#!/usr/bin/env python3
"""Export the highest-weighted n-grams per rare ICD-9 code (Section 7.4 / Appendix B).

Loads the trained MMAC-Net checkpoint selected by the config, runs a forward
pass over the test split, and for every label accumulates the per-label
attention weight (``model.get_input_attention()``) landing on each token
n-gram.  Writes ``supplementary/rare_code_top_ngrams.csv`` with columns
``icd9_code, rank, ngram, weight``.

Usage:
    python scripts/export_rare_code_top_ngrams.py \
        --config_path configs/MMACNet/MMACNet_mimic3_rare.yml \
        --ngram 4 --top 20
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from MMACNet.utils.configuration import Config
from MMACNet.utils.mapper import ConfigMapper
from MMACNet.utils.import_related_ops import pandas_related_ops

pandas_related_ops()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--ngram", type=int, default=4)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "supplementary" / "rare_code_top_ngrams.csv",
    )
    args = ap.parse_args()

    config = Config(path=args.config_path)
    dataset = ConfigMapper.get_object("datasets", config.dataset.name)(
        config.dataset.params.test
    )
    model = ConfigMapper.get_object("models", config.model.name)(config.model.params)

    ckpt_saver = ConfigMapper.get_object(
        "checkpoint_savers", config.trainer.params.checkpoint_saver.name
    )(config.trainer.params.checkpoint_saver.params)
    ckpt = ckpt_saver.get_best_checkpoint() or (
        ckpt_saver.get_latest_checkpoint() or [None, None]
    )[1]
    if ckpt is None:
        raise SystemExit("No checkpoint found; train the model first.")
    ckpt_saver.load_ckpt(model, ckpt)
    model.eval()

    inv_labels = dataset.inv_labels
    weights = defaultdict(lambda: defaultdict(float))

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn
    )
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            text = inputs["text"] if isinstance(inputs, dict) else inputs
            model(**inputs) if isinstance(inputs, dict) else model(inputs)
            attn = model.get_input_attention()
            attn = np.asarray(attn)
            if attn.shape[1] == text.shape[1]:
                attn = attn.transpose(0, 2, 1)
            for b in range(text.shape[0]):
                tokens = dataset.decode_tokens(
                    [t for t in text[b].tolist() if t != dataset.pad_idx]
                )
                gold = np.nonzero(labels[b].numpy())[0]
                for y in gold:
                    a = attn[b, y, : len(tokens)]
                    for i in range(len(tokens) - args.ngram + 1):
                        gram = " ".join(tokens[i : i + args.ngram])
                        weights[int(y)][gram] += float(a[i : i + args.ngram].sum())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["icd9_code", "rank", "ngram", "weight"])
        for y in sorted(weights):
            ranked = sorted(weights[y].items(), key=lambda kv: -kv[1])[: args.top]
            for rank, (gram, w) in enumerate(ranked, start=1):
                writer.writerow([inv_labels[y], rank, gram, round(w, 6)])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
