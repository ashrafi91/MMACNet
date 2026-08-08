import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def load_descriptions(dataset_dir: Path, static_dir: Path) -> Dict[str, str]:
    desc_path = dataset_dir / "desc.json"
    if desc_path.exists():
        return json.load(open(desc_path))

    static_desc = static_dir / "icd9_descriptions.txt"
    if static_desc.exists():
        desc_map: Dict[str, str] = {}
        with open(static_desc) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    code, text = parts[0], parts[1]
                    desc_map[code] = text
        if desc_map:
            return desc_map

    return {}


def build_similarity_matrix(labels, token_to_idx, embedding_matrix, static_dir, top_k=10, threshold=0.0):
    unk_idx = token_to_idx.get("<unk>", 0)
    desc_map = load_descriptions(labels["dataset_dir"], static_dir)

    idx_to_code = {i: c for c, i in labels.items() if c != "dataset_dir"}
    embed_tensor = torch.tensor(embedding_matrix, dtype=torch.float)
    embed_dim = embed_tensor.size(1)

    vectors = []
    for i in range(len(idx_to_code)):
        code = idx_to_code[i]
        desc = desc_map.get(code, code)
        tokens = desc.lower().split()
        idxs = [token_to_idx.get(t, unk_idx) for t in tokens if t]
        if not idxs:
            vectors.append(torch.zeros(embed_dim))
        else:
            vecs = embed_tensor[torch.tensor(idxs, dtype=torch.long)]
            vectors.append(vecs.mean(dim=0))

    desc_mat = torch.stack(vectors)
    desc_mat = F.normalize(desc_mat, p=2, dim=1)
    desc_mat[torch.isnan(desc_mat)] = 0.0

    sim = desc_mat @ desc_mat.t()
    sim[torch.isnan(sim)] = 0.0
    sim.fill_diagonal_(0.0)

    if threshold > 0:
        sim = torch.where(sim > threshold, sim, torch.zeros_like(sim))

    if top_k and top_k < sim.size(1):
        vals, idx = torch.topk(sim, k=top_k, dim=1)
        mask = torch.zeros_like(sim)
        mask.scatter_(1, idx, vals)
        sim = mask

    sim = torch.maximum(sim, sim.t())
    return sim


def main():
    parser = argparse.ArgumentParser(description="Generate label graph adjacency matrix")
    parser.add_argument("--dataset_dir", type=Path, default=Path("datasets/mimic3_full"))
    parser.add_argument(
        "--static_dir",
        type=Path,
        default=Path(os.environ.get("MIMIC_STATIC_DIR", "datasets/mimic3/static")),
        help="Directory containing icd9_descriptions.txt (or set MIMIC_STATIC_DIR)",
    )
    parser.add_argument("--word2vec_dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    word2vec_dir = args.word2vec_dir or dataset_dir / "word2vec"
    out_path = args.out or dataset_dir / "label_graph.npy"

    labels_path = dataset_dir / "labels.json"
    token_path = word2vec_dir / "token_to_idx.json"
    emb_path = word2vec_dir / "embedding_matrix.npy"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found at {labels_path}")
    if not token_path.exists() or not emb_path.exists():
        raise FileNotFoundError("word2vec token_to_idx.json or embedding_matrix.npy missing")

    labels = json.load(open(labels_path))
    labels["dataset_dir"] = dataset_dir  # pass through to description loader
    token_to_idx = json.load(open(token_path))
    embedding_matrix = np.load(emb_path)

    sim = build_similarity_matrix(
        labels, token_to_idx, embedding_matrix, args.static_dir, args.top_k, args.threshold
    )
    np.save(out_path, sim.cpu().numpy())
    print(f"Saved label graph to {out_path}")


if __name__ == "__main__":
    main()
