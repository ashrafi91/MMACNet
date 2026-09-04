"""The released MMAC-Net forward pass (Fig. 3 / Algorithm A1): per-label
attention, late-fusion tabular branch gated by ``tabular_modalities``, and the
optional description regulariser."""

import json
import os
import tempfile
import unittest

import numpy as np
import torch

from MMACNet.utils.configuration import Config
from MMACNet.utils.mapper import ConfigMapper
import MMACNet.models

EMBED_SIZE = 16
LABELS = ["401.9", "277.00", "759.83", "270.1", "250.00"]
VOCAB_TOKENS = ["hypertension", "diabetes", "renal", "cardiac", "sepsis", "course", "admission"]


def _build_model_dir(tmp, tabular=True):
    ds_dir = os.path.join(tmp, "dataset")
    static_dir = os.path.join(tmp, "static")
    os.makedirs(os.path.join(ds_dir, "word2vec"), exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    tokens = ["<pad>", "<unk>"] + VOCAB_TOKENS
    json.dump(
        {t: i for i, t in enumerate(tokens)},
        open(os.path.join(ds_dir, "word2vec", "token_to_idx.json"), "w"),
    )
    emb = np.random.default_rng(1).normal(size=(len(tokens), EMBED_SIZE)).astype("float32")
    emb[0] = 0.0
    np.save(os.path.join(ds_dir, "word2vec", "embedding_matrix.npy"), emb)

    json.dump(
        {c: i for i, c in enumerate(LABELS)},
        open(os.path.join(ds_dir, "labels.json"), "w"),
    )

    if tabular:
        meta = {
            "categorical": {
                "DRUG_TYPE": {"mapping": {"MAIN": 1, "BASE": 2}, "unk_index": 0, "num_classes": 3},
                "DRUG": {"mapping": {"aspirin": 1, "insulin": 2}, "unk_index": 0, "num_classes": 3},
                "PROD_STRENGTH": {"mapping": {"81mg": 1}, "unk_index": 0, "num_classes": 2},
                "ROUTE": {"mapping": {"PO": 1, "IV": 2}, "unk_index": 0, "num_classes": 3},
                "ORG_ITEMID": {"mapping": {"80293": 1}, "unk_index": 0, "num_classes": 2},
                "AB_ITEMID": {"mapping": {"90015": 1}, "unk_index": 0, "num_classes": 2},
                "INTERPRETATION": {"mapping": {"S": 1, "R": 2}, "unk_index": 0, "num_classes": 3},
            },
            "numerical": {
                "DOSE_VAL_RX": {"mean": 20.0, "std": 5.0},
                "DILUTION_VALUE": {"mean": 4.0, "std": 2.0},
            },
            "categorical_order": [
                "DRUG_TYPE", "DRUG", "PROD_STRENGTH", "ROUTE",
                "ORG_ITEMID", "AB_ITEMID", "INTERPRETATION",
            ],
            "numerical_order": ["DOSE_VAL_RX", "DILUTION_VALUE"],
        }
        json.dump(meta, open(os.path.join(ds_dir, "tabular_meta.json"), "w"))

    with open(os.path.join(static_dir, "icd9_descriptions.txt"), "w") as fh:
        fh.write("401.9 hypertension\n")
        fh.write("277.00 diabetes renal\n")
        fh.write("759.83 cardiac\n")
        fh.write("270.1 renal cardiac\n")
        fh.write("250.00 diabetes\n")
    return ds_dir, static_dir


def _model_config(ds_dir, static_dir, *, lmbda=0.0, modalities=("numerical", "categorical")):
    return Config(dic={
        "version": "mimic3",
        "dataset_dir": ds_dir,
        "mimic_dir": os.path.join(ds_dir, "does_not_exist"),
        "static_dir": static_dir,
        "word2vec_dir": os.path.join(ds_dir, "word2vec"),
        "num_classes": len(LABELS),
        "embed_size": EMBED_SIZE,
        "kernel_size": 3,
        "num_filter_maps": 6,
        "conv_block_depth": 3,
        "dropout": 0.6,
        "fc_dropout": 0.3,
        "conv_block_dropout": 0.0,
        "lmbda": lmbda,
        "init_code_emb": False,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "fc_layer_dims": [8, len(LABELS)],
        "use_batch_norm": True,
        "conv_activation": "relu",
        "use_se_block": True,
        "se_reduction": 2,
        "use_residual": True,
        "use_depthwise_separable": True,
        "tabular_hidden_dim": 5,
        "tabular_fusion": "late",
        "tabular_modalities": list(modalities),
    })


class ModelForwardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ds_dir, self.static_dir = _build_model_dir(self._tmp.name)
        torch.manual_seed(0)

    def _make(self, **kw):
        model = ConfigMapper.get_object("models", "MMACNet")(
            _model_config(self.ds_dir, self.static_dir, **kw)
        )
        model.eval()
        return model

    def test_multimodal_forward_shape(self):
        model = self._make()
        text = torch.randint(0, len(VOCAB_TOKENS) + 2, (2, 24))
        categorical = torch.randint(0, 2, (2, 7))
        numerical = torch.randn(2, 2)
        out = model(text, categorical=categorical, numerical=numerical)
        self.assertEqual(tuple(out.shape), (2, len(LABELS)))
        self.assertTrue(torch.isfinite(out).all())

    def test_per_label_attention_is_a_distribution(self):
        model = self._make()
        text = torch.randint(0, len(VOCAB_TOKENS) + 2, (2, 24))
        model(text, categorical=torch.zeros(2, 7, dtype=torch.long),
              numerical=torch.zeros(2, 2))
        attn = model.get_input_attention()
        self.assertEqual(attn.shape[:2], (2, len(LABELS)))
        np.testing.assert_allclose(attn.sum(axis=2), 1.0, rtol=1e-5, atol=1e-5)

    def test_notes_only_ablation_needs_no_tabular_inputs(self):
        model = self._make(modalities=())
        self.assertIsNone(model.tabular_mlp)
        text = torch.randint(0, len(VOCAB_TOKENS) + 2, (2, 24))
        out = model(text)
        self.assertEqual(tuple(out.shape), (2, len(LABELS)))

    def test_categorical_only_ablation(self):
        model = self._make(modalities=("categorical",))
        self.assertEqual(model.tabular_num_order, [])
        self.assertTrue(len(model.tabular_cat_order) == 7)
        out = model(
            torch.randint(0, 8, (2, 24)),
            categorical=torch.zeros(2, 7, dtype=torch.long),
        )
        self.assertEqual(tuple(out.shape), (2, len(LABELS)))

    def test_regularizer_zero_when_lambda_zero(self):
        model = self._make(lmbda=0.0)
        labels = torch.zeros(2, len(LABELS))
        labels[0, 1] = 1
        self.assertEqual(model.regularizer(labels), 0.0)

    def test_regularizer_finite_when_lambda_positive(self):
        model = self._make(lmbda=0.25)
        labels = torch.zeros(2, len(LABELS))
        labels[0, 1] = 1
        labels[1, 3] = 1
        reg = model.regularizer(labels)
        self.assertTrue(torch.is_tensor(reg))
        self.assertEqual(reg.dim(), 0)
        self.assertTrue(torch.isfinite(reg))
        self.assertGreaterEqual(float(reg), 0.0)


if __name__ == "__main__":
    unittest.main()
