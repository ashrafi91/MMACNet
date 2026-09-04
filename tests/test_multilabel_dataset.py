"""Section 3.5 / 4.1 at the tensor level: BaseDataset must turn one admission
into exactly one multi-hot label vector, and must feed the model the
aggregated (mode / mean) structured features -- z-scored with the train-fit
scaler, unseen categories mapped to the reserved index 0."""

import json
import os
import tempfile
import unittest

from tests import _synth

from MMACNet.datasets.base_dataset import BaseDataset


def _prepare(tmp):
    pipe, _ = _synth.make_pipeline(tmp, rare_subset=True)
    table = pipe.build_admission_table(
        pipe.extract_codes(), pipe.load_notes(),
        pipe.aggregate_structured(pipe.load_prescriptions(), pipe.load_microbiology()),
    )
    table = pipe.apply_rare_filter(table, pipe.load_rare_candidates())
    train, val, _test = pipe.split_data(table, "HADM_ID")
    vocab, _ = pipe.fit_label_vocab(train, val, _test)
    meta = pipe.fit_tabular_meta(train)

    enc_train = pipe.encode_split(train)
    enc_val = pipe.encode_split(val)
    tokens = sorted(
        {t for row in enc_train["TEXT"] + enc_val["TEXT"] for t in row}
    )
    ds_dir = os.path.join(tmp, "dataset")
    _synth.write_tiny_dataset_dir(ds_dir, enc_train, vocab, meta, tokens)
    with open(os.path.join(ds_dir, "val.json"), "w") as fh:
        json.dump(enc_val, fh)
    return ds_dir, vocab, meta, enc_train, enc_val


class MultiLabelDatasetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        (self.ds_dir, self.vocab, self.meta,
         self.enc_train, self.enc_val) = _prepare(self._tmp.name)
        self.train_ds = BaseDataset(_synth.data_common(self.ds_dir, "train.json"))
        self.val_ds = BaseDataset(_synth.data_common(self.ds_dir, "val.json"))

    def _row_index(self, ds, encoded, hadm):
        return encoded["HADM_ID"].index(hadm)


    def test_dataset_length_is_admission_count_not_code_count(self):
        self.assertEqual(len(self.train_ds), 3)
        total_codes = sum(len(c) for c in self.enc_train["LABELS"])
        self.assertGreater(total_codes, len(self.train_ds))

    def test_getitem_returns_one_multihot_vector(self):
        idx = self._row_index(self.train_ds, self.enc_train, "104")
        tokens, tabular, labels = self.train_ds[idx]
        self.assertEqual(labels.shape, (len(self.vocab),))
        self.assertEqual(int(labels.sum()), 2)
        self.assertEqual(labels[self.vocab["277.00"]], 1)
        self.assertEqual(labels[self.vocab["270.1"]], 1)
        self.assertEqual(labels[self.vocab["759.83"]], 0)

    def test_collate_stacks_to_batch_by_num_labels(self):
        batch = self.train_ds.collate_fn([self.train_ds[i] for i in range(len(self.train_ds))])
        inputs, labels = batch
        self.assertEqual(tuple(labels.shape), (3, len(self.vocab)))

        self.assertTrue((labels.sum(dim=1) >= 1).all())
        self.assertIn("categorical", inputs)
        self.assertIn("numerical", inputs)


    def test_categorical_feature_is_train_vocab_index_of_the_mode(self):
        order = self.meta["categorical_order"]
        drug_pos = order.index("DRUG")
        idx = self._row_index(self.train_ds, self.enc_train, "104")
        _tokens, tabular, _labels = self.train_ds[idx]
        expected = self.meta["categorical"]["DRUG"]["mapping"]["vancomycin"]
        self.assertEqual(tabular["categorical"][drug_pos], expected)

    def test_numerical_feature_is_zscored_mean_over_all_rows(self):
        order = self.meta["numerical_order"]
        dose_pos = order.index("DOSE_VAL_RX")
        m = self.meta["numerical"]["DOSE_VAL_RX"]

        expected = (150.0 - m["mean"]) / m["std"]
        idx = self._row_index(self.train_ds, self.enc_train, "104")
        _tokens, tabular, _labels = self.train_ds[idx]
        self.assertAlmostEqual(tabular["numerical"][dose_pos], expected, places=5)

    def test_unseen_category_maps_to_reserved_index_zero(self):

        order = self.meta["categorical_order"]
        route_pos = order.index("ROUTE")
        idx = self._row_index(self.val_ds, self.enc_val, "105")
        _tokens, tabular, _labels = self.val_ds[idx]
        self.assertEqual(tabular["categorical"][route_pos], 0)

    def test_missing_modality_admission_is_unk_and_zero(self):

        idx = self._row_index(self.val_ds, self.enc_val, "101")
        _tokens, tabular, labels = self.val_ds[idx]
        self.assertTrue(all(c == 0 for c in tabular["categorical"]))
        for v in tabular["numerical"]:
            self.assertAlmostEqual(v, 0.0, places=6)
        self.assertEqual(int(labels.sum()), 1)


if __name__ == "__main__":
    unittest.main()
