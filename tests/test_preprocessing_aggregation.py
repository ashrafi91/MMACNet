"""Section 3.5: one multi-hot vector per admission; ALL prescription and
microbiology records aggregated (never just the first)."""

import math
import tempfile
import unittest

from tests import _synth


class PreprocessingAggregationTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.pipe, self.save_dir = _synth.make_pipeline(
            self._tmp.name, rare_subset=True
        )
        self.code_df = self.pipe.extract_codes()
        self.notes_df = self.pipe.load_notes()
        self.presc_df = self.pipe.load_prescriptions()
        self.micro_df = self.pipe.load_microbiology()
        self.structured = self.pipe.aggregate_structured(
            self.presc_df, self.micro_df
        ).set_index("HADM_ID")
        self.table = self.pipe.build_admission_table(
            self.code_df, self.notes_df,
            self.structured.reset_index(),
        )


    def test_one_row_per_admission(self):
        self.assertEqual(len(self.table), len(_synth.ADMISSIONS))
        self.assertTrue(self.table["HADM_ID"].is_unique)

    def test_labels_are_lists_not_exploded(self):
        for codes in self.table["LABELS"]:
            self.assertIsInstance(codes, list)
        row100 = self.table.set_index("HADM_ID").loc["100", "LABELS"]
        self.assertEqual(sorted(row100), sorted(["401.9", "250.00", "277.00"]))


    def test_prescription_categorical_uses_mode_over_all_rows(self):





        self.assertEqual(self.structured.loc["100", "DRUG"], "aspirin")
        self.assertEqual(self.structured.loc["102", "ROUTE"], "SC")

    def test_prescription_numerical_uses_mean_over_all_rows(self):

        self.assertAlmostEqual(self.structured.loc["100", "DOSE_VAL_RX"], 20.0)
        self.assertNotAlmostEqual(self.structured.loc["100", "DOSE_VAL_RX"], 10.0)

        self.assertAlmostEqual(self.structured.loc["102", "DOSE_VAL_RX"], 6.0)


    def test_microbiology_categorical_uses_mode_over_all_rows(self):

        self.assertEqual(self.structured.loc["100", "ORG_ITEMID"], "80293")
        self.assertNotEqual(self.structured.loc["100", "ORG_ITEMID"], "80023")

    def test_microbiology_numerical_uses_mean_over_all_rows(self):

        self.assertAlmostEqual(
            self.structured.loc["100", "DILUTION_VALUE"], 14.0 / 3.0, places=6
        )

    def test_admission_without_structured_records(self):

        self.assertNotIn("101", self.structured.index)
        row = self.table.set_index("HADM_ID").loc["101"]
        for field in self.pipe.categorical_fields + self.pipe.numerical_fields:
            value = row[field]
            self.assertTrue(value is None or (isinstance(value, float) and math.isnan(value)))


    def test_rare_filter_drops_admissions_without_a_rare_code(self):
        rare = self.pipe.load_rare_candidates()
        filtered = self.pipe.apply_rare_filter(self.table, rare)
        self.assertNotIn("103", set(filtered["HADM_ID"]))
        self.assertEqual(len(filtered), 5)

    def test_rare_filter_restricts_labels_to_rare_codes(self):
        rare = self.pipe.load_rare_candidates()
        filtered = self.pipe.apply_rare_filter(self.table, rare).set_index("HADM_ID")
        self.assertEqual(filtered.loc["100", "LABELS"], ["277.00"])
        self.assertEqual(sorted(filtered.loc["104", "LABELS"]), ["270.1", "277.00"])

    def test_label_vocab_is_the_rare_codes_that_actually_occur(self):
        rare = self.pipe.load_rare_candidates()
        filtered = self.pipe.apply_rare_filter(self.table, rare)
        train, val, test = self.pipe.split_data(filtered, "HADM_ID")
        vocab, counts = self.pipe.fit_label_vocab(train, val, test)

        self.assertEqual(set(vocab), {"277.00", "759.83", "270.1"})
        self.assertTrue(set(vocab).issubset(_synth.RARE_CANDIDATES))
        self.assertEqual(set(vocab), set(counts) & set(vocab))

        self.assertEqual(sorted(vocab.values()), list(range(len(vocab))))


class PreprocessingEndToEndTest(unittest.TestCase):
    """The full driver produces exactly the files the trainer expects."""

    def test_preprocess_writes_consistent_artifacts(self):
        import json
        import os

        def _load(name):
            with open(os.path.join(save_dir, name)) as fh:
                return json.load(fh)

        with tempfile.TemporaryDirectory() as tmp:
            pipe, save_dir = _synth.make_pipeline(tmp, rare_subset=True)
            pipe.preprocess()

            labels = _load("labels.json")
            self.assertEqual(set(labels), {"277.00", "759.83", "270.1"})

            meta = _load("tabular_meta.json")
            self.assertEqual(
                meta["numerical_order"], ["DOSE_VAL_RX", "DILUTION_VALUE"]
            )
            self.assertEqual(len(meta["categorical_order"]), 7)

            self.assertNotIn("NG", meta["categorical"]["ROUTE"]["mapping"])
            self.assertIn("PO", meta["categorical"]["ROUTE"]["mapping"])

            train = _load("train.json")
            self.assertEqual(len(train["HADM_ID"]), 3)
            for codes in train["LABELS"]:
                self.assertIsInstance(codes, list)
            self.assertTrue(os.path.exists(
                os.path.join(save_dir, "word2vec", "embedding_matrix.npy")
            ))


if __name__ == "__main__":
    unittest.main()
