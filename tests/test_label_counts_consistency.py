"""Label counts must be IDENTICAL across the manuscript, the repository and the
supplementary files (task requirement 4)."""

import csv
import io
import unittest

from tests import _spec


MANUSCRIPT = {
    "rare_num_classes": 568,
    "full_num_classes": 8930,
    "rare_admissions": 39304,
    "all_admissions": 58976,
    "rare_unique_patients": 31515,
    "all_unique_patients": 46520,
    "rare_candidate_codes": 992,
}


class LabelCountConsistencyTest(unittest.TestCase):
    def setUp(self):
        self.spec = _spec.load_spec()


    def test_spec_matches_manuscript(self):
        d = self.spec["dataset"]
        self.assertEqual(d["rare"]["distinct_icd9_codes"], MANUSCRIPT["rare_num_classes"])
        self.assertEqual(d["all"]["distinct_icd9_codes"], MANUSCRIPT["full_num_classes"])
        self.assertEqual(d["rare"]["admissions"], MANUSCRIPT["rare_admissions"])
        self.assertEqual(d["all"]["admissions"], MANUSCRIPT["all_admissions"])
        self.assertEqual(d["rare"]["unique_patients"], MANUSCRIPT["rare_unique_patients"])
        self.assertEqual(d["all"]["unique_patients"], MANUSCRIPT["all_unique_patients"])
        self.assertEqual(
            d["rare_subset_construction"]["candidate_icd9_codes"],
            MANUSCRIPT["rare_candidate_codes"],
        )
        self.assertEqual(
            self.spec["model"]["num_classes"]["rare"], MANUSCRIPT["rare_num_classes"]
        )
        self.assertEqual(
            self.spec["model"]["num_classes"]["full"], MANUSCRIPT["full_num_classes"]
        )


    def test_config_num_classes_match_spec(self):
        rare = _spec.model_params(_spec.load_yaml(_spec.MMACNET_CONFIGS["rare"]))
        full = _spec.model_params(_spec.load_yaml(_spec.MMACNET_CONFIGS["full"]))
        self.assertEqual(rare["num_classes"], MANUSCRIPT["rare_num_classes"])
        self.assertEqual(full["num_classes"], MANUSCRIPT["full_num_classes"])
        for name, path in _spec.MMACNET_CONFIGS.items():
            if name.startswith("ablation/"):
                params = _spec.model_params(_spec.load_yaml(path))
                self.assertEqual(
                    params["num_classes"], MANUSCRIPT["rare_num_classes"], name
                )

    def test_config_fc_layer_dims_end_in_num_classes(self):
        rare = _spec.model_params(_spec.load_yaml(_spec.MMACNET_CONFIGS["rare"]))
        full = _spec.model_params(_spec.load_yaml(_spec.MMACNET_CONFIGS["full"]))
        self.assertEqual(rare["fc_layer_dims"], [1024, MANUSCRIPT["rare_num_classes"]])
        self.assertEqual(full["fc_layer_dims"], [1024, MANUSCRIPT["full_num_classes"]])


    def test_full_labels_json_length(self):
        labels = _spec.load_json("datasets/mimic3_full/labels.json")
        self.assertEqual(len(labels), MANUSCRIPT["full_num_classes"])
        self.assertEqual(sorted(labels.values()), list(range(len(labels))))

    def test_rare_labels_json_length_when_generated(self):
        path = _spec.REPO / "datasets/mimic3_rare/labels.json"
        if not path.exists():
            self.skipTest("rare labels.json not generated (needs MIMIC-III)")
        labels = _spec.load_json("datasets/mimic3_rare/labels.json")
        self.assertEqual(len(labels), MANUSCRIPT["rare_num_classes"])

    def test_candidate_codes_csv_count(self):
        text = (_spec.REPO / "supplementary/rare_icd9_candidate_codes.csv").read_text()
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(rows), MANUSCRIPT["rare_candidate_codes"])
        self.assertEqual(len({r["icd9_code"] for r in rows}), MANUSCRIPT["rare_candidate_codes"])

    def test_supplementary_readme_quotes_the_same_counts(self):
        readme = (_spec.REPO / "supplementary/README.md").read_text()
        for value in ("568", "8930", "992"):
            self.assertIn(value, readme)

    def test_reconciliation_doc_exists(self):
        self.assertTrue((_spec.REPO / "docs/RECONCILIATION.md").exists())


if __name__ == "__main__":
    unittest.main()
