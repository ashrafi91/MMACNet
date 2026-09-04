"""Every experimental setting must be IDENTICAL across the manuscript
(EXPERIMENT_SPEC.yaml, transcribed from Table 4 / Section 5.3) and the configs
(task requirement 4)."""

import unittest

from tests import _spec


TABLE4 = {
    "embedding_dimension": 100,
    "sequence_length": 1500,
    "kernel_size": 50,
    "num_filter_maps": 50,
    "conv_block_depth": 6,
    "regularisation_lambda": 0.25,
    "embedding_dropout": 0.6,
    "fc_dropout": 0.3,
    "batch_size": 512,
    "tabular_hidden_dim": 50,
}
SECTION_53 = {
    "learning_rate": 0.001,
    "lr_scheduler_t_max": 200,
    "max_epochs": 200,
    "early_stopping_patience": 15,
    "data_loader_workers": 16,
    "seed": 1337,
}


class SpecMatchesManuscriptTest(unittest.TestCase):
    def test_spec_model_block_equals_table4(self):
        m = _spec.load_spec()["model"]
        for key, value in TABLE4.items():
            got = m.get(key, _spec.load_spec()["training"].get(key))
            self.assertEqual(got, value, key)
        self.assertEqual(m["objective_function"], "bce")
        self.assertTrue(m["batch_normalization"])
        self.assertEqual(m["activation"], "relu")
        self.assertEqual(m["fusion_strategy"], "late")

    def test_spec_training_block_equals_section_53(self):
        t = _spec.load_spec()["training"]
        for key, value in SECTION_53.items():
            self.assertEqual(t[key], value, key)
        self.assertEqual(t["optimizer"], "adam")
        self.assertEqual(t["lr_scheduler"], "cosineanneal")
        self.assertEqual(t["early_stopping_metric"], "prec_at_8")
        self.assertTrue(t["drop_last"])
        self.assertTrue(t["train_loader_shuffle"])


class ConfigsMatchSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = _spec.load_spec()

    def _check_model(self, cfg, variant):
        p = _spec.model_params(cfg)
        s = self.spec["model"]
        self.assertEqual(p["embed_size"], s["embedding_dimension"])
        self.assertEqual(p["kernel_size"], s["kernel_size"])
        self.assertEqual(p["num_filter_maps"], s["num_filter_maps"])
        self.assertEqual(p["conv_block_depth"], s["conv_block_depth"])
        self.assertEqual(p["dropout"], s["embedding_dropout"])
        self.assertEqual(p["fc_dropout"], s["fc_dropout"])
        self.assertEqual(p["lmbda"], s["regularisation_lambda"])
        self.assertEqual(p["tabular_hidden_dim"], s["tabular_hidden_dim"])
        self.assertEqual(p["tabular_fusion"], s["fusion_strategy"])
        self.assertEqual(p["conv_activation"], s["activation"])
        self.assertIs(p["use_batch_norm"], True)
        self.assertIs(p["use_se_block"], True)
        self.assertIs(p["use_residual"], True)
        self.assertIs(p["use_depthwise_separable"], True)
        self.assertEqual(p["num_classes"], s["num_classes"][variant])
        self.assertEqual(p["fc_layer_dims"], s["fc_layer_dims"][variant])
        self.assertEqual(
            _spec.data_common(cfg)["max_length"], s["sequence_length"]
        )

    def _check_trainer(self, cfg):
        p = _spec.trainer_params(cfg)
        s = self.spec["training"]
        self.assertEqual(p["data_loader"]["batch_size"], s["batch_size"])
        self.assertEqual(p["data_loader"]["num_workers"], s["data_loader_workers"])
        self.assertIs(p["data_loader"]["drop_last"], True)
        self.assertIs(p["data_loader"]["shuffle"], True)
        self.assertEqual(p["loss"]["name"], "BinaryCrossEntropyLoss")
        self.assertEqual(p["optimizer"]["name"], "adam")
        self.assertEqual(p["optimizer"]["params"]["lr"], s["learning_rate"])
        self.assertEqual(p["lr_scheduler"]["name"], "cosineanneal")
        self.assertEqual(p["lr_scheduler"]["params"]["T_max"], s["lr_scheduler_t_max"])
        self.assertEqual(p["max_epochs"], s["max_epochs"])
        self.assertEqual(p["stopping_criterion"]["patience"], s["early_stopping_patience"])
        self.assertEqual(p["stopping_criterion"]["metric"]["name"], s["early_stopping_metric"])
        self.assertEqual(p["seed"], s["seed"])
        self.assertIs(p["tune_threshold_on_val"], True)

    def test_rare_config(self):
        cfg = _spec.load_yaml(_spec.MMACNET_CONFIGS["rare"])
        self._check_model(cfg, "rare")
        self._check_trainer(cfg)

    def test_full_config(self):
        cfg = _spec.load_yaml(_spec.MMACNET_CONFIGS["full"])
        self._check_model(cfg, "full")
        self._check_trainer(cfg)

    def test_ablation_configs_only_differ_in_modalities(self):
        for name, path in _spec.MMACNET_CONFIGS.items():
            if not name.startswith("ablation/"):
                continue
            cfg = _spec.load_yaml(path)
            self._check_model(cfg, "rare")
            self._check_trainer(cfg)

    def test_ablation_modalities_match_spec_rows(self):
        want = {
            frozenset(row["tabular_modalities"])
            for row in self.spec["ablation"]["rows"]
        }
        got = set()
        for name, path in _spec.MMACNET_CONFIGS.items():
            if not name.startswith("ablation/"):
                continue
            mods = _spec.model_params(_spec.load_yaml(path))["tabular_modalities"]
            got.add(frozenset(mods))
        self.assertEqual(got, want)


class PreprocessingConfigsMatchSpecTest(unittest.TestCase):
    def setUp(self):
        self.spec = _spec.load_spec()

    def test_word2vec_and_text_settings(self):
        pp = self.spec["preprocessing"]
        for variant, path in _spec.PREPROCESSING_CONFIGS.items():
            params = _spec.load_yaml(path)["preprocessing"]["params"]
            w2v = params["embedding"]["params"]["word2vec_params"]
            self.assertEqual(w2v["vector_size"], pp["word2vec"]["vector_size"])
            self.assertEqual(w2v["min_count"], pp["word2vec"]["min_count"])
            self.assertEqual(w2v["epochs"], pp["word2vec"]["epochs"])

            cnp = params["clinical_note_preprocessing"]
            self.assertIs(
                cnp["remove_stopwords"]["params"]["remove_common_medical_terms"],
                False,
            )
            self.assertEqual(
                cnp["stem_or_lemmatize"]["params"]["stemmer_name"],
                "nltk.WordNetLemmatizer",
            )
            self.assertEqual(
                cnp["truncate"]["params"]["max_length"], pp["notes"]["max_tokens"]
            )

            structured = params["structured"]
            self.assertEqual(
                structured["numerical_fields"], pp["structured"]["numerical_fields"]
            )
            self.assertEqual(
                structured["categorical_fields"],
                pp["structured"]["categorical_fields"],
            )

    def test_rare_flag(self):
        rare = _spec.load_yaml(_spec.PREPROCESSING_CONFIGS["rare"])["preprocessing"]["params"]
        full = _spec.load_yaml(_spec.PREPROCESSING_CONFIGS["full"])["preprocessing"]["params"]
        self.assertIs(rare["code_preprocessing"]["rare_subset"], True)
        self.assertIs(full["code_preprocessing"]["rare_subset"], False)
        self.assertIn(
            "rare_icd9_candidate_codes.csv",
            rare["code_preprocessing"]["rare_candidate_codes_csv"],
        )


if __name__ == "__main__":
    unittest.main()
