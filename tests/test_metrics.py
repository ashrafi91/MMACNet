"""Section 4.2 metric definitions and validation-only threshold tuning."""

import unittest

import numpy as np

from MMACNet.modules.metrics import (
    MacroAUC,
    MacroF1,
    MicroAUC,
    PrecAtK,
    tune_global_threshold,
)


class PrecAtKTest(unittest.TestCase):
    def test_precision_at_k_counts_hits_in_topk_over_k(self):
        y_true = np.array([[1, 1, 0, 0, 0]])
        p_pred = np.array([[0.90, 0.10, 0.80, 0.70, 0.20]])
        metric = PrecAtK({"k": 2})
        self.assertAlmostEqual(metric(y_true=y_true, p_pred=p_pred), 0.5)

    def test_precision_at_k_perfect(self):
        y_true = np.array([[1, 1, 1, 0, 0]])
        p_pred = np.array([[0.9, 0.8, 0.7, 0.1, 0.2]])
        self.assertAlmostEqual(PrecAtK({"k": 3})(y_true=y_true, p_pred=p_pred), 1.0)


class MacroF1AbsentLabelTest(unittest.TestCase):
    def test_never_predicted_label_contributes_zero_and_is_included(self):

        y_true = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
        y_pred = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        macro = MacroF1(None)(y_true=y_true, y_pred=y_pred)

        self.assertAlmostEqual(macro, (1.0 + (2 / 3) + 0.0) / 3, places=6)


class AucSanityTest(unittest.TestCase):
    def test_micro_and_macro_auc_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=(40, 4))
        y_true[:, 3] = 0
        p_pred = rng.random(size=(40, 4))
        self.assertTrue(0.0 <= MicroAUC(None)(y_true=y_true, p_pred=p_pred) <= 1.0)

        macro = MacroAUC({"num_process": 1})(y_true=y_true, p_pred=p_pred)
        self.assertTrue(0.0 <= macro <= 1.0)


class ThresholdTuningTest(unittest.TestCase):
    def test_tunes_below_half_when_positives_are_low_probability(self):


        y_true = np.array([[1, 0, 1, 0]] * 10)
        p_pred = np.where(y_true == 1, 0.30, 0.05).astype(float)
        threshold, score = tune_global_threshold(y_true, p_pred, metric="micro_f1")
        self.assertLessEqual(threshold, 0.30)
        self.assertGreater(threshold, 0.05)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_default_grid_returns_valid_threshold(self):
        y_true = np.array([[1, 0], [0, 1], [1, 1]])
        p_pred = np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.9]])
        threshold, score = tune_global_threshold(y_true, p_pred)
        self.assertTrue(0.0 < threshold < 1.0)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()
