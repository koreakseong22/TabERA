import unittest

import numpy as np
from scipy.special import expit, softmax
from sklearn.metrics import log_loss, roc_auc_score

from libs.eval import calculate_metric


class LogitEvaluationTest(unittest.TestCase):
    def test_logits_and_explicit_probability_bypass(self):
        for task, logits, y in (
            ("binclass", np.array([[-3.], [0.2], [2.], [-1.]]), np.array([0, 1, 0, 1])),
            ("multiclass", np.array([[3., 1., -2.], [2., 4., 0.], [-1., 0., 2.],
                                     [2., 1., 3.]]), np.array([0, 1, 2, 1])),
        ):
            with self.subTest(task=task):
                p = expit(logits).reshape(-1) if task == "binclass" else softmax(logits, axis=1)
                pred = (p > 0.5).astype(int) if task == "binclass" else p.argmax(axis=1)
                target = y if task == "binclass" else np.eye(3)[y]
                result = calculate_metric(y, pred, logits, task, "test")
                bypass = calculate_metric(y, pred, p, task, "test", prob=True)
                self.assertEqual(result, bypass)
                expected_auc = (roc_auc_score(target, p) if task == "binclass" else
                                roc_auc_score(target, p, average="macro", multi_class="ovr"))
                self.assertEqual(result["auroc_test"], expected_auc)
                self.assertAlmostEqual(result["logloss_test"], log_loss(target, p), places=12)
                double = calculate_metric(y, pred, p, task, "test")
                self.assertNotAlmostEqual(result["logloss_test"], double["logloss_test"], places=5)


if __name__ == "__main__":
    unittest.main()
