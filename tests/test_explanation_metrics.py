import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from build_explanation_manifest import sha256
from libs.eval import get_preds_and_probs
from libs.explanation_metrics import compute_explanation_metrics
from libs.reproduction_state import refresh_training_memory, save_checkpoint, snapshot
from libs.retrieval_audit import instrument_retrieval
from tests.test_reproduction_state import ReproductionStateTests


class ExplanationMetricTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = ReproductionStateTests()
        self.fixture.setUp()

    def make_refreshed(self, task):
        wrapper, dataset, state = self.fixture.make_state(task)
        refresh = refresh_training_memory(wrapper, dataset)
        self.assertTrue(refresh["parameter_match"])
        test_x = dataset._indv_dataset()[2][0]
        wrapper.model.eval()
        _, _, trace, audit = instrument_retrieval(
            wrapper.model, test_x, int(wrapper.params["batch_size"]))
        self.assertTrue(audit["metadata_off_on_equal"])
        return wrapper, dataset, state, refresh, trace

    def test_binary_and_multiclass_metrics_obey_locked_identities(self):
        for task in ("binclass", "multiclass"):
            with self.subTest(task=task):
                wrapper, dataset, _, _, trace = self.make_refreshed(task)
                result = compute_explanation_metrics(wrapper, dataset, trace)
                summary, rows = result["summary"], result["query_rows"]
                train_y = dataset._indv_dataset()[0][1].numpy().astype(int)
                test_y = dataset._indv_dataset()[2][1].numpy().astype(int)

                self.assertEqual(len(rows), len(test_y))
                self.assertLess(summary["decomposition_max_abs_error"], 1e-5)
                self.assertLessEqual(summary["accuracy_delta_identity_error"], 1e-12)
                self.assertAlmostEqual(
                    summary["final_acc"] - summary["regional_baseline_acc"],
                    summary["corrected_rate"] - summary["degraded_rate"])
                self.assertGreaterEqual(
                    summary["prediction_change_rate"] + 1e-12,
                    summary["corrected_rate"] + summary["degraded_rate"])
                if task == "binclass":
                    self.assertAlmostEqual(
                        summary["prediction_change_rate"],
                        summary["corrected_rate"] + summary["degraded_rate"])
                    self.assertIn("regional_logit", rows[0])
                    self.assertNotIn("regional_logit_class_0", rows[0])
                else:
                    self.assertIn("regional_logit_class_0", rows[0])

                global_majority = int(np.bincount(
                    train_y, minlength=dataset.n_classes).argmax())
                self.assertAlmostEqual(
                    summary["global_majority_acc"],
                    float(np.mean(test_y == global_majority)))
                groups = wrapper.model.prototype_layer.sample_groups
                expected_region_predictions = []
                saw_fallback = False
                for row in rows:
                    members = np.asarray(groups[row["query_region"]], dtype=int)
                    majority = (global_majority if len(members) == 0 else
                                int(np.bincount(train_y[members],
                                               minlength=dataset.n_classes).argmax()))
                    expected_region_predictions.append(majority)
                    self.assertEqual(row["region_majority_label"], majority)
                    self.assertEqual(row["label_gain_eligible"],
                                     row["eligible_label_gain"])
                    if row["label_gain_eligible"]:
                        self.assertEqual(row["fallback_type"], "none")
                        self.assertEqual(row["n_valid_neighbors"], summary["k"])
                        self.assertEqual(row["same_region_share"], 1.0)
                        self.assertAlmostEqual(
                            row["label_agreement_gain"],
                            row["retrieved_label_agreement"] -
                            row["expected_region_label_agreement"])
                    else:
                        self.assertIsNone(row["label_agreement_gain"])
                    if row["global_label_delta_eligible"]:
                        self.assertEqual(row["fallback_type"], "none")
                        self.assertTrue(row["label_gain_eligible"])
                        self.assertEqual(row["n_valid_neighbors"], summary["k"])
                        self.assertEqual(row["same_region_share"], 1.0)
                        self.assertIsNotNone(row["label_agreement_delta_vs_global"])
                    else:
                        self.assertIsNone(row["label_agreement_delta_vs_global"])
                    if row["fallback_invoked"]:
                        saw_fallback = True
                        self.assertFalse(row["global_label_delta_eligible"])
                        self.assertIsNone(row["label_agreement_delta_vs_global"])
                self.assertAlmostEqual(
                    summary["region_majority_acc"],
                    float(np.mean(np.asarray(expected_region_predictions) == test_y)))
                # The small fixture has fewer than k members in at least one
                # routed region, directly exercising fallback exclusion.
                self.assertTrue(saw_fallback)

                local = result["local_neighbor_rows"]
                global_rows = result["global_neighbor_rows"]
                for row in rows:
                    if row["global_knn_overlap_eligible"]:
                        left = {r["sample_id"] for r in local
                                if r["query_index"] == row["query_index"]}
                        right = {r["sample_id"] for r in global_rows
                                 if r["query_index"] == row["query_index"]}
                        self.assertEqual(row["global_knn_jaccard"],
                                         len(left & right) / len(left | right))

    def test_runner_writes_complete_immutable_artifact_set(self):
        from analyze_explanation_structure import parser, run

        wrapper, dataset, state, refresh, trace = self.make_refreshed("binclass")
        test_x, test_y = dataset._indv_dataset()[2]
        with torch.no_grad():
            logits = wrapper.model(test_x)["logits"]
        preds = get_preds_and_probs(logits, dataset.tasktype)[0]
        refreshed = snapshot(
            wrapper, dataset, state["identity"], logits, preds,
            state_kind="final_encoder_refreshed_for_explanation",
            parent_checkpoint_sha256="test-parent", refresh_audit=refresh)

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "openml_31/fold_1"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "checkpoint_refreshed.pt"
            save_checkpoint(checkpoint, refreshed)
            checkpoint_sha = sha256(checkpoint)
            (run_dir / "retrieval_trace.json").write_text(
                json.dumps(trace), encoding="utf-8")
            (run_dir / "audit_retrieval.json").write_text(json.dumps(dict(
                status="retrieval_instrumentation_passed",
                eligible_for_explanation_metrics=True,
                refreshed_checkpoint_sha256=checkpoint_sha)), encoding="utf-8")
            args = parser().parse_args([
                "--analysis-root", tmp, "--dataset-id", "31", "--fold", "1",
                "--gpu-id", "-1"])
            # A prior failed/interrupted attempt is cleaned and can resume.
            (run_dir / "audit_metrics.json").write_text(json.dumps(dict(
                status="failed_explanation_metrics",
                eligible_for_aggregation=False)), encoding="utf-8")
            (run_dir / "summary.json").write_text("partial", encoding="utf-8")
            (run_dir / "query_metrics.parquet").write_bytes(b"partial")
            self.assertEqual(run(args), 0)
            expected = {
                "query_metrics.parquet", "region_stats.parquet",
                "neighbors_tabera.parquet", "neighbors_global.parquet",
                "summary.json", "audit_metrics.json",
            }
            self.assertTrue(all((run_dir / name).is_file() for name in expected))
            summary = json.loads((run_dir / "summary.json").read_text())
            audit = json.loads((run_dir / "audit_metrics.json").read_text())
            self.assertEqual(summary["train_seed"], 10)
            self.assertEqual(audit["status"], "explanation_metrics_passed")
            self.assertTrue(audit["eligible_for_aggregation"])
            self.assertEqual(audit["integrity"]["metric_protocol_version"],
                             "explanation-analysis-v1")
            for field in ("metrics_runner_sha256", "metrics_code_sha256",
                          "retrieval_audit_sha256", "retrieval_trace_sha256"):
                self.assertEqual(len(audit["integrity"][field]), 64)
            query = pd.read_parquet(run_dir / "query_metrics.parquet")
            self.assertEqual(len(query), len(test_y))
            self.assertEqual(set(query["train_seed"]), {10})
            self.assertEqual(sha256(checkpoint), checkpoint_sha)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
