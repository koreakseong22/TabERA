import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from aggregate_explanation_analysis import TABLE2, TABLE3, parser, run
from build_explanation_manifest import sha256
from libs.explanation_metrics import METRIC_PROTOCOL_VERSION


class ExplanationAggregationTests(unittest.TestCase):
    def build_complete_fixture(self, root):
        metric_hash = sha256(Path(__file__).resolve().parents[1] /
                             "libs" / "explanation_metrics.py")
        manifest = {"runs": []}
        metrics = {name: 0.5 for name in set(TABLE2 + TABLE3)}
        for dataset_id in range(1, 22):
            for fold in range(1, 6):
                seed = fold * 10
                manifest["runs"].append(dict(
                    dataset_id=dataset_id, fold=fold, train_seed=seed))
                run_dir = root / f"openml_{dataset_id}" / f"fold_{fold}"
                run_dir.mkdir(parents=True)
                summary = dict(dataset_id=dataset_id, fold=fold, train_seed=seed,
                               tasktype="binclass", n_test=10, n_train=40, k=8,
                               **metrics)
                audit = dict(status="explanation_metrics_passed",
                             eligible_for_aggregation=True,
                             integrity=dict(metric_protocol_version=METRIC_PROTOCOL_VERSION,
                                            metrics_code_sha256=metric_hash))
                (run_dir / "summary.json").write_text(json.dumps(summary))
                (run_dir / "audit_metrics.json").write_text(json.dumps(audit))
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        return manifest_path

    def test_complete_current_protocol_and_seed_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.build_complete_fixture(root)
            args = parser().parse_args([
                "--analysis-root", str(root), "--manifest", str(manifest)])
            self.assertEqual(run(args), 0)
            audit = json.loads((root / "aggregate/aggregate_audit.json").read_text())
            self.assertEqual((audit["status"], audit["valid_runs"]), ("complete", 105))
            self.assertEqual(list(pd.read_csv(root / "aggregate/table3.csv")["metric"]), TABLE3)

            bad_audit_path = root / "openml_1/fold_1/audit_metrics.json"
            bad_audit = json.loads(bad_audit_path.read_text())
            bad_audit["integrity"]["metrics_code_sha256"] = "old-code"
            bad_audit_path.write_text(json.dumps(bad_audit))
            self.assertEqual(run(args), 2)
            audit = json.loads((root / "aggregate/aggregate_audit.json").read_text())
            self.assertIn("metric_code_mismatch", {p["status"] for p in audit["problems"]})

            bad_audit["integrity"]["metrics_code_sha256"] = sha256(
                Path(__file__).resolve().parents[1] / "libs" / "explanation_metrics.py")
            bad_audit_path.write_text(json.dumps(bad_audit))
            summary_path = root / "openml_1/fold_1/summary.json"
            summary = json.loads(summary_path.read_text())
            summary["train_seed"] = 999
            summary_path.write_text(json.dumps(summary))
            self.assertEqual(run(args), 2)
            audit = json.loads((root / "aggregate/aggregate_audit.json").read_text())
            self.assertIn("train_seed_mismatch", {p["status"] for p in audit["problems"]})


if __name__ == "__main__":
    unittest.main()
