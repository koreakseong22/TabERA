import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_explanation_analysis_batch import parser, run, stages_for


class ExplanationBatchTests(unittest.TestCase):
    def manifest(self, root, count=4):
        rows = [dict(dataset_id=dataset_id, fold=1, train_seed=10,
                     n_test=100 - dataset_id)
                for dataset_id in range(1, count + 1)]
        path = root / "manifest.json"
        path.write_text(json.dumps({"runs": rows}))
        return path

    def test_two_gpu_dry_run_assigns_each_run_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root)
            args = parser().parse_args([
                "--analysis-root", str(root), "--manifest", str(manifest),
                "--gpus", "0", "1", "--dry-run"])
            self.assertEqual(run(args), 0)
            report = json.loads((root / "batch_status.json").read_text())
            self.assertEqual(report["gpus"], [0, 1])
            self.assertEqual(report["counts"], {"pending": 4})
            self.assertEqual(len({(row["dataset_id"], row["fold"])
                                  for row in report["runs"]}), 4)
            self.assertEqual({row["gpu_id"] for row in report["runs"]}, {0, 1})

    def test_duplicate_gpu_workers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self.manifest(root, 1)
            args = parser().parse_args([
                "--analysis-root", str(root), "--manifest", str(manifest),
                "--gpus", "0", "0", "--dry-run"])
            with self.assertRaisesRegex(ValueError, "unique"):
                run(args)

    def test_physical_gpu_reaches_reproduction_and_later_stages_use_visible_zero(self):
        row = dict(dataset_id=31, fold=2)
        stages = stages_for(row, Path("analysis"), Path("manifest.json"), 1)
        train_command = stages[0][-1]
        refresh_command = stages[2][-1]
        self.assertEqual(train_command[train_command.index("--gpu-id") + 1], "1")
        self.assertEqual(refresh_command[refresh_command.index("--gpu-id") + 1], "0")


if __name__ == "__main__":
    unittest.main()
