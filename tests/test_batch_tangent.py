import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from libs.benchmark import arm_config, implementation_id, result_path
from libs.search_space import RECIPE_TAG
from reproduce import structure_result_path, with_structure


ROOT = Path(__file__).resolve().parents[1]


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / name), *map(str, args)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
    )


class TangentBatchRunnerTests(unittest.TestCase):
    def test_serial_dry_run_passes_structure_to_both_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(
                "run_final_benchmark.py",
                "--only_ds", 29, "--seeds", 1, "--savepath", tmp,
                "--run_hpo", "--dry_run",
                "--correction_geometry", "tangent",
                "--head_input_scale", "unit",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.count("--correction_geometry tangent"), 2)
        self.assertEqual(proc.stdout.count("--head_input_scale unit"), 2)
        self.assertIn("optimize.py", proc.stdout)
        self.assertIn("reproduce.py", proc.stdout)

    def test_parallel_resume_checks_tangent_result_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = with_structure(arm_config(), "tangent", "unit")
            path = structure_result_path(result_path(tmp, 1, 29, config=config), config)
            path.parent.mkdir(parents=True)
            path.touch()
            proc = run_script(
                "run_final_parallel.py", "--gpus", 0,
                "--only_ds", 29, "--seeds", 1, "--savepath", tmp,
                "--dry_run", "--correction_geometry", "tangent",
                "--head_input_scale", "unit",
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("0 jobs to run, 1 already", proc.stdout)

    def test_serial_aggregate_reads_and_separates_tangent_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = with_structure(arm_config(), "tangent", "unit")
            path = structure_result_path(result_path(tmp, 1, 29, config=config), config)
            path.parent.mkdir(parents=True)
            np.save(path, {
                "identity": {
                    "contract": {"config": config, "implementation": implementation_id()},
                    "dataset_id": 29,
                    "fold": 1,
                    "hpo_source": "own",
                    "unverified_hpo": False,
                },
                "Performance": {"acc_test": 0.8, "auroc_test": 0.9},
            }, allow_pickle=True)
            proc = run_script(
                "run_final_benchmark.py", "--only_ds", 29, "--seeds", 1,
                "--savepath", tmp, "--aggregate",
                "--correction_geometry", "tangent", "--head_input_scale", "unit",
            )
            output = (Path(tmp) / "results" /
                      f"tabera_final_per_fold{RECIPE_TAG}..geom=tangent..hs=unit.csv")
            self.assertTrue(output.is_file(), proc.stdout + proc.stderr)
            text = output.read_text(encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("auroc_test", text)
        self.assertIn("0.9", text)

    def test_batch_runners_reject_invalid_pair(self):
        commands = (
            ("run_final_benchmark.py", "--only_ds", 29, "--dry_run"),
            ("run_final_parallel.py", "--gpus", 0, "--only_ds", 29, "--dry_run"),
        )
        for name, *base in commands:
            with self.subTest(name=name):
                proc = run_script(
                    name, *base, "--correction_geometry", "tangent",
                    "--head_input_scale", "auto",
                )
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("Unsupported geometry/scale pair", proc.stderr)


if __name__ == "__main__":
    unittest.main()
