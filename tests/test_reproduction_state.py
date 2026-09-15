import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from build_explanation_manifest import build_manifest
from libs.benchmark import build_wrapper, contract
from libs.benchmark_config import FINAL_CONFIG
from libs.eval import get_preds_and_probs
from libs.reproduction_state import (compare_predictions, refresh_training_memory,
                                     restore_checkpoint, save_checkpoint, snapshot)
from tests.test_benchmark import dataset


class ReproductionStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def make_state(self, task):
        ds = dataset(task)
        cfg = dict(FINAL_CONFIG, correction_geometry="tangent", head_input_scale="unit")
        params = dict(embed_dim=8, embedder_layers=1, dropout=0.1, lr=.001, weight_decay=1e-6,
                      beta_lr_mult=1., ema_timescale="hl_1", num_bins=8, ple_d_embedding=12,
                      batch_size=16, n_prototypes=6, correction_geometry="tangent",
                      head_input_scale="unit", beta_param="sigmoid")
        torch.manual_seed(10)
        w = build_wrapper(ds, params, cfg, "cpu")
        w.epochs = 2
        w.regroup_log_every = 0
        (xt, yt), (xv, yv), (xe, ye) = ds._indv_dataset()
        w.fit(xt, yt, xv, yv)
        w.model.eval()
        with torch.no_grad():
            logits = w._forward_batched(xe)
        preds, _ = get_preds_and_probs(logits, task)
        identity = dict(dataset_id=31, fold=1, train_seed=10, tasktype=task,
                        params=params, contract=contract(cfg, ds), unverified_hpo=False)
        return w, ds, snapshot(w, ds, identity, logits, preds)

    def test_binary_multiclass_roundtrip_original_memory_and_predictions(self):
        for task in ("binclass", "multiclass"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                w, ds, state = self.make_state(task)
                xe, ye = ds._indv_dataset()[2]
                with torch.no_grad():
                    before = w.model(xe)
                path = Path(tmp) / "checkpoint.pt"
                save_checkpoint(path, state)
                restored, rds, _ = restore_checkpoint(path)
                with torch.no_grad():
                    after = restored.model(xe)
                for key in ("logits", "centroid_id", "topk_idx", "neighbor_mask"):
                    torch.testing.assert_close(after[key], before[key], rtol=0, atol=0)
                self.assertEqual(restored.model.prototype_layer.sample_groups, w.model.prototype_layer.sample_groups)
                torch.testing.assert_close(restored.model.memory.sample_ids, w.model.memory.sample_ids)
                torch.testing.assert_close(restored.model.feature_store._store, w.model.feature_store._store)
                self.assertEqual(contract(state["model_config"], rds), state["identity"]["contract"])
                refresh = refresh_training_memory(restored, rds)
                self.assertTrue(refresh["parameter_match"])
                self.assertTrue(refresh["centroid_match"])
                self.assertTrue(refresh["training_sample_id_unique"])
                self.assertTrue(refresh["training_sample_id_complete"])
                self.assertTrue(refresh["region_membership_complete"])
                with torch.no_grad():
                    refreshed_out = restored.model(xe)
                torch.testing.assert_close(refreshed_out["logits"], before["logits"], rtol=0, atol=0)
                torch.testing.assert_close(
                    restored.model.memory.sample_ids, torch.arange(len(ds._indv_dataset()[0][1])))
                refreshed_state = snapshot(
                    restored, rds, state["identity"], refreshed_out["logits"],
                    get_preds_and_probs(refreshed_out["logits"], task)[0],
                    state_kind="final_encoder_refreshed_for_explanation",
                    parent_checkpoint_sha256="test-parent", refresh_audit=refresh)
                refreshed_path = Path(tmp) / "checkpoint_refreshed.pt"
                save_checkpoint(refreshed_path, refreshed_state)
                restored_again, _, payload = restore_checkpoint(refreshed_path)
                self.assertEqual(payload["state_kind"], "final_encoder_refreshed_for_explanation")
                with torch.no_grad():
                    roundtrip_out = restored_again.model(xe)
                for key in ("logits", "centroid_id", "topk_idx", "neighbor_mask"):
                    torch.testing.assert_close(roundtrip_out[key], refreshed_out[key], rtol=0, atol=0)
                with self.assertRaises(FileExistsError):
                    save_checkpoint(path, state)
                bad = copy.deepcopy(state)
                del bad["state_dict"]["dev_head.weight"]
                save_checkpoint(Path(tmp) / "bad.pt", bad)
                with self.assertRaises(RuntimeError):
                    restore_checkpoint(Path(tmp) / "bad.pt")

    def test_same_accuracy_does_not_pass_reproduction(self):
        audit = compare_predictions(np.array([[1.], [-1.]]), [1, 0],
                                    np.array([[-1.], [1.]]), [0, 1], [1, 1])
        self.assertEqual(audit["accuracy_reference"], audit["accuracy_recomputed"])
        self.assertFalse(audit["passed"])
        with self.assertRaises(ValueError):
            compare_predictions([np.nan], [0], [0.], [0], [0])

    def test_memory_refresh_runner_preserves_original_and_saves_separate_state(self):
        from build_explanation_manifest import sha256
        from refresh_explanation_checkpoint import parser, run

        _, _, state = self.make_state("binclass")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "openml_31/fold_1"
            run_dir.mkdir(parents=True)
            original = run_dir / "checkpoint.pt"
            save_checkpoint(original, state)
            original_sha = sha256(original)
            common = dict(status="reproduction_passed", checkpoint_sha256=original_sha,
                          checkpoint_roundtrip={"passed": True},
                          benchmark_reproduction={"passed": True},
                          benchmark_accuracy_consistent=True,
                          eligible_for_memory_refresh=True, eligible_for_explanation=False)
            for mode, name in (("train", "audit_train.json"),
                               ("restore_only", "audit_restore.json")):
                (run_dir / name).write_text(
                    json.dumps(dict(common, execution_mode=mode)), encoding="utf-8")
            args = parser().parse_args(["--analysis-root", tmp, "--dataset-id", "31",
                                       "--fold", "1", "--gpu-id", "-1"])
            self.assertEqual(run(args), 0)
            self.assertEqual(sha256(original), original_sha)
            self.assertTrue((run_dir / "checkpoint_refreshed.pt").is_file())
            audit = json.loads((run_dir / "audit_memory_refresh.json").read_text())
            self.assertEqual(audit["status"], "memory_refresh_passed")
            self.assertTrue(audit["eligible_for_explanation"])
            self.assertTrue(audit["original_state_preserved"])
            self.assertTrue(audit["refreshed_state_saved_separately"])
            self.assertEqual(audit["prediction_invariance"]["max_abs_logit_error"], 0.0)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_manifest_and_runner_restore_with_reference_checks(self):
        from reproduce_with_checkpoint import parser, run
        w, ds, state = self.make_state("binclass")
        result = dict(identity=state["identity"], Prediction=state["test_preds"].numpy(),
                      Probability=state["test_logits"].numpy(), probability_representation="logits",
                      environment=state["environment"], Performance={"acc_test": float(
                          (state["test_preds"] == ds._indv_dataset()[2][1]).float().double().mean())})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = io.BytesIO()
            np.save(raw, result)
            archive = root / "results.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                member = tarfile.TarInfo("reproduce_logs/run.npy")
                member.size = len(raw.getvalue())
                tar.addfile(member, io.BytesIO(raw.getvalue()))
            out = root / "analysis"
            manifest = build_manifest(archive, out, expected_runs=1)
            row = manifest["runs"][0]
            self.assertEqual((row["fold"], row["train_seed"]), (1, 10))
            save_checkpoint(out / "openml_31/fold_1/checkpoint.pt", state)
            args = parser().parse_args(["--manifest", str(out / "manifest.json"), "--dataset-id", "31",
                                       "--fold", "1", "--gpu-id", "-1", "--restore-only"])
            self.assertEqual(run(args), 0)
            audit = json.loads((out / "openml_31/fold_1/audit_restore.json").read_text())
            self.assertEqual(audit["execution_mode"], "restore_only")
            self.assertTrue(audit["eligible_for_memory_refresh"])
            self.assertFalse(audit["eligible_for_explanation"])
            # Exercise the real train -> save -> restore path with the same
            # small two-epoch fixture, independently of the pre-saved state.
            train_args = parser().parse_args([
                "--manifest", str(out / "manifest.json"), "--dataset-id", "31", "--fold", "1",
                "--gpu-id", "-1", "--train", "--output", str(root / "new_run")])
            def short_wrapper(*args, **kwargs):
                wrapper = build_wrapper(*args, **kwargs)
                wrapper.epochs = 2
                wrapper.regroup_log_every = 0
                return wrapper
            with patch("libs.data.TabularDataset", return_value=ds), \
                 patch("libs.benchmark.build_wrapper", side_effect=short_wrapper):
                preflight_args = copy.copy(train_args)
                preflight_args.train = False
                self.assertEqual(run(preflight_args), 0)
                preflight_bytes = (root / "new_run/preflight.json").read_bytes()
                self.assertEqual(run(train_args), 0)
            self.assertTrue((root / "new_run/checkpoint.pt").is_file())
            train_bytes = (root / "new_run/audit_train.json").read_bytes()
            self.assertEqual(json.loads(train_bytes)["execution_mode"], "train")
            restore_args = copy.copy(train_args)
            restore_args.train = False
            restore_args.restore_only = True
            self.assertEqual(run(restore_args), 0)
            self.assertEqual((root / "new_run/audit_train.json").read_bytes(), train_bytes)
            self.assertEqual((root / "new_run/preflight.json").read_bytes(), preflight_bytes)
            self.assertFalse((root / "new_run/audit.json").exists())
            with patch("libs.reproduction_state.current_environment", return_value={"torch": "different"}), \
                 patch("libs.data.TabularDataset", side_effect=AssertionError("must fail before loading data")):
                self.assertEqual(run(restore_args), 2)
            blocked = json.loads((root / "new_run/audit_restore.json").read_text())
            self.assertEqual(blocked["status"], "blocked_provenance")
            self.assertEqual((root / "new_run/audit_train.json").read_bytes(), train_bytes)
            self.assertEqual((root / "new_run/preflight.json").read_bytes(), preflight_bytes)


if __name__ == "__main__":
    unittest.main()
