import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import torch

from libs.benchmark import (FINAL_CONFIG, UPSTREAM_COMMIT, build_wrapper, contract,
                            final_study_path, restore_params, select_trial, result_path)
from libs.eval import calculate_metric, get_preds_and_probs
from libs.search_space import resolve_dynamics


def dataset(task="binclass"):
    g = torch.Generator().manual_seed(12)
    x = torch.randn(60, 3, generator=g)
    y = ((x[:, 0] > 0).float() if task == "binclass" else
         (torch.arange(60) % 3 if task == "multiclass" else x[:, 0] * 2))
    pairs = [(x[:40], y[:40]), (x[40:50], y[40:50]), (x[50:], y[50:])]
    return SimpleNamespace(tasktype=task, n_classes={"binclass": 2, "multiclass": 3, "regression": None}[task],
                           n_features=3, X_cat=[], X_num=[0, 1, 2], X_cat_cardinality=[],
                           col_names=["a", "b", "c"], y_std=2., _indv_dataset=lambda: pairs)


def study_for(ds, count=100, config=FINAL_CONFIG):
    study = optuna.create_study(direction="minimize" if ds.tasktype == "regression" else "maximize")
    params = dict(embed_dim=8, embedder_layers=1, dropout=0., lr=.001, weight_decay=1e-6,
                  beta_lr_mult=1.0, ema_timescale="legacy_099")
    attrs = {k + "_actual": config[k] for k in
             ("correction_geometry", "head_input_scale", "beta_param", "tie_rule", "early_stop_metric")}
    attrs.update(n_prototypes_actual=6, batch_size_actual=64, benchmark_contract=contract(config, ds),
                 disable_dead_reinit_actual=bool(config["disable_dead_reinit"]), optimize_sha256="test")
    attrs.update(resolve_dynamics(dict(params, batch_size=64), 40))
    distributions = {k: optuna.distributions.CategoricalDistribution([v]) for k, v in params.items()}
    for i in range(count):
        study.add_trial(optuna.trial.create_trial(params=params, distributions=distributions,
                                                 value=i / 101, user_attrs=attrs))
    return study


class BenchmarkTests(unittest.TestCase):
    def test_hpo_entry_point_uses_shared_factory(self):
        import runpy
        import sys
        ds = dataset()
        def short_wrapper(*args, **kwargs):
            wrapper = build_wrapper(*args, **kwargs)
            wrapper.epochs = 2
            return wrapper
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["optimize.py", "--openml_id", "31", "--seed", "2", "--n_trials", "1",
                    "--savepath", tmp, "--gpu_id", "-1"]
            with patch.object(sys, "argv", argv), patch("libs.data.TabularDataset", return_value=ds), \
                    patch("libs.benchmark.build_wrapper", short_wrapper):
                runpy.run_path("optimize.py", run_name="__main__")
            study = joblib.load(final_study_path(tmp, 2, 31))
            self.assertEqual(len(study.trials), 1)
            self.assertEqual(study.best_trial.user_attrs["benchmark_contract"], contract(FINAL_CONFIG, ds))
            self.assertEqual(study.best_trial.params["beta_lr_mult"], 1.0)
            self.assertEqual(study.best_trial.params["ema_timescale"], "legacy_099")
            self.assertEqual(study.best_trial.user_attrs["ema_decay_actual"], 0.99)
            self.assertEqual(len(study.best_trial.user_attrs["beta_epoch_history"]), 2)
            self.assertIn("prediction_diagnostics_val", study.best_trial.user_attrs)
            self.assertIn("acc_test", study.best_trial.user_attrs)
            self.assertTrue(list(Path(tmp).rglob("*.npz")))

    def test_validation_only_blocks_test_evaluation_and_artifacts(self):
        import contextlib
        import io
        import runpy
        import sys
        torch.set_num_threads(1)
        for task, data_id, pilot in ((task, data_id, pilot)
                                    for task, data_id in (("binclass", 31), ("multiclass", 54), ("regression", 537))
                                    for pilot in ("joint", "dynamics2d")):
            ds = dataset(task)
            applied = []
            test_x = ds._indv_dataset()[2][0]
            def short_wrapper(*args, **kwargs):
                applied.append(dict(args[1]))
                wrapper = build_wrapper(*args, **kwargs)
                wrapper.epochs = 2
                forward = wrapper._forward_batched
                def guarded_forward(x, *a, **kw):
                    if torch.equal(x, test_x):
                        raise AssertionError("Pilot evaluated the held-out test split")
                    return forward(x, *a, **kw)
                wrapper._forward_batched = guarded_forward
                return wrapper
            def validation_metric(*args, **kwargs):
                if args[-1] == "test":
                    raise AssertionError("Pilot computed test metrics")
                return calculate_metric(*args, **kwargs)
            with self.subTest(task=task, pilot=pilot), tempfile.TemporaryDirectory() as tmp:
                argv = ["optimize.py", "--openml_id", str(data_id), "--seed", "2",
                        "--n_trials", "1", "--savepath", tmp, "--gpu_id", "-1", "--validation_only",
                        "--pilot_space", pilot]
                output = io.StringIO()
                with patch.object(sys, "argv", argv), \
                     patch("libs.data.TabularDataset", return_value=ds), \
                     patch("libs.benchmark.build_wrapper", short_wrapper), \
                     patch("libs.eval.calculate_metric", side_effect=validation_metric), \
                     patch("numpy.savez_compressed", side_effect=AssertionError("Pilot saved a prediction NPZ")), \
                     contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    runpy.run_path("optimize.py", run_name="__main__")
                paths = list(Path(tmp).rglob("*.pkl"))
                self.assertEqual(len(paths), 1)
                self.assertIn("..validation_only", paths[0].name)
                self.assertFalse(final_study_path(tmp, 2, data_id).exists())
                study = joblib.load(paths[0])
                self.assertTrue(study.user_attrs["validation_only"])
                trial = study.best_trial
                self.assertEqual(study.user_attrs["pilot_space"], pilot)
                if pilot == "dynamics2d":
                    from libs.search_space import LEGACY_ANCHOR
                    self.assertIn("..validation_only..pilot=dynamics2d", paths[0].name)
                    self.assertEqual(set(trial.params), {"beta_lr_mult", "ema_timescale"})
                    self.assertEqual(trial.params, {"beta_lr_mult": 1., "ema_timescale": "legacy_099"})
                    self.assertEqual(trial.user_attrs["fixed_hyperparameters"], LEGACY_ANCHOR)
                    self.assertEqual({k: applied[0][k] for k in LEGACY_ANCHOR}, LEGACY_ANCHOR)
                self.assertFalse(any(k.endswith("_test") for k in trial.user_attrs))
                self.assertEqual(trial.value, trial.user_attrs["rmse_val" if task == "regression" else "acc_val"])
                self.assertFalse(list(Path(tmp).rglob("*.npz")))
                for marker in ("acc_test", "rmse_test", "logloss_test"):
                    self.assertNotIn(marker, output.getvalue())
                    for csv in Path(tmp).rglob("*.csv"):
                        self.assertNotIn(marker, csv.read_text(encoding="utf-8"))
                with self.assertRaisesRegex(ValueError, "pilot study"):
                    select_trial(study, task)
                # A copied/renamed full-evaluation study must not resume as a pilot.
                study.set_user_attr("validation_only", False)
                joblib.dump(study, paths[0])
                with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "evaluation mode differs"):
                    runpy.run_path("optimize.py", run_name="__main__")

    def test_raw_loader_with_continuous_regression_target(self):
        import pandas as pd
        from libs.data import load_data
        frame = pd.DataFrame({"x": [1., 2., 3., 4.]})
        target = pd.Series([.2, 1.8, -2.4, .7])
        raw = SimpleNamespace(name="synthetic", default_target_attribute="y",
                              get_data=lambda **kw: (frame.copy(), target.copy(), [False], ["x"]))
        with patch("openml.datasets.get_dataset", return_value=raw):
            actual = load_data(123, tasktype="regression")
            np.testing.assert_array_equal(actual[1], target.to_numpy())

    def test_ensemble_averages_logits_and_keeps_regression_scale(self):
        from ensemble import combine
        members = [{"Probability": np.array([[-5.], [1.]], dtype=np.float32)},
                   {"Probability": np.array([[1.], [1.]], dtype=np.float32)}]
        pred, prob, logits = combine(members, "binclass")
        np.testing.assert_array_equal(logits, [[-2.], [1.]])
        torch.testing.assert_close(prob[:, 1], torch.sigmoid(torch.tensor([-2., 1.])))
        pred, _, _ = combine([{"Prediction": np.array([.2, 1.5])},
                              {"Prediction": np.array([.4, 2.])}], "regression")
        np.testing.assert_allclose(pred.numpy(), [.3, 1.75])

    def test_trial_selection_and_incomplete_hpo(self):
        for task in ("binclass", "regression"):
            study = study_for(dataset(task))
            self.assertEqual(select_trial(study, task, "init").number, 0)
            self.assertEqual(select_trial(study, task, "hyper", 1).number, 1 if task == "regression" else 98)
            with self.assertRaises(ValueError):
                select_trial(study_for(dataset(task), 2), task)
        ds = dataset()
        trial = study_for(ds).best_trial
        trial.user_attrs["head_input_scale_actual"] = "unit"
        with self.assertRaises(ValueError):
            restore_params(trial, 40)

    def test_official_metric_parity(self):
        repo = Path(__file__).resolve().parents[2] / "multitab"
        if not (repo / ".git").exists():
            self.skipTest("Optional pinned official git objects not available")
        source = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}",
                                          "-C", str(repo), "show", f"{UPSTREAM_COMMIT}:libs/eval.py"], text=True)
        namespace = {}
        exec(compile(source, "official_eval.py", "exec"), namespace)
        for task in ("binclass", "multiclass", "regression", "missing"):
            actual_task = "multiclass" if task == "missing" else task
            y = torch.tensor([0, 1, 0, 1]) if task != "multiclass" else torch.tensor([0, 1, 2, 1])
            logits = torch.tensor([[-1.], [2.], [.5], [1.]]) if actual_task != "multiclass" else torch.tensor([[1., 2., 3.], [2., 4., 1.], [0., 1., 3.], [1., 0., 2.]])
            pred, prob = get_preds_and_probs(logits, actual_task)
            ours = calculate_metric(y, pred, prob, actual_task, "test")
            official_y = np.eye(3)[y.numpy()] if actual_task == "multiclass" else y.numpy()
            official = namespace["calculate_metric"](official_y, pred.numpy(), logits.numpy(), actual_task, "test")
            self.assertEqual(set(ours), set(official))
            for key in ours:
                if official[key] is None or np.isnan(official[key]):
                    self.assertTrue(np.isnan(ours[key]))
                else:
                    self.assertAlmostEqual(ours[key], official[key], places=6)

    def test_official_split_parity(self):
        repo = Path(__file__).resolve().parents[2] / "multitab"
        if not (repo / ".git").exists():
            self.skipTest("Optional pinned official git objects not available")
        source = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
                                          "show", f"{UPSTREAM_COMMIT}:libs/data.py"], text=True)
        tree = ast.parse(source)
        names = {"one_hot", "split_data"}
        functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import LabelEncoder
        from libs.data import split_data
        def prep(*args, **kwargs):
            a, b, c, d, e, f = args[:6]
            return (a, d), (b, e), (c, f), 1.
        ns = dict(np=np, torch=torch, KFold=KFold, LabelEncoder=LabelEncoder, prep_data=prep)
        exec(compile(functions, "official_split.py", "exec"), ns)
        x = np.arange(1599, dtype=np.float32).reshape(-1, 1)
        y = np.arange(1599) % 100
        for seed in (0, 2, 9):
            official = ns["split_data"](x, y, "multiclass", seed=seed, device="cpu")
            ours = split_data(x, y, "multiclass", seed=seed, device="cpu")
            for actual, expected in zip(ours[:3], official[:3]):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b)

    def test_cached_1493_preprocessing_parity(self):
        import pandas as pd
        from libs.data import load_data, split_data
        root = Path(__file__).resolve().parents[1]
        repo = root.parent / "multitab"
        cache = root / "data_cache"
        if not (repo / ".git").exists() or not (cache / "1493_X.parquet").exists():
            self.skipTest("Optional official git objects or cached dataset unavailable")
        source = subprocess.check_output(["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo),
                                          "show", f"{UPSTREAM_COMMIT}:libs/data.py"], text=True)
        # Upstream assumes CUDA. Adapt device spelling only for CPU parity.
        source = source.replace("device = X_train.get_device()", "device = X_train.device")
        official = {}
        exec(compile(source, "official_data.py", "exec"), official)
        frame = pd.read_parquet(cache / "1493_X.parquet")
        # dtype=object reproduces what openml hands upstream's load_data. On
        # pandas 3 a bare pd.Series of these strings becomes an ArrowStringArray,
        # whose .values has no reshape, so upstream's `y.values.reshape(-1)`
        # raises before any comparison happens -- a fixture artefact of the
        # local pandas, not a parity difference.
        y = pd.Series(np.load(cache / "1493_y.npy", allow_pickle=True).reshape(-1), dtype=object)
        cats = [isinstance(d, pd.CategoricalDtype) for d in frame.dtypes]
        raw = SimpleNamespace(name="cached1493", default_target_attribute="y",
                              get_data=lambda **kw: (frame.copy(), y.copy(), cats, frame.columns.tolist()))
        with patch("openml.datasets.get_dataset", return_value=raw):
            ours = load_data(1493, "multiclass")
            theirs = official["load_data"](1493)
        for a, b in zip(ours[:5], theirs):
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        ours_split = split_data(ours[0], ours[1], "multiclass", num_indices=ours[4], seed=2, device="cpu")
        their_split = official["split_data"](theirs[0], theirs[1], "multiclass", num_indices=theirs[4], seed=2, device="cpu")
        for actual, expected in zip(ours_split[:3], their_split[:3]):
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b)

    def test_reassign_groups_leaves_model_state_untouched(self):
        # The post-training resync must be a pure reassignment. The earlier
        # path called the per-epoch regroup_update(), which could relocate a
        # dead centroid from the global RNG *after* the best checkpoint had
        # been restored -- so the evaluated model was not the saved one, and
        # two analyses of one checkpoint could disagree.
        torch.manual_seed(0)
        ds = dataset("binclass")
        params = dict(embed_dim=8, embedder_layers=1, dropout=0., lr=.001, weight_decay=1e-6,
                      n_prototypes=6, batch_size=64)
        pl = build_wrapper(ds, params, FINAL_CONFIG, "cpu").model.prototype_layer
        # Put every streak at the reinit threshold: the old path would have
        # relocated centroids here; the pure path must not.
        pl.dead_streak.fill_(int(pl.dead_reinit_patience))
        X = torch.randn(40, pl.centroid_emb.shape[1])
        state = {n: t.detach().clone() for n, t in list(pl.named_buffers()) + list(pl.named_parameters())}
        epoch = int(pl.current_epoch.item())
        stats = pl.reassign_groups(X)

        def same(a, b):   # NaN-safe: centroid_labels is a NaN-filled float buffer
            if a.is_floating_point():
                return a.shape == b.shape and bool(((a == b) | (a.isnan() & b.isnan())).all())
            return torch.equal(a, b)
        for n, t in list(pl.named_buffers()) + list(pl.named_parameters()):
            self.assertTrue(same(state[n], t.detach()), f"{n} changed during resync")
        self.assertEqual(int(pl.current_epoch.item()), epoch)
        self.assertEqual(stats["reinit_count"], 0)
        # Every sample lands in exactly one group, by the routing rule itself.
        self.assertEqual(sorted(i for g in pl.sample_groups for i in g), list(range(40)))
        q = torch.nn.functional.normalize(X, dim=-1)
        c = torch.nn.functional.normalize(pl.centroid_emb, dim=-1)
        expect = (q @ c.T).argmax(-1)
        for p, g in enumerate(pl.sample_groups):
            for i in g:
                self.assertEqual(int(expect[i]), p)

    def test_disable_dead_reinit_arm_is_separate_and_never_reinits(self):
        # The ablation arm must (1) resolve to its own study and result files
        # without touching the main arm's names, (2) reach the model, and (3)
        # never relocate a centroid even when every dead streak is far past
        # the default patience -- the exact thing the main arm does do.
        import reproduce
        from libs.benchmark import arm_config, final_study_path
        torch.set_num_threads(1)
        ds = dataset("binclass")
        main_cfg, nodr_cfg = arm_config(False), arm_config(True)
        self.assertEqual(main_cfg, FINAL_CONFIG)
        self.assertIn("..nodr", str(final_study_path("r", 1, 31, nodr_cfg)))
        self.assertNotIn("nodr", str(final_study_path("r", 1, 31)))
        self.assertIn("..nodr", str(result_path("r", 1, 31, config=nodr_cfg)))
        self.assertNotIn("nodr", str(result_path("r", 1, 31)))

        params = dict(embed_dim=8, embedder_layers=1, dropout=0., lr=.001, weight_decay=1e-6,
                      n_prototypes=6, batch_size=64)
        for cfg, expect_reinit in ((main_cfg, True), (nodr_cfg, False)):
            torch.manual_seed(0)
            pl = build_wrapper(ds, params, cfg, "cpu").model.prototype_layer
            pl.current_epoch.fill_(int(pl.regroup_warmup_epochs) + 5)   # past warmup
            pl.dead_streak.fill_(10_000)                                # every streak far past patience
            # The dead check only counts prototypes that receive *no* assignment
            # in this call, so route every sample to prototype 0: 1..5 are dead.
            X = pl.centroid_emb[0].detach().repeat(40, 1) + 1e-3 * torch.randn(40, 8)
            before = pl.centroid_emb.detach().clone()
            stats = pl.regroup_update(X)
            moved = not torch.equal(before, pl.centroid_emb.detach())
            self.assertEqual(int(stats.get("reinit_count", 0)) > 0, expect_reinit, cfg["disable_dead_reinit"])
            self.assertEqual(moved, expect_reinit, "centroid movement must follow the arm")
        self.assertGreaterEqual(int(build_wrapper(ds, params, nodr_cfg, "cpu").model.prototype_layer.dead_reinit_patience), 10**9)

        # End to end through reproduce.py: the flag reaches the model and the
        # result lands on the arm's own path with the arm's own contract.
        study = study_for(ds, config=nodr_cfg)
        captured = []
        def short_wrapper(*a, **k):
            w = build_wrapper(*a, **k); w.epochs = 3; captured.append(w); return w
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "datasets.json"
            metadata.write_text('{"1": {"tasktype": "binclass"}}')
            source = final_study_path(tmp, 2, 1, nodr_cfg)
            source.parent.mkdir(parents=True)
            joblib.dump(study, source)
            args = reproduce.parser().parse_args(["--openml_id", "1", "--seed", "2", "--savepath", tmp,
                                                  "--json", str(metadata), "--gpu_id", "-1",
                                                  "--disable_dead_reinit"])
            with patch("libs.data.TabularDataset", return_value=ds), patch("libs.benchmark.build_wrapper", short_wrapper):
                reproduce.run(args)
            out = result_path(tmp, 2, 1, config=nodr_cfg)
            self.assertTrue(out.exists())
            self.assertFalse(result_path(tmp, 2, 1).exists())
            saved = np.load(out, allow_pickle=True).item()
            self.assertTrue(saved["identity"]["contract"]["config"]["disable_dead_reinit"])
            pl = captured[-1].model.prototype_layer
            self.assertGreaterEqual(int(pl.dead_reinit_patience), 10**9)
            self.assertEqual(int(pl._diag_reinit_total.sum()), 0)

    def test_fixed_hp_ablation_reuses_main_study_for_the_nodr_model(self):
        # --hpo_source main: the main arm's study supplies the hyperparameters,
        # the nodr config builds the model, both contracts are recorded, and the
        # result lands on its own path. A nodr study must be refused there, and
        # the main arm's own result path must stay empty.
        import reproduce
        from libs.benchmark import arm_config, final_study_path, restore_params
        torch.set_num_threads(1)
        ds = dataset("binclass")
        main_cfg, nodr_cfg = arm_config(False), arm_config(True)
        main_study, nodr_study = study_for(ds), study_for(ds, config=nodr_cfg)
        # restore_params checks the arm the trial was searched under.
        restore_params(main_study.best_trial, 40, nodr_cfg, main_cfg)
        with self.assertRaises(ValueError):
            restore_params(nodr_study.best_trial, 40, nodr_cfg, main_cfg)
        captured = []
        def short_wrapper(*a, **k):
            w = build_wrapper(*a, **k); w.epochs = 3; captured.append(w); return w
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "datasets.json"
            metadata.write_text('{"1": {"tasktype": "binclass"}}')
            source = final_study_path(tmp, 2, 1, main_cfg)
            source.parent.mkdir(parents=True)
            joblib.dump(main_study, source)
            args = reproduce.parser().parse_args(["--openml_id", "1", "--seed", "2", "--savepath", tmp,
                                                  "--json", str(metadata), "--gpu_id", "-1",
                                                  "--disable_dead_reinit", "--hpo_source", "main"])
            with patch("libs.data.TabularDataset", return_value=ds), patch("libs.benchmark.build_wrapper", short_wrapper):
                reproduce.run(args)
            out = result_path(tmp, 2, 1, config=nodr_cfg, hpo_source="main")
            self.assertTrue(out.exists())
            self.assertFalse(result_path(tmp, 2, 1).exists())
            self.assertFalse(result_path(tmp, 2, 1, config=nodr_cfg).exists())
            saved = np.load(out, allow_pickle=True).item()
            idt = saved["identity"]
            self.assertEqual(idt["hpo_source"], "main")
            self.assertTrue(idt["contract"]["config"]["disable_dead_reinit"])
            self.assertFalse(idt["hpo_contract"]["config"]["disable_dead_reinit"])
            self.assertEqual(idt["optimize_sha256"], "test")
            self.assertIn("active_ratio", saved["prototype_diag"])
            self.assertGreaterEqual(int(captured[-1].model.prototype_layer.dead_reinit_patience), 10**9)
            # The early-stop-metric arm goes through the same path: own tag,
            # main study, the metric reaches the wrapper.
            captured.clear()
            esm_cfg = arm_config(early_stop_metric="logloss")
            self.assertIn("..esm=logloss", str(final_study_path("r", 1, 31, esm_cfg)))
            self.assertIn("..esm=logloss..hpo=main", str(result_path("r", 1, 31, config=esm_cfg, hpo_source="main")))
            args = reproduce.parser().parse_args(["--openml_id", "1", "--seed", "2", "--savepath", tmp,
                                                  "--json", str(metadata), "--gpu_id", "-1",
                                                  "--early_stop_metric", "logloss", "--hpo_source", "main"])
            with patch("libs.data.TabularDataset", return_value=ds), patch("libs.benchmark.build_wrapper", short_wrapper):
                reproduce.run(args)
            self.assertTrue(result_path(tmp, 2, 1, config=esm_cfg, hpo_source="main").exists())
            self.assertEqual(captured[-1].early_stop_metric, "logloss")
            self.assertEqual(captured[-1]._sel_key, "logloss_val")
            saved = np.load(result_path(tmp, 2, 1, config=esm_cfg, hpo_source="main"), allow_pickle=True).item()
            self.assertEqual(saved["identity"]["contract"]["config"]["early_stop_metric"], "logloss")
            self.assertEqual(saved["identity"]["hpo_contract"]["config"]["early_stop_metric"], FINAL_CONFIG["early_stop_metric"])
            # Without --hpo_source main, the same flag must look for the nodr study and refuse.
            args = reproduce.parser().parse_args(["--openml_id", "1", "--seed", "2", "--savepath", tmp,
                                                  "--json", str(metadata), "--gpu_id", "-1", "--disable_dead_reinit"])
            with patch("libs.data.TabularDataset", return_value=ds), self.assertRaises(FileNotFoundError):
                reproduce.run(args)

    def test_val_loss_early_stopping_has_multitab_semantics(self):
        # FINAL_CONFIG now stops on the batch-averaged validation loss and
        # evaluates the terminal model, as MultiTab's neural baselines do:
        #   (1) the selection key is val_loss and the monitor is the training
        #       loss function averaged over fixed-order validation batches,
        #   (2) training stops exactly `patience` epochs after the loss
        #       minimum (strict improvement, counter >= patience),
        #   (3) no restore: the model returned is the terminal one, so its
        #       recomputed validation loss equals the last history entry, not
        #       the minimum.
        torch.set_num_threads(1)
        self.assertEqual(FINAL_CONFIG["early_stop_metric"], "val_loss")
        ds = dataset("binclass")
        params = dict(embed_dim=8, embedder_layers=1, dropout=0., lr=.05, weight_decay=1e-6,
                      n_prototypes=6, batch_size=64)
        torch.manual_seed(0)
        w = build_wrapper(ds, params, FINAL_CONFIG, "cpu")
        w.epochs, w.patience = 60, 3
        self.assertTrue(w.terminal_checkpoint)
        self.assertEqual(w._sel_key, "val_loss")
        (xt, yt), (xv, yv), _ = ds._indv_dataset()
        w.fit(xt, yt, xv, yv)
        hist = w.val_loss_history
        self.assertEqual(len(hist), w.last_epoch)
        best = int(np.argmin(hist))                      # first minimum, strict '<'
        self.assertEqual(w.best_metric_epoch, best + 1)
        if w.last_epoch < w.epochs:                      # stopped early
            self.assertEqual(w.last_epoch - (best + 1), w.patience)
        # (3) terminal weights: recompute the MultiTab-style loss on the
        # returned model and compare with the last entry.
        w.model.eval()
        with torch.no_grad():
            lg = w._forward_batched(xv)
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(
                      lg[s:s + 64].view(yv[s:s + 64].shape), yv[s:s + 64].float())
                  for s in range(0, len(yv), 64)]
        self.assertAlmostEqual(float(torch.stack(losses).mean()), hist[-1], places=5)
        if w.last_epoch < w.epochs:
            self.assertNotAlmostEqual(hist[-1], min(hist), places=6)

    def test_val_loss_nan_does_not_fall_back_to_accuracy(self):
        torch.set_num_threads(1)
        ds = dataset("binclass")
        params = dict(embed_dim=8, embedder_layers=1, dropout=0., lr=.001,
                      weight_decay=1e-6, n_prototypes=6, batch_size=64)
        torch.manual_seed(0)
        w = build_wrapper(ds, params, FINAL_CONFIG, "cpu")
        w.epochs, w.patience = 10, 3
        (xt, yt), (xv, yv), _ = ds._indv_dataset()
        # Keep training finite while validation loss is NaN and accuracy
        # improves every epoch. Falling back would prevent the expected stop.
        with patch.object(w, "_forward_batched", return_value=torch.full((len(yv), 1), float("nan"))), \
             patch("libs.supervised.compute_metric", side_effect=[{"acc_val": i / 10} for i in range(10)]):
            w.fit(xt, yt, xv, yv)
        self.assertEqual(w.last_epoch, 4)
        self.assertEqual(w.best_epoch, 4)
        self.assertTrue(np.isnan(w.val_loss_history).all())
        self.assertFalse(w._sel_fallback_warned)
        self.assertIsNone(w._best_state)

    def test_early_stopping_counter_matches_upstream_multitab(self):
        # Transcription of multitab/libs/supervised.py EarlyStopping.on_epoch_end:
        # the first epoch sets best_value AND resets the counter, strict '<'
        # improvement, stop when counter >= patience. TabERA's step() must
        # stop on the same epoch for every curve -- including the edge case
        # where epoch 1 is the minimum for the whole run (stop at 1 + patience).
        from libs.supervised import EarlyStopping
        class Upstream:
            def __init__(self, patience):
                self.patience, self.best_value, self.counter = patience, None, 0
            def step(self, v):
                if self.best_value is None or v < self.best_value:
                    self.best_value, self.counter = v, 0
                else:
                    self.counter += 1
                return self.counter >= self.patience
        def stop_epoch(es, curve, call):
            for ep, v in enumerate(curve, 1):
                if call(es, v):
                    return ep
            return None
        rng = np.random.default_rng(0)
        curves = [[0.30] + [0.31 + 0.001 * i for i in range(60)],          # epoch-1 minimum
                  [0.5, 0.4, 0.4, 0.4, 0.35] + [0.36] * 40,                 # plateau (ties do not reset)
                  list(rng.random(80)), list(np.linspace(1.0, 0.1, 50)),  # random / never stops
                  [float("nan")] + [0.1] * 40,
                  [0.5, float("nan"), 0.4] + [float("nan")] * 40]
        for patience in (1, 3, 20):
            for curve in curves:
                up = stop_epoch(Upstream(patience), curve, lambda es, v: es.step(v))
                tb = stop_epoch(EarlyStopping(patience=patience), curve, lambda es, v: es.step(v, False))
                self.assertEqual(tb, up, f"patience={patience} curve={curve[:5]}")
        self.assertEqual(stop_epoch(EarlyStopping(patience=20), curves[0], lambda es, v: es.step(v, False)), 21)

    def test_cpu_reproduction_save_resume_and_conflict(self):
        import reproduce
        torch.set_num_threads(1)
        for task in ("binclass", "multiclass", "regression"):
            ds = dataset(task)
            study = study_for(ds)
            def short_wrapper(*args, **kwargs):
                wrapper = build_wrapper(*args, **kwargs)
                wrapper.epochs = 2
                return wrapper
            with tempfile.TemporaryDirectory() as tmp:
                metadata = Path(tmp) / "datasets.json"
                metadata.write_text('{"1": {"tasktype": "' + task + '"}}')
                source = final_study_path(tmp, 2, 1)
                source.parent.mkdir(parents=True)
                joblib.dump(study, source)
                args = reproduce.parser().parse_args(["--openml_id", "1", "--seed", "2", "--savepath", tmp,
                                                      "--json", str(metadata), "--gpu_id", "-1"])
                with patch("libs.data.TabularDataset", return_value=ds), patch("libs.benchmark.build_wrapper", short_wrapper):
                    reproduce.run(args)
                    path = result_path(tmp, 2, 1)
                    saved = np.load(path, allow_pickle=True).item()
                    self.assertEqual(saved["Prediction"].shape, (10,))
                    self.assertTrue(saved["time"] > 0)
                    self.assertEqual(saved["dynamics_provenance"]["ema_decay_actual"], 0.99)
                    self.assertTrue(saved["training_diagnostics"]["beta_epoch_history"])
                    if task != "regression":
                        pred, probs = get_preds_and_probs(torch.tensor(saved["Probability"]), task)
                        perf = calculate_metric(ds._indv_dataset()[2][1], pred, probs, task, "test")
                        for key, value in perf.items():
                            self.assertAlmostEqual(value, saved["Performance"][key])
                    with patch("libs.benchmark.build_wrapper", side_effect=AssertionError("must skip")):
                        reproduce.run(args)
                    saved["identity"]["train_seed"] = -1
                    np.save(path, saved)
                    with self.assertRaises(ValueError):
                        reproduce.run(args)


if __name__ == "__main__":
    unittest.main()
