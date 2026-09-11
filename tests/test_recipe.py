import math
import unittest
from unittest.mock import patch

import optuna
import torch

from libs.benchmark import (FINAL_CONFIG, build_wrapper, restore_params,
                            training_diagnostics, final_study_path, result_path)
from libs.search_space import (suggest_initial_trial, get_search_space,
                               resolve_dynamics, RECIPE_TAG)
from libs.supervised import classification_margin_diagnostics
from tests.test_benchmark import dataset, study_for


class RecipeTests(unittest.TestCase):
    def test_dynamics2d_samples_only_two_parameters(self):
        from libs.search_space import LEGACY_ANCHOR
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        study.enqueue_trial(suggest_initial_trial("dynamics2d"))
        seen = []
        for i in range(5):
            trial = study.ask()
            params = get_search_space(trial, n_train=800, pilot_space="dynamics2d")
            self.assertEqual({k: params[k] for k in LEGACY_ANCHOR}, LEGACY_ANCHOR)
            self.assertEqual(set(trial.params), {"beta_lr_mult", "ema_timescale"})
            self.assertEqual(params["batch_size"], 64)
            seen.append(trial.params)
            study.tell(trial, .5)
        self.assertEqual(seen[0], dict(beta_lr_mult=1., ema_timescale="hl_1"))
        self.assertGreater(len({p["beta_lr_mult"] for p in seen}), 1)

    def test_dynamics2d_requires_validation_only(self):
        import contextlib
        import io
        import runpy
        import sys
        with patch.object(sys, "argv", ["optimize.py", "--pilot_space", "dynamics2d"]), \
             contextlib.redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit) as stopped:
            runpy.run_path("optimize.py", run_name="__main__")
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("requires --validation_only", output.getvalue())

    def test_anchor_and_distributions(self):
        anchor = dict(embed_dim=128, embedder_layers=2, dropout=.1, lr=3e-4,
                      weight_decay=1e-5, beta_lr_mult=1., ema_timescale="hl_1",
                      num_bins=8, ple_d_embedding=12)
        self.assertEqual(suggest_initial_trial(), anchor)
        study = optuna.create_study()
        study.enqueue_trial(anchor)
        trial = study.ask()
        params = get_search_space(trial, n_train=800)
        self.assertEqual(trial.params, anchor)
        self.assertEqual(params["batch_size"], 64)
        self.assertTrue(trial.distributions["beta_lr_mult"].log)
        self.assertIsInstance(trial.distributions["beta_lr_mult"], optuna.distributions.FloatDistribution)
        self.assertEqual(trial.distributions["ema_timescale"].choices,
                         ("hl_05", "hl_1", "hl_3", "hl_10"))
        self.assertEqual(trial.distributions["embed_dim"].choices, (64, 128, 256))
        self.assertEqual(trial.distributions["num_bins"], optuna.distributions.IntDistribution(2, 128))
        self.assertEqual(trial.distributions["ple_d_embedding"], optuna.distributions.IntDistribution(8, 32, step=4))
        self.assertEqual(len(trial.params), 9)
        self.assertIn(RECIPE_TAG, str(final_study_path("r", 1, 31)))
        self.assertIn(RECIPE_TAG, str(result_path("r", 1, 31)))
        self.assertIn("..recipe=betaema1_plehpo_epochhl..", str(final_study_path("r", 1, 31)))
        self.assertNotIn("..recipe=betaema1_plehpo..", str(final_study_path("r", 1, 31)))
        self.assertNotIn("..recipe=betaema1..", str(result_path("r", 1, 31)))

    def test_ple_trial_roundtrip_and_train_only_edges(self):
        import joblib
        import tempfile
        from pathlib import Path
        ds = dataset()
        train_x = ds._indv_dataset()[0][0]
        # Held-out extremes must never affect the bin boundaries.
        ds._indv_dataset()[1][0].fill_(1000)
        ds._indv_dataset()[2][0].fill_(-1000)
        for bins, dim in ((2, 8), (37, 20), (128, 32)):
            trial = study_for(ds, count=1).best_trial
            trial.params.update(num_bins=bins, ple_d_embedding=dim)
            trial.user_attrs.update(num_bins_actual=bins, ple_d_embedding_actual=dim)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trial.pkl"
                joblib.dump(trial, path)
                restored = restore_params(joblib.load(path), 40)
            torch.manual_seed(19)
            hpo = build_wrapper(ds, dict(trial.params, n_prototypes=6, batch_size=64), FINAL_CONFIG, "cpu")
            torch.manual_seed(19)
            repro = build_wrapper(ds, restored, FINAL_CONFIG, "cpu")
            emb = repro.model.embedder
            self.assertEqual(tuple(emb.ple_emb_weight.shape), (3, bins, dim))
            self.assertEqual(emb.final_proj[1].in_features, 3 * dim)
            expected = torch.quantile(train_x, torch.linspace(0, 1, bins + 1), dim=0).T
            torch.testing.assert_close(emb.ple_edges, expected)
            torch.testing.assert_close(emb.ple_edges, hpo.model.embedder.ple_edges)
            self.assertEqual(repro.encoding_provenance["num_bins_actual"], bins)
            self.assertEqual(repro.encoding_provenance["ple_d_embedding_actual"], dim)
            loss = repro.model.embedder(train_x).square().mean()
            loss.backward()
            self.assertTrue(torch.isfinite(emb.ple_emb_weight.grad).all())
            with self.assertRaisesRegex(ValueError, "Precomputed PLE edges"):
                build_wrapper(ds, restored, FINAL_CONFIG, "cpu", torch.zeros(3, bins + 2))
            trial.user_attrs["num_bins_actual"] = 8 if bins != 8 else 9
            with self.assertRaisesRegex(ValueError, "num_bins_actual"):
                restore_params(trial, 40)

    def test_old_fixed_ple_trial_cannot_be_relabelled(self):
        trial = study_for(dataset(), count=1).best_trial
        trial.params.pop("ple_d_embedding")
        trial.user_attrs["benchmark_contract"]["recipe"] = "..recipe=betaema1"
        with self.assertRaisesRegex(ValueError, "ple_d_embedding"):
            restore_params(trial, 40)

    def test_half_life_including_partial_batches(self):
        for n, batch in ((118, 64), (800, 64), (104050, 1024)):
            for label, half_life in (("hl_05", .5), ("hl_1", 1), ("hl_3", 3), ("hl_10", 10)):
                p = dict(batch_size=batch, lr=.002, beta_lr_mult=4.2, ema_timescale=label)
                d = resolve_dynamics(p, n)
                steps = math.ceil(n / batch)
                self.assertEqual(d["steps_per_epoch"], steps)
                self.assertAlmostEqual(d["ema_decay_actual"] ** (steps * half_life), .5)
                self.assertAlmostEqual(d["ema_half_life_epochs_actual"], half_life)
                self.assertAlmostEqual(d["beta_lr_actual"], .0084)
        default_dynamics = resolve_dynamics(dict(batch_size=64, lr=.001), 118)
        self.assertEqual(default_dynamics["ema_timescale"], "hl_1")
        self.assertAlmostEqual(default_dynamics["ema_half_life_epochs_actual"], 1.0)
        for invalid in ("legacy_099", "typo"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                resolve_dynamics(dict(batch_size=64, lr=.001, ema_timescale=invalid), 118)

    def test_applied_groups_and_restore_guard(self):
        torch.set_num_threads(1)
        ds = dataset()
        trial = study_for(ds).best_trial
        trial.params.update(beta_lr_mult=4.2, ema_timescale="hl_3")
        trial.user_attrs.update(resolve_dynamics(dict(trial.params, batch_size=64), 40))
        p = restore_params(trial, 40)
        wrapper = build_wrapper(ds, p, FINAL_CONFIG, "cpu")
        self.assertEqual(wrapper.beta_lr_mult, 4.2)
        self.assertEqual(wrapper.model.prototype_layer.ema_decay,
                         trial.user_attrs["ema_decay_actual"])
        self.assertFalse(any("gamma_raw" in n for n, _ in wrapper.model.named_parameters()))
        real_adamw = torch.optim.AdamW
        captured = []
        def capture(*args, **kwargs):
            optimizer = real_adamw(*args, **kwargs)
            captured.extend((group["lr"], group["weight_decay"], list(group["params"]))
                            for group in optimizer.param_groups)
            return optimizer
        wrapper.epochs = 2
        (xt, yt), (xv, yv), _ = ds._indv_dataset()
        with patch("libs.supervised.torch.optim.AdamW", side_effect=capture):
            wrapper.fit(xt, yt, xv, yv)
        beta = wrapper.model.dev_beta_raw
        beta_group = next(g for g in captured if any(v is beta for v in g[2]))
        self.assertAlmostEqual(beta_group[0], p["lr"] * 4.2)
        self.assertEqual(beta_group[1], 0)
        self.assertEqual(len(training_diagnostics(wrapper)["beta_epoch_history"]), 2)
        trial.user_attrs["ema_decay_actual"] = .123
        with self.assertRaisesRegex(ValueError, "ema_decay_actual"):
            restore_params(trial, 40)

    def test_default_dynamics_matches_hl1(self):
        torch.set_num_threads(1)
        ds = dataset()
        p = dict(embed_dim=8, embedder_layers=1, dropout=.1, lr=.001,
                 weight_decay=1e-5, n_prototypes=6, batch_size=64)
        p.update({k: FINAL_CONFIG[k] for k in ("correction_geometry", "head_input_scale", "beta_param")})
        states = []
        for extra in ({}, dict(beta_lr_mult=1., ema_timescale="hl_1")):
            torch.manual_seed(19)
            w = build_wrapper(ds, dict(p, **extra), FINAL_CONFIG, "cpu")
            w.epochs = 2
            (xt, yt), (xv, yv), _ = ds._indv_dataset()
            w.fit(xt, yt, xv, yv)
            states.append({k: v.clone() for k, v in w.model.state_dict().items()})
        for k in states[0]:
            torch.testing.assert_close(states[0][k], states[1][k], rtol=0, atol=0, equal_nan=True)

    def test_margin_definitions_and_shift_invariance(self):
        final = torch.tensor([[3., 2., 1.], [0., 3., 1.]])
        region = torch.tensor([[0., 1., 4.], [0., 2., 1.]])
        d = classification_margin_diagnostics(final, region)
        self.assertEqual(d["region_to_final_class_change_rate"], .5)
        self.assertEqual(d["final_margin_mean"], 1.5)
        self.assertEqual(d["region_margin_mean"], 0.)
        self.assertEqual(d["correction_margin_mean"], 1.5)
        self.assertEqual(d, classification_margin_diagnostics(final + 10, region - 5))
        binary = classification_margin_diagnostics(torch.tensor([[2.], [-3.]]),
                                                    torch.tensor([[-1.], [-1.]]))
        self.assertEqual(binary["final_margin_mean"], 2.5)
        self.assertEqual(binary["region_margin_mean"], 0.)
        self.assertEqual(binary["region_to_final_class_change_rate"], .5)

    def test_diagnostic_forward_preserves_logits_and_state(self):
        ds = dataset("multiclass")
        p = restore_params(study_for(ds).best_trial, 40)
        w = build_wrapper(ds, p, FINAL_CONFIG, "cpu")
        w.model.eval()
        xv = ds._indv_dataset()[1][0]
        state = {k: v.clone() for k, v in w.model.state_dict().items()}
        with torch.no_grad():
            expected = w._forward_batched(xv)
            actual = w._forward_batched(xv, collect_diagnostics=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for k, v in w.model.state_dict().items():
            torch.testing.assert_close(v, state[k], rtol=0, atol=0, equal_nan=True)
        self.assertTrue(w.prediction_diagnostics["available"])


if __name__ == "__main__":
    unittest.main()
