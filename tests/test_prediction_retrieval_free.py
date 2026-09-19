"""The prediction path must not touch the evidence memory bank.

TabERA's README states that retrieval is not an input to the logits. That
was true functionally (changing k leaves z unchanged) but, until the
`retrieve` flag, forward() still ran the k-NN on every call. These tests pin
both halves of the decoupling:

  * computational: TabERAWrapper.predict / predict_proba / _forward_batched
    never call MemoryBank.retrieve;
  * functional: skipping retrieval leaves logits bit-identical
    (torch.equal, not assert_close).
"""
import unittest
from unittest.mock import patch

import torch

from libs.benchmark import build_wrapper
from libs.benchmark_config import FINAL_CONFIG
from tests.test_benchmark import dataset


def _trained_wrapper(task):
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
    return w, xv, xe


class PredictionRetrievalFreeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_skipping_retrieval_leaves_logits_bit_identical(self):
        for task in ("binclass", "multiclass", "regression"):
            with self.subTest(task=task):
                w, xv, xe = _trained_wrapper(task)
                model = w.model
                # Memory holds the training set, so the legacy call really retrieves.
                self.assertGreaterEqual(int(model.memory.filled.item()), model.k)
                with torch.no_grad():
                    legacy = model(xe)
                    gated = model(xe, retrieve=False)
                self.assertIsInstance(legacy["topk_idx"], torch.Tensor)
                self.assertIsInstance(legacy["neighbor_mask"], torch.Tensor)
                self.assertIsNone(gated["topk_idx"])
                self.assertIsNone(gated["neighbor_mask"])
                # Prediction-only also drops the host-syncing eval diagnostics.
                self.assertEqual(gated["dev_diag"], {})
                self.assertIn("dev_changed_rate", legacy["dev_diag"])
                for key in ("logits", "centroid_id", "context_emb", "query_emb", "correction"):
                    self.assertTrue(torch.equal(legacy[key], gated[key]), key)

    def test_wrapper_prediction_path_never_calls_memory_retrieve(self):
        for task in ("binclass", "multiclass"):
            with self.subTest(task=task):
                w, xv, xe = _trained_wrapper(task)
                model = w.model
                with torch.no_grad():
                    reference = model(xe)["logits"]
                    reference_val = model(xv)["logits"]

                def _forbidden(*args, **kwargs):
                    raise AssertionError("prediction path called MemoryBank.retrieve")

                with patch.object(model.memory, "retrieve", side_effect=_forbidden), torch.no_grad():
                    preds = w.predict(xe)
                    logits = w.predict_proba(xe, logit=True)
                    batched = w._forward_batched(xe)
                    val_logits = w._forward_batched(xv, collect_diagnostics=True)
                self.assertTrue(torch.equal(logits, reference))
                self.assertTrue(torch.equal(batched, reference))
                self.assertTrue(torch.equal(val_logits, reference_val))
                self.assertEqual(len(preds), len(xe))

    def test_explanation_and_default_calls_still_retrieve(self):
        w, xv, xe = _trained_wrapper("binclass")
        model = w.model
        real = model.memory.retrieve
        with patch.object(model.memory, "retrieve", wraps=real) as spy, torch.no_grad():
            default = model(xe)
            self.assertEqual(spy.call_count, 1)
            explained = model(xe, return_explanations=True)
            self.assertEqual(spy.call_count, 2)
            forced = model(xe, retrieve=True)
            self.assertEqual(spy.call_count, 3)
        for out in (default, explained, forced):
            self.assertIsInstance(out["topk_idx"], torch.Tensor)
        self.assertIn("explanations", explained)
        self.assertTrue(torch.equal(explained["logits"], default["logits"]))


if __name__ == "__main__":
    unittest.main()
