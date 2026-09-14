import copy
import unittest
import torch
from libs.tabera import TabERA
from analyze_e2e_split_head_auc import attach_head, correction_weight


class EndToEndHeadTests(unittest.TestCase):
    def test_initial_predictor_and_rng_parity(self):
        torch.manual_seed(17)
        base=TabERA(n_features=5,embed_dim=32,n_prototypes=4,memory_size=20,
                    correction_geometry="unit_tangent",head_input_scale="auto",n_output=1)
        x=torch.randn(20,5)
        base.eval()
        with torch.no_grad(): expected=base(x)["logits"]
        for mode in ("shared","match","free"):
            model=copy.deepcopy(base)
            state=torch.get_rng_state().clone()
            attach_head(model,mode)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            with torch.no_grad(): got=model(x)["logits"]
            self.assertLess(float((expected-got).abs().max()),1e-6)

    def test_matched_norms_and_trainable_correction(self):
        torch.manual_seed(5)
        model=TabERA(n_features=4,embed_dim=16,n_prototypes=3,memory_size=10,
                     correction_geometry="unit_tangent",head_input_scale="auto",n_output=1)
        attach_head(model,"match")
        self.assertIsNone(model.e2e_correction_head.bias)
        with torch.no_grad(): model.e2e_correction_head.weight.mul_(7.)
        torch.testing.assert_close(correction_weight(model).norm(),model.dev_head.weight.norm())
        loss=model(torch.randn(10,4))["logits"].square().mean()
        loss.backward()
        for w in (model.dev_head.weight,model.e2e_correction_head.weight,model.dev_beta_raw):
            self.assertIsNotNone(w.grad)
            self.assertTrue(torch.isfinite(w.grad).all())
        self.assertGreater(float(model.e2e_correction_head.weight.grad.norm()),0)


if __name__=="__main__": unittest.main()
