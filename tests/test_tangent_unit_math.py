import unittest

import torch
import torch.nn.functional as F

from libs.tabera import TabERA


def reference_correction(q, c, beta, geometry, eps=1e-6):
    """Independent statement of the documented Tangent geometry."""
    p = F.normalize(q - c, dim=-1)
    v = p - (p * c).sum(dim=-1, keepdim=True) * c
    s = v.norm(dim=-1, keepdim=True)
    if geometry == "tangent":
        return beta * v
    if geometry == "unit_tangent":
        return beta * v / s.clamp_min(eps)
    raise ValueError(geometry)


class TangentUnitMathTests(unittest.TestCase):
    def test_tangent_unit_full_forward_matches_equations(self):
        torch.manual_seed(101)
        model = TabERA(
            n_features=6,
            embed_dim=32,
            n_prototypes=5,
            memory_size=24,
            n_output=3,
            tasktype="multiclass",
            n_classes=3,
            correction_geometry="tangent",
            head_input_scale="unit",
            beta_param="sigmoid",
        ).eval()
        x = torch.randn(17, 6)

        with torch.no_grad():
            out = model(x)
            q, c = out["query_emb"], out["context_emb"]
            beta = torch.sigmoid(model.dev_beta_raw)
            d = reference_correction(q, c, beta, "tangent")
            h = c + d
            logits = F.linear(h, model.dev_head.weight, model.dev_head.bias)

        self.assertEqual(model.effective_gamma(), 1.0)
        torch.testing.assert_close(c.norm(dim=-1), torch.ones(len(x)),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(out["correction"], d, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(out["correction_mag"], d.norm(dim=-1),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(out["logits"], logits, atol=2e-6, rtol=2e-6)
        self.assertLess(float((d * c).sum(dim=-1).abs().max()), 2e-6)

    def test_tangent_and_unit_tangent_relation_including_epsilon(self):
        torch.manual_seed(102)
        model = TabERA(
            n_features=1,
            embed_dim=16,
            n_prototypes=3,
            memory_size=8,
            n_output=1,
            correction_geometry="tangent",
            head_input_scale="unit",
        )
        q = torch.randn(20, 16)
        c = F.normalize(torch.randn(20, 16), dim=-1)
        beta = torch.tensor([0.37])

        model.correction_geometry = "tangent"
        tangent, _, tangent_mag, tangent_log = model._compute_correction(q, c, beta)
        model.correction_geometry = "unit_tangent"
        unit, _, unit_mag, unit_log = model._compute_correction(q, c, beta)
        s = tangent_log["s"]

        expected_tangent = reference_correction(q, c, beta, "tangent")
        expected_unit = reference_correction(q, c, beta, "unit_tangent")
        torch.testing.assert_close(tangent, expected_tangent)
        torch.testing.assert_close(unit, expected_unit)
        torch.testing.assert_close(tangent_mag, beta * s, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(unit_mag, beta * s / s.clamp_min(1e-6),
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(tangent, s.unsqueeze(-1) * unit,
                                   atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(tangent_log["s"], unit_log["s"])

        # At the singular q == c point PyTorch normalize returns zero. The
        # implemented continuous safeguard therefore gives d == 0 for both.
        q0 = c[:3].clone().requires_grad_(True)
        for geometry in ("tangent", "unit_tangent"):
            model.correction_geometry = geometry
            d0, _, m0, _ = model._compute_correction(q0, c[:3], beta)
            self.assertTrue(torch.equal(d0, torch.zeros_like(d0)))
            self.assertTrue(torch.equal(m0, torch.zeros_like(m0)))
            self.assertTrue(torch.isfinite(torch.autograd.grad(d0.sum(), q0,
                                                               retain_graph=True)[0]).all())

    def test_tangent_gradient_matches_independent_formula(self):
        torch.manual_seed(103)
        model = TabERA(
            n_features=1,
            embed_dim=8,
            n_prototypes=2,
            memory_size=4,
            n_output=1,
            correction_geometry="tangent",
            head_input_scale="unit",
        ).double()
        q = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
        c = F.normalize(torch.randn(7, 8, dtype=torch.float64), dim=-1).detach().requires_grad_(True)
        beta = torch.tensor([0.43], dtype=torch.float64, requires_grad=True)
        weight = torch.randn(7, 8, dtype=torch.float64)

        actual = model._compute_correction(q, c, beta)[0]
        actual_grads = torch.autograd.grad((actual * weight).sum(), (q, c, beta))

        qr = q.detach().clone().requires_grad_(True)
        cr = c.detach().clone().requires_grad_(True)
        br = beta.detach().clone().requires_grad_(True)
        expected = reference_correction(qr, cr, br, "tangent")
        expected_grads = torch.autograd.grad((expected * weight).sum(), (qr, cr, br))

        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad,
                                       atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
