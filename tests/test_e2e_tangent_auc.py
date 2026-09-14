import copy
import unittest
import torch
from libs.tabera import TabERA
from analyze_e2e_tangent_auc import select_geometry


class TangentGeometryTests(unittest.TestCase):
    def test_geometry_only_changes_correction_magnitude(self):
        torch.manual_seed(9)
        base=TabERA(n_features=5,embed_dim=32,n_prototypes=4,memory_size=20,
                    correction_geometry="unit_tangent",head_input_scale="auto",n_output=1)
        x=torch.randn(20,5);unit=copy.deepcopy(base);tangent=copy.deepcopy(base)
        rng=torch.get_rng_state().clone();select_geometry(unit,"unit_tangent");select_geometry(tangent,"tangent")
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(unit.effective_gamma(),tangent.effective_gamma())
        self.assertEqual(float(unit.effective_beta().detach()),float(tangent.effective_beta().detach()))
        unit.eval();tangent.eval()
        with torch.no_grad():u,t=unit(x),tangent(x)
        torch.testing.assert_close(u["query_emb"],t["query_emb"])
        torch.testing.assert_close(u["context_emb"],t["context_emb"])
        torch.testing.assert_close(u["correction_dir"],t["correction_dir"],atol=2e-6,rtol=2e-6)
        s=u["geo"]["s"]
        torch.testing.assert_close(t["correction_mag"],u["correction_mag"]*s,atol=2e-6,rtol=2e-6)
        for model,out in ((unit,u),(tangent,t)):
            expected=model.dev_head(model.effective_gamma()*(out["context_emb"]+out["correction"]))
            torch.testing.assert_close(out["logits"],expected)


if __name__=="__main__":unittest.main()
