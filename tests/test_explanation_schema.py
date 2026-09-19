import unittest
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from libs.diagnostics import group_relative_feature_stats
from libs.explanation_schema import explanation_categories


class ExplanationSchemaTest(unittest.TestCase):
    def test_coded_categories_are_decoded_before_frequency_counting(self):
        columns = ["residence_since", "duration"]
        raw = np.array([[1, 6], [2, 12], [2, 18], [4, 24]], dtype=float)
        transformer = QuantileTransformer(n_quantiles=4).fit(raw)
        stored = torch.tensor(transformer.transform(raw), dtype=torch.float32)
        query = torch.tensor(transformer.transform([[4, 6]]), dtype=torch.float32)
        original = stored.clone()
        model = SimpleNamespace(
            column_names=columns,
            embedder=SimpleNamespace(cat_col_idx=[], num_col_idx=[0, 1]),
            prototype_layer=SimpleNamespace(sample_groups=[[0, 1, 2, 3]]),
            feature_store=SimpleNamespace(_filled=4, _store=stored),
        )
        stats = group_relative_feature_stats(
            model, {"centroid_id": torch.tensor([0])}, query,
            categorical_overrides=explanation_categories(31),
            quantile_transformer=transformer,
        )[0]
        self.assertEqual([d["feature_name"] for d in stats["numeric"]], ["duration"])
        category = stats["categorical"][0]
        self.assertEqual(category["value"], 4)
        self.assertEqual(category["group_freq"], 0.25)
        self.assertEqual(category["group_mode"], 2)
        self.assertEqual(category["group_mode_freq"], 0.5)
        self.assertTrue(torch.equal(stored, original))
        # Region profile: decoded codes, query-independent, and zero enrichment
        # when the region is the whole store.
        region = {d["feature_name"]: d for d in stats["region"]}
        self.assertEqual(set(region), {"residence_since", "duration"})
        self.assertEqual(region["residence_since"]["kind"], "categorical")
        self.assertEqual(region["residence_since"]["region_mode"], 2)
        self.assertEqual(region["residence_since"]["region_mode_freq"], 0.5)
        self.assertEqual(region["residence_since"]["region_mode_count"], 2)
        self.assertEqual(region["residence_since"]["region_n"], 4)
        self.assertEqual(category["group_count"], 1)
        self.assertEqual(category["group_n"], 4)
        self.assertAlmostEqual(region["residence_since"]["score"], 0.0)
        self.assertEqual(region["duration"]["kind"], "numeric")
        self.assertAlmostEqual(region["duration"]["score"], 0.0)
        self.assertLessEqual(stats["numeric"][0]["group_q25"], stats["numeric"][0]["group_median"])
        self.assertLessEqual(stats["numeric"][0]["group_median"], stats["numeric"][0]["group_q75"])
        with self.assertRaisesRegex(ValueError, "fitted numeric transformer"):
            group_relative_feature_stats(
                model, {"centroid_id": torch.tensor([0])}, query,
                categorical_overrides=explanation_categories(31),
            )

    def test_overrides_are_dataset_specific(self):
        self.assertEqual(explanation_categories(10), {})
        self.assertEqual(set(explanation_categories(31)), {
            "residence_since", "installment_commitment", "existing_credits", "num_dependents",
        })


if __name__ == "__main__":
    unittest.main()
