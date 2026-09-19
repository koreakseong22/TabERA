import unittest
from pathlib import Path

import reproduce
from libs.benchmark import arm_config, final_study_path, result_path
from libs.benchmark_config import FINAL_CONFIG


class TangentReproduceTests(unittest.TestCase):
    def parse(self, *extra):
        return reproduce.parser().parse_args(["--openml_id", "51", *extra])

    def test_unit_tangent_uses_own_study_and_separate_result(self):
        args = self.parse("--correction_geometry", "unit_tangent",
                          "--head_input_scale", "auto")
        config = reproduce.requested_config(args, arm_config)
        self.assertEqual(config["correction_geometry"], "unit_tangent")
        self.assertEqual(config["head_input_scale"], "auto")
        study = str(final_study_path("root", 1, 51, config))
        self.assertIn("..geom=unit_tangent", study)
        self.assertIn("..hs=auto", study)
        out = reproduce.structure_result_path(result_path("root", 1, 51, config=config), config)
        self.assertIn("..geom=unit_tangent..hs=auto", out.name)
        self.assertNotEqual(out, result_path("root", 1, 51))

    def test_default_paths_are_unchanged(self):
        args = self.parse()
        config = reproduce.requested_config(args, arm_config)
        self.assertEqual(config, FINAL_CONFIG)
        self.assertEqual(config["correction_geometry"], "tangent")
        self.assertEqual(config["head_input_scale"], "unit")
        self.assertEqual(config["early_stop_metric"], "val_loss")
        base = result_path("root", 1, 51, config=config)
        self.assertEqual(reproduce.structure_result_path(base, config), base)

    def test_structure_helper_copies_input(self):
        base = dict(FINAL_CONFIG)
        tangent = reproduce.with_structure(base, "tangent", "unit")
        self.assertEqual(base, FINAL_CONFIG)
        self.assertEqual(tangent["correction_geometry"], "tangent")
        self.assertEqual(tangent["head_input_scale"], "unit")

    def test_invalid_geometry_scale_pairs_fail(self):
        for geometry, scale in (("tangent", "auto"), ("unit_tangent", "unit")):
            args = self.parse("--correction_geometry", geometry,
                              "--head_input_scale", scale)
            with self.assertRaisesRegex(ValueError, "Unsupported geometry/scale"):
                reproduce.requested_config(args, arm_config)

    def test_unverified_override_is_limited_to_implementation_hash(self):
        self.assertTrue(reproduce.implementation_only_contract_diff([
            "implementation: recorded='old', expected='new'",
        ]))
        self.assertFalse(reproduce.implementation_only_contract_diff([]))
        self.assertFalse(reproduce.implementation_only_contract_diff([
            "implementation: recorded='old', expected='new'",
            "data: recorded='old', expected='new'",
        ]))
        self.assertFalse(reproduce.implementation_only_contract_diff([
            "config: recorded='old', expected='new'",
        ]))


if __name__ == "__main__":
    unittest.main()
