import contextlib
import io
import unittest

from analyze import print_explanation
from libs.explain_format import select_explanation_features


def _num(name, idx, value, *, below, equal=0.0, median=0.4, q25=0.3, q75=0.5):
    return {
        "feature_idx": idx, "feature_name": name, "kind": "numeric",
        "value": value, "group_mean": median, "group_median": median,
        "group_q25": q25, "group_q75": q75, "group_std": 0.1, "z": 0.0,
        "group_pct": below + equal / 2.0,
        "group_pct_below": below, "group_pct_equal": equal,
    }


def _cat(name, idx, value, *, freq, mode, mode_freq, n=20):
    return {
        "feature_idx": idx, "feature_name": name, "kind": "categorical",
        "value": value, "group_freq": freq, "group_mode": mode,
        "group_count": int(round(freq * n)), "group_n": n,
        "group_mode_freq": mode_freq, "rarity": 1.0 - freq,
        "differs_from_mode": value != mode, "absent_from_group": freq == 0.0,
        "ties_mode": False,
    }


def _region_num(name, idx, score, median=0.4):
    return {"feature_idx": idx, "feature_name": name, "kind": "numeric",
            "region_median": median, "region_q25": 0.3, "region_q75": 0.5,
            "global_median": 0.5, "score": score}


def _region_cat(name, idx, score, mode=1, mode_freq=0.6, n=20):
    return {"feature_idx": idx, "feature_name": name, "kind": "categorical",
            "region_mode": mode, "region_mode_freq": mode_freq,
            "region_mode_count": int(round(mode_freq * n)), "region_n": n,
            "global_mode_freq": mode_freq - score, "score": score}


def _default_group_stats(group_size=20):
    return {
        "numeric": [_num("amount", 1, 0.75, below=0.9, equal=0.1)],
        "categorical": [_cat("kind", 0, 1, freq=0.6, mode=1, mode_freq=0.6)],
        "region": [_region_cat("kind", 0, 0.3), _region_num("amount", 1, 0.2)],
        "group_size": group_size,
    }


def render_fixture(*, verbose=False, group_stats="default", proto_pred=1,
                   group_size=20, sink=None):
    """Render one fixed explanation, printed or collected through ``sink``.

    Module level, and shared with tests/test_explanation_png.py, so the image
    tests exercise the very output the text tests assert on rather than a
    second fixture that could drift from it.
    """
    neighbors = []
    for rank, (sample_id, label, similarity) in enumerate([
        (10, 1, 0.99),
        (11, 1, 0.98),
        (12, 0, 0.97),
        (13, 1, 0.96),
        (14, 0, 0.95),
        (15, 1, 0.94),
        (16, 1, 0.93),
        (17, 0, 0.92),
        (18, 1, 0.91),
    ]):
        neighbors.append({
            "rank": rank,
            "memory_idx": rank,
            "sample_id": sample_id,
            "similarity": similarity,
            "label": label,
            "features": {"kind": rank % 2, "amount": 0.1 * (rank + 1)},
        })
    if group_stats == "default":
        group_stats = _default_group_stats(group_size)

    explanation = {
        "input_features": {"kind": 1, "amount": 0.75},
        "prototype": {"assigned_group": "Centroid_2", "routing_confidence": 0.82},
        "local_evidence": {
            "group_size": group_size,
            "group_label_counts": {0: 3, 1: 17},
            "n_neighbors": 9,
            "label_counts": {0: 3, 1: 6},
        },
        "neighbors": neighbors,
        "group_stats": group_stats,
        "prototype_deviation": {
            "correction_geometry": "tangent",
            "head_input_scale": "unit",
            "beta": 0.4,
            "gamma": 1.0,
            "dev_norm": 0.2,
            "logit_proto": 1.5,
            "logit_dev": 0.3,
            "prob_proto": 0.82,
            "prob_proto_pred": 0.82,
            "prob_final": 0.90,
            "proto_pred": proto_pred,
            "argmax_changed": proto_pred != 1,
        },
        "region_position": None,
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print_explanation(
            [explanation], 0, ["kind", "amount"],
            cat_category_names={"kind": ["A", "B"]},
            num_cols=[1],
            pred_info={"pred_label": "good", "pred_confidence": 0.9,
                       "pred_code": 1},
            target_class_names=["bad", "good"],
            tasktype="binclass",
            verbose=verbose,
            sink=sink,
        )
    return output.getvalue()


class ExplanationOutputTest(unittest.TestCase):
    def _render(self, **kw):
        return render_fixture(**kw)


    def test_default_view_shows_three_blocks_without_researcher_detail(self):
        output = self._render()
        self.assertIn("good — 90.0%", output)
        self.assertIn("Input summary", output)
        self.assertIn("kind = B, amount = 0.750", output)
        self.assertIn("① Your assigned group", output)
        self.assertIn("20 training cases; good 17 (85%), bad 3 (15%)", output)
        # ① group profile is compared to the case only after selection
        self.assertIn("Group profile", output)
        self.assertRegex(output, r"kind\s+B \(60%\)\s+✓ same")
        # numeric profile rows carry the case's value and no ✓/≠
        self.assertRegex(output, r"amount\s+0\.400 \(median\)\s+0\.750")
        self.assertNotIn("≠", output)
        # ② only the row that departs from the group; the mode-matching value is not forced in
        self.assertIn("② How does this case compare within the group?", output)
        self.assertRegex(output, r"amount\s+0\.750\s+0\.400 \(median\)\s+higher than 90%")
        self.assertNotRegex(output, r"kind\s+B\s+B \(60%\)")
        # ③ three rows on the shared columns, then counts without shares
        self.assertIn("③ Retrieved past cases from this group", output)
        for i in (1, 2, 3):
            self.assertIn(f"Case {i}", output)
        self.assertNotIn("Case 4", output)
        self.assertIn("3 of 9 shown; 6 good, 3 bad", output)
        self.assertNotIn("67%", output)
        # no decision line when the class did not change
        self.assertNotIn("Group-based prediction", output)
        # the explained sample is identified, but nothing else internal
        self.assertIn("TabERA Explanation  # 0", output)
        self.assertNotIn("Region", output)
        self.assertNotIn("region", output)
        self.assertNotIn("#10", output)
        self.assertNotIn("similar", output.lower())
        self.assertNotIn("Prediction path", output)
        self.assertNotIn("Logit decomposition", output)
        self.assertNotIn("middle 50%", output)

    def test_rare_categorical_is_counted_not_rounded_to_a_percentage(self):
        gs = _default_group_stats(group_size=83)
        gs["numeric"] = [_num("amount", 1, 0.45, below=0.5, equal=0.1)]
        gs["categorical"] = [_cat("kind", 0, 0, freq=1 / 83, mode=1,
                                  mode_freq=48 / 83, n=83)]
        gs["region"] = [_region_cat("kind", 0, 0.3, mode_freq=48 / 83, n=83)]
        output = self._render(group_stats=gs, group_size=83)
        # 1/83 is "1%", and so is 0/83 rounded up: the count says which
        self.assertIn("seen in 1 of 83", output)
        self.assertNotIn("seen in 1%", output)
        # a mode of 48 cases is still a percentage
        self.assertIn("B (58%)", output)

    def test_absent_categorical_says_so(self):
        gs = _default_group_stats()
        gs["numeric"] = [_num("amount", 1, 0.45, below=0.5, equal=0.1)]
        gs["categorical"] = [_cat("kind", 0, 0, freq=0.0, mode=1, mode_freq=0.6)]
        self.assertIn("not seen in group", self._render(group_stats=gs))

    def test_typical_case_is_stated_not_padded(self):
        gs = _default_group_stats()
        gs["numeric"] = [_num("amount", 1, 0.45, below=0.5, equal=0.1)]
        output = self._render(group_stats=gs)
        self.assertIn("This case is typical of its group.", output)
        self.assertNotIn("typical in group    position", output)

    def test_small_group_shows_no_profile_and_no_positions(self):
        output = self._render(group_size=7)
        self.assertIn("7 training cases; good 17", output)
        self.assertIn("Group profile unavailable — only 7 training cases in this group.", output)
        self.assertIn("Too few training cases in this group (7) to place this case.", output)
        self.assertNotIn("(median)", output)
        # ③ still shows the cases, on the fallback columns
        self.assertRegex(output, r"kind\s+amount\s+outcome")
        self.assertIn("Case 3", output)

    def test_decision_line_appears_only_when_the_class_changed(self):
        output = self._render(proto_pred=0)
        self.assertIn("Group-based prediction: bad → Final prediction: good", output)
        self.assertIn("The case-specific adjustment changed the predicted class.", output)
        self.assertNotIn("Logit decomposition", output)

    def test_verbose_adds_ids_full_input_all_cases_and_decomposition(self):
        output = self._render(verbose=True)
        self.assertIn("TabERA Explanation  # 0 · Region 2", output)
        self.assertIn("Input case — 2 features", output)
        self.assertIn("Case 9", output)
        self.assertIn("#10", output)
        self.assertIn("similarity", output)
        self.assertIn("Prediction path", output)
        self.assertIn("↓ cosine routing", output)
        self.assertIn("good — 82.0%", output)
        self.assertIn("↓ sample-specific correction", output)
        self.assertIn("Group-based prediction: good → Final prediction: good (unchanged)", output)
        self.assertIn("Logit decomposition", output)
        self.assertIn("region       +1.5000", output)
        self.assertIn("correction   +0.3000", output)
        self.assertIn("final        +1.8000", output)
        self.assertIn("middle 50%", output)
        self.assertIn("0.300–0.500", output)
        self.assertIn("learned representation", output)


class SelectExplanationFeaturesTest(unittest.TestCase):
    def test_thresholds_are_applied_as_stated(self):
        gc = {
            "numeric": [_num("n_in", 0, 0.9, below=0.85),          # q = 0.85 → 2|q−½| = 0.70
                        _num("n_out", 1, 0.9, below=0.84),         # 0.68
                        _num("n_tie", 2, 0.9, below=0.80, equal=0.10)],  # midrank 0.85 → 0.70
            "categorical": [_cat("c_in", 3, 1, freq=0.09, mode=0, mode_freq=0.5),
                            _cat("c_out", 4, 1, freq=0.10, mode=0, mode_freq=0.5),
                            _cat("c_absent", 5, 1, freq=0.0, mode=0, mode_freq=0.5)],
            "region": [], "group_size": 50,
        }
        sel = select_explanation_features(gc, n_position=10)
        names = [d["feature_name"] for d in sel["position"]]
        # ranked by atypicality, ties by feature index; n_out and c_out fall below the rule
        self.assertEqual(names, ["c_absent", "c_in", "n_in", "n_tie"])

    def test_features_may_repeat_across_blocks_and_display_is_capped(self):
        region = [_region_num("a", 0, 0.9), _region_num("b", 1, 0.8), _region_num("c", 2, 0.7)]
        many = {
            "numeric": [_num("a", 0, 0.9, below=0.99), _num("d", 3, 0.9, below=0.90),
                        _num("e", 4, 0.9, below=0.88), _num("f", 5, 0.9, below=0.86)],
            "categorical": [], "region": region, "group_size": 50,
        }
        sel = select_explanation_features(many)
        # a is in ① and is the most atypical value: it stays in ② rather than being dropped
        self.assertEqual([d["feature_name"] for d in sel["position"]], ["a", "d", "e"])
        self.assertEqual(sel["display"], ["a", "b", "d", "c"])
        self.assertFalse(sel["too_small"])

    def test_display_set_never_introduces_a_feature_of_its_own(self):
        region = [_region_num("a", 0, 0.9), _region_num("b", 1, 0.8), _region_num("c", 2, 0.7),
                  _region_num("z", 9, 0.1)]
        sel = select_explanation_features({"numeric": [], "categorical": [],
                                           "region": region, "group_size": 50})
        self.assertEqual(sel["position"], [])
        self.assertEqual(sel["display"], ["a", "b", "c"])
        self.assertEqual(select_explanation_features(None)["display"], [])

    def test_small_group_is_gated(self):
        gc = _default_group_stats(group_size=9)
        sel = select_explanation_features(gc)
        self.assertTrue(sel["too_small"])
        self.assertEqual((sel["region"], sel["position"], sel["display"]), ([], [], []))
        self.assertFalse(select_explanation_features(_default_group_stats(10))["too_small"])


if __name__ == "__main__":
    unittest.main()
