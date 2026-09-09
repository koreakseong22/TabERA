# -*- coding: utf-8 -*-
"""
libs/explain_format.py
======================
Shared value formatting for the explanation views.

The terminal view (``analyze.print_explanation``) and the paper figure
(``libs.explain_figure``) render the *same* observer outputs. Every quantity a
reader sees -- a class name, a numeric value mapped back out of quantile space,
a categorical code resolved to its label, the within-region rank sentence --
therefore has to be produced in exactly one place, or the two views drift and a
figure in the paper stops matching the text the code prints.

⚠ Formatting only. Nothing here reads the model, and ``ExplanationFormatter``
  never decides what is shown: selection, ordering and display budgets stay
  with the caller. The one deliberate exception is ``build_neighbor_card`` --
  a *selection* helper kept in this module, outside the formatter class, so
  the text view and the figure pick the same feature values for a retrieved
  case.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from libs.prototypes import inverse_transform_numeric


class ExplanationFormatter:
    """Value formatting shared by every explanation view.

    Parameters mirror the dataset-derived arguments of ``print_explanation``;
    holding them on one object is what lets the figure be built from the same
    values the terminal printed rather than from a second implementation.
    """

    def __init__(self,
                 col_names: Optional[List[str]] = None,
                 cat_category_names: Optional[dict] = None,
                 quantile_transformer=None,
                 num_cols: Optional[List[int]] = None,
                 target_class_names: Optional[List[str]] = None,
                 tasktype: Optional[str] = None) -> None:
        self.col_names = col_names
        self.cat_category_names = cat_category_names
        self.quantile_transformer = quantile_transformer
        self.num_cols = num_cols
        self.target_class_names = target_class_names
        self.tasktype = tasktype
        self.name_to_idx = ({name: i for i, name in enumerate(col_names)}
                            if col_names else {})

    # ── labels ────────────────────────────────────────────────────
    def label_name(self, v) -> str:
        if v is None:
            return "?"
        if self.tasktype == "regression":
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)
        try:
            code = int(round(float(v)))
        except Exception:
            return str(v)
        if self.target_class_names and 0 <= code < len(self.target_class_names):
            return str(self.target_class_names[code])
        return str(code)

    @staticmethod
    def region_name(raw) -> str:
        raw = str(raw or "Region")
        if raw.startswith("Centroid_"):
            return "Region " + raw.split("Centroid_", 1)[1]
        if raw.startswith("Prototype_"):
            return "Region " + raw.split("Prototype_", 1)[1]
        return raw

    def dist_str(self, counts, total, max_items: int = 3) -> str:
        """Compact but exhaustive-in-mass class distribution.

        For many-class regions only the largest classes are named and the
        remaining probability mass is explicitly collapsed into ``others``.
        """
        counts = {int(k): int(v) for k, v in (counts or {}).items()}
        if not total or not counts:
            return "(no label summary)"
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ranked[:max_items]
        parts = [f"{self.label_name(k)} {c}/{total} ({c/total:.0%})" for k, c in shown]
        rest = sum(c for _, c in ranked[max_items:])
        if rest:
            parts.append(f"others {rest}/{total} ({rest/total:.0%})")
        return " · ".join(parts)

    def dist_items(self, counts, total, max_items: int = 3) -> List[dict]:
        """``dist_str`` as structured rows, for views that draw rather than print.

        Same ranking and same ``others`` collapse, so a stacked bar built from
        this cannot disagree with the printed line.
        """
        counts = {int(k): int(v) for k, v in (counts or {}).items()}
        if not total or not counts:
            return []
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        rows = [{"code": k, "name": self.label_name(k), "count": c,
                 "total": int(total), "prop": c / total}
                for k, c in ranked[:max_items]]
        rest = sum(c for _, c in ranked[max_items:])
        if rest:
            rows.append({"code": None, "name": "others", "count": rest,
                         "total": int(total), "prop": rest / total})
        return rows

    # ── numbers ───────────────────────────────────────────────────
    @staticmethod
    def pretty_num(x) -> Optional[str]:
        if x is None:
            return None
        x = float(x)
        ax = abs(x)
        if ax >= 1000:
            return f"{x:,.0f}"
        if ax >= 10:
            return (f"{x:,.1f}" if abs(x - round(x)) > 0.01 * max(ax, 1.0)
                    else f"{x:,.0f}")
        return f"{x:.2f}".rstrip("0").rstrip(".")

    def real_numeric(self, feature_idx: int, v: float):
        """Map a numeric value out of quantile space back to original units.

        ⚠ Applied to a group mean this is T⁻¹(mean(T(x))), which is not the
          arithmetic mean of the original column. Views call it a *reference*
          value for that reason.
        """
        if self.quantile_transformer is None or self.num_cols is None:
            return None
        return inverse_transform_numeric(
            self.quantile_transformer, self.num_cols, feature_idx, v)

    def fmt_num_value(self, name: str, uniform_val: float) -> str:
        if name in self.name_to_idx:
            real = self.real_numeric(self.name_to_idx[name], uniform_val)
            if real is not None:
                return self.pretty_num(real)
        return f"{float(uniform_val):.3f}"

    def fmt_cat_value(self, name: str, code_val: float) -> str:
        names_for_col = (self.cat_category_names.get(name)
                         if self.cat_category_names else None)
        code = int(round(float(code_val)))
        try:
            if names_for_col is not None:
                nm = (names_for_col[code] if not isinstance(names_for_col, dict)
                      else names_for_col.get(code))
                if nm is not None:
                    return str(nm)
        except (IndexError, KeyError, TypeError):
            pass
        return f"Category {code}"

    # ── within-region rank ────────────────────────────────────────
    @staticmethod
    def rank_position(d: Dict) -> Optional[str]:
        """Numeric feature rank inside the region, stated without a midrank.

        ``group_pct`` is a midrank (below + equal/2) and discrete features tie
        often, so "top 7%" derived from it can read as a strict ordering that
        does not hold. ``group_pct_below`` / ``group_pct_equal`` are exact
        shares of region training cases, so the sentence built from them is
        literally true: higher than X%, equal to Y%.
        """
        below = d.get("group_pct_below")
        equal = d.get("group_pct_equal")
        if below is None or equal is None:
            return None
        below, equal = float(below), float(equal)
        above = max(0.0, 1.0 - below - equal)
        if below >= above:
            s = f"higher than {below:.0%} of region cases"
        else:
            s = f"lower than {above:.0%} of region cases"
        if equal > 0.0005:
            s += f" · equal to {equal:.0%}"
        return s

    @staticmethod
    def centre_distance_position(rp: Dict) -> Optional[str]:
        """Rank of the sample's distance to the region centre.

        Stated as what ``within_region_position()`` returns -- a cosine
        distance and its rank among the region's training cases. It is not a
        typicality score, and the explained sample is not itself a member, so
        the comparison set is named.
        """
        if rp is None or rp.get("group_pct") is None:
            return None
        farther = float(rp["group_pct"])   # share of the region closer to the centre
        closer = 1.0 - farther
        if farther >= 0.995:
            return "Farther than every region training case"
        if closer >= 0.995:
            return "Closer than every region training case"
        if farther >= 0.5:
            return f"Farther than {farther:.0%} of region training cases"
        return f"Closer than {closer:.0%} of region training cases"

    # ── query-vs-neighbour gaps ───────────────────────────────────
    def gap_summary(self, nb: Dict, n: int = 2) -> List[str]:
        """The n largest query-to-neighbour differences, largest gap first.

        ``delta`` exists so the query value can be recovered as
        ``neighbour - delta``; for categoricals it is not a magnitude.
        """
        gp = nb.get("gaps") or []
        diff = [g for g in sorted(gp, key=lambda g: g["gap"], reverse=True)
                if g.get("gap", 0.0) > 1e-9][:n]
        parts = []
        for g in diff:
            name, nval, kind, delta = g["name"], g["neighbor_value"], g["kind"], g["delta"]
            qval = nval - delta
            if kind == "categorical":
                parts.append(f"{name}: {self.fmt_cat_value(name, qval)} vs "
                             f"{self.fmt_cat_value(name, nval)}")
            else:
                parts.append(f"{name}: {self.fmt_num_value(name, qval)} vs "
                             f"{self.fmt_num_value(name, nval)}")
        return parts


# ─────────────────────────────────────────────────────────────
# Neighbour card: which feature values a view shows beside a retrieved case
# ─────────────────────────────────────────────────────────────

def build_neighbor_card(nb: Dict, fmt: ExplanationFormatter, *,
                        max_matches: int = 2, max_differs: int = 1,
                        cat_rarity: Optional[Dict[str, float]] = None) -> Dict:
    """Pick the feature values to show beside one retrieved case.

    Selection, not formatting -- kept out of ExplanationFormatter on purpose,
    and in one function so the text view and the figure show the same card.

    ⚠ What this can and cannot say. The card is a *description* of feature
      values the query and the case hold: similarity itself is the cosine of
      their embeddings, and nothing here explains why that cosine is high.
      Views label the rows "matches" / "closest values" / "differs", never
      "why similar".

    ⚠ Numeric gaps are measured where the model saw the values: the [0,1]
      quantile space held in FeatureStore (feature_gaps is Gower on that
      space). Only the *displayed* numbers are mapped back to original units,
      so a small gap means close in rank, not close in raw units.

    Three rows, each optional, none padded to its budget:
      differs   reserved FIRST: the largest non-zero gaps, up to max_differs.
                Reserving it first means a two-feature dataset still shows a
                difference instead of both features being claimed as close.
      matches   exact equality (gap == 0), up to max_matches. Exact
                categorical matches all tie, so among them the value that is
                rarer in the region comes first (``cat_rarity`` = region
                frequency of the query's value, from
                group_relative_feature_stats): a match on a value 2% of the
                region holds says more than a match on the mode. Other ties
                keep input order.
      closest   only when there is no exact match at all: the smallest
                non-zero gaps, up to max_matches. "Closest" is rank language
                on purpose -- with no threshold, nothing here certifies that a
                gap of 0.6 is "similar", only that it is the smallest.
    A feature appears at most once per card, and a row that would be empty is
    simply absent: a short card is better than a card padded to look full.

    Returns {"matches": [...], "closest": [...], "differs": [...]}, formatted.
    """
    gaps = [g for g in (nb.get("gaps") or []) if g.get("gap") is not None]
    if not gaps:
        return {"matches": [], "closest": [], "differs": []}
    order = {g["name"]: i for i, g in enumerate(gaps)}
    rarity = cat_rarity or {}
    EPS = 1e-9

    def _vals(g):
        name, nval, delta = g["name"], g["neighbor_value"], g["delta"]
        qval = nval - delta
        if g["kind"] == "categorical":
            return name, fmt.fmt_cat_value(name, qval), fmt.fmt_cat_value(name, nval)
        return name, fmt.fmt_num_value(name, qval), fmt.fmt_num_value(name, nval)

    used = set()
    differs = []
    for g in sorted(gaps, key=lambda g: (-float(g["gap"]), order[g["name"]])):
        if len(differs) >= max_differs or float(g["gap"]) <= EPS:
            break
        name, q, n = _vals(g)
        differs.append(f"{name}: {q} ↔ {n}")
        used.add(name)

    exact = [g for g in gaps if float(g["gap"]) <= EPS and g["name"] not in used]
    exact.sort(key=lambda g: (
        rarity.get(g["name"], 1.0) if g["kind"] == "categorical" else 1.0,
        order[g["name"]]))
    matches = []
    for g in exact[:max_matches]:
        name, q, _ = _vals(g)
        matches.append(f"{name} = {q}")
        used.add(name)

    closest = []
    if not matches:
        near = [g for g in gaps if float(g["gap"]) > EPS and g["name"] not in used]
        near.sort(key=lambda g: (float(g["gap"]), order[g["name"]]))
        for g in near[:max_matches]:
            name, q, n = _vals(g)
            closest.append(f"{name}: {q} ↔ {n}")
            used.add(name)
    return {"matches": matches, "closest": closest, "differs": differs}
