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
  with the caller. The deliberate exceptions are the *selection* helpers kept
  in this module, outside the formatter class, so the text view and the
  figure pick the same rows: ``select_explanation_features`` (which features
  blocks ①/②/③ show) and ``build_neighbor_card`` (which values sit beside a
  retrieved case).
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

    def dist_str(self, counts, total, max_items: int = 3,
                 show_total: bool = True, sep: str = " · ") -> str:
        """Compact but exhaustive-in-mass class distribution.

        For many-class regions only the largest classes are named and the
        remaining probability mass is explicitly collapsed into ``others``.
        ``show_total=False`` drops the ``/total`` when the total was already
        stated on the same line (``260 training cases; good 250 (96%)``), and
        ``sep`` lets a caller match the separator of that line.
        """
        counts = {int(k): int(v) for k, v in (counts or {}).items()}
        if not total or not counts:
            return "(no label summary)"
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ranked[:max_items]
        tot = f"/{total}" if show_total else ""
        parts = [f"{self.label_name(k)} {c}{tot} ({c/total:.0%})" for k, c in shown]
        rest = sum(c for _, c in ranked[max_items:])
        if rest:
            parts.append(f"others {rest}{tot} ({rest/total:.0%})")
        return sep.join(parts)

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
        """Plain-language numeric rank among the group's training cases."""
        below = d.get("group_pct_below")
        equal = d.get("group_pct_equal")
        if below is None or equal is None:
            return None
        below, equal = float(below), float(equal)
        above = max(0.0, 1.0 - below - equal)
        if below >= above:
            s = f"higher than {below:.0%} of cases in this group"
        else:
            s = f"lower than {above:.0%} of cases in this group"
        return s

    @staticmethod
    def rank_position_short(d: Dict) -> Optional[str]:
        """``rank_position`` for a table cell: ``lower than 87%``."""
        s = ExplanationFormatter.rank_position(d)
        return s.replace(" of cases in this group", "") if s else None

    @staticmethod
    def cat_share(count, total, freq) -> str:
        """A categorical share, as a count while a percentage would overstate it.

        In a group of 83 a single case is "1%", and the next value up is "2%":
        the percentage moves in 1.2-point steps and reads more precise than the
        data is. Up to ``CAT_SHARE_AS_COUNT`` cases the count is printed
        instead, which is also what a reader of a small group wants to know.
        """
        if count is not None and total and int(count) <= CAT_SHARE_AS_COUNT:
            return f"{int(count)} of {int(total)}"
        return f"{float(freq):.0%}"

    def cat_position_short(self, d: Dict) -> str:
        """Group share of this case's categorical value, for a table cell."""
        if d.get("absent_from_group"):
            return "not seen in group"
        return "seen in " + self.cat_share(d.get("group_count"), d.get("group_n"),
                                           d.get("group_freq", 0.0))

    # ── region profile rows (block ①) ─────────────────────────────
    def typical_value(self, row: Dict) -> str:
        """The group's typical value for one profile row of
        ``group_stats["region"]`` -- mode with its share, or the median."""
        name = row["feature_name"]
        if row["kind"] == "categorical":
            share = self.cat_share(row.get("region_mode_count"), row.get("region_n"),
                                   row.get("region_mode_freq", 0.0))
            return f"{self.fmt_cat_value(name, row['region_mode'])} ({share})"
        return f"{self.fmt_num_value(name, row['region_median'])} (median)"

    def typical_range(self, d: Dict) -> Optional[str]:
        """Middle 50% of the region for a numeric sample row, original units."""
        lo, hi = d.get("group_q25"), d.get("group_q75")
        if lo is None or hi is None:
            return None
        name = d["feature_name"]
        return f"{self.fmt_num_value(name, lo)}–{self.fmt_num_value(name, hi)}"

    def region_row_comparison(self, row: Dict, sample: Optional[Dict]):
        """How this case relates to one typical characteristic of its region.

        Returns ``(case_value, agrees)``. The profile row was chosen without
        the sample (diagnostics._region_profile), so this is the only place
        the two meet -- after selection, never during it.

        ``agrees`` is a *categorical* judgement only: the case either holds
        the group's modal value or it does not, and a reader can check that
        from the two cells.

        ⚠ For a numeric row ``agrees`` is None, and the caller prints the
          value with no mark. A median is not something a value can equal, so
          any ✓ there would stand for an unstated band ("48 against a median
          of 42 -- within what?"), and a reader could not tell which band was
          meant. The band that does exist, the group's middle 50%, is shown
          under --explain_verbose as its own column rather than compressed
          into a tick. ``agrees`` is None as well when the sample row is
          missing (a feature constant within the group has no statistics).
        """
        name = row["feature_name"]
        if sample is None:
            return None, None
        if row["kind"] == "categorical":
            same = int(round(float(sample["value"]))) == int(row["region_mode"])
            return self.fmt_cat_value(name, sample["value"]), same
        return self.fmt_num_value(name, sample["value"]), None

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
            return "Farther than every training case in this group"
        if closer >= 0.995:
            return "Closer than every training case in this group"
        if farther >= 0.5:
            return f"Farther than {farther:.0%} of training cases in this group"
        return f"Closer than {closer:.0%} of training cases in this group"

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


# ─────────────────────────────────────────────────────────────
# Display rules for the default explanation view
# ─────────────────────────────────────────────────────────────
# ⚠ These are UI display rules, not paper claims and not detectors: they
#   decide what a reader sees first, never whether a value is "abnormal".
#   Stated once so the text view and any figure select the same rows.
NUM_ATYPICAL = 0.70        # numeric: 2·|q − ½| with q the within-group midrank percentile;
                           # 0.70 is "below the 15th or above the 85th percentile"
CAT_RARE_FREQ = 0.10       # categorical: the case's value held by <10% of the group
CAT_SHARE_AS_COUNT = 3     # at most this many cases, print "2 of 83" rather than "2%"
MIN_GROUP_SIZE = 10        # below this a median or mode is luck, so no profile and no
                           # positions are shown (same n<10 cut as the paper's Figure 2)
N_REGION_ROWS = 3          # block ①: group profile
N_POSITION_ROWS = 3        # block ②: where this case departs from the group
N_DISPLAY_FEATURES = 4     # shared feature set for Input summary and block ③


def atypicality(d: Dict) -> float:
    """How far from the middle of its group a value sits, on a 0-1 scale.

    numeric      2·|q − ½| with q the within-group midrank percentile
                 (``group_pct``); 0 at the median, 1 beyond every member
    categorical  1 − p_group(value); 1 when no member holds the value
    """
    if d.get("kind") == "categorical":
        return 1.0 - float(d.get("group_freq", 0.0))
    return 2.0 * abs(float(d.get("group_pct", 0.5)) - 0.5)


def select_explanation_features(gc: Optional[Dict], *,
                                n_region: int = N_REGION_ROWS,
                                n_position: int = N_POSITION_ROWS,
                                n_display: int = N_DISPLAY_FEATURES,
                                num_atypical: float = NUM_ATYPICAL,
                                cat_rare_freq: float = CAT_RARE_FREQ,
                                min_group_size: int = MIN_GROUP_SIZE) -> Dict:
    """Decide which features the three blocks of the explanation show.

    Selection, not formatting -- kept beside ``build_neighbor_card`` so every
    view applies one rule. ``gc`` is one entry of
    ``diagnostics.group_relative_feature_stats``.

    ① region    the first ``n_region`` rows of ``gc["region"]``, i.e. the
                features whose distribution in the group differs most from
                the whole training set (KS / TV distance, one 0-1 scale for
                both kinds), ranked without looking at this case. Comparing
                the case to them happens afterwards
                (``region_row_comparison``), so the rows cannot be the ones
                that happen to agree with it.
    ② position  where this case departs from its group: numeric values below
                the 15th or above the 85th within-group percentile
                (``atypicality >= num_atypical``), categorical values held by
                fewer than ``cat_rare_freq`` of the group (an absent value
                counts). Ranked by ``atypicality``. A feature may appear in
                both ① and ②: the two answer different questions (what the
                group is like; where this case sits in it), and meeting both
                criteria is itself information. When nothing qualifies the
                list is empty and the view says the case is typical -- it is
                never padded.
    gate        with fewer than ``min_group_size`` training cases a median,
                a mode or a percentile is luck, so ① and ② are both empty
                and ``too_small`` is set; the view says so instead of
                presenting the numbers as typical.
    display     the feature set reused by Input summary and block ③, at most
                ``n_display``: the first two of ① and of ②, then the remaining
                shown rows of each in turn. Every column of ③ therefore
                appears in ① or ②, and no block introduces a feature of its
                own.

    Returns {"region": [...], "position": [...], "display": [names],
             "by_name": {feature_name: sample row}, "too_small": bool}.
    """
    gc = gc or {}
    nums = list(gc.get("numeric") or [])
    cats = list(gc.get("categorical") or [])
    by_name = {d["feature_name"]: d for d in nums + cats}
    n_g = gc.get("group_size")
    too_small = n_g is not None and int(n_g) < int(min_group_size)
    if too_small:
        return {"region": [], "position": [], "display": [],
                "by_name": by_name, "too_small": True}

    region = list(gc.get("region") or [])[:max(0, int(n_region))]

    cands = []
    for d in nums:
        if atypicality(d) >= num_atypical:
            cands.append(d)
    for d in cats:
        if float(d.get("group_freq", 0.0)) < cat_rare_freq:
            cands.append(d)
    # Rounded so that 2·|0.85 − ½| and 2·|(0.80 + 0.10/2) − ½| tie by feature
    # index instead of by floating-point noise.
    cands.sort(key=lambda d: (-round(atypicality(d), 9), d["feature_idx"]))
    position = cands[:max(0, int(n_position))]

    display: List[str] = []

    def _add(name):
        if name not in display and len(display) < n_display:
            display.append(name)

    for r in region[:2]:
        _add(r["feature_name"])
    for d in position[:2]:
        _add(d["feature_name"])
    rest_r = [r["feature_name"] for r in region[2:]]
    rest_p = [d["feature_name"] for d in position[2:]]
    while (rest_r or rest_p) and len(display) < n_display:
        if rest_r:
            _add(rest_r.pop(0))
        if rest_p and len(display) < n_display:
            _add(rest_p.pop(0))
    return {"region": region, "position": position, "display": display,
            "by_name": by_name, "too_small": False}
