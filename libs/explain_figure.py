# -*- coding: utf-8 -*-
"""
libs/explain_figure.py
======================
Publication figure for a single TabERA explanation.

It renders the *same* observer outputs that ``analyze.print_explanation``
prints, formatted through the same ``ExplanationFormatter``, so a figure in the
paper cannot state a different number from the text the code prints.

Layout, and why it is split this way
────────────────────────────────────
    Prediction        the computation.  z = (W_eff*c + b) + W_eff*d is an
                      identity in *logit* space, so the panel shows the two
                      logit terms adding up, and the two probabilities beside
                      each other -- each its own sigmoid, never subtracted.
    (1) (2) (3)       description.  Retrieval and the within-region position
                      are not inputs to z. The rule drawn between the rows is
                      the figure's main claim, not decoration.

⚠ Display only. Nothing here computes, selects or re-derives a value: it draws
  what diagnostics.py produced and analyze.py already prints.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ── palette ──────────────────────────────────────────────────────
# Categorical slots 1-2 of the reference palette, validated on a white paper
# surface (lightness band, chroma floor, protan/deutan dE 24.7, normal-vision
# dE 33.6, contrast 4.42:1 / 3.20:1). Class identity only.
CLASS_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
RULE = "#c3c2b7"
# Emphasis pair for panel (3): the within-region position says nothing about
# class, so it must not borrow a class hue. One dark neutral + the track gray.
EMPH = "#52514e"
TRACK = "#e1e0d9"
# Two shades of one hue for the before/after dumbbell (blue 300 / 450); the
# baseline and the final probability are the same quantity at two stages.
SHADE_LO = "#6da7ec"
SHADE_HI = "#2a78d6"


def _rc():
    import matplotlib
    return {
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial", "Helvetica"],
        "text.color": INK,
        "pdf.fonttype": 42,      # embed as TrueType, not Type 3: journals ask for it
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


class _Canvas:
    """A single 0..1 x 0..1 axes with helpers in figure-relative units.

    Using one axes rather than a grid of them keeps every element on one
    coordinate system, so panel rules, bars and text can be positioned against
    each other directly.
    """

    def __init__(self, ax, w_in: float, h_in: float):
        self.ax = ax
        self.w_in, self.h_in = w_in, h_in
        self.aspect = h_in / w_in

    # ── primitives ───────────────────────────────────────────────
    def bar(self, x, y, w, h, color, *, round_end="right", z=3, alpha=1.0):
        """Horizontal bar, rounded data-end, square at the baseline.

        ⚠ Built as an explicit path rather than a FancyBboxPatch. The boxstyle
          route pads the box in the *mutated* space, so with a non-unit
          mutation_aspect the drawn body does not land where the caller asked
          and the square-off patch separates from it -- visible as a detached
          nub beside every stacked segment.
        """
        from matplotlib.patches import Rectangle, PathPatch
        from matplotlib.path import Path as MPath
        if w <= 0:
            return
        ry = min(h * 0.30, w * 0.5 / max(self.aspect, 1e-9))
        rx = ry * self.aspect          # equal radius in inches on both axes
        if round_end is None or w <= rx * 2.0:
            self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                                        edgecolor="none", zorder=z, alpha=alpha))
            return
        k = 0.5523                      # circle-to-Bezier constant
        x1, y1 = x + w, y + h
        if round_end == "right":
            verts = [(x, y), (x1 - rx, y),
                     (x1 - rx + k * rx, y), (x1, y + ry - k * ry), (x1, y + ry),
                     (x1, y1 - ry),
                     (x1, y1 - ry + k * ry), (x1 - rx + k * rx, y1), (x1 - rx, y1),
                     (x, y1), (x, y)]
        else:                            # rounded left end, square at the right
            verts = [(x1, y), (x + rx, y),
                     (x + rx - k * rx, y), (x, y + ry - k * ry), (x, y + ry),
                     (x, y1 - ry),
                     (x, y1 - ry + k * ry), (x + rx - k * rx, y1), (x + rx, y1),
                     (x1, y1), (x1, y)]
        codes = [MPath.MOVETO, MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO,
                 MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                 MPath.LINETO, MPath.CLOSEPOLY]
        self.ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color,
                                    edgecolor="none", zorder=z, alpha=alpha))

    def track(self, x, y, w, h, color=TRACK, z=1):
        from matplotlib.patches import Rectangle
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color,
                                    edgecolor="none", zorder=z))

    def text(self, x, y, s, *, size=7.2, color=INK2, ha="left", va="center",
             weight="normal", z=6, **kw):
        return self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                            fontweight=weight, zorder=z, **kw)

    def fit_text(self, x, y, s, max_x, *, size=6.3, color=INK2, **kw):
        """Left-aligned text that is never clipped by the panel edge.

        Measured, not guessed: a label that would overflow is re-anchored to
        the right edge, and only if it still will not fit is it shortened with
        an ellipsis. Cropping the end of a label is worse than shortening it
        deliberately.
        """
        t = self.text(x, y, s, size=size, color=color, **kw)
        try:
            rend = self.ax.figure.canvas.get_renderer()
        except Exception:
            return t
        inv = self.ax.transData.inverted()

        def _x1(txt):
            return txt.get_window_extent(renderer=rend).transformed(inv).x1

        if _x1(t) <= max_x:
            return t
        t.remove()
        t = self.text(max_x, y, s, size=size, color=color, ha="right", **kw)
        bb = t.get_window_extent(renderer=rend).transformed(inv)
        if bb.x0 >= x:
            return t
        t.remove()
        cut = s
        while len(cut) > 6:
            cut = cut[:-2]
            t = self.text(x, y, cut + "…", size=size, color=color, **kw)
            if _x1(t) <= max_x:
                return t
            t.remove()
        return self.text(x, y, cut + "…", size=size, color=color, **kw)

    def rule(self, x0, x1, y, color=GRID, lw=0.8, z=2):
        self.ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt",
                     zorder=z)

    def vrule(self, x, y0, y1, color=RULE, lw=0.8, z=2, ls="-"):
        self.ax.plot([x, x], [y0, y1], color=color, lw=lw, ls=ls,
                     solid_capstyle="butt", zorder=z)

    def dot(self, x, y, color, *, size=34, z=5, ring=SURFACE, ringw=1.6):
        self.ax.scatter([x], [y], s=size, c=color, edgecolors=ring,
                        linewidths=ringw, zorder=z)

    def panel_title(self, x, y, s, sub=None):
        self.text(x, y, s, size=8.4, color=INK, weight="bold")
        if sub:
            self.text(x, y - 0.031, sub, size=6.6, color=MUTED)

    def stacked(self, x, y, w, h, parts, *, gap_frac=0.006, z=3,
                label_min=0.16):
        """Part-to-whole bar. `parts` = [(prop, color, label), ...].

        Segments are separated by a gap in the surface colour, never a stroke.
        A label is drawn inside a segment only when it fits.
        """
        n = len(parts)
        gap = gap_frac * w if n > 1 else 0.0
        avail = w - gap * (n - 1)
        cx = x
        for i, (prop, color, label) in enumerate(parts):
            seg = avail * float(prop)
            end = "right" if i == n - 1 else (None if i else "left")
            self.bar(cx, y, seg, h, color, round_end=end, z=z)
            if label and prop >= label_min:
                self.text(cx + seg / 2, y + h / 2, label, size=6.4,
                          color=SURFACE, ha="center", weight="bold", z=z + 1)
            cx += seg + gap


def _class_color(code: Optional[int], pred_code: Optional[int]) -> str:
    """Colour for a target class. Identity, so a fixed slot per class code."""
    if code is None:
        return MUTED
    return CLASS_HUES[int(code) % len(CLASS_HUES)]


def render_explanation_figure(
    explanations: List[dict],
    sample_idx: int,
    out_path,
    *,
    col_names=None,
    cat_category_names=None,
    quantile_transformer=None,
    num_cols=None,
    pred_info: Optional[dict] = None,
    target_class_names=None,
    tasktype: Optional[str] = None,
    dataset_name: str = "",
    max_neighbors: int = 3,
    max_features: int = 2,   # display budget for the figure, not a threshold
    also_pdf: bool = True,
    dpi: int = 300,
) -> List[str]:
    """Render one explanation to PNG (and PDF) and return the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    from libs.explain_format import ExplanationFormatter

    fmt = ExplanationFormatter(
        col_names=col_names, cat_category_names=cat_category_names,
        quantile_transformer=quantile_transformer, num_cols=num_cols,
        target_class_names=target_class_names, tasktype=tasktype)

    e = explanations[sample_idx]
    proto = e.get("prototype") or {}
    le = e.get("local_evidence")
    nbrs = e.get("neighbors") or []
    dv = e.get("prototype_deviation")
    gc = e.get("group_stats")
    rp = e.get("region_position")
    pred_info = pred_info or {}
    pred_code = pred_info.get("pred_code")
    is_clf = tasktype != "regression"

    W_IN, H_IN = 9.0, 5.5
    with plt.rc_context(_rc()):
        fig = plt.figure(figsize=(W_IN, H_IN))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        C = _Canvas(ax, W_IN, H_IN)

        L, R = 0.035, 0.965

        # ── header ────────────────────────────────────────────────
        title = "TabERA explanation"
        if dataset_name:
            title += f" — {dataset_name}"
        title += f" · test sample #{sample_idx}"
        C.text(L, 0.965, title, size=10.4, color=INK, weight="bold")

        # Legend: two or more classes are never identified by colour alone.
        if is_clf and target_class_names is not None and len(target_class_names) <= 8:
            lx = R
            for code in range(len(target_class_names) - 1, -1, -1):
                nm = str(target_class_names[code])
                C.text(lx, 0.967, nm, size=7.2, color=INK2, ha="right")
                lx -= 0.010 + 0.0060 * len(nm)
                C.dot(lx, 0.967, _class_color(code, pred_code), size=26, ringw=0)
                lx -= 0.020

        # ══ PANEL A — the computation ════════════════════════════
        C.panel_title(L, 0.912, "Prediction",
                      "the computation — the only terms in $z$")

        AL0, AL1 = L, 0.430          # left half: probability
        AR0, AR1 = 0.520, R          # right half: the logit identity

        # -- A-left: the same quantity before and after the correction
        pp = (dv or {}).get("prob_proto")
        pf = (dv or {}).get("prob_final")
        label = pred_info.get("pred_label", "")
        conf = pred_info.get("pred_confidence")

        C.text(AL0, 0.835, "Predicted class", size=6.8, color=MUTED)
        C.text(AL0, 0.782, f"{label}", size=16.0, color=INK, weight="bold")
        if conf is not None:
            C.text(AL0, 0.742, f"{conf:.1%} confidence", size=8.0, color=INK2)

        if pp is not None and pf is not None:
            ty, th = 0.672, 0.020
            C.track(AL0, ty, AL1 - AL0, th, GRID)
            for frac, lab in ((0.0, "0%"), (0.5, "50%"), (1.0, "100%")):
                xf = AL0 + (AL1 - AL0) * frac
                C.vrule(xf, ty - 0.010, ty, RULE, lw=0.7)
                C.text(xf, 0.616, lab, size=6.0, color=MUTED, ha="center")
            x_pp = AL0 + (AL1 - AL0) * float(pp)
            x_pf = AL0 + (AL1 - AL0) * float(pf)
            C.ax.plot([x_pp, x_pf], [ty + th / 2, ty + th / 2], color=SHADE_HI,
                      lw=2.0, solid_capstyle="butt", zorder=4)
            C.dot(x_pp, ty + th / 2, SHADE_LO, size=42)
            C.dot(x_pf, ty + th / 2, SHADE_HI, size=54)
            # One label above its own dot, one below: they sit at nearly the
            # same x, so putting both on one side would detach them.
            C.text(x_pf, ty + th + 0.022, f"final {pf:.1%}", size=6.8,
                   color=INK, ha="center", weight="bold")
            C.text(x_pp, ty - 0.028, f"region baseline {pp:.1%}", size=6.8,
                   color=INK2, ha="center")
            C.text(AL0, 0.585,
                   "each end is its own sigmoid of the logits at right — the two\n"
                   "probabilities are never subtracted; only the logits add",
                   size=6.4, color=MUTED, va="top")

        # -- A-right: the exact logit decomposition ----------------
        if dv is not None and dv.get("logit_proto") is not None:
            lp = float(dv["logit_proto"]); ld = float(dv["logit_dev"])
            lf = lp + ld
            binary = (is_clf and target_class_names is not None
                      and len(target_class_names) == 2)
            if binary:
                pos_c, neg_c = _class_color(1, pred_code), _class_color(0, pred_code)
                head = ('one binary logit:  + favours "%s"   ·   − favours "%s"'
                        % (target_class_names[1], target_class_names[0]))
            else:
                pos_c = neg_c = SHADE_HI
                head = "the predicted class's own channel"
            C.text(AR0, 0.835, head, size=6.6, color=MUTED)

            rows = [("region baseline   $W_{\\mathrm{eff}}c+b$", lp),
                    ("correction   $W_{\\mathrm{eff}}d$", ld),
                    ("final   $z$", lf)]
            span = max(abs(lp), abs(ld), abs(lf), 1e-9) * 1.32
            zx = (AR0 + AR1) / 2
            half = (AR1 - AR0) / 2
            bh, y0, dy = 0.024, 0.775, 0.062
            C.vrule(zx, y0 - dy * 2 - 0.010, y0 + bh + 0.026, RULE, lw=0.8)
            for i, (nm, v) in enumerate(rows):
                yy = y0 - dy * i
                if i == 2:
                    C.rule(AR0, AR1, yy + bh + 0.030, GRID, lw=0.8)
                w = half * abs(v) / span
                col = pos_c if v >= 0 else neg_c
                if v >= 0:
                    C.bar(zx, yy, w, bh, col, round_end="right")
                else:
                    C.bar(zx - w, yy, w, bh, col, round_end="left")
                C.text(AR0, yy + bh + 0.014, nm, size=6.8, color=INK2)
                lx = zx + w + 0.010 if v >= 0 else zx - w - 0.010
                C.text(lx, yy + bh / 2, f"{v:+.4f}", size=7.2,
                       color=INK if i == 2 else INK2,
                       ha="left" if v >= 0 else "right",
                       weight="bold" if i == 2 else "normal")
            C.text(AR0, 0.600,
                   "an identity, not an approximation — the readout is one\n"
                   "shared linear map: $z=(W_{\\mathrm{eff}}c+b)+W_{\\mathrm{eff}}d$",
                   size=6.4, color=MUTED, va="top")

        # ── separating rule: computation above, description below ─
        SEP = 0.545
        C.rule(L, R, SEP, RULE, lw=1.0)
        C.text(L, SEP - 0.028,
               "Below: what this case is like. Neither retrieval nor the within-region "
               "position is an input to $z$, and neither is attribution.",
               size=6.8, color=MUTED)

        TT = 0.442        # panel titles on the lower row
        TOP = 0.360       # first content row

        # ══ PANEL B — the region ═════════════════════════════════
        bx0, bx1 = L, 0.285
        region = fmt.region_name(proto.get("assigned_group"))
        gsize = (le or {}).get("group_size")
        if gsize is None:
            ti = proto.get("target_info")
            gsize = ti.get("n") if ti else None
        C.panel_title(bx0, TT, "① Predictive region",
                      f"{region}" + (f" · {int(gsize)} training cases" if gsize else ""))
        gcnt = (le or {}).get("group_label_counts")
        if is_clf and gcnt and gsize:
            items = fmt.dist_items(gcnt, int(gsize), max_items=3)
            parts = [(it["prop"], _class_color(it["code"], pred_code),
                      f"{it['prop']:.0%}") for it in items]
            C.stacked(bx0, TOP, bx1 - bx0, 0.028, parts)
            C.text(bx0, TOP - 0.028,
                   " · ".join(f"{it['name']} {it['count']}/{it['total']}"
                              for it in items), size=6.6, color=INK2)
            C.text(bx0, TOP - 0.070,
                   "a regional anchor, not a class representative:\n"
                   "assignment and the EMA update never read labels",
                   size=6.4, color=MUTED, va="top")
        elif le is not None and le.get("group_mean") is not None:
            C.text(bx0, TOP, f"target mean {le['group_mean']:.4g} · "
                             f"std {le['group_std']:.4g}", size=7.0, color=INK2)

        # ══ PANEL C — the evidence ═══════════════════════════════
        cx0, cx1 = 0.325, 0.600
        nnb = int((le or {}).get("n_neighbors", len(nbrs)) or 0)
        # retrieve() expands beyond the assigned group only when the group
        # cannot supply k candidates; do not call that "inside the region".
        _gs = (le or {}).get("group_size")
        _scope = ("nearest, search expanded beyond the region"
                  if (_gs is not None and nnb and int(_gs) < nnb)
                  else "nearest inside the region")
        C.panel_title(cx0, TT, "② Similar past cases",
                      f"{nnb} {_scope} · evidence only")
        ry = TOP - 0.052
        if is_clf and le is not None and le.get("label_counts") and nnb:
            items = fmt.dist_items(le["label_counts"], nnb, max_items=3)
            parts = [(it["prop"], _class_color(it["code"], pred_code),
                      f"{it['count']}/{it['total']}") for it in items]
            C.stacked(cx0, TOP, cx1 - cx0, 0.028, parts)
            # The local mix is never shown without the group mix: 7/8 says
            # nothing until the region's own share is on the same scale.
            if gcnt and gsize:
                top_g = fmt.dist_items(gcnt, int(gsize), max_items=3)[0]
                gx = cx0 + (cx1 - cx0) * float(top_g["prop"])
                C.vrule(gx, TOP - 0.016, TOP + 0.028, INK, lw=1.0, ls=":")
                C.text(gx, TOP - 0.026,
                       f"whole region {top_g['prop']:.0%}", size=6.2,
                       color=INK2, ha="center")
                ry = TOP - 0.060
        rows = list(nbrs[:max_neighbors])
        contrast = None
        if is_clf and pred_code is not None:
            contrast = next((nb for nb in nbrs
                             if nb.get("label") is not None
                             and int(round(float(nb["label"]))) != int(pred_code)), None)
            if contrast is not None and contrast not in rows and max_neighbors > 0:
                rows = list(nbrs[:max_neighbors - 1]) + [contrast]
        for nb in rows:
            code = (int(round(float(nb["label"])))
                    if (is_clf and nb.get("label") is not None) else None)
            C.dot(cx0 + 0.005, ry, _class_color(code, pred_code), size=22, ringw=0)
            sid = nb.get("sample_id")
            sid_s = (f"train #{sid}" if sid is not None and sid >= 0
                     else f"memory #{nb['memory_idx']}")
            tag = "  · contrast" if (contrast is not None and nb is contrast) else ""
            C.text(cx0 + 0.018, ry,
                   f"{sid_s}   {fmt.label_name(nb.get('label'))}{tag}",
                   size=6.6, color=INK2)
            C.text(cx1, ry, f"{nb['similarity']:.3f}", size=6.6, color=MUTED,
                   ha="right")
            ry -= 0.032
        if contrast is not None:
            gaps = fmt.gap_summary(contrast, 2)
            if gaps:
                ry -= 0.006
                C.text(cx0, ry, "differs from the contrast case:", size=6.4,
                       color=MUTED)
                ry -= 0.026
                for g in gaps:
                    C.text(cx0 + 0.010, ry, g, size=6.4, color=INK2)
                    ry -= 0.024
        elif is_clf and nbrs:
            C.text(cx0, ry - 0.006,
                   f"no contrasting case among the {len(nbrs)} retrieved",
                   size=6.4, color=MUTED)

        # ══ PANEL D — the position ═══════════════════════════════
        dx0, dx1 = 0.640, R
        C.panel_title(dx0, TT, "③ Position relative to the region",
                      "shares of this region's training cases")
        # This panel stacks a label above every bar, so it starts a little
        # lower than the other two to clear its own subtitle.
        yy = TOP - 0.012
        BH = 0.022
        if gc:
            nums = list(gc.get("numeric") or [])
            cats = [d for d in (gc.get("categorical") or [])
                    if d.get("differs_from_mode", True) or d.get("absent_from_group", False)]
            budget = max(0, int(max_features))
            n_num = min(len(nums), (budget + 1) // 2) if (nums and cats) else min(len(nums), budget)
            n_cat = min(len(cats), budget - n_num)
            n_num = min(len(nums), n_num + max(0, budget - n_num - n_cat))
            n_cat = min(len(cats), budget - n_num)
            for d in nums[:n_num]:
                vr = fmt.real_numeric(d["feature_idx"], d["value"])
                mr = fmt.real_numeric(d["feature_idx"], d["group_mean"])
                v_s = fmt.pretty_num(vr) if vr is not None else f"{d['value']:.3f}"
                m_s = fmt.pretty_num(mr) if mr is not None else f"{d['group_mean']:.3f}"
                below = float(d.get("group_pct_below", 0.0))
                equal = float(d.get("group_pct_equal", 0.0))
                above = max(0.0, 1.0 - below - equal)
                C.text(dx0, yy + BH + 0.018, f"{d['feature_name']} = {v_s}",
                       size=6.9, color=INK)
                C.text(dx1, yy + BH + 0.018, f"region reference {m_s}",
                       size=6.3, color=MUTED, ha="right")
                # Emphasis, not identity: the band this case falls in is inked,
                # the rest is track. No class hue — position says nothing
                # about the label.
                C.stacked(dx0, yy, dx1 - dx0, BH,
                          [(below, TRACK, ""), (equal, EMPH, ""), (above, TRACK, "")],
                          label_min=2.0)
                side = "higher" if below >= above else "lower"
                share = below if below >= above else above
                txt = f"■ this case's band · {side} than {share:.0%}"
                if equal > 0.0005:
                    txt += f", equal to {equal:.0%}"
                C.fit_text(dx0, yy - 0.022, txt, dx1, size=6.3, color=INK2)
                yy -= 0.092
            for d in cats[:n_cat]:
                value = fmt.fmt_cat_value(d["feature_name"], d["value"])
                mode = fmt.fmt_cat_value(d["feature_name"], d["group_mode"])
                fq = float(d.get("group_freq", 0.0))
                mf = float(d.get("group_mode_freq", 0.0))
                C.fit_text(dx0, yy + BH + 0.018, f"{d['feature_name']} = {value}",
                           dx1, size=6.9, color=INK)
                # Both bars are shares of the same region, so one scale. The
                # two readings go on one caption line beneath them: a category
                # name can be long, and a label parked after a bar end has
                # nowhere to go when it reaches the panel edge.
                C.track(dx0, yy, dx1 - dx0, BH, "#f5f4f0")
                C.bar(dx0, yy, (dx1 - dx0) * fq, BH, EMPH)
                C.track(dx0, yy - 0.026, dx1 - dx0, BH * 0.60, "#f5f4f0")
                C.bar(dx0, yy - 0.026, (dx1 - dx0) * mf, BH * 0.60, GRID)
                C.fit_text(dx0, yy - 0.050,
                           f"■ {fq:.0%} this case's value   ·   "
                           f"▪ {mf:.0%} most common: {mode}",
                           dx1, size=6.3, color=INK2)
                yy -= 0.126
        if rp is not None:
            # A distance rank is a single position, not a share of the region,
            # so it is marked rather than filled: inking one side would read as
            # "this case's band" and mean something different from the rows
            # above.
            farther = float(rp["group_pct"])
            C.text(dx0, yy + BH + 0.018,
                   "distance to the region centre, in the representation",
                   size=6.9, color=INK)
            C.track(dx0, yy, dx1 - dx0, BH, TRACK)
            mx = dx0 + (dx1 - dx0) * farther
            C.vrule(mx, yy - 0.006, yy + BH + 0.006, EMPH, lw=1.4, z=4)
            C.dot(mx, yy + BH / 2, EMPH, size=30, z=5)
            C.text(dx0, yy - 0.024, "closest", size=6.0, color=MUTED)
            C.text(dx1, yy - 0.024, "farthest", size=6.0, color=MUTED, ha="right")
            C.text(dx0, yy - 0.048, (fmt.centre_distance_position(rp) or ""),
                   size=6.3, color=INK2)

        # ── footer ────────────────────────────────────────────────
        C.rule(L, R, 0.052, GRID, lw=0.8)
        C.text(L, 0.028,
               "① is the assignment the prediction used · ② is retrieved from that "
               "same region and never enters $z$ · ③ is descriptive statistics, "
               "not attribution.",
               size=6.4, color=MUTED)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = []
        png = out_path.with_suffix(".png")
        fig.savefig(png, dpi=dpi, facecolor=SURFACE)
        written.append(str(png))
        if also_pdf:
            pdf = out_path.with_suffix(".pdf")
            fig.savefig(pdf, facecolor=SURFACE)
            written.append(str(pdf))
        plt.close(fig)
    return written
