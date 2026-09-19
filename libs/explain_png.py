# -*- coding: utf-8 -*-
"""
libs/explain_png.py
===================
The terminal explanation, typeset as an image for a paper.

``analyze.print_explanation`` emits its output line by line through one
emitter (``out``). With ``sink=<list>`` those lines are collected instead of
printed, each tagged with the role it plays, and this module draws them. So
the image is not a second layout of the same data -- it is the *same lines*,
in the same order, with the same spacing. Nothing here recomputes, reorders,
re-wraps or re-words anything: give it rows whose text differs from the
terminal and it will faithfully draw the difference.

⚠ Not ``libs/explain_figure.py``. That module composes an editorial figure
  (bars, dumbbells, panels) from the observer outputs. This one is a
  typesetting pass over the text view, for when the paper wants to show the
  explanation a user actually sees.

Typography
──────────
```
titles, section headings   DejaVu Sans Bold
the prediction, and a decision that changed        DejaVu Sans Bold
prose                      DejaVu Sans
feature names/values, every aligned table  DejaVu Sans Mono
secondary notes            DejaVu Sans, dark gray
```
⚠ One deliberate departure from a literal reading of that list: a value
  *inside* a monospace table (an outcome, a case's own value) is set in
  DejaVu Sans **Mono** Bold, not DejaVu Sans Bold. A proportional face in a
  monospace row would move every column boundary after it; the monospace bold
  has the identical advance width, so the emphasis lands without shifting the
  table by one pixel. ``_assert_matched_advance`` checks that at render time
  and drops the emphasis rather than breaking alignment if it ever fails.

⚠ Alignment is preserved by construction, not by arithmetic: a monospace line
  is drawn as a single text object *including its leading spaces*, so the
  columns are placed by the font's own advances. Only proportional lines have
  their indent converted into an offset, where nothing has to line up.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ── palette ──────────────────────────────────────────────────────
BG = "#ffffff"      # pure white: the page, not a terminal
INK = "#111111"     # body text, near-black rather than pure black
MUTED = "#555555"   # secondary notes only
RULE = "#111111"    # the ━ dividers

# ── type scale, in points ────────────────────────────────────────
BASE = 9.0
LINE_H = 1.55 * BASE     # one text line, blank or not
BLANK_H = 0.62 * LINE_H  # a blank line, tightened for print
MARGIN_X = 14.0
MARGIN_Y = 14.0
RULE_LW = 1.1

# role -> (font kind, weight, size, colour)
STYLE = {
    "title":      ("sans", "bold",   BASE + 3.5, INK),
    "section":    ("sans", "bold",   BASE + 1.0, INK),
    "subsection": ("sans", "bold",   BASE,       INK),
    "value":      ("sans", "bold",   BASE + 3.5, INK),
    "text":       ("sans", "normal", BASE,       INK),
    "muted":      ("sans", "normal", BASE - 0.8, MUTED),
    "mono":       ("mono", "normal", BASE - 0.4, INK),
}
MONO_SIZE = STYLE["mono"][2]


# ── fonts ────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _font_path(kind: str, weight: str) -> str:
    """The DejaVu file itself, not a family name.

    Resolving by family would let a system font shadow DejaVu and change the
    figure between machines; a paper figure has to be reproducible, so the
    file is pinned and a substitution is an error rather than a surprise.
    """
    from matplotlib import font_manager as fm

    family = "DejaVu Sans Mono" if kind == "mono" else "DejaVu Sans"
    path = fm.findfont(fm.FontProperties(family=family, weight=weight),
                       fallback_to_default=False)
    if "DejaVu" not in Path(path).name:
        raise RuntimeError(
            f"{family} ({weight}) resolved to {path!r}; DejaVu ships with "
            f"matplotlib, so this means the font lookup found a substitute. "
            f"Refusing to render a figure in an unintended face.")
    return path


@lru_cache(maxsize=None)
def _ft(kind: str, weight: str, size: float):
    from matplotlib.ft2font import FT2Font

    font = FT2Font(_font_path(kind, weight))
    font.set_size(size, 72)   # 72 dpi -> one unit is one point
    return font


def _width(text: str, kind: str, weight: str, size: float) -> float:
    """Rendered advance width in points, kerning included."""
    if not text:
        return 0.0
    font = _ft(kind, weight, size)
    font.set_text(text, 0.0)
    return font.get_width_height()[0] / 64.0


@lru_cache(maxsize=None)
def _mono_char_w() -> float:
    """Advance of one monospace character; the indent unit for every line."""
    return _width("M" * 20, "mono", "normal", MONO_SIZE) / 20.0


@lru_cache(maxsize=None)
def _matched_advance() -> bool:
    """Whether mono bold steps exactly like mono regular (it should)."""
    a = _width("M" * 20, "mono", "normal", MONO_SIZE)
    b = _width("M" * 20, "mono", "bold", MONO_SIZE)
    return abs(a - b) < 1e-6


def _missing_glyphs(text: str, kind: str, weight: str) -> str:
    font = _ft(kind, weight, MONO_SIZE if kind == "mono" else BASE)
    return "".join(sorted({c for c in text
                           if not c.isspace() and font.get_char_index(ord(c)) == 0}))


# ── layout ───────────────────────────────────────────────────────

def _prepare(rows: Sequence[Tuple]) -> List[dict]:
    """One dict per emitted line: where it starts and how it is set."""
    out: List[dict] = []
    cw = _mono_char_w()
    for row in rows:
        role, text = row[0], row[1]
        bold = row[2] if len(row) > 2 else None
        if role == "blank":
            out.append({"role": "blank", "h": BLANK_H})
            continue
        if role == "rule":
            # ⚠ Width is filled in once the content width is known, not taken
            #   from the 60 characters the terminal draws. A divider that
            #   stops short of the text it divides reads as a mistake on a
            #   page; on a terminal the same run just wraps differently.
            out.append({"role": "rule", "h": LINE_H, "x": MARGIN_X, "w": 0.0})
            continue
        kind, weight, size, colour = STYLE.get(role, STYLE["text"])
        if kind == "mono":
            # Leading spaces stay in the string: the font places the columns.
            body, x = text, MARGIN_X
        else:
            indent = len(text) - len(text.lstrip(" "))
            body, x = text.strip(), MARGIN_X + indent * cw
        out.append({"role": role, "h": LINE_H, "x": x, "text": body,
                    "kind": kind, "weight": weight, "size": size,
                    "colour": colour,
                    "bold": bold if (kind == "mono" and _matched_advance()) else None,
                    "w": _width(body, kind, weight, size)})
    return out


def render_explanation_png(
    rows: Sequence[Tuple],
    out_path,
    *,
    dpi: int = 300,
    also_pdf: bool = True,
    warn=print,
) -> List[str]:
    """Draw collected explanation lines to ``<out_path>.png`` (and ``.pdf``).

    ``rows`` are the ``(role, text, bold_span)`` tuples ``print_explanation``
    appends to its ``sink``. Returns the paths written.

    A PDF is written beside the PNG because the content is text: a vector
    version stays sharp at any column width a journal chooses, while the PNG
    is the one asked for and the one most submission systems accept.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = _prepare(rows)
    if not lines:
        raise ValueError("no explanation lines to render")

    for ln in lines:
        if ln["role"] in ("blank", "rule"):
            continue
        miss = _missing_glyphs(ln["text"], ln["kind"], ln["weight"])
        if miss and warn is not None:
            warn(f"  !  [explain_png] {ln['kind']} face has no glyph for "
                 f"{miss!r}; it will render as a box. Line: {ln['text'][:60]!r}")

    content = [ln for ln in lines if ln["role"] not in ("blank", "rule")]
    width = (max((ln["x"] + ln["w"] for ln in content), default=60 * _mono_char_w())
             + MARGIN_X)
    for ln in lines:
        if ln["role"] == "rule":
            ln["w"] = width - 2 * MARGIN_X
    height = sum(ln["h"] for ln in lines) + 2 * MARGIN_Y

    fig = plt.figure(figsize=(width / 72.0, height / 72.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    from matplotlib.font_manager import FontProperties

    def _fp(kind, weight, size):
        return FontProperties(fname=_font_path(kind, weight), size=size)

    y = height - MARGIN_Y
    for ln in lines:
        y -= ln["h"]
        if ln["role"] == "blank":
            continue
        # Baseline a little above the cell floor, so ascenders and descenders
        # sit inside their own line rather than crowding the next one.
        yb = y + 0.30 * LINE_H
        if ln["role"] == "rule":
            ax.plot([ln["x"], ln["x"] + ln["w"]], [yb + 0.12 * LINE_H] * 2,
                    color=RULE, lw=RULE_LW, solid_capstyle="butt")
            continue
        text, kind, weight, size = ln["text"], ln["kind"], ln["weight"], ln["size"]
        span = ln["bold"]
        if not span:
            ax.text(ln["x"], yb, text, fontproperties=_fp(kind, weight, size),
                    color=ln["colour"], va="baseline", ha="left")
            continue
        # Emphasis inside a monospace row: three runs, each positioned from
        # the regular face, which the bold face steps identically to.
        s, t = span
        x0 = ln["x"]
        ax.text(x0, yb, text[:s], fontproperties=_fp(kind, weight, size),
                color=ln["colour"], va="baseline", ha="left")
        x1 = x0 + _width(text[:s], kind, weight, size)
        ax.text(x1, yb, text[s:t], fontproperties=_fp(kind, "bold", size),
                color=ln["colour"], va="baseline", ha="left")
        x2 = x0 + _width(text[:t], kind, weight, size)
        if text[t:]:
            ax.text(x2, yb, text[t:], fontproperties=_fp(kind, weight, size),
                    color=ln["colour"], va="baseline", ha="left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    png = out_path.with_suffix(".png")
    fig.savefig(png, dpi=dpi, facecolor=BG)
    written.append(str(png))
    if also_pdf:
        pdf = out_path.with_suffix(".pdf")
        fig.savefig(pdf, facecolor=BG)
        written.append(str(pdf))
    plt.close(fig)
    return written
