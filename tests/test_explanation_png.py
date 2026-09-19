import tempfile
import unittest
from pathlib import Path

from libs.explain_png import _prepare, render_explanation_png
from tests.test_explanation_output import render_fixture


def _collect(**kw):
    """The same explanation printed and collected, for comparison."""
    rows = []
    printed = render_fixture(**kw)
    render_fixture(sink=rows, **kw)
    return printed, rows


class ExplanationPngTest(unittest.TestCase):
    def test_sink_carries_exactly_the_printed_lines(self):
        # The image is only worth putting in a paper if it is the same output:
        # the rows the renderer draws must rebuild the printed text exactly.
        for kw in ({}, {"verbose": True}, {"group_size": 7}):
            with self.subTest(**kw):
                printed, rows = _collect(**kw)
                self.assertTrue(rows)
                self.assertEqual(printed, "\n".join(r[1] for r in rows) + "\n")

    def test_roles_follow_the_typography_contract(self):
        _, rows = _collect()
        by_role = {}
        for role, line, _bold in rows:
            by_role.setdefault(role, []).append(line)
        self.assertIn("  TabERA Explanation  # 0", by_role["title"])
        self.assertIn("  Prediction", by_role["section"])
        self.assertIn("  ① Your assigned group", by_role["section"])
        self.assertIn("     Group profile", by_role["subsection"])
        self.assertIn("     good — 90.0%", by_role["value"])
        self.assertEqual(by_role["rule"][0].strip(), "━" * 60)
        # every aligned row is monospace, and prose never is
        self.assertTrue(any(l.startswith("     Case 1") for l in by_role["mono"]))
        self.assertTrue(all("Case 1" not in l for l in by_role.get("text", [])))
        self.assertIn("     20 training cases; good 17 (85%), bad 3 (15%)",
                      by_role["text"])

    def test_secondary_notes_are_muted_only_in_the_researcher_view(self):
        _, rows = _collect()
        self.assertFalse([l for r, l, _ in rows if r == "muted"])
        _, rows = _collect(verbose=True)
        muted = [l for r, l, _ in rows if r == "muted"]
        self.assertTrue(any("Ranked by distribution shift" in l for l in muted))
        self.assertTrue(any("Routing mass" in l for l in muted))

    def test_bold_spans_mark_the_case_column_and_the_outcome(self):
        _, rows = _collect()
        marked = [(line, line[b[0]:b[1]]) for role, line, b in rows
                  if role == "mono" and b]
        self.assertTrue(marked)
        pieces = [p for _, p in marked]
        self.assertIn("✓ same", pieces)      # ① this-case column
        self.assertIn("0.750", pieces)       # ② this-case column
        self.assertIn("good", pieces)        # ③ outcome column
        for line, piece in marked:
            self.assertTrue(piece.strip(), f"empty emphasis in {line!r}")
            # the span is a whole cell, never a slice across a column boundary
            self.assertNotIn("  ", piece)
        # a header row is never emphasised
        self.assertNotIn("this case", pieces)
        self.assertNotIn("outcome", pieces)

    def test_layout_keeps_every_monospace_row_on_one_left_edge(self):
        # Indentation stays inside the string for monospace rows, so the font
        # places the columns; that is what makes the tables line up.
        _, rows = _collect()
        lines = _prepare(rows)
        self.assertEqual(len({round(l["x"], 6) for l in lines
                              if l["role"] == "mono"}), 1)
        self.assertTrue(all(l["w"] > 0 for l in lines
                            if l["role"] not in ("blank", "rule")))
        # proportional rows keep their indent as an offset instead
        indented = {round(l["x"], 6) for l in lines if l["role"] == "text"}
        headings = {round(l["x"], 6) for l in lines if l["role"] == "section"}
        self.assertTrue(min(indented) > min(headings))

    def test_render_writes_a_png_and_a_pdf(self):
        _, rows = _collect()
        with tempfile.TemporaryDirectory() as d:
            written = render_explanation_png(rows, Path(d) / "expl", dpi=100,
                                             warn=None)
            self.assertEqual([Path(p).suffix for p in written], [".png", ".pdf"])
            for p in written:
                self.assertGreater(Path(p).stat().st_size, 2000)

    def test_render_refuses_an_empty_explanation(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "no explanation lines"):
                render_explanation_png([], Path(d) / "expl")


if __name__ == "__main__":
    unittest.main()
