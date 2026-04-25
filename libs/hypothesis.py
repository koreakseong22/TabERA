"""
hypothesis.py
=============
User-defined hypothesis / rule engine.

A Hypothesis is a logical predicate over feature values that the model
will attempt to verify and use as a soft conditioning signal during
prototype routing and evidence selection.

Example
-------
>>> from hypo_tabr import Hypothesis, HypothesisEngine
>>> h1 = Hypothesis("age > 30 and income < 50000", label="low_income_adult")
>>> h2 = Hypothesis("credit_score >= 700", label="good_credit")
>>> engine = HypothesisEngine([h1, h2])
>>> scores = engine.evaluate(X_batch)   # Tensor[B, H]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hypothesis dataclass
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """
    A single user-defined hypothesis.

    Parameters
    ----------
    rule : str
        A Python-evaluable boolean expression over column names.
        Supported operators: >, <, >=, <=, ==, !=, and, or, not, in, (...)
        Numeric literals and string literals are allowed.
    label : str
        Human-readable name shown in explanations (e.g. "low_income_adult").
    weight : float
        Prior importance weight. Default 1.0.
    description : str
        Optional longer description for the explanation report.
    """
    rule: str
    label: str
    weight: float = 1.0
    description: str = ""
    _compiled: Optional[Callable] = field(default=None, init=False, repr=False)

    def compile(self, columns: List[str]) -> None:
        """
        Pre-compile the rule into a vectorised callable.

        The rule is safely evaluated column-by-column using numpy arrays.
        """
        # Sanitise: allow only safe tokens
        safe_pattern = re.compile(
            r"^[\w\s\d\.\,\+\-\*\/\%\(\)\[\]\>\<\=\!\&\|\'\"]+$"
        )
        if not safe_pattern.match(self.rule):
            raise ValueError(
                f"Hypothesis rule contains unsafe characters: '{self.rule}'"
            )

        col_set = set(columns)
        # Build a lambda that maps a dict {col: np.array} -> bool array
        code = self.rule
        # Replace column names with dict lookups
        for col in sorted(col_set, key=len, reverse=True):  # longest first
            if col in code:
                code = re.sub(r'\b' + re.escape(col) + r'\b', f"_d['{col}']", code)

        expr = f"lambda _d: ({code})"
        try:
            self._compiled = eval(expr)  # noqa: S307
        except SyntaxError as e:
            raise ValueError(f"Invalid hypothesis rule '{self.rule}': {e}") from e

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Evaluate this hypothesis on a batch.

        Parameters
        ----------
        data : dict[col_name -> np.ndarray of shape (B,)]

        Returns
        -------
        np.ndarray of shape (B,) with dtype float32 in [0.0, 1.0].
            Hard 0/1 for exact Boolean rules.
        """
        if self._compiled is None:
            raise RuntimeError("Call .compile(columns) before .evaluate()")
        result = self._compiled(data).astype(np.float32)
        return result


# ---------------------------------------------------------------------------
# HypothesisEngine
# ---------------------------------------------------------------------------

class HypothesisEngine(nn.Module):
    """
    Manages a collection of user-defined hypotheses and produces a soft
    hypothesis activation tensor for each input batch.

    The activations are used as a bias signal in prototype routing:
    - Each prototype can be associated with one or more hypotheses.
    - The engine promotes prototypes whose hypotheses are satisfied.

    Parameters
    ----------
    hypotheses : list[Hypothesis]
        The user-defined hypotheses.
    learnable_weights : bool
        If True, hypothesis weights are learnable (log-softmax parametrised).
        If False, fixed priors are used.
    temperature : float
        Temperature for soft activation blending.
    """

    def __init__(
        self,
        hypotheses: List[Hypothesis],
        learnable_weights: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.hypotheses = hypotheses
        self.temperature = temperature
        self.n_hypotheses = len(hypotheses)

        # Learnable log-weights per hypothesis
        prior = torch.tensor(
            [h.weight for h in hypotheses], dtype=torch.float32
        )
        if learnable_weights:
            self.log_weights = nn.Parameter(torch.log(prior))
        else:
            self.register_buffer("log_weights", torch.log(prior))

        self._columns: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_columns(self, columns: List[str]) -> None:
        """Compile all hypotheses against the dataset column list."""
        self._columns = columns
        for h in self.hypotheses:
            h.compile(columns)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X: torch.Tensor,
        column_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute hypothesis activation scores for a batch.

        Parameters
        ----------
        X : Tensor[B, F]   (float, already encoded)
        column_names : list[str] of length F  (required on first call)

        Returns
        -------
        h_scores : Tensor[B, H]
            Weighted soft activations in [0, 1] per hypothesis.
        """
        cols = column_names or self._columns
        if cols is None:
            raise RuntimeError("Provide column_names or call set_columns() first.")
        if self._columns is None:
            self.set_columns(cols)

        B = X.shape[0]
        X_np = X.detach().cpu().numpy()

        # Build col dict
        col_dict = {col: X_np[:, i] for i, col in enumerate(cols)}

        # Evaluate each hypothesis
        scores = np.stack(
            [h.evaluate(col_dict) for h in self.hypotheses], axis=1
        )  # (B, H)

        h_tensor = torch.tensor(scores, dtype=torch.float32, device=X.device)

        # Apply learned weights
        weights = torch.softmax(self.log_weights / self.temperature, dim=0)  # (H,)
        h_scores = h_tensor * weights.unsqueeze(0)  # (B, H)

        return h_scores

    # ------------------------------------------------------------------
    # Explanation helpers
    # ------------------------------------------------------------------

    def active_hypotheses(
        self,
        h_scores: torch.Tensor,
        threshold: float = 0.5,
    ) -> List[List[Tuple[str, float]]]:
        """
        For each sample in the batch, return the list of active hypotheses
        (score >= threshold) with their labels and scores.

        Returns
        -------
        list of lists: [ [(label, score), ...], ... ]  length B
        """
        out = []
        scores_np = h_scores.detach().cpu().numpy()
        for b in range(scores_np.shape[0]):
            active = [
                (self.hypotheses[i].label, float(scores_np[b, i]))
                for i in range(self.n_hypotheses)
                if scores_np[b, i] >= threshold
            ]
            out.append(active)
        return out

    def summary(self) -> str:
        lines = ["HypothesisEngine Summary", "=" * 40]
        weights = torch.softmax(self.log_weights, dim=0).detach().cpu().numpy()
        for i, h in enumerate(self.hypotheses):
            lines.append(
                f"[{i}] '{h.label}' (w={weights[i]:.3f})\n"
                f"     rule: {h.rule}\n"
                f"     desc: {h.description or '—'}"
            )
        return "\n".join(lines)
