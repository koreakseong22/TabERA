"""
libs/prototypes.py
==================
CentroidLayer — the prototype partition of the learned representation.

What it does
────────────
(1) Prototype memory
    centroid_emb (P, D), unit vectors. Not neural parameters: they are
    maintained as an EMA of the embeddings assigned to them, and unused ones
    are reinitialised from observed embeddings. Neither step reads labels.

(2) Hard routing with a straight-through estimator
    a = argmax_p cos(q, c_p).
    Forward takes the hard argmax; backward passes the softmax gradient
    (Bengio et al. 2013; the hard-assignment trick from VQ-VAE,
    van den Oord et al. 2017). This single assignment fixes both the
    prediction baseline c and the retrieval pool G(a).

(3) Group text labels — the material for explanation layer (1)
    label_groups_by_target() answers "which target does this group
    correspond to", which is the one thing layers (2) and (3) cannot say.
    label_all_groups() adds the group means of the features that
    characterise it, as raw values rather than qualitative bands.
    Both are computed in supervised.py right after each regroup_update().

    These used to be a separate module (libs/group_labels.py). They are
    helpers used only by CentroidLayer, so there was no reason to keep the
    split.

⚠ The prototype layer carries **no gradient-based objective of its own**.
  Assignment is discrete, the update is EMA, and dead-prototype recovery is
  maintenance. The only training signal in the model is cross-entropy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Group text-labelling helpers (formerly libs/group_labels.py)
# ─────────────────────────────────────────────────────────────
# These exist only to fill CentroidLayer.sample_groups / target_labels /
# group_labels, so they live in the same file. supervised.py imports them as
# `from libs.prototypes import label_all_groups, label_groups_by_target`.
#
# Two parts:
# 1) label_groups_by_target()  the main content of explanation layer (1):
#    which target this group corresponds to. Layers (2) and (3) cannot say it.
# 2) label_all_groups()        the group means of the features that most
#    characterise it, as raw values rather than qualitative bands.
#
# Ranking criterion: cross-group distinctiveness
# ──────────────────────────────────────────────
# The goal is to show what distinguishes each centroid *from the others*.
# Ranking by "how extreme is this group against the whole dataset" made
# features that are extreme in many groups win every time, so different
# centroids ended up described by the same feature. Ranking instead by "how
# far this group's value sits from the distribution of the other groups'
# values" (a robust z-score against the other groups) fixes that.
#
# Numeric features use the group median, which does not swing on an outlier
# when the group is small (often 1-10 samples). Categorical features show the
# most frequent category and its share. No qualitative bands ("very high",
# "moderate"): the raw value is already interpretable, and bands would need
# their own justification for where the boundaries fall.

@dataclass
class FeatureLabel:
    feature_idx:  int
    feature_name: str
    kind:         str    # "numeric" | "categorical"
    label:        str    # the group's actual value, e.g. "10.4" or "Category 2 (65%)"
    detail: dict          # raw values, for verification and debugging


def _group_stats_numeric(
    X_train: np.ndarray,                    # (N, F)
    valid_groups: Sequence[int],
    sample_groups: Sequence[Sequence[int]],
    feature_idx: int,
) -> Dict[int, float]:
    """Per-group median of the raw values: {group_idx: group_median}."""
    col = X_train[:, feature_idx]
    return {p: float(np.median(col[sample_groups[p]])) for p in valid_groups}


def _group_stats_categorical(
    X_train: np.ndarray,
    valid_groups: Sequence[int],
    sample_groups: Sequence[Sequence[int]],
    feature_idx: int,
    eps: float = 1e-6,
) -> Dict[int, dict]:
    """Per-group modal category with its share and lift: {group_idx: {...}}."""
    col = np.rint(X_train[:, feature_idx]).astype(int)
    out = {}
    for p in valid_groups:
        group_vals = col[sample_groups[p]]
        if len(group_vals) == 0:
            continue
        values, counts = np.unique(group_vals, return_counts=True)
        top_cat = int(values[np.argmax(counts)])
        group_prop   = float((group_vals == top_cat).mean())
        overall_prop = float((col == top_cat).mean())
        out[p] = {
            "top_category": top_cat,
            "group_prop":   group_prop,
            "lift":         group_prop / (overall_prop + eps),
        }
    return out


def _cross_group_distinctiveness(this_value: float, other_values: Sequence[float]) -> Optional[float]:
    """
    Robust z-score of `this_value` against `other_values`, the same feature's
    values in the other groups.

    Uses median and MAD (median absolute deviation) rather than mean and std,
    so a few extreme groups do not dominate. With fewer than two other groups
    to compare against (very small P) it cannot be computed and returns None;
    the caller falls back.
    """
    if len(other_values) < 2:
        return None
    others = np.asarray(other_values, dtype=float)
    med = np.median(others)
    mad = np.median(np.abs(others - med)) * 1.4826 + 1e-6  # scaled to match std under normality
    return float(abs(this_value - med) / mad)


def inverse_transform_numeric(qt, num_cols: Sequence[int], feature_idx: int, value: float) -> Optional[float]:
    """
    Numeric features are stored after prep_data() has pushed them through a
    QuantileTransformer into [0,1], which is why "0.328" carried no unit --
    there was no way to say how many marks a credit_amount of 0.328 is. Given
    the fitted transformer, this maps a value back to its original unit.

    QuantileTransformer treats each column independently (it learns a separate
    quantile mapping per column at fit time), so whatever fills the other
    columns cannot affect the inverse transform at feature_idx. Passing a
    dummy row with only this value substituted is therefore safe (verified).

    Returns None when qt is None or feature_idx is not a numeric column; the
    caller then displays the [0,1] value as-is.
    """
    if qt is None:
        return None
    try:
        col_pos = list(num_cols).index(feature_idx)
    except ValueError:
        return None
    dummy = np.full((1, len(num_cols)), 0.5)
    dummy[0, col_pos] = value
    try:
        return float(qt.inverse_transform(dummy)[0, col_pos])
    except Exception:
        return None


def label_all_groups(
    X_train: np.ndarray,
    sample_groups: Sequence[Sequence[int]],
    cat_cols: Sequence[int],
    num_cols: Sequence[int],
    col_names: Sequence[str],
    top_k: int = 5,
    min_group_size: int = 2,
    cat_category_names: Optional[Dict[str, Sequence[str]]] = None,
    quantile_transformer=None,
) -> Dict[int, List[FeatureLabel]]:
    """
    Called right after regroup_update() and cached.

    Returns {group_index: [FeatureLabel, ...]}: the top_k features, sorted by
    cross-group distinctiveness, so features unusual *for this group* come
    first.

    cat_category_names: {col_name: [original category strings, ...]}, as
    returned by load_data(). When given, categorical labels show the real name
    ("male single") instead of "Category 0". Without it they fall back to
    "Category N".

    quantile_transformer: the fitted QuantileTransformer returned by
    prep_data(). When given, numeric labels are mapped back to their original
    unit (credit_amount = 3271) instead of the [0,1] value.
    """
    valid_groups = [p for p, g in enumerate(sample_groups)
                     if g is not None and len(g) >= min_group_size]
    if not valid_groups:
        return {p: [] for p in range(len(sample_groups))}

    num_stats: Dict[int, Dict[int, float]] = {
        fi: _group_stats_numeric(X_train, valid_groups, sample_groups, fi)
        for fi in num_cols
    }
    cat_stats: Dict[int, Dict[int, dict]] = {
        fi: _group_stats_categorical(X_train, valid_groups, sample_groups, fi)
        for fi in cat_cols
    }

    result: Dict[int, List[FeatureLabel]] = {p: [] for p in range(len(sample_groups))}

    for p in valid_groups:
        candidates: List[FeatureLabel] = []

        for fi in num_cols:
            stats = num_stats[fi]
            if p not in stats:
                continue
            this_val = stats[p]
            others   = [v for q, v in stats.items() if q != p]
            dist = _cross_group_distinctiveness(this_val, others)
            if dist is None:
                dist = abs(this_val - float(np.median(list(stats.values()))))  # fallback

            real_val = inverse_transform_numeric(quantile_transformer, num_cols, fi, this_val)
            display_val = real_val if real_val is not None else this_val

            candidates.append(FeatureLabel(
                feature_idx=fi,
                feature_name=col_names[fi] if fi < len(col_names) else f"f{fi}",
                kind="numeric",
                label=f"{display_val:.3g}",
                detail={"group_value_uniform": this_val, "group_value_real": real_val,
                        "distinctiveness": dist},
            ))

        for fi in cat_cols:
            stats = cat_stats[fi]
            if p not in stats:
                continue
            top_cat, group_prop, lift = (stats[p]["top_category"], stats[p]["group_prop"], stats[p]["lift"])
            this_log = float(np.log2(lift + 1e-6))
            others_log = [float(np.log2(s["lift"] + 1e-6)) for q, s in stats.items() if q != p]
            dist = _cross_group_distinctiveness(this_log, others_log)
            if dist is None:
                dist = abs(this_log)  # fallback: distance from lift = 1 (log = 0)

            fname = col_names[fi] if fi < len(col_names) else f"f{fi}"
            names_for_col = cat_category_names.get(fname) if cat_category_names else None
            cat_display = (str(names_for_col[top_cat])
                            if names_for_col is not None and top_cat < len(names_for_col)
                            else f"Category {top_cat}")

            candidates.append(FeatureLabel(
                feature_idx=fi,
                feature_name=fname,
                kind="categorical",
                label=f"{cat_display} ({group_prop:.0%})",
                detail={"top_category": top_cat, "group_prop": group_prop, "lift": lift, "distinctiveness": dist},
            ))

        candidates.sort(key=lambda fl: fl.detail["distinctiveness"], reverse=True)
        result[p] = candidates[:top_k]

    return result




def label_groups_by_target(
    labels: np.ndarray,                      # (N,) MemoryBank labels: class index as float, or regression target
    sample_groups: Sequence[Sequence[int]],
    tasktype: str,                            # "multiclass" | "binclass" | "regression"
    class_names: Optional[Sequence[str]] = None,
    min_group_size: int = 2,
    second_class_threshold: float = 0.2,      # show the runner-up class when it reaches this share
) -> Dict[int, Optional[dict]]:
    """
    Summarise which target each group corresponds to -- the main content of
    explanation layer (1).

    - classification: the most frequent class and its share. The runner-up is
      returned as well once it reaches second_class_threshold, so a group
      straddling two classes is not presented as if it were pure.
    - regression: the percentile of the group's mean target within the overall
      distribution.

    Returns {group_idx: {...} or None}; None when the group is too small.
    """
    labels = np.asarray(labels)
    result: Dict[int, Optional[dict]] = {}

    for p, grp in enumerate(sample_groups):
        if grp is None or len(grp) < min_group_size:
            result[p] = None
            continue
        y_grp = labels[grp]

        if tasktype in ("multiclass", "binclass"):
            y_int = np.rint(y_grp).astype(int)
            vals, counts = np.unique(y_int, return_counts=True)
            order = np.argsort(-counts)
            top_cls   = int(vals[order[0]])
            top_count = int(counts[order[0]])
            top_prop  = float(top_count / len(y_int))
            top_name  = (class_names[top_cls]
                         if class_names is not None and top_cls < len(class_names)
                         else f"Class {top_cls}")

            second = None
            if len(order) > 1:
                second_cls   = int(vals[order[1]])
                second_count = int(counts[order[1]])
                second_prop  = float(second_count / len(y_int))
                if second_prop >= second_class_threshold:
                    second_name = (class_names[second_cls]
                                   if class_names is not None and second_cls < len(class_names)
                                   else f"Class {second_cls}")
                    second = {"class": second_cls, "name": second_name,
                              "prop": second_prop, "count": second_count}

            result[p] = {
                "kind": "classification",
                "top_class": top_cls, "top_class_name": top_name,
                "top_prop": top_prop, "top_count": top_count,
                "second": second, "n": len(y_int),
            }
        else:  # regression
            grp_mean   = float(np.mean(y_grp))
            percentile = float((labels <= grp_mean).mean()) * 100.0
            result[p] = {
                "kind": "regression",
                "group_mean": grp_mean, "percentile": percentile, "n": len(y_grp),
            }

    return result




class CentroidLayer(nn.Module):
    """
    Prototype partition over the learned representation.

    Parameters
    ──────────
    n_prototypes      : number of prototypes P
    embed_dim         : embedding dimension D
    n_features        : number of raw features F
    prototype_labels  : human-readable prototype names; "Centroid_i" if absent
    regroup_warmup_epochs : regroup_update starts publishing groups after this
                          epoch (0 = immediately)
    dead_reinit_patience : after this many consecutive regroup_update rounds
                          without a single assignment, the prototype is
                          reinitialised from an observed embedding (a random
                          sample plus small Gaussian noise), in the style of
                          Jukebox / SoundStream dead-code reset. 0 disables it.
    dropout           : dropout on the context vector
    col_names         : raw feature column names, for explanation output
    """

    def __init__(
        self,
        n_prototypes: int,
        embed_dim: int,
        n_features: int = 0,
        prototype_labels: Optional[List[str]] = None,
        regroup_warmup_epochs: int = 0,   # active immediately; no warmup
        freeze_centroid_after: "int | None" = None,   # see the note below
        dead_reinit_patience: int = 5,    # reinitialise after this many
        dead_reinit_noise_scale: float = 0.01,   # relative size of the Gaussian
                                            # noise added to the anchor vector on
                                            # reinitialisation
                                            # (noise_std = this * anchor.norm()).
                                            # The literature says only "small
                                            # Gaussian noise"; 0.01 is unverified.
        dropout: float = 0.0,
        col_names: Optional[List[str]] = None,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,   # van den Oord et al. (2017), Appendix, and
                                   # VQ-VAE-2(Razavi et al. 2019), Jukebox/SoundStream
                                   # the common default across EMA-based VQ
                                   # implementations. Not verified on this
                                   # project's data -- treat it like
                                   # dead_reinit_patience and noise_scale: a
                                   # literature default that is still a sweep
                                   # candidate.
        ema_eps: float = 1e-5,     # Laplace smoothing (VQ-VAE-2 Appendix A.1) —
                                   # keeps the denominator stable when the EMA
                                   # assignment count N_i approaches zero.
    ) -> None:
        super().__init__()
        self.P                 = n_prototypes
        self.D                 = embed_dim
        self.F                 = n_features
        self.regroup_warmup_epochs = regroup_warmup_epochs
        # ── Freeze centroid updates after this epoch ────────────────────
        # From this epoch on, centroid_emb must not move through any path:
        #     EMA           ema_update() returns early
        #     dead reinit   skipped inside regroup_update()
        # ⚠ Every path has to be blocked. Reinitialising a dead prototype is
        #   also an update, so leaving one open breaks the static condition.
        # ⚠ This is **not** freezing at initialisation
        #   (freeze_centroid_after=0). Freezing at 0 keeps the initial
        #   partition forever, which asks "is a random partition enough"
        #   rather than "is centroid updating needed". The question only holds
        #   when the frozen value is a partition that has already formed.
        # ⚠ The assignment step (reassignment inside regroup_update) keeps
        #   running: as the encoder moves, membership follows and only the
        #   centroids stand still. That is what isolates the update step.
        self.freeze_centroid_after = freeze_centroid_after
        self._centroid_frozen = False
        self.dead_reinit_patience  = dead_reinit_patience
        self.dead_reinit_noise_scale = dead_reinit_noise_scale
        self.col_names         = col_names or [f"f{i}" for i in range(n_features)]
        # ⚠ Cosine similarity, not a scaled logit. argmax(s*cos) == argmax(cos)
        #   for any s > 0, so a temperature cannot change which prototype is
        #   chosen -- only how peaked the straight-through gradient is, which
        #   the learning rate already covers.

        # ── Buffers: saved in the state_dict, no gradient ──
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))
        # For dead-prototype recovery: consecutive regroup_update rounds in
        # which this prototype received no assignment.
        self.register_buffer("dead_streak", torch.zeros(n_prototypes, dtype=torch.long))

        # ── Prototype memory. Not a learned parameter under EMA: see below ──
        self.centroid_emb = nn.Parameter(torch.empty(n_prototypes, embed_dim))
        nn.init.orthogonal_(self.centroid_emb)

        # ── EMA prototype memory ──────────────────────────────────
        # The standard arrangement (surveyed in Huh et al. 2023): the EMA
        # replaces the gradient-based codebook update that would pull a
        # centroid toward its assigned queries. One constraint is specific to
        # this project -- centroid_emb must stay a unit vector (CosFace
        # style) -- so the usual weighted-average update is followed by a
        # renormalisation, since the weighted average of two unit vectors is
        # not itself a unit vector.
        #
        # ⚠ Under EMA, centroid_emb is removed from the optimizer entirely
        #   (requires_grad = False). Any gradient-based push on the same
        #   tensor would be erased anyway: EMA overwrites centroid_emb.data
        #   wholesale every batch, so whatever optimizer.step() applied just
        #   before it disappears. When two mechanisms write one parameter --
        #   one accumulating a gradient, one overwriting -- the overwriting
        #   one wins. Centroid positions are therefore set purely by the
        #   running mean of the embeddings assigned to them.
        self.use_ema_codebook = use_ema_codebook
        self.ema_decay = ema_decay
        self.ema_eps   = ema_eps
        if use_ema_codebook:
            self.centroid_emb.requires_grad_(False)
            # N_i (the accumulated EMA assignment count) starts at 1. Starting
            # at 0 would make the denominator tiny right after the first batch
            # and send that first update flying.
            self.register_buffer("ema_cluster_size", torch.ones(n_prototypes))
            # m_i (the accumulated EMA sum of assigned embeddings) starts at
            # centroid_emb (a unit vector) * ema_cluster_size so that it agrees
            # with the initialisation immediately. Until the first
            # ema_update(), centroid_emb.data / ema_cluster_size therefore
            # equals centroid_emb.data and there is no jump.
            self.register_buffer(
                "ema_embed_sum",
                self.centroid_emb.data.clone() * self.ema_cluster_size.unsqueeze(-1)
            )

        # The IG-only baseline buffer was removed when explanation layer (3)
        # moved to an exact decomposition. n_features is kept in the signature
        # for compatibility.

        # ── Member sample indices per prototype; the retrieval pool G(a) ──
        # list of lists: sample_groups[p] = [idx, idx, ...]
        self.sample_groups: Optional[List[List[int]]] = None

        # ── Cached text labels per prototype ──
        # {p: [FeatureLabel, ...]}, filled by supervised.py right after each
        # regroup_update(). This cache backs the group description in
        # explanation layer (1).
        self.group_labels: Optional[Dict[int, list]] = None

        # ── Cached target distribution per prototype ──
        # The main content of explanation layer (1). Filled by supervised.py
        # via label_groups_by_target() right after each regroup_update().
        self.target_labels: Optional[Dict[int, Optional[dict]]] = None

        # ── Mean label per prototype ──
        self.register_buffer(
            'centroid_labels',
            torch.full((n_prototypes,), float('nan'))
        )

        # ── Labels ────────────────────────────────────────────
        self.labels = prototype_labels or [f"Centroid_{i}" for i in range(n_prototypes)]

        self.dropout = nn.Dropout(dropout)

        # ── State for the stability diagnostics ─────────────────
        # Observation variables used to check whether centroid drift -- the
        # failure that codebook losses target -- actually occurs here.
        # regroup_update() refreshes them every epoch and the result travels
        # through its return dict into supervised.regroup_history and meta.pkl.
        #
        # ⚠ Deliberately not register_buffer. Registering them would add keys
        #   to the state_dict and break --from_saved_state on existing
        #   checkpoints under load_state_dict(strict=True). Diagnostic state
        #   does not need saving. The only cost is that drift reads nan for
        #   the first epoch after resuming from a checkpoint.
        self._diag_prev_centroid = None    # (P, D)  centroids at end of last epoch
        self._diag_prev_assign   = None    # (N,)    assignments of last epoch
        self._diag_prev_active   = None    # (P,)    bool
        self._diag_prev_sizes    = None    # (P,)    long
        self._diag_reinit_mask   = None    # (P,)    reinitialised last epoch

        # ── History for evaluating dead-prototype recovery itself ──
        # Recovery is not a device that raises retrieval quality directly; it
        # makes dead centroids used again.
        # ⚠ The criterion is not "does it survive long" (the VQ view) but
        #   whether it goes on to receive assignments and form a usable
        #   cluster -- the centroid's purpose here is to partition retrieval.
        self._diag_reinit_total  = None    # (P,) long  cumulative reinit count
        self._diag_since_reinit  = None    # (P,) long  epochs since last reinit
                                           #            (-1 = never reinitialised)

    # ─────────────────────────────────────────────────────────
    # Initialisation: seed the centroids from training data
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize_from_data(
        self,
        X_emb: torch.Tensor,             # (N, D) training embeddings
        X_raw: Optional[torch.Tensor] = None,    # unused; kept for caller compatibility
        y_labels: Optional[torch.Tensor] = None, # unused; kept for caller compatibility
    ) -> None:
        """Seed the prototype memory from observed training embeddings.

        P rows are drawn uniformly from the embeddings the encoder produces
        before the first epoch. This is the rule dead-prototype recovery
        already uses, so the memory is created and repaired the same way.


        ⚠ No label is read. Initialisation, assignment and the EMA update are
          all class-agnostic.

        ⚠ This used to be k-means++. Across 10 datasets and 5 seeds the
          difference was accuracy -0.0017 (p = 0.670), AUROC -0.0040
          (p = 0.478) and active ratio -0.9% (p = 0.508) -- no case for
          keeping a geometry-specific initialisation procedure.
        """
        N   = X_emb.shape[0]
        dev = X_emb.device
        # Uniform sample of P observed embeddings, without replacement.
        idx = torch.randperm(N, device=dev)[: self.P]
        if idx.numel() < self.P:
            # Only reachable when P is set directly: P = floor(sqrt(N))
            # gives P <= N for every N. Falls back to replacement, which
            # duplicates prototypes -- dead-prototype recovery separates
            # them again once training starts.
            pad = torch.randint(0, N, (self.P - idx.numel(),), device=dev)
            idx = torch.cat([idx, pad])
        self.centroid_emb.data.copy_(F.normalize(X_emb[idx].float(), dim=-1))
        if hasattr(self, 'ema_cluster_size'):
            self.ema_cluster_size.fill_(1.0)
            self.ema_embed_sum.data = (
                self.centroid_emb.data.clone()
                * self.ema_cluster_size.unsqueeze(-1))
        print(f"  [CentroidLayer] {self.P} prototypes sampled from "
              f"{N} training embeddings")

    def ema_update(self, query_emb: torch.Tensor, hard_assignment: torch.Tensor) -> None:
        """
        The standard EMA form from van den Oord et al. (2017), Appendix, and
        VQ-VAE-2 (Razavi et al. 2019), Appendix A.1:

            N_i        <- decay * N_i + (1 - decay) * n_i
            m_i        <- decay * m_i + (1 - decay) * sum(query_emb assigned
                                                          to centroid i in
                                                          this batch)
            centroid_i <- m_i / N_i

        with two additions for this project's unit-sphere constraint:
          (a) Laplace smoothing, so a centroid whose N_i approaches zero does
              not get an unstable denominator (exactly as in VQ-VAE-2
              Appendix A.1);
          (b) renormalisation to a unit vector after every update, because the
              weighted average of two unit vectors is not a unit vector. This
              keeps the same invariant on the EMA path that the CosFace-style
              reprojection keeps after every optimizer step.

        query_emb is assumed to arrive detached (the caller's responsibility).
        This function is decorated with @torch.no_grad() so no gradient would
        form regardless, but passing an explicitly detached tensor makes the
        caller's intent clear.
        """
        # When centroids are frozen, the EMA path must stop as well.
        if getattr(self, "_centroid_frozen", False):
            return
        if not self.use_ema_codebook:
            return
        P = self.P
        one_hot     = F.one_hot(hard_assignment, num_classes=P).to(query_emb.dtype)  # (B, P)
        batch_count = one_hot.sum(dim=0)                                              # (P,)
        batch_sum   = one_hot.T @ query_emb                                           # (P, D)

        d = self.ema_decay
        self.ema_cluster_size.mul_(d).add_(batch_count, alpha=1 - d)
        self.ema_embed_sum.mul_(d).add_(batch_sum, alpha=1 - d)

        # Laplace smoothing, exactly as in VQ-VAE-2 Appendix A.1
        n = self.ema_cluster_size.sum()
        smoothed_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (n + P * self.ema_eps) * n
        )
        new_centroid = self.ema_embed_sum / smoothed_size.unsqueeze(-1)
        self.centroid_emb.data.copy_(F.normalize(new_centroid, dim=-1))

    # ─────────────────────────────────────────────────────────
    # (3) Refresh sample_groups (regroup_update, called at end of epoch)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def maybe_freeze_centroid(self, epoch: int) -> bool:
        """Stop all centroid movement once freeze_centroid_after is reached."""
        if self.freeze_centroid_after is None or self._centroid_frozen:
            return self._centroid_frozen
        if int(epoch) < int(self.freeze_centroid_after):
            return False
        self.centroid_emb.requires_grad_(False)
        self._centroid_frozen = True
        return True

    def regroup_update(
        self,
        X_emb: torch.Tensor,        # (N, D) all training embeddings, for assignment
        X_raw: Optional[torch.Tensor] = None,   # (N, F) raw features; unused, kept for caller compatibility
        assignments: Optional[torch.Tensor] = None,  # (N,) hard assignment
    ) -> Dict[str, float]:
        """
        Called at the end of an epoch to refresh sample_groups.

        What it does
        ────────────
        - refresh sample_groups, which bounds the retrieval search to G(a)
        - detect dead prototypes, so a collapse can stop the run early

        The group text labels for explanation layer (1) are computed and
        cached separately by the caller in supervised.py, immediately after
        this function refreshes sample_groups.

        Dead-prototype recovery
        ───────────────────────
        A centroid that received no assignment for dead_reinit_patience
        consecutive rounds is reinitialised from an observed embedding plus
        small noise, in the style of Jukebox (Dhariwal et al. 2020) and
        SoundStream (Zeghidour et al. 2021). Initialisation only guarantees a
        starting point; this keeps rescuing centroids that die later as the
        encoder drifts.

        Returns
        ───────
        stats: {"active_ratio": float, "min_cluster_size": int,
                "max_cluster_size": int, "reinit_count": int}
        """
        epoch = self.current_epoch.item()
        # ⚠ current_epoch is incremented *before* the warmup check, not at the
        # end of the function. It used to be incremented only at the bottom, a
        # point reachable only after passing warmup -- a circular dependency:
        # the counter had to grow to clear warmup, and warmup had to clear for
        # the counter to grow. With regroup_warmup_epochs > 0 every call
        # returned early, so warmup never ended (measured on vehicle: warmup=5
        # and warmup=10 both left active=0% and every sample_group empty for
        # the whole run). Incrementing first means the counter advances even
        # on an early return, so the warmup period actually finishes.
        self.current_epoch += 1
        # Freeze the centroids once the configured epoch is reached.
        # ⚠ Called here because regroup_update runs exactly once per epoch and
        #   is the only place that manages current_epoch. Adding a second call
        #   in the training loop would put two counters out of step.
        self.maybe_freeze_centroid(epoch)
        in_warmup = epoch < self.regroup_warmup_epochs

        if assignments is None:
            # Reassign against the current centroids, recomputed every call
            # including during warmup: the dead-prototype check below asks
            # whether a centroid received any assignment *this* epoch, and
            # that must hold regardless of warmup.
            q = F.normalize(X_emb.float(), dim=-1)
            c = F.normalize(self.centroid_emb, dim=-1)
            assignments = (q @ c.T).argmax(dim=-1)

        P = self.P
        assignments_cpu = assignments.cpu()
        new_groups: List[List[int]] = [[] for _ in range(P)]
        sizes = [0] * P
        for p in range(P):
            mask_cpu = (assignments_cpu == p).nonzero(as_tuple=True)[0]
            new_groups[p] = mask_cpu.tolist()
            sizes[p] = len(new_groups[p])

        # ── Stability diagnostics: snapshot only ────────────────────
        # Capture centroid_emb after this epoch's training and before any
        # reinitialisation. The metrics themselves are computed at the end of
        # the function, after reinit and after the sizes are recomputed: a
        # reinit rebuilds sample_groups and the sizes entirely, so a
        # size-based metric taken here would describe a different moment than
        # the active_ratio that gets returned.
        _diag_cur_centroid = self.centroid_emb.detach().clone()
        _reinit_jumps: List[float] = []
        _reinit_mask_now = torch.zeros(P, dtype=torch.bool,
                                       device=self.centroid_emb.device)
        _dev = self.centroid_emb.device
        if self._diag_reinit_total is None or self._diag_reinit_total.numel() != P:
            self._diag_reinit_total = torch.zeros(P, dtype=torch.long, device=_dev)
            self._diag_since_reinit = torch.full((P,), -1, dtype=torch.long, device=_dev)

        # Only the publication of sample_groups (the cache that actually
        # bounds retrieval) is deferred during warmup. Previously the whole
        # function returned early, which also disabled the dead-prototype
        # reinitialisation below. Reinit is a safeguard that should keep
        # running -- the opposite of what warmup is for. Measured on vehicle
        # with rwe20, that coupling made reinits pile up and fire all at once
        # (20 of 26) the moment warmup ended.
        if not in_warmup:
            self.sample_groups = new_groups

        # ── Dead-prototype recovery (Jukebox / SoundStream style) ───
        # Seeding at initialisation only helps at the starting line. As the
        # encoder drifts during training, a centroid that was alive can lose
        # its assignments and die -- the NSVQ paper shows collapse recurring
        # even from a perfect initialisation. This is a safeguard needed
        # throughout training, independently of how the centroids started.
        #
        # A centroid that received no assignment for dead_reinit_patience
        # consecutive rounds is moved to a randomly chosen embedding from this
        # epoch plus small Gaussian noise (the literature's phrasing:
        # "randomly sampled encoder outputs plus small Gaussian noise").
        # centroid_emb is always kept on the unit sphere, so the
        # reinitialised value is normalised too and the invariant holds.
        #
        # ⚠ This runs regardless of warmup. dead_streak has to accumulate from
        #   the start of training, otherwise centroids that begin dying during
        #   warmup go unnoticed until it ends and are then all relocated at
        #   once -- concentrating the instability rather than avoiding it
        #   (measured).
        n_reinit = 0
        if self.dead_reinit_patience > 0:
            with torch.no_grad():
                for p in range(P):
                    if sizes[p] == 0:
                        self.dead_streak[p] += 1
                    else:
                        self.dead_streak[p] = 0

                    if self.dead_streak[p].item() >= self.dead_reinit_patience:
                        src_idx = torch.randint(0, X_emb.shape[0], (1,)).item()
                        anchor  = X_emb[src_idx].float()
                        noise   = torch.randn_like(anchor) * self.dead_reinit_noise_scale * anchor.norm().clamp(min=1e-6)
                        new_vec = F.normalize(anchor + noise, dim=-1)
                        # Record the reinit jump separately from drift, so
                        # the signal the update rule is judged on is not
                        # swamped by relocation distance.
                        _reinit_jumps.append(float(torch.norm(
                            new_vec.to(self.centroid_emb.dtype)
                            - self.centroid_emb.data[p]
                        )))
                        _reinit_mask_now[p] = True
                        # Reinit is an update too, so it is skipped when
                        # centroids are frozen.
                        if getattr(self, "_centroid_frozen", False):
                            continue
                        self.centroid_emb.data[p] = new_vec.to(self.centroid_emb.dtype)
                        self.dead_streak[p] = 0
                        n_reinit += 1
                        # Under EMA, this centroid's accumulated statistics
                        # are reset as well. Otherwise the stale N_i and m_i
                        # built up while it was dead (both near zero) would
                        # pull the freshly placed centroid back toward its old
                        # position on the next ema_update().
                        if self.use_ema_codebook:
                            self.ema_cluster_size[p] = 1.0
                            self.ema_embed_sum.data[p] = new_vec.to(self.ema_embed_sum.dtype)

        # ⚠ When dead-prototype recovery relocates several centroids at once
        # (measured on jasmine: up to 11 in a single regroup_update, over 20%
        # of P=48), the new positions can intrude on the territory of
        # centroids that were *not* reinitialised. Routing is decided by
        # relative distance, so moving one centroid also shifts the Voronoi
        # boundaries of its neighbours.
        #
        # But sample_groups was already computed against the centroid_emb from
        # *before* the reinit, while what gets saved (best_state in
        # supervised.py) is the centroid_emb from *after* it -- two snapshots
        # of different moments. This was the root cause of the reassignment
        # agreement rate dropping to chance level or below. The reinitialised
        # centroid's own group was empty anyway; the damage was to the *other*
        # centroids' groups going stale.
        #
        # So if any reinit happened, assignments are recomputed against the
        # final centroid_emb and sample_groups / sizes are overwritten, which
        # keeps the two consistent. The cost is one more argmax over the
        # X_emb already in hand.
        #
        # Like the publication of sample_groups itself, this recomputation is
        # skipped during warmup: nothing was published this round, so there is
        # nothing to overwrite.
        if n_reinit > 0 and not in_warmup:
            with torch.no_grad():
                q_final = F.normalize(X_emb.float(), dim=-1)
                c_final = F.normalize(self.centroid_emb, dim=-1)
                assignments_final = (q_final @ c_final.T).argmax(dim=-1).cpu()
            new_groups = [[] for _ in range(P)]
            sizes = [0] * P
            for p in range(P):
                mask_cpu = (assignments_final == p).nonzero(as_tuple=True)[0]
                new_groups[p] = mask_cpu.tolist()
                sizes[p] = len(new_groups[p])
            self.sample_groups = new_groups

        # ── Stability metrics (after reinit and after sizes are rebuilt) ──
        # Drift is defined as "does it converge", not "how far did it move".
        #   healthy  dC: 0.8 -> 0.5 -> 0.3 -> 0.15 -> 0.07   (converging)
        #   drifting dC: 0.7 -> 0.6 -> 0.8 -> 0.5  -> 0.7    (never settles)
        # It is recorded as a per-epoch trajectory rather than one scalar, and
        # read as a trend. Centroids moving is not itself a failure -- early in
        # training they should. The failure is not settling.
        #
        # A reinit jump is not drift. Dead-prototype recovery throws a centroid
        # far by construction (measured: 2.7 per epoch on average), so mixing
        # the two buries the signal. Hence:
        #   _diag_prev_centroid  holds the value *after* reinit (end of epoch)
        #   _diag_cur_centroid   holds the value after training, *before* reinit
        # With that split, a reinitialised centroid's movement in the next
        # epoch reads as settling into its new place, and the jump does not
        # contaminate the measurement.
        #
        # centroid_emb is always kept on the unit sphere, so L2 distance is
        # monotone in cosine distance.
        _diag: Dict[str, float] = {}
        _sizes_t = torch.tensor(sizes, device=self.centroid_emb.device,
                                dtype=torch.long)
        _active = _sizes_t > 0

        if self._diag_prev_centroid is not None:
            _step = torch.norm(_diag_cur_centroid - self._diag_prev_centroid, dim=-1)
            _rm = self._diag_reinit_mask
            if _rm is None or _rm.numel() != P:
                _rm = torch.zeros_like(_active)
            _sel = _active & (~_rm)            # alive and not just reinitialised
            if bool(_sel.any()):
                _diag["drift_mean"] = float(_step[_sel].mean())
                _diag["drift_max"]  = float(_step[_sel].max())
            if bool(_rm.any()):
                # Movement of centroids reinitialised last epoch as they
                # settle; reported separately.
                _diag["drift_settle_mean"] = float(_step[_rm].mean())
        else:
            # First epoch, or right after resuming from a checkpoint (these
            # are plain attributes and are not restored).
            _diag["drift_mean"] = float("nan")

        # Assignment churn. A small centroid movement can already flip an
        # assignment, so this only means something read together with the
        # movement magnitude.
        #
        # ⚠ Do not read this value alone as "the centroids are unstable". The
        #   X_emb entering regroup are MemoryBank keys re-encoded every epoch,
        #   so a change in assignment has two possible causes:
        #     (a) the centroid moved, or (b) the embedding itself moved.
        #   assign_change_centroid_only below isolates (a) by assigning the
        #   same embeddings against the previous and the current centroids.
        #   Subtracting it from the total churn leaves roughly (b).
        _assign_final = (assignments_final if (n_reinit > 0 and not in_warmup)
                         else assignments_cpu)
        if (self._diag_prev_assign is not None
                and self._diag_prev_assign.numel() == _assign_final.numel()):
            _diag["assign_change_rate"] = float(
                (_assign_final != self._diag_prev_assign).float().mean()
            )
        if self._diag_prev_centroid is not None:
            with torch.no_grad():
                _q = F.normalize(X_emb.float(), dim=-1)
                _a_prev = (_q @ F.normalize(self._diag_prev_centroid.float(), dim=-1).T).argmax(-1)
                _a_cur  = (_q @ F.normalize(self.centroid_emb.detach().float(), dim=-1).T).argmax(-1)
                _diag["assign_change_centroid_only"] = float((_a_prev != _a_cur).float().mean())

        # active_delta: did the membership change even when the count did not?
        # (24 -> 24 with A and B dying while C and D appear is a different
        #  event, and active_centroids alone cannot tell them apart.)
        if self._diag_prev_active is not None and self._diag_prev_active.numel() == P:
            _diag["active_delta"] = int((_active ^ self._diag_prev_active).sum())
            _diag["active_died"]  = int((self._diag_prev_active & (~_active)).sum())
            _diag["active_born"]  = int(((~self._diag_prev_active) & _active).sum())

        # size shock: a large reshuffle among prototypes that stay alive
        # (20 -> 2, say). active_delta only catches deaths and births, so this
        # is the only place such an event shows up.
        if self._diag_prev_sizes is not None and self._diag_prev_sizes.numel() == P:
            _delta = (_sizes_t - self._diag_prev_sizes).abs()
            _base  = torch.clamp(self._diag_prev_sizes, min=1)
            _shock = (_delta >= 5) & (_delta.float() / _base.float() >= 0.5)
            _diag["size_shock_count"] = int(_shock.sum())
            _diag["size_shock_ids"]   = _shock.nonzero(as_tuple=True)[0].tolist()

        if _reinit_jumps:
            _diag["reinit_jump_mean"] = float(sum(_reinit_jumps) / len(_reinit_jumps))
            _diag["reinit_jump_max"]  = float(max(_reinit_jumps))

        # ── Metrics for dead-prototype recovery ────────────────────
        # ⚠ The criterion is not "does it survive long". That is the VQ-VAE
        #   view, where the codebook is the only information path and a dead
        #   code breaks the model. Here a centroid exists to produce a
        #   partition that retrieval can use, so the criterion follows that:
        #
        #     Case A  survives 20 epochs with cluster size 1
        #             -> long-lived and useless
        #     Case B  reinitialised every 3 epochs, each time representing a
        #             sparse region well
        #             -> possibly adaptation to a shifting distribution; fine
        #
        #   Jukebox and SoundStream aim for "become a used code", not "stay
        #   alive". So the primary metric is whether assignments arrive.
        #
        # Primary: do reinitialised centroids actually receive assignments?
        # Secondary: survival and repeat counts -- not criteria, but they
        # identify Case B.
        _prev_since = self._diag_since_reinit.clone()
        _ever = _prev_since >= 0
        self._diag_since_reinit = torch.where(_ever, _prev_since + 1, _prev_since)
        if bool(_reinit_mask_now.any()):
            self._diag_since_reinit[_reinit_mask_now] = 0
            self._diag_reinit_total[_reinit_mask_now] += 1

        _tot = self._diag_reinit_total
        if bool((_tot > 0).any()):
            _re = _tot > 0                      # ever reinitialised

            # Primary metric: does it receive assignments? This corresponds
            # directly to what recovery is for.
            _diag["reinit_assigned_rate"] = float((_re & _active).sum() / _re.sum())
            _diag["reinit_dead_now"]      = int((_re & (~_active)).sum())

            # Primary metric: is the resulting cluster large enough to be
            # useful for retrieval? k is not known here, so the size
            # distribution is emitted as-is and judged outside.
            _diag["reinit_size_mean"]   = float(_sizes_t[_re].float().mean())
            _diag["reinit_size_median"] = float(_sizes_t[_re].float().median())
            _diag["reinit_size_max"]    = int(_sizes_t[_re].max())
            # Reference: sizes of live centroids that were never
            # reinitialised.
            _never = (~_re) & _active
            if bool(_never.any()):
                _diag["never_reinit_size_mean"] = float(_sizes_t[_never].float().mean())

            # Secondary, not a criterion: identifies Case B (short-lived but
            # representative).
            _diag["reinit_repeat_rate"] = float((_tot >= 2).sum() / _re.sum())
            _diag["reinit_max_count"]   = int(_tot.max())
            _alive_re = _re & _active
            if bool(_alive_re.any()):
                _diag["reinit_age_mean"] = float(
                    self._diag_since_reinit[_alive_re].float().mean())

        # Update the state for the next epoch. The centroid stored is the
        # post-reinit (final) value.
        self._diag_prev_centroid = self.centroid_emb.detach().clone()
        self._diag_prev_assign   = _assign_final.clone()
        self._diag_prev_active   = _active.clone()
        self._diag_prev_sizes    = _sizes_t.clone()
        self._diag_reinit_mask   = _reinit_mask_now

        if in_warmup:
            # sample_groups was not published, but reinit_count reports what
            # actually happened during warmup, so the log shows how many
            # prototypes were already rescued before it ended.
            # The diagnostics are returned too: centroids move during warmup,
            # and leaving this interval empty would remove the first part of
            # any "does it converge" judgement.
            return {"active_ratio": 0.0, "min_cluster_size": 0,
                    "max_cluster_size": 0, "reinit_count": n_reinit, **_diag}

        # Statistics
        n_assigned = sum(1 for s in sizes if s > 0)

        return {
            "active_ratio":     n_assigned / self.P,
            "active_centroids": int(n_assigned),
            "pruned_this_epoch": 0,
            "reinit_count":     n_reinit,
            "min_cluster_size": int(min(s for s in sizes if s > 0)) if any(s > 0 for s in sizes) else 0,
            "max_cluster_size": int(max(s for s in sizes if s > 0)) if any(s > 0 for s in sizes) else 0,
            **_diag,
        }

    # ─────────────────────────────────────────────────────────
    # Temperature annealing hook, called once per epoch
    # ─────────────────────────────────────────────────────────

    def anneal(self, factor: Optional[float] = None) -> None:
        """
        No-op, kept so the training loop's call site stays valid.

        Temperature annealing became unnecessary when routing moved to a
        straight-through argmax: there is no temperature left to anneal. The
        method is retained because supervised.py calls model.anneal() once per
        epoch; removing it would only move the no-op to the caller.
        """
        pass

    # ─────────────────────────────────────────────────────────
    # Member indices of the assigned prototype group
    # ─────────────────────────────────────────────────────────

    def get_candidate_indices(
        self,
        hard_assignment: torch.Tensor,  # (B,)
        max_candidates: int = 5000,
    ) -> Optional[List[List[int]]]:
        """
        Return the member sample indices of each assigned prototype group.

        This is the same partition MemoryBank.retrieve() uses to bound the
        search to G(a) rather than the whole training split.

        ⚠ Nothing currently calls this method: retrieve() reads the cached
          groups directly. It is kept as the readable form of that lookup.

        Returns None if sample_groups has not been initialised yet.
        """
        if self.sample_groups is None:
            return None

        B = hard_assignment.shape[0]
        result = []
        for b in range(B):
            p = hard_assignment[b].item()
            grp = self.sample_groups[p]
            if len(grp) == 0:
                result.append(None)  # empty group: fall back to a full search
            else:
                result.append(grp[:max_candidates])
        return result

    # ─────────────────────────────────────────────────────────
    # Forward (Hierarchical-extended)
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        query_emb: torch.Tensor,                          # (B, D)
        top_m: int = 1,                                    # centroids to mix; always 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route a batch of query embeddings to prototypes.

        Parameters
        ──────────
        query_emb : (B, D) encoder output
        top_m     : number of centroids to mix. **The model always calls this
                    with the default 1**, which is the hard straight-through
                    routing the architecture is defined by. Values above 1
                    give a soft mixture over the top-M and exist only as an
                    exploratory path.

        Returns
        ───────
        context_emb      : (B, D)  the assigned centroid c (a weighted mix when
                                   top_m > 1)
        hard_assignment  : (B,)    the top-1 centroid. This single value fixes
                                   both the prediction baseline and the
                                   retrieval pool G(a)
        routing_probs    : (B, P)  full distribution; straight-through when
                                   top_m == 1
        topM_idx         : (B, M)  top-M centroid indices
        topM_weights     : (B, M)  top-M softmax weights (differentiable)
        top1_confidence  : (B,)    soft[hard_assignment].

            ⚠ routing_probs cannot serve as a per-sample confidence. Under the
              straight-through estimator its forward value is exactly one-hot
              by definition, so max() is exactly 1.0 for every sample. Anything
              that needs a confidence that actually varies per sample -- and
              that carries a gradient -- must use this value instead.

            Only the top-1 scalar is returned rather than the whole soft
            distribution: consumers use the one selected centroid, so this is
            sufficient and cheaper than carrying (B, P). The definition holds
            for top_m > 1 as well, since hard_assignment is still the top-1.

        With top_m = 1:
          - hard_assignment and routing_probs keep the straight-through
            behaviour
          - context_emb = centroid_emb[hard_assignment]
          - topM_idx    = hard_assignment.unsqueeze(1), shape (B, 1)
          - topM_weights = ones(B, 1)
        """
        # Cosine similarity logits
        q = F.normalize(query_emb, dim=-1)               # (B, D)
        c = F.normalize(self.centroid_emb, dim=-1)        # (P, D)
        # ⚠ No temperature. argmax(s*cos) == argmax(cos) for any s > 0, so a
        #   scale cannot change the assignment, the baseline c, or the logits.
        #   It only rescales the straight-through gradient -- an effect the
        #   learning rate already covers, and lr is searched. Measured on
        #   synthetic data, adding one changed nothing at convergence and was
        #   slightly worse early in training.
        logits = q @ c.T                                  # (B, P)

        # ── Full softmax; carries the straight-through backward gradient ─
        soft = F.softmax(logits, dim=-1)                  # (B, P)

        # ── Top-M selection ────────────────────────────
        top_m_eff = min(top_m, self.P)  # never exceed P
        topM_logits, topM_idx = logits.topk(top_m_eff, dim=-1)  # (B, M)

        # ── hard_assignment: top-1. Fixes both the prediction baseline
        #    and the retrieval pool G(a) ──
        hard_assignment = topM_idx[:, 0]                  # (B,)

        # ── top1_confidence: the soft probability at the selected position.
        #    Unlike routing_probs (one-hot in the forward value under the
        #    straight-through estimator) this actually varies per sample.
        #    Computed once here, independent of the top_m branch below.
        top1_confidence = soft.gather(1, hard_assignment.unsqueeze(1)).squeeze(1)  # (B,)

        # ── routing_probs ──────────────────────────────
        # top_m == 1: straight-through, so the backward gradient flows
        #             through the soft distribution
        # top_m > 1:  a plain softmax; the straight-through estimator has no
        #             meaning on the mixture path
        if top_m_eff == 1:
            # Straight-through: forward takes the hard argmax, backward
            # passes the soft gradient. Kept in eval mode as well.
            #
            # VQ-VAE (van den Oord et al. 2017) and Bengio et al. (2013)
            # define the estimator without a train/eval distinction:
            # forward = hard, backward = soft. Disabling it at eval time would
            # make d(context_emb)/d(query_emb) zero.
            #
            # That gradient path is no longer needed by anything, but keeping
            # it costs nothing: by value the forward result is identical to a
            # plain hard argmax, since soft + (hard - soft).detach() == hard.
            # Predictions are unaffected either way.
            hard_one_hot = F.one_hot(hard_assignment, self.P).float()  # (B, P)
            routing_probs = soft + (hard_one_hot - soft).detach()      # STE (always)

            # topM_weights = 1.0 for the single centroid
            topM_weights = torch.ones_like(topM_logits)   # (B, 1)

            # context_emb = routing_probs @ centroid_emb
            context_emb = self.dropout(routing_probs @ self.centroid_emb)
        else:
            # Mixture path.
            # routing_probs is returned as the soft distribution so callers
            # (diagnostics, explanations) can read the full spread, and the
            # gradient flows through it.
            routing_probs = soft

            # topM_weights is a softmax over the top-M and is differentiable,
            # so the gradient reaches the centroid selection.
            topM_weights = F.softmax(topM_logits, dim=-1) # (B, M)

            # context_emb is the weighted mixture of the top-M centroids.
            topM_centroids = self.centroid_emb[topM_idx]  # (B, M, D)
            context_emb = (
                topM_weights.unsqueeze(-1) * topM_centroids
            ).sum(dim=1)                                   # (B, D)
            context_emb = self.dropout(context_emb)

        return context_emb, hard_assignment, routing_probs, topM_idx, topM_weights, top1_confidence

    # ─────────────────────────────────────────────────────────
    # ⚠ entropy_loss was removed. It was defined but never connected to any
    # objective. For the record: it maximised the entropy of the *batch-mean*
    # routing distribution (the VQ-VAE-2 approach to codebook utilisation and
    # dead codes), which is a different goal from making each individual
    # sample's routing confident. That second goal would need a different
    # loss (per-sample entropy minimisation), and it is not being reinstated
    # without a separate case for it.




    # ─────────────────────────────────────────────────────────
    # Explanation helpers
    # ─────────────────────────────────────────────────────────

    def explain_routing(
        self,
        hard_assignment: torch.Tensor,   # (B,)
        routing_probs: torch.Tensor,     # (B, P)
        norm_mean: Optional[np.ndarray] = None,  # unused; kept for caller
        norm_std:  Optional[np.ndarray] = None,  # signature compatibility
        cos_sim: Optional[torch.Tensor] = None,  # (B, P) q_norm @ c_norm.T.
            # The quantity the assignment is actually made on. A softmax over
            # it compresses the range -- with P prototypes every value sits
            # near 1/P -- so the raw cosine and the cosine margin are reported
            # alongside. When None, cosine_similarity and cosine_margin are
            # None in the output dict.
    ) -> List[dict]:
        """
        Per-sample description of the prototype assignment.

        The main content of explanation layer (1) is target_labels, the output
        of label_groups_by_target(): which target this group mostly
        corresponds to. The runner-up groups carry their own target_info as
        well -- a runner-up is a group the sample nearly belonged to, so
        knowing its target is part of the context.

        group_feature_labels adds the group means of the features that most
        distinguish this group from the others, as raw values. Before the
        caches are filled (supervised.py not yet wired) these are None and an
        empty list.

        ⚠ The returned key is "routing_confidence", not "confidence". Used
          alone the word reads as "how sure the model is that the prediction
          is right", i.e. the classifier softmax. This is a different
          quantity: how strongly the query prefers the assigned centroid over
          the others, at the routing stage. The classifier downstream uses
          information beyond routing, so the two are independent.
          margin, others_mass and cosine_similarity come with it so that
          "is 55.6% high?" does not have to be judged from one absolute
          number.
        """
        pa   = hard_assignment.detach().cpu().numpy()
        pr   = routing_probs.detach().cpu().numpy()
        cs   = cos_sim.detach().cpu().numpy() if cos_sim is not None else None

        out  = []

        for b in range(pa.shape[0]):
            p     = int(pa[b])
            label = self.labels[p]
            conf  = float(pr[b, p])

            runner_idx = sorted(
                [i for i in range(self.P) if i != p],
                key=lambda i: -float(pr[b, i]),
            )[:2]
            runners = [
                {
                    "label":              self.labels[i],
                    "routing_confidence": float(pr[b, i]),
                    "target_info": (self.target_labels.get(i)
                                     if self.target_labels is not None else None),
                }
                for i in runner_idx
            ]

            # margin: the confidence gap between the assigned centroid and
            # the runner-up, so "55% vs 54%" reads differently from
            # "55% vs 10%".
            # others_mass: the probability mass held by every centroid outside
            # the assigned one and the runner-ups shown above, separating
            # "there are plenty of other candidates" from "these three are
            # essentially all of it".
            margin = conf - (runners[0]["routing_confidence"] if runners else 0.0)
            others_mass = 1.0 - conf - sum(r["routing_confidence"] for r in runners)
            cosine_similarity = float(cs[b, p]) if cs is not None else None

            # ⚠ cosine_margin is the quantity the assignment is actually made
            #   on: cos to the assigned centroid minus cos to the runner-up.
            #   The softmax-based margin above depends on a temperature, and
            #   the assignment does not -- argmax(s*cos) == argmax(cos) for any
            #   s > 0. So a softmax confidence describes the temperature as
            #   much as the geometry, while this describes only the geometry.
            cosine_margin = None
            if cs is not None and cs.shape[1] > 1:
                _row = cs[b].copy()
                _top = float(_row[p])
                _row[p] = -np.inf
                cosine_margin = _top - float(_row.max())

            # Main content of explanation layer (1): which target this group
            # corresponds to.
            target_info = (
                self.target_labels.get(p)
                if self.target_labels is not None else None
            )

            # Supporting information: the group means of its most
            # distinctive features.
            group_feature_labels = (
                self.group_labels.get(p, [])
                if self.group_labels is not None else []
            )

            out.append({
                "assigned_group":       label,
                "centroid_idx":         p,
                "routing_confidence":   conf,   # renamed from group_confidence
                "margin":               margin,
                "others_mass":          max(0.0, others_mass),  # clamp float error
                "cosine_similarity":    cosine_similarity,
                "cosine_margin":        cosine_margin,
                "runners_up":           runners,
                "target_info":          target_info,          # layer (1) main
                "group_feature_labels": group_feature_labels,  # supporting
            })
        return out

    def centroid_summary(self, top_n: int = 3) -> str:
        """
        Summarise every prototype: group size, target distribution, and the
        group means of its most distinctive features.

        top_n bounds how many features are shown per group.
        """
        lines = [f"CentroidLayer — {self.P} centroids", "─" * 44]

        for p in range(self.P):
            grp_size = (len(self.sample_groups[p])
                        if self.sample_groups else "?")
            line = f"  [{self.labels[p]}]  n={grp_size}"

            # Layer (1) content: target distribution
            tinfo = self.target_labels.get(p) if self.target_labels else None
            if tinfo is not None:
                if tinfo["kind"] == "classification":
                    line += f"  → {tinfo['top_class_name']} {tinfo['top_count']}/{tinfo['n']} ({tinfo['top_prop']:.0%})"
                else:
                    line += f"  → target≈{tinfo['group_mean']:.3g}(p{tinfo['percentile']:.0f})"

            # Group means of the most distinctive features
            labels_p = self.group_labels.get(p) if self.group_labels else None
            if labels_p:
                vals = ", ".join(f"{fl.feature_name}={fl.label}" for fl in labels_p[:top_n])
                line += f"  [{vals}]"
            lines.append(line)
        return "\n".join(lines)