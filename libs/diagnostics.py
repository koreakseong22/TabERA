"""
libs/diagnostics.py
===================
The observer layer.

Boundary
────────
```
tabera.py       predictor   tensor -> tensor. It does not *build* explanations.
diagnostics.py  observer    it *observes* the forward output and reconstructs them.
```

Every function here takes only `model` and the dict returned by `forward()`.
Nothing mutates the model, nothing builds a gradient, nothing affects training.

Why the split
─────────────
If explanation lived inside `forward()`:
  - explanation requirements would leak into the training/inference API,
  - benchmark and serving code would pay for work they never use,
  - the prediction graph and the diagnostic graph would share one function,
  - the model would have to know about external state such as FeatureStore.
So forward emits prediction state only, and all interpretation happens here.

⚠ What makes reconstruction possible (checked numerically before writing it)
```
neighbour embeddings  memory.keys[topk_idx]                  = keys_full[idx] in retrieve()
neighbour similarity  normalize(query_emb) . normalize(above) = the same expression retrieve() uses
neighbour labels      memory.labels[topk_idx]
logit_dev             logits - dev_head(context_emb)          = W.(beta*r), bias cancels
```
**Validity alone cannot be reconstructed**: a slot retrieval failed to fill has
`topk_idx = 0`, which is indistinguishable from "neighbour 0" outside the model.
That is why forward emits exactly one extra key, `out["neighbor_mask"]`.

⚠ This file does not judge
```
NO   if ambiguity_ratio > 1.15: ambiguous = True
YES  {"ambiguity_ratio": 1.02}          <- values only; judging belongs to analysis
```
And it does not truncate: the full ranking is returned. Top-N cuts are the
display layer's job. Truncating during computation blinds the analysis
scripts to the rest of the distribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


__all__ = [
    "retrieved_neighbors",
    "local_label_evidence",
    "prototype_deviation",
    "group_relative_feature_stats",
    "feature_gaps",
    "prototype_conditioning_overlap",
    # ⚠ These two were once missing here and reproduce.py raised
    #   AttributeError: the functions existed in the file but not in __all__.
    #   Update this list whenever a function is added.
    "prototype_class_alignment",
    "context_space_diversity",
]


# ─────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────

def _entropy(counts: Sequence[int], total: int) -> float:
    if total <= 0:
        return 0.0
    acc = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            acc -= p * float(np.log(p))
    return acc


def _cols(model, n_features: int) -> List[str]:
    return list(getattr(model, "column_names", None)
                or [f"f{i}" for i in range(n_features)])


def _cat_num_idx(model, n_features: int):
    cat = list(getattr(model.embedder, "cat_col_idx", []) or [])
    num = list(getattr(model.embedder, "num_col_idx", None)
               or [i for i in range(n_features) if i not in cat])
    return cat, num


# ─────────────────────────────────────────────────────────────
# (1) Reconstruct the retrieved neighbours
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieved_neighbors(
    model,
    out: Dict,
    feature_store=None,
) -> Optional[List[List[dict]]]:
    """
    Reconstruct the neighbour list from the forward output.

    Returns a list of dicts per sample, ordered by descending similarity
    (i.e. topk order): {rank, memory_idx, sample_id, similarity, label,
    features}.

    ⚠ `retrieve()` discards the similarity values, so they are recomputed
      here with **the same expression**: retrieve() also uses
      `q_norm @ self._keys_norm`, and the `nk` it returns is
      `keys_full[idx]`, the same tensor as `memory.keys[topk_idx]`.

    ⚠ Unfilled slots are not returned at all. Showing them as "a neighbour
      with similarity 0.000" would describe a case that does not exist.

    ⚠ `features` are the stored values as-is (numeric ones are [0,1]
      quantiles). Converting back to human-readable units is the display
      layer's job — neither the model nor this function knows the dataset's
      quantile_transformer.

    With feature_store=None, model.feature_store is used; if that is absent,
    `features` is None and only the retrieval itself is reconstructed.
    """
    topk = out.get("topk_idx")
    # ⚠ This used to read out["query_retr"], a retrieval-only representation.
    #   With retr_proj removed, retrieval runs on query_emb, so recomputing
    #   the similarity with the same expression requires query_emb here too.
    qr   = out.get("query_emb")
    if topk is None or qr is None:
        return None

    mask = out.get("neighbor_mask")
    fs   = feature_store if feature_store is not None else getattr(model, "feature_store", None)

    topk_c = topk.detach().cpu()
    qn = F.normalize(qr.detach(), dim=-1).unsqueeze(1)                  # (B,1,D)
    kn = F.normalize(model.memory.keys[topk].detach(), dim=-1)          # (B,k,D)
    sim = (qn * kn).sum(-1).cpu().numpy()                               # (B,k)

    lab = model.memory.labels[topk].detach().cpu().numpy()
    sid = model.memory.sample_ids[topk].detach().cpu().numpy()
    msk = mask.detach().cpu().numpy() if mask is not None else None
    feats = fs.retrieve(topk) if fs is not None else None               # [B][k] dict

    B, k = topk_c.shape
    result: List[List[dict]] = []
    for b in range(B):
        rows: List[dict] = []
        for j in range(k):
            if msk is not None and not bool(msk[b, j]):
                continue
            rows.append({
                "rank":       len(rows),
                "memory_idx": int(topk_c[b, j]),
                "sample_id":  int(sid[b, j]),
                "similarity": float(sim[b, j]),
                "label":      float(lab[b, j]),
                "features":   (feats[b][j] if feats is not None else None),
            })
        result.append(rows)
    return result


# ─────────────────────────────────────────────────────────────
# (2) Local label evidence (ambiguity evidence)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def local_label_evidence(model, out: Dict) -> Optional[List[dict]]:
    """
    Return the neighbours' label composition **together with the assigned
    prototype's full distribution**.

    ⚠ A neighbour majority must not be used as evidence for the prediction.
      Within one prototype, even raw features do not beat the majority
      baseline — measured. So "6 of 8 are A" is sampling noise around the
      group distribution.

    ⚠ It is still informative, for a different reason: after controlling for
      prototype purity, neighbour entropy still predicts misclassification
      significantly. The neighbours state **how mixed this region is**, not
      **what the answer is**.

    → Hence the local distribution is never emitted without the group
      distribution. Without both, "4/6" cannot even be called high or low
      (if the group is 82% A, then 4/6 = 67% is low).

    ⚠ No judgement is made. Only `ambiguity_ratio` is returned; where the
      line for "ambiguous region" falls is for the analysis scripts to set
      from the empirical distribution.

    ⚠ Scope: this retrieval is NN(q, G_p), not NN(q, D). The field is carried
      along so the caller cannot describe it wrongly.
    """
    topk = out.get("topk_idx")
    if topk is None:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    tasktype = getattr(model, "tasktype", None)
    n_mem    = int(model.memory.filled.item())
    mem_lab  = model.memory.labels[:n_mem].detach().cpu().numpy()
    nb_lab   = model.memory.labels[topk].detach().cpu().numpy()
    mask     = out.get("neighbor_mask")
    msk      = mask.detach().cpu().numpy() if mask is not None else None
    ha_np    = ha.detach().cpu().numpy()
    sg       = getattr(model.prototype_layer, "sample_groups", None)

    grp_cache: Dict[int, np.ndarray] = {}
    result: List[dict] = []
    for b in range(nb_lab.shape[0]):
        sel = [j for j in range(nb_lab.shape[1])
               if msk is None or bool(msk[b, j])]
        labs = nb_lab[b, sel]
        p = int(ha_np[b])
        if p not in grp_cache:
            ids = (sg[p] if (sg is not None and p < len(sg) and sg[p] is not None) else [])
            ids = [i for i in ids if 0 <= i < n_mem]
            grp_cache[p] = mem_lab[ids] if ids else mem_lab[:0]
        glab = grp_cache[p]

        d: Dict[str, object] = {
            "scope":       "prototype_conditioned",   # NN(q, G_p)
            "prototype":   p,
            "n_neighbors": int(len(labs)),
            "group_size":  int(glab.shape[0]),
        }
        if tasktype == "regression":
            # Regression has no "label composition"; compare local spread
            # against group spread. It answers the same question -- how much
            # does this region vary.
            d["local_mean"] = float(labs.mean()) if len(labs) else float("nan")
            d["local_std"]  = float(labs.std())  if len(labs) else float("nan")
            d["group_mean"] = float(glab.mean()) if glab.shape[0] else float("nan")
            d["group_std"]  = float(glab.std())  if glab.shape[0] else float("nan")
            d["dispersion_ratio"] = (
                float(d["local_std"] / d["group_std"])
                if (d["group_std"] == d["group_std"] and d["group_std"] > 1e-9)
                else float("nan"))
        else:
            lc: Dict[int, int] = {}
            for v in labs:
                kk = int(round(float(v)))
                lc[kk] = lc.get(kk, 0) + 1
            gc: Dict[int, int] = {}
            for v in glab:
                kk = int(round(float(v)))
                gc[kk] = gc.get(kk, 0) + 1
            d["label_counts"]       = lc
            d["group_label_counts"] = gc
            # ⚠ The key name carries its source; a bare `entropy` is avoided.
            #   The `entropy` in the npz export is that of the evidence_w
            #   (attention weight) distribution -- a constant log(k) here --
            #   and it really was misread as "neighbour uncertainty" from the
            #   name alone. Reusing the name with a second meaning would only
            #   compound that. This is the entropy of the retrieved
            #   neighbours' **label** distribution, H(Y_N(x)).
            d["label_entropy"]       = _entropy(lc.values(), len(labs))
            d["group_label_entropy"] = _entropy(gc.values(), int(glab.shape[0]))
            d["ambiguity_ratio"] = (
                float(d["label_entropy"] / d["group_label_entropy"])
                if d["group_label_entropy"] > 1e-9 else float("nan"))
        result.append(d)
    return result


# ─────────────────────────────────────────────────────────────
# (3) Prototype-relative deviation (exact additive decomposition)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_deviation(model, out: Dict) -> Optional[List[dict]]:
    """
    Magnitude, direction and logit contribution of the correction term d in
    h = c + d.

    ⚠ This decomposition is an **identity, not an approximation**. dev_head
      is a single nn.Linear(embed_dim, n_output), so
          logits = W.(c + d) + b = (W.c + b) + W.d
      and the two terms always sum exactly to logits. The bias is absorbed
      into the first term and never mixes into W.d. This differs in kind from
      SHAP or IG, which pick a baseline and approximate — here the structure
      is already additive and is simply written out.

    ⚠ d is reconstructed with **literally the same expression** as forward.
      Any separate approximation would make the explanation disagree with the
      computation.
          d = sigma(beta_raw) * normalize(q - c)

    ⚠ `dim_contrib` indexes **embedding dimensions**, not features. The
      embedder is PLE/PLR + MLP + LayerNorm, so there is no path back from an
      embedding dimension to an input feature, and gradients cannot bridge it
      either (the graph breaks at the categorical branch). What this supports
      is a statement about concentration — how few dimensions carry the
      correction — not about feature names.

    ⚠ No truncation: all D dimensions are returned. Taking the top N is the
      display layer's job.
    """
    # ⚠ This used to branch on fusion_mode. The final model has one path, so
    #   the only thing worth checking is that dev_head is a single Linear —
    #   that is the sole condition making the decomposition an identity.
    dev_head = getattr(model, "dev_head", None)
    if not isinstance(dev_head, torch.nn.Linear):
        return None
    q = out.get("query_emb")
    c = out.get("context_emb")
    lg = out.get("logits")
    if q is None or c is None or lg is None:
        return None

    q = q.detach(); c = c.detach(); lg = lg.detach()
    beta = torch.sigmoid(model.dev_beta_raw.detach())
    # This must be **literally the same expression** as forward. An
    # approximation here would make the explanation disagree with the actual
    # computation, and the disagreement would surface only as a non-zero
    # residual (which the caller checks).
    d = beta * F.normalize(q - c, dim=-1)

    W        = dev_head.weight.detach()          # (O, D)
    lg_proto = dev_head(c)                       # (B, O) = W·c + b
    lg_dev   = lg - lg_proto                     # (B, O) = W·d

    if lg.shape[-1] > 1:
        pred_m  = lg.argmax(dim=-1)
        proto_m = lg_proto.argmax(dim=-1)
        changed = pred_m != proto_m
    else:
        # ⚠ Binary and regression have n_output=1, so argmax is always 0. For
        #   binary the sign decides the class, so a sign flip is the test.
        pred_m  = torch.zeros(lg.shape[0], dtype=torch.long, device=lg.device)
        proto_m = pred_m
        changed = (lg.squeeze(-1) > 0) != (lg_proto.squeeze(-1) > 0)

    # ── Probability shift ───────────────────────────────────────
    # ⚠ dev_share (= |ld| / (|lp| + |ld|)) is a ratio of logit magnitudes and
    #   overstates the effect. Measured on credit-g: dev_share reads
    #   5.6-19.3% while the actual confidence moved 0.2-1.2 percentage
    #   points, because the logits sit in +-0.6 where the sigmoid is nearly
    #   linear. People read probabilities, so probabilities are what we emit.
    #
    # ⚠ The prototype-only probability must be measured on the *same* class:
    #   showing what the finally predicted class scored at the prototype
    #   stage is what makes the shift readable.
    tasktype = getattr(model, "tasktype", None)
    if tasktype == "regression":
        prob_proto = prob_final = None
        proto_pred = None
    elif lg.shape[-1] == 1:
        _pf = torch.sigmoid(lg.squeeze(-1))
        _pp = torch.sigmoid(lg_proto.squeeze(-1))
        _cls = (_pf > 0.5)
        prob_final = torch.where(_cls, _pf, 1 - _pf)
        prob_proto = torch.where(_cls, _pp, 1 - _pp)   # measured on the final predicted class
        proto_pred = (_pp > 0.5).long()
    else:
        _ar0 = torch.arange(lg.shape[0], device=lg.device)
        prob_final = torch.softmax(lg, -1)[_ar0, pred_m]
        prob_proto = torch.softmax(lg_proto, -1)[_ar0, pred_m]
        proto_pred = proto_m

    contrib = W[pred_m] * d                      # (B, D); sums to lg_dev[:, m]
    d_np       = d.norm(dim=-1).cpu().numpy()
    lgp_np     = lg_proto.cpu().numpy()
    lgd_np     = lg_dev.cpu().numpy()
    m_np       = pred_m.cpu().numpy()
    chg_np     = changed.cpu().numpy()
    pp_np      = prob_proto.cpu().numpy() if prob_proto is not None else None
    pf_np      = prob_final.cpu().numpy() if prob_final is not None else None
    ppred_np   = proto_pred.cpu().numpy() if proto_pred is not None else None
    contrib_np = contrib.cpu().numpy()

    result = []
    for b in range(lg.shape[0]):
        m  = int(m_np[b])
        lp = float(lgp_np[b, m])
        ld = float(lgd_np[b, m])
        result.append({
            "dev_norm":       float(d_np[b]),      # ||d||; r is a unit vector, so this equals beta
            "logit_proto":    lp,                  # W·c + b
            "logit_dev":      ld,                  # W·d
            # Share of the predicted channel taken by the correction. Logits
            # are signed, so a plain ratio can diverge; dividing by
            # |lp| + |ld| keeps it bounded. Read it as a magnitude share only.
            "dev_share":      float(abs(ld) / max(abs(lp) + abs(ld), 1e-12)),
            "argmax_changed": bool(chg_np[b]),
            "pred_channel":   m,
            # Probability of the finally predicted class at the prototype
            # stage, then after the correction. None for regression.
            "prob_proto":     (float(pp_np[b]) if pp_np is not None else None),
            "prob_final":     (float(pf_np[b]) if pf_np is not None else None),
            # Which class the prototype term alone would have predicted.
            "proto_pred":     (int(ppred_np[b]) if ppred_np is not None else None),
            "dim_contrib":    contrib_np[b].tolist(),   # all D, untruncated
            "n_dims":         int(contrib_np.shape[1]),
        })
    return result


# ─────────────────────────────────────────────────────────────
# (4) Group-relative feature statistics (feature space, descriptive)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def group_relative_feature_stats(
    model,
    out: Dict,
    X: torch.Tensor,
    feature_store=None,
) -> Optional[List[dict]]:
    """
    Compare a sample against the typical member of its own group, directly in
    raw feature space.

    ⚠ The axis differs from label_all_groups().
    ```
    label_all_groups              group A  vs  other groups   "features unusual for this group"
    group_relative_feature_stats  sample x vs  its own group  "features unusual for this sample"
    ```

    ⚠ **This is not attribution.** Place it beside prototype_deviation (the
      exact logit decomposition in embedding space) but do not connect the two
      causally: the embedder is non-linear, so there is no correspondence
      between embedding dimensions and features, and gradients cannot bridge
      it. "The prediction came out this way because of this feature" is not a
      sentence these values can support.

    ⚠ No filtering by group size. mean/std are unreliable for small groups,
      but choosing the cut-off here (say, drop n < 5) would make this an
      unjustified detector. `group_size` and `group_std` are always returned
      so the consumer can decide. Skipping z when `std < 1e-6` (constant
      within the group) is not a judgement — it avoids dividing by zero.

    ⚠ No truncation: all features are returned, sorted by |z| / rarity.

    Returns one dict per sample:
    {"numeric": [...], "categorical": [...], "group_size": n}
      numeric     : z = (x - group mean) / group std
      categorical : rarity = 1 - (frequency of this value within the group),
                    with group_mode / group_mode_freq attached so the consumer
                    can decide whether it matches the mode
    """
    fs = feature_store if feature_store is not None else getattr(model, "feature_store", None)
    sg = getattr(model.prototype_layer, "sample_groups", None)
    if fs is None or not sg:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    n_fill = fs._filled
    store  = fs._store[:n_fill].cpu().numpy()
    Xnp    = X.detach().cpu().numpy()
    n_feat = Xnp.shape[1]
    cat, num = _cat_num_idx(model, n_feat)
    cols     = _cols(model, n_feat)
    ha_np    = ha.detach().cpu().numpy()

    cache: Dict[int, np.ndarray] = {}
    result = []
    for b in range(Xnp.shape[0]):
        p = int(ha_np[b])
        if p not in cache:
            ids = sg[p] if (p < len(sg) and sg[p] is not None) else []
            ids = [i for i in ids if 0 <= i < n_fill]
            cache[p] = store[ids] if ids else store[:0]
        rows = cache[p]
        n_g  = int(rows.shape[0])

        num_out, cat_out = [], []
        if n_g > 0:
            for fi in num:
                if fi >= n_feat or fi >= rows.shape[1]:
                    continue
                col = rows[:, fi].astype(np.float64)
                mu, sd = float(col.mean()), float(col.std())
                val = float(Xnp[b, fi])
                if sd < 1e-6:      # numerical guard, not a judgement
                    continue
                # ⚠ Within-group percentile. It is invariant to monotone
                #   transforms, so measuring it in quantile space matches the
                #   percentile in the original units. z, by contrast, is
                #   computed in quantile space while the displayed value and
                #   the group representative are inverse-transformed back to
                #   real units -- the axes disagree, producing lines like
                #   "1 (group typical 1, z=-0.79)" where the numbers look the
                #   same yet a z is attached (seen across 20 credit-g cases).
                #   A percentile cannot have that mismatch by construction.
                # ⚠ Ties use midrank (below + equal/2). Discrete features
                #   repeat values, so a plain "fraction below" would distort
                #   the position.
                below = float((col < val).mean())
                equal = float((col == val).mean())
                num_out.append({
                    "feature_idx":  fi,
                    "feature_name": cols[fi] if fi < len(cols) else f"f{fi}",
                    "kind":         "numeric",
                    "value":        val,
                    "group_mean":   mu,
                    "group_std":    sd,
                    # z is dropped from the display but kept for analysis and
                    # sorting -- it can rank features better than a percentile.
                    "z":            (val - mu) / sd,
                    "group_pct":       below + equal / 2.0,   # midrank, 0~1
                    "group_pct_below": below,
                    "group_pct_equal": equal,
                })
            for fi in cat:
                if fi >= n_feat or fi >= rows.shape[1]:
                    continue
                code = int(round(float(Xnp[b, fi])))
                col  = np.rint(rows[:, fi]).astype(np.int64)
                freq = float((col == code).sum()) / n_g
                vals, cnts = np.unique(col, return_counts=True)
                cat_out.append({
                    "feature_idx":     fi,
                    "feature_name":    cols[fi] if fi < len(cols) else f"f{fi}",
                    "kind":            "categorical",
                    "value":           code,
                    "group_freq":      freq,
                    "group_mode":      int(vals[int(cnts.argmax())]),
                    "group_mode_freq": float(int(cnts.max())) / n_g,
                    "rarity":          1.0 - freq,
                })
        num_out.sort(key=lambda d: abs(d["z"]), reverse=True)
        cat_out.sort(key=lambda d: d["rarity"], reverse=True)
        result.append({"numeric": num_out, "categorical": cat_out, "group_size": n_g})
    return result


# ─────────────────────────────────────────────────────────────
# (5) Query-to-neighbour feature differences (all of them)
# ─────────────────────────────────────────────────────────────

def feature_gaps(query: Dict[str, float],
                 neighbour: Dict[str, float],
                 cat_names: set) -> List[dict]:
    """
    Return the per-feature difference between the query and one neighbour,
    for **every** feature.

    ⚠ The previous implementation (`_select_query_similar_features`) dropped
      any feature with `gap > 0.15` from the candidate set. It deleted
      information before showing the result, so the display showed only why
      the cases were similar and hid where they differed — a confirmation
      bias built into the output. Deciding what may be seen via a threshold
      is the same problem as building a detector. Nothing is cut here;
      sorting and truncation belong to the display layer.

    The gap follows the Gower definition. Categorical codes come from
    LabelEncoder and carry no order, so subtraction cannot express "how
    different" and the gap is 0/1 instead. (`delta` exists only so the
    display layer can recover the query value as `neighbour - delta`; the
    categorical delta must not be read as a magnitude.)

    Returns [{name, kind, query_value, neighbor_value, delta, gap}, ...] in
    input feature order. Sorting is left to the caller.
    """
    rows = []
    for k, v in neighbour.items():
        if k not in query:
            continue
        is_cat = k in cat_names
        if is_cat:
            gap = 0.0 if query[k] == v else 1.0
        else:
            gap = abs(v - query[k])
        rows.append({
            "name":           k,
            "kind":           "categorical" if is_cat else "numeric",
            "query_value":    query[k],
            "neighbor_value": v,
            "delta":          v - query[k],
            "gap":            gap,
        })
    return rows


# ─────────────────────────────────────────────────────────────
# (6) Q1 -- does prototype conditioning actually change the neighbour set?
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_conditioning_overlap(
    model,
    out: Dict,
    sample_ids: Optional[torch.Tensor] = None,
    n_permutations: int = 100,
    rng_seed: int = 0,
) -> Optional[Dict]:
    """
    Compare NN(q, G_p) against NN(q, D) within the same model.

    The claim under test
    ────────────────────
    The write-up calls the retrieval "prototype-conditioned". This checks
    whether the modifier has content.
        low overlap   -> the word "conditioned" is needed and accurate
        high overlap  -> the group constraint is effectively inert; the
                         wording must change and "the group constraint barely
                         changes the retrieved set" becomes a separate finding
    It is measured without assuming which way it comes out.

    ⚠ The model is not mutated. Mutating an attribute would let a diagnostic
      silently change later evaluation state. Instead this uses the override
      that MemoryBank.retrieve() already has (`hard_assignment=None` gives a
      global search). Adding a new argument would express one meaning in two
      ways and make contradictory calls possible.

    ⚠ The numbers are unreadable without a null baseline: drawing k=8 from a
      group of 60 lands near Jaccard 0.07 by chance alone. Both baselines are
      reported.
        analytic     given that m of the global top-k are inside this group,
                     k random draws from the group give E|A n B| = k*m/G
        permutation  actually draw n_permutations times and look at the spread
      `n_global_in_group` (= m) is the most directly readable value: if the
      global nearest neighbours were all inside the group already, then
      conditioning cannot change anything.

    ⚠ Nothing is truncated and nothing is judged. Every per-sample value is
      returned; summarising and interpreting belong to the analysis scripts.
      group_size always comes with it -- the same Jaccard means different
      things at |G| = 20 and |G| = 500.

    Returns {"per_sample": [...], "meta": {...}} or None.
      per_sample: sample_idx, group_id, group_size, k, n_local, n_global,
                  n_intersect, jaccard, top1_match, rank_corr,
                  n_global_in_group, null_jaccard_analytic,
                  null_jaccard_perm_mean, null_jaccard_perm_std,
                  fallback (bool), local_ids, global_ids
    """

    topk = out.get("topk_idx")
    # ⚠ This used to read out["query_retr"], a retrieval-only representation.
    #   With retr_proj removed, retrieval runs on query_emb, so recomputing
    #   the similarity with the same expression requires query_emb here too.
    qr   = out.get("query_emb")
    if topk is None or qr is None:
        return None
    ha = out.get("hard_group")
    if ha is None:
        ha = out.get("centroid_id")
    if ha is None:
        return None

    n_mem = int(model.memory.filled.item())
    k     = int(topk.shape[1])
    if n_mem < k:
        # If memory holds fewer than k entries, the group-constrained path
        # also falls back to a global search. The overlap then reads 1.0,
        # which does not mean the conditioning is ineffective -- it means the
        # conditioning was never applied.
        return {"per_sample": [], "meta": {
            "skipped": "memory.filled < k: conditioning was never applied",
            "memory_filled": n_mem, "k": k}}

    excl = sample_ids if getattr(model, "exclude_self_retrieval", False) else None
    # Global search: passing hard_assignment=None is the override itself,
    # so the model is never modified.
    _, _, g_idx = model.memory.retrieve(qr, k, hard_assignment=None, exclude_ids=excl)

    # Measure local and global similarity on one axis (normalised cosine)
    # so that distance_gap is comparable.
    _keys_n = F.normalize(model.memory.keys[:n_mem].detach(), dim=-1)
    _qn     = F.normalize(qr.detach(), dim=-1)
    _all    = _qn @ _keys_n.T                                   # (B, n_mem)
    sim_l   = _all.gather(1, topk.clamp(0, n_mem - 1)).cpu().numpy()
    sim_g   = _all.gather(1, g_idx.clamp(0, n_mem - 1)).cpu().numpy()

    mask  = out.get("neighbor_mask")
    msk   = mask.detach().cpu().numpy() if mask is not None else None
    l_np  = topk.detach().cpu().numpy()
    g_np  = g_idx.detach().cpu().numpy()
    ha_np = ha.detach().cpu().numpy()
    sg    = getattr(model.prototype_layer, "sample_groups", None)
    rng   = np.random.default_rng(rng_seed)

    grp_cache: Dict[int, np.ndarray] = {}
    per_sample = []
    for b in range(l_np.shape[0]):
        sel = [j for j in range(k) if msk is None or bool(msk[b, j])]
        A = l_np[b, sel]                      # conditioned top-k, descending similarity
        B = g_np[b]                           # global top-k
        p = int(ha_np[b])
        if p not in grp_cache:
            ids = (sg[p] if (sg is not None and p < len(sg) and sg[p] is not None) else [])
            grp_cache[p] = np.array([i for i in ids if 0 <= i < n_mem], dtype=np.int64)
        G = grp_cache[p]
        gsize = int(G.shape[0])

        sim_l_b = sim_l[b, sel] if len(sel) else np.array([])
        setA, setB = set(A.tolist()), set(B.tolist())
        inter = setA & setB
        union = setA | setB
        jac = (len(inter) / len(union)) if union else float("nan")

        # Rank correlation: rank within A vs rank within B, over the intersection
        if len(inter) >= 2:
            ra = {v: i for i, v in enumerate(A.tolist())}
            rb = {v: i for i, v in enumerate(B.tolist())}
            xs = np.array([ra[v] for v in inter], dtype=float)
            ys = np.array([rb[v] for v in inter], dtype=float)
            from scipy.stats import spearmanr          # noqa: F401 (optional dependency)
            rc = float(spearmanr(xs, ys).statistic)
        else:
            rc = float("nan")

        # Null model: given that m of the global top-k fall inside this
        # group, how much would k random draws from the group overlap?
        m = int(len(setB & set(G.tolist()))) if gsize else 0
        if gsize >= 1:
            exp_i = k * m / gsize
            null_a = exp_i / max(2 * k - exp_i, 1e-12)
            draws = []
            take = min(k, gsize)
            for _ in range(n_permutations):
                R = set(rng.choice(G, size=take, replace=False).tolist())
                iu = len(R & setB)
                draws.append(iu / max(len(R | setB), 1))
            null_p_mean, null_p_std = float(np.mean(draws)), float(np.std(draws))
        else:
            null_a = null_p_mean = null_p_std = float("nan")

        # ⚠ Jaccard alone is not enough. At the same overlap, "the candidates
        #   outside the group were nearly as close" and "something much closer
        #   existed outside the group" are entirely different, and only the
        #   second means the constraint actually cost something. Subtract the
        #   local and global rank-1 and rank-k values on the same cosine axis.
        _sl = sim_l_b                        # cosine to this sample's local neighbours
        _sg_ = sim_g[b]                      # global top-k cosine, descending
        gap_top1 = float(_sg_[0] - _sl.max()) if len(_sl) else float("nan")
        gap_topk = float(_sg_[-1] - _sl.min()) if len(_sl) else float("nan")

        per_sample.append({
            "sample_idx":  b,
            "group_id":    p,
            "group_size":  gsize,
            "k":           k,
            "n_local":     int(len(setA)),
            "n_global":    int(len(setB)),
            "n_intersect": int(len(inter)),
            "jaccard":     jac,
            "top1_match":  (bool(A[0] == B[0]) if len(A) and len(B) else None),
            "rank_corr":   rc,
            "n_global_in_group":      m,
            "null_jaccard_analytic":  float(null_a),
            "null_jaccard_perm_mean": null_p_mean,
            "null_jaccard_perm_std":  null_p_std,
            # Similarity cost paid by the group constraint (>= 0). Near zero
            # means nothing closer existed outside the group, i.e. the
            # constraint was effectively free.
            "distance_gap_top1": gap_top1,
            "distance_gap_topk": gap_topk,
            # ⚠ Fallback samples went through the expanded search, so the
            #   group constraint was already loose for them. Whether to
            #   exclude them belongs to the analysis stage, but the exclusion
            #   rate is itself a result: at 30% fallback, describing the
            #   retrieval as "group-constrained" is only half true.
            "fallback":    bool(gsize < k or (msk is not None and not msk[b].all())),
            "local_ids":   A.tolist(),
            "global_ids":  B.tolist(),
        })

    return {"per_sample": per_sample, "meta": {
        "memory_filled": n_mem, "k": k,
        "exclude_self_retrieval": bool(getattr(model, "exclude_self_retrieval", False)),
        "n_permutations": n_permutations,
    }}


# ─────────────────────────────────────────────────────────────
# (7) What do prototypes represent -- density mode or class anchor?
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def prototype_class_alignment(model) -> Optional[Dict]:
    """Measure whether prototypes end up dividing the classes between them.

    Why this is needed
    ──────────────────
    `P >= C` is **necessary but not sufficient**. The prototype term alone can
    produce at most P distinct argmax values, so P < C forces the correction
    term to take over -- but even at P = C, all C centroids could point at the
    same class. What raising P is meant to buy is
        wanted:   a predictive anchor -- classes are divided up while
                  structure remains inside each group
        unwanted: class memory -- centroids collapse onto class labels
    and the two are told apart by alignment and H(Y_G).

    Returns
    ───────
        alignment        mean_p max_y P(y|p). 1/C is chance, 1.0 is pure
        alignment_std    spread across prototypes
        group_entropy    mean_p H(Y|p). Near 0 means class memory
        n_prototypes     the configured P
        n_effective      number of non-empty prototypes
        n_eff_entropy    exp(H(assignment distribution)) -- an effective count
                         that also reflects size imbalance (only 12 of P being
                         alive and one live group holding 61% are different
                         problems)
        dead_ratio       fraction that are empty
        per_prototype    [{p, size, top_class, top_prop, entropy}, ...]

    ⚠ Only the training distribution is read (sample_groups + memory.labels).
      No test label is touched, so this is safe to log during training.
    ⚠ Regression has no classes, so None is returned.
    """
    if getattr(model, "tasktype", None) == "regression":
        return None
    sg = getattr(model.prototype_layer, "sample_groups", None)
    if not sg:
        return None
    n_mem = int(model.memory.filled.item())
    if n_mem == 0:
        return None
    lab = model.memory.labels[:n_mem].detach().cpu().numpy().round().astype(int)
    C = int(lab.max()) + 1

    rows, sizes, aligns, ents = [], [], [], []
    for p, ids in enumerate(sg):
        ids = [i for i in (ids or []) if 0 <= i < n_mem]
        if not ids:
            rows.append({"p": p, "size": 0, "top_class": None,
                         "top_prop": None, "entropy": None})
            sizes.append(0)
            continue
        y = lab[ids]
        cnt = np.bincount(y, minlength=C)
        prop = cnt / cnt.sum()
        top = int(prop.argmax())
        nz = prop[prop > 0]
        H = float(-(nz * np.log(nz)).sum())
        rows.append({"p": p, "size": len(ids), "top_class": top,
                     "top_prop": float(prop[top]), "entropy": H})
        sizes.append(len(ids))
        aligns.append(float(prop[top]))
        ents.append(H)

    sizes_a = np.asarray(sizes, dtype=np.float64)
    tot = sizes_a.sum()
    # Size-weighted effective count. Counting only live prototypes misses the
    # case where one absorbs most of the data (measured on phoneme: 42 of 65
    # alive, yet a single group held 61% of the training set).
    p_assign = sizes_a[sizes_a > 0] / tot if tot > 0 else np.array([1.0])
    n_eff_H = float(np.exp(-(p_assign * np.log(p_assign)).sum()))

    return {
        "alignment":     float(np.mean(aligns)) if aligns else float("nan"),
        "alignment_std": float(np.std(aligns))  if aligns else float("nan"),
        "group_entropy": float(np.mean(ents))   if ents   else float("nan"),
        "chance_alignment": 1.0 / C,
        "n_prototypes":  len(sg),
        "n_effective":   int((sizes_a > 0).sum()),
        "n_eff_entropy": n_eff_H,
        "dead_ratio":    float((sizes_a == 0).mean()),
        "n_classes":     C,
        "per_prototype": rows,
    }


# ─────────────────────────────────────────────────────────────
# (8) Has the context space collapsed -- prototype vocabulary collapse?
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def context_space_diversity(model) -> Optional[Dict]:
    """Measure whether the prototypes actually supply distinct contexts.

    Why dead ratio is not enough
    ────────────────────────────
    ⚠ Uniform prototype usage is **not** the goal. A centroid here is not the
      cluster center of a uniform partition but a latent context anchor
      pointing at a mode of the data manifold. If the real distribution is
          A 70% / B 20% / C 5% / D 5%
      then a good prototype allocation is just as uneven. Optimising for
      uniformity (say KL(usage || Uniform)) would turn the model into a
      clustering method.

    ⚠ The real failure is **vocabulary collapse**. If P = 50 is configured but
      the model only ever produces 12 distinct context directions, then
      `context_emb` is effectively "one prototype plus noise" -- the same
      situation as an LLM with a 50k vocabulary that uses 12 tokens. That is
      not an imbalance problem; it is **discarded representational capacity**.

    What is measured
    ────────────────
        usage_entropy_eff   exp(H(usage distribution)): how many are in
                            effective use
        gini                usage concentration (0 uniform, 1 monopoly).
                            Reported for reference, not as a target
        top1_share          fraction taken by the largest prototype
        context_eff_rank    **the key one.** exp of the entropy of the
                            eigenvalue spectrum of the covariance of the
                            context_emb that training samples actually
                            receive, i.e. how many dimensions the context
                            space really uses
        context_eff_rank_uniform
                            the effective rank a uniform usage distribution
                            would have given. The ratio of the two is the
                            representational dimensionality lost to the skew
        centroid_cos_mean/min
                            pairwise cosine among live centroids. Near 1 means
                            the centroids have bunched into one direction

    ⚠ context_eff_rank is bounded by embed_dim, and by P when P < embed_dim.
    """
    pl = getattr(model, "prototype_layer", None)
    sg = getattr(pl, "sample_groups", None) if pl is not None else None
    if pl is None or not sg:
        return None
    C_emb = pl.centroid_emb.detach()                       # (P, D)
    P, D = C_emb.shape
    n_mem = int(model.memory.filled.item())
    sizes = np.array([len([i for i in (g or []) if 0 <= i < n_mem]) for g in sg],
                     dtype=np.float64)
    tot = sizes.sum()
    if tot <= 0:
        return None
    w = sizes / tot                                        # usage distribution, shape (P,)

    nz = w[w > 0]
    H = float(-(nz * np.log(nz)).sum())
    gini = float((np.abs(sizes[:, None] - sizes[None, :]).sum())
                 / (2 * len(sizes) * sizes.sum())) if tot > 0 else float("nan")

    def _eff_rank(weights):
        # Weighted covariance of the context_emb a sample receives. Since
        # context_emb = c_p, this is the covariance over prototypes.
        # Cov = Σ_p w_p (c_p - μ)(c_p - μ)^T,  μ = Σ_p w_p c_p
        wt = torch.as_tensor(weights, dtype=C_emb.dtype, device=C_emb.device)
        mu = (wt[:, None] * C_emb).sum(0, keepdim=True)
        Xc = (C_emb - mu) * wt[:, None].sqrt()
        ev = torch.linalg.svdvals(Xc) ** 2
        ev = ev[ev > 0]
        if ev.numel() == 0:
            return 0.0
        p = ev / ev.sum()
        return float(torch.exp(-(p * p.log()).sum()))

    alive = sizes > 0
    er_actual = _eff_rank(w)
    er_uniform = _eff_rank(alive / max(alive.sum(), 1))     # uniform over the live prototypes only

    Cn = F.normalize(C_emb[torch.as_tensor(alive)], dim=-1)
    G = (Cn @ Cn.T).cpu().numpy()
    iu = np.triu_indices(G.shape[0], k=1)
    pair = G[iu] if len(iu[0]) else np.array([np.nan])

    return {
        "n_prototypes":  int(P),
        "embed_dim":     int(D),
        "n_alive":       int(alive.sum()),
        "usage_entropy_eff": float(np.exp(H)),
        "gini":          gini,
        "top1_share":    float(sizes.max() / tot),
        "context_eff_rank":         er_actual,
        "context_eff_rank_uniform": er_uniform,
        # Fraction of representational dimensionality lost to the imbalance.
        # Near 1 means the skew barely cost anything -- uneven usage is not
        # collapse by itself.
        "eff_rank_ratio": (er_actual / er_uniform) if er_uniform > 0 else float("nan"),
        "centroid_cos_mean": float(np.mean(pair)),
        "centroid_cos_max":  float(np.max(pair)),
    }