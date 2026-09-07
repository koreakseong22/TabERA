## Train and evaluate the best TabERA configuration found by optimize.py.
## Paper: TabERA — Tabular Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import os, argparse, time

# ── Set CUDA_VISIBLE_DEVICES before torch is imported ──────
_parser_pre = argparse.ArgumentParser(add_help=False)
_parser_pre.add_argument("--gpu_id", type=int, default=0)
_parser_pre.add_argument("--deterministic", action="store_true")
_pre, _ = _parser_pre.parse_known_args()
if _pre.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.gpu_id)
if _pre.deterministic:
    # For torch.use_deterministic_algorithms(True) to make certain cuBLAS
    # operations deterministic on CUDA >= 10.2, this variable must be set
    # *before* the CUDA context is created, i.e. before torch is imported.
    # Setting it afterwards is silently ignored, so it is handled here in the
    # pre-parser alongside --gpu_id.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from libs.data         import TabularDataset
from libs.search_space import params_to_model_kwargs, study_pkl_tag, HPO_TRAINING_SCHEDULE
from libs.data         import get_batch_size
from libs.supervised   import TabERAWrapper
from libs.tabera         import TabERA
from libs.prototypes     import inverse_transform_numeric
from libs                import diagnostics as diag
from libs.eval         import calculate_metric, get_preds_and_probs, get_criterion
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────
# Explanation output
# ─────────────────────────────────────────────────────────────

def _fmt_signed(x: float, decimals: int = 4) -> str:
    """
    Format a signed decimal.

    A very small negative value (-0.00003) rounds to "-0.0000" in Python,
    which reads as a meaningful negative when it is really zero. The random
    null mean in rank_correlation is the typical case: the expected
    correlation between random rankings is 0, so tiny negatives are common.
    Adding +0.0 after round() normalises negative zero to positive zero
    (IEEE 754: -0.0 + 0.0 == 0.0).
    """
    v = round(x, decimals) + 0.0
    return f"{v:.{decimals}f}"


def _fmt_pval(p: float, n_draws: int) -> str:
    """
    Format an empirical p-value from a bootstrap or permutation test.

    When none of n_draws resamples exceeds the observed value (count = 0), the
    naive output is p = 0.0000. That does not mean the probability is exactly
    zero -- it is the resolution limit of the test, and all that can be said
    is that p is below 1/n_draws. Both p_shap_vs_null in rank_correlation and
    p_vs_null in interaction_check have this issue, hence a shared helper.
    """
    if p <= 0.0:
        return f"<{1.0 / n_draws:.4g}"
    return f"{p:.4f}"


def _fmt_class(name: str, count: int, n: int, prop: float) -> str:
    """Format one class as: "name" count/n (prop%).

    Used consistently for region class summaries.
    """
    return f"\"{name}\" {count}/{n} ({prop:.0%})"


def _format_target_info(tinfo) -> str:
    """target_info(label_groups_by_target() result) as a short string."""
    if tinfo is None:
        return "(no target info)"
    if tinfo["kind"] == "classification":
        s = _fmt_class(tinfo['top_class_name'], tinfo['top_count'], tinfo['n'], tinfo['top_prop'])
        if tinfo["second"] is not None:
            s += ", " + _fmt_class(tinfo['second']['name'], tinfo['second']['count'],
                                    tinfo['n'], tinfo['second']['prop'])
        return s
    else:
        return f"target≈{tinfo['group_mean']:.3g}(p{tinfo['percentile']:.0f})"


def _split_by_kind(labels, get_kind, get_str):
    """Split items by kind (numeric / categorical) into two string lists."""
    num_strs, cat_strs = [], []
    for item in labels:
        (num_strs if get_kind(item) == "numeric" else cat_strs).append(get_str(item))
    return num_strs, cat_strs


def print_explanation(explanations: list, sample_idx: int, col_names: list,
                       cat_category_names: dict = None,
                       quantile_transformer=None, num_cols: list = None,
                       pred_info: dict = None,
                       target_class_names: list = None,
                       tasktype: str = None,
                       max_neighbors: int = 3,
                       max_features: int = 3,
                       max_gaps: int = 2,
                       verbose: bool = False,
                       max_dims: int = 5) -> None:
    """Print a compact, user-facing TabERA explanation.

    The default view deliberately answers only four questions:
      1) What was predicted?
      2) Which predictive region contains this case?
      3) What do the nearest past cases in that region look like?
      4) Where does this case differ from its region?

    All values come from the same observer outputs as before; this function is
    display-only and does not change prediction, routing, retrieval, or any
    diagnostic computation.  ``--explain_verbose`` exposes the researcher
    details (routing mass, exact predicted-channel logit decomposition,
    entropies, and raw representation distance) that are intentionally hidden
    from the default user view.

    ``max_*`` values are display budgets only, never decision thresholds.
    """
    e = explanations[sample_idx]
    proto = e.get("prototype") or {}
    le = e.get("local_evidence")
    nbrs = e.get("neighbors") or []
    dv = e.get("prototype_deviation")
    gc = e.get("group_stats")
    rp = e.get("region_position")

    # ── Small display helpers ──────────────────────────────────────
    name_to_idx = {name: i for i, name in enumerate(col_names)} if col_names else {}

    def _label_name(v):
        if v is None:
            return "?"
        if tasktype == "regression":
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)
        try:
            code = int(round(float(v)))
        except Exception:
            return str(v)
        if target_class_names and 0 <= code < len(target_class_names):
            return str(target_class_names[code])
        return str(code)

    def _region_name(raw):
        raw = str(raw or "Region")
        if raw.startswith("Centroid_"):
            return "Region " + raw.split("Centroid_", 1)[1]
        if raw.startswith("Prototype_"):
            return "Region " + raw.split("Prototype_", 1)[1]
        return raw

    def _dist_str(counts, total, max_items=3):
        """Compact but exhaustive-in-mass class distribution.

        For many-class regions only the largest classes are named and the
        remaining probability mass is explicitly collapsed into ``others``.
        """
        counts = {int(k): int(v) for k, v in (counts or {}).items()}
        if not total or not counts:
            return "(no label summary)"
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        shown = ranked[:max_items]
        parts = [f"{_label_name(k)} {c}/{total} ({c/total:.0%})" for k, c in shown]
        rest = sum(c for _, c in ranked[max_items:])
        if rest:
            parts.append(f"others {rest}/{total} ({rest/total:.0%})")
        return " · ".join(parts)

    def _pretty_num(x):
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

    def _real_numeric(feature_idx, v):
        if quantile_transformer is None or num_cols is None:
            return None
        return inverse_transform_numeric(quantile_transformer, num_cols, feature_idx, v)

    def _fmt_num_value(name: str, uniform_val: float) -> str:
        if name in name_to_idx:
            real = _real_numeric(name_to_idx[name], uniform_val)
            if real is not None:
                return _pretty_num(real)
        return f"{float(uniform_val):.3f}"

    def _fmt_cat_value(name: str, code_val: float) -> str:
        names_for_col = cat_category_names.get(name) if cat_category_names else None
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

    def _profile_position(pct):
        if pct is None or pct != pct:
            return None
        pct = float(pct)
        if pct >= 0.9995:
            return "highest in region"
        if pct <= 0.0005:
            return "lowest in region"
        if pct >= 0.5:
            top = max(1, int(round((1.0 - pct) * 100)))
            return f"top {top}% in region"
        bot = max(1, int(round(pct * 100)))
        return f"bottom {bot}% in region"

    def _gap_summary(nb, n=max_gaps):
        gp = nb.get("gaps") or []
        diff = [g for g in sorted(gp, key=lambda g: g["gap"], reverse=True)
                if g.get("gap", 0.0) > 1e-9][:n]
        parts = []
        for g in diff:
            name, nval, kind, delta = g["name"], g["neighbor_value"], g["kind"], g["delta"]
            qval = nval - delta
            if kind == "categorical":
                parts.append(f"{name}: {_fmt_cat_value(name, qval)} vs {_fmt_cat_value(name, nval)}")
            else:
                parts.append(f"{name}: {_fmt_num_value(name, qval)} vs {_fmt_num_value(name, nval)}")
        return parts

    # ── Header ────────────────────────────────────────────────────
    print(f"\n{'━'*48}")
    print(f"  TabERA Explanation — Sample #{sample_idx}")
    print(f"{'━'*48}")

    # ── Prediction ────────────────────────────────────────────────
    print("\n  Prediction")
    if pred_info is not None:
        label = pred_info.get("pred_label", "prediction")
        conf = pred_info.get("pred_confidence")
        if conf is None:
            print(f"     → {label}")
        else:
            print(f"     → {label} — {conf:.1%}")

        if dv is not None and dv.get("prob_final") is not None:
            pp = dv.get("prob_proto")
            pf = dv.get("prob_final")
            changed = bool(dv.get("argmax_changed"))
            if changed:
                proto_code = dv.get("proto_pred")
                proto_label = _label_name(proto_code)
                print(f"     Region-only decision: {proto_label} → Final: {label}")
                if pp is not None:
                    print(f"     Final-class confidence at region baseline: {pp:.1%} → {pf:.1%}")
            elif pp is not None:
                print(f"     Region-only prediction: {label} {pp:.1%}")
                print(f"     Final prediction:       {label} {pf:.1%}")
    elif dv is not None and tasktype == "regression":
        print(f"     region={dv['logit_proto']:+.4f}  sample={dv['logit_dev']:+.4f}  "
              f"final={dv['logit_proto'] + dv['logit_dev']:+.4f}")

    # ── ① Predictive region ──────────────────────────────────────
    region = _region_name(proto.get("assigned_group"))
    group_size = None
    if le is not None:
        group_size = le.get("group_size")
    if group_size is None:
        tinfo = proto.get("target_info")
        group_size = tinfo.get("n") if tinfo is not None else None

    print("\n  ① Predictive region")
    if group_size is not None:
        print(f"     {region} — {int(group_size)} training cases")
    else:
        print(f"     {region}")

    if tasktype == "regression":
        if le is not None and le.get("group_mean") is not None:
            print(f"     Region target: mean {le['group_mean']:.4g} · std {le['group_std']:.4g}")
        elif proto.get("target_info") is not None:
            print(f"     Region target: {_format_target_info(proto['target_info'])}")
    else:
        gcnt = (le or {}).get("group_label_counts")
        if gcnt:
            print(f"     Outcomes: {_dist_str(gcnt, int(group_size), max_items=3)}")
        elif proto.get("target_info") is not None:
            print(f"     Outcomes: {_format_target_info(proto['target_info'])}")

    # ── ② Similar past cases ─────────────────────────────────────
    print("\n  ② Similar past cases — shown as evidence only")
    if le is not None:
        nnb = int(le.get("n_neighbors", len(nbrs)))
        if tasktype == "regression":
            print(f"     {nnb} nearest cases in this region: "
                  f"mean {le.get('local_mean', float('nan')):.4g} · "
                  f"std {le.get('local_std', float('nan')):.4g}")
        else:
            lcnt = le.get("label_counts") or {}
            print(f"     Retrieved outcomes: {_dist_str(lcnt, nnb, max_items=3)}")
        # The retrieval implementation expands beyond the assigned group only
        # when the group itself cannot supply k candidates. This note prevents
        # the UI from falsely calling such a result purely within-region.
        if (le.get("group_size") is not None and nnb
                and int(le["group_size"]) < nnb):
            print(f"     Note: search expanded beyond the assigned region "
                  f"(region had {int(le['group_size'])} candidates).")

    if not nbrs:
        print("     No retrieved cases available.")
    else:
        pred_code = (pred_info or {}).get("pred_code")
        contrast = None
        if tasktype != "regression" and pred_code is not None:
            contrast = next((nb for nb in nbrs
                             if nb.get("label") is not None
                             and int(round(float(nb["label"]))) != int(pred_code)), None)

        # "Representative" rather than strictly "most similar": if a contrast
        # exists outside the display budget, reserve the last slot for the
        # closest contrast so counter-evidence is not hidden by presentation.
        top = list(nbrs[:max_neighbors])
        if contrast is not None and contrast not in top and max_neighbors > 0:
            top = list(nbrs[:max(0, max_neighbors - 1)]) + [contrast]

        print("     Representative cases:")
        for nb in top:
            sid = nb.get("sample_id")
            sid_str = f"train #{sid}" if sid is not None and sid >= 0 else f"memory #{nb['memory_idx']}"
            tag = ""
            if tasktype != "regression" and pred_code is not None and nb.get("label") is not None:
                if int(round(float(nb["label"]))) != int(pred_code):
                    tag = " · contrast"
            print(f"       {sid_str:<12s} {_label_name(nb.get('label')):<14s} "
                  f"similarity {nb['similarity']:.3f}{tag}")

        if tasktype != "regression" and pred_code is not None:
            if contrast is None:
                print(f"     No contrasting case among the {len(nbrs)} retrieved cases.")
            else:
                diffs = _gap_summary(contrast)
                if diffs:
                    print("     Compared with the contrast case:")
                    for diff in diffs:
                        print(f"       {diff}")

    # ── ③ How this case differs from its region ────────────────────
    if (gc and (gc.get("numeric") or gc.get("categorical"))) or rp is not None:
        print("\n  ③ How this case differs from its region")

    if gc and (gc.get("numeric") or gc.get("categorical")):
        nums = list(gc.get("numeric") or [])
        # In a section explicitly about departures, categorical values equal
        # to the region mode do not earn scarce display space. This is a
        # semantic filter (equal vs different), not a numeric threshold.
        cats = [d for d in (gc.get("categorical") or [])
                if d.get("differs_from_mode", True) or d.get("absent_from_group", False)]

        budget = max(0, int(max_features))
        chosen_num, chosen_cat = [], []
        if budget > 0:
            if nums and cats:
                n_num = min(len(nums), (budget + 1) // 2)
                n_cat = min(len(cats), budget - n_num)
                # Fill any unused categorical slots with numeric ones, then
                # vice versa. Total output never exceeds max_features.
                n_num = min(len(nums), n_num + max(0, budget - n_num - n_cat))
                n_cat = min(len(cats), budget - n_num)
                if n_num + n_cat < budget:
                    n_cat = min(len(cats), n_cat + (budget - n_num - n_cat))
                chosen_num, chosen_cat = nums[:n_num], cats[:n_cat]
            elif nums:
                chosen_num = nums[:budget]
            else:
                chosen_cat = cats[:budget]

        if chosen_num or chosen_cat:
            print("     Values that stand out:")

        for d in chosen_num:
            vr = _real_numeric(d["feature_idx"], d["value"])
            mr = _real_numeric(d["feature_idx"], d["group_mean"])
            v_s = _pretty_num(vr) if vr is not None else f"{d['value']:.3f}"
            m_s = _pretty_num(mr) if mr is not None else f"{d['group_mean']:.3f}"
            pos = _profile_position(d.get("group_pct"))
            pos_s = f" · {pos}" if pos else ""
            print(f"       • {d['feature_name']}={v_s}")
            detail = f"typical in region: {m_s}"
            if pos:
                detail += f" · {pos}"
            print(f"         {detail}")

        for d in chosen_cat:
            value = _fmt_cat_value(d["feature_name"], d["value"])
            mode = _fmt_cat_value(d["feature_name"], d["group_mode"])
            if d.get("absent_from_group"):
                share = "not seen among region cases"
            else:
                share = f"{d.get('group_freq', 0.0):.0%} in region"
            print(f"       • {d['feature_name']}={value}")
            print(f"         {share} · most common: {mode} {d.get('group_mode_freq', 0.0):.0%}")

    if rp is not None:
        farther = float(rp["group_pct"])    # share of region closer to centre
        closer = 1.0 - farther
        if farther >= 0.995:
            where = "Less typical than every other case in this region"
        elif closer >= 0.995:
            where = "More typical than every other case in this region"
        elif farther >= 0.5:
            where = f"Less typical than {farther:.0%} of cases in this region"
        else:
            where = f"More typical than {closer:.0%} of cases in this region"
        print("     Overall position:")
        print(f"       {where}")

    # ── Researcher-only detail ─────────────────────────────────────
    if verbose:
        print("\n  Advanced diagnostics")
        if dv is not None:
            if dv.get("logit_proto") is not None and dv.get("logit_dev") is not None:
                print(f"     Predicted-channel logit: region {dv['logit_proto']:+.4f} "
                      f"+ correction {dv['logit_dev']:+.4f}")
        if proto.get("routing_confidence") is not None:
            print(f"     Routing mass: assigned {proto['routing_confidence']:.1%}", end="")
            if proto.get("others_mass") is not None:
                print(f" · others {proto['others_mass']:.1%}")
            else:
                print()
            for r in proto.get("runners_up") or []:
                if r.get("target_info") is None:
                    continue
                print(f"       runner-up {r['label']}: {r['routing_confidence']:.1%} "
                      f"({_format_target_info(r['target_info'])})")
        if le is not None and tasktype != "regression":
            if le.get("label_entropy") is not None:
                print(f"     Label entropy: neighbours {le['label_entropy']:.3f} "
                      f"· region {le['group_label_entropy']:.3f}")
            ar = le.get("ambiguity_ratio")
            if ar is not None and ar == ar:
                print(f"     Local/region entropy ratio: {ar:.3f}")
        labels = proto.get("group_feature_labels") or []
        if labels:
            shown = [f"{fl.feature_name}={fl.label}" for fl in labels[:max_features]]
            print(f"     Region-characteristic features: {', '.join(shown)}")
        if rp is not None:
            print(f"     Cosine distance to region centre: {rp['distance']:.4f} "
                  f"(region min/median/max {rp['group_min']:.4f}/"
                  f"{rp['group_median']:.4f}/{rp['group_max']:.4f})")

    print(f"{'━'*48}")




# ─────────────────────────────────────────────────────────────
# Calibration analysis: routing confidence vs prediction confidence
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Per-centroid statistics (size / purity / cohesion) on the training set
# ─────────────────────────────────────────────────────────────

def compute_centroid_train_stats(model, X_train, y_train, tasktype: str,
                                  class_names=None, batch_size: int = 256):
    """
    Compute per-centroid size, purity and cohesion on the training set.

    size     number of members in the group
    purity   share of the most frequent target within the group
    cohesion how tightly the members sit around their own centroid

    This duplicates the calculation in the centroid_representativeness
    ablation, which only prints and pickles its results and therefore cannot
    be reused. Having it as a function lets run_calibration_analysis()
    correlate group size against purity directly -- testing whether larger
    centroids are less pure and therefore predict worse. The existing ablation
    code is left untouched: the duplication is accepted in exchange for not
    risking a regression there.

    Returns {centroid_idx: {"size": int, "purity": float|None,
                            "cohesion": float, "gap": float|None}}.
    purity and gap are classification-only (None for regression); gap is
    purity minus the global baseline, the overall most-frequent target
    share.
    """
    model.eval()
    P = model.prototype_layer.P
    sample_groups = model.prototype_layer.sample_groups
    target_labels = model.prototype_layer.target_labels
    if sample_groups is None:
        return {}

    y_train_np = y_train.detach().cpu().numpy()
    global_majority_prop = None
    if tasktype in ("multiclass", "binclass"):
        y_int = np.rint(y_train_np).astype(int)
        _, counts = np.unique(y_int, return_counts=True)
        global_majority_prop = float(counts.max() / counts.sum())

    with torch.no_grad():
        c_norm = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
        q_chunks = []
        for start in range(0, X_train.shape[0], batch_size):
            q_chunks.append(
                F.normalize(model.embedder(X_train[start:start + batch_size]), dim=-1).cpu()
            )
        q_all = torch.cat(q_chunks)
    c_norm_cpu = c_norm.cpu()

    stats = {}
    for p in range(P):
        grp = sample_groups[p] if sample_groups is not None else None
        size = len(grp) if grp else 0
        if size == 0:
            continue
        idx_t = torch.as_tensor(grp, dtype=torch.long)
        q_grp = q_all[idx_t]
        cohesion = float((q_grp @ c_norm_cpu[p]).mean())

        tl = target_labels.get(p) if target_labels is not None else None
        purity, gap = None, None
        if tl is not None and tl.get("kind") == "classification":
            purity = tl["top_prop"]
            gap = purity - global_majority_prop if global_majority_prop is not None else None

        # Label entropy H(y|c) = -sum p(y|c) log p(y|c). Unlike purity, which
        # looks only at the largest class share, this reflects the whole class
        # distribution within the group. For three classes, (0.5, 0.5, 0.0)
        # and (0.5, 0.25, 0.25) have the same purity (0.5) but different
        # entropy -- the first is lower, spanning only two classes. It
        # captures how widely a group is spread across classes, which purity
        # cannot see. Classification only.
        entropy = None
        if tasktype in ("multiclass", "binclass"):
            y_grp_int = np.rint(y_train_np[grp]).astype(int)
            _, grp_counts = np.unique(y_grp_int, return_counts=True)
            p_y = grp_counts / grp_counts.sum()
            entropy = float(-(p_y * np.log(p_y + 1e-12)).sum())

        stats[p] = {"size": size, "purity": purity, "cohesion": cohesion,
                     "gap": gap, "entropy": entropy}

    return stats


# ─────────────────────────────────────────────────────────────
# ECE, as a reusable standalone function
# ─────────────────────────────────────────────────────────────

def compute_ece(pred_confidence: np.ndarray, corrects: np.ndarray, n_bins: int = 5) -> float:
    """
    Standard ECE (Guo et al. 2017), the same definition
    run_calibration_analysis uses internally: the bin-size-weighted mean of
    |accuracy - mean_confidence| per bin, extracted as a standalone function.

    It separates two things a rising logloss cannot distinguish on its own --
    whether the probabilities themselves degraded (a calibration problem), or
    whether accuracy held while the probability distribution moved for another
    reason such as a change in logit scale.
    """
    pred_confidence = np.asarray(pred_confidence)
    corrects = np.asarray(corrects)
    n_total = len(corrects)
    if n_total == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (pred_confidence >= lo) & (pred_confidence <= hi)
        else:
            mask = (pred_confidence >= lo) & (pred_confidence < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(corrects[mask].mean())
        mean_conf = float(pred_confidence[mask].mean())
        ece += (n / n_total) * abs(acc - mean_conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────
# Calibration analysis entry point
# ─────────────────────────────────────────────────────────────



























def run_calibration_analysis(model, X_test, y_test, tasktype: str,
                              batch_size: int = 512, n_bins: int = 5,
                              X_train=None, y_train=None, class_names=None):
    """Compare routing_confidence (layer 1) and prediction_confidence
    (layer 2) against actual accuracy over the whole test set.

    --explain walks a few samples (n_explain) in detail; this answers "is the
    final prediction trustworthy even when routing is ambiguous" as a
    statistic over the entire test set rather than from one sample. Everything
    expensive -- feature summaries, neighbour text -- is therefore stripped
    out, leaving only routing_confidence, prediction_confidence and whether
    the prediction was correct.

    ⚠ routing_confidence used to be binned on the same fixed 0/20/40/60/80/100%
    grid as prediction_confidence, which was wrong. Measured on adult
    (P = 190), all 4,523 test samples landed in the single 0-20% bin -- and
    that must not be read as "routing collapsed". The absolute scale of
    routing_confidence = softmax(cos(q, c)) depends structurally on the
    prototype count: the uniform baseline is 1/P. Unlike prediction_confidence, which is a real
    probability where 0-100% means the same thing everywhere, a fixed percent
    grid offers no basis for judging whether a value is low -- neither across
    datasets with different P nor within one dataset. So:
    (a) the distribution itself is reported first (mean, median, std, min,
        max, p90, p99), and
    (b) the bins are **percentiles** (bottom 20%, 20-40%, ..., top 20%) rather
        than absolute confidence percentages. Whatever P is, this actually
        answers the intended question: within this test
        set, is there an accuracy difference between the samples where routing
        was relatively ambiguous and those where it was confident.
    prediction_confidence is a real probability and keeps its fixed bins.

    Returns {
      "routing_stats": {"mean":.., "median":.., "std":.., "min":.., "max":..,
                        "p90":.., "p99":.., "n_prototypes":..,
                        "uniform_baseline":..}
      "routing_bins": [(lo_pct, hi_pct, lo_conf, hi_conf, n, acc), ...]
                      # percentile-based
      "prediction_bins": [(lo, hi, n, acc, mean_conf), ...]
                      # fixed confidence-percentage bins
      "prediction_ece": float,   # Expected Calibration Error
      "n_total": int,
      "overall_acc": float,
    }
    """
    if tasktype == "regression":
        raise ValueError("calibration_analysis is classification-only "
                         "(binclass/multiclass): regression has no notion "
                         "of routing or prediction confidence.")

    model.eval()
    routing_confs, pred_confs, corrects, assigned_centroids, margins = [], [], [], [], []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start:start + batch_size]
            y_batch = y_test[start:start + batch_size]
            # forward now builds only the routing explanation; neighbour
            # assembly and group statistics moved to diagnostics, so there is
            # no cost left to switch off here.
            out = model(X_batch, return_explanations=True)

            explanations = out.get("explanations", [])
            if not explanations:
                # An early batch where the memory bank is not yet filled;
                # skipped for the same reason as the "no explanations" case
                # under --explain.
                continue

            pred_idx, pred_probs = get_preds_and_probs(out["logits"][:len(explanations)], tasktype)

            for i, exp in enumerate(explanations):
                routing_confs.append(exp["prototype"]["routing_confidence"])
                assigned_centroids.append(int(exp["prototype"]["centroid_idx"]))
                margins.append(exp["prototype"]["margin"])  # top1 - runner-up routing prob
                idx = int(pred_idx[i].item())
                pred_confs.append(float(pred_probs[i, idx].item()))
                y_i = int(y_batch[i].item()) if tasktype == "multiclass" else int(y_batch[i].item())
                corrects.append(int(idx == y_i))

    routing_confs      = np.array(routing_confs)
    pred_confs         = np.array(pred_confs)
    corrects           = np.array(corrects)
    assigned_centroids = np.array(assigned_centroids)
    margins             = np.array(margins)
    n_total             = len(corrects)

    if n_total == 0:
        raise RuntimeError("calibration_analysis: no valid samples at all "
                            "(the memory bank may never have filled across "
                            "the whole test set).")

    sample_groups = getattr(getattr(model, "prototype_layer", None), "sample_groups", None)
    if sample_groups is not None:
        centroid_sizes = np.array([len(sample_groups[c]) for c in assigned_centroids])
    else:
        centroid_sizes = np.full(n_total, -1)  # before sample_groups is cached (abnormal)

    n_prototypes = getattr(getattr(model, "prototype_layer", None), "P", None)

    # N_eff = exp(H(assignment distribution)). "How many centroids are alive"
    # and "how many carry traffic evenly" are different quantities, confirmed
    # repeatedly in measurement (alive = 139 while a handful took more than
    # half the test traffic). A uniform distribution gives N_eff = P, total
    # concentration on one centroid gives N_eff = 1, so it summarises in one
    # number how many prototypes are effectively working.
    # test_n_eff uses the distribution this run's test samples actually
    # reached; train_n_eff uses the sample_groups size distribution over the
    # whole training split, which has far more samples and is the more stable
    # of the two.
    def _n_eff(counts: np.ndarray) -> float:
        counts = counts[counts > 0]
        if counts.sum() == 0:
            return 0.0
        p = counts / counts.sum()
        h = -(p * np.log(p + 1e-12)).sum()
        return float(np.exp(h))

    _, test_counts = np.unique(assigned_centroids, return_counts=True)
    test_n_eff = _n_eff(test_counts)
    train_n_eff = None
    if sample_groups is not None:
        train_counts = np.array([len(g) for g in sample_groups if g])
        train_n_eff = _n_eff(train_counts)

    routing_stats = {
        "mean":   float(routing_confs.mean()),
        "median": float(np.median(routing_confs)),
        "std":    float(routing_confs.std()),
        "min":    float(routing_confs.min()),
        "max":    float(routing_confs.max()),
        "p90":    float(np.percentile(routing_confs, 90)),
        "p99":    float(np.percentile(routing_confs, 99)),
        "n_prototypes": n_prototypes,
        "uniform_baseline": (1.0 / n_prototypes) if n_prototypes else None,
        "test_n_eff":  test_n_eff,
        "train_n_eff": train_n_eff,
    }

    # Spearman correlations between routing and prediction confidence, and
    # between centroid_size and accuracy, so a hypothesis such as "larger
    # centroids are less accurate" comes with a number rather than a claim.
    # Spearman because `correct` is binary 0/1, where a rank-based measure
    # distorts less than Pearson (its interpretation is close to
    # point-biserial), and because it still detects a monotone but non-linear
    # relation between the two confidences.
    from scipy.stats import spearmanr
    corr_routing_vs_pred, _      = spearmanr(routing_confs, pred_confs)
    corr_routing_vs_correct, _   = spearmanr(routing_confs, corrects)
    corr_margin_vs_correct, _    = spearmanr(margins, corrects)
    corr_centroidsize_vs_correct, _ = (
        spearmanr(centroid_sizes, corrects) if sample_groups is not None else (float("nan"), None)
    )
    correlations = {
        "routing_vs_prediction_confidence": float(corr_routing_vs_pred),
        "routing_vs_correct":               float(corr_routing_vs_correct),
        "routing_margin_vs_correct":        float(corr_margin_vs_correct),
        "centroid_size_vs_correct":         float(corr_centroidsize_vs_correct),
    }

    # Join the training-set centroid purity and cohesion (from
    # compute_centroid_train_stats) onto the samples, and correlate at the
    # centroid level. Skipped entirely when X_train / y_train are absent, so
    # existing callers that omit them keep working.
    # This tests the hypothesis "larger centroid -> lower purity -> worse
    # prediction" from both directions: per sample (centroid_purity against
    # correct) and per centroid (size against purity, purity against
    # test_accuracy). It implements the three-stage analysis of centroid
    # statistics, then centroid-level correlation, then sample-level
    # correlation.
    centroid_train_stats = {}
    centroid_level_correlations = {}
    centroid_table = []
    centroid_purities  = np.full(n_total, np.nan)
    centroid_cohesions = np.full(n_total, np.nan)

    if X_train is not None and y_train is not None:
        centroid_train_stats = compute_centroid_train_stats(
            model, X_train, y_train, tasktype, class_names=class_names
        )
        for i, c in enumerate(assigned_centroids):
            st = centroid_train_stats.get(int(c))
            if st is not None:
                if st["purity"] is not None:
                    centroid_purities[i] = st["purity"]
                centroid_cohesions[i] = st["cohesion"]

        _valid_purity = ~np.isnan(centroid_purities)
        if _valid_purity.sum() >= 2:
            corr_purity_vs_correct, _ = spearmanr(centroid_purities[_valid_purity], corrects[_valid_purity])
            correlations["centroid_purity_vs_correct"] = float(corr_purity_vs_correct)
        _valid_cohesion = ~np.isnan(centroid_cohesions)
        if _valid_cohesion.sum() >= 2:
            corr_cohesion_vs_correct, _ = spearmanr(centroid_cohesions[_valid_cohesion], corrects[_valid_cohesion])
            correlations["centroid_cohesion_vs_correct"] = float(corr_cohesion_vs_correct)

        # Centroid-level correlation: one value per centroid rather than per
        # sample. test_accuracy is the mean accuracy of the test samples
        # assigned to that centroid.
        _centroid_ids  = sorted(centroid_train_stats.keys())

        centroid_table = []
        for c in _centroid_ids:
            mask = (assigned_centroids == c)
            st = centroid_train_stats[c]
            centroid_table.append({
                "centroid": c, "train_count": st["size"], "test_count": int(mask.sum()),
                "purity": st["purity"], "entropy": st["entropy"], "cohesion": st["cohesion"],
                "test_accuracy": float(corrects[mask].mean()) if mask.sum() > 0 else None,
            })

        _sizes, _purities, _cohesions, _test_accs = [], [], [], []
        for c in _centroid_ids:
            mask = (assigned_centroids == c)
            if mask.sum() == 0:
                continue  # no test sample assigned here: test_accuracy is undefined
            st = centroid_train_stats[c]
            _sizes.append(st["size"])
            _purities.append(st["purity"] if st["purity"] is not None else np.nan)
            _cohesions.append(st["cohesion"])
            _test_accs.append(float(corrects[mask].mean()))
        _sizes, _purities, _cohesions, _test_accs = map(np.array, (_sizes, _purities, _cohesions, _test_accs))

        if len(_sizes) >= 2:
            _valid = ~np.isnan(_purities)
            if _valid.sum() >= 2:
                r, _ = spearmanr(_sizes[_valid], _purities[_valid])
                centroid_level_correlations["size_vs_purity"] = float(r)
                r, _ = spearmanr(_purities[_valid], _test_accs[_valid])
                centroid_level_correlations["purity_vs_test_accuracy"] = float(r)
            r, _ = spearmanr(_cohesions, _test_accs)
            centroid_level_correlations["cohesion_vs_test_accuracy"] = float(r)
            centroid_level_correlations["n_centroids"] = int(len(_sizes))


    def _fixed_bin_stats(confs, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # Only the last bin includes hi (<=); the rest are [lo, hi), so
            # that 100% is not dropped from the final bin.
            if hi == edges[-1]:
                mask = (confs >= lo) & (confs <= hi)
            else:
                mask = (confs >= lo) & (confs < hi)
            n = int(mask.sum())
            acc = float(corrects[mask].mean()) if n > 0 else float("nan")
            mean_conf = float(confs[mask].mean()) if n > 0 else float("nan")
            rows.append({"lo": lo, "hi": hi, "n": n, "acc": acc, "mean_conf": mean_conf})
        return rows

    def _percentile_bin_stats(confs, n_bins):
        # Bin edges come from percentiles. Where many values coincide the
        # edges can collide and leave a bin with n = 0. That is itself
        # information about how concentrated the distribution is, so it is
        # left uncorrected.
        pct_edges = np.linspace(0, 100, n_bins + 1)
        conf_edges = np.percentile(confs, pct_edges)
        rows = []
        for i in range(n_bins):
            lo_pct, hi_pct = pct_edges[i], pct_edges[i + 1]
            lo_conf, hi_conf = conf_edges[i], conf_edges[i + 1]
            if i == n_bins - 1:
                mask = (confs >= lo_conf) & (confs <= hi_conf)
            else:
                mask = (confs >= lo_conf) & (confs < hi_conf)
            n = int(mask.sum())
            acc = float(corrects[mask].mean()) if n > 0 else float("nan")
            mean_centroid_size = (
                float(centroid_sizes[mask].mean())
                if n > 0 and sample_groups is not None else None
            )
            rows.append({"lo_pct": lo_pct, "hi_pct": hi_pct,
                         "lo_conf": float(lo_conf), "hi_conf": float(hi_conf),
                         "n": n, "acc": acc, "mean_centroid_size": mean_centroid_size})
        return rows

    routing_bins    = _percentile_bin_stats(routing_confs, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prediction_bins = _fixed_bin_stats(pred_confs, edges)

    # ECE: the bin-size-weighted mean of |accuracy - mean_confidence| per bin
    # (the standard definition from Guo et al. 2017, "On Calibration of Modern
    # Neural Networks")
    ece = sum(
        (b["n"] / n_total) * abs(b["acc"] - b["mean_conf"])
        for b in prediction_bins if b["n"] > 0
    )

    return {
        "routing_stats":    routing_stats,
        "routing_bins":     routing_bins,
        "prediction_bins":  prediction_bins,
        "prediction_ece":   float(ece),
        "n_total":          n_total,
        "overall_acc":      float(corrects.mean()),
        "correlations":     correlations,
        "centroid_train_stats":         centroid_train_stats,        # {centroid_idx: {size,purity,cohesion,gap,entropy}}
        "centroid_table":               centroid_table,  # [{centroid,train_count,test_count,...}]
        "centroid_level_correlations":  centroid_level_correlations,  # size_vs_purity etc.
        # Raw per-sample arrays, so a scatter plot or a further correlation
        # can be done without recomputing anything. They show patterns the bin
        # statistics hide -- whether a handful of centroids are the problem or
        # the effect is general.
        "per_sample": {
            "routing_confidence":    routing_confs.tolist(),
            "routing_margin":        margins.tolist(),
            "prediction_confidence": pred_confs.tolist(),
            "assigned_centroid":     assigned_centroids.tolist(),
            "centroid_size":         centroid_sizes.tolist(),
            "centroid_purity":       centroid_purities.tolist(),   # training set; needs X_train
            "centroid_cohesion":     centroid_cohesions.tolist(),  # same condition as above
            "correct":               corrects.tolist(),
        },
    }


def print_calibration_analysis(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Calibration Analysis (test set, n={result['n_total']})")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {result['overall_acc']:.1%}")
    print(f"  Prediction ECE (Expected Calibration Error): {result['prediction_ece']:.4f}")
    print(f"  (a lower ECE means confidence matches how often it is actually right;")
    print(f"   under 0.05 is generally good, over 0.15 is markedly over/underconfident)")

    rs = result["routing_stats"]
    print(f"\n  Routing confidence distribution (n_prototypes={rs['n_prototypes']}, "
          f"uniform baseline={rs['uniform_baseline']:.2%}; far above it means routing "
          f"concentrates on particular centroids, close to it means near-uniform):")
    print(f"    mean={rs['mean']:.2%}  median={rs['median']:.2%}  std={rs['std']:.2%}  "
          f"min={rs['min']:.2%}  max={rs['max']:.2%}  p90={rs['p90']:.2%}  p99={rs['p99']:.2%}")
    print(f"  (binned by percentile, not absolute %: routing_confidence depends")
    print(f"   structurally on n_prototypes, so fixed % bins cannot compare across P)")

    print(f"\n  Effective prototype count (N_eff = exp(entropy); {rs['n_prototypes']} if uniform, "
          f"1 if all traffic goes to one centroid):")
    if rs.get("train_n_eff") is not None:
        print(f"    train N_eff = {rs['train_n_eff']:.1f}  (over the full training distribution)")
    print(f"    test  N_eff = {rs['test_n_eff']:.1f}  (over what this run's test set reached)")
    print(f"  (different from the 'alive' centroid count in the [Regroup] log: alive means")
    print(f"   'not dead', N_eff means 'how evenly traffic is shared'. A large alive count")
    print(f"   with a much smaller N_eff means most of them sit idle.)")

    print(f"\n  {'Routing confidence (percentile)':<34s}{'conf range':<20s}{'n':>6s}{'accuracy':>10s}{'mean centroid_size':>20s}")
    for b in result["routing_bins"]:
        acc_str = f"{b['acc']:.1%}" if b["n"] > 0 else "  n/a"
        range_str = f"{b['lo_conf']:.2%}-{b['hi_conf']:.2%}"
        size_str = f"{b['mean_centroid_size']:.0f}" if b["n"] > 0 and b["mean_centroid_size"] is not None else "  n/a"
        print(f"  {b['lo_pct']:>3.0f}–{b['hi_pct']:>3.0f}pct{'':<20s}{range_str:<20s}{b['n']:>6d}{acc_str:>10s}{size_str:>20s}")

    print(f"\n  {'Prediction confidence':<24s}{'n':>8s}{'accuracy':>12s}{'mean conf':>12s}")
    for b in result["prediction_bins"]:
        lo_pct, hi_pct = int(b["lo"] * 100), int(b["hi"] * 100)
        acc_str  = f"{b['acc']:.1%}" if b["n"] > 0 else "  n/a"
        conf_str = f"{b['mean_conf']:.1%}" if b["n"] > 0 else "  n/a"
        print(f"  {lo_pct:>3d}–{hi_pct:>3d}%{'':<16s}{b['n']:>8d}{acc_str:>12s}{conf_str:>12s}")

    corr = result["correlations"]
    print(f"\n  Spearman correlations (whole test set, per sample):")
    print(f"    routing_confidence vs prediction_confidence : {corr['routing_vs_prediction_confidence']:+.3f}")
    print(f"    routing_confidence vs correct(0/1)          : {corr['routing_vs_correct']:+.3f}")
    print(f"    routing_margin(top1-runnerup1) vs correct   : {corr['routing_margin_vs_correct']:+.3f}")
    print(f"    centroid_size vs correct(0/1)               : {corr['centroid_size_vs_correct']:+.3f}")
    if "centroid_purity_vs_correct" in corr:
        print(f"    centroid_purity(train) vs correct(0/1)      : {corr['centroid_purity_vs_correct']:+.3f}")
    if "centroid_cohesion_vs_correct" in corr:
        print(f"    centroid_cohesion(train) vs correct(0/1)    : {corr['centroid_cohesion_vs_correct']:+.3f}")
    print(f"  (a clearly negative routing_confidence vs correct means the more confident")
    print(f"   routing is, the more it errs; a clear centroid_size/purity/cohesion vs correct")
    print(f"   means that centroid property relates to failure -- though one coefficient")
    print(f"   cannot establish causation, so plot the per_sample arrays directly.)")

    ct = result.get("centroid_table", [])
    if ct:
        n_zero_test = sum(1 for r in ct if r["test_count"] == 0)
        print(f"\n  Train vs test usage per centroid ({len(ct)} centroids have train samples, "
              f"of which {n_zero_test} received no test sample):")
        print(f"  [note] many centroids with test_count=0 does not by itself mean they died:")
        print(f"   the task may simply concentrate in a few regions. Compare train_count too")
        print(f"   -- i.e. check whether they were barely used during training either.")
        _top = sorted(ct, key=lambda r: -r["train_count"])[:15]
        print(f"\n  {'Centroid':<10}{'train_n':>9}{'test_n':>8}{'purity':>9}{'entropy':>9}{'cohesion':>10}{'test_acc':>10}")
        for r in _top:
            purity_str = f"{r['purity']:.1%}" if r['purity'] is not None else "  n/a"
            entropy_str = f"{r['entropy']:.3f}" if r['entropy'] is not None else "  n/a"
            acc_str = f"{r['test_accuracy']:.1%}" if r['test_accuracy'] is not None else "  n/a"
            print(f"  Centroid_{r['centroid']:<4}{r['train_count']:>9}{r['test_count']:>8}"
                  f"{purity_str:>9}{entropy_str:>9}{r['cohesion']:>10.4f}{acc_str:>10}")
        print(f"  (top 15 by train_count; the full table is in result['centroid_table'])")

    clc = result.get("centroid_level_correlations", {})
    if clc:
        print(f"\n  Spearman correlations (per centroid, one value each, n_centroids={clc.get('n_centroids', '?')}):")
        print(f"  [note] with few centroids (under 10, say) these coefficients rest on a very")
        print(f"   small sample and have wide intervals -- do not over-read extreme values.")
        if "size_vs_purity" in clc:
            print(f"    size vs purity            : {clc['size_vs_purity']:+.3f}  "
                  f"(negative means larger centroids are less pure)")
        if "purity_vs_test_accuracy" in clc:
            print(f"    purity vs test_accuracy   : {clc['purity_vs_test_accuracy']:+.3f}  "
                  f"(positive means purer centroids also do better on test)")
        if "cohesion_vs_test_accuracy" in clc:
            print(f"    cohesion vs test_accuracy : {clc['cohesion_vs_test_accuracy']:+.3f}")
        print(f"  (if all three point the expected way -- size vs purity negative, purity and")
        print(f"   cohesion vs accuracy positive -- the size/purity/failure path holds per centroid)")

    accs = [b["acc"] for b in result["routing_bins"] if b["n"] > 0 and not np.isnan(b["acc"])]
    is_monotonic_nondecreasing = all(a <= b + 0.03 for a, b in zip(accs, accs[1:]))  # 3%p slack
    max_drop = max((accs[i] - accs[i+1] for i in range(len(accs)-1)), default=0.0)

    print(f"\n  Reading:")
    if is_monotonic_nondecreasing and max_drop < 0.05:
        print(f"    - Accuracy across routing-confidence percentile bins is flat or monotonically")
        print(f"      increasing, consistent with retrieval/fusion compensating for routing's")
        print(f"      relative uncertainty (though one metric does not prove causation).")
    else:
        print(f"    - Accuracy across routing-confidence percentile bins is not monotone (largest "
              f"drop {max_drop:.1%}p). The cause cannot be read off this alone; possibilities:")
        print(f"        1) top-percentile samples concentrate in particular centroids, usually")
        print(f"           large and impure -> check the mean centroid_size column and the")
        print(f"        2) routing and the final prediction look at different information")
        print(f"           -> check the routing_confidence vs prediction_confidence correlation")
        print(f"        3) chance in this one test set or seed (sample size, training noise)")
        print(f"           -> check whether it reproduces under a different --train_seed")
        print(f"      Do not settle on one of these -- dig into the per_sample arrays.")
    print(f"    - Accuracy in a prediction-confidence bin is clearly below mean_conf")
    print(f"      (especially the 80-100% bin) -> overconfidence, i.e. poor calibration.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────



REFINE_V1 = dict(lr=1e-2, wd=0.0, epochs=500, patience=50, null=True)


def _readout_refine_tag(args) -> str:
    """Filename tag for the post-training readout refinement.

    The final benchmark uses exactly one setting (REFINE_V1), tagged
    ``..readoutRefineV1``. Any other combination gets its values spelled out
    so two settings can never write to the same file -- the CLI exposes
    --refine_lr / --refine_wd / --refine_epochs / --refine_patience /
    --refine_no_null, and without this an lr=1e-2 run and an lr=1e-3 run
    would overwrite each other.
    """
    if not getattr(args, "final_readout_refine", False):
        return ""
    cur = dict(lr=args.refine_lr, wd=args.refine_wd, epochs=args.refine_epochs,
               patience=args.refine_patience, null=not args.refine_no_null)
    if cur == REFINE_V1:
        return "..readoutRefineV1"
    t = "..readoutRefine"
    if cur["lr"] != REFINE_V1["lr"]:            t += f"_lr{cur['lr']:g}"
    if cur["wd"] != REFINE_V1["wd"]:            t += f"_wd{cur['wd']:g}"
    if cur["epochs"] != REFINE_V1["epochs"]:    t += f"_ep{cur['epochs']}"
    if cur["patience"] != REFINE_V1["patience"]: t += f"_pat{cur['patience']}"
    if not cur["null"]:                          t += "_noNull"
    return t


def run_single_seed(
    dataset, X_train, y_train, X_val, y_val, X_test, y_test, y_std,
    output_dim, tasktype, openml_id, dataset_info, device, log_dir, env_info,
    args, train_seed, do_analysis, study_dir=None,
):
    """Train, evaluate and optionally analyse for one train_seed, given a
    dataset and HPO study that are both independent of train_seed and loaded
    once in main().

    optimize.py loads the dataset once and reuses it across 100 trials. This
    file used to reload it in every process run -- once per seed -- paying the
    OpenML fetch, NaN preprocessing, KFold and QuantileTransformer
    cost five times for --train_seeds with five values. That logic (about
    2,400 lines inline in main()) moved here unchanged, so main() loads the
    dataset and study once and calls this function per seed, the same pattern
    as optimize.py.

    do_analysis : whether whichever of --explain / --calibration_analysis /
      --linear_probe are enabled should actually run for this seed. With
      several --train_seeds, leaving them all on multiplies the log by the
      number of seeds, so main() decides this by comparing against
      --explain_seed (default: the last seed).

    Returns {"train_seed": train_seed, "val_metrics": dict,
             "test_metrics": dict}. With two or more --train_seeds, main()
    collects these and prints a mean +- std summary.
    """
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if len(getattr(args, '_train_seed_list', [train_seed])) > 1 or train_seed != args.seed:
        print(f"  [train_seed={train_seed}] seeds init and batch order (the data split still uses --seed={args.seed})")

    # ── Output identity ─────────────────────────────────────
    # Any setting that changes the trained model or selected checkpoint must be
    # represented in the output tag. Structural settings restored from a saved
    # checkpoint are appended after the checkpoint configuration is known.
    actual_geometry = args.correction_geometry
    actual_beta_param = args.beta_param
    actual_head_input_scale = args.head_input_scale
    _save_tag = (f"..k{args.k_override}" if args.k_override is not None else "") \
              + ("..allowSelfRet" if args.allow_self_retrieval else "") \
              + ("..cat_concat" if args.cat_combine == "concat" else "") \
              + ("..cat_onehot" if args.cat_combine == "onehot" else "") \
              + ("..num_ple" if args.num_embedding == "ple" else "") \
              + ("..num_plr" if args.num_embedding == "plr_lite" else "") \
              + (f"..ed{args.embed_dim_override}" if args.embed_dim_override is not None else "") \
              + (f"..do{args.dropout_override:g}" if args.dropout_override is not None else "") \
              + (f"..bs{args.batch_size_override}" if args.batch_size_override is not None else "") \
              + (f"..rwe{args.regroup_warmup_epochs_override}" if args.regroup_warmup_epochs_override is not None else "") \
              + ("..defersel" if args.defer_early_stopping else "") \
              + (f"..me{args.min_epochs}" if args.min_epochs else "") \
              + ("..nodr" if args.disable_dead_reinit else "") \
              + (f"..drp{args.dead_reinit_patience_override}" if args.dead_reinit_patience_override is not None else "") \
              + (f"..drn{args.dead_reinit_noise_scale_override:g}" if args.dead_reinit_noise_scale_override is not None else "") \
              + _readout_refine_tag(args) \
              + (f"..trainseed{train_seed}" if train_seed != args.seed else "") \
              + ("..deterministic" if args.deterministic else "") \
              + (f"..{args.run_tag}" if args.run_tag is not None else "")
    # run_tag goes straight into the filename, so it is validated.
    # Why: PowerShell variables are **case-insensitive**. In a script using
    # both $S (a checkpoint path) and $s (a seed), `--run_tag stab_s$s`
    # expanded to `stab_s` plus the whole path and died at the np.save step
    # with OSError(Errno 22) -- after training and inference had finished.
    # Minutes of computation thrown away, so it is caught at startup.
    # so it has to be caught at startup.
    if args.run_tag is not None:
        _bad = [c for c in ('/', '\\', ':', '=', '*', '?', '"', '<', '>', '|') if c in args.run_tag]
        if _bad or len(args.run_tag) > 64:
            raise SystemExit(
                f"--run_tag cannot be used in a filename: {args.run_tag!r}\n"
                + (f"  contains forbidden characters: {_bad}\n" if _bad else "")
                + (f"  length {len(args.run_tag)} > 64\n" if len(args.run_tag) > 64 else "")
                + "  PowerShell variables are case-insensitive: using $S (a path) "
                  "and $s (a seed)\n  together makes them one variable, so the path "
                  "leaks into the tag. Use distinct names.")

    _saved_state = None
    if args.from_saved_state:
        # ── --from_saved_state: no study file needed; the saved model_kwargs
        # are used as-is. Training is skipped, so --epochs and --patience are
        # ignored.
        print(f"  [--from_saved_state] loading {args.from_saved_state} (skipping training)")
        # From PyTorch 2.6, torch.load() defaults to weights_only=True and
        # rejects the custom classes inside sample_groups and group_labels
        # (FeatureLabel and friends) as not being on the allow-list. This file
        # weights_only=False is stated explicitly.
        _saved_state = torch.load(args.from_saved_state, map_location=device, weights_only=False)
        model_kwargs = dict(_saved_state["model_kwargs"])
        _key_cfg = {k: model_kwargs.get(k) for k in
                    ("k", "embed_dim", "n_prototypes", "correction_geometry")
                    if k in model_kwargs}
        print(f"  [from_saved_state] actual checkpoint settings: "
              + ", ".join(f"{k}={v}" for k, v in _key_cfg.items()))
        # ⚠ The model comes from the checkpoint, so the checkpoint decides the
        #   geometry -- filenames and meta.pkl must follow the model, not the
        #   CLI. Without this a tangent checkpoint reopened without the flag
        #   would write its results under the additive name and record
        #   "additive" in freeze_deviations. That is the failure the comment
        #   above describes, applied to a structural variable.
        # gamma is non-persistent, so structural settings must be restored from
        # checkpoint model_kwargs rather than inferred from state_dict loading.
        actual_beta_param = model_kwargs.get("beta_param", "sigmoid")
        actual_head_input_scale = model_kwargs.get("head_input_scale", "unit")
        for _nm, _got, _cli in (("beta_param", actual_beta_param, args.beta_param),
                                ("head_input_scale", actual_head_input_scale,
                                 args.head_input_scale)):
            if _got != _cli:
                print(f"  [from_saved_state] --{_nm} {_cli} ignored; using "
                      f"{_got} from the checkpoint.")
        actual_geometry = model_kwargs.get("correction_geometry", "additive")
        if actual_geometry != args.correction_geometry:
            print(f"  [from_saved_state] --correction_geometry "
                  f"{args.correction_geometry} ignored; using "
                  f"{actual_geometry} from the checkpoint.")
        best_params  = _saved_state.get("best_params", {})
        if best_params:
            print(f"  Params (as saved): {best_params}")
        # Compatibility fallback for checkpoints that store n_train but not memory_size.
        if "memory_size" not in model_kwargs:
            fallback_size = _saved_state.get("n_train")
            if fallback_size is not None:
                model_kwargs = {**model_kwargs, "memory_size": fallback_size}
                print(f"  !  old-format file (no memory_size): substituting n_train={fallback_size}."
                      f" sample_groups may also be missing; check the warnings below.")
        if args.batch_size_override is not None:
            print(f"  ⚠️  --batch_size_override only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.regroup_warmup_epochs_override is not None:
            print(f"  ⚠️  --regroup_warmup_epochs_override only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.defer_early_stopping or args.min_epochs:
            print(f"  ⚠️  --defer_early_stopping / --min_epochs only apply when "
                  f"retraining; --from_saved_state skips training, so they are "
                  f"ignored (the checkpoint already fixes which epoch was "
                  f"selected).")
        if args.dead_reinit_patience_override is not None:
            print(f"  ⚠️  --dead_reinit_patience_override only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.dead_reinit_noise_scale_override is not None:
            print(f"  ⚠️  --dead_reinit_noise_scale_override only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.dropout_override is not None:
            print(f"  ⚠️  --dropout_override only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.train_seed is not None:
            print(f"  ⚠️  --train_seed only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")
        if args.deterministic:
            print(f"  ⚠️  --deterministic only applies when retraining; "
                  f"--from_saved_state skips training, so the flag is ignored.")

    else:
        # ── Load the study ─────────────────────────────────────
        # ⚠ Must use the **same study_pkl_tag()** as optimize.py to point at
        #   the same file. If the arguments change on one side only, this
        #   fails with a silent FileNotFoundError -- which is exactly what
        #   happened when the --num_embedding default changed.
        _study_tag = study_pkl_tag(
            cat_combine=args.cat_combine,
            num_embedding=args.num_embedding,
            n_prototypes=args.n_prototypes,
            disable_dead_reinit=args.disable_dead_reinit,
            num_bins=args.num_bins,
            cat_embed_dim=args.cat_embed_dim,
            # Only set for studies explicitly tagged with a batch-size override.
            batch_size=args.study_batch_size,
            # params_geometry selects the source study only; the target geometry
            # is applied to model_kwargs below.
            correction_geometry=(args.params_geometry
                                 if args.params_geometry is not None
                                 else args.correction_geometry),
            # Source-study configuration; target structure is applied separately.
            beta_param=("sigmoid" if args.params_variant == "legacy"
                        else args.beta_param),
            # matched is a target-only intervention and always reads a unit-scale source study.
            head_input_scale=("unit" if (args.params_variant == "legacy"
                                         or args.head_input_scale == "matched")
                              else args.head_input_scale),
            tie_rule=("first" if args.params_variant == "legacy"
                      else args.tie_rule),
            early_stop_metric=args.early_stop_metric,
        )
        # params_seed changes only the HPO-study source; outputs remain under the current seed.
        _study_dir = study_dir or log_dir
        fname = os.path.join(_study_dir, f"data={openml_id}{_study_tag}..model=tabera.pkl")
        if not os.path.exists(fname):
            _hint_flags = ""
            if args.num_embedding != "ple":
                _hint_flags += f" --num_embedding {args.num_embedding}"
            if args.cat_combine != "onehot":
                _hint_flags += f" --cat_combine {args.cat_combine}"
            if args.n_prototypes is not None:
                _hint_flags += f" --n_prototypes {args.n_prototypes}"
            if args.disable_dead_reinit:
                _hint_flags += " --disable_dead_reinit"
            # Without this the suggested command creates an *additive* study,
            # and the next reproduce run stops on the arm-mismatch guard.
            if args.correction_geometry != "additive":
                _hint_flags += f" --correction_geometry {args.correction_geometry}"
            if args.early_stop_metric != "accuracy":
                _hint_flags += f" --early_stop_metric {args.early_stop_metric}"
            # The hint reconstructs the source-study command, not the target structure.
            if args.params_variant is None:
                if args.beta_param != "sigmoid":
                    _hint_flags += f" --beta_param {args.beta_param}"
                # matched is reproduce-only; its source study uses unit scaling.
                if args.head_input_scale not in ("unit", "matched"):
                    _hint_flags += f" --head_input_scale {args.head_input_scale}"
            _hint_cmd = f"optimize.py --openml_id {openml_id} --seed {args.seed}{_hint_flags}"
            raise FileNotFoundError(
                f"no optimisation log at: {fname}\n"
                f"run {_hint_cmd} first."
            )

        study       = joblib.load(fname)
        best_params = study.best_params
        print(f"  Best trial #{study.best_trial.number}  val={study.best_value:.4f}")
        # P follows the sqrt(N) rule and is not in best_params, so
        # reproduction requires the actual value optimize.py stored in
        # user_attrs.
        best_params["n_prototypes"] = study.best_trial.user_attrs["n_prototypes_actual"]
        print(f"  n_prototypes (from optimize.py): {best_params['n_prototypes']}")
        # batch_size is stored as a trial attribute rather than in best_params.
        # Reproduce the recorded value; fall back to the same MultiTab batch rule.
        _bs_actual = study.best_trial.user_attrs.get("batch_size_actual")
        if _bs_actual is None:
            _bs_actual = get_batch_size(len(X_train))
            print(f"  !  study has no batch_size_actual; "
                  f"recomputed with get_batch_size(len(X_train))={_bs_actual}.")
        best_params.setdefault("batch_size", int(_bs_actual))
        # correction_geometry is also stored as a trial attribute, so restore it
        # explicitly to keep HPO and reproduction structurally aligned.
        _geom_actual = study.best_trial.user_attrs.get(
            "correction_geometry_actual", "additive")
        _geom_expected = (args.params_geometry if args.params_geometry is not None
                          else args.correction_geometry)
        if _geom_actual != _geom_expected:
            raise SystemExit(
                f"\n[stopped] The study was built with correction_geometry="
                f"'{_geom_actual}', but this run expected '{_geom_expected}'.\n"
                f"  file: {fname}\n"
                f"  Reproducing the tuned configuration under a different "
                f"geometry trains a different model than the one HPO selected."
                f"\n  -> pass --correction_geometry {_geom_actual}, or "
                f"--params_geometry {_geom_actual} to borrow its "
                f"hyperparameters on purpose.")
        if args.params_geometry is not None:
            # Controlled intervention: hyperparameters from one arm's study,
            # geometry from the CLI. Everything downstream names the run after
            # the geometry that is actually trained.
            print(f"  [params_geometry] hyperparameters from the "
                  f"'{_geom_actual}' study; training with "
                  f"correction_geometry='{args.correction_geometry}'.")
        best_params["correction_geometry"] = args.correction_geometry
        actual_geometry = args.correction_geometry
        actual_beta_param = args.beta_param
        actual_head_input_scale = args.head_input_scale
        if args.params_variant == "legacy":
            print(f"  [params_variant=legacy] hyperparameters from a "
                  f"sigmoid/unit/first study; training with "
                  f"beta_param={args.beta_param}, "
                  f"head_input_scale={args.head_input_scale}, "
                  f"tie_rule={args.tie_rule}.")
        # Reuse the selection metric recorded by the study.
        _esm_actual = study.best_trial.user_attrs.get(
            "early_stop_metric_actual", "accuracy")
        if _esm_actual != args.early_stop_metric:
            raise SystemExit(
                f"\n[stopped] The study was built with early_stop_metric="
                f"'{_esm_actual}', but this run was given "
                f"'{args.early_stop_metric}'.\n"
                f"  file: {fname}\n"
                f"  The selection metric decides which checkpoint the run "
                f"returns; reproducing under a different one yields a "
                f"different model than HPO selected.\n"
                f"  -> pass --early_stop_metric {_esm_actual}.")
        print(f"  early_stop_metric (from optimize.py): {_esm_actual}")
        print(f"  correction_geometry (from optimize.py): {_geom_actual}")
        print(f"  Params: {best_params}")
        # P and batch_size are fold-derived protocol values, so verify them when
        # hyperparameters are borrowed from another seed.
        if study_dir is not None and study_dir != log_dir:
            _P_rule = int(len(X_train) ** 0.5)
            _B_rule = int(get_batch_size(len(X_train)))
            _P_used, _B_used = int(best_params["n_prototypes"]), int(best_params["batch_size"])
            print(f"  [params_seed] protocol rule for this fold: "
                  f"P={_P_rule} B={_B_rule}  |  loaded: P={_P_used} B={_B_used}")
            if (_P_used, _B_used) != (_P_rule, _B_rule) and not args.allow_pb_mismatch:
                raise RuntimeError(
                    f"[params_seed] P/batch_size mismatch: loaded (P={_P_used}, "
                    f"B={_B_used}) vs this fold's protocol rule (P={_P_rule}, "
                    f"B={_B_rule}). Both values are derived from N_train rather than "
                    f"selected by HPO. Use --allow_pb_mismatch to override this check.")

        # PLE bin edges come from the train split only, to avoid leakage.
        num_bin_edges = None
        if args.num_embedding == "ple" and len(dataset.X_num) > 0:
            X_num_train = X_train[:, dataset.X_num]
            q = torch.linspace(0.0, 1.0, args.num_bins + 1, device=X_num_train.device)
            num_bin_edges = torch.quantile(X_num_train, q, dim=0).T.contiguous()

        # ── Build the model ────────────────────────────────────
        model_kwargs = params_to_model_kwargs(best_params, dataset.n_features, output_dim)

        # ── Structural target override (MUST be after params_to_model_kwargs) ──
        # The source study supplies non-structural hyperparameters. Apply the target
        # structure only after params_to_model_kwargs(); tie_rule belongs to wrapper
        # checkpoint selection rather than the TabERA architecture.
        model_kwargs["correction_geometry"] = args.correction_geometry
        model_kwargs["beta_param"] = args.beta_param
        model_kwargs["head_input_scale"] = args.head_input_scale

        # ── Overrides applied to the retrained model ────────────────────
        #   filename still picked up their tags, but nothing reached
        #   model_kwargs -- so a run with --disable_dead_reinit produced a
        #   file named "..nodr.." whose contents were byte-identical to the
        #   default run. Three conditions that looked like an ablation were
        #   the same experiment. Any new override must be wired here, not
        #   only into the tag.
        if args.k_override is not None:
            print(f"  [--k_override] k: {model_kwargs.get('k')} -> {args.k_override}")
            model_kwargs["k"] = args.k_override
        if args.embed_dim_override is not None:
            print(f"  [--embed_dim_override] embed_dim: "
                  f"{model_kwargs.get('embed_dim')} -> {args.embed_dim_override}")
            model_kwargs["embed_dim"] = args.embed_dim_override
        if args.dropout_override is not None:
            print(f"  [--dropout_override] dropout: "
                  f"{model_kwargs.get('dropout')} -> {args.dropout_override}")
            model_kwargs["dropout"] = args.dropout_override
        if args.regroup_warmup_epochs_override is not None:
            print(f"  [--regroup_warmup_epochs_override] regroup_warmup_epochs: "
                  f"{model_kwargs.get('regroup_warmup_epochs', 0)} -> "
                  f"{args.regroup_warmup_epochs_override}")
            model_kwargs["regroup_warmup_epochs"] = args.regroup_warmup_epochs_override
        if args.disable_dead_reinit:
            # A patience above any reachable epoch count switches recovery off
            # without adding a branch to CentroidLayer.
            model_kwargs["dead_reinit_patience"] = 10 ** 9
            print(f"  [--disable_dead_reinit] dead-prototype recovery off "
                  f"(patience=1e9, so no reinit event can fire)")
        if args.dead_reinit_patience_override is not None:
            _old_p = model_kwargs.get("dead_reinit_patience", 5)
            model_kwargs["dead_reinit_patience"] = args.dead_reinit_patience_override
            print(f"  [--dead_reinit_patience_override] dead_reinit_patience: "
                  f"{_old_p} -> {args.dead_reinit_patience_override}")
        if args.dead_reinit_noise_scale_override is not None:
            _old_n = model_kwargs.get("dead_reinit_noise_scale", 0.01)
            model_kwargs["dead_reinit_noise_scale"] = args.dead_reinit_noise_scale_override
            print(f"  [--dead_reinit_noise_scale_override] dead_reinit_noise_scale: "
                  f"{_old_n} -> {args.dead_reinit_noise_scale_override}")

    # ── Training provenance: HPO source vs actual trained configuration ──
    # Store the same provenance in checkpoints and meta files. Saved checkpoints
    # override CLI provenance when they are reopened for analysis.
    _training_provenance = {
        "correction_geometry_actual": actual_geometry,
        "beta_param_actual": actual_beta_param,
        "head_input_scale_actual": actual_head_input_scale,
        "tie_rule_actual": args.tie_rule,
        "early_stop_metric_actual": args.early_stop_metric,
        "params_geometry": args.params_geometry,
        "params_variant": args.params_variant,
        # Record the HPO-study seed in run identity even though it does not alter
        # the current data split or training RNG.
        "params_seed": args.params_seed,
        "params_correction_geometry_source": (args.params_geometry
                                              if args.params_geometry is not None
                                              else actual_geometry),
        "params_beta_param_source": ("sigmoid" if args.params_variant == "legacy"
                                     else actual_beta_param),
        # Keep provenance source-scale resolution identical to study lookup.
        "params_head_input_scale_source": (
            "unit" if (args.params_variant == "legacy"
                       or actual_head_input_scale == "matched")
            else actual_head_input_scale),
        "params_tie_rule_source": ("first" if args.params_variant == "legacy"
                                   else args.tie_rule),
    }
    if args.from_saved_state:
        _tp_ckpt = _saved_state.get("training_provenance")
        if _tp_ckpt:
            _training_provenance = dict(_tp_ckpt)
            _training_provenance["restored_from_checkpoint"] = True
        else:
            _training_provenance = {k: "unknown_legacy_assumed" for k in _training_provenance}
            _training_provenance["restored_from_checkpoint"] = False

    # ── Run identity comes from provenance, not from the current CLI ─────
    # This keeps analysis outputs tied to the configuration that produced the
    # checkpoint, including HPO source and checkpoint-selection settings.
    _actual_tie_rule = _training_provenance.get("tie_rule_actual", args.tie_rule)
    _actual_esm = _training_provenance.get("early_stop_metric_actual",
                                           args.early_stop_metric)
    _actual_params_seed = _training_provenance.get("params_seed", args.params_seed)
    _actual_params_geometry = _training_provenance.get("params_geometry",
                                                       args.params_geometry)
    _actual_params_variant = _training_provenance.get("params_variant",
                                                      args.params_variant)
    # Map unknown legacy provenance to stable filename defaults.
    if _actual_tie_rule == "unknown_legacy_assumed":        _actual_tie_rule = "first"
    if _actual_esm == "unknown_legacy_assumed":             _actual_esm = "accuracy"
    if _actual_params_seed == "unknown_legacy_assumed":     _actual_params_seed = None
    if _actual_params_geometry == "unknown_legacy_assumed": _actual_params_geometry = None
    if _actual_params_variant == "unknown_legacy_assumed":  _actual_params_variant = None

    # ── Now that the arm is known, name the outputs after it ──────────
    # Both branches above have set `actual_geometry`: from the study on the
    # training path, from the checkpoint's model_kwargs under
    # --from_saved_state. Everything downstream (filenames, meta.pkl,
    # reproduce_logs) reads this one variable, never args.correction_geometry.
    # Include HPO source overrides in filenames to prevent collisions.
    if _actual_params_seed is not None:
        _save_tag += f"..paramsseed={_actual_params_seed}"
    if _actual_params_geometry is not None:
        _save_tag += f"..paramsgeom={_actual_params_geometry}"
    if actual_geometry != "additive":
        _save_tag += f"..geom={actual_geometry}"
    # Saved-state analysis preserves the checkpoint's own run identity.
    if actual_beta_param != "sigmoid":
        _save_tag += f"..bp={actual_beta_param}"
    if actual_head_input_scale != "unit":
        _save_tag += f"..hs={actual_head_input_scale}"
    # tie_rule changes which checkpoint is returned and is part of run identity.
    if _actual_tie_rule != "first":
        _save_tag += f"..tie={_actual_tie_rule}"
    if _actual_params_variant is not None:
        _save_tag += f"..paramsvar={_actual_params_variant}"
    if _actual_esm != "accuracy":
        _save_tag += f"..esm={_actual_esm}"
    # beta_lr_mult changes optimization dynamics and therefore run identity.
    if args.beta_lr_mult != 1.0 and not args.from_saved_state:
        _save_tag += f"..blrm{args.beta_lr_mult:g}"

    # ⚠ memory_size must equal n_train. Left at the default (10000), the ring
    model_kwargs.update(dict(
        memory_size=len(y_train),
        exclude_self_retrieval=(not args.allow_self_retrieval),
        cat_col_idx=list(dataset.X_cat),
        num_col_idx=list(dataset.X_num),
        cat_cardinalities=list(dataset.X_cat_cardinality),
        cat_combine=args.cat_combine,
        cat_embed_dim=args.cat_embed_dim,
        num_embedding=args.num_embedding,
        num_bin_edges=num_bin_edges,
    ))

    # Neighbour label encoding needs the task type to choose between
    # nn.Embedding (classification) and nn.Linear (regression). It goes inside
    # model_kwargs so that it survives save and reload under
    # --from_saved_state (like the plr_* values
    # setdefault for the same reason: reloading a newer checkpoint through
    # --from_saved_state must not overwrite values already in model_kwargs.)
    model_kwargs.setdefault("tasktype", tasktype)
    model_kwargs.setdefault(
        "n_classes",
        output_dim if tasktype == "multiclass" else (2 if tasktype == "binclass" else None),
    )

    model = TabERA(**model_kwargs, column_names=dataset.col_names)
    # Record beta at the true training start. Under --from_saved_state there is
    # no new training start, so beta_init_is_training_start is marked False.
    _beta_raw_init = float(model.dev_beta_raw.detach().mean())
    _beta_init = float(model.effective_beta().detach().mean())

    # ── Train (skipped and restored under --from_saved_state) ───
    wrapper = TabERAWrapper(
        model, best_params, tasktype,
        device=str(device), epochs=args.epochs, patience=args.patience,
        # Checkpoint-selection timing.
        defer_early_stopping=args.defer_early_stopping,
        min_epochs=args.min_epochs,
        # Needed for group text labelling: the group description in layer (1)
        # is a text summary rather than a medoid, and this cache backs it.
        cat_cols=list(dataset.X_cat), num_cols=list(dataset.X_num),
        col_names=dataset.col_names,
        cat_category_names=dataset.cat_category_names,
        target_class_names=dataset.target_class_names,
        quantile_transformer=dataset.quantile_transformer,
        # Silent unless asked. These lines track how the partition formed,
        # which is a development concern rather than a result.
        # Same default as optimize.py: every 10 epochs. --regroup_log_every 0
        # turns the lines off entirely.
        regroup_log_every=args.regroup_log_every,
        time_epoch=args.time_epoch,
        log_beta=args.log_beta,
        beta_lr_mult=args.beta_lr_mult,
        refresh_on_best=args.refresh_on_best,
        early_stop_metric=args.early_stop_metric,
        # Wrapper-only selection setting; not a TabERA constructor argument.
        tie_rule=args.tie_rule,
    )
    wrapper._data_id = args.openml_id
    if _saved_state is not None:
        # ── Skip training; restore the saved state as-is ──────
        # Allow only the explicitly supported missing keys when restoring a checkpoint.
        _ALLOWED_MISSING = ("dev_head.", "dev_beta_raw", "dev_gamma_raw",
                            "prototype_layer_2.")
        _miss, _unexp = model.load_state_dict(
            _saved_state["state_dict"], strict=False)
        _bad = [k for k in _miss if not k.startswith(_ALLOWED_MISSING)]
        if _bad or _unexp:
            raise RuntimeError(
                f"checkpoint mismatch: unexpected missing {_bad[:5]}, "
                f"unexpected {list(_unexp)[:5]}. The saved structure differs "
                f"from the current model (check the checkpoint model_kwargs).")
        if _miss:
            print(f"  [--from_saved_state] {len(_miss)} new parameters keep "
                  f"their initial values: {[k for k in _miss][:3]}...")
        # Items the state_dict does not capture, being plain Python attributes
        # rather than buffers. sample_groups is required for
        # group-constrained retrieval and retrieve() misbehaves without it;
        # group_labels and target_labels are the text labels for layer (1);
        # feature_store._store holds the raw feature values for layer (2).
        model.prototype_layer.sample_groups = _saved_state.get("sample_groups")
        model.prototype_layer.group_labels  = _saved_state.get("group_labels")
        model.prototype_layer.target_labels = _saved_state.get("target_labels")
        fs_state = _saved_state.get("feature_store_state")
        if fs_state is not None and model.feature_store is not None:
            # Compatibility with feature-store states that do not contain sample_ids.
            if len(fs_state) == 4:
                store, ptr, filled, sample_ids = fs_state
            else:
                store, ptr, filled = fs_state
                sample_ids = torch.full((model.feature_store.max_size,), -1, dtype=torch.long)
                print(f"  !  the saved feature_store_state has no sample_ids: "
                      f"this looks like an older checkpoint. Skipping ID checks.")
            model.feature_store._store       = store.to(device)
            model.feature_store._ptr         = ptr
            model.feature_store._filled      = filled
            model.feature_store._sample_ids  = sample_ids.to(device)
        if model.prototype_layer.sample_groups is not None:
            try:
                model.memory.cache_sample_groups(
                    model.prototype_layer.sample_groups,
                    device,
                    centroid_emb=model.prototype_layer.centroid_emb.detach(),
                )
                _cg = getattr(model.memory, "_cached_groups", None)
                if _cg is None:
                    print("  !  cache_sample_groups ran but the cache is empty "
                          "(every group is empty): falling back to a global search.")
                else:
                    _sz = model.memory._cached_group_sizes
                    print(f"  [group cache] rebuilt: P={_cg.shape[0]}, "
                          f"largest group={_cg.shape[1]}, median={int(_sz.median().item())}, "
                          f"k={model.k}")
                    _n_fb = int((_sz < model.k).sum().item())
                    if _n_fb:
                        print(f"  !  {_n_fb}/{len(_sz)} groups are smaller than k({model.k}); "
                              f"those samples fall back to cross-group or global "
                              f"search in retrieve() (see fallback_mask in tabera.py).")
            except Exception as _ce:
                print(f"  !  group cache rebuild failed: {type(_ce).__name__}: {_ce} "
                      f"-- falling back to a global search.")
        if model.prototype_layer.sample_groups is None:
            print(f"  !  the saved state has no sample_groups; this file appears"
                  f" to predate --from_saved_state support."
                  f" Group-constrained retrieval and layers (1)(2) may be wrong.")
        if args.refresh_on_best:
            # ── Meta-diagnostic: memory staleness ───────────────────
            # memory.keys[i] is a one-off snapshot computed at some point
            # during training, **with a dropout mask applied**, whereas an
            # inference query is a deterministic embedding in eval mode.
            # Before a refresh the two are
            #     memory space = noisy embedding manifold
            #     test  space  = deterministic embedding manifold
            # different spaces, and comparing train against test routing in
            # that state mixes **encoder drift, dropout noise and distribution
            # shift** together.
            #
            # This block quantifies what the refresh actually changed -- a
            # meta-diagnostic bounding the confidence of every train-side
            # routing metric computed so far (k-coverage, occupancy
            # correlation, group size distribution).
            _stale_prev = None
            try:
                _nm0 = int(model.memory.filled.item())
                if _nm0 > 0:
                    _stale_prev = model.memory.keys[:_nm0].detach().float().clone()
                    _stale_prev_assign = None
                    _cN0 = torch.nn.functional.normalize(
                        model.prototype_layer.centroid_emb.detach().float(), dim=-1)
                    _stale_prev_assign = (
                        torch.nn.functional.normalize(_stale_prev, dim=-1) @ _cN0.T
                    ).argmax(-1)
            except Exception:
                _stale_prev = None

            refresh_stats = model.refresh_memory_keys()
            if refresh_stats is not None:
                if args.verbose:
                    print(f"  [--refresh_on_best] recomputed {refresh_stats['n_refreshed']} "
                          f"memory.keys slots with the frozen weights")

                if _stale_prev is not None:
                    try:
                        import torch.nn.functional as _sF
                        _new = model.memory.keys[:_stale_prev.shape[0]].detach().float()
                        # 1) Representation drift. The mean alone is not
                        #    enough; p5 catches the case where only a few
                        #    samples moved far.
                        _cos = _sF.cosine_similarity(_stale_prev, _new, dim=-1).cpu().numpy()
                        _new_assign = (_sF.normalize(_new, dim=-1) @ _cN0.T).argmax(-1)
                        _agree = float((_new_assign == _stale_prev_assign).float().mean())
                        # 3) Geometry drift: change in distance to the
                        #    assigned centroid.
                        _d_old = float((1 - (_sF.normalize(_stale_prev, dim=-1) @ _cN0.T).max(-1).values).mean())
                        _d_new = float((1 - (_sF.normalize(_new, dim=-1) @ _cN0.T).max(-1).values).mean())
                        print(f"  [memory staleness] cos(q_memory, q_refresh): "
                              f"mean={_cos.mean():.4f} std={_cos.std():.4f} "
                              f"p5={np.percentile(_cos,5):.4f} p50={np.percentile(_cos,50):.4f} "
                              f"p95={np.percentile(_cos,95):.4f}")
                        print(f"  [memory staleness] assignment agreement={_agree*100:.1f}%  "
                              f"| centroid dist {_d_old:.4f} → {_d_new:.4f}")
                        # Also written to the npz: judging afterwards how far
                        # this run's diagnostics can be trusted requires having
                        # it inside the result file.
                        globals()["_MEMORY_STALENESS"] = dict(
                            cos_mean=float(_cos.mean()), cos_std=float(_cos.std()),
                            cos_p5=float(np.percentile(_cos, 5)),
                            cos_p50=float(np.percentile(_cos, 50)),
                            cos_p95=float(np.percentile(_cos, 95)),
                            assign_agreement=_agree,
                            centroid_dist_before=_d_old, centroid_dist_after=_d_new,
                        )
                        if _cos.mean() < 0.9 or _agree < 0.9:
                            print(f"  !  [memory staleness] train-side routing metrics from "
                                  f"before the refresh were computed under a **different "
                                  f"encoder state**; recheck train k-coverage, occupancy "
                                  f"correlation and group size.")
                    except Exception as _se:
                        print(f"  [memory staleness] diagnostic failed: {type(_se).__name__}: {_se}")

                regroup_stats = wrapper._resync_groups_after_refresh()
                if regroup_stats is not None:
                    print(f"  [--refresh_on_best] resynced sample_groups on the clean "
                          f"embeddings (active={regroup_stats.get('active_ratio', 0)*100:.0f}%, "
                          f"reinit={regroup_stats.get('reinit_count', 0)})")
        else:
            print(f"  [--from_saved_state] restored (no retraining from epoch 0)")
        _fit_time = float("nan")   # No training was performed.
    else:
        _fit_st = time.time()
        wrapper.fit(X_train, y_train, X_val, y_val)
        _fit_time = time.time() - _fit_st

    # ── Post-training readout refinement ──────────────────
    # Frozen-h refit of the shared linear readout. Not a hyperparameter
    # search dimension: lr / wd / epochs / patience / null-inclusion are all
    # fixed here and are never re-selected per dataset. The encoder, the
    # centroids, the routing and beta do not move, so
    #     z = (W c_a + b) + beta * W r
    # stays an exact decomposition of the logits.
    _refine_info = {"enabled": False}
    _refine_time = 0.0
    if getattr(args, "final_readout_refine", False):
        if tasktype == "regression":
            print("  [readout refine] skipped (regression)")
        else:
            _rst = time.time()
            # Raw predictions are recorded first for the ablation row. The
            # main benchmark uses the refined model, fixed in advance; the raw
            # numbers are for auditing, never for picking between the two.
            _raw_preds_test = wrapper.predict(X_test)
            _raw_probs_test = wrapper.predict_proba(X_test)
            _refine_info = wrapper.refine_readout(
                X_train, y_train, X_val, y_val,
                lr=args.refine_lr, weight_decay=args.refine_wd,
                epochs=args.refine_epochs, patience=args.refine_patience,
                include_null=not args.refine_no_null,
            )
            _refine_info["raw_test_metrics"] = calculate_metric(
                y_test, _raw_preds_test, _raw_probs_test, tasktype, "test")
            _refine_time = time.time() - _rst
    # Include readout-refinement time in the reported fit time when enabled.
    _base_fit_time = _fit_time
    _fit_time = _base_fit_time + _refine_time
    _refine_info["base_fit_time"] = float(_base_fit_time)
    _refine_info["readout_refine_time"] = float(_refine_time)
    _refine_info["total_fit_time"] = float(_fit_time)

    # ── Evaluate ──────────────────────────────────────────
    preds_val  = wrapper.predict(X_val)
    preds_test = wrapper.predict(X_test)
    probs_val  = wrapper.predict_proba(X_val)  if tasktype != "regression" else None
    probs_test = wrapper.predict_proba(X_test) if tasktype != "regression" else None

    if tasktype == "regression":
        val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, None, tasktype, "val")
        test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, None, tasktype, "test")
    else:
        val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
        test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

    print(f"\n  {env_info}  {openml_id}  {dataset_info['name']}  tabera  {log_dir}")
    print(f"  val  : {val_metrics}")
    print(f"  test : {test_metrics}")

    # ── Save results in the MultiTab-compatible layout ─────────────────
    # The analysis pipeline can read TabERA and MultiTab baselines through the
    # same path pattern and result keys:
    #     reproduce_logs/seed={S}/data={D}/model={m}..init_hps=False..deep=0..hyper=0.npy
    #     -> {"Prediction", "Probability", "time", "Performance"}
    #
    # TabERA stores final probabilities in "Probability"; MultiTab baseline files
    # may store logits there. "Performance" is already computed from the proper
    # probability representation in both pipelines.
    try:
        _mt_dir = os.path.join(args.savepath, 'reproduce_logs',
                               f'seed={args.seed}', f'data={openml_id}')
        os.makedirs(_mt_dir, exist_ok=True)
        # Encode structural/source variants in the filename to prevent collisions.
        _mt_variant = ("" if actual_geometry == "additive"
                       else f"_geom-{actual_geometry}")
        # Add non-default training seeds to the filename to prevent collisions.
        if _actual_params_seed is not None:
            _mt_variant += f"_paramsseed-{_actual_params_seed}"
        if _actual_params_geometry is not None:
            _mt_variant += f"_paramsgeom-{_actual_params_geometry}"
        if actual_beta_param != "sigmoid":
            _mt_variant += f"_bp-{actual_beta_param}"
        if actual_head_input_scale != "unit":
            _mt_variant += f"_hs-{actual_head_input_scale}"
        if _actual_tie_rule != "first":
            _mt_variant += f"_tie-{_actual_tie_rule}"
        if _actual_params_variant is not None:
            _mt_variant += f"_paramsvar-{_actual_params_variant}"
        if _actual_esm != "accuracy":
            _mt_variant += f"_esm-{_actual_esm}"
        if args.beta_lr_mult != 1.0 and not args.from_saved_state:
            _mt_variant += f"_blrm-{args.beta_lr_mult:g}"
        if train_seed != args.seed:
            _mt_variant += f"_trainseed-{train_seed}"
        _mt_fname = os.path.join(
            _mt_dir, f'model=tabera{_mt_variant}'
                     '..init_hps=False..deep=0..hyper=0.npy')
        _to_np = lambda t: (t.detach().cpu().numpy()
                            if isinstance(t, torch.Tensor) else
                            (None if t is None else np.asarray(t)))
        np.save(_mt_fname, {
            "Prediction":  _to_np(preds_test),
            "Probability": _to_np(probs_test),
            "time":        float(_fit_time),
            "Performance": {k: float(v) for k, v in test_metrics.items()},
            # TabERA-specific metadata; baseline aggregation ignores this field.
            "Performance_val": {k: float(v) for k, v in val_metrics.items()},
            "best_params": best_params,
            "readout_refinement": _refine_info,
        })
        print(f"  saved: {_mt_fname}")
        # Save pre-refinement metrics separately for audit/ablation only.
        if _refine_info.get("enabled") and _refine_info.get("raw_test_metrics"):
            _raw_fname = os.path.join(
                _mt_dir, f'model=tabera{_mt_variant}_raw'
                         '..init_hps=False..deep=0..hyper=0.npy')
            np.save(_raw_fname, {
                "Prediction":  _to_np(_raw_preds_test),
                "Probability": _to_np(_raw_probs_test),
                "time":        float(_refine_info["base_fit_time"]),
                "Performance": {k: float(v) for k, v
                                in _refine_info["raw_test_metrics"].items()},
                "best_params": best_params,
                "note": "pre-refinement TabERA. ablation only, not for selection.",
            })
            print(f"  saved: {_raw_fname}")
    except Exception as _e:
        print(f"  !  failed to save reproduce_logs: {type(_e).__name__}: {_e}")


    # ── Linear Probe ───────────────────────────────────────────
    # Fit a separate linear classifier (or Ridge for regression) on each
    # representation to measure how much target information it carries. This
    # asks a different question from a shuffle ablation, which looks at the
    # prediction instead. It separates "the information is not there" from
    # "the information is there but the head cannot use it". Embeddings are
    # extracted from a model loaded via --from_saved_state and fitted
    # separately with sklearn; TabERA itself is untouched.
    if args.linear_probe and do_analysis:
        print(f"\n{'='*60}")
        print(f"  Linear probe: how much information query_emb / context_emb carry")
        print(f"{'='*60}")
        model.eval()

        # ⚠ agg_emb no longer exists (no aggregator is built). The probe
        #   targets are query_emb and context_emb; the upper-bound measurement
        #   in section 14-5 is also relative to h0, so these two suffice.
        def _extract_embeddings(X, batch_size=512):
            qs, cs = [], []
            with torch.no_grad():
                for start in range(0, len(X), batch_size):
                    _out = model(X[start:start + batch_size])
                    qs.append(_out["query_emb"].cpu())
                    cs.append(_out["context_emb"].cpu())
            return torch.cat(qs).numpy(), torch.cat(cs).numpy()

        q_tr, c_tr = _extract_embeddings(X_train)
        q_te, c_te = _extract_embeddings(X_test)

        import numpy as _np
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, r2_score

        if tasktype == "regression":
            y_tr_np = (y_train * y_std).cpu().numpy()
            y_te_np = (y_test * y_std).cpu().numpy()
        else:
            y_tr_np = y_train.cpu().numpy().astype(int)
            y_te_np = y_test.cpu().numpy().astype(int)

        probe_results = {}
        reprs = {
            "query_emb":     (q_tr, q_te),
            "context_emb":   (c_tr, c_te),
            "concat(q+c)":   (_np.concatenate([q_tr, c_tr], axis=1),
                              _np.concatenate([q_te, c_te], axis=1)),
        }

        # Two variants that normalise each block before concatenation, to test
        # the scale-imbalance hypothesis. If normalising restores performance,
        # scale is plausibly the cause; if it does not, scale alone does not
        # explain it (leaving multicollinearity or a collapse in
        # representation geometry).
        def _l2_normalize_blocks(*blocks_tr_te):
            """L2-normalise each (tr, te) pair per sample, then concatenate."""
            tr_parts, te_parts = [], []
            for tr, te in blocks_tr_te:
                tr_n = tr / (_np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
                te_n = te / (_np.linalg.norm(te, axis=1, keepdims=True) + 1e-8)
                tr_parts.append(tr_n)
                te_parts.append(te_n)
            return _np.concatenate(tr_parts, axis=1), _np.concatenate(te_parts, axis=1)

        def _standardize_blocks(*blocks_tr_te):
            """StandardScaler each (tr, te) pair (fitted on train, giving
            per-dimension zero-mean/unit-variance like LayerNorm), then
            concatenate."""
            from sklearn.preprocessing import StandardScaler
            tr_parts, te_parts = [], []
            for tr, te in blocks_tr_te:
                _scaler = StandardScaler()
                tr_parts.append(_scaler.fit_transform(tr))
                te_parts.append(_scaler.transform(te))
            return _np.concatenate(tr_parts, axis=1), _np.concatenate(te_parts, axis=1)

        reprs["concat(q+c)_l2norm"] = _l2_normalize_blocks((q_tr, q_te), (c_tr, c_te))
        reprs["concat(q+c)_standardized"] = _standardize_blocks((q_tr, q_te), (c_tr, c_te))

        for _name, (_tr, _te) in reprs.items():
            if tasktype == "regression":
                _clf = Ridge(alpha=1.0)
                _clf.fit(_tr, y_tr_np)
                _score = float(r2_score(y_te_np, _clf.predict(_te)))
                _metric_name = "R2"
            else:
                _clf = LogisticRegression(max_iter=2000)
                _clf.fit(_tr, y_tr_np)
                _score = float(accuracy_score(y_te_np, _clf.predict(_te)))
                _metric_name = "acc"
            probe_results[_name] = _score
            print(f"  {_name:28s} linear probe {_metric_name}={_score:.4f}")

        # Representation similarity: does a branch carry new information, or
        # does it point essentially the same way as query_emb? Cosine gives an
        # intuitive per-sample direction similarity; linear CKA measures
        # alignment of the whole representation space and is invariant to
        # scale and rotation, a stricter multivariate measure. The two can
        # disagree in either direction, so both are reported.
        def _linear_cka(X: "_np.ndarray", Y: "_np.ndarray") -> float:
            Xc = X - X.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            hsic = _np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
            norm_x = _np.linalg.norm(Xc.T @ Xc, ord="fro")
            norm_y = _np.linalg.norm(Yc.T @ Yc, ord="fro")
            return float(hsic / (norm_x * norm_y + 1e-12))

        def _mean_cosine(X: "_np.ndarray", Y: "_np.ndarray"):
            xn = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            yn = Y / (_np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
            cos = (xn * yn).sum(axis=1)
            return {"mean": float(cos.mean()), "std": float(cos.std())}

        similarity_results = {}
        for _pair_name, _X, _Y in [
            ("query_vs_context", q_te, c_te),
        ]:
            _cka = _linear_cka(_X, _Y)
            _cos = _mean_cosine(_X, _Y)
            similarity_results[_pair_name] = {"cka": _cka, "cosine_mean": _cos["mean"], "cosine_std": _cos["std"]}
            print(f"  [similarity] {_pair_name:18s} CKA={_cka:.4f}  "
                  f"cosine={_cos['mean']:+.4f}±{_cos['std']:.4f}")

        probe_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_linear_probe.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump({
                "probe_results": probe_results,
                "similarity_results": similarity_results,
                "openml_id": openml_id, "seed": args.seed, "tasktype": tasktype,
            }, f)
        print(f"\n  saved: {probe_path}")

    if args.calibration_analysis and do_analysis:
        if tasktype == "regression":
            print(f"\n  !  --calibration_analysis is classification-only; "
                  f"skipping for this dataset ({tasktype}).")
        else:
            calib_result = run_calibration_analysis(
                model, X_test, y_test, tasktype,
                X_train=X_train, y_train=y_train,
                class_names=getattr(dataset, "target_class_names", None),
            )
            print_calibration_analysis(calib_result)
            calib_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_calibration.pkl"
            with open(calib_path, "wb") as f:
                pickle.dump({**calib_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  saved: {calib_path}")








    # ── Save results ──────────────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_meta.pkl"

    model.eval()
    _pred_batch_size = 512
    _logits_chunks = []
    with torch.no_grad():
        for _start in range(0, len(X_test), _pred_batch_size):
            _out = model(X_test[_start:_start + _pred_batch_size])
            _logits_chunks.append(_out["logits"].cpu())
    logits = torch.cat(_logits_chunks, dim=0).numpy()
    np.save(str(pred_path), logits)




    # Finalise the gate statistics accumulated above as a batch-weighted mean.
    # Left empty when there is nothing to accumulate.
    _final_gate_stats = {"mean": {}, "var": {}, "entropy": None,
                          "logit_mean": {}, "logit_gap": None}

    meta = {
        "openml_id":   openml_id,
        "tasktype":    tasktype,
        "best_params": best_params,
        "readout_refinement": _refine_info,
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "seed":        args.seed,
        "train_seed":  train_seed,
        "centroid_geometry_diag": wrapper.centroid_geometry_diag,
        # Per-epoch routing-stability diagnostics.
        "regroup_history": wrapper.regroup_history,
        # Epoch selected by the configured validation rule and the first eligible epoch.
        "best_epoch": getattr(wrapper, "best_epoch", None),
        "selection_open_epoch": getattr(wrapper, "selection_open_epoch", None),
        # ── Design Lock v1 §4: selection resolution ────────────────────
        # Number of validation-accuracy ties encountered under the configured tie rule.
        "tie_rule": _training_provenance.get("tie_rule_actual"),
        # Same source-vs-actual provenance stored in the checkpoint.
        "training_provenance": _training_provenance,
        "best_correct": getattr(wrapper, "best_correct", None),
        "n_val": getattr(wrapper, "n_val", None),
        "n_best_ties": getattr(wrapper, "n_best_ties", None),
        # Effective gamma is reconstructed from model_kwargs, not state_dict.
        "head_input_gamma": (model.effective_gamma()
                             if hasattr(model, "effective_gamma") else 1.0),
        # ── Axis 2: prototype behaviour ──────────────────────────────
        # Prototype diagnostics recorded with the run.
        "prototype_alignment": diag.prototype_class_alignment(model),
        "context_diversity":   diag.context_space_diversity(model),
        "beta_history": getattr(wrapper, "beta_history", []),
        "beta_lr_mult": getattr(wrapper, "beta_lr_mult", 1.0),
        # dev_beta_raw / effective β at model construction, before fit() and
        # before any state is loaded. ⚠ Under --from_saved_state this is the
        # fresh-construction default, not the checkpoint's value, and there is
        # no training start point at all -- hence the flag below.
        "beta_raw_init": _beta_raw_init,
        "beta_init": _beta_init,
        "beta_init_is_training_start": args.from_saved_state is None,
        # ── Config freeze: did this run deviate from the defaults? ──────
        # ⚠ A static check can only inspect the **defaults in the code**; it
        #   cannot see a flag overriding them at run time. Recording what
        #   differed from the defaults in the result file itself is what makes
        #   the condition identifiable when a table is built later. A P=35
        #   study really was lost to a P=100 run.
        "freeze_deviations": {
            k: v for k, v in {
                "n_prototypes":  args.n_prototypes,
                "beta_lr_mult":  args.beta_lr_mult,
                # The update-rule condition must be identifiable from meta.pkl alone
                "disable_dead_reinit": args.disable_dead_reinit,
                # Selection timing changes which checkpoint is returned, so a
                # results table cannot be read without knowing it.
                "defer_early_stopping": args.defer_early_stopping,
                "min_epochs":    args.min_epochs,
                "epochs":        args.epochs,
                "patience":      args.patience,
                "num_bins":      args.num_bins,
                "cat_combine":   args.cat_combine,
                "num_embedding": args.num_embedding,
                # Use the actual restored/trained structure, not the current CLI default.
                "correction_geometry": actual_geometry,
                "beta_param": actual_beta_param,
                "head_input_scale": actual_head_input_scale,
                # Use restored provenance rather than the current CLI.
                "tie_rule": _actual_tie_rule,
                "params_seed": _actual_params_seed,
                "params_geometry": _actual_params_geometry,
                "params_variant": _actual_params_variant,
                "early_stop_metric": _actual_esm,
            }.items()
            if v != {"n_prototypes": None, "beta_lr_mult": 1.0,
                     "disable_dead_reinit": False,
                     "defer_early_stopping": False, "min_epochs": 0,
                     "epochs": HPO_TRAINING_SCHEDULE["epochs"],
                     "patience": HPO_TRAINING_SCHEDULE["patience"],
                     # ⚠ Changed from 0.005 to 0.0 when the default moved.
                     #   Afterwards 0 is the default and not a deviation.
                     "num_bins": 8,
                     "cat_combine": "onehot", "num_embedding": "ple",
                     "correction_geometry": "additive",
                     "beta_param": "sigmoid", "head_input_scale": "unit",
                     "tie_rule": "first",
                     "params_seed": None,
                     "params_geometry": None, "params_variant": None,
                     "early_stop_metric": "accuracy"}[k]
        },
        # The --time_epoch measurements are stored for the same reason.
        "epoch_timing": getattr(wrapper, "_timing", {}),
        # ── Optimizer update budget ────────────────────────────────
        # Record the actual number of optimizer steps; use ceil because the
        # training loop includes the final partial batch.
        # The training loop is
        #   `for start in range(0, len(y_train), batch_size)`, so the final
        #   partial batch also takes an optimizer step (1067: 1687/128 → 14,
        #   not 13). This number feeds the beta-timescale budget estimate.
        "steps_per_epoch": (
            -(-len(X_train) // best_params["batch_size"])
            if best_params.get("batch_size") else None),
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only if args.deterministic else None,
        "exclude_self_retrieval": (not args.allow_self_retrieval),
        "dev_beta_final": float(model.effective_beta().detach().mean().item()),
        "cat_embedding": True,
        "cat_combine": args.cat_combine,
        "cat_embed_dim": args.cat_embed_dim if args.cat_combine == "concat" else None,
        "num_embedding": args.num_embedding,
        "num_bins": args.num_bins if args.num_embedding == "ple" else None,
        "plr_n_frequencies": args.plr_n_frequencies if args.num_embedding == "plr_lite" else None,
        "plr_freq_scale": args.plr_freq_scale if args.num_embedding == "plr_lite" else None,
        "plr_out_dim": args.plr_out_dim if args.num_embedding == "plr_lite" else None,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  saved: {pred_path}")
    if wrapper.centroid_geometry_diag is not None:
        _diag = wrapper.centroid_geometry_diag
        if args.verbose:
            print(f"  centroid_geometry_diag: "
                  f"reinit_per_epoch={_diag.get('reinit_per_epoch', float('nan')):.3f}  "
                  f"active_ratio_std={_diag.get('active_ratio_std', float('nan')):.4f}  "
                  f"margin_percentile={_diag.get('margin_percentile', float('nan')):.3f}  "
                  f"avg_inter_dist_final={_diag.get('avg_inter_dist_final', float('nan')):.3f} "
                  f"(compare with the avg_inter_dist logged at init: a clear drop "
                  f"by the end means the centroids have bunched together)")

    # ── Save the model state (for --from_saved_state) ─────────
    # model_kwargs already has every architecture flag merged in (via the
    # model_kwargs.update() above). They are absent from best_params, which
    # only covers what Optuna searched, so without this a --from_saved_state
    # restore would silently fall back to the defaults.
    #
    # Items state_dict() does not capture were missed here too, the same
    # problem as in the best-checkpoint snapshot in supervised.py:
    # sample_groups, group_labels and target_labels are plain Python
    # attributes rather than buffers, and feature_store is not an nn.Module.
    # Without them a --from_saved_state restore produces wrong output for
    # layers (1) and (2) -- and without sample_groups, group-constrained
    # retrieval breaks entirely.
    state_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_model_state.pt"
    fs = model.feature_store
    torch.save({
        "state_dict":     model.state_dict(),
        "model_kwargs":   model_kwargs,
        "best_params":    best_params,
        "readout_refinement": _refine_info,
        "sample_groups":  model.prototype_layer.sample_groups,
        "group_labels":   model.prototype_layer.group_labels,
        "target_labels":  model.prototype_layer.target_labels,
        "feature_store_state": (
            (fs._store.detach().cpu(), fs._ptr, fs._filled, fs._sample_ids.detach().cpu())
            if fs is not None else None
        ),
        "col_names":    dataset.col_names,
        "n_train":      len(X_train),
        "tasktype":     tasktype,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "seed":         args.seed,
        "train_seed":   train_seed,
        "deterministic": args.deterministic,
        # Store selection/HPO-source provenance separately from model_kwargs.
        "training_provenance": _training_provenance,
    }, str(state_path))
    print(f"  saved: {state_path}")

    # ── Explanation output ───────────────────────────────
    if args.explain and do_analysis:
        print(f"\n{'='*52}")
        print(f"  TabERA Explanations (--explain)")
        print(f"{'='*52}")

        model.eval()
        n_show = min(args.n_explain, len(y_test))
        X_show = X_test[:n_show]

        with torch.no_grad():
            out = model(X_show, return_explanations=True)

        explanations = out.get("explanations", [])

        # Prediction confidence: the classifier softmax for classification, or
        # the predicted value for regression. Computed here so the display can
        # keep it separate from routing confidence (layer 1) from the start.
        # get_preds_and_probs uses the same logic as the metric computation in
        # eval.py, so its definition cannot drift from test_metrics.
        pred_idx, pred_probs = get_preds_and_probs(out["logits"], tasktype)
        pred_infos = []
        for b in range(n_show):
            if tasktype == "regression":
                pred_val = float(pred_idx[b].item()) * y_std
                pred_infos.append({"pred_label": f"{pred_val:.4g}", "pred_confidence": None})
            else:
                idx = int(pred_idx[b].item())
                conf = float(pred_probs[b, idx].item())
                label = (dataset.target_class_names[idx]
                         if getattr(dataset, "target_class_names", None) else str(idx))
                # pred_code decides supporting vs contrasting. With only the
                # label string there is nothing to compare against the
                # neighbours' integer label codes.
                pred_infos.append({"pred_label": label, "pred_confidence": conf,
                                    "pred_code": idx})

        _nbrs = diag.retrieved_neighbors(model, out)
        _le   = diag.local_label_evidence(model, out)
        _pdv  = diag.prototype_deviation(model, out)
        _gst  = diag.group_relative_feature_stats(model, out, X_show)
        _wrp  = (diag.within_region_position(model, out)
                 if getattr(args, "refresh_on_best", False) else None)

        cat_names = {dataset.col_names[i] for i in dataset.X_cat}
        X_show_cpu = X_show.detach().cpu().numpy()
        for b, exp in enumerate(explanations):
            query_dict = {name: float(X_show_cpu[b, i])
                          for i, name in enumerate(dataset.col_names)}
            exp["neighbors"]           = (_nbrs[b] if _nbrs else [])
            exp["local_evidence"]      = (_le[b]   if _le   else None)
            exp["prototype_deviation"] = (_pdv[b]  if _pdv  else None)
            exp["group_stats"]         = (_gst[b]  if _gst  else None)
            exp["region_position"]     = (_wrp[b]  if _wrp  else None)
            for nb in exp["neighbors"]:
                if nb.get("features"):
                    # Attach every gap; sorting and truncation belong to the
                    # display layer.
                    nb["gaps"] = diag.feature_gaps(
                        query_dict, nb["features"], cat_names)
        if not explanations:
            print("  (no explanations — memory bank has not been filled yet)")
            print("  → try increasing epochs or n_trials.")
        else:
            for i in range(n_show):
                print_explanation(explanations, i, dataset.col_names,
                                   cat_category_names=dataset.cat_category_names,
                                   quantile_transformer=dataset.quantile_transformer,
                                   num_cols=list(dataset.X_num),
                                   pred_info=pred_infos[i],
                                   target_class_names=getattr(
                                       dataset, "target_class_names", None),
                                   tasktype=tasktype,
                                   verbose=getattr(args, "explain_verbose", False))

    return {"train_seed": train_seed, "val_metrics": val_metrics, "test_metrics": test_metrics}


def main():

    parser = argparse.ArgumentParser(description="TabERA Reproduce Best Config")
    parser.add_argument("--gpu_id",    type=int, default=0)
    parser.add_argument("--openml_id", type=int, required=True)
    parser.add_argument("--savepath",  type=str, default=".",
                        help="parent directory containing optim_logs")
    parser.add_argument("--seed",      type=int, default=1,
                        help="evaluation seed. It selects the data split -- with "
                             "KFold(random_state=42) fixed in libs/data.py it decides "
                             "which fold becomes test -- and, when --train_seed / "
                             "--train_seeds are not given, it is ALSO used for training "
                             "initialization and batch order (train_seed defaults to "
                             "--seed). Use --train_seed to vary training randomness "
                             "while holding the data split fixed.")
    parser.add_argument("--train_seed", type=int, default=None,
                        help=(
                            "seed for training init and batch order only: it is passed "
                            "to torch.manual_seed and np.random.seed instead of --seed. "
                            "The data split (--seed, the fold selection in "
                            "TabularDataset) is unaffected. Left unset, --seed is used, "
                            "as before. To measure run-to-run variance, hold --seed "
                            "fixed and vary this across N runs: on the same "
                            "train/val/test split, weight initialisation, the batch "
                            "order (torch.randperm in supervised.py) and dead-prototype "
                            "reinit (torch.randint/torch.randn in tabera.py) all come "
                            "from the global torch RNG this seed sets, so changing it "
                            "alone perturbs the training trajectory and nothing else. "
                            "No effect with --from_saved_state, which skips training."
                        ))
    parser.add_argument("--train_seeds", type=int, nargs="+", default=None,
                        help=(
                            "plural form of --train_seed: run several in one go, e.g. "
                            "--train_seeds 1 2 3 4 5. Like optimize.py, the dataset and "
                            "HPO study load once and only training repeats per seed "
                            "(run_single_seed), removing the per-process dataset load "
                            "cost of launching one shell process per seed. When given, "
                            "--train_seed is ignored. Cannot be combined with "
                            "--from_saved_state, which loads one checkpoint saved at one "
                            "seed, so iterating over seeds is meaningless there and "
                            "raises an error. With two or more seeds, a mean +- std "
                            "summary of the val/test metrics is printed at the end."
                        ))
    parser.add_argument("--explain_seed", type=int, default=None,
                        help=(
                            "with --train_seeds, which seed runs whichever of "
                            "--explain / --calibration_analysis / --linear_probe are "
                            "enabled. Defaults to the last seed in --train_seeds. "
                            "Printing the --explain text for every seed makes the log "
                            "unmanageable, so the detailed analysis is limited to one "
                            "representative seed and the rest keep only val/test "
                            "metrics. A value not in --train_seeds is an error."
                        ))
    parser.add_argument("--deterministic", action="store_true",
                        help="enable strict deterministic PyTorch/CUDA execution where supported")
    parser.add_argument("--deterministic_warn_only", action="store_true",
                        help="warn instead of failing on operations without deterministic implementations")
    parser.add_argument("--run_tag", type=str, default=None,
                        help="append an arbitrary tag to output filenames")
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    parser.add_argument("--epochs",    type=int, default=HPO_TRAINING_SCHEDULE["epochs"],
                        help="number of training epochs; default matches optimize.py HPO schedule")
    parser.add_argument("--patience",  type=int, default=HPO_TRAINING_SCHEDULE["patience"],
                        help="early-stopping patience; default matches optimize.py HPO schedule")
    parser.add_argument("--defer_early_stopping", action="store_true",
                        help="delay checkpoint selection and patience counting until selection is open")
    parser.add_argument("--min_epochs", type=int, default=0,
                        help="minimum epoch before checkpoint selection can open")
    parser.add_argument("--n_explain", type=int, default=3,
                        help="number of test samples to explain")
    parser.add_argument("--n_prototypes", type=int, default=None,
                        help="override the study prototype count when reproducing a matching tagged study")
    parser.add_argument("--beta_lr_mult", type=float, default=1.0,
                        help="learning-rate multiplier for dev_beta_raw")
    parser.add_argument("--log_beta", action="store_true",
                        help="record beta values and gradients during training")
    parser.add_argument("--time_epoch", action="store_true",
                        help="record and print per-epoch timing diagnostics")
    parser.add_argument("--verbose", action="store_true",
                        help="print additional run/provenance diagnostics")
    parser.add_argument("--explain_verbose", action="store_true",
                        help=("append researcher diagnostics to the compact explanation: "
                              "routing mass, predicted-channel logit decomposition, label "
                              "entropy, region-characteristic features, and raw representation "
                              "distance. The default --explain view is user-facing and concise."))
    parser.add_argument("--explain",   action="store_true",
                        help="print the feature explanation after training")
    parser.add_argument("--final_readout_refine", action="store_true",
                        help=(
                            "Refit only the shared linear readout (W,b) on frozen h. "
                            "Encoder, centroids, routing, and beta remain fixed, so the "
                            "exact logit decomposition is preserved. This is a fixed "
                            "post-training procedure, not an HPO dimension."))
    parser.add_argument("--refine_lr", type=float, default=1e-2)
    parser.add_argument("--refine_wd", type=float, default=0.0)
    parser.add_argument("--refine_epochs", type=int, default=500)
    parser.add_argument("--refine_patience", type=int, default=50)
    parser.add_argument("--refine_no_null", action="store_true",
                        help="exclude the null readout candidate (W0,b0) from validation selection")
    parser.add_argument("--params_seed", type=int, default=None,
                        help=(
                            "read HPO parameters from another seed directory without "
                            "changing the current data fold or training RNG"))
    parser.add_argument("--beta_param", type=str, default="sigmoid",
                        choices=["sigmoid", "centered"],
                        help=("beta parameterization used by the checkpoint/model; "
                              "final TabERA uses sigmoid"))
    parser.add_argument("--head_input_scale", type=str, default="unit",
                        choices=["unit", "auto", "matched"],
                        help=("head-input scale; final TabERA uses auto with unit_tangent"))
    parser.add_argument("--tie_rule", type=str, default="first",
                        choices=["first", "latest"],
                        help=("checkpoint tie rule for validation accuracy; final benchmark uses first"))
    parser.add_argument("--params_variant", type=str, default=None,
                        choices=["legacy"],
                        help="select a legacy HPO-study source without changing the target structure")
    parser.add_argument("--params_geometry", type=str, default=None,
                        choices=["additive", "chord", "tangent", "unit_tangent"],
                        help=(
                            "read HPO parameters from a study built with another correction geometry"))
    parser.add_argument("--allow_pb_mismatch", action="store_true",
                        help="allow P/batch-size mismatches when borrowing HPO parameters across seeds")
    parser.add_argument("--from_saved_state", type=str, default=None,
                        help=(
                            "give the *_model_state.pt path saved by an earlier run to "
                            "skip training entirely, restore that state and rerun only "
                            "--explain and the analyses. The optimize.py study file is "
                            "not needed either, since model_kwargs is read from this "
                            "file. Other arguments such as --n_explain still apply. "
                            "seed and openml_id must match what was saved for the data "
                            "split to agree -- the values given on this command line "
                            "are used, so pass the same ones."
                        ))
    parser.add_argument("--linear_probe", action="store_true",
                        help=(
                            "extract query_emb and context_emb from a model loaded via "
                            "--from_saved_state and fit sklearn LogisticRegression "
                            "(classification) or Ridge (regression) on each separately, "
                            "then compare test performance. This distinguishes whether "
                            "a representation lacks the information entirely from "
                            "whether the head simply does not use it: in the first case "
                            "the standalone probe scores far below query_emb, in the "
                            "second it comes close. TabERA itself is not retrained -- "
                            "only embeddings are extracted and sklearn is fitted "
                            "separately. --from_saved_state is not required; without "
                            "it, the model just trained is used."
                            "when routing is ambiguous, as a statistic over the whole set."
                        ))
    parser.add_argument("--calibration_analysis", action="store_true",
                        help=(
                            "compare routing confidence (layer 1, relative dominance in "
                            "prototype space) and prediction confidence (the classifier "
                            "softmax) against actual accuracy over the whole test set. "
                            "Unlike --explain, which walks individual samples, this "
                            "answers whether the final prediction is trustworthy even "
                            "A flat accuracy curve across routing-confidence bins is "
                            "evidence that retrieval and fusion compensate for routing "
                            "uncertainty; a high ECE on prediction confidence -- "
                            "especially accuracy falling short of confidence in the top "
                            "bins -- indicates overconfidence. TabERA is not retrained."
                        ))
    parser.add_argument("--allow_self_retrieval", action="store_true",
                        help=(
                            "do not exclude self-retrieval. By default a MemoryBank "
                            "slot whose sample_id matches the query (the query itself, "
                            "stored in an earlier epoch) is dropped from the candidates; "
                            "this flag restores the older behaviour of keeping it. "
                            "Excluding is the more correct implementation because "
                            "MemoryBank stores and returns the label, so a self-retrieved "
                            "slot hands back the query's own ground truth. Note that the "
                            "outlier path (very large centroid groups, rare) still does "
                            "not apply the exclusion even with the default on, so use "
                            "this flag to compare exactly against older results."
                            "when it is on -- use this flag to compare exactly against "
                            "results produced under the older behaviour."
                        ))
    parser.add_argument("--cat_combine", type=str, default="onehot", choices=["sum", "concat", "onehot"],
                        help=(
                            "how categorical embeddings are combined. 'onehot' (the "
                            "default) follows the TabR/ModernNCA line: plain one-hot with "
                            "no learned parameters, one reserved span per column and no "
                            "mixing. 'sum' adds per-column embeddings of width embed_dim "
                            "-- the initial implementation, kept for compatibility with "
                            "older checkpoints. 'concat' is the original Guo & Berkhahn "
                            "(2016) form: small per-column embeddings (--cat_embed_dim) "
                        ))
    parser.add_argument("--cat_embed_dim", type=int, default=16,
                        help="per-column embedding width when cat_combine=concat.")
    parser.add_argument("--num_embedding", type=str, default="ple",
                        choices=["linear", "ple", "plr_lite"],
                        help=(
                            "numeric feature encoding. 'ple' (the default) is "
                            "PiecewiseLinearEmbeddings(activation=False, Gorishniy et al. 2022) — "
                            "the same structure TabM (Gorishniy et al. 2024) recommends "
                            "by default: a learnable per-feature (n_bins, d_embedding) "
                            "weight contracted with the bin encoding. It previously "
                            "emitted the raw bin vector (PiecewiseLinearEncoding), which "
                            "differed from the TabM default. Measured on four datasets "
                            "(profb/vehicle/credit-g/jasmine): collapsed validation "
                            "trials fell to zero against three for PLR, and dropping the "
                            "PLR hyperparameters shrank the HPO search space. Top-5 test "
                            "performance still favoured PLR on 3 of 4, and centroid "
                            "margin_percentile was lower under PLE on all four (cause "
                            "unknown) -- so the case for PLE is avoiding catastrophic "
                            "failure and simplifying the search, not a performance win. "
                            "'plr_lite' is the previous default; 'linear' projects raw values directly."
                        ))
    parser.add_argument("--num_bins", type=int, default=8,
                        help="bins per column when num_embedding=ple (default 8, changed "
                             "from 48 after better calibration was observed on several datasets).")
    parser.add_argument("--correction_geometry", type=str, default="additive",
                        choices=["additive", "chord", "tangent", "unit_tangent"],
                        help=(
                            "Phase A arm: the geometry of d = h - c. Must match the "
                            "value optimize.py ran with -- it selects the study file "
                            "(..geom=NAME) and, if the study records a different arm, "
                            "this run stops rather than train a different model than "
                            "the one that was tuned. Ignored with --from_saved_state, "
                            "where the value stored in the checkpoint's model_kwargs "
                            "applies."))
    parser.add_argument("--early_stop_metric", type=str, default="accuracy",
                        choices=["accuracy", "logloss", "auroc", "bacc"],
                        help=(
                            "Validation metric that selects best_state and drives the "
                            "patience counter. Must match what optimize.py ran with: "
                            "it selects the study file (..esm=NAME) and this run stops "
                            "if the study recorded a different one. Default 'accuracy' "
                            "is the legacy behaviour. No effect with "
                            "--from_saved_state, which skips training."))
    parser.add_argument("--plr_n_frequencies", type=int, default=16,
                        help="number of periodic frequencies per column when num_embedding=plr_lite (default 16).")
    parser.add_argument("--plr_freq_scale", type=float, default=0.01,
                        help="frequency initialisation scale for num_embedding=plr_lite "
                             "(default 0.01; the TabR paper suggests LogUniform[0.01, 100.0]).")
    parser.add_argument("--plr_out_dim", type=int, default=8,
                        help="output width per column when num_embedding=plr_lite (default 8).")
    parser.add_argument("--regroup_log_every", type=int, default=10,
                        help=(
                            "how often to print the [Regroup] log, in epochs (default "
                            "10). Lower it to 1 or 2 to follow active_ratio and reinit "
                            "more closely -- for instance when a 10-epoch interval "
                            "leaves it unclear whether the final stretch actually "
                            "settled. No effect with --from_saved_state, which skips training."
                        ))
    parser.add_argument("--k_override", type=int, default=None,
                        help=(
                            "override only k (the number of retrieved neighbours) from "
                            "best_params and retrain with everything else unchanged. "
                            "Mainly used to separate whether a drop under sharp "
                            "attention comes from the sharpening itself or from a large "
                            "k where only a few neighbours end up used, inflating the "
                            "estimator variance (at k=48 with n_eff around 1.4, the "
                            "variance is over 30x that of averaging 48). It changes "
                            "weight shapes, so it breaks loading under --from_saved_state."
                        ))
    parser.add_argument("--embed_dim_override", type=int, default=None,
                        help=(
                            "override embed_dim from best_params and retrain with "
                            "everything else unchanged. Across jasmine, mfeat-zernike "
                            "and ada_agnostic, HPO runs that moved embed_dim up (to 256) "
                            "consistently reduced the retrieval-branch gradient and "
                            "runs that moved it down (to 64) increased it (jasmine "
                            "64->256 down, mfeat-zernike 256->64 up, ada_agnostic "
                            "128->256 down). The loss weights moved the same way on all "
                            "three and could not explain the split, whereas embed_dim "
                            "split in the matching direction. HPO changed dropout, lr "
                            "and layer count alongside it, so this is a correlation and "
                            "the flag isolates embed_dim alone. Note that embed_dim "
                            "changes the weight shapes, so a checkpoint saved at a "
                            "different embed_dim cannot be loaded -- and "
                            "--from_saved_state skips retraining anyway."
                        ))
    parser.add_argument("--dropout_override", type=float, default=None,
                        help=(
                            "override dropout from best_params and retrain with "
                            "everything else unchanged. dropout sits inside "
                            "TabularEmbedder (ResidualMLP) and perturbs query_emb on "
                            "every forward, so this checks whether it is one cause of "
                            "routing churn -- the repeated dead/reinit cycle of "
                            "centroids. No effect with --from_saved_state."
                        ))
    parser.add_argument("--study_batch_size", type=int, default=None,
                        help=("Point at a study produced by "
                              "`optimize.py --batch_size N` (filename tag "
                              "..B{N}). Selects **which study file to load**; "
                              "it does not change training. Leave unset for "
                              "protocol runs -- those studies carry no ..B tag. "
                              "Do not confuse with --batch_size_override, which "
                              "changes the batch size used when retraining."))
    parser.add_argument("--batch_size_override", type=int, default=None,
                        help=(
                            "override batch_size from best_params and retrain with "
                            "everything else unchanged. Used to gather evidence for "
                            "dropping batch_size from the HPO search in favour of a "
                            "fixed value chosen by dataset size (the standard practice "
                            "in the TabR line): sweep several values (64/128/256/512) "
                            "on the same best_params to see how sensitive validation "
                            "performance is and how that relates to dataset size. "
                            "No effect with --from_saved_state. It is applied to "
                            "best_params (TabERAWrapper.params), not model_kwargs, "
                            "since batch_size affects the training loop only."
                        ))
    parser.add_argument("--regroup_warmup_epochs_override", type=int, default=None,
                        help="override centroid regroup warmup epochs and retrain")
    parser.add_argument("--disable_dead_reinit", action="store_true",
                        help="disable dead-centroid reinitialization")
    parser.add_argument("--dead_reinit_patience_override", type=int, default=None,
                        help="override dead-centroid reinitialization patience")
    parser.add_argument("--dead_reinit_noise_scale_override", type=float, default=None,
                        help="override dead-centroid reinitialization noise scale")
    parser.add_argument("--refresh_on_best", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="re-encode training memory after restoring the best checkpoint")
    parser.add_argument("--shap_background", type=int, default=50,
                        help="background-sample count for KernelSHAP diagnostics")
    parser.add_argument("--shap_nsamples", type=int, default=None,
                        help="perturbation-sample count for KernelSHAP; None uses SHAP auto")
    parser.add_argument("--shap_repeats", type=int, default=1,
                        help="repeat KernelSHAP diagnostics to estimate Monte Carlo variability")
    args = parser.parse_args()

    # Deterministic mode is strict by default; unsupported CUDA operations raise.
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        print(f"  [--deterministic] cudnn.deterministic=True, benchmark=False, "
              f"use_deterministic_algorithms(True, warn_only={args.deterministic_warn_only})"
              + (f" -- CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '(unset!)')}"
                 if torch.cuda.is_available() else " (no CUDA; on CPU most operations are deterministic anyway)"))
    # CUDA_VISIBLE_DEVICES exposes the selected physical GPU as logical cuda:0.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── Load the data ─────────────────────────────────────
    with open(args.json, "r") as f:
        data_info = json.load(f)

    openml_id    = str(args.openml_id)
    dataset_info = data_info[openml_id]
    tasktype     = dataset_info["tasktype"]
    print(f"[TabERA Reproduce] {dataset_info['fullname']} (id={openml_id}, task={tasktype})")

    # Track data-loading time separately from model fitting.
    _t_data_start = time.time()
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    print(f"  [timing] dataset load: {time.time() - _t_data_start:.1f}s")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std      = dataset.y_std
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")

    # ── Load the best params ───────────────────────────────
    # Matches the save path used by optimize.py
    if not args.savepath.endswith("optim_logs"):
        log_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
    else:
        log_dir = args.savepath
    # Ensure the current seed output directory exists.
    os.makedirs(log_dir, exist_ok=True)

    # params_seed changes the study source only; outputs stay in log_dir.
    if args.params_seed is not None and args.params_seed != args.seed:
        if args.savepath.endswith("optim_logs"):
            raise ValueError("--params_seed cannot be used when --savepath itself ends "
                             "with optim_logs because seed-specific study folders cannot be separated.")
        study_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.params_seed}")
        print(f"  [params_seed] study source: {study_dir}  |  outputs: {log_dir}")
    else:
        study_dir = log_dir

    # Optional repeated training on the same data split and HPO configuration.
    if args.train_seeds:
        if args.from_saved_state:
            raise ValueError(
                "--train_seeds cannot be combined with --from_saved_state: "
                "the latter loads a single checkpoint saved at one seed, so "
                "iterating over seeds is meaningless. Use --train_seed for a "
                "single seed."
            )
        train_seed_list = args.train_seeds
    else:
        train_seed_list = [args.train_seed if args.train_seed is not None else args.seed]
    # Used by run_single_seed() to word its log lines (single run vs one of
    # several seeds). Attached to args temporarily; not a CLI option.
    args._train_seed_list = train_seed_list

    if args.explain_seed is not None:
        if args.explain_seed not in train_seed_list:
            raise ValueError(
                f"--explain_seed={args.explain_seed} is not in --train_seeds({train_seed_list})."
            )
        explain_seed = args.explain_seed
    else:
        explain_seed = train_seed_list[-1]

    results = []
    for _ts in train_seed_list:
        do_analysis = (_ts == explain_seed)
        result = run_single_seed(
            dataset, X_train, y_train, X_val, y_val, X_test, y_test, y_std,
            output_dim, tasktype, openml_id, dataset_info, device, log_dir, env_info,
            args, _ts, do_analysis, study_dir=study_dir,
        )
        results.append(result)

    # With two or more seeds, print a mean +- std summary. reproduce.py now
    # also covers re-confirming the best config across several initialisations
    # (robust evaluation), so it ends with a summary rather than a list of
    # per-seed numbers.
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary across {len(results)} train_seeds: {train_seed_list}")
        print(f"{'='*60}")
        for split_name, key_dict_name in [("val", "val_metrics"), ("test", "test_metrics")]:
            metric_keys = sorted(results[0][key_dict_name].keys())
            for key in metric_keys:
                vals = np.array([r[key_dict_name][key] for r in results])
                indiv = ', '.join(f"{v:.4f}" for v in vals)
                print(f"  {key:16s} mean={vals.mean():.4f}  std={vals.std():.4f}  (seeds: {indiv})")



if __name__ == "__main__":
    main()
