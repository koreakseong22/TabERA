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
from libs import diagnostics as diag
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

    Both the top and the runner-up go through this one function so the two
    cannot drift apart in style -- they previously used different bracket
    conventions.
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


# _select_query_similar_features and _select_query_dissimilar_features were
# merged into feature_gaps() in libs/diagnostics.py.
#
# The reason was not duplication but the max_gap = 0.15 threshold. It decided
# that any feature with gap > 0.15 was dropped from the candidates, so
# information was deleted **before** the result was shown -- displaying only
# why cases were similar and hiding where they differed. An unjustified
# constant deciding what may be seen is the same problem as a detector
# threshold.
#
# feature_gaps() now returns the gap for every feature, and sorting and the
# top-N cut happen only in print_explanation below, i.e. in the display
# layer.

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
                       max_features: int = 4,
                       max_gaps: int = 3,
                       verbose: bool = False,
                       max_dims: int = 5) -> None:
    """
    ⚠ Display layer. The max_* arguments below bound the number of console
      lines; they are not decision criteria and make no claim.
      libs/diagnostics.py returns the full ranking and the truncation happens
      only here. Analysis scripts should call diagnostics directly to see the
      whole distribution.

    ⚠ This function does not judge. Producing a sentence like "ambiguous
      region" would require a baseline drawn from an empirical distribution,
      and printing a single sample offers no reference distribution. It
      prints values only.
    """
    e = explanations[sample_idx]

    print(f"\n{'━'*52}")
    print(f"  TabERA Explanation — Sample #{sample_idx}")
    print(f"{'━'*52}")

    # Prediction confidence (the classifier softmax) is shown separately from
    # routing confidence (layer 1 below) from the outset, because they are
    # never the same value. The classifier uses information beyond routing, so
    # a prediction can be confident where routing was ambiguous and the other
    # way round. Putting both on screen together prevents conflating them.
    if pred_info is not None:
        print(f"\n  Prediction")
        print(f"     → {pred_info['pred_label']}")
        if pred_info.get("pred_confidence") is not None:
            print(f"     Prediction confidence: {pred_info['pred_confidence']:.1%}  "
                  f"(classifier output — separate from routing confidence below)")

        # ── Where the prediction came from ──────────────────────────
        # dev_head is a single Linear, so logits = (W*c + b) + W*(beta*r) is an
        # identity: the region baseline and the sample-specific term are the
        # two halves of the computation that produced the prediction, not an
        # attribution fitted to it afterwards.
        #
        # ⚠ The two probabilities are shown side by side and never subtracted.
        #   Each is a separate softmax, and softmax is non-linear, so their
        #   difference is not the sample's "contribution" in probability space.
        #   The *direction* is exact, since it is the sign of the logit term.
        _dv0 = e.get("prototype_deviation")
        if _dv0 is not None and _dv0.get("prob_final") is not None:
            _lb = pred_info.get("pred_label", "prediction")
            print(f"     Region baseline prediction: {_lb} {_dv0['prob_proto']:.1%}"
                  f"   (shared by every sample in this region)")
            print(f"     Final prediction:           {_lb} {_dv0['prob_final']:.1%}")
            # ⚠ The logit term is printed as a number, the two probabilities
            #   are not subtracted. In logit space the split is exact -- the
            #   two terms sum to the logit to floating-point error -- so the
            #   value is a real quantity, not a share inferred from the
            #   probabilities. Showing only a direction invited the reader to
            #   fill the gap by subtracting 55.8% from 56.1%, which is the one
            #   reading the decomposition does not support.
            _ld = _dv0.get("logit_dev")
            if _ld is not None:
                # ⚠ n_output = 1 for binary: the sign of the logit decides the
                #   class, so a positive term points at the positive class,
                #   not necessarily at the predicted one.
                if (tasktype == "binclass" and target_class_names
                        and len(target_class_names) == 2):
                    _toward = target_class_names[1] if _ld >= 0 else target_class_names[0]
                else:
                    # Multiclass: the value is the term on the predicted
                    # channel, so its sign is relative to that class directly.
                    _toward = _lb if _ld >= 0 else f"away from {_lb}"
                print(f"     Sample-specific logit:      {_ld:+.4f} toward {_toward}"
                      f"   (region baseline logit {_dv0['logit_proto']:+.4f})")
            # ⚠ This line belongs here, not in the region profile. It is a
            #   statement about the prediction, and it is the centre of the
            #   explanation wherever P < C: on ds=1493 the decision changes for
            #   70.5% of samples, against 0 of 800 on credit-g.
            if _dv0.get("argmax_changed"):
                _pc0 = _dv0.get("proto_pred")
                _pn0 = (target_class_names[_pc0]
                        if (target_class_names and _pc0 is not None
                            and 0 <= _pc0 < len(target_class_names)) else _pc0)
                print(f"     → the sample-specific component changes the decision: "
                      f"\"{_pn0}\" → \"{_lb}\"")

    # ① Prototype routing (target distribution — which class does this group represent?)
    proto = e["prototype"]
    print(f"\n  ① Region")

    # The group's target distribution, the main content of layer 1
    # (label_groups_by_target(), cached right after regroup_update()). To keep
    # it from overlapping with layer 2 (the neighbours' raw feature values),
    # it states only what kind of group this is, not a feature summary.
    tinfo = proto.get("target_info")
    if tinfo is not None:
        if tinfo["kind"] == "classification":
            target_str = _fmt_class(tinfo['top_class_name'], tinfo['top_count'], tinfo['n'], tinfo['top_prop'])
            if tinfo["second"] is not None:
                target_str += ", also " + _fmt_class(tinfo['second']['name'], tinfo['second']['count'],
                                                       tinfo['n'], tinfo['second']['prop'])
        else:
            target_str = (f"target mean {tinfo['group_mean']:.3g} "
                           f"(percentile {tinfo['percentile']:.0f}, n={tinfo['n']})")
    else:
        target_str = "(no group target info — target_labels may not have been cached during training)"

    # A bare "confidence" would be read as the classifier's prediction
    # confidence. This is a different quantity: how strongly the query prefers
    # its assigned centroid over the others, at the routing stage. margin,
    # others and cosine are shown alongside so the number is read in context
    # rather than on its own.
    print(f"     Assigned prototype: \"{proto['assigned_group']}\"")
    # Routing confidence / margin / cosine similarity are not printed. The
    # values do vary per sample, but **without a reference they cannot be
    # judged**: at P = 28 the uniform share is 3.6%, and there is no way to
    # tell whether 16.5% is high or low. The routing distribution below
    # carries the same information in comparable form, with the runner-up
    # groups' label distributions attached. The raw values remain in
    # explanations[b]['prototype'] for diagnostics.
    print(f"     Prototype label distribution: {target_str}")

    if proto["runners_up"]:
        print(f"     Routing distribution:")
        print(f"       • {proto['assigned_group']:<20s} {proto['routing_confidence']:>6.1%}  (assigned)")
        # ⚠ A prototype with no members is not shown even when its routing
        #   probability is high. `(no target info)` does not mean the
        #   explanation is thin; it means there is nothing to explain (the
        #   prototype is dead or was just reinitialised). To a reader that is
        #   confusion, not information.
        #   ⚠ This is a validity filter in the display layer, not a model
        #     change: the routing probabilities are untouched and the Others
        #     mass is unaffected.
        _shown = 0
        for r in proto["runners_up"]:
            if r.get("target_info") is None:
                continue
            _shown += 1
            print(f"       • {r['label']:<20s} {r['routing_confidence']:>6.1%}  "
                  f"({_format_target_info(r['target_info'])})")
        if _shown < len(proto["runners_up"]) and verbose:
            _hid = len(proto["runners_up"]) - _shown
            print(f"       {_hid} prototype(s) with no members hidden"
                  f" (they have routing mass but no group to describe)")
        print(f"       • {'Others':<20s} {proto['others_mass']:>6.1%}")

    # Group means of the features that most distinguish this group from the
    # others (label_all_groups, top K by cross-group distinctiveness). Numeric
    # and categorical are listed separately: mixing them puts values on wholly
    # different scales -- a raw ratio next to a category code with a share --
    # on one line.
    labels = proto.get("group_feature_labels", [])
    if labels:
        num_strs, cat_strs = _split_by_kind(
            labels, get_kind=lambda fl: fl.kind,
            get_str=lambda fl: f"{fl.feature_name}={fl.label}",
        )
        print(f"     Characteristic features:")
        if num_strs:
            print(f"       numeric:     {',  '.join(num_strs)}")
        if cat_strs:
            print(f"       categorical: {',  '.join(cat_strs)}")

    # ② Local evidence — prototype-conditioned
    le = e.get("local_evidence")
    nbrs = e.get("neighbors")
    _single = False          # are the neighbours a single class? (set in layer 2)
    _skip_contrast = False   # has the summary line above already said 'no counter-example'?
    name_to_idx = {name: i for i, name in enumerate(col_names)} if col_names else {}

    # Layer 2 addresses three things at once.
    #
    # (a) What the neighbour label distribution means.
    #     Printing "6 of the 8 nearest are yes" is read as "so, yes". But
    #     within one prototype, even raw features do not beat the majority
    #     baseline (measured), so that 6/8 is sampling noise around the group
    #     distribution, not evidence. The one supported reading is how mixed
    #     this region is: after controlling for purity, entropy still predicts
    #     misclassification significantly on 6 of 6.
    #     -> Always print it beside the full group distribution and summarise
    #        it as relative ambiguity.
    #
    # (b) State the scope. This retrieval is NN(q, G_p), not NN(q, D).
    #     Writing only "similar cases" reads as a search over all the data.
    #
    # (c) Counter-examples. The display used to pick only the features close
    #     to the query (_select_query_similar_features dropped anything with
    #     gap > 0.15 from the candidates), showing why cases were similar and
    #     hiding where they differed -- confirmation bias built into the
    #     output. For a neighbour whose outcome differs from the prediction,
    #     the most **divergent** features are shown as well.
    _p = le.get("prototype") if le else None
    _scope = f"NN(q, G_{_p})" if _p is not None else "NN(q, G_p)"
    print(f"\n  ② Evidence")
    # ⚠ The five lines below are correct but do not belong on every sample.
    #   Across 14 cases they become 70 lines that nobody reads after the
    #   first. A rule for reading the output needs stating once, so it goes
    #   under verbose only -- it is a property of the explanation format, not
    #   a fact about this sample.
    if verbose:
        print(f"     Search is restricted to the assigned prototype group, not the whole training set.")
        print(f"     The neighbours' outcome distribution is not evidence for the prediction. Label")
        print(f"       separation inside a prototype is limited, so reading a neighbour majority as")
        print(f"       support would present sampling noise from the group distribution as a reason.")
        print(f"       What this section carries is one quantity: whether the decision was made in")
        print(f"       a typical part of its region or an ambiguous one.")

    # ── Local distribution vs group distribution ─────────────────
    def _fmt_label(v):
        if v is None:
            return "?"
        if tasktype == "regression":
            return f"{v:.4g}"
        code = int(round(v))
        if target_class_names and 0 <= code < len(target_class_names):
            return str(target_class_names[code])
        return str(code)

    if le is not None:
        if tasktype == "regression":
            if le.get("group_std") == le.get("group_std"):   # not nan
                print(f"\n     neighbourhood (k={le['n_neighbors']})  "
                      f"mean {le['local_mean']:.4g}  std {le['local_std']:.4g}")
                print(f"     whole group (n={le['group_size']})      "
                      f"mean {le['group_mean']:.4g}  std {le['group_std']:.4g}")
                _dr = le.get("dispersion_ratio")
                if _dr is not None and _dr == _dr:
                    print(f"     -> local / group entropy = {_dr:.2f}  "
                          f"(threshold to be set from the empirical distribution)")
        else:
            def _dist_str(counts, total):
                if not total:
                    return "(empty)"
                return ",  ".join(
                    f"{_fmt_label(float(k))} {c}/{total} ({c/total:.0%})"
                    for k, c in sorted(counts.items(), key=lambda x: -x[1]))
            _lc = {int(k): v for k, v in (le.get("label_counts") or {}).items()}
            _gc = {int(k): v for k, v in (le.get("group_label_counts") or {}).items()}
            # When all k neighbours share one label, three things hold
            # **automatically at once**: H(label) = 0, relative ambiguity 0,
            # and no contrasting case. Stating them separately says the same
            # fact three times (7 of 14 credit-g cases), so they collapse into
            # one sentence. The group distribution stays, since it provides
            # the contrast.
            # ⚠ A single class does **not** mean "no counterexample". The
            #   neighbours can all be the class **opposite** the prediction
            #   (measured on synthetic data: predicted yes, 6 of 6 neighbours
            #   no), in which case every one of them is a counterexample. The
            #   collapse is only valid after checking agreement with the
            #   predicted class.
            _one = (len(_lc) == 1 and le["n_neighbors"] > 0)
            _pcode = (pred_info or {}).get("pred_code")
            _same_as_pred = (_one and _pcode is not None
                              and next(iter(_lc)) == int(_pcode))
            _single = _one and _same_as_pred
            # The label mentions that this is a label-distribution entropy.
            # Printing a bare 'entropy' would be indistinguishable from the
            # evidence_w entropy.
            # ⚠ Both distributions are printed **always**, even when the
            #   neighbourhood collapses to one class. A previous version
            #   replaced the neighbourhood line with a sentence in that case,
            #   which dropped the very comparison this block exists for: with
            #   only the group line left, "all 8 neighbours disagree" reads as
            #   a fact about 8 samples when the region is entirely that class
            #   -- a far stronger statement. The sentence is now a note
            #   *after* the numbers, not a replacement for them.
            _skip_contrast = _single
            print(f"\n     neighbourhood (k={le['n_neighbors']})   "
                  f"{_dist_str(_lc, le['n_neighbors'])}   H(label) {le['label_entropy']:.3f}")
            print(f"     region                "
                  f"{_dist_str(_gc, le['group_size'])}   H(label) {le['group_label_entropy']:.3f}")
            if _single:
                print(f"     → single-class region, no contrasting case")
            elif _one:
                # Every neighbour disagrees with the prediction. Whether the
                # region as a whole also disagrees changes what this means, so
                # the note says which of the two it is.
                _whole = (len(_gc) == 1)
                print(f"     ⚠ the prediction is "
                      f"{(pred_info or {}).get('pred_label', '?')} while "
                      + ("the neighbourhood and the entire region are"
                         if _whole else "every neighbour is")
                      + f" {_fmt_label(float(next(iter(_lc))))}")
            _ar = le.get("ambiguity_ratio")
            if (not _single) and _ar is not None and _ar == _ar:
                # ⚠ No judgement is made. Phrases like "a more mixed region"
                #   used to be attached at 1.15 and 0.85, values with no
                #   justification behind them. Where "high" begins has to come
                #   from the empirical distribution over the evaluation set,
                #   which is the analysis scripts' job.
                print(f"     -> relative ambiguity {_ar:.2f}  "
                      f"(local entropy / group entropy; the threshold for "
                      f"calling a region ambiguous is not fixed here)")

    # ── Case list: supporting and contrasting ────────────────────
    def _fmt_cat_value(name: str, code_val: float) -> str:
        # ⚠ A failed name lookup must not kill the whole explanation. Losing
        #   a name costs display quality, but an exception here removes the
        #   entire explanation for that sample. Whether the mapping is a list
        #   or a dict, and whether the code is out of range, everything falls
        #   back safely to "Category N".
        names_for_col = cat_category_names.get(name) if cat_category_names else None
        code = int(code_val)
        try:
            if names_for_col is not None:
                nm = (names_for_col[code] if not isinstance(names_for_col, dict)
                      else names_for_col.get(code))
                if nm is not None:
                    # ⚠ The LabelEncoder integer is not shown. It is an
                    #   internal index with no meaning to a reader, and it
                    #   appeared in every categorical line of layers 2 and 3.
                    return f"{name}={nm}"
        except (IndexError, KeyError, TypeError):
            pass
        return f"{name}=Category {code}"

    def _fmt_num_value(name: str, uniform_val: float) -> str:
        if quantile_transformer is not None and num_cols is not None and name in name_to_idx:
            real_val = inverse_transform_numeric(quantile_transformer, num_cols,
                                                  name_to_idx[name], uniform_val)
            if real_val is not None:
                return f"{name}={real_val:.3g}"
        return f"{name}={uniform_val:.3f}"

    def _fmt_items(items):
        num_strs, cat_strs = _split_by_kind(
            items, get_kind=lambda it: it[2],
            get_str=lambda it: (_fmt_cat_value(it[0], it[1])
                                 if it[2] == "categorical" else _fmt_num_value(it[0], it[1])),
        )
        return num_strs, cat_strs

    if not nbrs:
        print(f"\n     (no neighbours — the memory bank held fewer than k, or this "
              f"is output from a version without the 'neighbors' key)")
    else:
        # Split by agreement with the prediction. Case-based explanation only
        # works if counterexamples are visible as counterexamples.
        _pc = (pred_info or {}).get("pred_code")
        if tasktype == "regression" or _pc is None:
            groups = [("Cases (no outcome contrast)", nbrs)]
        else:
            sup = [n for n in nbrs
                   if n["label"] is not None and int(round(n["label"])) == int(_pc)]
            con = [n for n in nbrs
                   if n["label"] is not None and int(round(n["label"])) != int(_pc)]
            groups = [("Outcome-matched cases", sup),
                      ("Outcome-contrasting cases", con)]

        for title, rows in groups:
            if not rows:
                # If "single-class region (no counterexample)" was already
                # stated above, the same fact is not printed twice.
                if title.startswith("Contrasting") and not _skip_contrast:
                    print(f"\n     Outcome-contrasting cases — none "
                          f"(all {len(nbrs)} nearest share the prediction)")
                continue
            print(f"\n     {title}")
            _is_con = title.startswith("Contrasting")
            for nb in rows[:max_neighbors]:
                sid = nb.get("sample_id")
                sid_str = (f"train #{sid}" if sid is not None and sid >= 0
                           else f"mem #{nb['memory_idx']}")
                print(f"       #{nb['rank']+1}  similarity={nb['similarity']:.3f}"
                      f"   → {_fmt_label(nb.get('label'))}   [{sid_str}]")
                # A gap <= 0.15 filter used to decide which features were
                # even candidates. Now the full set of gaps arrives and only
                # the closest max_features are taken -- no constant decides
                # what may be shown.
                gp = nb.get("gaps") or []
                if gp:
                    near = sorted(gp, key=lambda g: g["gap"])[:max_features]
                    ns, cs = _fmt_items([(g["name"], g["neighbor_value"], g["kind"])
                                          for g in near])
                    if ns:
                        print(f"            close on numeric:     {', '.join(ns)}")
                    if cs:
                        print(f"            close on categorical: {', '.join(cs)}")
                # A counterexample always shows where it differs.
                # ⚠ A gap of 0 is not a difference. Simply taking the largest
                #   gaps would print "duration 48 -> 48" even when every gap
                #   is 0 (duplicate rows, or an anomaly such as
                #   self-retrieval), leaving the reader asking what differs.
                #   An eps filter also absorbs floating-point error.
                _GAP_EPS = 1e-9
                df = ([(g["name"], g["neighbor_value"], g["kind"], g["delta"])
                       for g in sorted(gp, key=lambda g: g["gap"], reverse=True)
                       if g["gap"] > _GAP_EPS][:max_gaps]
                      if gp else [])
                if _is_con and df:
                    # Numeric values live in quantile space ([0,1] uniform),
                    # so a difference reads directly as a **percentile**
                    # difference (-0.498 is about 50 percentiles lower). Both
                    # endpoints are also shown in their original units, so the
                    # axis matches the other numeric displays in layers (1)
                    # and (2), which are inverse-transformed.
                    for name, nval, kind, delta in df:
                        if kind == "categorical":
                            qv = nval - delta
                            print(f"            differs: {name}  "
                                  f"query {_fmt_cat_value(name, qv).split('=', 1)[1]}"
                                  f" -> neighbour {_fmt_cat_value(name, nval).split('=', 1)[1]}")
                        else:
                            qv = nval - delta
                            print(f"            differs: {name}  "
                                  f"query {_fmt_num_value(name, qv).split('=', 1)[1]}"
                                  f" -> neighbour {_fmt_num_value(name, nval).split('=', 1)[1]}"
                                  f"   ({delta*100:+.0f} pct)")
            if len(rows) > max_neighbors:
                print(f"       ... (+{len(rows) - max_neighbors} more)")

    # ⚠ The evidence_w (attention weight) output was removed. The final model
    #   has no aggregator, so those weights are a uniform constant and their
    #   entropy is fixed at log(k); reading that as "the neighbours were used
    #   evenly" would be reading a fact that is not there. What layer (2) can
    #   speak to is the neighbours' **label distribution** (local_evidence).

    # Level 3: retrieval signal magnitude.
    # Deliberately not called a "contribution": there is no guarantee that a
    # norm is proportional to the actual effect on the prediction, the same
    # reason the naming was corrected in layer (2) above.
    # What is reported here is magnitude only, not causal attribution.
    # ③ Query-direction correction (Level 2.5)
    # Section 4-2: the argmax agrees 99.4% of the time, but this term decides
    # the probability. Using the confidence in an explanation while hiding the
    # term that sets it would be a faithfulness problem.
    #
    # ⚠ Naming correction (section 9). This was called "Prototype-relative
    #   Deviation", which is inaccurate: ||c|| is fixed at 1 while ||q||
    #   ranges from 7 to 1197, so
    #       r = normalize(q − c) ≈ normalize(q) = q̂
    #   and the measured cos(r, q) is 0.994 to 1.000 across every dataset and
    #   both update rules. Subtracting c has essentially no effect on the
    #   direction, so this term is a **query-direction correction**, not a
    #   deviation from the prototype. The decomposition identity
    #   (logits = W*c + W*(beta*r)) remains exact.
    dv = e.get("prototype_deviation")
    if dv is not None:
        print(f"\n  ③ Region profile")
        # ⚠ This decomposition is not an approximation. dev_head is a single
        #   Linear, so logits = (W*c + b) + W*(beta*r) is an identity and the
        #   two terms always sum to the actual logits (residual 0.000e+00 in
        #   the smoke test). It differs in kind from SHAP or IG, which pick a
        #   baseline and approximate.
        if verbose:
            print(f"     (dev_head is a single Linear, so logits = (W·c + b) + W·(β·r) — an exact decomposition)")
            print(f"     r = normalize(q−c), but ‖q‖ >> ‖c‖ = 1, so it points along q"
                  f" (cos(r, q) ~ 1.00). It is not a deviation from the prototype.")
        # Reports the probability shift rather than a share of the deviation.
        #
        # ⚠ dev_share is a ratio of logit magnitudes and **overstates the
        #   effect**. Measured on credit-g it reads 5.6-19.3% while the actual
        #   confidence moved 0.2-1.2 percentage points, because the logits sit
        #   in +-0.6 where the sigmoid is nearly linear. People read
        #   probabilities, so probabilities are shown.
        #
        # ⚠ The "decision unchanged" line stays. On credit-g it is dead
        #   weight -- 800 of 800 are unchanged -- but where P < C (ds=1493:
        #   35 prototypes for 100 classes) 70.5% change, and there this line
        #   becomes the centre of the explanation.
        if dv.get("prob_final") is not None:
            _pp, _pf = dv["prob_proto"], dv["prob_final"]
            _lab = (pred_info or {}).get("pred_label", "prediction")
            # ⚠ This value depends only on W*c, so it is **identical for
            #   every sample assigned to the same prototype** (measured on
            #   credit-g: all 7 samples of Centroid_16 read 65.1%). Calling it
            #   "prototype only" would suggest a per-sample number, so the
            #   name states that it is a group-level baseline. What varies per
            #   sample is the shift.
            # ⚠ Printed as two independent predictions, never as an additive
            #   split. prob_proto is softmax(W*c + b) and prob_final is
            #   softmax(W*h + b); the decomposition is exact in logit space,
            #   but softmax is non-linear, so 72.1% + 1.7% = 73.8% does not
            #   hold. Showing a "+1.7%p contribution" would assert exactly
            #   that. The direction, by contrast, is exact -- it follows the
            #   sign of the logit term.
            if verbose:
                print(f"     region baseline prediction: {_lab} {_pp:.1%}"
                      f"   (shared by every sample in this region)")
                print(f"     final prediction:           {_lab} {_pf:.1%}")
        else:
            # Regression has no probability, so the logit decomposition stands
            print(f"     prototype={dv['logit_proto']:+.4f}"
                  f"   query_dir={dv['logit_dev']:+.4f}"
                  f"   final={dv['logit_proto'] + dv['logit_dev']:+.4f}")

        # Deviation concentration and dim_contrib were removed from the
        # display. An embedding dimension number is something a reader can do
        # nothing with; across 25 cases it never proved useful. It still has
        # value as a reported statistic, so the full vector remains in
        # diagnostics.prototype_deviation()'s dim_contrib for analysis
        # scripts.

    # (3b) Group contrast in feature space -- the readable axis
    gc = e.get("group_stats")
    if gc and (gc.get("numeric") or gc.get("categorical")):
        print(f"\n     against the region (feature space)")
        if verbose:
            print(f"     (the group typical value is the inverse transform of a mean taken in quantile space, not an arithmetic mean)")
        # ⚠ A **different axis** from layer (3) above, which is the exact
        #   logit decomposition in embedding space. This is descriptive
        #   statistics, not attribution: "the prediction came out this way
        #   because of this feature" is not a sentence these values support.
        if verbose:
            print(f"     (descriptive statistics against the same group — not attribution; do not read causally with the decomposition above)")
        # diagnostics returns every feature sorted by |z|; the cut is here.
        for d in gc.get("numeric", [])[:max_features]:
            # Back to original units; without a quantile_transformer the
            # [0,1] percentile is shown as-is.
            _real = (lambda x: inverse_transform_numeric(
                        quantile_transformer, num_cols, d["feature_idx"], x)
                     ) if (quantile_transformer is not None and num_cols is not None) \
                    else (lambda x: None)
            _vr, _mr = _real(d["value"]), _real(d["group_mean"])

            def _fmt(x, fallback):
                # ⚠ 6.19e+03 is unreadable; a thousands separator is used.
                # ⚠ Rounding values that look integral (existing_credits = 1
                #   against a group mean of 1) produces "these are the same,
                #   so why is z -0.74?". They are actually 1.0 against 1.4, so
                #   one decimal place is kept.
                if x is None:
                    return f"{fallback:.3f}"
                ax = abs(x)
                if ax >= 1000:
                    return f"{x:,.0f}"
                if ax >= 10:
                    # ⚠ A 1e-9 tolerance trips on the inverse transform's
                    #   floating-point error (~1e-5) and prints 392 as
                    #   "392.0". The test uses a relative tolerance scaled to
                    #   the value.
                    return (f"{x:,.1f}" if abs(x - round(x)) > 0.01 * max(ax, 1.0)
                            else f"{x:,.0f}")
                return f"{x:.2f}".rstrip("0").rstrip(".")

            _v_s  = _fmt(_vr, d["value"])
            _mu_s = _fmt(_mr, d["group_mean"])
            # ⚠ Integer-valued features (existing_credits,
            #   installment_commitment) round to the same integer on both
            #   sides, producing "1 (group typical 1, z=-0.79)" -- identical
            #   numbers with a z attached (about 27% of numeric lines across
            #   14 cases). The reader asks why there is a z at all. The values
            #   are really 1 against 1.4. Writing "same as typical" would
            #   discard the 1.4, so the precision of the typical value is
            #   raised **until the two are distinguishable**.
            if (_v_s == _mu_s and _vr is not None and _mr is not None
                    and abs(_vr - _mr) > 1e-6):
                for _p in (1, 2, 3):
                    _cand = f"{_mr:,.{_p}f}"
                    if _cand != _v_s:
                        _mu_s = _cand
                        break
            # A ratio only means something for positive continuous quantities
            # (amounts, durations). It is uninterpretable across zero or
            # negative values, so it is omitted there.
            _ratio = ""
            if _vr is not None and _mr is not None and _mr > 0 and _vr > 0:
                _r = _vr / _mr
                if _r >= 1.15 or _r <= 0.87:      # a display cut, not a judgement
                    _ratio = f"{_r:.1f}x, " if _r >= 1 else f"{1/_r:.1f}x lower, "
            # ⚠ Do not call this a "mean". It is inverse_transform(mean(q)),
            #   the quantile-space mean mapped back, not an arithmetic mean in
            #   the original space. On a skewed distribution such as
            #   credit_amount it lands below the arithmetic mean, closer to
            #   the median. The quantile transform is monotone, so the **sign
            #   always agrees with z** and the comparison itself is valid.
            #   (z is computed in quantile space -- a different axis.)
            # ⚠ The display uses a **percentile**. z lives in quantile space
            #   while the value and typical value shown here are
            #   inverse-transformed to original units -- different axes, which
            #   is what produces "1 (typical 1, z=-0.79)" on discrete
            #   features. A percentile is invariant under a monotone
            #   transform, so both axes agree. z remains in the diagnostics
            #   return value for analysis.
            _pct = d.get("group_pct")
            if _pct is None:
                _pos = f"z={d['z']:+.2f}"
            elif _pct >= 0.5:
                # ⚠ "top 0%" reads as a contradiction. When nothing in the
                #   group is larger, say so instead of printing a share that
                #   rounds to zero.
                _pos = ("highest in the region" if _pct >= 0.9995
                        else f"top {(1 - _pct) * 100:.0f}%")
            else:
                _pos = ("lowest in the region" if _pct <= 0.0005
                        else f"bottom {_pct * 100:.0f}%")
            # ⚠ |group mean - global mean| / global std is printed on every
            #   line and never used to hide one. Near 0 means this group's
            #   distribution barely differs from the dataset, so "unusual
            #   within the group" reads the same as "unusual overall" -- the
            #   percentile beside it carries no group-specific information.
            #   Measured range: 0.09 (ds=31, largest group) to 0.71 (ds=1489).
            #   Gating on it would need a cut-off nobody can justify, and the
            #   reader could not tell a hidden feature from an absent one.
            _gvg = d.get("group_vs_global")
            _gvg_s = ("" if _gvg is None or _gvg != _gvg
                      else f",  |Δmean|/σ {_gvg:.2f}")
            print(f"       {d['feature_name']}={_v_s}"
                  f"   (group typical {_mu_s},  {_ratio}{_pos}{_gvg_s})")
        for d in gc.get("categorical", [])[:max_features]:
            # ⚠ Nothing is filtered. Values equal to the mode used to be
            #   skipped, but when the display layer decides what to hide, the
            #   reader cannot even tell that the feature was examined. This
            #   sample's share and the group mode are **always printed
            #   together**, leaving the judgement to the reader.
            # The model does not know the dataset's cat_category_names and
            # returns "Category N"; the mapping to real names happens here.
            _v  = _fmt_cat_value(d["feature_name"], d["value"]).split("=", 1)[1]
            _mv = _fmt_cat_value(d["feature_name"], d["group_mode"]).split("=", 1)[1]
            # ⚠ No |Δmean|/σ here. A categorical column has no standard
            #   deviation to divide by, and inventing one would attach a
            #   number that looks comparable to the numeric lines but is not.
            #   Prevalence and the group mode are the natural statistics.
            # ⚠ 0% is a real value, not a rounding artefact: the group is
            #   made of training rows and this sample is not one of them, so
            #   "no case in this region holds this value" can happen. Writing
            #   it as a percentage invites the reader to treat it as a
            #   vanishing share of something that exists.
            if d.get("absent_from_group"):
                _share = "no case in this region"
            else:
                _share = f"{d['group_freq']:.0%} of group"
            if not d.get("differs_from_mode", True):
                _same = "  (= mode)" if not d.get("ties_mode") else "  (ties the mode)"
            else:
                _same = ""
            print(f"       {d['feature_name']}={_v}"
                  f"   ({_share}; mode {_mv}, {d['group_mode_freq']:.0%}){_same}")

    # ── Position within the region, in representation space ─────────
    # The percentiles above answer "where does this sample sit in its region"
    # in the original columns. This answers the same question in the space the
    # region was actually formed in. The two can disagree, and that is worth
    # seeing: a sample can be central in the representation while holding an
    # extreme raw value, or the reverse.
    #
    # ⚠ Only present when the memory keys were refreshed (see the call site).
    #   Without that, memory holds training-time embeddings taken under a
    #   dropout mask while the query is deterministic, and the rank would be
    #   against a different representation.
    #
    # ⚠ No typicality verdict. The distance and its rank are shown; whether
    #   that makes the sample atypical is not decided here, since a cluster
    #   need not be spherical.
    _rp = e.get("region_position")
    if _rp is not None:
        # ⚠ group_pct is (d_group < d_me).mean(): the share of the region that
        #   is **closer to the centre than this sample**. A large value
        #   therefore means this sample is far out, not close in. Reading it
        #   the other way produced exactly inverted output -- a sample at
        #   distance 0.393, the farthest in its region, was described as
        #   "closer to the centre than 100% of the region".
        _farther = _rp["group_pct"]          # share of the region closer in
        _closer  = 1.0 - _farther            # share of the region farther out
        print(f"\n     position in the representation")
        # ⚠ Strict `<` means the extremes really can reach 0% and 100%. A
        #   percentage reads oddly there ("farther than 100%"), so the two
        #   ends are worded instead of numbered.
        if _farther >= 0.995:
            _where = "farther from the centre than every other case in this region"
        elif _closer >= 0.995:
            _where = "closer to the centre than every other case in this region"
        elif _farther >= 0.5:
            _where = f"farther from the centre than {_farther:.0%} of the region"
        else:
            _where = f"closer to the centre than {_closer:.0%} of the region"
        print(f"       distance to region centre {_rp['distance']:.3f}"
              f"   ({_where})")
        if verbose:
            print(f"       region spread: min {_rp['group_min']:.3f}"
                  f" / median {_rp['group_median']:.3f}"
                  f" / max {_rp['group_max']:.3f}   (cosine distance)")

    # The Representation Magnitude block was removed. beta is a model-level
    # constant and read literally the same on 25 of 25 samples (0.1039), and
    # ||query_emb|| is an absolute value with no reference point -- the block
    # carried no information. The values remain in
    # explanations[b]["retrieval_signal"] for diagnostics.

    print(f"{'━'*52}")


# ─────────────────────────────────────────────────────────────
# Integrated Gradients (Sundararajan et al. 2017, ICML) was removed.
# ─────────────────────────────────────────────────────────────
# compute_integrated_gradients and make_logit_target_fn were deleted:
#   1. IG breaks fundamentally on categorical features. The cast in
#      _encode_categorical() (x.round().long()) severs the autograd graph, so
#      the gradient of a categorical column is always exactly zero (reproduced
#      on a toy example). On an all-categorical dataset such as splice it
#      crashes outright with RuntimeError.
#   2. IG assumes a continuous path integral from baseline to input and does
#      not fit discrete inputs. The literature classifies it the same way:
#      the model must be differentiable, which rules out direct application to
#      discrete inputs without a workaround.
#   3. SHAP (Shapley values) evaluates the function repeatedly rather than
#      differentiating it, so the problem does not arise, and it has the
#      theoretical footing of being the unique allocation satisfying
#      efficiency, symmetry, dummy and additivity (Lundberg & Lee 2017).
# The SHAP computation lives inline inside the rank_correlation ablation (a
# model_predict closure plus shap.KernelExplainer); it is not reused widely
# enough to warrant a top-level function.


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

    # centroid_size: how many training/memory samples belong to the centroid
    # each sample was assigned to. Rather than concluding directly from "there
    # is a band with high routing confidence but low accuracy", this is the
    # minimum needed to check whether that band concentrates in one very large
    # centroid. assigned_centroid alone does not show the size.
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

        # A full train_count vs test_count table. Whether "only a few
        # centroids were used at test time" is a utilisation problem or simply
        # a task that concentrates in a few regions cannot be told apart
        # without the empty ones, so centroids with test_n = 0 are kept.
        # Unlike the correlation lists below, this table filters nothing.
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
            # Mean size of the centroids these samples were assigned to, so
            # "do the high- or low-confidence bands concentrate in large
            # centroids" can be read straight off the accuracy table without
            # plotting anything.
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

    # Flat is not asserted to be good unconditionally: accuracy has been
    # measured as non-monotone across percentiles, dropping sharply in the top
    # bins in particular. In that case the output does not claim to know the
    # cause and only points at what to look at next.
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



def run_single_seed(
    dataset, X_train, y_train, X_val, y_val, X_test, y_test, y_std,
    output_dim, tasktype, openml_id, dataset_info, device, log_dir, env_info,
    args, train_seed, do_analysis,
):
    """Train, evaluate and optionally analyse for one train_seed, given a
    dataset and HPO study that are both independent of train_seed and loaded
    once in main().

    optimize.py loads the dataset once and reuses it across 100 trials. This
    file used to reload it in every process run -- once per seed -- paying the
    OpenML fetch, NaN preprocessing, StratifiedKFold and QuantileTransformer
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
    # Moved here from main(), where it ran before the dataset load right after
    # train_seed was computed. Loading the dataset does not depend on
    # train_seed, so behaviour is identical -- and "this function reseeds with
    # this seed on every call" is clearer at the entry point.
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if len(getattr(args, '_train_seed_list', [train_seed])) > 1 or train_seed != args.seed:
        print(f"  [train_seed={train_seed}] seeds init and batch order (the data split still uses --seed={args.seed})")

    # ⚠ Any fusion_mode other than the default must appear in the filename.
    #   Listing modes one by one left proto_dev_retr (and proto_dev_vec,
    #   proto_only, ...) untagged, so their checkpoints overwrote the default
    #   run's: the study tag told them apart but the checkpoint name did not.
    # ── Filename tag ───────────────────────────────────────────
    # ⚠ Any non-default setting **must** appear in the filename, or outputs
    #   from different conditions overwrite each other under one name. Add new
    #   flags here as well.
    # ⚠ Tags for removed components (fusion_mode / L_nbr / aggregator /
    #   retr_proj ...) are absent; those outputs come from
    #   legacy/v3ema2_full/.
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
        # was written by this same code, not received from elsewhere, so
        # weights_only=False is stated explicitly.
        _saved_state = torch.load(args.from_saved_state, map_location=device, weights_only=False)
        # ⚠ Older checkpoints carry nbr_k / nbr_tau / nbr_neg_margin as
        #   constructor arguments, which were removed when they became
        #   constants.
        # ⚠ The old-checkpoint compatibility layer (strip_legacy_kwargs) is
        #   gone; load pre-cleanup .pt files from legacy/v3ema2_full/.
        model_kwargs = dict(_saved_state["model_kwargs"])
        # Print the **actual** structural settings of the loaded checkpoint.
        # meta.pkl records args, which disagree with the real model under
        # --from_saved_state: running an ablation on a checkpoint trained with
        # one setting still leaves the default in meta. A correct result was
        # once misread as "wrong checkpoint" because of this.
        _key_cfg = {k: model_kwargs.get(k) for k in
                    ("k", "embed_dim", "n_prototypes")
                    if k in model_kwargs}
        print(f"  [from_saved_state] actual checkpoint settings: "
              + ", ".join(f"{k}={v}" for k, v in _key_cfg.items()))
        best_params  = _saved_state.get("best_params", {})
        if best_params:
            print(f"  Params (as saved): {best_params}")
        # Files written before --from_saved_state existed have no memory_size
        # in model_kwargs: it used to be passed to TabERA(...) as a separate
        # kwarg and never merged into the dict. Rebuilding the model then
        # picks the TabERA default (10000), which disagrees with the
        # checkpoint's real size (n_train) and breaks loading. n_train was in
        # the old format, so it substitutes.
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
            # optimize.py --batch_size 로 만든 pilot study 를 가리킬 때만 준다.
            # 본 실험 study 는 태그가 없으므로 기본값 None 이 맞다.
            batch_size=args.study_batch_size,
        )
        fname = os.path.join(log_dir, f"data={openml_id}{_study_tag}..model=tabera.pkl")
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
        # ⚠ batch_size 는 optimize.py 가 space dict 에 직접 써넣는 값이라
        #   trial.suggest_* 를 거치지 않고, 따라서 study.best_params 에 없다
        #   (n_prototypes / k 와 같은 상황). 여기 기본값이 틀리면 HPO 와 최종
        #   학습이 서로 다른 batch size 로 돌게 된다.
        #
        #   [2026-08 프로토콜 정정] 예전 기본값은 256 이었다. 이제 batch size 는
        #   MultiTab 과 동일하게 get_batch_size(len(X_train)) 을 따르므로
        #   256 고정은 27개 벤치마크 중 25개에서 mismatch 를 만든다
        #   (예: credit-g -- HPO 는 B=64 로 탐색, 최종 학습은 B=256).
        #
        #   순서: optimize.py 가 기록한 실제 값이 있으면 그것을 쓰고, 없으면
        #   (프로토콜 태그 이전의 옛 study) 같은 규칙으로 다시 계산한다.
        _bs_actual = study.best_trial.user_attrs.get("batch_size_actual")
        if _bs_actual is None:
            _bs_actual = get_batch_size(len(X_train))
            print(f"  !  study에 batch_size_actual이 없습니다(옛 형식). "
                  f"get_batch_size(len(X_train))={_bs_actual} 로 재계산합니다.")
        best_params.setdefault("batch_size", int(_bs_actual))
        print(f"  Params: {best_params}")

        # PLE bin edges come from the train split only, to avoid leakage.
        num_bin_edges = None
        if args.num_embedding == "ple" and len(dataset.X_num) > 0:
            X_num_train = X_train[:, dataset.X_num]
            q = torch.linspace(0.0, 1.0, args.num_bins + 1, device=X_num_train.device)
            num_bin_edges = torch.quantile(X_num_train, q, dim=0).T.contiguous()

        # ── Build the model ────────────────────────────────────
        model_kwargs = params_to_model_kwargs(best_params, dataset.n_features, output_dim)

        # ── Overrides applied to the retrained model ────────────────────
        # ⚠ These were lost once. The CLI flags stayed defined and the study
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

    # ⚠ memory_size must equal n_train. Left at the default (10000), the ring
    #   buffer keeps accumulating each epoch until filled > n_train, and the
    #   slot indices in sample_groups run past the training arrays (measured:
    #   n_train=240 with filled=720 -> IndexError). Group-constrained
    #   retrieval also assumes the memory *is* the training split.
    model_kwargs.update(dict(
        memory_size=len(y_train),
        exclude_self_retrieval=(not args.allow_self_retrieval),
        # ⚠ Feature encoding. Without these, cat_col_idx is None and the model
        #   silently takes the raw-encoding path regardless of --cat_combine
        #   and --num_embedding -- categorical columns go in as LabelEncoder
        #   integers and the PLE bin edges computed just above are discarded.
        #
        #   These were lost once. optimize.py kept passing them while
        #   reproduce.py did not, so HPO tuned one architecture and the final
        #   run trained a different one. Nothing raised: cat_col_idx=None is a
        #   valid configuration, so the model built and trained without
        #   complaint on features it was never meant to see that way.
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

    # ── Train (skipped and restored under --from_saved_state) ───
    wrapper = TabERAWrapper(
        model, best_params, tasktype,
        device=str(device), epochs=args.epochs, patience=args.patience,
        # Checkpoint-selection timing. Defaults (False / 0) reproduce the
        # previous behaviour exactly.
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
    )
    wrapper._data_id = args.openml_id
    if _saved_state is not None:
        # ── Skip training; restore the saved state as-is ──────
        # ⚠ Loading an older checkpoint (proto_head) into the current model
        #   (dev_head, dev_beta_raw) with strict=True fails immediately.
        #   Passing strict=False instead would let typos and genuine
        #   mismatches through silently, producing a model that runs with
        #   weights that were never loaded -- more dangerous.
        #   So **only the intended missing keys** are allowed and anything
        #   else fails at once.
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
            # Older checkpoints store a 3-tuple (store, ptr, filled). Without
            # sample_ids everything is filled with -1 (unknown), and the ID
            # comparison in dual_space_faithfulness reports "cannot verify".
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
        # ⚠ Restoring sample_groups alone is not enough. What retrieve()
        # actually reads is memory._cached_groups, a plain attribute rather
        # than a registered buffer, so it never enters the state_dict. As a
        # result --from_saved_state runs were hitting the condition below and
        # searching **globally, with no group constraint**:
        #     if hard_assignment is None or cached is None or n < k:  (tabera.py:570)
        # Measured: the share of retrieved neighbours that fall inside the
        # query's own group was 0.235 on ds=1493, 0.528 on ds=46 and 0.668 on
        # ds=1489 -- it should be 1.000 when the constraint applies. Every
        # purity, margin, n_eff and topk diagnostic produced under
        # from_saved_state was therefore a global-search number. The cache is
        # rebuilt here.
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
        # memory.keys may still be noisy if --refresh_on_best was off when the
        # checkpoint was written (the default) or if it predates the flag. When
        # this run enables it, the refresh happens once here, right after
        # loading. If the checkpoint was already refreshed this simply rewrites
        # the same values -- effectively a no-op.
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
                        # 2) Routing drift. A low cosine does not affect the
                        #    routing analysis unless a centroid boundary was
                        #    crossed, so the two are measured separately.
                        #    A global rotation lowers cosine while assignments
                        #    hold; a boundary crossing lowers cosine and
                        #    changes them.
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
    else:
        wrapper.fit(X_train, y_train, X_val, y_val)

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
    # This used to forward all of X_test in one call. On datasets with heavy
    # centroid concentration (adult, where max_cluster_size runs into the
    # thousands) retrieve() then tried to process that whole cluster at full
    # batch width and the memory demand exploded -- a measured 25GB request,
    # ending in CUDA OOM. --calibration_analysis, which batches at 512,
    # finished on the same model and data; only this call site was unbatched.
    # It now follows the same pattern as run_calibration_analysis().
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
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "seed":        args.seed,
        "train_seed":  train_seed,
        # The HPO trials in optimize.py already store reinit_per_epoch and
        # active_ratio_std in study.pkl via trial.set_user_attr(), but they
        # were absent from this final run's meta.pkl -- so reading the
        # stability metrics of the one adopted model meant digging back into
        # study.pkl. These are copied straight from what the wrapper already
        # computed; no calculation changes.
        "centroid_geometry_diag": wrapper.centroid_geometry_diag,
        # The full per-epoch routing-stability series (active_ratio and
        # friends). Only centroid_geometry_diag -- a single final snapshot --
        # used to be stored, which made time-aligned analysis impossible.
        # regroup_history is recorded every epoch and can be zipped by epoch
        # with any other per-epoch series.
        "regroup_history": wrapper.regroup_history,
        # ⚠ Which epoch the returned weights come from. This had to be
        #   re-derived by scanning regroup_history for the max val_score,
        #   which only works when that series was recorded and breaks as soon
        #   as ties or a changed selection rule enter. It is the single most
        #   load-bearing number for the ds=46 / ds=31 selection analysis
        #   (best_epoch vs. the trajectory maximum of active_centroids), so it
        #   is recorded directly. selection_open_epoch is the first epoch that
        #   was eligible at all, which separates "nothing better existed" from
        #   "better existed but could not be selected".
        "best_epoch": getattr(wrapper, "best_epoch", None),
        "selection_open_epoch": getattr(wrapper, "selection_open_epoch", None),
        # ── Axis 2: prototype behaviour ──────────────────────────────
        # Metrics supporting the claim that a prototype is a density-driven
        # anchor rather than a class prototype, with a granularity that adapts
        # to task complexity.
        # ⚠ These used to be computed after the fact from checkpoints, which
        #   means reopening every checkpoint to build a results table and
        #   silently disagreeing with older numbers whenever the calculation
        #   changes. They are recorded at training time instead.
        "prototype_alignment": diag.prototype_class_alignment(model),
        "context_diversity":   diag.context_space_diversity(model),
        # ⚠ The dev_beta_raw trajectory recorded under --log_beta; an empty
        #   list when it was off. Console output alone cannot be re-analysed
        #   later, so it goes into meta as well.
        "beta_history": getattr(wrapper, "beta_history", []),
        "beta_lr_mult": getattr(wrapper, "beta_lr_mult", 1.0),
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
            }.items()
            if v != {"n_prototypes": None, "beta_lr_mult": 1.0,
                     "disable_dead_reinit": False,
                     "defer_early_stopping": False, "min_epochs": 0,
                     "epochs": HPO_TRAINING_SCHEDULE["epochs"],
                     "patience": HPO_TRAINING_SCHEDULE["patience"],
                     # ⚠ Changed from 0.005 to 0.0 when the default moved.
                     #   Afterwards 0 is the default and not a deviation.
                     "num_bins": 8,
                     "cat_combine": "onehot", "num_embedding": "ple"}[k]
        },
        # The --time_epoch measurements are stored for the same reason.
        "epoch_timing": getattr(wrapper, "_timing", {}),
        # ── optimizer update budget ────────────────────────────────
        # ⚠ Epoch count alone cannot compare training effort across datasets.
        #   With batch fixed at 256, steps per epoch scale with N and range
        #   from 2 to 16 (ds=54 takes 2 per epoch, ds=1489 takes 16 -- an
        #   eightfold gap). Early stopping absorbs some of this but does not
        #   control it, so the actual update count is recorded. It is needed
        #   whenever batch_size is revisited.
        "steps_per_epoch": (
            (len(X_train) // best_params["batch_size"])
            if best_params.get("batch_size") else None),
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only if args.deterministic else None,
        "exclude_self_retrieval": (not args.allow_self_retrieval),
        # The learned beta, so meta.pkl alone shows how much query-direction
        # correction this run settled on. It ranges from 0.10 to 0.73 across
        # datasets (section 12-6) and must be read alongside any reproduction.
        "dev_beta_final": float(
            torch.sigmoid(model.dev_beta_raw.detach()).mean().item()),
        # Diagnostic fields kept for meta.pkl compatibility. The fusion modes
        # they described were removed, so they stay None here.
                # Residual-fusion coefficients; that mode was removed, so None.
                        # Gate statistics from the gated_sum mode, which was
        # removed; the fields remain empty.
        "fusion_gate_mean_final": (
        ),
        "fusion_gate_var_final": (
        ),
        "fusion_gate_entropy_final": (
        ),
        # Gate temperature and pre-softmax logit statistics from the same
        # removed mode.
        "fusion_gate_logit_mean_final": (
        ),
        "fusion_gate_logit_gap_final": (
        ),
        "cat_embedding": True,  # records that categorical embedding is in use
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

        # Explanation material is built by the observer layer, not the model.
        # forward emits prediction state only (logits, topk_idx, context_emb,
        # neighbor_mask) and the four functions below observe it and
        # reconstruct the rest. Equivalence was verified numerically (maximum
        # error 3e-08).
        _nbrs = diag.retrieved_neighbors(model, out)
        _le   = diag.local_label_evidence(model, out)
        _pdv  = diag.prototype_deviation(model, out)
        _gst  = diag.group_relative_feature_stats(model, out, X_show)
        # ⚠ Only when the memory keys were refreshed. Otherwise memory.keys
        #   holds embeddings taken during training under a dropout mask while
        #   the query is deterministic -- a percentile computed across the two
        #   would rank the sample against a different representation. Measured
        #   group-structure agreement (ARI) before refresh: 0.006 on ds=1489.
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
                        help="use the same seed as optimize.py. It selects the data "
                             "split only: with KFold(random_state=42) fixed in "
                             "libs/data.py, it decides which fold becomes test, and "
                             "has no effect on init or batch order")
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
                        help=(
                            "retrain with torch.use_deterministic_algorithms(True), "
                            "cudnn.deterministic=True and cudnn.benchmark=False. It "
                            "separates whether the variance observed across "
                            "--train_seed (test score, active_ratio_std, reinit "
                            "count, early-stopping epoch) comes from GPU "
                            "non-determinism or from the architecture's own chaotic "
                            "sensitivity: with this on, repeat N runs varying only "
                            "--train_seed at a fixed --seed. If the variance largely "
                            "disappears, GPU non-determinism dominates; if it "
                            "remains, chaotic sensitivity does. "
                            "CUBLAS_WORKSPACE_CONFIG is set automatically before the "
                            "torch import (alongside --gpu_id) when this is on. Some "
                            "operations have no deterministic implementation and stop "
                            "with a RuntimeError; report the operation name."
                        ))
    parser.add_argument("--deterministic_warn_only", action="store_true",
                        help=(
                            "escape hatch for when --deterministic stops with a RuntimeError. "
                            "run operations without a deterministic implementation "
                            "anyway, warning instead of erroring. This breaks the "
                            "full determinism guarantee, so check the console "
                            "warnings. No effect without --deterministic."
                        ))
    parser.add_argument("--run_tag", type=str, default=None,
                        help=(
                            "arbitrary tag appended to the filename (e.g. 'r1', 'r2'). "
                            "When repeating an identical --seed / --train_seed / "
                            "--deterministic combination N times to measure pure GPU "
                            "non-determinism, the filenames would collide and "
                            "overwrite. Default None means no tag."
                        ))
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    parser.add_argument("--epochs",    type=int, default=HPO_TRAINING_SCHEDULE["epochs"],
                        help=(
                            "the default matches the HPO trials in optimize.py "
                            "(see HPO_TRAINING_SCHEDULE in libs/search_space.py). It "
                            "used to default to 200 against the HPO's 100, so a "
                            "script named after reproducing the best config actually "
                            "trained on a different schedule (measured on adult "
                            "(1590): reproduce.py trained longer yet reached a lower "
                            "val acc than the best HPO trial, with worse centroid "
                            "concentration). Override this explicitly to experiment "
                            "with a different schedule -- knowing that it is then a "
                            "separate experiment, not a reproduction."
                        ))
    parser.add_argument("--patience",  type=int, default=HPO_TRAINING_SCHEDULE["patience"],
                        help="default taken from HPO_TRAINING_SCHEDULE; see --epochs above.")
    parser.add_argument("--defer_early_stopping", action="store_true",
                        help=(
                            "[2026-08] Delay BOTH best_state selection and the "
                            "early-stopping patience counter until selection is "
                            "actually open (past regroup warmup and past "
                            "--min_epochs). Off by default so existing studies "
                            "reproduce unchanged.\n"
                            "Why: best_state was already gated on warmup, but "
                            "es.step() ran from epoch 1, so a warmup score that "
                            "later epochs could not beat spent patience on epochs "
                            "that were never selectable. Measured on ds=46: all "
                            "five low-utilisation runs reached active=50 at some "
                            "epoch, yet best_epoch landed at 8-13 with "
                            "active=14-24 and the run ended ~20 epochs later. "
                            "credit-g (31) has the same shape (best at epoch 9 "
                            "with active=2, trajectory max 28).\n"
                            "⚠ Not a fix for every low-utilisation run. On ds=934 "
                            "some runs have assign_change_rate falling to "
                            "0.01-0.04 with active decaying after the best epoch "
                            "and no later recovery -- there the best epoch really "
                            "is the best available. Check that a run's trajectory "
                            "max exceeds its active@best before expecting this to "
                            "help."))
    parser.add_argument("--min_epochs", type=int, default=0,
                        help=(
                            "[2026-08] Hard floor on the epoch at which checkpoint "
                            "selection may open, independent of "
                            "regroup_warmup_epochs (a short warmup does not "
                            "exclude the best_epoch=1 runs seen on ds=934). "
                            "Applies to best_state on its own; combine with "
                            "--defer_early_stopping to gate patience on the same "
                            "epoch. Ignored with --from_saved_state."))
    parser.add_argument("--n_explain", type=int, default=3,
                        help="number of test samples to explain")
    parser.add_argument("--n_prototypes", type=int, default=None,
                        help=("pass the same value used to build the study with "
                              "--n_prototypes in optimize.py; the study filename "
                              "carries a ..P{n} tag. Not needed with --from_saved_state."))
    parser.add_argument("--beta_lr_mult", type=float, default=1.0,
                        help=("learning-rate multiplier for dev_beta_raw (1.0 keeps "
                              "the previous behaviour). beta was measured rising "
                              "monotonically for the whole run and being cut off by "
                              "early stopping; this checks whether an equilibrium exists. The (0,1) bound stays."))
    parser.add_argument("--log_beta", action="store_true",
                        help=("record the dev_beta_raw value and gradient per epoch "
                              "and print them after training, to tell whether beta "
                              "staying near its initial value is an equilibrium or "
                              "a stall. Off by default: it syncs on every batch."))
    parser.add_argument("--time_epoch", action="store_true",
                        help=("print a table of cumulative time per epoch phase "
                              "(regroup_update, cache_sample_groups, feature_store "
                              "transfer, label computation) after training. Off by "
                              "default: accurate timing requires CUDA syncs."))
    parser.add_argument("--verbose", action="store_true",
                        help=(
                            "Also print the lines that describe how a run got "
                            "where it did rather than what it produced: which "
                            "study was loaded, memory refresh, and centroid "
                            "geometry. Training progress is controlled "
                            "separately by --regroup_log_every."))
    parser.add_argument("--explain_verbose", action="store_true",
                        help=("include the interpretation notes in the explanation "
                              "output. By default the fixed text that repeats for "
                              "every sample (5 warning lines in layer 2, 2 lines of "
                              "group-contrast notes, 1 decomposition note in layer 3) "
                              "is omitted -- the rules need reading once, and 14 samples of it exceed 100 lines."))
    parser.add_argument("--explain",   action="store_true",
                        help="print the feature explanation after training")
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
                        help=(
                            "set CentroidLayer.regroup_warmup_epochs and retrain "
                            "(default 0, i.e. active immediately, which is what has "
                            "been used in practice). Deferring regroup through the "
                            "early phase, when the straight-through estimator and "
                            "dead-prototype reinit are unstable, shows how routing "
                            "stability (active_ratio_std, reinit_per_epoch) and final "
                            "performance change. It is a model parameter, so it cannot "
                            "be combined with --from_saved_state."
                        ))
    parser.add_argument("--disable_dead_reinit", action="store_true",
                        help=(
                            "disable dead-prototype recovery entirely. Unlike a large "
                            "patience override, no reinitialisation event occurs at "
                            "all."
                            "⚠ The question this answers has changed: not whether it "
                            "prevents collapse, but whether it escapes a local minimum "
                            "that cross-entropy cannot leave on its own. With a "
                            "prototype-only head, CE itself creates collapse pressure, "
                            "since every sample in a centroid receives the same prediction."
                        ))
    parser.add_argument("--dead_reinit_patience_override", type=int, default=None,
                        help=(
                            "set CentroidLayer.dead_reinit_patience and retrain "
                            "(default 5, unverified: Jukebox and NSVQ use a different "
                            "criterion -- a usage rate below a threshold -- rather than "
                            "N consecutive epochs). A larger value leaves a dead "
                            "centroid unattended longer before it is reinitialised (an "
                            "event that overwrites parameters randomly, without a "
                            "gradient) but makes that intervention rarer -- a trade-off "
                            "against training stability. Cannot be used with --from_saved_state."
                        ))
    parser.add_argument("--dead_reinit_noise_scale_override", type=float, default=None,
                        help=(
                            "set CentroidLayer.dead_reinit_noise_scale and retrain "
                            "(default 0.01, unverified: the standard deviation of the "
                            "Gaussian noise added to the anchor vector on "
                            "reinitialisation is this value times anchor.norm()). The "
                            "original papers say only 'small Gaussian noise'. It decides "
                            "how far the reinitialised centroid lands from the anchor; "
                            "0 copies the anchor exactly. Cannot be used with --from_saved_state."
                        ))
    parser.add_argument("--refresh_on_best", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "on by default; disable with --no-refresh_on_best. "
                            "⚠ Turning it off makes **every retrieval diagnostic "
                            "invalid**. The memory.keys stored during training are "
                            "embeddings from an earlier point with dropout applied, a "
                            "different function from a test query encoded with the "
                            "final weights. Measured group-structure agreement (ARI): "
                            "ds=14 0.518, ds=46 0.067, ds=1489 0.006, rising to 1.000 "
                            "after a refresh. Missing this setting once invalidated a "
                            "large body of diversity and routing conclusions, so it "
                            "defaults to on. Right after best_state (and "
                            "feature_store) are restored, memory.keys is re-encoded "
                            "from the raw features with the current frozen weights. "
                            "Values stored during training were one-off snapshots "
                            "taken under some dropout mask and were not a "
                            "deterministic function of the raw features; with this on, "
                            "memory.keys[i] == embedder(feature_store._store[i]) holds "
                            "to floating-point error. With --from_saved_state the "
                            "refresh runs only if the checkpoint was not already "
                            "refreshed -- otherwise it is effectively a no-op."
                        ))
    parser.add_argument("--shap_background", type=int, default=50,
                        help=(
                            "number of background samples for the SHAP KernelExplainer "
                            "in rank_correlation (default 50). Measured: raising this "
                            "alone while nsamples is small relative to the feature "
                            "count can make agreement worse (jasmine, F=144: background "
                            "50->200 alone moved rho from 0.53 down to 0.36). Raise it "
                            "only once nsamples is adequate via --shap_nsamples."
                        ))
    parser.add_argument("--shap_nsamples", type=int, default=None,
                        help=(
                            "nsamples (perturbation samples) for the SHAP "
                            "KernelExplainer in rank_correlation. None uses the SHAP "
                            "library's own auto formula (2*n_features + 2048). "
                            "Measured: this was previously pinned at 100 regardless of "
                            "n_features to save cost, and on jasmine (F=144) raising "
                            "nsamples 100->500 alone lifted rho from 0.53 to 0.63. When "
                            "nsamples is small relative to F, the weighted regression "
                            "KernelSHAP solves internally becomes underdetermined and "
                            "the estimate is systematically biased. The auto formula "
                            "avoids that by scaling with F, which makes it the better "
                            "default; give an integer to override it for experiments."
                        ))
    parser.add_argument("--shap_repeats", type=int, default=1,
                        help=(
                            "how many times to repeat the SHAP computation in "
                            "rank_correlation, to diagnose the KernelExplainer's own "
                            "Monte Carlo noise -- how much the values move for the same "
                            "sample depending on the background and nsamples draws. "
                            "Default 1 means no repetition and no extra cost. Two or "
                            "more recomputes SHAP with a different random background "
                            "each time and reports the std of corr_shap across repeats."
                        ))
    args = parser.parse_args()
    # The centroid initialisation line lives in libs.prototypes, which does

    # ── Defaults for retired options ──────────────────────────────
    # The entries below were retired and **removed from the CLI**. Code in
    # many places still reads args.<name>, so rather than deleting every
    # reference the value at retirement is injected here. Execution always
    # takes that value, which makes the feature dead code and shrinks only
    # the CLI surface.









    # --deterministic separates GPU non-determinism from the architecture's
    # own chaotic sensitivity. cudnn.deterministic and benchmark are always
    # safe to set, but use_deterministic_algorithms(True) raises a
    # RuntimeError on any operation without a deterministic implementation.
    # Letting that propagate, rather than suppressing it with
    # --deterministic_warn_only, is deliberate: the error names the operation,
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        print(f"  [--deterministic] cudnn.deterministic=True, benchmark=False, "
              f"use_deterministic_algorithms(True, warn_only={args.deterministic_warn_only})"
              + (f" -- CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '(unset!)')}"
                 if torch.cuda.is_available() else " (no CUDA; on CPU most operations are deterministic anyway)"))
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
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

    # Separates data-loading time from training time. optimize.py loads the
    # dataset once outside objective() and 100 trials reuse it, whereas
    # reproduce.py loads it afresh in every process. The OpenML fetch, NaN
    # preprocessing, StratifiedKFold and QuantileTransformer costs all land
    # here, so this tells apart whether "reproduce.py feels slower than an
    # optimize.py trial" is training itself or this loading cost.
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

    # --train_seeds support: like optimize.py, the dataset and study load once
    # in main() and only run_single_seed() repeats per seed. The training,
    # evaluation and analysis logic that used to be inline in main() now lives
    # entirely in run_single_seed() -- see its definition above.
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
            args, _ts, do_analysis,
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