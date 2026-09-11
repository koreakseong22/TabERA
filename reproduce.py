"""Final TabERA reproduction. Exploratory analysis lives in analyze.py."""
import argparse
import os
from pathlib import Path
from libs.benchmark_config import FINAL_CONFIG


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openml_id", type=int, required=True)
    p.add_argument("--seed", type=int, choices=range(10), default=1, help="MultiTab data fold")
    p.add_argument("--gpu_id", type=int, default=0, help="physical GPU; -1 for CPU")
    p.add_argument("--savepath", default=".")
    p.add_argument("--json", default=str(Path(__file__).with_name("dataset_id.json")))
    p.add_argument("--mode", choices=["best", "init", "deep", "hyper", "all"], default="best")
    p.add_argument("--member", type=int, choices=range(5), default=0)
    p.add_argument("--correction_geometry", choices=[FINAL_CONFIG["correction_geometry"]], default=FINAL_CONFIG["correction_geometry"])
    p.add_argument("--head_input_scale", choices=[FINAL_CONFIG["head_input_scale"]], default=FINAL_CONFIG["head_input_scale"])
    p.add_argument("--disable_dead_reinit", action="store_true",
                   help=("ablation arm: dead-prototype recovery off (dead_reinit_patience=1e9, so the reinit "
                         "block in regroup_update never fires). Reads the ..nodr study written by "
                         "optimize.py --disable_dead_reinit and writes model=tabera..nodr.. results; the "
                         "main arm's files are untouched"))
    p.add_argument("--early_stop_metric", choices=["val_loss", "accuracy", "logloss", "auroc", "bacc"],
                   default=FINAL_CONFIG["early_stop_metric"],
                   help=("validation criterion for early stopping. Default val_loss is the MultiTab protocol: "
                         "batch-averaged validation loss, patience 20, and the model at the epoch training "
                         "stopped is evaluated (no best-checkpoint restore). Every other value is an ablation "
                         "arm that restores the best checkpoint by that metric; it reads the ..esm=NAME study "
                         "written by optimize.py --early_stop_metric and writes model=tabera..esm=NAME.. results"))
    p.add_argument("--hpo_source", choices=["own", "main"], default="own",
                   help=("where an ablation arm takes its hyperparameters from. own (default): the arm's own "
                         "study, i.e. a fair comparison of two tuned models. main: the MAIN arm's study with the "
                         "same best trial, so lr/dropout/embed_dim are held fixed and only the arm's one variable "
                         "changes -- the fixed-HP ablation. Results are written as model=tabera<arm>..hpo=main.. "
                         "and record both contracts"))
    p.add_argument("--allow_unverified_study", action="store_true",
                   help="accept old trials lacking code/data provenance after your own audit; result marked unverified")
    p.add_argument("--overwrite", action="store_true", help="explicitly replace an existing result")
    p.add_argument("--audit_only", action="store_true", help="validate study and data without training")
    return p


def prototype_diag(wrapper):
    """Compact prototype-utilisation summary of a finished fit.

    From the per-epoch regroup records the wrapper keeps: the last epoch's
    utilisation, and reinit totals over the run. Plain floats so the payload
    stays a small dict; None where the wrapper recorded nothing.
    """
    hist = list(getattr(wrapper, "regroup_history", None) or [])
    last = hist[-1] if hist else {}
    keys = ("active_ratio", "dead_ratio", "n_eff_entropy", "n_eff_inv_simpson", "top1_share",
            "min_cluster_size", "max_cluster_size")
    out = {k: (float(last[k]) if last.get(k) is not None else None) for k in keys}
    out["reinit_total"] = float(sum(r.get("reinit_count", 0) or 0 for r in hist)) if hist else None
    out["epochs_logged"] = len(hist)
    geo = getattr(wrapper, "centroid_geometry_diag", None) or {}
    out["reinit_per_epoch"] = geo.get("reinit_per_epoch")
    out["active_ratio_std"] = geo.get("active_ratio_std")
    return out


def run(args):
    import json
    import random
    import time
    import hashlib
    import importlib.metadata
    import joblib
    import numpy as np
    import torch
    from libs.data import TabularDataset, _LAST_LOAD_DIAG
    from libs.eval import calculate_metric, get_preds_and_probs
    from libs.benchmark import (UPSTREAM_COMMIT, arm_config, is_main_arm, contract, contract_diff,
                                final_study_path, select_trial, restore_params, build_wrapper,
                                result_path, atomic_save, training_diagnostics)

    if args.mode not in ("deep", "hyper") and args.member:
        raise ValueError("--member only applies to --mode deep/hyper")
    with open(args.json, encoding="utf-8") as stream:
        info = json.load(stream)[str(args.openml_id)]
    task = info["tasktype"]
    # One config object drives the study lookup, the contract, the model and
    # the result path, so an arm can never be half-applied.
    config = arm_config(args.disable_dead_reinit, args.early_stop_metric)
    if args.hpo_source == "main" and is_main_arm(config):
        raise ValueError("--hpo_source main only applies to an ablation arm "
                         "(--disable_dead_reinit and/or a non-default --early_stop_metric)")
    # Fixed-HP ablation: the study (and the contract it is audited against)
    # belong to the main arm; the model is built with the arm's config.
    hpo_config = arm_config() if args.hpo_source == "main" else config
    source = final_study_path(args.savepath, args.seed, args.openml_id, hpo_config)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. Run optimize.py for this dataset/fold and final structure first.")
    study = joblib.load(source)
    select_trial(study, task)
    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    dataset = TabularDataset(args.openml_id, task, device=device, seed=args.seed)
    if _LAST_LOAD_DIAG.get("invalid_num_cols"):
        raise ValueError("Data loader dropped columns outside the official protocol")
    (xt, yt), (xv, yv), (xe, ye) = dataset._indv_dataset()
    expected = contract(config, dataset)          # this run's own contract
    expected_hpo = contract(hpo_config, dataset)  # what the study must have been searched under
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    unverified = False
    for trial in completed:
        restore_params(trial, len(yt), config, hpo_config)
        recorded = trial.user_attrs.get("benchmark_contract")
        if recorded is None:
            if not args.allow_unverified_study:
                raise ValueError("Study lacks code/data provenance. Audit it before using --allow_unverified_study, or rerun HPO in a new --savepath.")
            unverified = True
        elif recorded != expected_hpo:
            diff = contract_diff(recorded, expected_hpo)
            hint = ""
            if [d for d in diff if d.startswith("implementation:")] == diff:
                # Only the code hash moved. That still means this study was
                # searched under different library code, but it tells the
                # reader the data and the configuration are intact -- and that
                # the decision is whether the change could affect training.
                hint = ("\n  Only the library-code hash differs (data and config match). "
                        "implementation_id() covers libs/{benchmark,benchmark_config,data,eval,"
                        "search_space,supervised,tabera,prototypes}.py, so any edit to those "
                        "invalidates studies recorded before it.")
            raise ValueError(
                f"Trial {trial.number}: this study was recorded under a different contract; "
                f"rerun HPO or use a study that matches.\n  " + "\n  ".join(diff) + hint)
    print(f"[HPO audit] {len(completed)} completed; unverified_hpo={unverified}; {source}")
    if args.audit_only:
        return
    modes = ([("init", 0), ("best", 0)] + [("deep", i) for i in range(1, 5)] +
             [("hyper", i) for i in range(1, 5)]) if args.mode == "all" else [(args.mode, args.member)]
    for mode, member in modes:
        if mode == "hyper" and member >= len(completed) and args.mode == "all":
            print(f"[unavailable] hyper={member}: only {len(completed)} completed trials")
            continue
        trial = select_trial(study, task, mode, member)
        params = restore_params(trial, len(yt), config, hpo_config)
        # Upstream leaves its RNG implicit. Explicit member seeds make resume
        # independent of which preceding artifacts already exist.
        train_seed = args.seed * 10 + (member if mode == "deep" else 0)
        identity = dict(contract=expected, hpo_source=args.hpo_source, hpo_contract=expected_hpo,
                        optimize_sha256=trial.user_attrs.get("optimize_sha256"),
                        trial=trial.number, params=params,
                        dataset_id=args.openml_id, fold=args.seed, tasktype=task,
                        train_seed=train_seed, unverified_hpo=unverified,
                        reproduction_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        study_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        path = result_path(args.savepath, args.seed, args.openml_id, mode, member, config, args.hpo_source)
        if path.exists() and not args.overwrite:
            saved = np.load(path, allow_pickle=True).item()
            if not isinstance(saved, dict) or saved.get("identity") != identity:
                raise ValueError(f"Existing result is incompatible: {path}. Use a new --savepath or --overwrite.")
            if not all(k in saved for k in ("Prediction", "Probability", "Performance", "time")):
                raise ValueError(f"Incomplete result: {path}")
            print(f"[skip verified] {path}")
            continue
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)
        wrapper = build_wrapper(dataset, params, config, device)
        wrapper._data_id = args.openml_id
        if config["disable_dead_reinit"]:
            _pat = int(wrapper.model.prototype_layer.dead_reinit_patience)
            assert _pat >= 10**9, f"disable_dead_reinit did not reach the model (patience={_pat})"
            print(f"[arm] dead-prototype recovery OFF (dead_reinit_patience={_pat})")
        if config["early_stop_metric"] != FINAL_CONFIG["early_stop_metric"]:
            assert wrapper.early_stop_metric == config["early_stop_metric"], \
                f"early_stop_metric did not reach the wrapper ({wrapper.early_stop_metric})"
            print(f"[arm] early_stop_metric={wrapper.early_stop_metric} "
                  f"(selection key {wrapper._sel_key}, main arm: {FINAL_CONFIG['early_stop_metric']})")
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        wrapper.fit(xt, yt, xv, yv)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        wrapper.model.eval()
        with torch.no_grad():
            logits = wrapper._forward_batched(xe)
            val_logits = wrapper._forward_batched(xv, collect_diagnostics=True)
        pred, prob = get_preds_and_probs(logits, task)
        vp, vprob = get_preds_and_probs(val_logits, task)
        scale = dataset.y_std if task == "regression" else 1.
        perf = calculate_metric(ye * scale, pred * scale, prob, task, "test")
        val_perf = calculate_metric(yv * scale, vp * scale, vprob, task, "val")
        clean = lambda values: {k: float(v) if np.isfinite(v) else None for k, v in values.items()}
        payload = dict(Prediction=pred.detach().cpu().numpy(),
                       Probability=None if task == "regression" else logits.detach().cpu().numpy(),
                       time=elapsed, Performance=clean(perf), Performance_val=clean(val_perf),
                       identity=identity, upstream_commit=UPSTREAM_COMMIT,
                       probability_representation="logits", prediction_scale="standardized" if task == "regression" else "class_index",
                       head_gamma=float(wrapper.model.effective_gamma()),
                       # Prototype-utilisation diagnostics, so an arm comparison
                       # can read utilisation next to the metrics it moved.
                       prototype_diag=prototype_diag(wrapper),
                       dynamics_provenance=wrapper.dynamics_provenance,
                       encoding_provenance=wrapper.encoding_provenance,
                       training_diagnostics=training_diagnostics(wrapper),
                       prediction_diagnostics_val=wrapper.prediction_diagnostics,
                       environment={p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn", "optuna")})
        atomic_save(path, payload)
        print(path)
        print(payload["Performance"])


def main():
    args = parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id >= 0 else ""
    run(args)


if __name__ == "__main__":
    main()
