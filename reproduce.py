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
    p.add_argument("--allow_unverified_study", action="store_true",
                   help="accept old trials lacking code/data provenance after your own audit; result marked unverified")
    p.add_argument("--overwrite", action="store_true", help="explicitly replace an existing result")
    p.add_argument("--audit_only", action="store_true", help="validate study and data without training")
    return p


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
    from libs.benchmark import (FINAL_CONFIG, UPSTREAM_COMMIT, contract, final_study_path,
                                select_trial, restore_params, build_wrapper, result_path, atomic_save)

    if args.mode not in ("deep", "hyper") and args.member:
        raise ValueError("--member only applies to --mode deep/hyper")
    with open(args.json, encoding="utf-8") as stream:
        info = json.load(stream)[str(args.openml_id)]
    task = info["tasktype"]
    source = final_study_path(args.savepath, args.seed, args.openml_id)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}. Run optimize.py for this dataset/fold and final structure first.")
    study = joblib.load(source)
    select_trial(study, task)
    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    dataset = TabularDataset(args.openml_id, task, device=device, seed=args.seed)
    if _LAST_LOAD_DIAG.get("invalid_num_cols"):
        raise ValueError("Data loader dropped columns outside the official protocol")
    (xt, yt), (xv, yv), (xe, ye) = dataset._indv_dataset()
    expected = contract(FINAL_CONFIG, dataset)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    unverified = False
    for trial in completed:
        restore_params(trial, len(yt))
        recorded = trial.user_attrs.get("benchmark_contract")
        if recorded is None:
            if not args.allow_unverified_study:
                raise ValueError("Study lacks code/data provenance. Audit it before using --allow_unverified_study, or rerun HPO in a new --savepath.")
            unverified = True
        elif recorded != expected:
            raise ValueError(f"Trial {trial.number}: code, data or configuration differs from HPO; rerun HPO")
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
        params = restore_params(trial, len(yt))
        # Upstream leaves its RNG implicit. Explicit member seeds make resume
        # independent of which preceding artifacts already exist.
        train_seed = args.seed * 10 + (member if mode == "deep" else 0)
        identity = dict(contract=expected, trial=trial.number, params=params,
                        dataset_id=args.openml_id, fold=args.seed, tasktype=task,
                        train_seed=train_seed, unverified_hpo=unverified,
                        reproduction_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        study_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        path = result_path(args.savepath, args.seed, args.openml_id, mode, member)
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
        wrapper = build_wrapper(dataset, params, FINAL_CONFIG, device)
        wrapper._data_id = args.openml_id
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
            val_logits = wrapper._forward_batched(xv)
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
