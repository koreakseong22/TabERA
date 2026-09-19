"""Inference latency of the final TabERA model, prediction-only vs with evidence.

Companion to ``../multitab/scripts/bench_inference_tabr.py``, which times TabR
with the same protocol module (libs/inference_timing.py). One JSON per
(dataset, fold) goes to --out.

Modes timed (all on the same trained model, eval mode, no_grad):
  prediction_only       forward(retrieve=False): encoder -> prototype routing
                        -> head. Never touches the memory bank. This is what
                        predict / predict_proba run.
  prediction_retrieval  forward(retrieve=True): the same plus k-NN evidence
                        retrieval inside the assigned region.
  prediction_explain    forward(return_explanations=True): retrieval plus the
                        per-sample explanation payload (routing explanation).
  api_predict_proba     full split through TabERAWrapper.predict_proba, the
                        public API (chunks by the tuned batch size).

The model is trained once here exactly as reproduce.py --mode best does
(same study, trial selection, params, train seed), because the benchmark
archives keep predictions, not checkpoints. Training time is not reported:
it is not part of this protocol.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from libs.benchmark_config import FINAL_CONFIG
from reproduce import with_structure


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--openml_id", type=int, required=True)
    p.add_argument("--seed", type=int, choices=range(10), default=1, help="MultiTab data fold")
    p.add_argument("--gpu_id", type=int, default=0, help="physical GPU; -1 for CPU")
    p.add_argument("--savepath", default=".", help="root holding optim_logs/ of the final studies")
    p.add_argument("--json", default=str(Path(__file__).with_name("dataset_id.json")))
    # Same structure switches as reproduce.py, so the study path and the model
    # are the ones the benchmark numbers came from.
    p.add_argument("--correction_geometry", choices=["unit_tangent", "tangent"],
                   default=FINAL_CONFIG["correction_geometry"])
    p.add_argument("--head_input_scale", choices=["auto", "unit"],
                   default=FINAL_CONFIG["head_input_scale"])
    p.add_argument("--out", default="results/inference_latency")
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 512])
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--full_pass_repeats", type=int, default=20)
    p.add_argument("--split", choices=["test", "val"], default="test")
    p.add_argument("--tile", action="store_true",
                   help="repeat rows so every dataset is timed at the requested batch size even when "
                        "its split is smaller; needed for a samples/s comparison across datasets")
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (pin for CPU runs)")
    p.add_argument("--overwrite", action="store_true")
    return p


def run(args):
    import numpy as np
    import torch
    from libs.data import TabularDataset
    from libs.benchmark import (arm_config, final_study_path, select_trial, restore_params,
                                build_wrapper, atomic_save)
    from libs.inference_timing import run_protocol, environment, resident_memory_mb, time_full_pass
    import joblib

    if args.threads:
        torch.set_num_threads(args.threads)
    with open(args.json, encoding="utf-8") as stream:
        info = json.load(stream)[str(args.openml_id)]
    task = info["tasktype"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (f"model=tabera..geom={args.correction_geometry}..hs={args.head_input_scale}"
                          f"..data={args.openml_id}..seed={args.seed}.json")
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path}")
        return

    config = with_structure(arm_config(), args.correction_geometry, args.head_input_scale)
    source = final_study_path(args.savepath, args.seed, args.openml_id, config)
    if not source.is_file():
        raise FileNotFoundError(f"Missing {source}")
    study = joblib.load(source)
    device_str = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda"):
        torch.cuda.set_device(0)
    device = torch.device(device_str)
    dataset = TabularDataset(args.openml_id, task, device=device_str, seed=args.seed)
    (xt, yt), (xv, yv), (xe, ye) = dataset._indv_dataset()
    trial = select_trial(study, task, "best", 0)
    params = restore_params(trial, len(yt), config, config)

    train_seed = args.seed * 10                     # reproduce.py --mode best
    random.seed(train_seed); np.random.seed(train_seed); torch.manual_seed(train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(train_seed)
    wrapper = build_wrapper(dataset, params, config, device_str)
    wrapper._data_id = args.openml_id
    t0 = time.perf_counter()
    wrapper.fit(xt, yt, xv, yv)
    fit_s = time.perf_counter() - t0
    model = wrapper.model.eval()

    X = (xe if args.split == "test" else xv).to(device)
    n_mem = int(model.memory.filled.item())
    if n_mem < model.k:
        raise RuntimeError(f"memory holds {n_mem} < k={model.k}; retrieval mode would be warm-up only")

    fns = {
        "prediction_only":      lambda xb: model(xb, retrieve=False)["logits"],
        "prediction_retrieval": lambda xb: model(xb, retrieve=True)["logits"],
        "prediction_explain":   lambda xb: model(xb, return_explanations=True),
    }
    # Guard the protocol's premise: the three modes agree on the logits.
    with torch.no_grad():
        probe = X[: min(64, len(X))]
        z0 = fns["prediction_only"](probe)
        assert torch.equal(z0, fns["prediction_retrieval"](probe))
        assert torch.equal(z0, fns["prediction_explain"](probe)["logits"])

    resident = resident_memory_mb(device)
    timing = run_protocol(fns, X, args.batch_sizes, args.repeats, args.warmup,
                          full_pass_batch=max(args.batch_sizes), full_pass_repeats=args.full_pass_repeats,
                          seed=args.seed, tile=args.tile)
    api = time_full_pass(lambda xb: wrapper.predict_proba(xb, logit=True), X,
                         batch_size=len(X), repeats=args.full_pass_repeats)
    timing["api_predict_proba"] = {"full_pass": api}

    payload = {
        "model": "tabera",
        "correction_geometry": args.correction_geometry, "head_input_scale": args.head_input_scale,
        "dataset_id": args.openml_id, "dataset": info.get("fullname"), "tasktype": task,
        "fold": args.seed, "train_seed": train_seed, "split": args.split,
        "n_train": int(len(yt)), "n_eval": int(len(X)), "n_features": int(xt.shape[1]),
        "n_prototypes": int(model.prototype_layer.centroid_emb.shape[0]),
        "k": int(model.k), "memory_filled": n_mem,
        "study": str(source), "trial": trial.number, "params": params,
        "fit_s_not_protocol": fit_s,
        "protocol": {"batch_sizes": args.batch_sizes, "repeats": args.repeats, "warmup": args.warmup,
                     "full_pass_repeats": args.full_pass_repeats, "tile": args.tile,
                     "excluded": ["data loading", "input host->device copy", "model construction",
                                  "training", "memory-bank construction"]},
        "resident_mb_after_setup": resident,
        "environment": dict(environment(device), physical_gpu_id=args.gpu_id),
        "timing": timing,
    }
    atomic_save(out_path, payload) if out_path.suffix == ".npy" else out_path.write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(out_path)
    for mode, rec in timing.items():
        for key, st in rec.items():
            if isinstance(st, dict) and "ms_median" in st:
                print(f"  {mode:22s} {key:10s} rows={st['batch_rows']:5d} "
                      f"median={st['ms_median']:8.3f} ms  p90={st['ms_p90']:8.3f}  "
                      f"{st['samples_per_s']:10.0f} samples/s")


def main():
    args = parser().parse_args()
    # Same convention as reproduce.py: the physical GPU is selected by
    # remapping, before run() imports torch, so the process sees exactly one
    # device and `cuda:0` inside is that GPU. Without this, --gpu_id 1 would
    # silently land on GPU 0 and two parallel workers would share one device,
    # which for a timing experiment means both sets of numbers are contended.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id >= 0 else ""
    run(args)


if __name__ == "__main__":
    main()
