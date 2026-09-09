"""Average final TabERA member logits before sigmoid/softmax, as in MultiTab."""
import argparse
from pathlib import Path
import json

import numpy as np
import torch

from libs.benchmark import arm_config, arm_tag, atomic_save, contract, result_path
from libs.data import TabularDataset
from libs.eval import calculate_metric, get_preds_and_probs
from libs.search_space import RECIPE_TAG


def combine(members, task):
    key = "Prediction" if task == "regression" else "Probability"
    arrays = [np.asarray(m[key]) for m in members]
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Ensemble members have incompatible shapes or non-finite predictions")
    mean = torch.from_numpy(np.mean(np.stack(arrays), axis=0))
    if task == "regression":
        return mean.reshape(-1), None, None
    pred, prob = get_preds_and_probs(mean, task)
    return pred, prob, mean.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openml_id", type=int, required=True)
    p.add_argument("--seed", type=int, choices=range(10), default=1)
    p.add_argument("--savepath", default=".")
    p.add_argument("--type", choices=["deep", "hyper", "all"], default="deep")
    p.add_argument("--members", type=int, choices=range(2, 6), default=5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--disable_dead_reinit", action="store_true",
                   help="ablation arm: combine the ..nodr members written by reproduce.py --disable_dead_reinit")
    p.add_argument("--early_stop_metric", choices=["val_loss", "accuracy", "logloss", "auroc", "bacc"], default=None,
                   help="ablation arm: combine the ..esm=NAME members")
    args = p.parse_args()
    info = json.loads(Path(__file__).with_name("dataset_id.json").read_text(encoding="utf-8"))
    task = info[str(args.openml_id)]["tasktype"]
    ds = TabularDataset(args.openml_id, task, device="cpu", seed=args.seed)
    config = arm_config(args.disable_dead_reinit, args.early_stop_metric)
    expected = contract(config, ds)
    modes = ([("best", 0)] + [("deep", i) for i in range(1, 5)] + [("hyper", i) for i in range(1, 5)]
             if args.type == "all" else [("best", 0)] + [(args.type, i) for i in range(1, args.members)])
    members = []
    for mode, i in modes:
        path = result_path(args.savepath, args.seed, args.openml_id, mode, i, config)
        member = np.load(path, allow_pickle=True).item()
        if not isinstance(member, dict) or member.get("identity", {}).get("contract") != expected:
            raise ValueError(f"Incompatible member: {path}")
        if task != "regression" and member.get("probability_representation") != "logits":
            raise ValueError(f"Expected logits: {path}")
        members.append(member)
    if len({m["identity"]["study_sha256"] for m in members}) != 1:
        raise ValueError("Ensemble members come from different HPO snapshots")
    pred, prob, logits = combine(members, task)
    y = ds._indv_dataset()[2][1]
    scale = ds.y_std if task == "regression" else 1.
    metrics = calculate_metric(y * scale, pred * scale, prob, task, "test")
    deep = 5 if args.type == "all" else args.members if args.type == "deep" else 0
    hyper = 5 if args.type == "all" else args.members if args.type == "hyper" else 0
    out = (Path(args.savepath) / f"ensemble_logs/seed={args.seed}/data={args.openml_id}/"
           f"model=tabera{RECIPE_TAG}{arm_tag(config)}..init_hps=False..deep={deep}..hyper={hyper}.npy")
    payload = dict(Prediction=pred.numpy(), Probability=logits,
                   Performance={k: float(v) if np.isfinite(v) else None for k, v in metrics.items()},
                   time=sum(m["time"] for m in members), members=[m["identity"] for m in members],
                   probability_representation="logits")
    if out.exists() and not args.overwrite:
        previous = np.load(out, allow_pickle=True).item()
        if isinstance(previous, dict) and previous.get("members") == payload["members"]:
            print(f"[skip verified] {out}")
            return
        raise ValueError(f"Existing incompatible ensemble: {out}; use --overwrite or new --savepath")
    atomic_save(out, payload)
    print(out)
    print(payload["Performance"])


if __name__ == "__main__":
    main()
