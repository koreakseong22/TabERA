"""Paired training-seed stability test for Unit- and Tangent-selected HPs.

This is a controlled diagnostic, not an official reproduction. It reads each
geometry's independently selected seed-1 HPO trial, then retrains both under
the *current* package environment with identical train seeds 11--14. Stored
study contracts are checked except for their package environment, whose
difference is the confound this experiment removes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

# OpenML reads this setting while libs.data is imported.
os.environ.setdefault("OPENML_CACHE_DIR", str(Path("data_cache/openml").resolve()))

import joblib
import numpy as np
import torch

from libs.benchmark_config import FINAL_CONFIG
from libs.benchmark import (build_wrapper, data_signature, final_study_path,
                            implementation_id, restore_params, select_trial,
                            training_diagnostics)
from libs.data import TabularDataset
from libs.eval import calculate_metric, get_preds_and_probs


GEOMETRIES = ("unit_tangent", "tangent")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()), encoding="utf-8")


def geometry_config(geometry: str) -> dict:
    config = dict(FINAL_CONFIG)
    if geometry == "tangent":
        config.update(correction_geometry="tangent", head_input_scale="unit")
    elif geometry != "unit_tangent":
        raise ValueError(geometry)
    return config


def clean(values: dict) -> dict:
    return {k: float(v) if np.isfinite(v) else None for k, v in values.items()}


def study_source(args, data: int, geometry: str) -> Path:
    root = args.unit_root if geometry == "unit_tangent" else args.tangent_root
    return final_study_path(root, args.fold, data, geometry_config(geometry))


def audit_source(study, dataset, config, task: str) -> tuple[object, dict]:
    trial = select_trial(study, task)
    recorded = trial.user_attrs.get("benchmark_contract")
    if not recorded:
        raise ValueError("Selected trial lacks benchmark_contract")
    expected_fields = {
        "implementation": implementation_id(),
        "data": data_signature(dataset),
        "protocol": recorded.get("protocol"),
        "recipe": recorded.get("recipe"),
        "schedule": recorded.get("schedule"),
        "config": config,
    }
    for key, expected in expected_fields.items():
        if recorded.get(key) != expected:
            raise ValueError(
                f"Selected trial contract mismatch for {key}: "
                f"study={recorded.get(key)!r}, expected={expected!r}")
    return trial, recorded


def run_one(args, dataset, task: str, data: int, member: int, geometry: str):
    config = geometry_config(geometry)
    source = study_source(args, data, geometry)
    if not source.is_file():
        raise FileNotFoundError(source)
    study = joblib.load(source)
    trial, source_contract = audit_source(study, dataset, config, task)
    train_y = dataset._indv_dataset()[0][1]
    params = restore_params(trial, len(train_y), config)
    train_seed = args.fold * 10 + member
    environment = {p: importlib.metadata.version(p)
                   for p in ("torch", "numpy", "scikit-learn", "optuna")}
    identity = dict(data=data, fold=args.fold, member=member,
                    train_seed=train_seed, geometry=geometry,
                    selected_trial=trial.number, selected_params=params,
                    study_sha256=sha256(source), source_contract=source_contract,
                    current_implementation=implementation_id(),
                    current_environment=environment,
                    provenance="same_environment_retraining_of_own_hpo_selected_hp")
    path = args.output / f"data={data}_member={member}_geometry={geometry}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError(f"Existing result identity differs: {path}")
        return record

    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    wrapper = build_wrapper(dataset, params, config, args.device)
    wrapper._data_id = data
    (xt, yt), (xv, yv), (xe, ye) = dataset._indv_dataset()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    wrapper.fit(xt, yt, xv, yv)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not wrapper.terminal_checkpoint or wrapper.best_epoch != wrapper.last_epoch:
        raise ValueError("Expected val-loss patience with terminal checkpoint")
    wrapper.model.eval()
    performance = {}
    with torch.inference_mode():
        for name, (x, y) in zip(("val", "test"), ((xv, yv), (xe, ye))):
            logits = wrapper._forward_batched(x)
            pred, prob = get_preds_and_probs(logits, task)
            performance[name] = clean(calculate_metric(y, pred, logits, task, name))
    record = dict(identity=identity, training_seconds=elapsed,
                  performance=performance,
                  training=training_diagnostics(wrapper))
    write_json(path, record)
    return record


def summarize(output: Path, records: list[dict]) -> dict:
    rows = []
    for record in records:
        row = {k: record["identity"][k]
               for k in ("data", "fold", "member", "train_seed", "geometry",
                         "selected_trial")}
        for split in ("val", "test"):
            row.update({f"{split}_{k}": v
                        for k, v in record["performance"][split].items()})
        row["last_epoch"] = record["training"]["last_epoch"]
        row["beta_final"] = record["training"]["beta_final"]
        rows.append(row)
    with (output / "per_run.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(r["data"], r["member"], r["geometry"]): r for r in rows}
    pairs = []
    for data, member in sorted({(r["data"], r["member"]) for r in rows}):
        unit = lookup[data, member, "unit_tangent"]
        tangent = lookup[data, member, "tangent"]
        pair = dict(data=data, member=member, train_seed=unit["train_seed"])
        source_metric = {"auc": "auroc", "acc": "acc", "f1": "f1",
                         "logloss": "logloss"}
        for metric, source_name in source_metric.items():
            pair[metric] = (tangent[f"test_{source_name}_test"] -
                            unit[f"test_{source_name}_test"])
        pairs.append(pair)

    means = []
    for data in sorted({p["data"] for p in pairs}):
        group = [p for p in pairs if p["data"] == data]
        means.append(dict(data=data, n=len(group),
                          **{m: float(np.mean([p[m] for p in group]))
                             for m in ("auc", "acc", "f1", "logloss")},
                          auc_wins=sum(p["auc"] > 0 for p in group),
                          auc_losses=sum(p["auc"] < 0 for p in group)))
    macro = {m: float(np.mean([d[m] for d in means]))
             for m in ("auc", "acc", "f1", "logloss")}
    non51 = float(np.mean([d["auc"] for d in means if d["data"] != 51]))
    all_auc = [p["auc"] for p in pairs]
    member_mean_auc = {
        member: float(np.mean([p["auc"] for p in pairs if p["member"] == member]))
        for member in sorted({p["member"] for p in pairs})
    }
    best_member = max(member_mean_auc, key=member_mean_auc.get)
    remaining_members = [m for m in member_mean_auc if m != best_member]
    mean_auc_without_best_member = float(np.mean([
        p["auc"] for p in pairs if p["member"] in remaining_members
    ]))
    gate = dict(dataset_equal_mean=macro,
                positive_datasets=sum(d["auc"] > 0 for d in means),
                non51_mean_auc=non51,
                fold_wins=sum(x > 0 for x in all_auc),
                fold_losses=sum(x < 0 for x in all_auc),
                member_mean_auc=member_mean_auc,
                best_member=best_member,
                mean_auc_without_best_member=mean_auc_without_best_member,
                pass_datasets=sum(d["auc"] > 0 for d in means) >= 2,
                pass_auc=macro["auc"] > .005,
                pass_non51=non51 >= 0,
                pass_acc=macro["acc"] >= -.01,
                pass_f1=macro["f1"] >= -.01,
                pass_seed_sensitivity=mean_auc_without_best_member > 0)
    gate["all"] = all(v for k, v in gate.items() if k.startswith("pass_"))
    result = dict(pairs=pairs, dataset_means=means, gate=gate)
    write_json(output / "paired_summary.json", result)

    lines = ["# Paired own-HPO geometry stability", "",
             "Fold 1; current environment; members/train seeds 1-4 / 11-14.", "",
             "| Data | Tangent-Unit AUC | ACC | F1 | LogLoss | AUC W/L |",
             "|---|---:|---:|---:|---:|---:|"]
    for d in means:
        lines.append(f"| {d['data']} | {d['auc']:+.4f} | {d['acc']:+.4f} | "
                     f"{d['f1']:+.4f} | {d['logloss']:+.4f} | "
                     f"{d['auc_wins']}/{d['auc_losses']} |")
    lines += ["", f"Dataset-equal mean: AUC {macro['auc']:+.4f}, "
              f"ACC {macro['acc']:+.4f}, F1 {macro['f1']:+.4f}, "
              f"LogLoss {macro['logloss']:+.4f}.",
              f"Without best member {best_member}: AUC "
              f"{mean_auc_without_best_member:+.4f}.",
              f"Gate: {'PASS' if gate['all'] else 'FAIL'}. "]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", type=int, default=[51, 1067, 31])
    parser.add_argument("--members", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--unit-root", type=Path, default=Path("."))
    parser.add_argument("--tangent-root", type=Path, default=Path("tangent_hpo_pilot"))
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/hpo_geometry_stability"))
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault("OPENML_CACHE_DIR", str(Path("data_cache/openml").resolve()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id >= 0 else ""
    args.device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for data in args.datasets:
        with open("dataset_id.json", encoding="utf-8") as stream:
            task = json.load(stream)[str(data)]["tasktype"]
        dataset = TabularDataset(data, task, device=args.device, seed=args.fold)
        for member in args.members:
            for geometry in GEOMETRIES:
                print(f"[run] data={data} member={member} geometry={geometry}", flush=True)
                record = run_one(args, dataset, task, data, member, geometry)
                records.append(record)
                print(f"[done] test={record['performance']['test']}", flush=True)
    summarize(args.output, records)


if __name__ == "__main__":
    main()
