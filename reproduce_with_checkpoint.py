"""Reproduce one manifest run, preserving its original inference state.

Default: provenance preflight only. --train opts into a new reproduction (no
HPO); --restore-only audits an already saved checkpoint. Library/data/code
mismatches fail closed. Existing benchmark artifacts are never overwritten.
"""
import argparse
import json
import os
from pathlib import Path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def select_run(manifest, dataset_id, fold):
    rows = [r for r in manifest["runs"] if r["dataset_id"] == dataset_id and r["fold"] == fold]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one run for dataset={dataset_id}, fold={fold}")
    row = rows[0]
    identity = row["identity"]
    for key, expected in (("dataset_id", dataset_id), ("fold", fold), ("train_seed", row["train_seed"]),
                          ("params", row["selected_params"]), ("tasktype", row["tasktype"])):
        if identity[key] != expected:
            raise ValueError(f"Manifest/identity mismatch: {key}")
    if identity["contract"]["config"] != row["model_config"]:
        raise ValueError("Manifest config/identity mismatch")
    return row


def run(args):
    import random
    import time
    import numpy as np
    import torch
    from build_explanation_manifest import sha256
    from libs.benchmark import build_wrapper, contract, contract_diff, implementation_id
    from libs.data import TabularDataset
    from libs.eval import get_preds_and_probs
    from libs.reproduction_state import (compare_predictions, current_environment, restore_checkpoint,
                                         save_checkpoint, snapshot, get_cuda_runtime_info)

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = select_run(manifest, args.dataset_id, args.fold)
    output = Path(args.output) if args.output else manifest_path.parent / f"openml_{args.dataset_id}/fold_{args.fold}"
    output.mkdir(parents=True, exist_ok=True)
    execution_mode = "restore_only" if args.restore_only else "train" if args.train else "preflight"
    audit_path = output / {"restore_only": "audit_restore.json", "train": "audit_train.json",
                           "preflight": "preflight.json"}[execution_mode]
    reference_path = (manifest_path.parent / row["reference_path"]).resolve()
    if sha256(reference_path) != row["reference_sha256"]:
        raise ValueError("Benchmark reference checksum mismatch")
    with np.load(reference_path, allow_pickle=False) as reference:
        ref_logits, ref_preds = reference["logits"].copy(), reference["predictions"].copy()
    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    original_identity = row["identity"]
    expected = original_identity["contract"]
    environment = current_environment()
    audit = dict(execution_mode=execution_mode,
                 dataset_id=args.dataset_id, fold=args.fold, train_seed=row["train_seed"],
                 manifest_sha256=sha256(manifest_path), reference_sha256=row["reference_sha256"],
                 runner_sha256=sha256(__file__),
                 checkpoint_code_sha256=sha256(Path(__file__).parent / "libs/reproduction_state.py"),
                 device=device, cuda_version=torch.version.cuda,
                 device_name=torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
                 runtime=get_cuda_runtime_info(),
                 environment=environment, expected_environment=expected["environment"],
                 implementation_hash=implementation_id(), data_verified=False,
                 checkpoint_available=(output / "checkpoint.pt").is_file(),
                 eligible_for_explanation=False)
    differences = []
    if implementation_id() != expected["implementation"]:
        differences.append("implementation hash differs from benchmark")
    if environment != expected["environment"] or environment != row["environment"]:
        differences.append("library versions differ from benchmark")
    if original_identity.get("unverified_hpo"):
        differences.append("source benchmark marked unverified_hpo")
    audit["provenance_differences"] = differences
    if differences:
        audit["status"] = "blocked_provenance"
        write_json(audit_path, audit)
        print(json.dumps(audit, indent=2))
        return 2

    checkpoint_path = output / "checkpoint.pt"
    if args.restore_only:
        wrapper, dataset, payload = restore_checkpoint(checkpoint_path, device)
        if payload["identity"] != original_identity:
            raise ValueError("Checkpoint belongs to another benchmark run")
    else:
        dataset = TabularDataset(args.dataset_id, row["tasktype"], device=device, seed=args.fold)
    actual_contract = contract(row["model_config"], dataset)
    differences = contract_diff(expected, actual_contract)
    audit["provenance_differences"] = differences
    audit["data_verified"] = actual_contract["data"] == expected["data"]
    if differences:
        audit["status"] = "blocked_contract"
        write_json(audit_path, audit)
        print(json.dumps(audit, indent=2))
        return 2
    audit["status"] = "preflight_passed"
    write_json(audit_path, audit)
    if not args.train and not args.restore_only:
        print(f"[preflight passed] dataset={args.dataset_id}, fold={args.fold}, train_seed={row['train_seed']}")
        return 0
    (xt, yt), (xv, yv), (xe, ye) = dataset._indv_dataset()
    if not args.restore_only:
        if checkpoint_path.exists():
            raise FileExistsError(f"Use --restore-only to audit existing {checkpoint_path}")
        seed = row["train_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        wrapper = build_wrapper(dataset, row["selected_params"], row["model_config"], device)
        wrapper._data_id = args.dataset_id
        start = time.perf_counter()
        wrapper.fit(xt, yt, xv, yv)
        audit["training_seconds"] = time.perf_counter() - start
        wrapper.model.eval()
        with torch.no_grad():
            logits = wrapper._forward_batched(xe)
        preds, _ = get_preds_and_probs(logits, row["tasktype"])
        payload = snapshot(wrapper, dataset, original_identity, logits, preds)
        save_checkpoint(checkpoint_path, payload)
        # Always audit disk restoration, not just the in-memory model.
        del wrapper
        wrapper, dataset, payload = restore_checkpoint(checkpoint_path, device)
        xe, ye = dataset._indv_dataset()[2]
    with torch.no_grad():
        restored_logits = wrapper._forward_batched(xe)
    restored_preds, _ = get_preds_and_probs(restored_logits, row["tasktype"])
    audit["checkpoint_roundtrip"] = compare_predictions(
        restored_logits, restored_preds, payload["test_logits"], payload["test_preds"], ye)
    audit["benchmark_reproduction"] = compare_predictions(
        restored_logits, restored_preds, ref_logits, ref_preds, ye)
    saved_acc = row["performance"]["acc_test"]
    audit["benchmark_accuracy_consistent"] = bool(abs(
        audit["benchmark_reproduction"]["accuracy_reference"] - saved_acc) < 1e-12)
    passed = (audit["checkpoint_roundtrip"]["passed"] and audit["benchmark_reproduction"]["passed"]
              and audit["benchmark_accuracy_consistent"])
    audit.update(checkpoint_available=True, checkpoint_sha256=sha256(checkpoint_path),
                 status="reproduction_passed" if passed else "failed_reproduction",
                 # Refresh invariance and explanation audits are deliberately pending.
                 eligible_for_memory_refresh=bool(passed), eligible_for_explanation=False)
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))
    return 0 if passed else 3


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default="analysis_results/manifest.json")
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(10))
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--output", help="New run directory; defaults beside the manifest reference")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--train", action="store_true")
    action.add_argument("--restore-only", action="store_true")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id >= 0 else ""
    raise SystemExit(run(args))
