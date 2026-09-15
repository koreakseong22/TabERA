"""Create and audit a final-encoder memory for explanation analysis."""
import argparse
import json
from pathlib import Path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def require_reproduced(audit, mode, checkpoint_sha256):
    required = (
        audit.get("execution_mode") == mode,
        audit.get("status") == "reproduction_passed",
        (audit.get("checkpoint_roundtrip") or {}).get("passed") is True,
        (audit.get("benchmark_reproduction") or {}).get("passed") is True,
        audit.get("benchmark_accuracy_consistent") is True,
        audit.get("eligible_for_memory_refresh") is True,
        audit.get("eligible_for_explanation") is False,
        audit.get("checkpoint_sha256") == checkpoint_sha256,
    )
    if not all(required):
        raise ValueError(f"{mode} reproduction audit did not pass every locked gate")


def run(args):
    import torch
    from build_explanation_manifest import sha256
    from libs.eval import get_preds_and_probs
    from libs.reproduction_state import (compare_predictions, refresh_training_memory,
                                         restore_checkpoint, save_checkpoint, snapshot)

    root = Path(args.analysis_root).resolve()
    run_dir = root / f"openml_{args.dataset_id}" / f"fold_{args.fold}"
    original_path = run_dir / "checkpoint.pt"
    refreshed_path = run_dir / "checkpoint_refreshed.pt"
    audit_path = run_dir / "audit_memory_refresh.json"
    original_sha = sha256(original_path)
    train_audit = json.loads((run_dir / "audit_train.json").read_text(encoding="utf-8"))
    restore_audit = json.loads((run_dir / "audit_restore.json").read_text(encoding="utf-8"))
    require_reproduced(train_audit, "train", original_sha)
    require_reproduced(restore_audit, "restore_only", original_sha)

    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    wrapper, dataset, original = restore_checkpoint(original_path, device)
    identity = original["identity"]
    if identity["dataset_id"] != args.dataset_id or identity["fold"] != args.fold:
        raise ValueError("Checkpoint identity does not match requested run")
    if original["state_kind"] != "original_post_fit_no_analysis_refresh":
        raise ValueError("Input is not an original post-fit checkpoint")
    test_x, test_y = dataset._indv_dataset()[2]
    with torch.no_grad():
        before_logits = wrapper._forward_batched(test_x)
    before_preds, _ = get_preds_and_probs(before_logits, dataset.tasktype)
    original_reference = compare_predictions(
        before_logits, before_preds, original["test_logits"], original["test_preds"], test_y)
    if not original_reference["passed"]:
        raise ValueError("Original checkpoint no longer reproduces its saved inference output")

    refresh = refresh_training_memory(wrapper, dataset)
    with torch.no_grad():
        after_logits = wrapper._forward_batched(test_x)
    after_preds, _ = get_preds_and_probs(after_logits, dataset.tasktype)
    invariance = compare_predictions(
        after_logits, after_preds, before_logits, before_preds, test_y)
    checks = (
        refresh["parameter_match"], refresh["centroid_match"],
        refresh["training_sample_id_unique"], refresh["training_sample_id_complete"],
        refresh["region_membership_complete"],
        refresh["memory_filled"] == refresh["n_train"], invariance["passed"],
    )
    if not all(checks):
        audit = dict(status="failed_memory_refresh", dataset_id=args.dataset_id,
                     fold=args.fold, original_checkpoint_sha256=original_sha,
                     refresh=refresh, prediction_invariance=invariance,
                     original_state_preserved=sha256(original_path) == original_sha,
                     refreshed_state_saved_separately=False,
                     eligible_for_explanation=False)
        write_json(audit_path, audit)
        print(json.dumps(audit, indent=2))
        return 3

    refreshed = snapshot(
        wrapper, dataset, identity, after_logits, after_preds,
        state_kind="final_encoder_refreshed_for_explanation",
        parent_checkpoint_sha256=original_sha, refresh_audit=refresh)
    save_checkpoint(refreshed_path, refreshed)
    refreshed_sha = sha256(refreshed_path)
    restored, restored_dataset, restored_payload = restore_checkpoint(refreshed_path, device)
    restored_x, restored_y = restored_dataset._indv_dataset()[2]
    with torch.no_grad():
        restored_logits = restored._forward_batched(restored_x)
    restored_preds, _ = get_preds_and_probs(restored_logits, restored_dataset.tasktype)
    roundtrip = compare_predictions(
        restored_logits, restored_preds, after_logits, after_preds, restored_y)
    original_preserved = sha256(original_path) == original_sha
    kind_ok = (restored_payload["state_kind"] == "final_encoder_refreshed_for_explanation"
               and restored_payload.get("parent_checkpoint_sha256") == original_sha)
    passed = all(checks) and roundtrip["passed"] and original_preserved and kind_ok
    audit = dict(status="memory_refresh_passed" if passed else "failed_memory_refresh",
                 dataset_id=args.dataset_id, fold=args.fold,
                 original_checkpoint_sha256=original_sha,
                 refreshed_checkpoint_sha256=refreshed_sha,
                 original_reference=original_reference, refresh=refresh,
                 prediction_invariance=invariance, refreshed_checkpoint_roundtrip=roundtrip,
                 original_state_preserved=bool(original_preserved),
                 refreshed_state_saved_separately=bool(refreshed_path != original_path and kind_ok),
                 eligible_for_explanation=bool(passed))
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2))
    return 0 if passed else 3


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", default="analysis_results")
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(10))
    p.add_argument("--gpu-id", type=int, default=0)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
