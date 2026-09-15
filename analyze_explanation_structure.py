"""Compute locked explanation metrics for one eligible refreshed checkpoint."""
import argparse
import json
from pathlib import Path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    import pandas as pd
    import torch
    from build_explanation_manifest import sha256
    from libs.explanation_metrics import METRIC_PROTOCOL_VERSION, compute_explanation_metrics
    from libs.reproduction_state import restore_checkpoint

    run_dir = Path(args.analysis_root).resolve() / f"openml_{args.dataset_id}" / f"fold_{args.fold}"
    checkpoint = run_dir / "checkpoint_refreshed.pt"
    checkpoint_sha = sha256(checkpoint)
    retrieval_audit_path = run_dir / "audit_retrieval.json"
    retrieval_trace_path = run_dir / "retrieval_trace.json"
    retrieval_audit = json.loads(retrieval_audit_path.read_text(encoding="utf-8"))
    if not (retrieval_audit.get("status") == "retrieval_instrumentation_passed"
            and retrieval_audit.get("eligible_for_explanation_metrics") is True
            and retrieval_audit.get("refreshed_checkpoint_sha256") == checkpoint_sha):
        raise ValueError("Retrieval instrumentation did not pass the locked gate")
    stored_trace = json.loads(retrieval_trace_path.read_text(encoding="utf-8"))
    metric_outputs = [run_dir / name for name in (
        "query_metrics.parquet", "region_stats.parquet", "neighbors_tabera.parquet",
        "neighbors_global.parquet", "summary.json")]
    audit_path = run_dir / "audit_metrics.json"
    existing_audit = None
    if audit_path.exists():
        try:
            existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if existing_audit and existing_audit.get("status") == "explanation_metrics_passed":
        raise FileExistsError("Successful metric artifacts already exist")
    # Derived files without a successful audit are untrusted remnants of a
    # failed or interrupted attempt. Removing them makes the metric stage
    # resumable while preserving every successfully audited result.
    for path in metric_outputs:
        path.unlink(missing_ok=True)
    audit_path.unlink(missing_ok=True)
    targets = [*metric_outputs, audit_path]
    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    wrapper, dataset, payload = restore_checkpoint(checkpoint, device)
    state_before = {name: value.detach().cpu().clone()
                    for name, value in wrapper.model.state_dict().items()}
    result = compute_explanation_metrics(wrapper, dataset, stored_trace)
    state_after = wrapper.model.state_dict()
    state_match = all(
        (torch.allclose(before, state_after[name].detach().cpu(), rtol=0, atol=0, equal_nan=True)
         if torch.is_floating_point(before) or torch.is_complex(before)
         else torch.equal(before, state_after[name].detach().cpu()))
        for name, before in state_before.items())
    identity = payload["identity"]
    if (int(identity["dataset_id"]), int(identity["fold"])) != (args.dataset_id, args.fold):
        raise ValueError("Checkpoint identity does not match requested dataset/fold")
    train_seed = int(identity["train_seed"])
    summary = dict(dataset_id=args.dataset_id, fold=args.fold, train_seed=train_seed,
                   tasktype=dataset.tasktype, **result["summary"])
    runner_path = Path(__file__).resolve()
    metrics_path = runner_path.parent / "libs" / "explanation_metrics.py"
    integrity = dict(
        metric_protocol_version=METRIC_PROTOCOL_VERSION,
        metrics_runner_sha256=sha256(runner_path),
        metrics_code_sha256=sha256(metrics_path),
        retrieval_audit_sha256=sha256(retrieval_audit_path),
        retrieval_trace_sha256=sha256(retrieval_trace_path),
        source_checkpoint_sha256=checkpoint_sha,
        source_checkpoint_preserved=sha256(checkpoint) == checkpoint_sha,
        model_state_match=bool(state_match),
        retrieval_metadata_off_on_equal=result["instrumentation"]["metadata_off_on_equal"],
        decomposition_passed=result["summary"]["decomposition_max_abs_error"] < 1e-5,
        prediction_identity_passed=result["summary"]["accuracy_delta_identity_error"] <= 1e-12,
    )
    passed = all(value for key, value in integrity.items()
                 if key.endswith("_passed") or key.endswith("_match") or
                 key.endswith("_equal") or key.endswith("_preserved"))
    if not passed:
        raise ValueError(f"Explanation metric integrity failed: {integrity}")
    common = dict(dataset_id=args.dataset_id, fold=args.fold, train_seed=train_seed)
    pd.DataFrame([dict(common, **row) for row in result["query_rows"]]).to_parquet(targets[0], index=False)
    pd.DataFrame([dict(common, **row) for row in result["region_rows"]]).to_parquet(targets[1], index=False)
    pd.DataFrame([dict(common, **row) for row in result["local_neighbor_rows"]]).to_parquet(targets[2], index=False)
    pd.DataFrame([dict(common, **row) for row in result["global_neighbor_rows"]]).to_parquet(targets[3], index=False)
    write_json(targets[4], summary)
    audit = dict(status="explanation_metrics_passed", dataset_id=args.dataset_id,
                 fold=args.fold, train_seed=train_seed, integrity=integrity,
                 eligible_for_aggregation=True)
    write_json(targets[5], audit)
    print(json.dumps(dict(audit=audit, summary=summary), indent=2))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", default="analysis_results")
    p.add_argument("--dataset-id", type=int, required=True)
    p.add_argument("--fold", type=int, required=True, choices=range(10))
    p.add_argument("--gpu-id", type=int, default=0)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        run_dir = (Path(args.analysis_root).resolve() /
                   f"openml_{args.dataset_id}" / f"fold_{args.fold}")
        run_dir.mkdir(parents=True, exist_ok=True)
        audit_path = run_dir / "audit_metrics.json"
        existing = None
        if audit_path.exists():
            try:
                existing = json.loads(audit_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        # Never replace a prior successful audit with a failed re-run record.
        if not (existing and existing.get("status") == "explanation_metrics_passed"):
            write_json(audit_path, dict(
                status="failed_explanation_metrics", dataset_id=args.dataset_id,
                fold=args.fold, error_type=type(exc).__name__, error=str(exc),
                eligible_for_aggregation=False))
        raise
