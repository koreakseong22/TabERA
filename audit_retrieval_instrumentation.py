"""Audit retrieval branch metadata against an eligible refreshed checkpoint."""
import argparse
import json
from pathlib import Path


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    import torch
    from build_explanation_manifest import sha256
    from libs.reproduction_state import restore_checkpoint
    from libs.retrieval_audit import instrument_retrieval

    run_dir = Path(args.analysis_root).resolve() / f"openml_{args.dataset_id}" / f"fold_{args.fold}"
    source = run_dir / "checkpoint_refreshed.pt"
    source_sha = sha256(source)
    refresh_audit = json.loads((run_dir / "audit_memory_refresh.json").read_text(encoding="utf-8"))
    if not (refresh_audit.get("status") == "memory_refresh_passed"
            and refresh_audit.get("eligible_for_explanation") is True
            and refresh_audit.get("refreshed_checkpoint_sha256") == source_sha):
        raise ValueError("Refreshed checkpoint did not pass the locked memory-refresh gate")
    device = "cuda:0" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu"
    wrapper, dataset, payload = restore_checkpoint(source, device)
    if payload.get("state_kind") != "final_encoder_refreshed_for_explanation":
        raise ValueError("Expected a final-encoder refreshed checkpoint")
    if payload["identity"]["dataset_id"] != args.dataset_id or payload["identity"]["fold"] != args.fold:
        raise ValueError("Checkpoint identity does not match requested run")
    test_x = dataset._indv_dataset()[2][0]
    _, _, trace, instrumentation = instrument_retrieval(
        wrapper.model, test_x, int(wrapper.params.get("batch_size", 512)))
    passed = instrumentation["metadata_off_on_equal"] and instrumentation["model_state_match"]
    audit = dict(status="retrieval_instrumentation_passed" if passed else
                         "failed_retrieval_instrumentation",
                 dataset_id=args.dataset_id, fold=args.fold,
                 refreshed_checkpoint_sha256=source_sha,
                 instrumentation=instrumentation,
                 eligible_for_explanation_metrics=bool(passed))
    write_json(run_dir / "retrieval_trace.json", trace)
    write_json(run_dir / "audit_retrieval.json", audit)
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
