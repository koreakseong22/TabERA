"""Resume the locked reproduction-to-metrics pipeline over manifest runs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def passed(path, status, eligible=None):
    value = read_json(path)
    return bool(value and value.get("status") == status and
                (eligible is None or value.get(eligible) is True))


def command(script, dataset_id, fold, gpu_id, extra=()):
    return [sys.executable, str(ROOT / script), "--dataset-id", str(dataset_id),
            "--fold", str(fold), "--gpu-id", str(gpu_id), *extra]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", default="analysis_results")
    p.add_argument("--manifest", default="analysis_results/manifest.json")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--only-dataset", type=int, nargs="+")
    p.add_argument("--only-fold", type=int, nargs="+", choices=range(1, 6))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(args.analysis_root).resolve()
    manifest = json.loads(Path(args.manifest).resolve().read_text(encoding="utf-8"))
    rows = [row for row in manifest["runs"]
            if (not args.only_dataset or row["dataset_id"] in args.only_dataset)
            and (not args.only_fold or row["fold"] in args.only_fold)]
    results = []
    for row in rows:
        ds, fold = row["dataset_id"], row["fold"]
        run_dir = root / f"openml_{ds}" / f"fold_{fold}"
        effective_gpu = 0 if args.gpu_id >= 0 else -1
        stages = [
            ("train", run_dir / "audit_train.json", "reproduction_passed",
             "eligible_for_memory_refresh", command(
                 "reproduce_with_checkpoint.py", ds, fold, effective_gpu,
                 ("--manifest", str(Path(args.manifest).resolve()),
                  "--output", str(run_dir), "--train"))),
            ("restore", run_dir / "audit_restore.json", "reproduction_passed",
             "eligible_for_memory_refresh", command(
                 "reproduce_with_checkpoint.py", ds, fold, effective_gpu,
                 ("--manifest", str(Path(args.manifest).resolve()),
                  "--output", str(run_dir), "--restore-only"))),
            ("refresh", run_dir / "audit_memory_refresh.json", "memory_refresh_passed",
             "eligible_for_explanation", command(
                 "refresh_explanation_checkpoint.py", ds, fold, effective_gpu,
                 ("--analysis-root", str(root)))),
            ("retrieval", run_dir / "audit_retrieval.json",
             "retrieval_instrumentation_passed", "eligible_for_explanation_metrics",
             command("audit_retrieval_instrumentation.py", ds, fold, effective_gpu,
                     ("--analysis-root", str(root)))),
            ("metrics", run_dir / "audit_metrics.json", "explanation_metrics_passed",
             "eligible_for_aggregation", command(
                 "analyze_explanation_structure.py", ds, fold, effective_gpu,
                 ("--analysis-root", str(root)))),
        ]
        outcome = "success"
        for name, audit, status, eligible, cmd in stages:
            if passed(audit, status, eligible):
                continue
            if args.dry_run:
                print(subprocess.list2cmdline(cmd))
                outcome = "pending"
                break
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id) if args.gpu_id >= 0 else ""
            code = subprocess.call(cmd, cwd=ROOT, env=env)
            if code or not passed(audit, status, eligible):
                current = read_json(audit)
                outcome = ((current or {}).get("status") or
                           ("missing" if not audit.exists() else "incomplete"))
                break
        results.append(dict(dataset_id=ds, fold=fold, status=outcome))
        print(f"dataset={ds} fold={fold}: {outcome}", flush=True)
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = dict(total=len(results), counts=counts, runs=results)
    (root / "batch_status.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(total=len(results), counts=counts), indent=2))
    return 0 if all(row["status"] in ("success", "pending") for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
