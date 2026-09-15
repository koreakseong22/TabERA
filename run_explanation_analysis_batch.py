"""Dynamically resume the locked explanation pipeline across one or more GPUs."""
import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

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


def stages_for(row, root, manifest_path, physical_gpu):
    ds, fold = row["dataset_id"], row["fold"]
    run_dir = root / f"openml_{ds}" / f"fold_{fold}"
    # reproduce_with_checkpoint.py sets CUDA_VISIBLE_DEVICES itself, so it
    # must receive the physical ID. Later stage scripts inherit the worker's
    # visibility mask and therefore address that one visible device as cuda:0.
    reproduce_gpu = physical_gpu
    stage_gpu = -1 if physical_gpu < 0 else 0
    return [
        ("train", run_dir / "audit_train.json", "reproduction_passed",
         "eligible_for_memory_refresh", command(
             "reproduce_with_checkpoint.py", ds, fold, reproduce_gpu,
             ("--manifest", str(manifest_path), "--output", str(run_dir), "--train"))),
        ("restore", run_dir / "audit_restore.json", "reproduction_passed",
         "eligible_for_memory_refresh", command(
             "reproduce_with_checkpoint.py", ds, fold, reproduce_gpu,
             ("--manifest", str(manifest_path), "--output", str(run_dir), "--restore-only"))),
        ("refresh", run_dir / "audit_memory_refresh.json", "memory_refresh_passed",
         "eligible_for_explanation", command(
             "refresh_explanation_checkpoint.py", ds, fold, stage_gpu,
             ("--analysis-root", str(root)))),
        ("retrieval", run_dir / "audit_retrieval.json",
         "retrieval_instrumentation_passed", "eligible_for_explanation_metrics",
         command("audit_retrieval_instrumentation.py", ds, fold, stage_gpu,
                 ("--analysis-root", str(root)))),
        ("metrics", run_dir / "audit_metrics.json", "explanation_metrics_passed",
         "eligible_for_aggregation", command(
             "analyze_explanation_structure.py", ds, fold, stage_gpu,
             ("--analysis-root", str(root)))),
    ]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", default="analysis_results")
    p.add_argument("--manifest", default="analysis_results/manifest.json")
    devices = p.add_mutually_exclusive_group()
    devices.add_argument("--gpus", type=int, nargs="+",
                         help="physical GPU IDs; one dynamic worker per ID")
    devices.add_argument("--gpu-id", type=int,
                         help="legacy single-device form; -1 selects CPU")
    p.add_argument("--only-dataset", type=int, nargs="+")
    p.add_argument("--only-fold", type=int, nargs="+", choices=range(1, 6))
    p.add_argument("--dry-run", action="store_true")
    return p


def run(args):
    root = Path(args.analysis_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest["runs"]
            if (not args.only_dataset or row["dataset_id"] in args.only_dataset)
            and (not args.only_fold or row["fold"] in args.only_fold)]
    rows.sort(key=lambda row: (-int(row.get("n_test", 0)), row["dataset_id"], row["fold"]))
    physical_gpus = args.gpus if args.gpus is not None else [
        args.gpu_id if args.gpu_id is not None else 0]
    if len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("GPU IDs must be unique")
    if any(gpu < -1 for gpu in physical_gpus) or (-1 in physical_gpus and len(physical_gpus) > 1):
        raise ValueError("Use CPU (-1) alone or one or more nonnegative GPU IDs")

    if args.dry_run:
        results = []
        for position, row in enumerate(rows):
            gpu = physical_gpus[position % len(physical_gpus)]
            outcome = "success"
            for _, audit, status, eligible, cmd in stages_for(
                    row, root, manifest_path, gpu):
                if passed(audit, status, eligible):
                    continue
                print(f"[gpu {gpu}] {subprocess.list2cmdline(cmd)}")
                outcome = "pending"
                break
            results.append(dict(dataset_id=row["dataset_id"], fold=row["fold"],
                                gpu_id=gpu, status=outcome))
    else:
        jobs = queue.Queue()
        for row in rows:
            jobs.put(row)
        results, lock = [], threading.Lock()
        log_dir = root / "batch_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        def worker(gpu):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if gpu < 0 else str(gpu)
            while True:
                try:
                    row = jobs.get_nowait()
                except queue.Empty:
                    return
                ds, fold = row["dataset_id"], row["fold"]
                outcome = "success"
                log_path = log_dir / f"openml_{ds}_fold_{fold}.log"
                for name, audit, status, eligible, cmd in stages_for(
                        row, root, manifest_path, gpu):
                    if passed(audit, status, eligible):
                        continue
                    with log_path.open("a", encoding="utf-8") as log:
                        log.write(f"\n===== {name} =====\n{subprocess.list2cmdline(cmd)}\n")
                        log.flush()
                        code = subprocess.call(cmd, cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    if code or not passed(audit, status, eligible):
                        current = read_json(audit)
                        outcome = ((current or {}).get("status") or
                                   ("missing" if not audit.exists() else "incomplete"))
                        break
                result = dict(dataset_id=ds, fold=fold, gpu_id=gpu, status=outcome,
                              log_path=str(log_path.relative_to(root)))
                with lock:
                    results.append(result)
                    print(f"[gpu {gpu}] dataset={ds} fold={fold}: {outcome} "
                          f"({jobs.qsize()} queued)", flush=True)
                jobs.task_done()

        threads = [threading.Thread(target=worker, args=(gpu,), daemon=True)
                   for gpu in physical_gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    results.sort(key=lambda row: (row["dataset_id"], row["fold"]))
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = dict(total=len(results), gpus=physical_gpus, counts=counts, runs=results)
    (root / "batch_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(dict(total=len(results), gpus=physical_gpus, counts=counts), indent=2))
    return 0 if all(row["status"] in ("success", "pending") for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
