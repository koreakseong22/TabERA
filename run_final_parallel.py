"""
run_final_parallel.py -- final 25-dataset x 5-seed benchmark, dispatched dynamically over GPUs.

Every (dataset, seed) pair is one job: optimize.py (up to 100 trials) -> reproduce.py.
Jobs sit in one shared queue; each worker is pinned to a GPU and pulls the next
job as soon as it finishes the previous one, so GPUs never wait for each other.
Jobs are ordered largest-dataset-first so the long ones do not end up alone at
the tail of a single GPU.

Resumable: optimize.py continues an existing study and reproduce.py skips a
verified existing result, so re-running this script after a crash or Ctrl+C
simply carries on. A job whose reproduce result already exists is not launched
at all.

    python run_final_parallel.py --gpus 0 1                       # all 25 x seeds 1-5
    python run_final_parallel.py --gpus 0 1 --per_gpu 2           # two jobs per GPU
    python run_final_parallel.py --gpus 0 --only_ds 31 1493 --seeds 1 2
    python run_final_parallel.py --gpus 0 1 --dry_run
    python run_final_benchmark.py --seeds 1 2 3 4 5 --savepath . --aggregate   # afterwards

Per-job logs: <savepath>/final_logs/<recipe>/ds<id>_seed<k>.log; a running ledger in
<savepath>/final_logs/<recipe>/progress.tsv (job, gpu, status, seconds, finished-at).
"""
from __future__ import annotations
import argparse, csv, json, os, queue, subprocess, sys, threading, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BENCHMARK = [51, 25, 334, 470, 29, 40981, 31, 934, 1067, 41143, 1043, 1489, 40536, 846, 1486, 151,
             10, 11, 54, 1493, 14, 22, 46, 1459, 41027]


def dataset_size(ds):
    """Rows in the cached target vector, for longest-first ordering; 0 if not cached."""
    try:
        import numpy as np
        return int(len(np.load(ROOT / "data_cache" / f"{ds}_y.npy", allow_pickle=True)))
    except Exception:
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=int, nargs="+", required=True, help="physical GPU ids; repeat an id to run several workers on it")
    p.add_argument("--per_gpu", type=int, default=1, help="concurrent jobs per listed GPU")
    p.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=[1, 2, 3, 4, 5])
    p.add_argument("--only_ds", type=int, nargs="+")
    p.add_argument("--savepath", default=".", help="same root for optimize.py and reproduce.py (existing seed-1 studies live in '.')")
    p.add_argument("--mode", choices=["best", "all"], default="best")
    p.add_argument("--n_trials", type=int, default=100, help="HPO budget; anything but 100 is a smoke test, not a benchmark")
    p.add_argument("--skip_hpo", action="store_true", help="reproduce only (studies must already be complete)")
    p.add_argument("--allow_unverified_study", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    info = json.loads((ROOT / "dataset_id.json").read_text(encoding="utf-8"))
    ids = args.only_ds or BENCHMARK
    unknown = set(ids) - {int(k) for k in info}
    if unknown:
        p.error(f"Unknown datasets: {sorted(unknown)}")
    from libs.benchmark import result_path, arm_config
    config = arm_config()
    save = Path(args.savepath).resolve()
    from libs.search_space import RECIPE_TAG
    log_dir = save / "final_logs" / RECIPE_TAG.removeprefix("..recipe=")
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger = log_dir / "progress.tsv"

    # Longest-processing-time-first: big datasets first, then by seed.
    order = sorted(ids, key=lambda d: -dataset_size(d))
    jobs, done = [], []
    for ds in order:
        for seed in args.seeds:
            if result_path(save, seed, ds, config=config).exists():
                done.append((ds, seed))
            else:
                jobs.append((ds, seed))
    print(f"{len(jobs)} jobs to run, {len(done)} already have a reproduce result, "
          f"{len(args.gpus) * args.per_gpu} workers on GPUs {args.gpus}")
    if args.dry_run:
        for ds, seed in jobs:
            print(f"  ds={ds:<6} seed={seed}  N={dataset_size(ds)}  {info[str(ds)]['name']}")
        return

    q: "queue.Queue[tuple[int, int]]" = queue.Queue()
    for j in jobs:
        q.put(j)
    lock = threading.Lock()
    running: dict[tuple[int, int], subprocess.Popen] = {}
    failed: list[tuple[int, int, str]] = []
    stop = threading.Event()

    def record(ds, seed, gpu, status, seconds):
        with lock, ledger.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerow([f"ds={ds}", f"seed={seed}", f"gpu={gpu}", status, f"{seconds:.0f}",
                                                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        print(f"[{datetime.now():%H:%M:%S}] gpu{gpu} ds={ds} seed={seed} {status} ({seconds:.0f}s)  "
              f"remaining {q.qsize()}", flush=True)

    def run_step(name, cmd, log, ds, seed, gpu):
        step_start = time.time()
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] gpu{gpu} ds={ds} seed={seed} "
              f"START {name}", flush=True)
        with log.open("a", encoding="utf-8") as out:
            out.write(f"\n===== {name} {datetime.now():%Y-%m-%d %H:%M:%S} =====\n{subprocess.list2cmdline(cmd)}\n")
            out.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
            with lock:
                running[(ds, seed)] = proc
            code = proc.wait()
            with lock:
                running.pop((ds, seed), None)
        step_seconds = time.time() - step_start
        outcome = "DONE" if code == 0 else f"FAILED(rc={code})"
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] gpu{gpu} ds={ds} seed={seed} "
              f"{outcome} {name} elapsed={step_seconds:.0f}s", flush=True)
        return code

    def worker(gpu):
        while not stop.is_set():
            try:
                ds, seed = q.get_nowait()
            except queue.Empty:
                return
            log = log_dir / f"ds{ds}_seed{seed}.log"
            base = ["--openml_id", str(ds), "--seed", str(seed), "--gpu_id", str(gpu), "--savepath", str(save),
                    "--correction_geometry", "unit_tangent", "--head_input_scale", "auto"]
            t0 = time.time()
            status = "ok"
            if not args.skip_hpo:
                code = run_step("optimize", [sys.executable, str(ROOT / "optimize.py"), *base, "--n_trials", str(args.n_trials)],
                                log, ds, seed, gpu)
                if code != 0:
                    status = f"hpo_failed({code})"
            if status == "ok":
                code = run_step("reproduce", [sys.executable, str(ROOT / "reproduce.py"), *base, "--mode", args.mode,
                                              *(["--allow_unverified_study"] if args.allow_unverified_study else [])],
                                log, ds, seed, gpu)
                if code != 0:
                    status = f"reproduce_failed({code})"
            if status != "ok":
                with lock:
                    failed.append((ds, seed, status))
            record(ds, seed, gpu, status, time.time() - t0)
            q.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True)
               for gpu in args.gpus for _ in range(args.per_gpu)]
    t_start = time.time()
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
    except KeyboardInterrupt:
        stop.set()
        with lock:
            procs = list(running.values())
        for proc in procs:
            proc.terminate()
        print("\ninterrupted: running jobs terminated; re-run the same command to resume", flush=True)
        raise SystemExit(130)
    print(f"\nfinished in {(time.time() - t_start) / 60:.1f} min: {len(jobs) - len(failed)} ok, {len(failed)} failed")
    for ds, seed, status in failed:
        print(f"  ds={ds} seed={seed}: {status}   see {log_dir / f'ds{ds}_seed{seed}.log'}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
