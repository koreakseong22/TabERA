"""
run_final_benchmark.py — official 25-dataset benchmark for TabERA.

This is not diagnostic tooling. It runs the final, frozen procedure:

    optimize.py   (HPO over the locked search space)
        -> reproduce.py --final_readout_refine
               train the best configuration
               refit the shared readout on the frozen h
               evaluate once on test

The readout refinement is a fixed post-training step, not a search
dimension. lr, weight decay, epochs, patience and null-inclusion are all
fixed below and are never re-selected per dataset.

    lr = 1e-2, wd = 0, epochs = 500, patience = 50, include_null = True

Model selection is closed. All architecture and procedure decisions were
made on the diagnostic datasets before this benchmark was run; nothing here
picks between raw TabERA and the refined model. Raw test metrics are still
recorded inside each run (meta["readout_refinement"]["raw_test_metrics"])
for the ablation row, but they are never used to choose.

Reporting is two-layer. Eight of these datasets were used repeatedly in the
diagnostic analyses that shaped the final design (geometry variants, matched
no-partition controls, the h-probe, the readout probe, the k-means ablation
and the Q3 evidence experiment). Dataset 1493 informed the regional
calibration formula and the P<K discussion, and dataset 25 was examined
before the N_train<100 exclusion rule was written. Results on those ten are
reported, but they are not independent confirmation, so the fifteen
remaining datasets are summarised separately.

Usage
-----
  python run_final_benchmark.py --gpu_id 0                 # HPO already done
  python run_final_benchmark.py --gpu_id 0 --run_hpo       # run optimize.py too
  python run_final_benchmark.py --shard 0 --n_shards 4     # split across GPUs
  python run_final_benchmark.py --aggregate
"""
from __future__ import annotations

import argparse, glob, os, pickle, subprocess, sys
from pathlib import Path

# optimize.py / reproduce.py 와 **같은 함수**로 study 파일명을 만든다.
# 이 tag 를 runner 가 따로 하드코딩했다가 존재하는 study 125개를 전부 missing
# 으로 보고했다. PROTOCOL_TAG 가 바뀌어도 자동으로 따라가야 한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libs.search_space import study_pkl_tag  # noqa: E402

# ── The benchmark ────────────────────────────────────────────
# (openml_id, name, tasktype)
BENCHMARK = [
    (51, "heart-h", "binclass"),
    (25, "colic", "binclass"),
    (334, "monks-problems-2", "binclass"),
    (470, "profb", "binclass"),
    (29, "credit-approval", "binclass"),
    (40981, "Australian", "binclass"),
    (31, "credit-g", "binclass"),
    (934, "socmob", "binclass"),
    (1067, "kc1", "binclass"),
    (41143, "jasmine", "binclass"),
    (1043, "ada-agnostic", "binclass"),
    (1489, "phoneme", "binclass"),
    (40536, "SpeedDating", "binclass"),
    (846, "elevators", "binclass"),
    (1486, "nomao", "binclass"),
    (151, "electricity", "binclass"),
    (10, "lymph", "multiclass"),
    (11, "balance-scale", "multiclass"),
    (54, "vehicle", "multiclass"),
    (1493, "one-hundred-plants-texture", "multiclass"),
    (14, "mfeat-fourier", "multiclass"),
    (22, "mfeat-zernike", "multiclass"),
    (46, "splice", "multiclass"),
    (1459, "artificial-characters", "multiclass"),
    (41027, "jungle_chess_2pcs_raw_endgame_complete", "multiclass"),
]

# Datasets whose results informed the final design. Not independent.
DIAGNOSTIC_IDS = {31, 22, 1043, 1067, 1486, 1489, 151, 41143}
REFERENCE_IDS = {1493}      # regional-calibration formula, P<K regime
AUDITED_IDS = {25}          # examined before the N_train<100 rule was fixed

# Pre-declared exclusion: after complete-case preprocessing colic (25) leaves
# about 61 rows, i.e. N_train ~ 49. The rule "exclude datasets with
# N_train < 100 from the aggregate" was fixed during the diagnostic phase, not
# after seeing benchmark numbers. The dataset is still run and reported in the
# appendix; it is only kept out of the headline and the statistics.
EXCLUDED_IDS = {25}

TOUCHED = DIAGNOSTIC_IDS | REFERENCE_IDS | AUDITED_IDS
EVALUATED = [d for d, _, _ in BENCHMARK if d not in EXCLUDED_IDS]
UNTOUCHED = [d for d in EVALUATED if d not in TOUCHED]

# Frozen refinement settings. Do not tune these per dataset.
REFINE = dict(lr=1e-2, wd=0.0, epochs=500, patience=50)

# reproduce.py 의 _readout_refine_tag() 가 REFINE_V1 에 붙이는 태그와 같아야 한다.
SAVE_TAG = "..cat_onehot..num_ple..readoutRefineV1"

# 기본 프로토콜(onehot + PLE, override 없음)의 study tag.
STUDY_TAG = study_pkl_tag(
    cat_combine="onehot", num_embedding="ple", n_prototypes=None,
    disable_dead_reinit=False, num_bins=8, cat_embed_dim=16, batch_size=None,
)


def study_path(savepath, seed, ds):
    return Path(savepath) / "optim_logs" / f"seed={seed}" / \
        f"data={ds}{STUDY_TAG}..model=tabera.pkl"


def result_path(savepath, seed, ds):
    return Path(savepath) / "optim_logs" / f"seed={seed}" / \
        f"data={ds}{SAVE_TAG}..seed{seed}_meta.pkl"


def run(cmd, dry):
    print("  $ " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n_trials", type=int, default=100,
                    help="HPO budget. Must match the baselines' budget.")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--savepath", type=str, default=".")
    ap.add_argument("--run_hpo", action="store_true",
                    help="call optimize.py when the study file is missing")
    ap.add_argument("--only_ds", type=int, nargs="*", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--audit_hpo", action="store_true",
                    help="실행 전 점검: seed 별 study 가 실제로 존재하고 "
                         "COMPLETE trial 수가 --n_trials 와 같은지 확인한다")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.savepath, args.seeds)
        return
    if args.audit_hpo:
        audit_hpo(args.savepath, args.seeds, args.n_trials)
        return

    # 사전 규칙(N_train<100)으로 aggregate 에서 빠지는 dataset 은 HPO 도 돌리지
    # 않는다. colic 은 5 trials 만 있는데 나머지 95개를 채워도 appendix 한 줄에만
    # 쓰인다.
    ids = [d for d, _, _ in BENCHMARK if d not in EXCLUDED_IDS]
    if args.only_ds:
        ids = [d for d in ids if d in args.only_ds]
    ids = [d for i, d in enumerate(ids) if i % args.n_shards == args.shard]
    print(f"benchmark {len(BENCHMARK)} datasets | shard {args.shard}/"
          f"{args.n_shards} -> {len(ids)} | seeds {args.seeds}")
    print(f"touched (not independent): {sorted(TOUCHED)}")
    print(f"excluded from aggregate (N_train<100): {sorted(EXCLUDED_IDS)}")
    print(f"evaluated: {len(EVALUATED)}   untouched: {len(UNTOUCHED)} "
          f"-> {sorted(UNTOUCHED)}\n")

    done = skipped = failed = 0
    for ds in ids:
        for sd in args.seeds:
            out = result_path(args.savepath, sd, ds)
            if out.is_file():
                print(f"[skip] ds={ds} seed={sd} (already done)")
                skipped += 1
                continue
            sp = study_path(args.savepath, sd, ds)
            if not sp.is_file():
                hpo = [sys.executable, "optimize.py", "--openml_id", str(ds),
                       "--seed", str(sd), "--n_trials", str(args.n_trials),
                       "--gpu_id", str(args.gpu_id),
                       "--savepath", args.savepath]
                if not args.run_hpo:
                    print(f"[need HPO] ds={ds} seed={sd}: {sp} missing")
                    print("  $ " + " ".join(hpo))
                    skipped += 1
                    continue
                print(f"\n### HPO ds={ds} seed={sd}")
                if run(hpo, args.dry_run) != 0:
                    print(f"  [FAIL] optimize.py ds={ds} seed={sd}")
                    failed += 1
                    continue
            print(f"\n### FINAL ds={ds} seed={sd}")
            cmd = [sys.executable, "reproduce.py", "--openml_id", str(ds),
                   "--seed", str(sd), "--gpu_id", str(args.gpu_id),
                   "--savepath", args.savepath,
                   "--final_readout_refine",
                   "--refine_lr", str(REFINE["lr"]),
                   "--refine_wd", str(REFINE["wd"]),
                   "--refine_epochs", str(REFINE["epochs"]),
                   "--refine_patience", str(REFINE["patience"])]
            if run(cmd, args.dry_run) == 0:
                done += 1
            else:
                print(f"  [FAIL] reproduce.py ds={ds} seed={sd}")
                failed += 1

    print(f"\ndone {done}, skipped {skipped}, failed {failed}")
    if not args.dry_run:
        aggregate(args.savepath, args.seeds)


def audit_hpo(savepath, seeds, expected):
    """Check the HPO protocol before spending 12,500 trials.

    reproduce.py also supports --params_seed, i.e. reusing one fold's study
    across folds, so the presence of per-seed directories does not by itself
    prove that every fold got its own full search. This counts COMPLETE
    trials per (dataset, seed) and flags anything that differs from the
    expected budget. A mismatch here means TabERA and the baselines did not
    get the same budget, which no amount of good numbers will fix.
    """
    import joblib
    n_need = len([d for d, _, _ in BENCHMARK if d not in EXCLUDED_IDS]) * len(seeds)
    print(f"expected {expected} COMPLETE trials per (dataset, seed)")
    print(f"target: {n_need} studies "
          f"({len(BENCHMARK) - len(EXCLUDED_IDS)} datasets x {len(seeds)} seeds)")
    print(f"looking for: data=<id>{STUDY_TAG}..model=tabera.pkl\n")
    bad = missing = ok = 0
    for ds, name, _ in BENCHMARK:
        if ds in EXCLUDED_IDS:
            print(f"  ds={ds:<6}{name[:26]:<28}  [excluded: N_train<100, "
                  f"HPO 불필요]")
            continue
        line, flags = f"  ds={ds:<6}{name[:26]:<28}", []
        for sd in seeds:
            f = study_path(savepath, sd, ds)
            if not f.is_file():
                line += "  --"; missing += 1; continue
            try:
                st = joblib.load(f)
                n = sum(1 for t in st.trials if str(t.state) == "TrialState.COMPLETE")
            except Exception as e:
                line += "  ER"; flags.append(f"seed{sd}:{type(e).__name__}")
                bad += 1; continue
            line += f"  {n:>3}"
            if n != expected:
                flags.append(f"seed{sd}={n}"); bad += 1
            else:
                ok += 1
        print(line + ("   <- " + ", ".join(flags) if flags else ""))
    print(f"\n  ok {ok}, mismatched {bad}, missing {missing}")
    if missing:
        # 전부 missing 이면 tag 불일치일 가능성이 높다. 실제 파일을 보여준다.
        found = sorted(glob.glob(os.path.join(
            savepath, "optim_logs", "seed=*", "*model=tabera.pkl")))
        if found:
            print(f"\n  ⚠ 디스크에는 study {len(found)}개가 있다. 예:")
            for f in found[:3]:
                print(f"      {os.path.basename(f)}")
            print(f"    기대한 tag: '{STUDY_TAG}'  — 다르면 STUDY_TAG 를 맞출 것")
        else:
            print(f"\n  {savepath}/optim_logs 아래에 study 파일이 없다. "
                  f"경로가 맞는지 확인할 것 (--savepath)")
    if bad:
        print("  ⚠ trial 수가 다르면 baseline 과 동일 예산이 아니다. "
              "실행 전에 확인할 것")


def aggregate(savepath, seeds):
    import numpy as np
    import pandas as pd

    rows = []
    name_of = {d: n for d, n, _ in BENCHMARK}
    task_of = {d: t for d, _, t in BENCHMARK}
    for sd in seeds:
        for ds, _, _ in BENCHMARK:
            f = result_path(savepath, sd, ds)
            if not f.is_file():
                continue
            with open(f, "rb") as fh:
                meta = pickle.load(fh)
            tm = meta.get("test_metrics") or meta.get("Performance") or {}
            ri = meta.get("readout_refinement") or {}
            raw = ri.get("raw_test_metrics") or {}
            r = dict(ds=ds, name=name_of[ds], tasktype=task_of[ds], seed=sd,
                     touched=ds in TOUCHED,
                     null_selected=ri.get("null_selected"),
                     val_before=ri.get("val_loss_before"),
                     val_after=ri.get("val_loss_after"))
            for k, v in tm.items():
                r[k.replace("_test", "")] = float(v)
            for k, v in raw.items():
                r["raw_" + k.replace("_test", "")] = float(v)
            rows.append(r)

    if not rows:
        print("no results found")
        return
    raw_df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    out = Path(savepath) / "final_benchmark"
    out.mkdir(exist_ok=True)
    raw_df.to_csv(out / "benchmark_raw.csv", index=False)

    g = raw_df.groupby(["ds", "name", "tasktype", "touched"])
    ag = g.mean(numeric_only=True).drop(columns=["seed"], errors="ignore")
    ag["n_seed"] = g["seed"].nunique().values
    ag = ag.reset_index()
    ag.to_csv(out / "benchmark_by_dataset.csv", index=False)

    # bacc 는 TabERA 전용 진단 지표라 baseline 과 직접 비교하면 안 된다.
    # CSV 에는 남기고 비교 표에서는 뺀다.
    metrics = [m for m in ("acc", "auroc", "f1", "logloss") if m in ag.columns]
    ag["excluded"] = ag.ds.isin(EXCLUDED_IDS)
    ev = ag[~ag.excluded]
    print(f"\n=== all {len(ag)} datasets (dataset = unit of analysis) ===")
    print(ag[["ds", "name", "tasktype", "touched", "n_seed"] + metrics]
          .round(4).to_string(index=False))

    for label, sub in ((f"ALL EVALUATED (n={len(ev)})", ev),
                       (f"UNTOUCHED (n={len(ev[~ev.touched])})", ev[~ev.touched])):
        if not len(sub):
            continue
        print(f"\n=== {label} ===")
        for m in metrics:
            print(f"  {m:<9} mean={sub[m].mean():.4f}  median={sub[m].median():.4f}")
        # refinement vs raw, for the ablation row only
        for m in metrics:
            rm = "raw_" + m
            if rm not in sub.columns or sub[rm].isna().all():
                continue
            d = sub[m] - sub[rm]
            if m == "logloss":
                d = -d
            print(f"  refine vs raw {m:<9} improved {int((d > 0).sum())}/"
                  f"{d.notna().sum()}  median={d.median():+.4f}")
        if "null_selected" in sub.columns and sub.null_selected.notna().any():
            print(f"  null candidate selected (mean over folds): "
                  f"{sub.null_selected.mean():.3f}")

    exc = ag[ag.excluded]
    if len(exc):
        print(f"\n=== EXCLUDED (pre-declared N_train<100 rule; appendix only) ===")
        print(exc[["ds", "name", "n_seed"] + metrics].round(4).to_string(index=False))

    miss = [(d, s) for d, _, _ in BENCHMARK for s in seeds
            if not result_path(savepath, s, d).is_file()]
    if miss:
        print(f"\n  missing {len(miss)} runs, e.g. {miss[:6]}")
    print("\n  touched datasets are reported but are not independent "
          "confirmation; the untouched block is the generalisation summary")


if __name__ == "__main__":
    main()