"""Run final TabERA on an explicit set of MultiTab datasets and folds.

The one sanctioned ablation arm, dead-prototype recovery off, runs through the
same script with --disable_dead_reinit: it is passed to optimize.py and
reproduce.py alike, so the arm gets its own HPO study (..nodr), its own
contract and its own result files, and never touches the main arm's outputs.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", choices=range(10), default=[1, 2, 3, 4, 5])
    p.add_argument("--only_ds", type=int, nargs="+")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--savepath", default=".")
    p.add_argument("--run_hpo", action="store_true")
    p.add_argument("--mode", choices=["best", "all"], default="best")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--allow_unverified_study", action="store_true")
    p.add_argument("--disable_dead_reinit", action="store_true",
                   help="ablation arm: dead-prototype recovery off, for HPO, reproduction and aggregation alike")
    p.add_argument("--early_stop_metric", choices=["val_loss", "accuracy", "logloss", "auroc", "bacc"], default=None,
                   help="ablation arm: validation metric for checkpoint selection/patience (MultiTab neural baselines: val_loss, terminal checkpoint)")
    p.add_argument("--hpo_source", choices=["own", "main"], default="own",
                   help=("for an ablation arm: own = the arm's own HPO (two tuned models); main = reuse the main "
                         "arm's study and hold the hyperparameters fixed (the arm's variable is the only change). "
                         "main needs no --run_hpo and reads/writes <arm>..hpo=main files"))
    args = p.parse_args()
    from libs.benchmark import arm_config, is_main_arm
    config = arm_config(args.disable_dead_reinit, args.early_stop_metric)
    if args.hpo_source == "main":
        if is_main_arm(config):
            p.error("--hpo_source main only applies with an ablation arm (--disable_dead_reinit / --early_stop_metric)")
        if args.run_hpo:
            p.error("--hpo_source main reuses the main arm's study; drop --run_hpo")
    root = Path(__file__).resolve().parent
    info = json.loads((root / "dataset_id.json").read_text(encoding="utf-8"))
    ids = args.only_ds or [int(k) for k in info]
    unknown = set(ids) - {int(k) for k in info}
    if unknown:
        p.error(f"Unknown datasets: {sorted(unknown)}")
    if args.aggregate:
        import numpy as np
        from libs.benchmark import arm_tag, implementation_id, result_path
        from libs.search_space import RECIPE_TAG
        rows = []
        for seed in args.seeds:
            for ds in ids:
                path = result_path(args.savepath, seed, ds, config=config, hpo_source=args.hpo_source)
                if not path.exists():
                    print(f"[missing] {path}")
                    continue
                saved = np.load(path, allow_pickle=True).item()
                if not isinstance(saved, dict) or "identity" not in saved:
                    raise ValueError(f"Unrecognized result: {path}")
                identity = saved["identity"]
                recorded = identity.get("contract", {})
                if (recorded.get("config") != config or recorded.get("implementation") != implementation_id()
                        or identity.get("dataset_id") != ds or identity.get("fold") != seed
                        or identity.get("hpo_source", "own") != args.hpo_source):
                    raise ValueError(f"Incompatible result: {path}")
                diag = {f"proto_{k}": v for k, v in (saved.get("prototype_diag") or {}).items()}
                rows.append(dict(data=ds, seed=seed, task=info[str(ds)]["tasktype"],
                                 unverified_hpo=saved["identity"]["unverified_hpo"],
                                 **saved["Performance"], **diag))
        hpo = "" if args.hpo_source == "own" else f"..hpo={args.hpo_source}"
        out = Path(args.savepath) / f"results/tabera_final_per_fold{RECIPE_TAG}{arm_tag(config)}{hpo}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            writer.writeheader()
            writer.writerows(rows)
        print(out)
        return
    failed = 0
    arm = (["--disable_dead_reinit"] if args.disable_dead_reinit else []) + \
          (["--early_stop_metric", args.early_stop_metric] if args.early_stop_metric else [])
    for ds in ids:
        for seed in args.seeds:
            base = ["--openml_id", str(ds), "--seed", str(seed), "--gpu_id", str(args.gpu_id),
                    "--savepath", str(Path(args.savepath).resolve()),
                    "--correction_geometry", "unit_tangent", "--head_input_scale", "auto"]
            commands = []
            if args.run_hpo:
                commands.append([sys.executable, str(root / "optimize.py"), *base, *arm, "--n_trials", "100"])
            commands.append([sys.executable, str(root / "reproduce.py"), *base, *arm,
                             *(["--hpo_source", args.hpo_source] if args.hpo_source != "own" else []),
                             "--mode", args.mode,
                             *(["--allow_unverified_study"] if args.allow_unverified_study else [])])
            for command in commands:
                print(subprocess.list2cmdline(command), flush=True)
                if not args.dry_run and subprocess.call(command, cwd=root):
                    failed += 1
                    break
    if failed:
        raise SystemExit(f"{failed} dataset/fold runs failed")


if __name__ == "__main__":
    main()
