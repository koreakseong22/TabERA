"""Aggregate the inference-latency JSONs into the paper table and the
latency-vs-N figure.

Input: results/inference_latency/*.json written by bench_inference.py
(TabERA) and ../multitab/scripts/bench_inference_tabr.py (TabR).

Output (in --out, default the input directory):
  inference_latency.csv     one row per dataset and fold
  inference_latency.md      the same as a Markdown table plus the summary
  latency_vs_n.png/.pdf     batch-1 ms/sample against N_train, log-log

Speedup is TabR / TabERA prediction-only on the same dataset and fold,
summarised as median and geometric mean over datasets. Evidence overhead is
TabERA prediction_retrieval minus prediction_only. Files whose environment
(GPU, torch) differs from the first one are reported, since numbers taken on
different hardware must not be pooled.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Categorical slots in fixed order (validated default palette): TabR first,
# TabERA prediction-only second, TabERA with evidence third.
SERIES = {
    "TabR prediction": "#2a78d6",
    "TabERA prediction-only": "#eb6834",
    "TabERA prediction + evidence": "#1baf7a",
}


def load(dir_):
    recs = []
    for p in sorted(Path(dir_).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        d["_file"] = p.name
        recs.append(d)
    return recs


def stat(d, mode, key, field):
    try:
        return d["timing"][mode][key][field]
    except KeyError:
        return None


def build_rows(recs, b_small, b_large):
    by = defaultdict(dict)
    for d in recs:
        by[(d["dataset_id"], d["fold"])][d["model"]] = d
    rows = []
    for (ds, fold), models in sorted(by.items()):
        r = {"dataset_id": ds, "fold": fold}
        a = models.get("tabera")
        t = models.get("tabr")
        src = a or t
        r["dataset"] = src.get("dataset")
        r["tasktype"] = src.get("tasktype")
        r["n_train"] = src["n_train"]
        r["n_eval"] = src["n_eval"]
        if t:
            r["tabr_ms_b1"] = stat(t, "prediction", f"batch={b_small}", "ms_per_sample_median")
            r["tabr_sps_b512"] = stat(t, "prediction", f"batch={b_large}", "samples_per_s")
            r["tabr_backend"] = f"{t.get('index_backend')}{' (GPU)' if t.get('faiss_gpu') else ' (CPU)'}"
            r["tabr_n_candidates"] = t.get("n_candidates")
            r["tabr_params_borrowed_from"] = t.get("params_borrowed_from")
        if a:
            r["tabera_ms_b1"] = stat(a, "prediction_only", f"batch={b_small}", "ms_per_sample_median")
            r["tabera_sps_b512"] = stat(a, "prediction_only", f"batch={b_large}", "samples_per_s")
            r["tabera_ev_ms_b1"] = stat(a, "prediction_retrieval", f"batch={b_small}", "ms_per_sample_median")
            r["tabera_ev_sps_b512"] = stat(a, "prediction_retrieval", f"batch={b_large}", "samples_per_s")
            r["n_prototypes"] = a.get("n_prototypes")
            r["tabera_arm"] = f"{a.get('correction_geometry')}/{a.get('head_input_scale')}"
        if a and t and r.get("tabr_ms_b1") and r.get("tabera_ms_b1"):
            r["speedup_b1"] = r["tabr_ms_b1"] / r["tabera_ms_b1"]
            r["speedup_b512"] = r["tabera_sps_b512"] / r["tabr_sps_b512"]
        if a and r.get("tabera_ev_ms_b1") is not None:
            r["evidence_overhead_ms_b1"] = r["tabera_ev_ms_b1"] - r["tabera_ms_b1"]
        rows.append(r)
    return rows


def summary(rows):
    out = {}
    for key in ("speedup_b1", "speedup_b512"):
        v = [r[key] for r in rows if r.get(key)]
        if v:
            out[key] = {"n": len(v), "median": statistics.median(v),
                        "geomean": math.exp(statistics.fmean(math.log(x) for x in v)),
                        "min": min(v), "max": max(v)}
    v = [r["evidence_overhead_ms_b1"] for r in rows if r.get("evidence_overhead_ms_b1") is not None]
    if v:
        out["evidence_overhead_ms_b1"] = {"n": len(v), "median": statistics.median(v)}
    return out


def environments(recs):
    envs = {}
    for d in recs:
        e = d.get("environment", {})
        envs[d["_file"]] = (e.get("gpu") or e.get("device"), e.get("torch"))
    distinct = sorted(set(envs.values()), key=str)
    return distinct, envs


def fmt(x, nd=3):
    return "" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_table(rows, summ, distinct_envs, out_dir, b_small, b_large):
    cols = ["dataset_id", "dataset", "fold", "n_train", "n_prototypes", "tabr_n_candidates",
            "tabr_ms_b1", "tabera_ms_b1", "speedup_b1",
            "tabr_sps_b512", "tabera_sps_b512", "speedup_b512",
            "tabera_ev_ms_b1", "evidence_overhead_ms_b1", "tabr_backend", "tabera_arm",
            "tabr_params_borrowed_from"]
    with open(out_dir / "inference_latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["# Inference latency: TabR vs TabERA", ""]
    lines.append(f"Environment(s): {'; '.join(f'{g}, torch {t}' for g, t in distinct_envs)}")
    gpus = sorted({g for g, _ in distinct_envs}, key=str)
    if len(gpus) > 1:
        lines.append("**More than one GPU in this directory: do not pool these rows.**")
    elif len(distinct_envs) > 1:
        lines.append("Note: TabR (multitab venv) and TabERA (TabERA venv) ran under different torch builds "
                     "on the same GPU; state both versions with the numbers.")
    lines.append("")
    lines.append(f"batch={b_small}: ms/sample (median over calls). batch={b_large}: samples/s. "
                 "Speedup = TabR / TabERA prediction-only. Evidence overhead = TabERA prediction_retrieval "
                 "minus prediction_only at batch 1.")
    lines.append("")
    hdr = ["dataset", "N_train", "P", "TabR ms", "TabERA ms", "speedup", "TabR samples/s",
           "TabERA samples/s", "speedup", "TabERA+evidence ms", "evidence overhead ms", "TabR index"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in sorted(rows, key=lambda r: r["n_train"]):
        lines.append("| " + " | ".join([
            f"{r['dataset_id']} ({r.get('dataset') or ''})", str(r["n_train"]), fmt(r.get("n_prototypes")),
            fmt(r.get("tabr_ms_b1")), fmt(r.get("tabera_ms_b1")), fmt(r.get("speedup_b1"), 2),
            fmt(r.get("tabr_sps_b512"), 0), fmt(r.get("tabera_sps_b512"), 0), fmt(r.get("speedup_b512"), 2),
            fmt(r.get("tabera_ev_ms_b1")), fmt(r.get("evidence_overhead_ms_b1")), r.get("tabr_backend") or ""]) + " |")
    lines.append("")
    lines.append("## Summary over datasets")
    for key, s in summ.items():
        if "geomean" in s:
            lines.append(f"- {key}: n={s['n']}, median {s['median']:.2f}x, geometric mean {s['geomean']:.2f}x, "
                         f"range {s['min']:.2f}x to {s['max']:.2f}x")
        else:
            lines.append(f"- {key}: n={s['n']}, median {s['median']:.3f} ms")
    borrowed = [r for r in rows if r.get("tabr_params_borrowed_from")]
    if borrowed:
        lines.append("")
        lines.append("TabR hyperparameters borrowed from another dataset's study (no own HPO): " +
                     ", ".join(f"{r['dataset_id']} <- {r['tabr_params_borrowed_from']}" for r in borrowed))
    (out_dir / "inference_latency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def plot(rows, out_dir, b_small):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = {name: [] for name in SERIES}
    for r in rows:
        if r.get("tabr_ms_b1"):
            pts["TabR prediction"].append((r["n_train"], r["tabr_ms_b1"], r["dataset_id"]))
        if r.get("tabera_ms_b1"):
            pts["TabERA prediction-only"].append((r["n_train"], r["tabera_ms_b1"], r["dataset_id"]))
        if r.get("tabera_ev_ms_b1"):
            pts["TabERA prediction + evidence"].append((r["n_train"], r["tabera_ev_ms_b1"], r["dataset_id"]))
    if not any(pts.values()):
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    markers = {"TabR prediction": "o", "TabERA prediction-only": "s", "TabERA prediction + evidence": "^"}
    for name, color in SERIES.items():
        p = sorted(pts[name])
        if not p:
            continue
        xs, ys = [q[0] for q in p], [q[1] for q in p]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.6, zorder=2)
        ax.scatter(xs, ys, s=28, color=color, marker=markers[name], edgecolor="white",
                   linewidth=0.8, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Training-set size $N$")
    ax.set_ylabel(f"Latency, ms / sample (batch {b_small})")
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "latency_vs_n.png")
    fig.savefig(out_dir / "latency_vs_n.pdf")
    return out_dir / "latency_vs_n.png"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/inference_latency")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_small", type=int, default=1)
    p.add_argument("--batch_large", type=int, default=512)
    args = p.parse_args()
    recs = load(args.dir)
    if not recs:
        raise SystemExit(f"no JSON in {args.dir}")
    out_dir = Path(args.out or args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(recs, args.batch_small, args.batch_large)
    summ = summary(rows)
    distinct, _ = environments(recs)
    lines = write_table(rows, summ, distinct, out_dir, args.batch_small, args.batch_large)
    print("\n".join(lines))
    fig = plot(rows, out_dir, args.batch_small)
    print(f"\nwrote {out_dir / 'inference_latency.csv'}, {out_dir / 'inference_latency.md'}"
          + (f", {fig}" if fig else ""))


if __name__ == "__main__":
    main()
