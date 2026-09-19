"""Aggregate the inference-latency JSONs into the paper table and figure.

Input: results/inference_latency*/ *.json written by bench_inference.py
(TabERA) and ../multitab/scripts/bench_inference_tabr.py (TabR).

Output (in --out, default the input directory):
  inference_latency.csv     one row per dataset and fold
  inference_latency.md      the same as a Markdown table plus the summary
  latency_vs_n.png/.pdf     two panels against N_train, log-log

Two measurements are reported, and they answer different questions.

  batch=1        online latency: one sample, one forward. Dominated by fixed
                 per-call overhead, so it is the setting where an
                 architectural difference is hardest to see.
  full_pass      time to score the entire test split in chunks. This is the
                 deployment-relevant number. Both the query count and the
                 candidate pool grow with the dataset, so a growing ratio
                 here means TabR's per-query candidate search is the term
                 that scales.

The intermediate "batch=<b>" entries are summarised only when every dataset
was actually timed at that batch size (runner flag --tile). Without tiling a
split smaller than b is timed at its own size, which keeps the per-dataset
ratio valid but makes samples/s incomparable across datasets.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# The 21 OpenML datasets shared by the TabERA and TabR benchmarks.
COMMON21 = [10, 11, 14, 22, 25, 29, 31, 46, 51, 54, 334, 470, 846, 934,
            1043, 1067, 1459, 1489, 1493, 40981, 41143]

# Categorical slots in fixed order (validated default palette).
SERIES = {
    "TabR prediction": "#2a78d6",
    "TabERA prediction-only": "#eb6834",
    "TabERA prediction + evidence": "#1baf7a",
}
MARKERS = {"TabR prediction": "o", "TabERA prediction-only": "s",
           "TabERA prediction + evidence": "^"}
TABR_MODE = "prediction"


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
    except (KeyError, TypeError):
        return None


def build_rows(recs, b_small, b_large):
    by = defaultdict(dict)
    for d in recs:
        by[(d["dataset_id"], d["fold"])][d["model"]] = d
    rows = []
    for (ds, fold), models in sorted(by.items()):
        a, t = models.get("tabera"), models.get("tabr")
        src = a or t
        r = {"dataset_id": ds, "fold": fold, "dataset": src.get("dataset"),
             "tasktype": src.get("tasktype"), "n_train": src["n_train"],
             "n_eval": src["n_eval"]}
        if t:
            r["tabr_ms_b1"] = stat(t, TABR_MODE, f"batch={b_small}", "ms_per_sample_median")
            r["tabr_ms_full"] = stat(t, TABR_MODE, "full_pass", "ms_median")
            r["tabr_ms_large"] = stat(t, TABR_MODE, f"batch={b_large}", "ms_median")
            r["batch_rows_large"] = stat(t, TABR_MODE, f"batch={b_large}", "batch_rows")
            r["tiled_large"] = bool(stat(t, TABR_MODE, f"batch={b_large}", "tiled"))
            r["tabr_n_candidates"] = t.get("n_candidates")
            backend = t.get("index_backend")
            r["tabr_backend"] = f"{backend} {'GPU' if t.get('faiss_gpu') else 'CPU'}"
        if a:
            r["tabera_ms_b1"] = stat(a, "prediction_only", f"batch={b_small}", "ms_per_sample_median")
            r["tabera_ms_full"] = stat(a, "prediction_only", "full_pass", "ms_median")
            r["tabera_ms_large"] = stat(a, "prediction_only", f"batch={b_large}", "ms_median")
            r["tabera_ev_ms_b1"] = stat(a, "prediction_retrieval", f"batch={b_small}",
                                        "ms_per_sample_median")
            r["tabera_ev_ms_full"] = stat(a, "prediction_retrieval", "full_pass", "ms_median")
            r["tabera_ex_ms_full"] = stat(a, "prediction_explain", "full_pass", "ms_median")
            r["n_prototypes"] = a.get("n_prototypes")
            r["tabera_arm"] = f"{a.get('correction_geometry')}/{a.get('head_input_scale')}"
            if r.get("batch_rows_large") is None:
                r["batch_rows_large"] = stat(a, "prediction_only", f"batch={b_large}", "batch_rows")
                r["tiled_large"] = bool(stat(a, "prediction_only", f"batch={b_large}", "tiled"))
        for name, num, den in (("speedup_b1", "tabr_ms_b1", "tabera_ms_b1"),
                               ("speedup_full", "tabr_ms_full", "tabera_ms_full"),
                               ("speedup_large", "tabr_ms_large", "tabera_ms_large")):
            if r.get(num) and r.get(den):
                r[name] = r[num] / r[den]
        for name, ev, base in (("evidence_overhead_ms_b1", "tabera_ev_ms_b1", "tabera_ms_b1"),
                               ("evidence_overhead_ms_full", "tabera_ev_ms_full", "tabera_ms_full")):
            if r.get(ev) is not None and r.get(base) is not None:
                r[name] = r[ev] - r[base]
        rows.append(r)
    return rows


def geomean(v):
    return math.exp(statistics.fmean(math.log(x) for x in v))


def summary(rows):
    out = {}
    for key, label in (("speedup_b1", "Speedup, batch 1 (online latency)"),
                       ("speedup_full", "Speedup, full test split")):
        v = [r[key] for r in rows if r.get(key)]
        if v:
            out[label] = {"n": len(v), "median": statistics.median(v), "geomean": geomean(v),
                          "min": min(v), "max": max(v), "unit": "x"}
    tiled = [r.get("tiled_large") for r in rows if "tiled_large" in r]
    if tiled and all(tiled) and any(r.get("speedup_large") for r in rows):
        v = [r["speedup_large"] for r in rows if r.get("speedup_large")]
        out["Speedup, fixed batch (tiled)"] = {
            "n": len(v), "median": statistics.median(v), "geomean": geomean(v),
            "min": min(v), "max": max(v), "unit": "x"}
    for key, label in (("evidence_overhead_ms_b1", "Evidence retrieval overhead, batch 1"),
                       ("evidence_overhead_ms_full", "Evidence retrieval overhead, full split")):
        v = [r[key] for r in rows if r.get(key) is not None]
        if v:
            out[label] = {"n": len(v), "median": statistics.median(v),
                          "min": min(v), "max": max(v), "unit": "ms"}
    return out


def environments(recs):
    envs = defaultdict(set)
    for d in recs:
        e = d.get("environment", {})
        envs[d["model"]].add((e.get("gpu") or e.get("device"), e.get("torch"),
                              e.get("physical_gpu_id")))
    return envs


def fmt(x, nd=3):
    if x is None:
        return ""
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def write_table(rows, summ, envs, out_dir, b_small, b_large):
    cols = ["dataset_id", "dataset", "tasktype", "fold", "n_train", "n_eval", "n_prototypes",
            "tabr_n_candidates", "tabr_ms_b1", "tabera_ms_b1", "speedup_b1",
            "tabr_ms_full", "tabera_ms_full", "speedup_full",
            "tabr_ms_large", "tabera_ms_large", "speedup_large", "batch_rows_large", "tiled_large",
            "tabera_ev_ms_b1", "evidence_overhead_ms_b1",
            "tabera_ev_ms_full", "evidence_overhead_ms_full", "tabera_ex_ms_full",
            "tabr_backend", "tabera_arm"]
    with open(out_dir / "inference_latency.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["n_train"]))

    L = ["# Inference latency: TabR vs TabERA", ""]
    for model in sorted(envs):
        for gpu, torch_v, gid in sorted(envs[model], key=str):
            gid_s = "" if gid is None else f", GPU index {gid}"
            L.append(f"- {model}: {gpu}, torch {torch_v}{gid_s}")
    all_gpus = {g for s in envs.values() for g, _, _ in s}
    all_torch = {t for s in envs.values() for _, t, _ in s}
    if len(all_gpus) > 1:
        L += ["", "**More than one GPU in this directory: do not pool these rows.**"]
    elif len(all_torch) > 1:
        L += ["", "**The two models ran under different torch builds. State both versions with "
                  "the numbers, or re-time one model under the other's runtime.**"]
    L += ["",
          f"`batch {b_small}` is online latency in ms per sample. `full split` is the time to "
          "score the whole test split in chunks, in ms. Speedup is TabR divided by TabERA "
          "prediction-only; above 1 means TabERA is faster. Evidence overhead is TabERA with "
          "retrieval minus prediction-only.", ""]
    tiled = [r.get("tiled_large") for r in rows if "tiled_large" in r]
    if tiled and not all(tiled):
        short = [r["dataset_id"] for r in rows if r.get("batch_rows_large") not in (None, b_large)]
        L += [f"The `batch {b_large}` columns are kept out of the summary: {len(short)} of "
              f"{len(rows)} datasets have a test split smaller than {b_large}, so they were timed "
              "at their own size. Per-dataset ratios stay valid; samples/s across datasets does "
              "not. Re-run the runners with `--tile` for a fixed-batch comparison.", ""]

    hdr = ["dataset", "N train", "n test", "P", f"TabR ms (b{b_small})",
           f"TabERA ms (b{b_small})", "speedup", "TabR ms (full)", "TabERA ms (full)",
           "speedup", "evidence +ms (full)"]
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for r in sorted(rows, key=lambda r: r["n_train"]):
        L.append("| " + " | ".join([
            f"{r['dataset_id']} {r.get('dataset') or ''}".strip(),
            str(r["n_train"]), str(r["n_eval"]), fmt(r.get("n_prototypes")),
            fmt(r.get("tabr_ms_b1")), fmt(r.get("tabera_ms_b1")), fmt(r.get("speedup_b1"), 2),
            fmt(r.get("tabr_ms_full")), fmt(r.get("tabera_ms_full")), fmt(r.get("speedup_full"), 2),
            fmt(r.get("evidence_overhead_ms_full")),
        ]) + " |")
    L += ["", "## Summary over datasets", ""]
    for label, s in summ.items():
        if s["unit"] == "x":
            L.append(f"- {label}: median {s['median']:.2f}x, geometric mean {s['geomean']:.2f}x, "
                     f"range {s['min']:.2f}x to {s['max']:.2f}x (n={s['n']})")
        else:
            L.append(f"- {label}: median {s['median']:.3f} ms, "
                     f"range {s['min']:.3f} to {s['max']:.3f} ms (n={s['n']})")
    backends = {r.get("tabr_backend") for r in rows if r.get("tabr_backend")}
    if backends:
        L += ["", f"TabR candidate search: {', '.join(sorted(backends))}. "
                  "With a CPU index the query embeddings leave the GPU and the results come back "
                  "each call; that round trip is part of TabR's measured latency."]
    (out_dir / "inference_latency.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return L


def plot(rows, out_dir, b_small):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    panels = [
        ("tabr_ms_b1", "tabera_ms_b1", "tabera_ev_ms_b1",
         f"Online latency (batch {b_small})", "ms per sample"),
        ("tabr_ms_full", "tabera_ms_full", "tabera_ev_ms_full",
         "Scoring the full test split", "ms per pass"),
    ]
    # Each point is one dataset, so the points are NOT connected: a line would
    # read as a trajectory through a single system. The dashed line is an
    # explicit least-squares fit in log-log, whose slope (annotated at its
    # right end) is the scaling exponent in N.
    def tick(v, _):
        return f"{v:g}"

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), dpi=200)
    handles = None
    for ax, (kt, ka, ke, title, ylab) in zip(axes, panels):
        data = {
            "TabR prediction": [(r["n_train"], r[kt]) for r in rows if r.get(kt)],
            "TabERA prediction-only": [(r["n_train"], r[ka]) for r in rows if r.get(ka)],
            "TabERA prediction + evidence": [(r["n_train"], r[ke]) for r in rows if r.get(ke)],
        }
        for name, color in SERIES.items():
            p = sorted(data[name])
            if not p:
                continue
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            if len(p) >= 4:
                lx = [math.log10(x) for x in xs]
                ly = [math.log10(y) for y in ys]
                mx, my = statistics.fmean(lx), statistics.fmean(ly)
                sxx = sum((x - mx) ** 2 for x in lx)
                slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx if sxx else 0.0
                fx = [min(xs), max(xs)]
                fy = [10 ** (my + slope * (math.log10(x) - mx)) for x in fx]
                ax.plot(fx, fy, color=color, linewidth=1.0, linestyle=(0, (4, 3)),
                        alpha=0.75, zorder=2)
                # The slope is this panel's own fit, so it is annotated in the
                # panel rather than in the shared legend, where one number
                # would be read as applying to both panels.
                ax.annotate(f"{slope:+.2f}", xy=(fx[1], fy[1]), xytext=(3, 0),
                            textcoords="offset points", color=color, fontsize=7,
                            va="center", ha="left", zorder=4,
                            annotation_clip=False)
            ax.scatter(xs, ys, s=22, color=color, marker=MARKERS[name], edgecolor="white",
                       linewidth=0.7, zorder=3, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(tick))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(tick))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_minor_formatter(NullFormatter())
        ax.margins(x=0.12)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Training-set size $N$", fontsize=8.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.grid(True, which="major", color="#e8e8e8", linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    if handles:
        fig.legend(*handles, frameon=False, fontsize=7.4, loc="lower center",
                   ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out_dir / "latency_vs_n.png")
    fig.savefig(out_dir / "latency_vs_n.pdf")
    return out_dir / "latency_vs_n.png"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="results/inference_latency")
    p.add_argument("--out", default=None)
    p.add_argument("--batch_small", type=int, default=1)
    p.add_argument("--batch_large", type=int, default=512)
    p.add_argument("--datasets", default="common21",
                   help="'common21' (default), 'all', or a comma-separated list of OpenML ids")
    args = p.parse_args()
    recs = load(args.dir)
    if not recs:
        raise SystemExit(f"no JSON in {args.dir}")
    if args.datasets != "all":
        wanted = (COMMON21 if args.datasets == "common21"
                  else [int(x) for x in args.datasets.split(",")])
        recs = [d for d in recs if d["dataset_id"] in wanted]
        for model in ("tabr", "tabera"):
            have = {d["dataset_id"] for d in recs if d["model"] == model}
            missing = [x for x in wanted if x not in have]
            if missing:
                print(f"[missing] {model}: {missing}")
    out_dir = Path(args.out or args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(recs, args.batch_small, args.batch_large)
    summ = summary(rows)
    lines = write_table(rows, summ, environments(recs), out_dir,
                        args.batch_small, args.batch_large)
    print("\n".join(lines))
    fig = plot(rows, out_dir, args.batch_small)
    print(f"\nwrote {out_dir / 'inference_latency.csv'}, {out_dir / 'inference_latency.md'}"
          + (f", {fig}" if fig else ""))


if __name__ == "__main__":
    main()
