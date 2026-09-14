"""Frozen binary readout screening, isolated from the benchmark implementation.

Audit: python analyze_split_head_auc.py --audit
Run:   python analyze_split_head_auc.py --manifest checkpoints.json
Or explicitly permit new same-HP training: --rebuild-from-results

Manifest: [{"data": 51, "seed": 1, "checkpoint": "path/to/state.pt"}, ...].
Checkpoints must reproduce the corresponding current result's test logits and
validation metrics. Older architectures are never substituted automatically.
Caches and result files live under --output; benchmark files are never written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

GRID = tuple(10.0 ** np.arange(-4, 5))
ARMS = ("Q", "Shared", "Split-match", "Split-free")
VERSION = 1


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, Path):
            return str(x)
        return x
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pair_auc(y, score, region):
    """Exact tie-aware pair accounting without an O(N^2) matrix."""
    y, score, region = np.asarray(y), np.asarray(score), np.asarray(region)
    def wins(yy, ss):
        neg = np.sort(ss[yy == 0])
        pos = ss[yy == 1]
        return float((np.searchsorted(neg, pos, side="left") +
                      np.searchsorted(neg, pos, side="right")).sum() / 2), len(neg) * len(pos)
    total_wins, total_n = wins(y, score)
    same_wins, same_n = 0., 0
    for r in np.unique(region):
        mask = region == r
        w, n = wins(y[mask], score[mask])
        same_wins += w
        same_n += n
    cross_n = total_n - same_n
    return dict(auc=total_wins / total_n if total_n else None,
                same_auc=same_wins / same_n if same_n else None,
                cross_auc=(total_wins - same_wins) / cross_n if cross_n else None,
                same_pairs=same_n, cross_pairs=cross_n,
                same_pair_fraction=same_n / total_n if total_n else None)


def metrics(y, score, region):
    pred = np.asarray(score) > 0
    return dict(pair_auc(y, score, region), acc=accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0),
                logloss=float(np.mean(np.logaddexp(0, score) - y * score)))


def linear_fit(x, y, C, penalty, maxiter):
    """Sum BCE + ||W||^2/(2C); intercept unpenalized.

    RMS conditioning is a solver coordinate change, NOT altered features or
    regularization. Returned weights are in the original input coordinates.
    """
    scale = np.maximum(np.sqrt(np.mean(x * x, axis=0)), 1e-8)
    xx = x / scale
    pp = np.broadcast_to(penalty, (x.shape[1],)) / scale ** 2
    def objective(t):
        z = xx @ t[:-1] + t[-1]
        e = expit(z) - y
        f = np.sum(np.logaddexp(0, z) - y * z) + .5 / C * np.dot(pp, t[:-1] ** 2)
        g = np.r_[xx.T @ e + pp * t[:-1] / C, e.sum()]
        return f, g
    start = np.zeros(x.shape[1] + 1)
    start[-1] = np.log(y.mean() / (1 - y.mean()))
    fit = minimize(objective, start, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-7, maxls=50))
    return dict(w=fit.x[:-1] / scale, b=float(fit.x[-1]),
                objective=float(fit.fun), success=bool(fit.success), nit=int(fit.nit),
                message=str(fit.message), grad_inf=float(np.max(np.abs(fit.jac))))


def match_objective(t, c, d, y, C):
    """wc=a, wd=||a|| v/||v||, hence exact binary norm equality.

    Both norms contribute half to the split penalty, so its restriction to
    wc=wd equals the shared penalty. This avoids a hidden 2x L2 difference.
    """
    dim = c.shape[1]
    a, v, b = t[:dim], t[dim:2 * dim], t[-1]
    r, s = max(np.linalg.norm(a), 1e-15), max(np.linalg.norm(v), 1e-15)
    u = v / s
    z = c @ a + r * (d @ u) + b
    e = expit(z) - y
    du = d.T @ e
    f = np.sum(np.logaddexp(0, z) - y * z) + .5 / C * np.dot(a, a)
    ga = c.T @ e + np.dot(du, u) * a / r + a / C
    gv = r / s * (du - u * np.dot(u, du))
    return float(f), np.r_[ga, gv, e.sum()]


def match_fit(c, d, y, C, shared, free, maxiter):
    dim = c.shape[1]
    a = shared["w"]
    if np.linalg.norm(a) < 1e-12:
        a = np.random.default_rng(0).normal(size=dim) * 1e-4
    # Three fixed initializations. Choose by TRAIN penalized objective only.
    fa, fv = free["w"][:dim], free["w"][dim:]
    if min(np.linalg.norm(fa), np.linalg.norm(fv)) < 1e-12:
        fa, fv = a, -a
    starts = [np.r_[a, a, shared["b"]], np.r_[fa, fv, free["b"]],
              np.r_[a, -a, shared["b"]]]
    candidates = []
    for start in starts:
        fit = minimize(match_objective, start, args=(c, d, y, C), jac=True,
                       method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-12, gtol=1e-7, maxls=50))
        candidates.append(fit)
    fit = min(candidates, key=lambda f: f.fun)
    a, v = fit.x[:dim], fit.x[dim:2 * dim]
    wd = np.linalg.norm(a) * v / np.linalg.norm(v)
    return dict(w=np.r_[a, wd], b=float(fit.x[-1]), objective=float(fit.fun),
                success=bool(fit.success), nit=int(fit.nit), message=str(fit.message),
                grad_inf=float(np.max(np.abs(fit.jac))),
                norm_error=float(abs(np.linalg.norm(a) - np.linalg.norm(wd))),
                starts=[dict(objective=float(f.fun), success=bool(f.success), nit=int(f.nit)) for f in candidates])


def designs(cache, split):
    g = float(cache["gamma"])
    c, d = g * cache[f"{split}_c"], g * cache[f"{split}_d"]
    return {"Q": cache[f"{split}_q"], "Shared": c + d,
            "Split-match": np.c_[c, d], "Split-free": np.c_[c, d]}


def screen(cache, grid=GRID, maxiter=1500):
    train, val = designs(cache, "train"), designs(cache, "val")
    y, yv = cache["train_y"], cache["val_y"]
    if set(np.unique(y)) != {0, 1} or set(np.unique(yv)) != {0, 1}:
        return dict(status="undefined_train_or_validation_auc")
    selected, trials = {}, []
    for C in grid:
        shared = linear_fit(train["Shared"], y, C, 1., maxiter)
        free = linear_fit(train["Split-free"], y, C, .5, maxiter)
        dim = train["Shared"].shape[1]
        fits = {"Shared": shared, "Split-free": free,
                "Q": linear_fit(train["Q"], y, C, 1., maxiter),
                "Split-match": match_fit(train["Split-free"][:, :dim],
                                         train["Split-free"][:, dim:], y, C, shared, free, maxiter)}
        for arm in ARMS:
            fit = fits[arm]
            auc = float(roc_auc_score(yv, val[arm] @ fit["w"] + fit["b"]))
            entry = dict(arm=arm, C=C, val_auc=auc, **fit)
            trials.append(entry)
            # Grid sorted ascending; ties retain stronger regularization.
            if arm not in selected or auc > selected[arm]["val_auc"]:
                selected[arm] = entry
    # No test scores or labels are used until ALL selections have been fixed.
    test = designs(cache, "test")
    for arm, fit in selected.items():
        score = test[arm] @ fit["w"] + fit["b"]
        fit["test"] = metrics(cache["test_y"], score, cache["test_region"])
        fit["test_logits"] = score
        fit["val"] = metrics(yv, val[arm] @ fit["w"] + fit["b"], cache["val_region"])
        if arm.startswith("Split"):
            dim = len(fit["w"]) // 2
            fit["wc_norm"] = float(np.linalg.norm(fit["w"][:dim]))
            fit["wd_norm"] = float(np.linalg.norm(fit["w"][dim:]))
    baselines = {name: metrics(cache["test_y"], cache[f"test_{key}"], cache["test_region"])
                 for name, key in (("Original", "logits"), ("Region-only", "region_logits"))}
    return dict(status="complete", selected=selected, trials=trials, baselines=baselines)


def result_file(root, data, seed):
    from libs.search_space import RECIPE_TAG
    return root / "reproduce_logs" / f"seed={seed}" / f"data={data}" / (
        f"model=tabera{RECIPE_TAG}..init_hps=False..deep=0..hyper=0.npy")


def extract(model, dataset):
    import torch
    model.eval()
    model.requires_grad_(False)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    output = dict(gamma=np.array(model.effective_gamma()),
                  beta=np.array(float(model.effective_beta().detach().mean())))
    with torch.inference_mode():
        for name, (x, y) in zip(("train", "val", "test"), dataset._indv_dataset()):
            arrays = {k: [] for k in ("q", "c", "d", "h", "region", "logits", "region_logits")}
            for xx in x.split(512):
                out = model(xx)
                q, c, d = out["query_emb"], out["context_emb"], out["correction"]
                z = out["logits"].flatten()
                rz = model.dev_head(model.effective_gamma() * c).flatten()
                expected = rz + torch.nn.functional.linear(d, model.effective_W()).flatten()
                if not torch.allclose(z, expected, atol=2e-5, rtol=2e-5):
                    raise ValueError("Shared logit decomposition failed")
                shared_design_logits = model.dev_head(model.effective_gamma() * (c + d)).flatten()
                shared_design_error = float((z - shared_design_logits).abs().max())
                if shared_design_error >= 2e-5:
                    raise ValueError(f"Shared design gamma placement failed: {shared_design_error}")
                output["shared_design_max_abs_diff"] = np.array(max(
                    float(output.get("shared_design_max_abs_diff", 0.)), shared_design_error))
                for key, tensor in dict(q=q, c=c, d=d, h=c+d, region=out["centroid_id"],
                                        logits=z, region_logits=rz).items():
                    arrays[key].append(tensor.cpu().numpy())
            for key, parts in arrays.items():
                output[f"{name}_{key}"] = np.concatenate(parts).astype(np.int64 if key == "region" else np.float64)
            output[f"{name}_y"] = y.cpu().numpy().astype(np.int64).reshape(-1)
    for k, v in model.state_dict().items():
        if not torch.allclose(before[k], v, rtol=0, atol=0, equal_nan=True):
            raise ValueError(f"Extraction mutated frozen state: {k}")
    return output


def get_cache(args, data, seed, entry):
    os.environ.setdefault("OPENML_CACHE_DIR", str(args.root.resolve() / "data_cache" / "openml"))
    import torch
    from libs.benchmark import build_wrapper, data_signature, implementation_id
    from libs.data import TabularDataset
    from libs.tabera import TabERA
    source = result_file(args.root, data, seed)
    saved = np.load(source, allow_pickle=True).item()
    identity = saved["identity"]
    if identity["dataset_id"] != data or identity["fold"] != seed or identity["tasktype"] != "binclass":
        raise ValueError("Source result identity mismatch")
    checkpoint = Path(entry["checkpoint"]) if entry else None
    mode = "existing_checkpoint" if checkpoint else "same_hp_retraining"
    cache_path = args.output / f"data={data}_seed={seed}_features.npz"
    provenance = dict(version=VERSION, data=data, seed=seed, mode=mode,
                      source_result=str(source), source_sha256=sha256(source),
                      implementation=implementation_id(),
                      checkpoint=str(checkpoint) if checkpoint else None,
                      checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                      source_identity=identity)
    if cache_path.exists():
        meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta["provenance"] != provenance:
            raise ValueError(f"Cache provenance changed: {cache_path}. Use a new output directory.")
        return dict(np.load(cache_path, allow_pickle=False)), meta
    device = "cuda:0" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    dataset = TabularDataset(data, "binclass", device=device, seed=seed)
    if data_signature(dataset) != identity["contract"]["data"]:
        raise ValueError("Data signature differs from source result")
    start = time.perf_counter()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("identity") is not None:
            if state["identity"] != identity:
                raise ValueError("Checkpoint identity does not match source")
            wrapper = build_wrapper(dataset, identity["params"], identity["contract"]["config"], device)
            model = wrapper.model
        else:
            if state.get("seed") != seed:
                raise ValueError("Checkpoint fold mismatch")
            kwargs = dict(state["model_kwargs"])
            model = TabERA(**kwargs).to(device)
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        if not args.rebuild_from_results:
            raise FileNotFoundError("No latest checkpoint. Supply --manifest or explicitly --rebuild-from-results.")
        train_seed = identity["train_seed"]
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)
        wrapper = build_wrapper(dataset, identity["params"], identity["contract"]["config"], device)
        wrapper._data_id = data
        wrapper.epochs = identity["contract"]["schedule"]["epochs"]
        wrapper.patience = identity["contract"]["schedule"]["patience"]
        (xt, yt), (xv, yv), _ = dataset._indv_dataset()
        wrapper.fit(xt, yt, xv, yv)
        model = wrapper.model
        torch.save(dict(state_dict=model.state_dict(), identity=identity,
                        provenance=provenance, actual_implementation=implementation_id()),
                   args.output / f"data={data}_seed={seed}_retrained.pt")
    if model.correction_geometry != "unit_tangent" or model.split_head:
        raise ValueError("Expected shared Unit Tangent checkpoint")
    cache = extract(model, dataset)
    original = np.asarray(saved["Probability"]).reshape(-1)
    if original.shape != cache["test_logits"].shape:
        raise ValueError("Test output shape mismatch")
    max_diff = float(np.max(np.abs(original - cache["test_logits"])))
    if checkpoint and not np.allclose(original, cache["test_logits"], atol=2e-5, rtol=2e-5):
        raise ValueError(f"Checkpoint does not reproduce latest test logits (max diff {max_diff})")
    if checkpoint:
        observed = metrics(cache["val_y"], cache["val_logits"], cache["val_region"])
        for key in ("auc", "acc", "f1", "logloss"):
            ref = saved["Performance_val"].get({"auc": "auroc_val"}.get(key, key + "_val"))
            if ref is not None and not np.isclose(observed[key], ref, atol=2e-5, rtol=2e-5):
                raise ValueError(f"Checkpoint validation {key} mismatch")
    meta = dict(provenance=provenance, feature_seconds=time.perf_counter()-start,
                max_test_logit_difference_from_source=max_diff,
                note="New same-HP training; not the original checkpoint" if not checkpoint else "Verified source outputs",
                actual_environment={p: importlib.metadata.version(p)
                                    for p in ('torch', 'numpy', 'scikit-learn', 'optuna')},
                split_sizes={s: len(cache[f"{s}_y"]) for s in ("train", "val", "test")})
    np.savez_compressed(cache_path, **cache)
    write_json(cache_path.with_suffix(".json"), meta)
    return cache, meta


def summarize(output, records):
    rows = []
    for record in records:
        if record["result"]["status"] != "complete":
            continue
        row = dict(data=record["data"], seed=record["seed"], mode=record["meta"]["provenance"]["mode"])
        for arm, fit in record["result"]["selected"].items():
            row.update({f"{arm}_{k}": v for k, v in fit["test"].items()})
            row.update({f"{arm}_{k}": fit[k] for k in ("C", "val_auc", "success")})
        for arm, result in record["result"]["baselines"].items():
            row.update({f"{arm}_{k}": v for k, v in result.items()})
        rows.append(row)
    if rows:
        with (output / "per_run.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Frozen readout AUROC screening", "", "Test is evaluated after validation selects C. "
             "Undefined AUC stays missing; no fold replacement. Seed is a data fold, not an independent replication.", "",
             "| Data | Seed | Retrained-original | Q | Shared | Split-match | Split-free | Same-pair % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "NA" if x is None or not np.isfinite(x) else f"{x:.4f}"
    for r in rows:
        values = [r[f"{a}_auc"] for a in ("Original", *ARMS)]
        fraction = r["Shared_same_pair_fraction"]
        lines.append(f"| {r['data']} | {r['seed']} | " + " | ".join(map(fmt, values)) +
                     f" | {fmt(100*fraction if fraction is not None else None)} |")
    lines += ["", "Gains below are paired against Shared-refit, not the original head.", ""]
    for arm in ("Split-match", "Split-free"):
        good = [r for r in rows if r[f"{arm}_auc"] is not None and r["Shared_auc"] is not None]
        gains = [r[f"{arm}_auc"]-r["Shared_auc"] for r in good]
        if gains:
            lines.append(f"- {arm}: mean gain {np.mean(gains):+.4f}; wins/ties/losses "
                         f"{sum(g>0 for g in gains)}/{sum(g==0 for g in gains)}/{sum(g<0 for g in gains)} (N={len(gains)}).")
            for data in sorted({r['data'] for r in good}):
                group = [r for r in good if r['data']==data]
                lines.append(f"  - Data {data}: mean gain "
                             f"{np.mean([r[f'{arm}_auc']-r['Shared_auc'] for r in group]):+.4f}; "
                             f"mean ACC change {np.mean([r[f'{arm}_acc']-r['Shared_acc'] for r in group]):+.4f}; "
                             f"mean F1 change {np.mean([r[f'{arm}_f1']-r['Shared_f1'] for r in group]):+.4f}.")
    lines += ["", "Split-match is nonconvex; solver success and all three fixed-start training objectives are saved. "
              "A low Q-probe score does not prove that the encoder contains no useful nonlinear information."]
    (output / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("diagnostics/frozen_split_auc"))
    p.add_argument("--datasets", nargs="+", type=int, default=[51, 1067, 25])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--rebuild-from-results", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--maxiter", type=int, default=1500)
    args = p.parse_args()
    os.environ.setdefault("OPENML_CACHE_DIR", str(args.root.resolve() / "data_cache" / "openml"))
    args.output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else []
    manifest = {(e["data"], e["seed"]): e for e in entries}
    if len(manifest) != len(entries):
        raise ValueError("Duplicate manifest entries")
    audit = []
    for data in args.datasets:
        for seed in args.seeds:
            src = result_file(args.root, data, seed)
            entry = manifest.get((data, seed))
            saved = np.load(src, allow_pickle=True).item() if src.is_file() else {}
            audit.append(dict(data=data, seed=seed, result_exists=src.is_file(),
                              checkpoint=entry.get("checkpoint") if entry else None,
                              checkpoint_exists=bool(entry and Path(entry["checkpoint"]).is_file()),
                              source_val_auc=saved.get("Performance_val", {}).get("auroc_val"),
                              source_fit_seconds=saved.get("time")))
    write_json(args.output / "audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit:
        return
    if any(not r["result_exists"] for r in audit):
        raise FileNotFoundError("Missing source results; see audit.json")
    if not args.rebuild_from_results and any(not r["checkpoint_exists"] for r in audit):
        raise FileNotFoundError("Latest checkpoints missing; see audit.json. No training performed.")
    import torch
    torch.set_num_threads(1)
    config = dict(version=VERSION, grid=GRID, maxiter=args.maxiter,
                  penalty="sum BCE + penalty/(2*C); shared/Q: ||w||^2; split: (||wc||^2+||wd||^2)/2",
                  selection="maximum val logit AUROC; ties choose smallest C",
                  match_starts="shared, projected free, reversed shared; lowest train objective wins",
                  script_sha256=sha256(Path(__file__)))
    config_path = args.output / "experiment.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old["script_sha256"] != config["script_sha256"] or old["maxiter"] != args.maxiter:
            raise ValueError("Experiment changed; choose a new --output directory")
    write_json(config_path, config)
    records = []
    with threadpool_limits(limits=1):
        for data in args.datasets:
            for seed in args.seeds:
                print(f"[run] data={data} seed={seed}", flush=True)
                cache, meta = get_cache(args, data, seed, manifest.get((data, seed)))
                target = args.output / f"data={data}_seed={seed}_readout.json"
                if target.exists():
                    record = json.loads(target.read_text(encoding="utf-8"))
                    if record["meta"] != meta:
                        raise ValueError("Readout cache identity changed")
                else:
                    start = time.perf_counter()
                    record = dict(data=data, seed=seed, meta=meta, result=screen(cache, maxiter=args.maxiter),
                                  readout_seconds=time.perf_counter()-start)
                    write_json(target, record)
                records.append(record)
                summarize(args.output, records)
                print(f"[done] {target} ({record['readout_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
