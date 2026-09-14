"""Fixed-HP end-to-end head experiment. No HPO or primary-selection changes.

python analyze_e2e_split_head_auc.py --datasets 51 1067 31 --seeds 1 2 3
Prediction adapter is installed before optimizer construction. The ordinary
TabERA encoder/routing/EMA/training implementation is used unchanged. Checkpoint
reloading requires build_wrapper -> attach_head(mode) -> load_state_dict.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from analyze_split_head_auc import metrics, result_file, sha256, write_json

MODES = ("shared", "match", "free")


def correction_weight(model):
    if model.e2e_head_mode == "shared":
        return model.dev_head.weight
    w = model.e2e_correction_head.weight
    if model.e2e_head_mode == "match":
        # Exactly the existing match_w1 convention: radius carries no gradient
        # from the correction branch into Wc. Frozen fitting used full gradients.
        w = model.dev_head.weight.norm().detach() * w / w.norm().clamp_min(1e-12)
    return w


def _readout_hook(model, inputs, out):
    g = model.effective_gamma()
    c, d = out["context_emb"], out["correction"]
    region = model.dev_head(g * c)
    z = region + F.linear(g * d, correction_weight(model))
    out["logits"] = z
    # Keep diagnostics consumed by the standard training wrapper truthful.
    if not model.training:
        diag = out["dev_diag"]
        diag["dev_changed_rate"] = float(((z > 0) != (region > 0)).float().mean())
        diag["dev_unique_logits"] = float(len(torch.unique(z.round(decimals=4), dim=0)))
    return out


def attach_head(model, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if model.split_head or model.correction_geometry != "unit_tangent" or model.dev_head.out_features != 1:
        raise ValueError("This experiment requires binary shared Unit Tangent base models")
    if hasattr(model, "e2e_head_mode"):
        raise ValueError("Adapter already installed")
    model.e2e_head_mode = mode
    if mode != "shared":
        # No initialization, hence no changes to CPU/CUDA RNG streams.
        model.e2e_correction_head = copy.deepcopy(model.dev_head)
        model.e2e_correction_head.register_parameter("bias", None)
        model.register_forward_hook(_readout_hook)
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    if state[3]:
        torch.cuda.set_rng_state_all(state[3])


def digest_state(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("e2e_correction_head."):
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def snapshot(model, identity, epoch, val_auc=None):
    return dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                identity=identity, epoch=epoch, val_auc=val_auc,
                sample_groups=copy.deepcopy(model.prototype_layer.sample_groups),
                group_labels=copy.deepcopy(model.prototype_layer.group_labels),
                target_labels=copy.deepcopy(model.prototype_layer.target_labels))


def head_stats(model):
    wc = float(model.dev_head.weight.detach().norm())
    wd = float(correction_weight(model).detach().norm())
    beta = float(model.effective_beta().detach().mean())
    ratio = wd / wc if wc > 0 else None
    return dict(beta=beta, wc_norm=wc, wd_norm=wd, ratio=ratio,
                beta_ratio=beta*ratio if ratio is not None else None,
                wd_raw_norm=float(model.e2e_correction_head.weight.detach().norm())
                if model.e2e_head_mode != "shared" else wc,
                gamma=model.effective_gamma())


def evaluate(model, dataset):
    model.eval()
    result = {}
    with torch.inference_mode():
        for name, (x, y) in zip(("train", "val", "test"), dataset._indv_dataset()):
            zs, rs = [], []
            for xx in x.split(512):
                out = model(xx)
                g = model.effective_gamma()
                expected = model.dev_head(g*out["context_emb"]) + F.linear(g*out["correction"], correction_weight(model))
                if not torch.allclose(out["logits"], expected, atol=2e-5, rtol=2e-5):
                    raise ValueError("Prediction decomposition failed")
                zs.append(out["logits"].flatten().cpu().numpy())
                rs.append(out["centroid_id"].cpu().numpy())
            z, r = np.concatenate(zs), np.concatenate(rs)
            result[name] = dict(metrics(y.cpu().numpy().reshape(-1), z, r),
                                occupied_regions=int(len(np.unique(r))), logits=z, region=r)
    return result


def run_one(args, data, seed, mode, reference):
    from libs.benchmark import build_wrapper, data_signature, implementation_id, training_diagnostics
    from libs.data import TabularDataset
    import libs.supervised as supervised
    source = result_file(args.root, data, seed)
    saved = np.load(source, allow_pickle=True).item()
    old = saved["identity"]
    config = old["contract"]["config"]
    if old["dataset_id"] != data or old["fold"] != seed or old["tasktype"] != "binclass":
        raise ValueError("Source identity mismatch")
    if config["early_stop_metric"] != "val_loss" or config["correction_geometry"] != "unit_tangent":
        raise ValueError("Expected latest terminal/Unit Tangent source")
    identity = dict(data=data, seed=seed, mode=mode, source_sha256=sha256(source),
                    source_identity=old, implementation=implementation_id(),
                    adapter_sha256=sha256(Path(__file__)),
                    actual_environment={p: importlib.metadata.version(p) for p in
                                        ("torch", "numpy", "scikit-learn", "optuna")},
                    norm_rule="stopgrad(norm(Wc))*Wd_raw/norm(Wd_raw)" if mode == "match" else mode,
                    primary_selection="val_loss patience, terminal checkpoint",
                    secondary_selection="best val AUROC, first strict maximum, saved only",
                    optimizer="unchanged AdamW lr/wd including the added correction weight")
    stem = args.output / f"data={data}_seed={seed}_head={mode}"
    path = Path(str(stem)+".json")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError("Existing run differs; use another output directory")
        reference.setdefault((data, seed), record["initial"]["base_state_sha256"])
        return record
    dataset = TabularDataset(data, "binclass", device=args.device, seed=seed)
    if data_signature(dataset) != old["contract"]["data"]:
        raise ValueError("Data signature mismatch")
    seed_all(old["train_seed"])
    wrapper = build_wrapper(dataset, old["params"], config, args.device)
    wrapper._data_id = data
    wrapper.epochs = old["contract"]["schedule"]["epochs"]
    wrapper.patience = old["contract"]["schedule"]["patience"]
    model = wrapper.model
    initial_hash = digest_state(model)
    if (data, seed) in reference and reference[data, seed] != initial_hash:
        raise ValueError("Base initial state differs between heads")
    reference[data, seed] = initial_hash
    state = rng_state()
    was_training = model.training
    model.eval()
    xt, yt = dataset._indv_dataset()[0]
    with torch.no_grad():
        base_logits = model(xt)["logits"].clone()
        attach_head(model, mode)
        split_logits = model(xt)["logits"]
        error = float((base_logits-split_logits).abs().max())
    if error >= 1e-6:
        raise ValueError(f"Initial predictor parity failed: {error}")
    if digest_state(model) != initial_hash:
        raise ValueError("Parity check changed the base model")
    restore_rng(state)
    # fit() initializes centroids BEFORE its first model.train() call. Restore
    # training mode so its embedder/dropout and RNG consumption match baseline.
    model.train(was_training)
    initial = dict(base_state_sha256=initial_hash, max_abs_logit_diff=error, **head_stats(model))
    best = dict(auc=-float("inf"), epoch=None, snapshot=None)
    epoch_stats = []
    original_metric = supervised.compute_metric
    def observe(logits, labels, tasktype, *a, **kw):
        values = original_metric(logits, labels, tasktype, *a, **kw)
        epoch = len(epoch_stats)+1
        auc = values.get("auroc_val")
        epoch_stats.append(dict(epoch=epoch, val_metrics=dict(values), **head_stats(model)))
        if auc is not None and np.isfinite(auc) and auc > best["auc"]:
            best.update(auc=float(auc), epoch=epoch,
                        snapshot=snapshot(model, identity, epoch, float(auc)))
        return values
    start = time.perf_counter()
    (xt, yt), (xv, yv), _ = dataset._indv_dataset()
    # Observation only; no changes to metrics returned, RNG, patience or state.
    with patch.object(supervised, "compute_metric", side_effect=observe):
        wrapper.fit(xt, yt, xv, yv)
    if not wrapper.terminal_checkpoint or wrapper.best_epoch != wrapper.last_epoch:
        raise ValueError("Primary checkpoint protocol changed")
    if len(epoch_stats) != wrapper.last_epoch:
        raise ValueError("Validation observer epoch mismatch")
    if best["snapshot"] is not None:
        torch.save(best["snapshot"], str(stem)+"_best_val_auc.pt")
    torch.save(snapshot(model, identity, wrapper.last_epoch), str(stem)+"_terminal.pt")
    record = dict(identity=identity, initial=initial, training_seconds=time.perf_counter()-start,
                  terminal_epoch=wrapper.last_epoch, best_val_loss_epoch=wrapper.best_metric_epoch,
                  best_val_auc_epoch=best["epoch"], best_val_auc=best["auc"],
                  final_head=head_stats(model), training=training_diagnostics(wrapper),
                  epoch_head_stats=epoch_stats, performance=evaluate(model, dataset))
    write_json(path, record)
    return record


def summarize(output, records):
    rows = []
    for r in records:
        i = r["identity"]
        row = dict(data=i["data"], seed=i["seed"], head=i["mode"],
                   terminal_epoch=r["terminal_epoch"], best_val_auc_epoch=r["best_val_auc_epoch"],
                   best_val_loss_epoch=r["best_val_loss_epoch"], **r["final_head"])
        for s in ("train", "val", "test"):
            row.update({f"{s}_{k}":v for k,v in r["performance"][s].items() if k not in ("logits", "region")})
        row["dead_ratio_final"] = 1-r["training"]["active_ratio_final"]
        row["reinit_total"] = r["training"]["reinit_total"]
        rows.append(row)
    with (output / "per_run.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lookup = {(r["data"],r["seed"],r["head"]):r for r in rows}
    lines = ["# Fixed-HP end-to-end head comparison", "",
             "Same selected HPs retrained per arm. Primary selection: validation loss patience, terminal model.",
             "Secondary best-validation-AUROC weights saved only; no secondary test evaluation.", "",
             "| Data | Seed | Shared AUC | Match AUC | Free AUC | Match gain | Free gain |", "|---|---:|---:|---:|---:|---:|---:|"]
    paired = []
    for data, seed in sorted({(r["data"],r["seed"]) for r in rows}):
        if any((data,seed,h) not in lookup for h in MODES):
            continue
        a,b,c = [lookup[data,seed,h] for h in MODES]
        lines.append(f"| {data} | {seed} | {a['test_auc']:.4f} | {b['test_auc']:.4f} | {c['test_auc']:.4f} | {b['test_auc']-a['test_auc']:+.4f} | {c['test_auc']-a['test_auc']:+.4f} |")
        for head, alt in (("match",b),("free",c)):
            pair = dict(data=data,seed=seed,head=head)
            for key in ("auc","acc","f1","logloss","same_auc","cross_auc"):
                v,w = alt[f"test_{key}"], a[f"test_{key}"]
                pair[key] = v-w if v is not None and w is not None else None
            paired.append(pair)
    aggregates=[]
    for head in ("match","free"):
        lines += ["",f"## {head}: paired differences from shared", "",
                  "| Data | N | AUC gain | ACC gain | F1 gain | Same AUC gain | Cross AUC gain | AUC W/T/L |",
                  "|---|---:|---:|---:|---:|---:|---:|---|"]
        for data in sorted({r["data"] for r in paired}):
            group=[r for r in paired if r["data"]==data and r["head"]==head]
            if not group: continue
            means={k:float(np.mean([r[k] for r in group if r[k] is not None]))
                   if any(r[k] is not None for r in group) else None
                   for k in ("auc","acc","f1","same_auc","cross_auc")}
            agg=dict(data=data,head=head,n=len(group),**means)
            aggregates.append(agg)
            vals=["NA" if means[k] is None else f"{means[k]:+.4f}" for k in means]
            gains=[r["auc"] for r in group]
            lines.append(f"| {data} | {len(group)} | "+" | ".join(vals)+f" | {sum(x>0 for x in gains)}/{sum(x==0 for x in gains)}/{sum(x<0 for x in gains)} |")
        group=[r for r in aggregates if r["head"]==head]
        if group:
            lines += ["",f"Dataset-equal mean AUC gain: {np.mean([r['auc'] for r in group]):+.4f}; "
                      f"ACC: {np.mean([r['acc'] for r in group]):+.4f}; F1: {np.mean([r['f1'] for r in group]):+.4f}."]
    write_json(output/"paired_differences.json",dict(pairs=paired,dataset_means=aggregates))
    (output/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",nargs="+",type=int,default=[51,1067,31])
    p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3])
    p.add_argument("--root",type=Path,default=Path("."))
    p.add_argument("--output",type=Path,default=Path("diagnostics/e2e_split_auc"))
    p.add_argument("--device",default="cuda",choices=["cpu","cuda"])
    args=p.parse_args()
    os.environ.setdefault("OPENML_CACHE_DIR",str(args.root.resolve()/"data_cache/openml"))
    args.device="cuda:0" if args.device=="cuda" and torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"protocol.json",dict(datasets=args.datasets,seeds=args.seeds,heads=MODES,
               source="latest result HP/fold/train seed",initialization="Wc=Wd=W0, one bias, no extra RNG",
               primary="val_loss/terminal",secondary="best val AUROC saved only",
               gamma="both branches",norm="existing match_w1 detached Wc radius",
               gate=dict(min_improved_datasets=2,min_mean_auc_gain=.005,min_mean_acc_gain=-.01,min_mean_f1_gain=-.01),
               source_code_sha256=sha256(Path(__file__))))
    reference={}; records=[]
    for data in args.datasets:
        for seed in args.seeds:
            for mode in MODES:
                print(f"[run] data={data} seed={seed} head={mode}",flush=True)
                record=run_one(args,data,seed,mode,reference)
                records.append(record)
                summarize(args.output,records)
                print(f"[done] data={data} seed={seed} head={mode} auc={record['performance']['test']['auc']:.4f}",flush=True)


if __name__=="__main__":
    main()
