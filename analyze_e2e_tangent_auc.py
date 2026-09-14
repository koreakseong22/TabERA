"""Fixed-HP Unit Tangent versus Tangent end-to-end experiment.

The base model is constructed as the latest Unit Tangent model. For the tangent
arm only ``correction_geometry`` is changed after construction, preserving the
Unit arm's fixed gamma, every initialized parameter and all RNG state.

Example:
  python analyze_e2e_tangent_auc.py --datasets 51 1067 31 --seeds 1 2 3
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
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analyze_split_head_auc import metrics, pair_auc, result_file, sha256, write_json
from analyze_e2e_split_head_auc import seed_all, rng_state, restore_rng, digest_state

GEOMETRIES = ("unit_tangent", "tangent")


def select_geometry(model, geometry):
    """Select geometry without reconstructing gamma or consuming RNG."""
    if geometry not in GEOMETRIES:
        raise ValueError(geometry)
    if model.split_head or model.dev_head.out_features != 1:
        raise ValueError("This screening requires the binary shared head")
    if model.correction_geometry != "unit_tangent" or model.head_input_scale != "auto":
        raise ValueError("Base must be the final Unit Tangent + auto-gamma model")
    model.correction_geometry = geometry
    model.geometry_experiment_gamma_source = "unit_tangent_auto"
    return model


def geometry_stats(model, x, y):
    model.eval()
    zs, regions, ss, ds = [], [], [], []
    with torch.inference_mode():
        for xx in x.split(512):
            out = model(xx)
            s = out["geo"].get("s")
            if s is None:
                raise ValueError("Tangent geometry did not emit s")
            d = out["correction_mag"]
            expected = (model.effective_beta() * s if model.correction_geometry == "tangent"
                        else model.effective_beta() * torch.minimum(
                            s / model.correction_eps, torch.ones_like(s)))
            if not torch.allclose(d, expected, atol=2e-5, rtol=2e-5):
                raise ValueError("Correction magnitude does not match implemented geometry")
            z_expected = model.dev_head(model.effective_gamma() *
                                        (out["context_emb"] + out["correction"]))
            if not torch.allclose(out["logits"], z_expected, atol=2e-5, rtol=2e-5):
                raise ValueError("Shared-head decomposition failed")
            zs.append(out["logits"].flatten().cpu().numpy())
            regions.append(out["centroid_id"].cpu().numpy())
            ss.append(s.cpu().numpy())
            ds.append(d.cpu().numpy())
    z, region, s, d = map(np.concatenate, (zs, regions, ss, ds))
    base = metrics(y.cpu().numpy().reshape(-1), z, region)
    base.update(logits=z, region=region, s_mean=float(s.mean()), s_median=float(np.median(s)),
                s_q10=float(np.quantile(s,.1)), s_q90=float(np.quantile(s,.9)),
                d_mean=float(d.mean()), d_median=float(np.median(d)),
                d_q10=float(np.quantile(d,.1)), d_q90=float(np.quantile(d,.9)),
                occupied_regions=int(len(np.unique(region))),
                correction_cv=float(d.std()/d.mean()) if d.mean()>0 else None)
    return base


def partition_agreement(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = int(a.max())+1, int(b.max())+1
    table = np.zeros((na,nb),dtype=np.int64)
    np.add.at(table,(a,b),1)
    row,col=linear_sum_assignment(-table)
    return dict(ari=float(adjusted_rand_score(a,b)),
                nmi=float(normalized_mutual_info_score(a,b)),
                hungarian_agreement=float(table[row,col].sum()/len(a)),
                unit_occupied=int(len(np.unique(a))),tangent_occupied=int(len(np.unique(b))))


def run_one(args, data, seed, geometry, reference):
    from libs.benchmark import build_wrapper, data_signature, implementation_id, training_diagnostics
    from libs.data import TabularDataset
    source=result_file(args.root,data,seed)
    saved=np.load(source,allow_pickle=True).item(); old=saved["identity"]; config=old["contract"]["config"]
    if old["tasktype"]!="binclass" or old["dataset_id"]!=data or old["fold"]!=seed:
        raise ValueError("Source identity mismatch")
    if config["correction_geometry"]!="unit_tangent" or config["head_input_scale"]!="auto" or config["early_stop_metric"]!="val_loss":
        raise ValueError("Expected final Unit Tangent/auto/val-loss source")
    identity=dict(data=data,seed=seed,geometry=geometry,source_sha256=sha256(source),source_identity=old,
                  implementation=implementation_id(),runner_sha256=sha256(Path(__file__)),
                  actual_environment={p:importlib.metadata.version(p) for p in ("torch","numpy","scikit-learn","optuna")},
                  intervention="correction_geometry only; gamma retained from unit_tangent auto",
                  selection="val_loss patience; terminal checkpoint",hpo="none; source Unit Tangent best HP")
    stem=args.output/f"data={data}_seed={seed}_geometry={geometry}"; path=Path(str(stem)+".json")
    if path.exists():
        record=json.loads(path.read_text(encoding="utf-8"))
        if record["identity"]!=identity: raise ValueError("Existing run identity differs")
        reference.setdefault((data,seed),record["initial"]["base_state_sha256"])
        return record
    dataset=TabularDataset(data,"binclass",device=args.device,seed=seed)
    if data_signature(dataset)!=old["contract"]["data"]: raise ValueError("Data signature mismatch")
    seed_all(old["train_seed"])
    wrapper=build_wrapper(dataset,old["params"],config,args.device)
    wrapper._data_id=data; wrapper.epochs=old["contract"]["schedule"]["epochs"];wrapper.patience=old["contract"]["schedule"]["patience"]
    model=wrapper.model; initial_hash=digest_state(model)
    if (data,seed) in reference and reference[data,seed]!=initial_hash: raise ValueError("Initial parameter state differs between geometries")
    reference[data,seed]=initial_hash
    state=rng_state(); was_training=model.training
    select_geometry(model,geometry)
    # Eval-mode observation is fully rolled back before fit.
    xt,yt=dataset._indv_dataset()[0]
    initial_eval=geometry_stats(model,xt,yt)
    if digest_state(model)!=initial_hash: raise ValueError("Initial diagnostic changed model state")
    restore_rng(state);model.train(was_training)
    initial=dict(base_state_sha256=initial_hash,gamma=model.effective_gamma(),
                 beta=float(model.effective_beta().detach()),s_mean=initial_eval["s_mean"],s_median=initial_eval["s_median"],
                 d_mean=initial_eval["d_mean"],d_median=initial_eval["d_median"])
    start=time.perf_counter();(xt,yt),(xv,yv),_=dataset._indv_dataset();wrapper.fit(xt,yt,xv,yv)
    if not wrapper.terminal_checkpoint or wrapper.best_epoch!=wrapper.last_epoch: raise ValueError("Primary checkpoint protocol changed")
    performance={}
    for name,(x,y) in zip(("train","val","test"),dataset._indv_dataset()): performance[name]=geometry_stats(model,x,y)
    history=wrapper.regroup_history
    aucs=[(int(r["epoch"]),r.get("val_auroc_val")) for r in history if r.get("val_auroc_val") is not None]
    best_auc=max(aucs,key=lambda z:z[1]) if aucs else (None,None)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},identity=identity,
                    epoch=wrapper.last_epoch),str(stem)+"_terminal.pt")
    record=dict(identity=identity,initial=initial,training_seconds=time.perf_counter()-start,
                terminal_epoch=wrapper.last_epoch,best_val_loss_epoch=wrapper.best_metric_epoch,
                best_observed_val_auc_epoch=best_auc[0],best_observed_val_auc=best_auc[1],
                final=dict(beta=float(model.effective_beta().detach()),gamma=model.effective_gamma()),
                training=training_diagnostics(wrapper),performance=performance)
    write_json(path,record);return record


def summarize(output,records):
    rows=[]
    for r in records:
        i=r["identity"];row=dict(data=i["data"],seed=i["seed"],geometry=i["geometry"],
            terminal_epoch=r["terminal_epoch"],best_val_loss_epoch=r["best_val_loss_epoch"],
            best_observed_val_auc_epoch=r["best_observed_val_auc_epoch"],**r["final"])
        for split in ("train","val","test"):
            row.update({f"{split}_{k}":v for k,v in r["performance"][split].items() if k not in ("logits","region")})
        row["dead_ratio_final"]=1-r["training"]["active_ratio_final"];row["reinit_total"]=r["training"]["reinit_total"]
        rows.append(row)
    with (output/"per_run.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lookup={(r["data"],r["seed"],r["geometry"]):r for r in rows}; pairs=[]; agreements=[]
    lines=["# Unit Tangent vs Tangent, fixed HP E2E","",
           "Shared head; same Unit-selected HPs and gamma; val-loss patience; terminal checkpoint.","",
           "| Data | Seed | Unit AUC | Tangent AUC | AUC gain | ACC gain | F1 gain | LogLoss gain |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for data,seed in sorted({(r["data"],r["seed"]) for r in rows}):
        if any((data,seed,g) not in lookup for g in GEOMETRIES):continue
        u,t=lookup[data,seed,"unit_tangent"],lookup[data,seed,"tangent"]
        p=dict(data=data,seed=seed)
        for k in ("auc","acc","f1","logloss","same_auc","cross_auc","s_mean","s_median","d_mean","d_median"):
            a,b=u[f"test_{k}"],t[f"test_{k}"];p[k]=b-a if a is not None and b is not None else None
        pairs.append(p)
        lines.append(f"| {data} | {seed} | {u['test_auc']:.4f} | {t['test_auc']:.4f} | {p['auc']:+.4f} | {p['acc']:+.4f} | {p['f1']:+.4f} | {p['logloss']:+.4f} |")
        ur=np.asarray(next(r for r in records if r['identity']['data']==data and r['identity']['seed']==seed and r['identity']['geometry']=='unit_tangent')['performance']['test']['region'])
        tr=np.asarray(next(r for r in records if r['identity']['data']==data and r['identity']['seed']==seed and r['identity']['geometry']=='tangent')['performance']['test']['region'])
        agreements.append(dict(data=data,seed=seed,**partition_agreement(ur,tr)))
    means=[];lines += ["","## Dataset means","","| Data | AUC gain | ACC gain | F1 gain | LogLoss change | AUC W/T/L |","|---|---:|---:|---:|---:|---|"]
    for data in sorted({p['data'] for p in pairs}):
        g=[p for p in pairs if p['data']==data];m={k:float(np.mean([p[k] for p in g if p[k] is not None])) for k in ("auc","acc","f1","logloss")};means.append(dict(data=data,n=len(g),**m));a=[p['auc'] for p in g]
        lines.append(f"| {data} | {m['auc']:+.4f} | {m['acc']:+.4f} | {m['f1']:+.4f} | {m['logloss']:+.4f} | {sum(x>0 for x in a)}/{sum(x==0 for x in a)}/{sum(x<0 for x in a)} |")
    if means:
        macro={k:float(np.mean([m[k] for m in means])) for k in ("auc","acc","f1","logloss")};g=[p['auc'] for p in pairs];without=((sum(g)-max(g))/(len(g)-1) if len(g)>1 else g[0])
        gate=dict(mean=macro,positive_datasets=sum(m['auc']>0 for m in means),wins=sum(x>0 for x in g),ties=sum(x==0 for x in g),losses=sum(x<0 for x in g),mean_auc_without_best_fold=float(without),pass_auc=macro['auc']>=.005,pass_datasets=sum(m['auc']>0 for m in means)>=2,pass_acc=macro['acc']>=-.01,pass_f1=macro['f1']>=-.01,pass_best_fold_sensitivity=without>0)
        gate['all']=all(v for k,v in gate.items() if k.startswith('pass_'));lines += ["",f"Dataset-equal mean: AUC {macro['auc']:+.4f}, ACC {macro['acc']:+.4f}, F1 {macro['f1']:+.4f}, LogLoss {macro['logloss']:+.4f}.",f"Gate: {'PASS' if gate['all'] else 'FAIL'}."]
    else:gate={}
    write_json(output/"paired_differences.json",dict(pairs=pairs,dataset_means=means,partition_agreement=agreements,gate=gate))
    (output/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--datasets",nargs="+",type=int,default=[51,1067,31]);p.add_argument("--seeds",nargs="+",type=int,default=[1,2,3]);p.add_argument("--geometries",nargs="+",choices=GEOMETRIES,default=list(GEOMETRIES));p.add_argument("--root",type=Path,default=Path("."));p.add_argument("--output",type=Path,default=Path("diagnostics/e2e_tangent_auc"));p.add_argument("--device",choices=["cpu","cuda"],default="cuda");args=p.parse_args()
    if set(args.geometries)!=set(GEOMETRIES):raise ValueError("Controlled comparison requires both geometries")
    os.environ.setdefault("OPENML_CACHE_DIR",str(args.root.resolve()/"data_cache/openml"));args.device="cuda:0" if args.device=="cuda" and torch.cuda.is_available() else "cpu";torch.set_num_threads(1);args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"protocol.json",dict(datasets=args.datasets,seeds=args.seeds,geometries=args.geometries,head="shared",gamma="Unit Tangent auto gamma fixed in both arms",hpo="none; Unit-selected HP",selection="val_loss patience; terminal",gate=dict(min_positive_datasets=2,min_auc=.005,min_acc=-.01,min_f1=-.01),runner_sha256=sha256(Path(__file__))))
    reference={};records=[]
    for data in args.datasets:
        for seed in args.seeds:
            for geometry in args.geometries:
                print(f"[run] data={data} seed={seed} geometry={geometry}",flush=True);r=run_one(args,data,seed,geometry,reference);records.append(r);summarize(args.output,records);print(f"[done] test_auc={r['performance']['test']['auc']:.4f}",flush=True)


if __name__=="__main__":main()
