"""Audit completed E2E runs and compare scores on shared-reference pair sets."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze_split_head_auc import pair_auc, write_json


def audit(root):
    runs=[json.loads(p.read_text(encoding="utf-8")) for p in root.glob("data=*_seed=*_head=*.json")]
    lookup={(r['identity']['data'],r['identity']['seed'],r['identity']['mode']):r for r in runs}
    expected={(d,s,h) for d in (51,1067,31) for s in (1,2,3) for h in ('shared','match','free')}
    assert set(lookup)==expected, (len(lookup),expected-set(lookup))
    shared_rows=[]
    max_parity=0.; max_norm_error=0.
    for data,seed in sorted({(d,s) for d,s,h in expected}):
        group=[lookup[data,seed,h] for h in ('shared','match','free')]
        assert len({r['initial']['base_state_sha256'] for r in group})==1
        assert all(r['identity']['source_identity']==group[0]['identity']['source_identity'] for r in group)
        for r in group:
            mode=r['identity']['mode']
            max_parity=max(max_parity,r['initial']['max_abs_logit_diff'])
            assert r['initial']['max_abs_logit_diff']<1e-6
            hist=r['epoch_head_stats']
            assert len(hist)==r['terminal_epoch']==r['training']['last_epoch']
            eligible=[v for v in hist if v['val_metrics'].get('auroc_val') is not None]
            best=max(eligible,key=lambda v:v['val_metrics']['auroc_val'])
            assert best['epoch']==r['best_val_auc_epoch']
            stem=root/f'data={data}_seed={seed}_head={mode}'
            assert Path(str(stem)+'_terminal.pt').is_file()
            assert Path(str(stem)+'_best_val_auc.pt').is_file()
            if mode=='match':
                for v in hist:
                    err=abs(v['ratio']-1.)
                    max_norm_error=max(max_norm_error,err)
                    assert err<2e-6, (data,seed,err)
        # Labels are recovered from the same data fold, not inferred from scores.
        import os
        os.environ.setdefault('OPENML_CACHE_DIR',str(Path('data_cache/openml').resolve()))
        from libs.data import TabularDataset
        dataset=TabularDataset(data,'binclass',device='cpu',seed=seed)
        y=dataset._indv_dataset()[2][1].numpy().reshape(-1)
        shared_region=np.asarray(group[0]['performance']['test']['region'])
        for r in group:
            score=np.asarray(r['performance']['test']['logits'])
            shared_rows.append(dict(data=data,seed=seed,head=r['identity']['mode'],
                                    **pair_auc(y,score,shared_region)))
    with (root/'shared_partition_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(shared_rows[0]));w.writeheader();w.writerows(shared_rows)
    paired=json.loads((root/'paired_differences.json').read_text(encoding='utf-8'))
    gates={}
    for mode in ('match','free'):
        means=[r for r in paired['dataset_means'] if r['head']==mode]
        gains=[r['auc'] for r in paired['pairs'] if r['head']==mode]
        gain=float(np.mean([r['auc'] for r in means])); acc=float(np.mean([r['acc'] for r in means]));f1=float(np.mean([r['f1'] for r in means]))
        without_best=float((sum(gains)-max(gains))/(len(gains)-1))
        f1_1067=next(r['f1'] for r in means if r['data']==1067)
        gates[mode]=dict(mean_auc_gain=gain,mean_acc_gain=acc,mean_f1_gain=f1,
                         positive_datasets=sum(r['auc']>0 for r in means),
                         mean_auc_without_best_fold=without_best,f1_gain_1067=f1_1067,
                         pass_auc=gain>=.005,pass_datasets=sum(r['auc']>0 for r in means)>=2,
                         pass_acc=acc>=-.01,pass_f1=f1>=-.01,
                         pass_best_fold_sensitivity=without_best>0,
                         pass_f1_1067=f1_1067>=-.01)
        gates[mode]['all_gates']=all(v for k,v in gates[mode].items() if k.startswith('pass_'))
    summary=dict(runs=len(runs),max_initial_logit_error=max_parity,
                 max_epoch_matched_norm_ratio_error=max_norm_error,gates=gates)
    write_json(root/'audit.json',summary)
    print(json.dumps(summary,indent=2))
    for d in (51,1067,31):
        for h in ('match','free'):
            diffs={k:[] for k in ('same_auc','cross_auc')}
            for s in (1,2,3):
                a=next(r for r in shared_rows if (r['data'],r['seed'],r['head'])==(d,s,'shared'))
                b=next(r for r in shared_rows if (r['data'],r['seed'],r['head'])==(d,s,h))
                for k in diffs:
                    if a[k] is not None and b[k] is not None: diffs[k].append(b[k]-a[k])
            print('shared-reference pair gains',d,h,{k:float(np.mean(v)) if v else None for k,v in diffs.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('diagnostics/e2e_split_auc_v2'))
    audit(p.parse_args().output)
