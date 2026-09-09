"""Joint read-out of dynamics2d pilot studies (validation-only). Read-only."""
import ast, glob, sys, json
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
info = json.load(open("dataset_id.json", encoding="utf-8"))
rows = []
for f in sorted(glob.glob("pilot_dynamics2d/optim_logs/seed=1/data=*pilot=dynamics2d*.csv")):
    ds = int(f.split("data=")[1].split("..")[0])
    df = pd.read_csv(f); df = df[df.state == "COMPLETE"].copy()
    df["ds"] = ds; df["task"] = info[str(ds)]["tasktype"]; df["name"] = info[str(ds)]["name"]
    pdv = df["user_attrs_prediction_diagnostics_val"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else {})
    df["class_change"] = pdv.apply(lambda d: d.get("region_to_final_class_change_rate"))
    df["corr_margin"] = pdv.apply(lambda d: d.get("correction_margin_mean"))
    bh = df["user_attrs_beta_epoch_history"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else [])
    df["beta_0"] = bh.apply(lambda h: h[0]["beta"] if h else np.nan)
    # still moving in one direction at the end: last-5-epoch slope sign == overall sign and |slope| > 1e-3/epoch
    def tail_slope(h):
        if len(h) < 6: return np.nan
        b = [e["beta"] for e in h[-5:]]; return (b[-1] - b[0]) / 4
    df["beta_tail_slope"] = bh.apply(tail_slope)
    rows.append(df)
all_ = pd.concat(rows, ignore_index=True)
higher = all_.task != "regression"
all_["rel"] = all_.groupby("ds")["value"].transform(lambda v: v - v[all_.loc[v.index, "number"] == 0].iloc[0])
all_["rank"] = all_.groupby("ds")["value"].rank(ascending=False, method="min")
c = dict(v="value", m="params_beta_lr_mult", e="params_ema_timescale", bf="user_attrs_beta_final",
         r="user_attrs_reinit_per_epoch", a="user_attrs_active_ratio_std", ch="user_attrs_routing_churn_mean",
         le="user_attrs_last_epoch", be="user_attrs_best_metric_epoch", ne="user_attrs_n_eff_entropy",
         st="user_attrs_steps_per_epoch")
print("=== per dataset: anchor vs best, top-5 composition ===")
for ds, g in all_.groupby("ds"):
    g = g.sort_values("value", ascending=False)
    a = g[g.number == 0].iloc[0]; b = g.iloc[0]; top = g.head(5)
    print(f"ds={ds:<5} {a['name']:<4} {a['task']:<10} steps/ep={int(a[c['st']]):>3} P={int(a['user_attrs_n_prototypes_actual']):>3} | "
          f"anchor={a['value']:.4f} (rank {int(a['rank'])}/25)  best={b['value']:.4f} mult={b[c['m']]:.2f} ema={b[c['e']]} | "
          f"top5 mult=[{', '.join(f'{x:.1f}' for x in top[c['m']])}]  ema={dict(top[c['e']].value_counts())}")
print("\n=== per dataset x EMA: mean val (rel. to anchor), reinit/epoch, active_std, churn, last_epoch  (n) ===")
t = all_.groupby(["ds", c["e"]]).agg(n=("value", "size"), rel=("rel", "mean"), reinit=(c["r"], "mean"),
                                       act_std=(c["a"], "mean"), churn=(c["ch"], "mean"), last_ep=(c["le"], "mean")).round(3)
print(t.unstack(0).to_string())
print("\n=== pooled by EMA (mean rel-to-anchor value, mean rank) ===")
print(all_.groupby(c["e"]).agg(n=("value", "size"), rel=("rel", "mean"), rank=("rank", "mean"), reinit=(c["r"], "mean"), act_std=(c["a"], "mean")).round(3).to_string())
print("\n=== pooled by beta_lr_mult bin ===")
all_["mbin"] = pd.cut(all_[c["m"]], [0.99, 2, 5, 12, 30.01], labels=["1-2", "2-5", "5-12", "12-30"])
print(all_.groupby("mbin", observed=True).agg(n=("value", "size"), rel=("rel", "mean"), rank=("rank", "mean"),
      beta_final=(c["bf"], "mean"), tail_slope=("beta_tail_slope", "mean"), class_change=("class_change", "mean"),
      corr_margin=("corr_margin", "mean"), last_ep=(c["le"], "mean")).round(4).to_string())
print("\n=== per dataset: spearman(mult, value), beta_final range, trials with beta still rising (tail slope>1e-3) ===")
from scipy.stats import spearmanr
for ds, g in all_.groupby("ds"):
    rho, p = spearmanr(g[c["m"]], g["value"])
    rising = (g["beta_tail_slope"] > 1e-3).mean()
    print(f"ds={ds:<5} rho={rho:+.2f} (p={p:.2f})  beta_final {g[c['bf']].min():.3f}..{g[c['bf']].max():.3f}  "
          f"rising_at_end={rising:.0%}  mult>=20: {(g[c['m']]>=20).sum()} trials, their ranks {sorted(g[g[c['m']]>=20]['rank'].astype(int).tolist())}")
print("\n=== top-5 per dataset (full) ===")
cols = ["ds", "number", "value", "rel", c["m"], c["e"], c["bf"], c["r"], c["a"], c["ch"], "class_change", "corr_margin", c["be"], c["le"]]
print(all_.sort_values(["ds", "value"], ascending=[True, False]).groupby("ds").head(5)[cols].round(4).to_string(index=False))
