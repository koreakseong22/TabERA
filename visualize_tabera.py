# -*- coding: utf-8 -*-
"""
visualize_tabera.py
===================
Per-dataset diagnostic figures. These are not the architecture diagram in the
paper — they are drawn from a trained checkpoint and describe what one run
produced, one set per dataset and seed.

    x -> Encoder -> q -> prototype assignment -> c
                            |- prediction   z = W*c + beta*W*(q-c)
                            '- retrieval    NN(q, G_p) -> evidence

  1  Prototype partition       how the space is divided
  2  Prototype profiles        what each region is
  3  Prediction decomposition  how prototype and query combine
  4  Retrieval evidence        what accompanies the decision
  5  Prototype geometry        how the prototypes sit relative to each other

Panel 3 is the central one: dev_head is a single Linear and W is shared, so the
two terms add to the logits exactly rather than approximating them.

Panel 4 deliberately keeps retrieval off the prediction path. Six post-hoc
injections, a jointly trained decoder and a probe upper bound all found no
consistent predictive gain, so retrieval is drawn as an evidence branch. Its
right-hand panel asks whether the evidence agrees with the decision, not
whether it produced it.

Usage
-----
python visualize_tabera.py --openml_id 54 --seed 1
python visualize_tabera.py --state path/to/..._model_state.pt
"""

import argparse
import glob
import os
import sys
import types
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
sys.modules.setdefault("openml", types.ModuleType("openml"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG = "#ffffff"
QC = "#c0392b"
MARK = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

# One place for the type scale. The defaults are too small once these are
# dropped into a two-column layout and scaled down.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.linewidth": .4,
    "grid.alpha": .35,
})
TITLE = 12.5     # panel titles
LABEL = 11       # axis labels
TICK = 10        # tick / annotation text
NOTE = 9.5       # caveats


# ─────────────────────────────────────────────────────────────
def load(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st["state_dict"]
    n = st["n_train"]
    C = sd["prototype_layer.centroid_emb"]
    K = sd["memory.keys"][:n]
    y = sd["memory.labels"][:n].numpy().astype(int).ravel()
    if "dev_head.weight" not in sd:
        raise SystemExit(
            f"  this checkpoint has no dev_head; fusion_mode="
            f"{st['model_kwargs'].get('fusion_mode')!r}.\n"
            f"  These figures decompose z = W*c + beta*W*(q-c), which only\n"
            f"  exists under proto_dev. Point --state at a proto_dev run.")
    W, b = sd["dev_head.weight"], sd["dev_head.bias"]
    beta = float(torch.sigmoid(sd["dev_beta_raw"]))

    grp = torch.zeros(n, dtype=torch.long)
    for p, g in enumerate(st["sample_groups"]):
        for i in (g or []):
            if 0 <= i < n:
                grp[i] = p
    c = C[grp]
    r = F.normalize(K - c, dim=-1)
    z_proto = c @ W.T + b
    z_query = (beta * r) @ W.T

    fs = st["feature_store_state"]
    return dict(n=n, C=C, K=K, Kn=F.normalize(K, dim=-1), y=y,
                grp=grp.numpy(), beta=beta,
                z_proto=z_proto.numpy(), z_query=z_query.numpy(),
                logits=(z_proto + z_query).numpy(),
                active=[p for p in torch.unique(grp).tolist()
                        if int((grp == p).sum()) >= 2],
                X=fs[0][:n].numpy(),
                cols=(st.get("col_names")
                      or st["model_kwargs"].get("column_names")
                      or [f"f{i}" for i in range(fs[0].shape[1])]),
                names=st.get("target_class_names"),
                k=st["model_kwargs"]["k"])


def labels_of(d):
    ncl = int(d["y"].max()) + 1
    return (d["names"] or [str(i) for i in range(ncl)])[:ncl], ncl


def neighbours(d, i):
    """Top-k inside the assigned prototype, self excluded."""
    m = np.where(d["grp"] == d["grp"][i])[0]
    if len(m) <= d["k"]:
        return m[m != i]
    s = (d["Kn"][m] @ d["Kn"][i]).numpy()
    s[m == i] = -2
    return m[np.argsort(-s)[:d["k"]]]


def profiles(d):
    """One row per prototype: size, purity, entropy, locality, raw contrast."""
    lab, ncl = labels_of(d)
    mu, sdv = d["X"].mean(0), d["X"].std(0) + 1e-8
    rows = []
    for p in sorted(d["active"], key=lambda p: -(d["grp"] == p).sum()):
        m = d["grp"] == p
        cnt = np.bincount(d["y"][m], minlength=ncl).astype(float)
        pr = cnt / cnt.sum()
        nz = pr[pr > 0]
        rows.append(dict(
            p=p, n=int(m.sum()), dist=pr, purity=float(pr.max()),
            H=abs(float(-(nz * np.log(nz)).sum())),
            # Kn, not K: ||q|| runs to the hundreds, so an unnormalised dot
            # product is not a cosine and lands well outside [-1, 1].
            loc=float((d["Kn"][m] @ F.normalize(d["C"][p], dim=-1)).mean()),
            raw=float(np.median(np.abs(d["X"][m].mean(0) - mu) / sdv)),
            top=lab[int(pr.argmax())]))
    return rows


# ─────────────────────────────────────────────────────────────
def fig1(d, out, seed=0):
    """Projection of the prototype partition.

    Colour is prototype, marker is class: colouring by class shows that the
    encoder separates classes but says nothing about how the prototypes cut
    the space. Faint lines mark assignment. No Voronoi cell is drawn --
    assignment is argmax cosine in D dimensions and a 2-D boundary would be a
    claim the projection cannot support."""
    from sklearn.decomposition import PCA
    Z = np.vstack([d["Kn"].numpy(), F.normalize(d["C"], dim=-1).numpy()])
    pca = PCA(2, random_state=seed).fit(Z)
    Q2 = pca.transform(d["Kn"].numpy())
    C2 = pca.transform(F.normalize(d["C"], dim=-1).numpy())
    rows = profiles(d)
    rng = np.random.default_rng(seed)
    cmap = plt.get_cmap("tab20")

    fig = plt.figure(figsize=(15.0, 7.2), facecolor=BG)
    gs = GridSpec(1, 2, width_ratios=[1.55, 1], wspace=.2)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")
    for j, rw in enumerate(rows):
        p, col = rw["p"], cmap(j % 20)
        m = np.where(d["grp"] == p)[0]
        for cl in np.unique(d["y"][m]):
            mm = m[d["y"][m] == cl]
            ax.scatter(Q2[mm, 0], Q2[mm, 1], s=15, alpha=.5, color=col,
                       marker=MARK[cl % len(MARK)], linewidths=0)
        for i in rng.choice(m, size=min(10, len(m)), replace=False):
            ax.plot([Q2[i, 0], C2[p, 0]], [Q2[i, 1], C2[p, 1]],
                    color=col, lw=.3, alpha=.22, zorder=1)
        ax.scatter(C2[p, 0], C2[p, 1], s=80 + 300 * rw["n"] / d["n"],
                   marker="X", color=col, edgecolors="black",
                   linewidths=.6 + 2.0 * rw["purity"], zorder=5)
        ax.annotate(f"P{p}", (C2[p, 0], C2[p, 1]), fontsize=TICK,
                    xytext=(6, 6), textcoords="offset points")
    ev = pca.explained_variance_ratio_
    ax.set_title("Panel 1  Projection of the prototype partition\n"
                 "colour = prototype,  marker = class,  X size = |G_p|,  "
                 "edge width = purity", fontsize=TITLE)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=LABEL)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=LABEL)
    ax.text(.02, .02,
            f"assignment is argmax cos(q, c) over {d['K'].shape[1]} dims; "
            f"PC1+PC2 hold {ev[:2].sum():.0%}",
            transform=ax.transAxes, fontsize=NOTE, style="italic", color="#555")

    axr = fig.add_subplot(gs[1])
    yy = np.arange(len(rows))
    axr.barh(yy, [r["n"] for r in rows],
             color=[cmap(j % 20) for j in range(len(rows))],
             edgecolor="black", linewidth=.4)
    axr.set_yticks(yy)
    axr.set_yticklabels([f"P{r['p']}" for r in rows], fontsize=TICK)
    axr.invert_yaxis()
    axr.set_xscale("log")
    axr.set_xlabel("group size |G_p| (log)", fontsize=LABEL)
    share = sum(r["n"] for r in rows[:2]) / d["n"]
    axr.set_title(f"{len(rows)} active of {d['C'].shape[0]};  "
                  f"top-2 hold {share:.0%}", fontsize=TITLE)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig2(d, out):
    """Prototype profiles. A pie per centroid cannot be read past a handful of
    prototypes and hides n; a row of the quantities that define a region can."""
    rows = profiles(d)
    lab, ncl = labels_of(d)
    fig = plt.figure(figsize=(4.4 + .5 * ncl + 7.0, 2.3 + .42 * len(rows)),
                     facecolor=BG)
    gs = GridSpec(1, 5, width_ratios=[max(ncl, 3), 2.2, 2.2, 2.2, 2.2],
                  wspace=.55)

    ax = fig.add_subplot(gs[0])
    M = np.stack([r["dist"] for r in rows])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(ncl))
    ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=TICK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"P{r['p']}  n={r['n']}" for r in rows], fontsize=TICK)
    ax.set_title("P(y | G_p)", fontsize=TITLE)
    plt.colorbar(im, ax=ax, fraction=.05, pad=.03)

    panels = [("purity", "purity", "#4878CF", (0, 1)),
              ("H", "H(Y | G_p)", "#B47CC7", None),
              ("loc", "mean cos(q, c_p)\nlocality", "#6ACC65", None),
              ("raw", "|Δmean| / σ\nraw contrast", "#D65F5F", None)]
    yy = np.arange(len(rows))
    for gi, (key, title, col, xlim) in enumerate(panels, 1):
        a = fig.add_subplot(gs[gi])
        a.barh(yy, [r[key] for r in rows], color=col,
               edgecolor="black", linewidth=.4)
        a.set_yticks(yy)
        a.set_yticklabels([])
        a.invert_yaxis()
        a.set_title(title, fontsize=LABEL)
        a.tick_params(labelsize=TICK)
        a.grid(axis="x", linestyle="--", lw=.3, alpha=.4)
        if xlim:
            a.set_xlim(*xlim)
        elif key == "loc":
            # These cosines sit between about 0.9 and 1.0; an axis anchored at
            # zero draws every bar full-width and hides the differences.
            lo = min(r[key] for r in rows)
            a.set_xlim(max(0.0, lo - (1 - lo) * .6), 1.0)
        if key == "raw":
            # Below roughly 0.3 the group's raw statistics sit close to the
            # global ones, so "compared with similar cases" does not hold for
            # that group even when it is tight in embedding space.
            a.axvline(.3, color="#333", lw=.8, ls="--")
    n_eff = float(np.exp(-sum((r["n"] / d["n"]) * np.log(r["n"] / d["n"])
                              for r in rows)))
    fig.suptitle(f"Panel 2  Prototype profiles — what each region is"
                 f"   [{len(rows)} active, N_eff {n_eff:.1f}]",
                 fontsize=14, y=1.0)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig3(d, out, n_show=6, seed=0):
    """z = W*c + beta*W*(q-c). One sample per prototype, largest first: within
    a prototype the predictions are nearly identical because W*c is shared and
    the query term is small, so random sampling repeats itself."""
    rng = np.random.default_rng(seed)
    lab, _ = labels_of(d)
    ncl = d["logits"].shape[1]
    binary = ncl == 1
    rows = profiles(d)[:n_show]
    idx = [int(rng.choice(np.where(d["grp"] == r["p"])[0])) for r in rows]

    # Share x across panels: with different limits per panel a short bar in
    # one row can look longer than a tall bar in another.
    fig, axes = plt.subplots(len(idx), 1, figsize=(10.5, 1.85 * len(idx)),
                             facecolor=BG, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        zp, zq, zt = d["z_proto"][i], d["z_query"][i], d["logits"][i]
        if binary:
            ax.barh(["logit", "β·W·(q−c)", "W·c"], [zt[0], zq[0], zp[0]],
                    color=["#555555", "#D65F5F", "#4878CF"], height=.6)
            ax.axvline(0, color="black", lw=.8)
            pred, pp = int(zt[0] > 0), int(zp[0] > 0)
        else:
            w, yy = .27, np.arange(ncl)
            ax.barh(yy - w, zp, height=w, color="#4878CF", label="W·c")
            ax.barh(yy, zq, height=w, color="#D65F5F", label="β·W·(q−c)")
            ax.barh(yy + w, zt, height=w, color="#555555", label="logit")
            ax.set_yticks(yy)
            ax.set_yticklabels(lab, fontsize=TICK)
            ax.invert_yaxis()
            pred, pp = int(np.argmax(zt)), int(np.argmax(zp))
        # Shade the predicted row so the winner is visible without reading
        # every bar end.
        if not binary:
            ax.axhspan(pred - .45, pred + .45, color="#f0f0f0", zorder=0)
        flip = pred != pp
        ax.set_title(
            f"sample #{i}   prototype P{d['grp'][i]}   true={lab[d['y'][i]]}"
            f"   prototype alone={lab[pp]}   final={lab[pred]}"
            + ("   ← the query term flips it" if flip else ""),
            fontsize=TICK, loc="left", color=QC if flip else "black")
        ax.tick_params(labelsize=TICK)
        ax.grid(axis="x", linestyle="--", lw=.3, alpha=.4)
    if not binary:
        axes[0].legend(fontsize=TICK, ncol=3, loc="upper right")
    axes[-1].set_xlabel("logit contribution", fontsize=LABEL)

    pa = (d["z_proto"][:, 0] > 0).astype(int) if binary else d["z_proto"].argmax(1)
    fa = (d["logits"][:, 0] > 0).astype(int) if binary else d["logits"].argmax(1)
    fig.suptitle(
        f"Panel 3  Prediction decomposition   z = W·c + β·W·(q−c)   "
        f"[β = {d['beta']:.3f};  the query term changes argmax on "
        f"{(pa != fa).mean():.1%} of train]", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig4(d, out, seed=0, stride=5):
    """Evidence chain for one query, and whether evidence tracks the decision.

    The right panel reports alignment, not usefulness. A correlation between
    prototype confidence and neighbour agreement says the neighbourhood is
    consistent with what was predicted; it does not say retrieval produced or
    improved the prediction. R1 tested that directly and found no gain, so the
    figure keeps retrieval off the prediction path."""
    rng = np.random.default_rng(seed)
    lab, _ = labels_of(d)
    big = max(d["active"], key=lambda p: (d["grp"] == p).sum())
    pool = np.where(d["grp"] == big)[0]
    qi = int(rng.choice(pool))
    nb = neighbours(d, qi)
    binary = d["logits"].shape[1] == 1
    pred = int(d["logits"][qi, 0] > 0) if binary else int(d["logits"][qi].argmax())

    fig = plt.figure(figsize=(16.0, 6.4), facecolor=BG)
    gs = GridSpec(1, 3, width_ratios=[1.25, 1.0, 1.15], wspace=.32)

    ax = fig.add_subplot(gs[0])
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(.5, .95, f"query #{qi}", ha="center", fontsize=TITLE, weight="bold",
            color=QC, transform=ax.transAxes)
    ax.text(.5, .875, "↓   argmax cos(q, c)", ha="center", fontsize=TICK,
            transform=ax.transAxes, color="#666")
    ax.text(.5, .80, f"prototype P{big}   (n = {len(pool)})", ha="center",
            fontsize=TITLE, transform=ax.transAxes)
    ax.text(.5, .725, f"↓   NN(q, G_{big}),  k = {len(nb)}", ha="center",
            fontsize=TICK, transform=ax.transAxes, color="#666")
    for r, j in enumerate(nb):
        same = d["y"][j] == d["y"][qi]
        ax.text(.05, .64 - r * .072,
                f"#{j:<5} {lab[d['y'][j]]:<12} "
                f"cos {float(d['Kn'][j] @ d['Kn'][qi]):.3f}",
                fontsize=TICK, family="monospace", transform=ax.transAxes,
                color="#2c5aa0" if same else QC)
    agree = int((d["y"][nb] == d["y"][qi]).sum())
    ax.text(.5, .04, f"{agree} / {len(nb)} share the query's class",
            ha="center", fontsize=LABEL, weight="bold", transform=ax.transAxes)
    ax.set_title(f"Panel 4  Retrieval as evidence — outside the prediction path\n"
                 f"prediction: {lab[pred]}", fontsize=TITLE)

    axm = fig.add_subplot(gs[1])
    mu, sdv = d["X"][pool].mean(0), d["X"][pool].std(0) + 1e-8
    z = (d["X"][qi] - mu) / sdv
    top = np.argsort(-np.abs(z))[:6]
    axm.barh(range(len(top)), z[top],
             color=[QC if v > 0 else "#4878CF" for v in z[top]])
    axm.set_yticks(range(len(top)))
    axm.set_yticklabels([str(d["cols"][t])[:20] for t in top], fontsize=TICK)
    axm.invert_yaxis()
    axm.axvline(0, color="black", lw=.8)
    axm.set_xlabel("z within the group", fontsize=LABEL)
    dm = float(np.median(np.abs(mu - d["X"].mean(0)) / (d["X"].std(0) + 1e-8)))
    axm.set_title(f"how the query differs from G_{big}\n"
                  f"group vs global mean: {dm:.3f} σ", fontsize=TITLE)

    axr = fig.add_subplot(gs[2])
    conf, agr = [], []
    for i in range(0, d["n"], stride):
        m = np.where(d["grp"] == d["grp"][i])[0]
        if len(m) < d["k"] + 1:
            continue
        nn = neighbours(d, i)
        zp = d["z_proto"][i]
        conf.append(float(abs(zp[0])) if binary
                    else float(torch.softmax(torch.from_numpy(zp), 0).max()))
        agr.append(float((d["y"][nn] == d["y"][i]).mean()))
    conf, agr = np.array(conf), np.array(agr)
    axr.scatter(conf, agr + rng.normal(0, .012, len(agr)), s=13, alpha=.4,
                color="#4878CF", linewidths=0)
    rho = float(np.corrcoef(conf, agr)[0, 1]) if len(conf) > 2 else float("nan")
    # W*c is identical for every member of a prototype, so the points form one
    # vertical band per prototype. The correlation is therefore across
    # prototypes, not across samples within one.
    axr.set_xlabel("prototype-only confidence  (one band per prototype)",
                   fontsize=LABEL)
    axr.set_ylabel("neighbour label agreement", fontsize=LABEL)
    # Alignment, not usefulness. A correlation here says the neighbourhood is
    # consistent with what was predicted; it does not say retrieval produced
    # or improved the prediction. R1 tested that and found no gain.
    axr.set_title(f"evidence aligns with the decision\n"
                  f"(alignment, not contribution)   r = {rho:+.3f}"
                  f"   n = {len(conf)}", fontsize=TITLE)
    axr.grid(linestyle="--", lw=.3, alpha=.4)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def fig5(d, out):
    """Pairwise cosine between centroids, ordered by group size.

    Panels 1 and 2 show where each prototype sits and what it holds; this shows
    how the prototypes sit relative to each other. Two centroids at cosine 0.96
    point almost the same way, which the projection in panel 1 cannot reveal."""
    rows = profiles(d)
    order = [r["p"] for r in rows]
    Cn = F.normalize(d["C"], dim=-1)[order]
    S = (Cn @ Cn.T).numpy()

    side = max(6.0, min(11.0, 1.8 + .22 * len(order)))
    fig, ax = plt.subplots(figsize=(side + 1.2, side), facecolor=BG)
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    # Label each prototype only while the ticks stay legible; past that they
    # overlap into a grey band and hide the matrix.
    if len(order) <= 20:
        lab = [f"P{p}" for p in order]
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, rotation=90, fontsize=TICK)
        ax.set_yticks(range(len(lab)))
        ax.set_yticklabels(lab, fontsize=TICK)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{len(order)} prototypes, largest first", fontsize=LABEL)
    iu = np.triu_indices(len(S), 1)
    ax.set_title(f"Panel 5  cos(c_p, c_q), ordered by |G_p|\n"
                 f"off-diagonal mean {S[iu].mean():+.2f}, "
                 f"max {S[iu].max():+.2f}", fontsize=TITLE)
    ax.grid(False)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, default=None)
    ap.add_argument("--openml_id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--savepath", type=str, default="figures")
    args = ap.parse_args()

    path = args.state
    if path is None:
        if args.openml_id is None:
            ap.error("--state or --openml_id required")
        # Ablation runs write their tag into the filename, but runs that
        # differ only by fusion_mode do not, and older checkpoints predate
        # proto_dev entirely. Filename filtering is not enough: open each
        # candidate and keep the ones the final model can actually read.
        EXP = ("trainseed", "beta_x5", "beta_base", "nocommit", "grad_cb",
               "nodr", "nbr0", "labelOnly", "fusion_")
        raw = [x for x in sorted(glob.glob(
            f"**/data={args.openml_id}..*seed{args.seed}_model_state.pt",
            recursive=True))
            if not any(t in os.path.basename(x) for t in EXP)]
        cand, other = [], []
        for x in raw:
            try:
                ck = torch.load(x, map_location="cpu", weights_only=False)
                fm = ck["model_kwargs"].get("fusion_mode")
                ok = (fm == "proto_dev"
                      and "dev_head.weight" in ck["state_dict"])
            except Exception:
                fm, ok = "unreadable", False
            (cand if ok else other).append((x, fm))
        if not cand:
            msg = "\n".join(f"      {x}   fusion_mode={fm}" for x, fm in other[:6])
            ap.error(f"no proto_dev checkpoint for openml_id={args.openml_id}"
                     + (f"\n    found instead:\n{msg}" if other else ""))
        cand.sort(key=lambda t: (len(os.path.dirname(t[0])),
                                 len(os.path.basename(t[0]))))
        path = cand[0][0]
        if len(cand) > 1:
            print(f"  {len(cand)} proto_dev candidates; using the first. "
                  f"Pass --state to choose.")
            for x, _ in cand[:4]:
                print(f"      {x}")
        if other:
            print(f"  ({len(other)} skipped: not proto_dev or missing dev_head)")
    print(f"  state: {path}")
    fm = torch.load(path, map_location="cpu",
                    weights_only=False)["model_kwargs"].get("fusion_mode")
    print(f"  fusion_mode: {fm}")
    if fm != "proto_dev":
        print(f"  ⚠ the final model is proto_dev; this checkpoint is '{fm}'")

    d = load(path)
    print(f"  n_train={d['n']}  P={d['C'].shape[0]}  "
          f"active={len(d['active'])}  beta={d['beta']:.3f}")
    # Mirror reproduce.py's layout: figures/seed=N/ next to optim_logs/seed=N/.
    out_dir = os.path.join(args.savepath, f"seed={args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    tag = os.path.basename(path).split("..")[0]
    for i, fn in enumerate([fig1, fig2, fig3, fig4, fig5], 1):
        out = os.path.join(out_dir, f"fig{i}_{tag}.png")
        fn(d, out)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()