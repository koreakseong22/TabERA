# -*- coding: utf-8 -*-
"""
visualize_prototype_geometry.py
===============================
TabERA -- prototype geometry and conditional partitions (4 panels).

Reads a checkpoint written by reproduce.py and draws:

  (a) Embedding space      all q_i, centroids c_p, assignment
  (b) Prototype geometry   centroid PCA + pairwise cosine heatmap
  (c) Prototype statistics |G_p| vs within-group cosine, class entropy
  (d) Raw-feature contrast standardised group-to-global mean difference

Why these four
--------------
(a) and (c) show what the model is defined to do: the encoder learns a
prediction-oriented embedding and the EMA prototypes partition it.

(d) is the counterpart. A partition that is coherent in embedding space is
not necessarily distinguishable in the original features, and how far the two
agree varies by dataset. Putting them side by side is the point of the figure,
not an afterthought.

Usage
-----
python visualize_prototype_geometry.py --state path/to/..._model_state.pt
python visualize_prototype_geometry.py --openml_id 31 --seed 1
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
sys.modules.setdefault("openml", types.ModuleType("openml"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BG = "#ffffff"
ACCENT = "#c0392b"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--",
    "grid.linewidth": .4, "grid.alpha": .35,
})


# ─────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────
def load_state(path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    n = st["n_train"]
    sd = st["state_dict"]
    C = sd["prototype_layer.centroid_emb"]
    K = F.normalize(sd["memory.keys"][:n], dim=-1)
    fs = st["feature_store_state"]
    X = fs[0][:n].numpy()
    # feature_store rows, memory.keys and memory.labels all sit in memory-slot
    # order, so they line up without reindexing. fs[3] holds the original row
    # ids and is only needed when cross-referencing the source dataframe.
    y = sd["memory.labels"][:n].numpy().astype(int).ravel()
    groups = st["sample_groups"]
    active = [p for p, g in enumerate(groups)
              if len([i for i in (g or []) if 0 <= i < n]) >= 2]
    return dict(C=C, K=K, X=X, y=y, groups=groups, active=active, n=n,
                cols=st.get("col_names") or
                     st["model_kwargs"].get("column_names") or
                     [f"f{i}" for i in range(X.shape[1])])


def assignment(d):
    a = np.full(d["n"], -1, dtype=int)
    for p, g in enumerate(d["groups"]):
        for i in (g or []):
            if 0 <= i < d["n"]:
                a[i] = p
    return a


# ─────────────────────────────────────────────────────────────
# panels
# ─────────────────────────────────────────────────────────────
def panel_a(ax, d, a, seed=0):
    """Embedding space: samples coloured by prototype, centroids overlaid."""
    from sklearn.decomposition import PCA
    Z = np.concatenate([d["K"].numpy(), F.normalize(d["C"], dim=-1).numpy()])
    P2 = PCA(n_components=2, random_state=seed).fit(Z)
    Q2, C2 = P2.transform(d["K"].numpy()), P2.transform(
        F.normalize(d["C"], dim=-1).numpy())
    cmap = plt.get_cmap("tab20")
    order = sorted(d["active"], key=lambda p: -(a == p).sum())
    rng = np.random.default_rng(seed)
    for j, p in enumerate(order):
        m = np.where(a == p)[0]
        if len(m) == 0:
            continue
        col = cmap(j % 20)
        ax.scatter(Q2[m, 0], Q2[m, 1], s=14, alpha=.55, color=col, linewidths=0)
        # A few assignment lines per prototype. Drawing all of them turns the
        # panel into a solid block and hides the sample cloud.
        for i in rng.choice(m, size=min(12, len(m)), replace=False):
            ax.plot([Q2[i, 0], C2[p, 0]], [Q2[i, 1], C2[p, 1]],
                    color=col, lw=.35, alpha=.30, zorder=1)
    ax.scatter(C2[order, 0], C2[order, 1], s=120, marker="X",
               c="white", edgecolors="black", linewidths=1.6, zorder=5)
    ev = P2.explained_variance_ratio_
    ax.set_title(f"(a) Embedding space\n"
                 f"{len(d['active'])} active prototypes, lines = assignment",
                 fontsize=12)
    ax.set_xlabel(f"PC1 ({ev[0]:.0%})", fontsize=12)
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})", fontsize=12)
    if ev[0] > .6:
        # One direction carrying most of the variance is worth stating: the
        # 2-D picture then shows far less of the geometry than it appears to.
        ax.text(.02, .02, f"PC1 alone carries {ev[0]:.0%} — 2-D view is partial",
                transform=ax.transAxes, fontsize=9, style="italic", color="#555")


def panel_b(ax, d):
    """Pairwise cosine between centroids."""
    # Ordered by group size so the heatmap and panel (a) refer to prototypes
    # in the same sequence.
    order = sorted(d["active"], key=lambda p: -len(
        [i for i in (d["groups"][p] or []) if 0 <= i < d["n"]]))
    Cn = F.normalize(d["C"], dim=-1)[order]
    S = (Cn @ Cn.T).numpy()
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("(b) Prototype geometry\ncos(c_p, c_q),  ordered by |G_p|",
                 fontsize=12)
    # Label every prototype only while the ticks stay legible; past that the
    # text overlaps into a grey band and hides the matrix.
    if len(order) <= 20:
        lab = [f"G{p}" for p in order]
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, rotation=90, fontsize=8)
        ax.set_yticks(range(len(lab)))
        ax.set_yticklabels(lab, fontsize=8)
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{len(order)} prototypes, largest first", fontsize=12)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    iu = np.triu_indices(len(S), 1)
    ax.set_xlabel(f"off-diagonal mean {S[iu].mean():+.2f}", fontsize=12)


def panel_c(ax, d, a):
    """Group size against within-group cosine; colour = label entropy."""
    Cn = F.normalize(d["C"], dim=-1)
    sz, wc, ent = [], [], []
    for p in d["active"]:
        m = np.where(a == p)[0]
        cs = (d["K"][m] @ Cn[p]).numpy()
        sz.append(len(m)); wc.append(cs.mean())
        if d["y"] is not None:
            _, c = np.unique(d["y"][m], return_counts=True)
            q = c / c.sum(); ent.append(float(-(q * np.log(q)).sum()))
        else:
            ent.append(0.0)
    sc = ax.scatter(sz, wc, s=70, c=ent, cmap="viridis",
                    edgecolors="black", linewidths=.6)
    if len(sz) <= 20:
        for x_, y_, p in zip(sz, wc, d["active"]):
            ax.annotate(f"G{p}", (x_, y_), fontsize=6, xytext=(4, 4),
                        textcoords="offset points", color="#444")
    ax.set_xscale("log")
    ax.set_title("(c) Prototype statistics", fontsize=12)
    ax.set_xlabel("group size  |G_p|  (log)", fontsize=12)
    ax.set_ylabel("mean cos(q, c_p)", fontsize=12)
    cb = plt.colorbar(sc, ax=ax, fraction=.046, pad=.04)
    cb.set_label("H(Y | G_p)", fontsize=12)


def panel_d(ax, d, a, topk=8):
    """Standardised group-to-global mean difference in raw features."""
    X = d["X"]
    mu, sd = X.mean(0), X.std(0) + 1e-8
    order = sorted(d["active"], key=lambda p: -(a == p).sum())[:6]
    M = np.stack([np.abs(X[a == p].mean(0) - mu) / sd for p in order])
    keep = np.argsort(-M.mean(0))[:topk]
    im = ax.imshow(M[:, keep], cmap="magma", aspect="auto", vmin=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"G{p} (n={(a == p).sum()})" for p in order], fontsize=9)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([str(d["cols"][i])[:11] for i in keep],
                       rotation=45, ha="right", fontsize=9)
    ax.set_title("(d) Raw-feature contrast\n|group mean - global mean| / sigma",
                 fontsize=12)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    # This median is the quantity that separates the two regimes: a partition
    # can be tight in embedding space (panel c) and still sit on top of the
    # global feature distribution, in which case "compared with similar cases"
    # does not hold for that group.
    med = float(np.median(M))
    ax.set_xlabel(f"median |Δmean|/σ = {med:.3f}"
                  f"   ({'little raw contrast' if med < .3 else 'raw-space local'})",
                  fontsize=10, fontweight="bold")


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
        # --run_tag writes the tag into the filename, so ablation runs sit
        # alongside the default one. Drop the known tags and prefer the
        # shortest remaining name.
        # Filename filtering alone is not enough: runs differing only by
        # fusion_mode share a name, and older checkpoints predate proto_dev.
        # Open each candidate and keep the readable proto_dev ones.
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
                ok = ck["model_kwargs"].get("fusion_mode") == "proto_dev"
            except Exception:
                ok = False
            (cand if ok else other).append(x)
        if other:
            print(f"  ({len(other)} skipped: not proto_dev)")
        cand.sort(key=lambda p: (len(os.path.dirname(p)),
                                 len(os.path.basename(p))))
        if not cand:
            ap.error(f"no checkpoint for openml_id={args.openml_id}")
        path = cand[0]
        if len(cand) > 1:
            # Runs that differ only by fusion_mode share a filename and are
            # told apart by directory, which this cannot see. Say which one
            # was taken rather than picking silently.
            print(f"  ⚠ {len(cand)} candidates; using the first. "
                  f"Pass --state to choose explicitly.")
            for c in cand[:4]:
                print(f"      {c}")
    print(f"  state: {path}")
    _fm = torch.load(path, map_location="cpu",
                     weights_only=False)["model_kwargs"].get("fusion_mode")
    print(f"  fusion_mode: {_fm}")

    d = load_state(path)
    a = assignment(d)
    print(f"  n_train={d['n']}  P={len(d['groups'])}  active={len(d['active'])}")

    fig = plt.figure(figsize=(14.5, 10), facecolor=BG)
    gs = GridSpec(2, 2, figure=fig, hspace=.32, wspace=.26)
    for ax_pos, fn in [((0, 0), panel_a), ((0, 1), panel_b),
                       ((1, 0), panel_c), ((1, 1), panel_d)]:
        ax = fig.add_subplot(gs[ax_pos])
        ax.set_facecolor("white")
        fn(ax, d, a) if fn is not panel_b else fn(ax, d)

    tag = os.path.basename(path).split("..")[0]
    fig.suptitle(f"TabERA -- prototype geometry and conditional partitions"
                 f"   [{tag}]", fontsize=12, y=.98)
    # Mirror reproduce.py's layout: figures/seed=N/ next to optim_logs/seed=N/.
    out_dir = os.path.join(args.savepath, f"seed={args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"prototype_geometry_{tag}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=BG)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()