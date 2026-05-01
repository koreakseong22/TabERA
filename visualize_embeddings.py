"""
visualize_embeddings.py
=======================
TabERA — Embedding Space 3-Figure 시각화

한 번 실행으로 세 가지 독립 Figure를 생성합니다.

  Figure A : 임베딩 공간 구조
             — t-SNE/PCA 투영 + label별 컬러링 + centroid 위치
             — "embedder가 class를 얼마나 잘 분리했는가"

  Figure B : Centroid가 데이터 구조를 발견했는가
             — centroid별 담당 class 비율을 파이 차트로 오버레이
             — "P개 centroid가 class와 무관하게 데이터를 분할했는가"

  Figure C : Retrieval이 centroid 그룹 내에서 일어나는가
             — query 1개 클로즈업, 그룹 내 이웃 k개, evidence_w 수치
             — "grouped KNN의 효율성과 설명가능성"

사용법
------
# 첫 실행 (학습 + 임베딩 추출 + 세 Figure 생성)
python visualize_embeddings.py --openml_id 54 --seed 1 --proj tsne

# reproduce.py로 학습한 최적 모델 사용 (권장)
python visualize_embeddings.py --openml_id 54 --seed 1 --proj tsne --from_state

# pkl 재사용 (학습 없이 Figure만 재생성)
python visualize_embeddings.py --openml_id 54 --seed 1 --proj tsne --from_pkl

출력
----
  figures/A_embed_{id}_seed{seed}_{proj}.png
  figures/B_centroid_{id}_seed{seed}_{proj}.png
  figures/C_retrieval_{id}_seed{seed}_{proj}.png
  figures/data_{id}_seed{seed}_{proj}.pkl
"""

import os
import argparse
import json
import pickle
import joblib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from libs.data       import TabularDataset
from libs.supervised import TabERAWrapper
from libs.tabera     import TabERA

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# params_to_model_kwargs 인라인 (순환 import 방지)
# ─────────────────────────────────────────────────────────────

def params_to_model_kwargs(params: dict, n_features: int, n_output: int) -> dict:
    return {
        "n_features":      n_features,
        "embed_dim":       params["embed_dim"],
        "n_prototypes":    params["n_prototypes"],
        "k":               params["k"],
        "embedder_layers": params["embedder_layers"],
        "dropout":         params["dropout"],
        "n_output":        n_output,
        "loss_weights": {
            "diversity":   params["loss_diversity"],
            "commitment":  params["loss_commitment"],
            "entropy":     params["loss_entropy"],
        },
    }


# ─────────────────────────────────────────────────────────────
# 색상
# ─────────────────────────────────────────────────────────────

PALETTE = [
    "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7",
    "#C4AD66", "#77BEDB", "#E78AC3", "#A6D854",
    "#FFD92F", "#E5C494", "#B3B3B3", "#66C2A5",
]
QUERY_COLOR = "#c0392b"
BG_COLOR    = "#f8f8f6"


# ─────────────────────────────────────────────────────────────
# 임베딩 추출
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, X_all, y_all, device, chunk=512):
    model.eval()
    model = model.to(device)
    all_emb, all_assign, all_prob, all_ew, all_topk = [], [], [], [], []

    for start in range(0, len(X_all), chunk):
        xb  = X_all[start:start+chunk].to(device)
        out = model(xb, return_explanations=False)
        emb = model.embedder(xb)
        all_emb.append(emb.cpu().numpy())
        all_assign.append(out["hard_group"].cpu().numpy())
        all_prob.append(out["routing"].cpu().numpy())
        all_ew.append(out["evidence_w"].cpu().numpy())
        all_topk.append(out["topk_idx"].cpu().numpy())

    return dict(
        emb          = np.concatenate(all_emb,    axis=0),
        hard_assign  = np.concatenate(all_assign, axis=0),
        routing_prob = np.concatenate(all_prob,   axis=0),
        evidence_w   = np.concatenate(all_ew,     axis=0),
        topk_idx     = np.concatenate(all_topk,   axis=0),
        y            = y_all.cpu().numpy().astype(int),
        centroid_emb = model.prototype_layer.centroid_emb.detach().cpu().numpy(),
        centroid_x   = (model.prototype_layer.centroid_x.cpu().numpy()
                        if model.prototype_layer.centroid_x is not None else None),
        P            = model.prototype_layer.P,
        k            = model.k,
    )


# ─────────────────────────────────────────────────────────────
# 2D 투영
# ─────────────────────────────────────────────────────────────

def project(emb, centroid_emb, method="tsne", seed=42):
    emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    cen_n = centroid_emb / (np.linalg.norm(centroid_emb, axis=1, keepdims=True) + 1e-8)
    N = len(emb_n)

    if method == "pca":
        proj      = PCA(n_components=2, random_state=seed)
        X2d       = proj.fit_transform(emb_n)
        C2d       = proj.transform(cen_n)
        explained = proj.explained_variance_ratio_.tolist()
    else:
        perp  = min(30, max(5, N // 10))
        proj  = TSNE(n_components=2, perplexity=perp, random_state=seed,
                     max_iter=1000, init="pca", learning_rate="auto")
        all2d = proj.fit_transform(np.vstack([emb_n, cen_n]))
        X2d, C2d = all2d[:N], all2d[N:]
        explained = None

    return X2d, C2d, explained


# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────

def make_cmap(classes):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(classes))}


def dominant_label(y, hard_assign, p):
    idx = np.where(hard_assign == p)[0]
    if len(idx) == 0:
        return -1
    return int(np.bincount(y[idx]).argmax())


def centroid_class_dist(y, hard_assign, p, n_classes):
    idx = np.where(hard_assign == p)[0]
    if len(idx) == 0:
        return np.zeros(n_classes)
    counts = np.bincount(y[idx], minlength=n_classes).astype(float)
    return counts / counts.sum()


def pick_best_query(X2d, hard_assign, C2d, evidence_w, topk_idx, n_train, k_show=5):
    """
    Figure C에 적합한 query + centroid 그룹 선정.

    선정 로직 (하드 필터링):
      1. 그룹 내 이웃 비율 threshold (0.8 → 0.6 → 0.4 순으로 완화)를 만족하는
         후보 중에서
      2. evidence_w 엔트로피가 가장 높은 샘플 선택
         (weight가 여러 이웃에 고르게 퍼진 케이스 → 바 차트가 의미 있게 보임)
      3. 각 그룹에서 centroid와 중간 거리 샘플만 후보로 사용
         (outlier 제외)
    """
    P = len(C2d)

    def score_candidates(min_grp_ratio):
        best_si, best_ci, best_ent = -1, -1, -1.0
        for p in range(P):
            idx_p = np.where(hard_assign == p)[0]
            if len(idx_p) < 10:
                continue

            # 중간 거리 후보 (1/4 ~ 3/4 구간)
            dists    = np.linalg.norm(X2d[idx_p] - C2d[p], axis=1)
            sorted_i = np.argsort(dists)
            lo = len(sorted_i) // 4
            hi = len(sorted_i) * 3 // 4
            candidates = idx_p[sorted_i[lo:hi]]

            for si in candidates:
                # ① 그룹 내 이웃 비율
                raw   = np.clip(topk_idx[si], 0, n_train - 1)
                valid = [int(ni) for ni in raw if int(ni) < len(hard_assign)]
                in_grp = sum(1 for ni in valid if hard_assign[ni] == p)
                grp_ratio = in_grp / max(len(valid), 1)

                if grp_ratio < min_grp_ratio:
                    continue

                # ② evidence_w 엔트로피
                ew  = evidence_w[si][:k_show]
                ew  = ew / (ew.sum() + 1e-8)
                ent = float(-np.sum(ew * np.log(ew + 1e-8)))

                if ent > best_ent:
                    best_ent = ent
                    best_si, best_ci = int(si), int(p)

        return best_si, best_ci

    # threshold 순차 완화
    for threshold in [0.8, 0.6, 0.4]:
        si, ci = score_candidates(threshold)
        if si != -1:
            return si, ci

    # 최후 fallback: 가장 큰 그룹의 중간 샘플
    best_p = max(range(P), key=lambda p: (hard_assign == p).sum())
    idx_p  = np.where(hard_assign == best_p)[0]
    dists  = np.linalg.norm(X2d[idx_p] - C2d[best_p], axis=1)
    si     = int(idx_p[np.argsort(dists)[len(dists) // 3]])
    return si, best_p


def topk_in_2d(topk_idx, n_train, si, k_show=5):
    """
    모델이 실제로 반환한 이웃을 그대로 사용.
    그룹 내/외 필터링 없음 — 정직한 시각화.
    """
    raw     = topk_idx[si]
    clipped = np.clip(raw, 0, n_train - 1)
    return [int(ni) for ni in clipped[:k_show]]


def proj_labels(method, explained):
    if method == "pca" and explained:
        return (f"PC1 ({explained[0]*100:.1f}% var.)",
                f"PC2 ({explained[1]*100:.1f}% var.)")
    return "t-SNE dim 1", "t-SNE dim 2"


def ax_setup(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=10, pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.4)


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # bbox_inches="tight"는 annotation 극단값이 있을 때 figsize를 폭발시킬 수 있음
    # 고정 크기로 저장 후 plt 전체 정리
    fig.savefig(path, dpi=150)
    plt.close(fig)
    plt.close("all")
    print(f"  [저장] {path}")


# ═════════════════════════════════════════════════════════════
# Figure A — 임베딩 공간 구조
# ═════════════════════════════════════════════════════════════

def draw_figure_A(X2d, y, C2d, hard_assign,
                  method, explained, dataset_name, openml_id, seed,
                  col_names, centroid_x, out_path):
    """
    전체 샘플 label별 컬러링 + centroid 위치(▲) + 그룹 경계.
    retrieval 선 없음 — "embedder가 만든 공간 구조"에만 집중.
    """
    classes = sorted(np.unique(y).tolist())
    cmap    = make_cmap(classes)
    P       = len(C2d)
    xl, yl  = proj_labels(method, explained)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # 샘플 산점도
    for c in classes:
        idx = np.where(y == c)[0]
        ax.scatter(X2d[idx, 0], X2d[idx, 1],
                   c=cmap[c], s=16, alpha=0.50,
                   linewidths=0, zorder=2, label=f"Class {c}")

    # centroid 그룹 경계 타원
    for p in range(P):
        idx_p = np.where(hard_assign == p)[0]
        if len(idx_p) < 3:
            continue
        pts     = X2d[idx_p]
        cx_, cy_ = pts.mean(0)
        sx = pts[:, 0].std() * 1.9 + 0.1
        sy = pts[:, 1].std() * 1.9 + 0.1
        dom = dominant_label(y, hard_assign, p)
        col = cmap.get(dom, "#888888")
        ax.add_patch(mpatches.Ellipse(
            (cx_, cy_), sx*2, sy*2,
            edgecolor=col, facecolor=col,
            linewidth=0, alpha=0.08, zorder=1))
        ax.add_patch(mpatches.Ellipse(
            (cx_, cy_), sx*2, sy*2,
            edgecolor=col, facecolor="none",
            linewidth=1.2, linestyle="--",
            alpha=0.45, zorder=3))

    # centroid 삼각형
    for p in range(P):
        dom = dominant_label(y, hard_assign, p)
        col = cmap.get(dom, "#888888")
        ax.scatter(C2d[p, 0], C2d[p, 1],
                   s=200, marker="^", c=col,
                   edgecolors="white", linewidths=1.4, zorder=6)
        ax.annotate(f"C{p}",
                    xy=(C2d[p, 0], C2d[p, 1]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7.5, fontweight="bold",
                    color=col, zorder=7)

    handles = [mpatches.Patch(color=cmap[c], label=f"Class {c}", alpha=0.8)
               for c in classes]
    handles += [
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor="gray", markersize=9,
               label="Centroid (centroid_emb)"),
        mpatches.Patch(facecolor="#aaa", edgecolor="#666",
                       linestyle="--", alpha=0.3,
                       label="Centroid group boundary"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8,
              framealpha=0.9, edgecolor="#ccc", ncol=2)
    ax_setup(ax,
             f"Figure A — Embedding Space Structure\n"
             f"Dataset: {dataset_name}  (id={openml_id}, seed={seed})",
             xl, yl)
    fig.tight_layout()
    save_fig(fig, out_path)


# ═════════════════════════════════════════════════════════════
# Figure B — Centroid가 데이터 구조를 발견했는가
# ═════════════════════════════════════════════════════════════

def draw_figure_B(X2d, y, C2d, hard_assign,
                  method, explained, dataset_name, openml_id, seed,
                  col_names, centroid_x, out_path):
    """
    centroid 위치에 파이 차트 오버레이.
    각 centroid의 class 비율 → P개 centroid ≠ C개 class가 시각적으로 명확.
    """
    classes  = sorted(np.unique(y).tolist())
    n_cls    = len(classes)
    cmap     = make_cmap(classes)
    colors   = [cmap[c] for c in classes]
    P        = len(C2d)
    xl, yl   = proj_labels(method, explained)

    fig, ax  = plt.subplots(figsize=(9, 7), dpi=150)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    # 샘플 (흐리게, 배경 역할)
    for c in classes:
        idx = np.where(y == c)[0]
        ax.scatter(X2d[idx, 0], X2d[idx, 1],
                   c=cmap[c], s=12, alpha=0.22,
                   linewidths=0, zorder=1)

    # centroid 그룹 경계 타원
    for p in range(P):
        idx_p = np.where(hard_assign == p)[0]
        if len(idx_p) < 3:
            continue
        pts      = X2d[idx_p]
        cx_, cy_ = pts.mean(0)
        sx = pts[:, 0].std() * 1.9 + 0.1
        sy = pts[:, 1].std() * 1.9 + 0.1
        dom = dominant_label(y, hard_assign, p)
        col = cmap.get(dom, "#888")
        ax.add_patch(mpatches.Ellipse(
            (cx_, cy_), sx*2, sy*2,
            edgecolor=col, facecolor="none",
            linewidth=1.3, linestyle="--",
            alpha=0.5, zorder=2))

    # 파이 차트 (inset axes)
    # ax 범위 먼저 확정 (canvas.draw() 대신 autoscale만 사용)
    ax.autoscale()
    ax.figure.tight_layout()
    xlim    = ax.get_xlim()
    ylim    = ax.get_ylim()
    ax_bbox = ax.get_position()
    pw = 0.052   # figure fraction 단위 파이 너비
    ph = pw * (fig.get_size_inches()[0] / fig.get_size_inches()[1])

    for p in range(P):
        idx_p = np.where(hard_assign == p)[0]
        if len(idx_p) == 0:
            continue
        dist = centroid_class_dist(y, hard_assign, p, n_cls)
        n_p  = len(idx_p)
        cx_, cy_ = C2d[p]

        # data → figure fraction 변환
        xn = (cx_ - xlim[0]) / (xlim[1] - xlim[0])
        yn = (cy_ - ylim[0]) / (ylim[1] - ylim[0])
        fx = ax_bbox.x0 + xn * ax_bbox.width
        fy = ax_bbox.y0 + yn * ax_bbox.height

        pie_ax = fig.add_axes([fx - pw/2, fy - ph/2, pw, ph])
        pie_ax.pie(dist, colors=colors, startangle=90,
                   wedgeprops=dict(linewidth=0.4, edgecolor="white"))
        pie_ax.set_aspect("equal")

        # centroid 번호 주석
        ax.annotate(
            f"C{p}\nn={n_p}",
            xy=(cx_, cy_),
            xytext=(12, 12), textcoords="offset points",
            fontsize=7, fontweight="bold", color="#333",
            zorder=9,
        )

    handles = [mpatches.Patch(color=cmap[c], label=f"Class {c}", alpha=0.85)
               for c in classes]
    handles.append(mpatches.Patch(
        facecolor="white", edgecolor="#333",
        label="Centroid (pie = class ratio)"))
    ax.legend(handles=handles, loc="best", fontsize=8,
              framealpha=0.92, edgecolor="#ccc", ncol=2)
    ax_setup(ax,
             f"Figure B — Did Centroids Discover Data Structure?\n"
             f"Pie = class distribution per centroid group  "
             f"(P={P} centroids, C={n_cls} classes)",
             xl, yl)
    fig.tight_layout()
    save_fig(fig, out_path)


# ═════════════════════════════════════════════════════════════
# Figure C — Retrieval이 centroid 그룹 내에서 일어나는가
# ═════════════════════════════════════════════════════════════

def draw_figure_C(X2d, y, C2d, hard_assign, evidence_w, topk_idx,
                  n_train, method, explained, dataset_name, openml_id, seed,
                  col_names, centroid_x, out_path, k_show=5):
    """
    query 1개 클로즈업.
    - 해당 centroid 그룹만 강조, 나머지 샘플은 흐리게
    - 그룹 내 이웃 k개 + evidence_w 바 차트
    - centroid_x 상위 feature 주석
    """
    classes = sorted(np.unique(y).tolist())
    cmap    = make_cmap(classes)
    P       = len(C2d)
    xl, yl  = proj_labels(method, explained)

    si, ci  = pick_best_query(X2d, hard_assign, C2d,
                               evidence_w, topk_idx, n_train, k_show=k_show)
    ql      = int(y[si])
    nbr_idx = topk_in_2d(topk_idx, n_train, si, k_show=k_show)
    ew_raw  = evidence_w[si][:len(nbr_idx)]
    ew_norm = ew_raw / (ew_raw.sum() + 1e-8)

    # 이웃의 그룹 내/외 여부 (MemoryBank 기준 — 실제 모델 동작)
    # ni는 MemoryBank 인덱스 = train 데이터 인덱스
    # n_train 초과 시 범위 밖 → fallback으로 처리
    nbr_in_grp = []
    for ni in nbr_idx:
        ni = int(ni)
        if ni >= n_train or ni >= len(hard_assign):
            nbr_in_grp.append(False)
        else:
            nbr_in_grp.append(bool(hard_assign[ni] == ci))

    # 클로즈업 범위: 그룹 샘플 + centroid + query 기준
    # t-SNE 왜곡으로 타원 밖에 찍히는 이웃은 별도 표시
    # fallback 이웃은 극단 좌표일 수 있으므로 범위 계산에서 제외
    idx_grp = np.where(hard_assign == ci)[0]
    ref_pts = np.vstack([X2d[idx_grp], C2d[ci:ci+1], X2d[si:si+1]])
    for ni, ig in zip(nbr_idx, nbr_in_grp):
        ni = int(ni)
        if ig and ni < len(X2d):
            ref_pts = np.vstack([ref_pts, X2d[ni:ni+1]])
    margin  = np.clip((ref_pts.max(axis=0) - ref_pts.min(axis=0)) * 0.38, 0.5, None)
    xlim    = (ref_pts[:,0].min()-margin[0], ref_pts[:,0].max()+margin[0])
    ylim    = (ref_pts[:,1].min()-margin[1], ref_pts[:,1].max()+margin[1])

    # 범위 상한 cap: 전체 데이터 범위의 70% 이내로 제한 (극단 좌표 방지)
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    max_x  = (X2d[:,0].max() - X2d[:,0].min()) * 0.70
    max_y  = (X2d[:,1].max() - X2d[:,1].min()) * 0.70
    if x_span > max_x:
        mid = (xlim[0] + xlim[1]) / 2
        xlim = (mid - max_x/2, mid + max_x/2)
    if y_span > max_y:
        mid = (ylim[0] + ylim[1]) / 2
        ylim = (mid - max_y/2, mid + max_y/2)

    # 레이아웃: 메인 + 우측 바 차트
    fig = plt.figure(figsize=(11, 6.5), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    gs  = fig.add_gridspec(1, 2, width_ratios=[2.8, 1], wspace=0.07)
    ax  = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])
    ax.set_facecolor(BG_COLOR)
    axb.set_facecolor(BG_COLOR)

    # 전체 샘플 (그룹 외 흐리게)
    for c in classes:
        idx_c   = np.where(y == c)[0]
        out_grp = [i for i in idx_c if hard_assign[i] != ci]
        in_grp  = [i for i in idx_c
                   if hard_assign[i] == ci
                   and i != si and i not in nbr_idx]
        if out_grp:
            ax.scatter(X2d[out_grp, 0], X2d[out_grp, 1],
                       c=cmap[c], s=12, alpha=0.10,
                       linewidths=0, zorder=1)
        if in_grp:
            ax.scatter(X2d[in_grp, 0], X2d[in_grp, 1],
                       c=cmap[c], s=22, alpha=0.55,
                       linewidths=0, zorder=2)

    # 그룹 경계 타원 (강조)
    pts_g    = X2d[idx_grp]
    cx_g, cy_g = pts_g.mean(0)
    sx_g = pts_g[:,0].std() * 2.0 + 0.15
    sy_g = pts_g[:,1].std() * 2.0 + 0.15
    dom   = dominant_label(y, hard_assign, ci)
    col_g = cmap.get(dom, "#888")
    ax.add_patch(mpatches.Ellipse(
        (cx_g, cy_g), sx_g*2, sy_g*2,
        edgecolor=col_g, facecolor=col_g,
        linewidth=0, alpha=0.10, zorder=1))
    ax.add_patch(mpatches.Ellipse(
        (cx_g, cy_g), sx_g*2, sy_g*2,
        edgecolor=col_g, facecolor="none",
        linewidth=2.0, linestyle="--",
        alpha=0.85, zorder=3))

    # retrieval 화살표 (evidence_w 비례 굵기·투명도)
    qx, qy = X2d[si]

    # 타원 중심·반경 (t-SNE 왜곡 감지용)
    ell_cx, ell_cy = pts_g.mean(0)
    ell_sx = pts_g[:,0].std() * 2.0 + 0.15
    ell_sy = pts_g[:,1].std() * 2.0 + 0.15

    def is_outside_ellipse(px, py):
        """2D 타원 경계 밖에 시각적으로 위치하는지 판단."""
        return ((px - ell_cx)**2 / (ell_sx**2) +
                (py - ell_cy)**2 / (ell_sy**2)) > 1.0

    for k, (ni, ew, in_grp) in enumerate(zip(nbr_idx, ew_norm, nbr_in_grp)):
        ni = int(ni)
        if ni >= len(X2d):
            continue
        nx, ny = X2d[ni]
        alpha  = float(np.clip(ew * 3.5, 0.15, 0.9))
        lw     = 1.0 + ew * 5.0
        ax.annotate(
            "", xy=(nx, ny), xytext=(qx, qy),
            arrowprops=dict(
                arrowstyle="-|>" if in_grp else "->",
                color=QUERY_COLOR,
                alpha=alpha, lw=lw, mutation_scale=10,
                linestyle="solid" if in_grp else "dashed"),
            zorder=4)

    # 이웃 포인트
    for k, (ni, ew, in_grp) in enumerate(zip(nbr_idx, ew_norm, nbr_in_grp)):
        ni = int(ni)
        if ni >= len(X2d):
            continue
        nx, ny = X2d[ni]
        c      = int(y[ni])
        sz     = 80 + ew * 280

        # t-SNE 왜곡 감지: 실제 그룹 내지만 시각적으로 타원 밖에 찍힌 경우
        proj_distorted = in_grp and is_outside_ellipse(nx, ny)

        if in_grp:
            edge_color = QUERY_COLOR
            edge_lw    = 1.8
            txt_color  = QUERY_COLOR
        else:
            edge_color = "#888888"
            edge_lw    = 1.2
            txt_color  = "#888888"

        ax.scatter(nx, ny,
                   s=sz, c=cmap.get(c, "#888"),
                   edgecolors=edge_color, linewidths=edge_lw,
                   zorder=5, alpha=0.9)

        # 라벨 구성
        label_str = f"k{k+1}  {ew*100:.0f}%"
        if proj_distorted:
            # 실제 그룹 내이지만 투영 왜곡으로 타원 밖에 보임
            label_str += "\n(proj. out)"
            txt_color = "#999999"
        elif not in_grp:
            label_str += "\n(fallback)"

        ax.annotate(
            label_str,
            xy=(nx, ny),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color=txt_color,
            fontweight="bold", zorder=7)

    # centroid 삼각형 + centroid_x 주석
    ax.scatter(C2d[ci,0], C2d[ci,1],
               s=260, marker="^", c=col_g,
               edgecolors="white", linewidths=1.6, zorder=8)
    if centroid_x is not None and col_names:
        cx_feat = centroid_x[ci]
        top2    = np.argsort(np.abs(cx_feat))[::-1][:2]
        feat_str = "\n".join(
            f"{col_names[i]}: {cx_feat[i]:.3f}" for i in top2)
        ax.annotate(
            f"C{ci}\n{feat_str}",
            xy=(C2d[ci,0], C2d[ci,1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=7.5, color=col_g, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="white", ec=col_g, alpha=0.85, lw=0.8),
            zorder=9)

    # query 사각형
    ax.scatter(qx, qy, s=150, marker="s",
               c=QUERY_COLOR,
               edgecolors="white", linewidths=1.6, zorder=9)
    ax.annotate(
        f"Query\n(Class {ql})",
        xy=(qx, qy), xytext=(-48, 8),
        textcoords="offset points",
        fontsize=8, fontweight="bold",
        color=QUERY_COLOR, zorder=10)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # 그룹 밖 이웃 수 계산
    n_fallback   = sum(1 for ig in nbr_in_grp if not ig)
    # t-SNE 왜곡으로 타원 밖에 찍힌 실제 그룹 내 이웃 수
    nbr_proj_out = [in_grp and is_outside_ellipse(X2d[int(ni),0], X2d[int(ni),1])
                    for ni, in_grp in zip(nbr_idx, nbr_in_grp)
                    if int(ni) < len(X2d)]
    # 길이 맞추기
    while len(nbr_proj_out) < len(nbr_idx):
        nbr_proj_out.append(False)
    n_proj_out = sum(nbr_proj_out)

    subtitle = f"Query (■) → top-{k_show} neighbors from assigned group"
    if n_fallback > 0:
        subtitle += f"  ({n_fallback} fallback)"
    if n_proj_out > 0:
        subtitle += f"  ({n_proj_out} proj. out: t-SNE distortion)"
    ax_setup(ax,
             f"Figure C — Retrieval within Centroid Group C{ci}\n{subtitle}",
             xl, yl)

    # 범례
    handles = [mpatches.Patch(color=cmap[c], label=f"Class {c}", alpha=0.8)
               for c in classes]
    handles += [
        Line2D([0],[0], marker="^", color="w",
               markerfacecolor=col_g, markersize=9,
               label=f"Centroid C{ci}"),
        Line2D([0],[0], marker="s", color="w",
               markerfacecolor=QUERY_COLOR, markersize=8,
               label="Query sample"),
        Line2D([0],[0], marker="o", color="w",
               markerfacecolor="#aaa",
               markeredgecolor=QUERY_COLOR, markersize=8,
               label="In-group neighbor"),
    ]
    if n_proj_out > 0:
        handles.append(
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#aaa",
                   markeredgecolor="#999999", markersize=8,
                   label="In-group (proj. out: t-SNE distortion)"))
    if n_fallback > 0:
        handles.append(
            Line2D([0],[0], marker="o", color="w",
                   markerfacecolor="#aaa",
                   markeredgecolor="#888888", markersize=8,
                   label="Fallback neighbor (out-of-group)"))
    ax.legend(handles=handles, loc="lower left",
              fontsize=7.5, framealpha=0.9,
              edgecolor="#ccc", ncol=2)

    # evidence_w 바 차트 (proj. out은 회색 점선 구분)
    labels_b      = [f"k{i+1}" for i in range(len(nbr_idx))]
    bar_colors    = [cmap.get(int(y[int(ni)]), "#888") for ni in nbr_idx]
    edge_colors_b = []
    for ig, po in zip(nbr_in_grp, nbr_proj_out):
        if not ig:
            edge_colors_b.append("#888888")   # fallback
        elif po:
            edge_colors_b.append("#999999")   # proj. out
        else:
            edge_colors_b.append(QUERY_COLOR) # 정상 in-group
    axb.barh(labels_b[::-1], ew_norm[::-1]*100,
             color=bar_colors[::-1],
             edgecolor=edge_colors_b[::-1], linewidth=0.8, alpha=0.85)
    for i, (ew, ni, ig, po) in enumerate(
            zip(ew_norm[::-1], nbr_idx[::-1],
                nbr_in_grp[::-1], nbr_proj_out[::-1])):
        if not ig:
            suffix, tc = " ↩", "#888888"
        elif po:
            suffix, tc = " ↗", "#999999"
        else:
            suffix, tc = "", "#333"
        axb.text(ew*100 + 0.5, i, f"{ew*100:.1f}%{suffix}",
                 va="center", ha="left", fontsize=8, color=tc)
    axb.set_xlabel("Attention weight (%)", fontsize=8)
    axb.set_title("Evidence weights\n(evidence_w)", fontsize=9, pad=6)
    axb.set_xlim(0, ew_norm.max() * 135)
    axb.tick_params(labelsize=8)
    axb.grid(axis="x", linestyle="--", linewidth=0.35, alpha=0.4)

    fig.suptitle(
        f"Dataset: {dataset_name}  (id={openml_id}, seed={seed})",
        fontsize=9, y=0.99)
    fig.tight_layout()
    save_fig(fig, out_path)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TabERA 3-Figure Embedding Visualizer")
    parser.add_argument("--gpu_id",      type=int,  default=0)
    parser.add_argument("--openml_id",   type=int,  required=True)
    parser.add_argument("--seed",        type=int,  default=1)
    parser.add_argument("--savepath",    type=str,  default=".",
                        help="optim_logs가 있는 상위 경로")
    parser.add_argument("--json",        type=str,  default="dataset_id.json")
    parser.add_argument("--epochs",      type=int,  default=200)
    parser.add_argument("--patience",    type=int,  default=30)
    parser.add_argument("--proj",        type=str,  default="tsne",
                        choices=["pca", "tsne"])
    parser.add_argument("--max_samples", type=int,  default=3000)
    parser.add_argument("--k_show",      type=int,  default=5,
                        help="Figure C에서 보여줄 이웃 수")
    parser.add_argument("--out_dir",     type=str,  default="figures",
                        help="figures 루트 경로 (seed 하위 폴더 자동 생성)")
    parser.add_argument("--from_pkl",    action="store_true",
                        help="학습 없이 기존 pkl 재사용")
    parser.add_argument("--from_state",  action="store_true",
                        help="reproduce.py가 저장한 최적 model state 로드 (권장)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    oid      = str(args.openml_id)
    # figures/seed={seed}/ 구조 자동 생성
    out_dir  = os.path.join(args.out_dir, f"seed={args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir,
                            f"data_{oid}_seed{args.seed}_{args.proj}.pkl")

    # ── from_state: reproduce.py가 저장한 최적 모델 로드 ────
    if args.from_state:
        with open(args.json, "r") as f:
            data_info = json.load(f)
        dataset_info = data_info[oid]
        tasktype     = dataset_info["tasktype"]
        dataset_name = dataset_info.get("fullname", dataset_info.get("name", oid))

        if not args.savepath.endswith("optim_logs"):
            log_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
        else:
            log_dir = args.savepath

        state_path = os.path.join(log_dir,
                                  f"data={oid}..seed{args.seed}_model_state.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"model state 없음: {state_path}\n"
                f"먼저 reproduce.py --openml_id {oid} --seed {args.seed} 를 실행하세요.")

        print(f"[TabERA Visualize] {dataset_name} (id={oid})  → state 로드")
        ckpt = torch.load(state_path, map_location=device)

        col_names  = ckpt["col_names"]
        n_train    = ckpt["n_train"]
        model_kwargs = ckpt["model_kwargs"]
        print(f"  val : {ckpt['val_metrics']}")
        print(f"  test: {ckpt['test_metrics']}")

        model = TabERA(**model_kwargs, column_names=col_names,
                       memory_size=min(n_train * 2, 10_000))
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        model.eval()
        print("  model state 로드 완료.")

        # 데이터 로드 (임베딩 추출용)
        dataset = TabularDataset(args.openml_id, tasktype,
                                 device=device, seed=args.seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()

        X_all = torch.cat([X_train, X_val, X_test], dim=0)
        y_all = torch.cat([
            y_train.long() if y_train.dtype != torch.long else y_train,
            y_val.long()   if y_val.dtype   != torch.long else y_val,
            y_test.long()  if y_test.dtype  != torch.long else y_test,
        ], dim=0)

        N = len(X_all)
        if N > args.max_samples:
            rng  = np.random.default_rng(args.seed)
            keep = np.sort(rng.choice(N, args.max_samples, replace=False))
            X_all   = X_all[keep]
            y_all   = y_all[keep]
            n_train = int((keep < len(X_train)).sum())
            print(f"  샘플 수 제한: {N} → {args.max_samples}")

        print("  임베딩 추출 중...")
        emb_data = extract_embeddings(model, X_all, y_all, device)

        print(f"  {args.proj.upper()} 투영 중...")
        X2d, C2d, explained = project(
            emb_data["emb"], emb_data["centroid_emb"],
            method=args.proj, seed=args.seed)

        y           = emb_data["y"]
        hard_assign = emb_data["hard_assign"]
        evidence_w  = emb_data["evidence_w"]
        topk_idx    = emb_data["topk_idx"]
        centroid_x  = emb_data["centroid_x"]

        with open(pkl_path, "wb") as f:
            pickle.dump(dict(
                X2d=X2d, C2d=C2d, y=y,
                hard_assign=hard_assign,
                evidence_w=evidence_w,
                topk_idx=topk_idx,
                n_train=n_train,
                explained=explained,
                dataset_name=dataset_name,
                col_names=col_names,
                centroid_x=centroid_x,
            ), f)
        print(f"  [저장] {pkl_path}")

    # ── pkl 재사용 ────────────────────────────────────────
    elif args.from_pkl:
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(
                f"pkl 없음: {pkl_path}\n먼저 --from_pkl 없이 실행하세요.")
        print(f"  pkl 재사용: {pkl_path}")
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        X2d, C2d     = d["X2d"], d["C2d"]
        y            = d["y"]
        hard_assign  = d["hard_assign"]
        evidence_w   = d["evidence_w"]
        topk_idx     = d["topk_idx"]
        n_train      = d["n_train"]
        explained    = d["explained"]
        dataset_name = d["dataset_name"]
        col_names    = d["col_names"]
        centroid_x   = d["centroid_x"]

    else:
        # ── 데이터 + 모델 로드 + 학습 ────────────────────
        with open(args.json, "r") as f:
            data_info = json.load(f)
        dataset_info = data_info[oid]
        tasktype     = dataset_info["tasktype"]
        dataset_name = dataset_info.get("fullname", dataset_info.get("name", oid))
        print(f"[TabERA Visualize] {dataset_name} (id={oid}, task={tasktype})")

        dataset    = TabularDataset(args.openml_id, tasktype,
                                    device=device, seed=args.seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
        output_dim = dataset.n_classes if tasktype == "multiclass" else 1
        col_names  = dataset.col_names

        if not args.savepath.endswith("optim_logs"):
            log_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
        else:
            log_dir = args.savepath

        fname = os.path.join(log_dir, f"data={oid}..model=tabera.pkl")
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"최적화 로그 없음: {fname}\n"
                f"먼저 optimize.py --openml_id {oid} --seed {args.seed} 를 실행하세요.")

        study       = joblib.load(fname)
        best_params = study.best_params
        print(f"  Best trial #{study.best_trial.number}  val={study.best_value:.4f}")

        # optimize.py가 실제 사용한 n_prototypes 그대로 복원
        best_params["n_prototypes"] = study.best_trial.user_attrs["n_prototypes_actual"]
        print(f"  n_prototypes (from optimize.py): {best_params['n_prototypes']}")

        best_params.setdefault("loss_entropy",    0.01)
        best_params.setdefault("loss_diversity",  0.01)
        best_params.setdefault("loss_commitment", 0.01)
        best_params.setdefault("n_heads",         4)
        best_params.setdefault("anneal_factor",   0.97)

        model_kwargs = params_to_model_kwargs(best_params, dataset.n_features, output_dim)
        model = TabERA(
            **model_kwargs,
            column_names=col_names,
            memory_size=min(int(len(y_train) * 2), 10_000),
        )
        wrapper = TabERAWrapper(
            model, best_params, tasktype,
            device=str(device), epochs=args.epochs, patience=args.patience,
        )
        wrapper._data_id = args.openml_id
        wrapper.fit(X_train, y_train, X_val, y_val)
        print("  학습 완료.")

        X_all = torch.cat([X_train, X_val, X_test], dim=0)
        y_all = torch.cat([
            y_train.long() if y_train.dtype != torch.long else y_train,
            y_val.long()   if y_val.dtype   != torch.long else y_val,
            y_test.long()  if y_test.dtype  != torch.long else y_test,
        ], dim=0)
        n_train = len(X_train)

        N = len(X_all)
        if N > args.max_samples:
            rng  = np.random.default_rng(args.seed)
            keep = np.sort(rng.choice(N, args.max_samples, replace=False))
            X_all   = X_all[keep]
            y_all   = y_all[keep]
            n_train = int((keep < len(X_train)).sum())
            print(f"  샘플 수 제한: {N} → {args.max_samples}")

        print("  임베딩 추출 중...")
        emb_data = extract_embeddings(model, X_all, y_all, device)

        print(f"  {args.proj.upper()} 투영 중...")
        X2d, C2d, explained = project(
            emb_data["emb"], emb_data["centroid_emb"],
            method=args.proj, seed=args.seed)

        y           = emb_data["y"]
        hard_assign = emb_data["hard_assign"]
        evidence_w  = emb_data["evidence_w"]
        topk_idx    = emb_data["topk_idx"]
        centroid_x  = emb_data["centroid_x"]

        with open(pkl_path, "wb") as f:
            pickle.dump(dict(
                X2d=X2d, C2d=C2d, y=y,
                hard_assign=hard_assign,
                evidence_w=evidence_w,
                topk_idx=topk_idx,
                n_train=n_train,
                explained=explained,
                dataset_name=dataset_name,
                col_names=col_names,
                centroid_x=centroid_x,
            ), f)
        print(f"  [저장] {pkl_path}")

    # ── 공통 kwargs ───────────────────────────────────────
    common = dict(
        method=args.proj, explained=explained,
        dataset_name=dataset_name,
        openml_id=oid, seed=args.seed,
        col_names=col_names, centroid_x=centroid_x,
    )

    # ══ Figure A ══════════════════════════════════════════
    print("\n  Figure A 생성 중...")
    draw_figure_A(
        X2d, y, C2d, hard_assign,
        out_path=os.path.join(
            out_dir, f"A_embed_{oid}_seed{args.seed}_{args.proj}.png"),
        **common)

    # ══ Figure B ══════════════════════════════════════════
    print("  Figure B 생성 중...")
    draw_figure_B(
        X2d, y, C2d, hard_assign,
        out_path=os.path.join(
            out_dir, f"B_centroid_{oid}_seed{args.seed}_{args.proj}.png"),
        **common)

    # ══ Figure C ══════════════════════════════════════════
    print("  Figure C 생성 중...")
    draw_figure_C(
        X2d, y, C2d, hard_assign, evidence_w, topk_idx,
        n_train=n_train, k_show=args.k_show,
        out_path=os.path.join(
            out_dir, f"C_retrieval_{oid}_seed{args.seed}_{args.proj}.png"),
        **common)

    print(f"\n완료!  {out_dir}/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
    