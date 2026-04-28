"""
libs/prototypes.py
==================
CentroidLayer — Dual-Space Prototype Representation
    (이중 공간 프로토타입 기반 계층적 설명가능 검색 레이어)

가설 핵심 구현
──────────────
(1) 이중 공간 저장
    - centroid_emb  (P, D) : 임베딩 공간 — STE routing + FAISS 마스킹용
    - centroid_x    (P, F) : 원본 feature 공간 — 역정규화 없이 직접 해석 가능
    → "Centroid #3: alcohol=10.2, pH=3.3" 형태의 즉각적 설명

(2) Straight-Through Estimator (STE) Hard Routing + FAISS 범위 제한
    - O(N) → O(P + k·log k) 복잡도 개선
    - 배정된 centroid 그룹 내 샘플 인덱스만 FAISS 검색 후보로 제한
    - train: Straight-Through Estimator (STE) / eval: argmax hard
    - STE: forward=argmax(hard), backward=softmax gradient 통과
    - 근거: VQ-VAE(van den Oord, 2017) 표준 설계 + commitment loss

(3) EMA 기반 centroid 업데이트
    - 매 에폭 후 배정된 샘플 임베딩·원본 X의 평균으로 갱신
    - centroid_emb_new = τ * centroid_emb + (1-τ) * batch_mean_emb
    - centroid_x_new   = τ * centroid_x   + (1-τ) * batch_mean_x
    - 학습 초기 고정(freeze) → epoch > warmup 후 활성화

이론적 근거
───────────
- Dual-Space Prototype Representation (본 가설)
- Straight-Through Estimator (Bengio et al. 2013)
- VQ-VAE hard assignment trick (van den Oord et al. 2017)
- ODC EMA centroid update (Zhan et al. 2020)

하위 호환성
───────────
PrototypeLayer = CentroidLayer  (alias 유지, 기존 tabr.py 수정 불필요)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CentroidLayer(nn.Module):
    """
    이중 공간 프로토타입 표현 레이어.

    Parameters
    ──────────
    n_prototypes      : 초기 centroid 수 P\n                        (학습 중 비활성 centroid pruning이 적용되므로,\n                         예상 군집 수보다 다소 크게 시작하는 것을 권장)
    embed_dim         : 임베딩 차원 D
    n_features        : 원본 feature 수 F (이중 공간 저장용)
    prototype_labels  : centroid 의미론적 이름 (없으면 "Centroid_i" 자동 생성)
    ema_momentum      : EMA 업데이트 모멘텀 (0.9~0.99 권장)
    ema_warmup_epochs : 이 에폭 이후부터 EMA 활성화 (기본 0 = 즉시)
    dropout           : 컨텍스트 벡터 드롭아웃
    col_names         : 원본 feature 컬럼명 (설명 출력용)
    """

    def __init__(
        self,
        n_prototypes: int,
        embed_dim: int,
        n_features: int = 0,
        prototype_labels: Optional[List[str]] = None,
        ema_momentum: float = 0.95,
        ema_warmup_epochs: int = 0,   # 즉시 활성화 (warmup 없음)
        prune_threshold: float = 0.005,
        prune_patience: int = 5,
        min_active_centroids: int = 2,
        dropout: float = 0.0,
        col_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.P                 = n_prototypes
        self.D                 = embed_dim
        self.F                 = n_features
        self.ema_momentum      = ema_momentum
        self.ema_warmup_epochs = ema_warmup_epochs
        self.prune_threshold   = prune_threshold
        self.prune_patience    = prune_patience
        self.min_active_centroids = max(1, min_active_centroids)
        self.col_names         = col_names or [f"f{i}" for i in range(n_features)]

        # ── 온도 (register_buffer: 저장되지만 gradient 없음) ──
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))

        # ── (1) 이중 공간 centroid ────────────────────────────
        # 임베딩 공간: 학습 가능 파라미터 (routing + FAISS 마스킹)
        self.centroid_emb = nn.Parameter(torch.empty(n_prototypes, embed_dim))
        nn.init.orthogonal_(self.centroid_emb)

        # 원본 feature 공간: buffer (학습 안 됨, EMA로만 갱신)
        if n_features > 0:
            self.register_buffer("centroid_x", torch.zeros(n_prototypes, n_features))
            self.register_buffer("centroid_x_initialized", torch.tensor(False))
        else:
            self.centroid_x = None
            self.centroid_x_initialized = None

        # ── centroid별 샘플 인덱스 그룹 (FAISS 범위 제한용) ──
        # list of lists: sample_groups[p] = [idx, idx, ...]
        self.sample_groups: Optional[List[List[int]]] = None

        # ── centroid별 평균 레이블 (Rank-Consistency Loss용) ──
        self.register_buffer(
            'centroid_labels',
            torch.full((n_prototypes,), float('nan'))
        )
        self.register_buffer("usage_ema", torch.zeros(n_prototypes))
        self.register_buffer("inactive_streak", torch.zeros(n_prototypes, dtype=torch.long))
        self.register_buffer("active_mask", torch.ones(n_prototypes, dtype=torch.bool))

        # ── 레이블 ────────────────────────────────────────────
        self.labels = prototype_labels or [f"Centroid_{i}" for i in range(n_prototypes)]

        self.dropout = nn.Dropout(dropout)

    # ─────────────────────────────────────────────────────────
    # 초기화: 훈련 데이터로 centroid 설정
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize_from_data(
        self,
        X_emb: torch.Tensor,             # (N, D) 훈련 임베딩
        X_raw: Optional[torch.Tensor] = None,   # (N, F) 원본 feature
        y_labels: Optional[torch.Tensor] = None, # (N,) 레이블 (소수 클래스 보장용)
    ) -> None:
        """
        KMeans++ 스타일 초기화.

        Arthur & Vassilvitskii (SODA 2007) "k-means++: The Advantages of
        Careful Seeding"의 거리 기반 확률적 시딩을 구현합니다.

        알고리즘
        ────────
        1. 첫 centroid: 균등 무작위 선택
        2. 이후 각 centroid: 기존 centroid와의 최소 거리²에 비례한
           확률로 다음 centroid 선택
           → 멀리 있는 점이 선택될 확률이 높아져 dead centroid 방지

        y_labels가 주어지면 소수 클래스 보장
        ──────────────────────────────────
        P개 선택 후에도 특정 클래스가 대표 centroid를 갖지 못하면
        해당 클래스의 대표 샘플(클래스 평균에 가장 가까운 샘플)로
        마지막 centroid를 대체합니다.
        """
        N   = X_emb.shape[0]
        dev = X_emb.device
        X_n = F.normalize(X_emb.float(), dim=-1)  # (N, D) 정규화

        # ── Step 1: KMeans++ 시딩 ────────────────────────────────
        selected_idx = []

        # 첫 centroid: 균등 무작위
        first = torch.randint(N, (1,), device=dev).item()
        selected_idx.append(first)

        for _ in range(self.P - 1):
            ctrs  = X_n[torch.tensor(selected_idx, device=dev)]
            sims  = X_n @ ctrs.T
            max_sim, _ = sims.max(dim=1)
            dists_sq = (1.0 - max_sim).clamp(min=0.0) ** 2
            dists_sq[torch.tensor(selected_idx, device=dev)] = 0.0
            if dists_sq.sum() < 1e-10:
                nxt = torch.randint(N, (1,), device=dev).item()
            else:
                nxt = torch.multinomial(dists_sq, 1).item()
            selected_idx.append(nxt)

        # ── Step 2: 소수 클래스 보장 (y_labels 있을 때) ────────────
        if y_labels is not None:
            y_cpu = y_labels.cpu()
            unique_cls = y_cpu.unique().tolist()
            sel_labels = y_cpu[torch.tensor(selected_idx)].tolist()

            for cls in unique_cls:
                if cls not in sel_labels:
                    # 해당 클래스 샘플들의 평균 임베딩에 가장 가까운 샘플
                    cls_mask = (y_cpu == cls).nonzero(as_tuple=True)[0]
                    cls_emb  = X_n[cls_mask.to(dev)].mean(0, keepdim=True)
                    dists    = 1.0 - (X_n[cls_mask.to(dev)] @ cls_emb.T).squeeze()
                    rep_idx  = cls_mask[dists.argmin().item()].item()
                    # 가장 큰 클러스터에 해당하는 마지막 selected를 교체
                    sel_t    = torch.tensor(selected_idx)
                    sel_lbl  = y_cpu[sel_t]
                    majority = sel_lbl.long().bincount().argmax().item()
                    maj_pos  = (sel_lbl == majority).nonzero(as_tuple=True)[0]
                    if len(maj_pos) > 1:
                        selected_idx[maj_pos[-1].item()] = rep_idx

        # ── Step 3: centroid 등록 ────────────────────────────────
        idx_t = torch.tensor(selected_idx, device=dev)
        self.centroid_emb.data = X_n[idx_t]

        if X_raw is not None and self.centroid_x is not None:
            self.centroid_x.data = X_raw[idx_t].float()
            self.centroid_x_initialized.fill_(True)
        self.usage_ema.zero_()
        self.inactive_streak.zero_()
        self.active_mask.fill_(True)

        self.sample_groups = [[] for _ in range(self.P)]

        # 초기화 품질 로그: centroid 간 평균 코사인 거리
        sim_mat  = self.centroid_emb.data @ self.centroid_emb.data.T
        mask     = ~torch.eye(self.P, dtype=torch.bool, device=dev)
        avg_dist = (1.0 - sim_mat[mask]).mean().item()
        print(f"  [CentroidLayer] KMeans++ {self.P} centroids "
              f"from {N} samples. avg_inter_dist={avg_dist:.3f}")

    # ─────────────────────────────────────────────────────────
    # (3) EMA 업데이트 (에폭 종료 후 호출)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ema_update(
        self,
        X_emb: torch.Tensor,        # (N, D) 전체 훈련 임베딩 — assignment 계산용
        X_raw: Optional[torch.Tensor] = None,   # (N, F) 원본 feature — centroid_x EMA용
        assignments: Optional[torch.Tensor] = None,  # (N,) hard assignment
    ) -> Dict[str, float]:
        """
        centroid_x만 EMA로 갱신합니다 (설명용 원본 feature 공간).
        centroid_emb는 gradient + commitment loss로 학습되므로 EMA 불필요.

        변경 이유
        ─────────
        centroid_emb EMA와 commitment loss가 같은 방향으로 중복 작동.
        EMA 제거 → gradient 신호 단일화 → 학습 속도 향상 + 신호 명확화.
        centroid_x는 gradient가 없으므로 EMA가 유일한 갱신 수단 → 유지.

        유지되는 기능
        ─────────────
        - centroid_x EMA: 설명 품질 보장
        - dead centroid 재초기화: 학습 안정성
        - sample_groups 갱신: 캐시 기반 검색

        Returns
        ───────
        stats: {"active_ratio": float, "min_cluster_size": int, "max_cluster_size": int}
        """
        epoch = self.current_epoch.item()
        if epoch < self.ema_warmup_epochs:
            return {"active_ratio": 0.0, "min_cluster_size": 0, "max_cluster_size": 0}

        if assignments is None:
            # 현재 centroid 기준으로 재배정
            q = F.normalize(X_emb.float(), dim=-1)
            c = F.normalize(self.centroid_emb, dim=-1)
            assignments = (q @ c.T).argmax(dim=-1)

        m = self.ema_momentum
        sizes = []
        active = 0
        new_groups: List[List[int]] = [[] for _ in range(self.P)]

        for p in range(self.P):
            mask = (assignments == p)
            size = int(mask.sum().item())
            sizes.append(size)
            # nonzero → 1D 리스트로 변환
            nz = mask.nonzero(as_tuple=True)[0]
            new_groups[p] = nz.tolist()

            if size == 0:
                continue
            active += 1

            # centroid_emb EMA 제거 → gradient + commitment loss로 학습
            # centroid_x EMA 유지 → 설명용 원본 feature 공간 (gradient 없음)
            if X_raw is not None and self.centroid_x is not None:
                mean_x = X_raw[mask].float().mean(dim=0)
                self.centroid_x.data[p] = m * self.centroid_x.data[p] + (1 - m) * mean_x

        # ── Usage EMA 기반 dead centroid pruning ───────────────
        n_samples = max(int(assignments.numel()), 1)
        batch_usage = torch.tensor(sizes, device=self.usage_ema.device, dtype=self.usage_ema.dtype) / n_samples
        self.usage_ema.mul_(m).add_((1 - m) * batch_usage)

        low_usage = self.usage_ema < self.prune_threshold
        self.inactive_streak[low_usage] += 1
        self.inactive_streak[~low_usage] = 0

        prune_cand = (self.inactive_streak >= self.prune_patience) & self.active_mask
        active_count = int(self.active_mask.sum().item())
        max_prunable = max(0, active_count - self.min_active_centroids)
        pruned_this_epoch = 0
        if prune_cand.any() and max_prunable > 0:
            cand_idx = prune_cand.nonzero(as_tuple=True)[0]
            order = torch.argsort(self.usage_ema[cand_idx])  # 사용률 낮은 순
            to_prune = cand_idx[order[:max_prunable]]
            self.active_mask[to_prune] = False
            self.inactive_streak[to_prune] = 0
            pruned_this_epoch = int(to_prune.numel())

        for p in range(self.P):
            if not self.active_mask[p]:
                new_groups[p] = []

        self.sample_groups = new_groups
        self.current_epoch += 1

        # 통계
        n_assigned = sum(1 for s in sizes if s > 0)

        return {
            "active_ratio":     n_assigned / self.P,
            "active_centroids": int(self.active_mask.sum().item()),
            "pruned_this_epoch": pruned_this_epoch,
            "min_cluster_size": int(min(sizes)) if sizes else 0,
            "max_cluster_size": int(max(s for s in sizes if s > 0)) if any(s > 0 for s in sizes) else 0,
        }

    # ─────────────────────────────────────────────────────────
    # 온도 어닐링 (에폭마다 호출)
    # ─────────────────────────────────────────────────────────

    def anneal(self, factor: Optional[float] = None) -> None:
        """
        (하위 호환) 온도 어닐링 인터페이스.
        factor가 None이면 tau_anneal_rate 기반 지수 감소.
        """
        pass  # STE 전환으로 annealing 불필요

    # ─────────────────────────────────────────────────────────
    # FAISS 마스킹용 인덱스 반환
    # ─────────────────────────────────────────────────────────

    def get_candidate_indices(
        self,
        hard_assignment: torch.Tensor,  # (B,)
        max_candidates: int = 5000,
    ) -> Optional[List[List[int]]]:
        """
        배정된 centroid 그룹의 샘플 인덱스를 반환합니다.
        FAISS 검색 범위를 해당 그룹으로 제한하는 데 사용합니다.
        O(N) → O(P + k·log k) 복잡도 개선의 핵심.

        Returns None if sample_groups not yet initialized.
        """
        if self.sample_groups is None:
            return None

        B = hard_assignment.shape[0]
        result = []
        for b in range(B):
            p = hard_assignment[b].item()
            grp = self.sample_groups[p]
            if len(grp) == 0:
                result.append(None)  # 빈 그룹: 전체 검색으로 fallback
            else:
                result.append(grp[:max_candidates])
        return result

    # ─────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        query_emb: torch.Tensor,                          # (B, D)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        ───────
        context_emb    : (B, D)  — centroid 혼합 컨텍스트 (미분 가능)
        hard_assignment: (B,)    — Top-1 centroid 인덱스 (FAISS 마스킹 + 설명용)
        routing_probs  : (B, P)  — STE용 soft routing 확률
        """
        # 코사인 유사도 로짓
        q = F.normalize(query_emb, dim=-1)               # (B, D)
        c = F.normalize(self.centroid_emb, dim=-1)        # (P, D)
        logits = q @ c.T                                  # (B, P)

        # 가설 바이어스 주입

        # (2) STE routing (Bengio et al., 2013 + VQ-VAE, van den Oord et al., 2017)
        # forward: hard argmax (이산, 결정론적)
        # backward: softmax gradient 통과
        # collapse 방지: entropy loss (VQ-VAE-2, Razavi et al., NeurIPS 2019)
        hard_assignment = logits.argmax(dim=-1)              # (B,)
        hard_one_hot = F.one_hot(hard_assignment, self.P).float()  # (B, P)
        soft = F.softmax(logits, dim=-1)                     # (B, P)

        if self.training:
            routing_probs = soft + (hard_one_hot - soft).detach()
        else:
            routing_probs = hard_one_hot

        # 컨텍스트: routing_probs로 centroid 혼합
        context_emb = self.dropout(routing_probs @ self.centroid_emb)  # (B, D)

        return context_emb, hard_assignment, routing_probs

    # ─────────────────────────────────────────────────────────
    # Auxiliary Losses (기존 tabr.py 호환)
    # ─────────────────────────────────────────────────────────

    def diversity_loss(self) -> torch.Tensor:
        """Centroid 붕괴 방지: off-diagonal cosine similarity 최소화.
        clamp(max=1e4): STE collapse 시 nan 전파 방지
        """
        c = F.normalize(self.centroid_emb, dim=-1)
        sim = c @ c.T
        mask = 1.0 - torch.eye(self.P, device=sim.device)
        loss = (sim.pow(2) * mask).sum() / (self.P * (self.P - 1))
        return loss.clamp(max=1e4)  # nan 방지

    def entropy_loss(self, routing_probs: torch.Tensor) -> torch.Tensor:
        """
        Codebook utilization 향상 — 배정 분포의 entropy 최대화.

        근거: VQ-VAE-2 (Razavi et al., NeurIPS 2019)
        배치 내 평균 routing 분포의 entropy를 최대화하여
        모든 centroid가 고르게 사용되도록 유도.

        routing_probs: (B, P) — STE output (soft 값 보유, gradient 연결됨)
        soft collapse 시 avg_probs 편중 → entropy 낮음 → loss 높음
        → gradient가 centroid_emb를 분산 방향으로 당김
        """
        avg_probs = routing_probs.mean(dim=0)              # (P,) 배치 평균
        entropy   = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        return -entropy  # entropy 최대화 → loss 최소화

    def rank_consistency_loss(
        self,
        centroid_labels: Optional[torch.Tensor],  # (P,) 각 centroid의 평균 레이블
        margin: float = 0.1,
    ) -> torch.Tensor:
        """
        Rank-Consistency Loss.

        제언: 임베딩 공간에서 centroid가 레이블 순서대로
        일직선(manifold) 위에 놓이도록 제약합니다.

        핵심 아이디어
        ─────────────
        레이블 순서: label_i < label_j 이면
        임베딩 거리 d(c_i, c_k) < d(c_i, c_j) (k가 i에 더 가까운 레이블)

        구현: Triplet-style
          anchor   = centroid_a
          positive = 레이블상 anchor에 더 가까운 centroid_p
          negative = 레이블상 anchor에서 더 먼 centroid_n
          loss = max(0, d(a,p) - d(a,n) + margin)

        centroid_labels가 None이면 0 반환 (초기 에폭 안전 처리)
        """
        if centroid_labels is None or self.P < 3:
            return torch.tensor(0.0, device=self.centroid_emb.device)

        c    = F.normalize(self.centroid_emb, dim=-1)  # (P, D)
        dist = 1.0 - c @ c.T                           # (P, P) cosine distance
        lbl  = centroid_labels.float()                 # (P,)

        # 레이블 차이 행렬
        lbl_diff = (lbl.unsqueeze(0) - lbl.unsqueeze(1)).abs()  # (P, P)

        loss = torch.tensor(0.0, device=self.centroid_emb.device)
        n_triplets = 0

        # triplet 샘플링: 각 anchor a에 대해
        # positive p = a와 레이블이 가장 가까운 centroid
        # negative n = a와 레이블이 가장 먼 centroid
        for a in range(self.P):
            sorted_by_lbl = lbl_diff[a].argsort()  # 레이블 거리 오름차순
            # positive: 레이블 거리 2번째 (0번째는 자기 자신)
            p = sorted_by_lbl[1].item()
            # negative: 레이블 거리 가장 큰 것
            n = sorted_by_lbl[-1].item()
            if p == n:
                continue
            triplet = dist[a, p] - dist[a, n] + margin
            loss = loss + torch.clamp(triplet, min=0.0)
            n_triplets += 1

        return loss / max(n_triplets, 1)

    def cosine_similarity_matrix(self) -> torch.Tensor:
        """진단용: centroid 간 cosine similarity 행렬 반환 (P, P)."""
        c = F.normalize(self.centroid_emb.detach(), dim=-1)
        return (c @ c.T).cpu()

    def commitment_loss(
        self, query_emb: torch.Tensor, hard_assignment: torch.Tensor
    ) -> torch.Tensor:
        """쿼리를 배정된 centroid 방향으로."""
        assigned = self.centroid_emb[hard_assignment]
        return F.mse_loss(query_emb, assigned.detach())

    # ─────────────────────────────────────────────────────────
    # 설명 헬퍼 (기존 tabr.py 호환 + 원본 feature 값 추가)
    # ─────────────────────────────────────────────────────────

    def explain_routing(
        self,
        hard_assignment: torch.Tensor,   # (B,)
        routing_probs: torch.Tensor,     # (B, P)
        norm_mean: Optional[np.ndarray] = None,  # (F,) 역정규화용
        norm_std:  Optional[np.ndarray] = None,  # (F,) 역정규화용
    ) -> List[dict]:
        """
        샘플별 centroid 배정 설명.
        norm_mean/norm_std가 주어지면 centroid_x를 역정규화하여
        원본 feature 값으로 출력합니다.
        예: alcohol=-0.816 → alcohol=10.24
        """
        pa   = hard_assignment.detach().cpu().numpy()
        pr   = routing_probs.detach().cpu().numpy()

        # centroid_x 역정규화: x_original = x_norm * std + mean
        if self.centroid_x is not None:
            cx = self.centroid_x.detach().cpu().numpy()  # (P, F)
            if norm_mean is not None and norm_std is not None:
                cx = cx * norm_std + norm_mean  # 역정규화
        else:
            cx = None

        out  = []
        ncol = min(len(self.col_names), self.F) if self.F > 0 else 0

        for b in range(pa.shape[0]):
            p     = int(pa[b])
            label = self.labels[p]
            conf  = float(pr[b, p])

            runners = sorted(
                [(self.labels[i], float(pr[b, i]))
                 for i in range(self.P) if i != p],
                key=lambda x: -x[1],
            )

            # 원본 feature 값 (이중 공간의 핵심)
            centroid_features: Dict[str, float] = {}
            if cx is not None and ncol > 0:
                for fi in range(ncol):
                    centroid_features[self.col_names[fi]] = float(cx[p, fi])

            out.append({
                "assigned_group":    label,
                "centroid_idx":      p,
                "group_confidence":  conf,
                "runners_up":        runners[:2],
                "centroid_features": centroid_features,  # ← 이중 공간 설명
            })
        return out

    def centroid_summary(self, top_n: int = 3) -> str:
        """
        전체 centroid의 원본 feature 평균값을 요약 출력.
        역정규화 없이 직접 해석 가능한 형태.
        """
        lines = [f"CentroidLayer — {self.P} centroids", "─" * 44]
        cx = (self.centroid_x.detach().cpu().numpy()
              if self.centroid_x is not None else None)
        ncol = min(len(self.col_names), self.F, top_n * 2) if self.F > 0 else 0

        for p in range(self.P):
            grp_size = (len(self.sample_groups[p])
                        if self.sample_groups else "?")
            line = f"  [{self.labels[p]}]  n={grp_size}"
            if cx is not None and ncol > 0:
                vals = ", ".join(
                    f"{self.col_names[fi]}={cx[p, fi]:.3f}"
                    for fi in range(ncol)
                )
                line += f"  {vals}"
            lines.append(line)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 하위 호환 alias (tabera.py import 수정 불필요)
# ─────────────────────────────────────────────────────────────
PrototypeLayer = CentroidLayer
