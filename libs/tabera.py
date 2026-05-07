"""
libs/tabera.py
============
TabERA — Dual-Space Prototype Explainable TabR Model.

MultiTab의 tabr.py를 대체하는 파일입니다.
아래 세 가지 구조적 혁신을 TabR 위에 통합합니다.

  2. CentroidLayer        : Dual-Space Prototype + STE Routing + EMA
  3. AttentionAggregator  : TabR 방식 similarity 기반 이웃 집계

Forward 흐름
------------
  X_query
    ↓  TabularEmbedder
  query_emb (B, D)
    ├─ CentroidLayer(query_emb) → context_emb (B, D) + routing + FAISS mask
    └─ MemoryBank.retrieve → k neighbours + neighbour_labels
         ↓ AttentionAggregator
       agg_emb (B, D) + evidence_weights (B, k)
    ↓
  [query_emb ‖ context_emb ‖ agg_emb] → PredictionHead → ŷ
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.prototypes import CentroidLayer, PrototypeLayer  # PrototypeLayer = CentroidLayer alias
from libs.evidence   import AttentionAggregator


# ─────────────────────────────────────────────────────────────
# 보조 블록
# ─────────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(x)


class TabularEmbedder(nn.Module):
    """수치형 특성을 공유 임베딩 공간으로 투영."""
    def __init__(self, n_features: int, embed_dim: int, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, embed_dim))
        self.blocks = nn.Sequential(*[ResidualMLP(embed_dim, embed_dim * 2, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class MemoryBank(nn.Module):
    """학습 임베딩 저장소 (KNN 검색용)."""
    def __init__(self, max_size: int, embed_dim: int):
        super().__init__()
        self.max_size = max_size
        self.register_buffer("keys",   torch.zeros(max_size, embed_dim))
        self.register_buffer("vals",   torch.zeros(max_size, embed_dim))
        self.register_buffer("labels", torch.zeros(max_size))
        self.register_buffer("ptr",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, keys, vals, labels):
        B   = keys.shape[0]
        ptr = self.ptr.item()
        end = min(ptr + B, self.max_size)
        n   = end - ptr
        self.keys[ptr:end]   = keys[:n].detach()
        self.vals[ptr:end]   = vals[:n].detach()
        self.labels[ptr:end] = labels[:n].float().detach()
        self.ptr    = torch.tensor(end % self.max_size, dtype=torch.long)
        self.filled = torch.tensor(min(self.filled.item() + n, self.max_size), dtype=torch.long)

    @torch.no_grad()
    def cache_sample_groups(
        self,
        sample_groups: "List[List[int]]",
        device: "torch.device",
    ) -> None:
        """
        sample_groups를 GPU 텐서로 미리 변환·캐시.
        EMA 업데이트 후 에폭당 1번만 호출 → retrieve 내부 변환 비용 제거.
        패딩: 가장 큰 그룹 크기에 맞춰 -1로 채움.
        """
        P   = len(sample_groups)
        max_g = max((len(g) for g in sample_groups), default=0)
        if max_g == 0:
            self._cached_groups      = None
            self._cached_group_sizes = None
            return

        # (P, max_g) 패딩 텐서 (-1 = 패딩)
        padded = torch.full((P, max_g), -1, dtype=torch.long)
        for p, g in enumerate(sample_groups):
            if g:
                padded[p, :len(g)] = torch.tensor(g, dtype=torch.long)

        self._cached_groups      = padded.to(device)         # (P, max_g)
        self._cached_group_sizes = torch.tensor(
            [len(g) for g in sample_groups], dtype=torch.long, device=device
        )                                                     # (P,)

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,                       # (B, D)
        k: int,
        hard_assignment: "Optional[torch.Tensor]" = None,
        sample_groups:   "Optional[List[List[int]]]" = None,  # 미사용 (캐시 우선)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        완전 벡터화 k-NN 검색.

        반환: (nk, nv, neighbour_labels, top_k_idx)
          nk              : (B, k, D) — 이웃 key 임베딩
          nv              : (B, k, D) — 이웃 value 임베딩
          neighbour_labels: (B, k)    — 이웃 레이블 (TabR 방식 value 구성용)
          top_k_idx       : (B, k)    — MemoryBank 내 실제 인덱스 (FeatureStore 조회용)
        """
        n   = self.filled.item()
        B   = query.shape[0]
        D   = query.shape[1]
        dev = query.device

        keys_full   = self.keys[:n]    # (n, D)
        vals_full   = self.vals[:n]    # (n, D)
        labels_full = self.labels[:n]  # (n,)
        q_norm      = F.normalize(query, dim=-1)  # (B, D)

        # ── 캐시 없거나 초기화 전 → 전체 검색 fallback ──────────
        cached = getattr(self, '_cached_groups', None)
        if hard_assignment is None or cached is None or n < k:
            keys_all = F.normalize(keys_full, dim=-1)
            sim      = q_norm @ keys_all.T
            _, idx   = sim.topk(min(k, n), dim=-1)
            idx      = idx.clamp(0, n - 1)
            neighbour_labels = labels_full[idx]              # (B, k)
            return keys_full[idx], vals_full[idx], neighbour_labels, idx

        # ── 완전 벡터화 (for loop 없음) ────────────────────────
        ha        = hard_assignment.to(dev)             # (B,)
        grp_sizes = self._cached_group_sizes[ha]        # (B,)

        # fallback 여부 판단: 그룹 크기 < k 인 샘플
        fallback_mask = grp_sizes < k                   # (B,) bool
        normal_mask   = ~fallback_mask

        # 결과 버퍼 (fallback 샘플은 zeros 유지)
        out_nk    = torch.zeros(B, k, D,          device=dev)
        out_nv    = torch.zeros(B, k, D,          device=dev)
        out_labels = torch.zeros(B, k,            device=dev)
        top_k_idx = torch.zeros(B, k, dtype=torch.long, device=dev)

        # ── 정상 샘플: 청크 단위 처리 (OOM 방지) ───────────────────
        if normal_mask.any():
            nm_idx   = normal_mask.nonzero(as_tuple=True)[0]  # (Bn,)
            ha_nm    = ha[nm_idx]                              # (Bn,)
            q_nm     = q_norm[nm_idx]                          # (Bn, D)

            cand_idx   = self._cached_groups[ha_nm]            # (Bn, max_g)
            max_g      = cand_idx.shape[1]
            valid_mask = (cand_idx >= 0)                       # (Bn, max_g)
            safe_idx   = cand_idx.clamp(min=0, max=n - 1)
            k_eff      = min(k, max_g)

            Bn = nm_idx.shape[0]

            # 청크 크기: max_g × D × 4bytes 기준 256MB 이하로 제한
            # 예) max_g=5000, D=256 → 5000×256×4 = 5MB/샘플 → chunk=50이면 250MB
            chunk = max(1, min(Bn, (256 * 1024 * 1024) // max(max_g * D * 4, 1)))

            i_final_all    = torch.zeros(Bn, k_eff, dtype=torch.long, device=dev)
            out_nk_chunk   = torch.zeros(Bn, k_eff, D, device=dev)
            out_nv_chunk   = torch.zeros(Bn, k_eff, D, device=dev)
            out_labels_chunk = torch.zeros(Bn, k_eff, device=dev)

            for start in range(0, Bn, chunk):
                end = min(start + chunk, Bn)
                sl  = slice(start, end)

                si_c  = safe_idx[sl]      # (c, max_g)
                vm_c  = valid_mask[sl]    # (c, max_g)
                q_c   = q_nm[sl]          # (c, D)

                keys_c = F.normalize(
                    keys_full[si_c.reshape(-1)].view(end - start, max_g, D),
                    dim=-1,
                )                                              # (c, max_g, D)

                sim_c = torch.bmm(
                    q_c.unsqueeze(1), keys_c.transpose(1, 2)
                ).squeeze(1)                                   # (c, max_g)
                sim_c = sim_c.masked_fill(~vm_c, -1e9)

                _, top_c   = sim_c.topk(k_eff, dim=-1)        # (c, k)
                i_final_c  = si_c.gather(1, top_c)            # (c, k)

                i_final_all[sl]      = i_final_c
                out_nk_chunk[sl]     = keys_full[i_final_c.reshape(-1)].view(end - start, k_eff, D)
                out_nv_chunk[sl]     = vals_full[i_final_c.reshape(-1)].view(end - start, k_eff, D)
                out_labels_chunk[sl] = labels_full[i_final_c.clamp(0, n - 1)]

            out_nk[nm_idx]     = out_nk_chunk
            out_nv[nm_idx]     = out_nv_chunk
            out_labels[nm_idx] = out_labels_chunk
            top_k_idx[nm_idx]  = i_final_all

        # fallback 샘플: zeros 유지 (초기 1~2 에폭만 해당)

        return out_nk, out_nv, out_labels, top_k_idx


class FeatureStore:
    """
    순수 설명용 저장소 — 검색과 완전 독립.

    가설 완결: "이웃 #1이 42% 기여(fixed_acidity=7.1 유사성)"에서
    fixed_acidity=7.1을 제공하는 모듈.

    설계 원칙
    ─────────
    - MemoryBank의 ptr과 동기화 (같은 순서로 저장)
    - 원본 X(역정규화 완료값) 저장
    - nn.Module이 아님 → gradient 없음, 순수 numpy/tensor 저장소
    - retrieve(top_k_indices) → X 값 dict 반환
    """

    def __init__(
        self,
        max_size: int,
        n_features: int,
        col_names: Optional[List[str]] = None,
    ) -> None:
        self.max_size   = max_size
        self.n_features = n_features
        self.col_names  = col_names or [f"f{i}" for i in range(n_features)]
        self._store  = torch.zeros(max_size, n_features)
        self._ptr    = 0
        self._filled = 0

    @torch.no_grad()
    def update(self, X_raw: torch.Tensor) -> None:
        B   = X_raw.shape[0]
        end = min(self._ptr + B, self.max_size)
        n   = end - self._ptr
        self._store[self._ptr:end] = X_raw[:n].detach().cpu().float()
        self._ptr    = end % self.max_size
        self._filled = min(self._filled + n, self.max_size)

    @torch.no_grad()
    def retrieve(self, indices: torch.Tensor) -> List[Dict[str, float]]:
        idx_cpu = indices.detach().cpu().clamp(0, self._filled - 1)
        if idx_cpu.dim() == 1:
            rows = self._store[idx_cpu]
            return [
                {self.col_names[fi]: float(rows[ki, fi]) for fi in range(self.n_features)}
                for ki in range(rows.shape[0])
            ]
        else:
            B, k = idx_cpu.shape
            result = []
            for b in range(B):
                rows = self._store[idx_cpu[b]]
                result.append([
                    {self.col_names[fi]: float(rows[ki, fi]) for fi in range(self.n_features)}
                    for ki in range(k)
                ])
            return result

    def top_features(self, sample_dict: Dict[str, float], n: int = 6) -> Dict[str, float]:
        return dict(sorted(sample_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n])

    def __repr__(self) -> str:
        return f"FeatureStore(max_size={self.max_size}, n_features={self.n_features}, filled={self._filled})"


# ─────────────────────────────────────────────────────────────
# TabERA
# ─────────────────────────────────────────────────────────────

class TabERA(nn.Module):
    """
    Parameters
    ----------
    n_features        : 입력 특성 수 (전처리 후)
    embed_dim         : 공유 임베딩 차원 D
    n_prototypes      : 프로토타입 수 P
    k                 : KNN 이웃 수
    prototype_labels  : 프로토타입 의미론적 이름
    n_output          : 출력 차원 (1: binary/regression, C: multiclass)
    memory_size       : 메모리 뱅크 최대 크기
    embedder_layers   : TabularEmbedder ResidualMLP 수
    dropout           : 전역 드롭아웃
    loss_weights      : 보조 손실 가중치 {'diversity': .., 'commitment': .., 'entropy': .., 'cohesion': ..}
                        cohesion=0.0 이면 Phase 1과 동일 (기본값, 하위 호환)
    column_names      : 특성 컬럼명 (설명 출력용)
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 128,
        n_prototypes: int = 8,
        k: int = 16,
        prototype_labels: Optional[List[str]] = None,
        n_output: int = 1,
        memory_size: int = 10_000,
        embedder_layers: int = 2,
        dropout: float = 0.1,
        loss_weights: Optional[Dict[str, float]] = None,
        column_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.k            = k
        self.embed_dim    = embed_dim
        self.n_output     = n_output
        self.n_features   = n_features
        self.loss_weights = loss_weights or {
            "diversity":  0.01,
            "commitment": 0.01,
            "entropy":    0.01,
        }
        self.column_names = column_names

        # ── 임베더 ──────────────────────────────────
        self.embedder = TabularEmbedder(n_features, embed_dim, embedder_layers, dropout)

        # ── CentroidLayer (Dual-Space Prototype) ────────
        self.prototype_layer = CentroidLayer(
            n_prototypes=n_prototypes,
            embed_dim=embed_dim,
            n_features=n_features,
            prototype_labels=prototype_labels,
            dropout=dropout,
            col_names=column_names,
        )

        # ── TabR 방식 이웃 집계 ───────────────────────
        self.ot_selector = AttentionAggregator(
            embed_dim=embed_dim,
            k=k,
            n_features=n_features,
            n_output=n_output,
            dropout=dropout,
        )

        # ── 메모리 뱅크 (검색 전용) ──────────────────
        self.memory = MemoryBank(memory_size, embed_dim)

        # ── FeatureStore (설명 전용) ──────────────────
        self._feature_store: Optional[FeatureStore] = None
        if column_names and n_features > 0:
            self._feature_store = FeatureStore(
                max_size=memory_size,
                n_features=n_features,
                col_names=column_names,
            )

        # ── 예측 헤드: [query ‖ context ‖ agg] → ŷ ──
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_output),
        )

    # ── Forward ────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,                          # (B, F)
        labels: Optional[torch.Tensor] = None,    # (B,) 학습 시 메모리 업데이트용
        return_explanations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # 1. 임베딩
        query_emb = self.embedder(X)               # (B, D)

        # 2. 프로토타입 라우팅 (CentroidLayer)
        # soft_probs: 순수 softmax 확률 — Phase 2 soft cohesion loss에 사용
        # routing_probs: STE (forward=hard, backward=soft gradient)
        context_emb, hard_assignment, routing_probs, soft_probs = self.prototype_layer(query_emb)

        # 3. 그룹별 부분 검색을 위한 sample_groups
        sample_groups = self.prototype_layer.sample_groups

        # 4. KNN 검색 + TabR 방식 집계
        if self.memory.filled.item() >= self.k:
            nk, nv, neighbour_labels, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                hard_assignment=hard_assignment,
                sample_groups=sample_groups,
            )
            agg_emb, evidence_w, feature_imp, attn_w = self.ot_selector(
                query_emb, nk, nv, neighbour_labels
            )
        else:
            agg_emb        = torch.zeros_like(query_emb)
            evidence_w     = torch.full((X.shape[0], self.k), 1.0 / self.k, device=X.device)
            feature_imp    = None
            attn_w         = None
            topk_idx       = torch.zeros(X.shape[0], self.k, dtype=torch.long, device=X.device)

        # 5. 예측
        combined = torch.cat([query_emb, context_emb, agg_emb], dim=-1)
        logits   = self.head(combined)

        # 6. 메모리 업데이트 (학습 시)
        if self.training and labels is not None:
            self.memory.update(query_emb.detach(), context_emb.detach(), labels.float())
            if self._feature_store is not None:
                self._feature_store.update(X)

        # 7. 보조 손실
        aux_loss = torch.tensor(0.0, device=X.device)
        if self.training:
            aux_loss = (
                self.loss_weights["diversity"]  * self.prototype_layer.diversity_loss()
                + self.loss_weights["commitment"] * self.prototype_layer.commitment_loss(query_emb, hard_assignment)
                + self.loss_weights.get("entropy", 0.01) * self.prototype_layer.entropy_loss(routing_probs)
            )

            # ── Phase 2: Soft Feature Cohesion Loss ─────────────────
            # 목적: soft routing probability가 feature-space prototype proximity와
            #        정렬되도록 유도 (embedding routing → feature space 해석 가능성 강화)
            #
            # 설계 원칙:
            #   - C_x는 EMA buffer (gradient 없음) → C_x.detach() 필수
            #   - soft_probs (진짜 softmax) 사용 → routing_probs(STE)와 구분
            #   - X는 이미 전처리된 normalized feature → scale 통일
            #   - / X.size(1) 로 feature 차원 정규화 → λ_cohesion 튜닝 안정화
            #
            # gradient 흐름:
            #   L_cohesion → soft_probs → logits → query_emb → ResidualMLP
            #                                     → centroid_emb
            #   (C_x는 EMA로만 갱신, gradient 없음)
            lw_cohesion = self.loss_weights.get("cohesion", 0.0)
            if lw_cohesion > 0 and self.prototype_layer.centroid_x is not None:
                C_x    = self.prototype_layer.centroid_x.detach()  # (P, F)
                dist_x = torch.cdist(X.float(), C_x.float(), p=2).pow(2) / X.size(1)  # (B, P)
                L_cohesion = (soft_probs * dist_x).sum(dim=1).mean()
                aux_loss = aux_loss + lw_cohesion * L_cohesion

        out = {
            "logits":      logits,
            "aux_loss":    aux_loss,
            "routing":     routing_probs,
            "hard_group":  hard_assignment,
            "evidence_w":  evidence_w,
            "feature_imp": feature_imp,
            "attn_w":      attn_w,
            "topk_idx":    topk_idx,
        }

        if return_explanations:
            proto_exp = self.prototype_layer.explain_routing(hard_assignment, routing_probs)
            ev_exp    = self.ot_selector.explain_evidence(evidence_w)
            feat_exp  = (
                self.ot_selector.explain_feature_match(
                    feature_imp, evidence_w,
                    self.column_names or [f"f{i}" for i in range(self.n_features)]
                )
                if feature_imp is not None else
                [None for _ in range(X.shape[0])]
            )
            out["explanations"] = [
                {
                    "prototype":     proto_exp[b],
                    "evidence":      ev_exp[b],
                    "feature_match": feat_exp[b],
                }
                for b in range(X.shape[0])
            ]

        return out

    @property
    def feature_store(self) -> Optional[FeatureStore]:
        return self._feature_store

    def anneal(self, factor: float = 0.95) -> None:
        self.prototype_layer.anneal(factor)

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        lines = ["=" * 48, "TabERA", "=" * 48,
                 f"  Parameters  : {total:,}",
                 f"  Embed dim   : {self.embed_dim}",
                 f"  Centroids   : {self.prototype_layer.P}",
                 f"  KNN k       : {self.k}",
                 f"  Dual-Space  : {'ON' if self.prototype_layer.F > 0 else 'OFF'}"]
        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        return "\n".join(lines)
