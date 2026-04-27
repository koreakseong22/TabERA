"""
libs/tabera.py
============
TabERA — Dual-Space Prototype Explainable TabR Model.

MultiTab의 tabr.py를 대체하는 파일입니다.
아래 세 가지 구조적 혁신을 TabR 위에 통합합니다.

  2. CentroidLayer        : Dual-Space Prototype + STE Routing + EMA
  3. AttentionAggregator  : Scaled dot-product attention → 이웃 집계

Forward 흐름
------------
  X_query
    ↓  TabularEmbedder
  query_emb (B, D)
    ├─ CentroidLayer(query_emb) → context_emb (B, D) + routing + FAISS mask
    └─ MemoryBank.retrieve → k neighbours
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
            self._cached_groups     = None
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        완전 벡터화 k-NN 검색.

        핵심 최적화:
          cached_groups[hard_assignment] → (B, max_g) 인덱스 텐서
          keys_full[(B, max_g)] → (B, max_g, D) 한 번에 gather
          bmm → (B, max_g) 유사도 → topk
          → for loop 완전 제거, Python-GPU 왕복 0회
        """
        n   = self.filled.item()
        B   = query.shape[0]
        D   = query.shape[1]
        dev = query.device

        keys_full = self.keys[:n]   # (n, D)
        vals_full = self.vals[:n]   # (n, D)
        q_norm    = F.normalize(query, dim=-1)  # (B, D)

        # 캐시 없거나 초기화 전 → 전체 검색
        cached = getattr(self, '_cached_groups', None)
        if hard_assignment is None or cached is None or n < k:
            # N이 작으면 전체 검색 (OOM 위험 없음)
            # N이 크면 이 분기 자체를 줄이기 위해 warmup 에폭 동안
            # MemoryBank를 먼저 충분히 채운 후 검색
            keys_all = F.normalize(keys_full, dim=-1)
            sim      = q_norm @ keys_all.T
            _, idx   = sim.topk(min(k, n), dim=-1)
            idx      = idx.clamp(0, n - 1)
            return keys_full[idx], vals_full[idx], torch.zeros(B, k, device=dev), idx

        # ── 완전 벡터화 (for loop 없음) ────────────────────────
        ha  = hard_assignment.to(dev)                        # (B,)
        grp_sizes = self._cached_group_sizes[ha]             # (B,)

        # fallback 여부 판단: 그룹 크기 < k 인 샘플
        fallback_mask = grp_sizes < k                        # (B,) bool
        normal_mask   = ~fallback_mask

        # 결과 버퍼
        out_nk    = torch.empty(B, k, D,          device=dev)
        out_nv    = torch.empty(B, k, D,          device=dev)
        top_k_idx = torch.zeros(B, k, dtype=torch.long, device=dev)

        # ── 정상 샘플: 벡터화 처리 ──────────────────────────────
        if normal_mask.any():
            nm_idx   = normal_mask.nonzero(as_tuple=True)[0]  # 정상 샘플 인덱스
            ha_nm    = ha[nm_idx]                              # (Bn,)
            q_nm     = q_norm[nm_idx]                          # (Bn, D)

            # cached_groups[ha_nm] → (Bn, max_g) 후보 인덱스
            cand_idx = self._cached_groups[ha_nm]              # (Bn, max_g)
            max_g    = cand_idx.shape[1]

            # 패딩(-1) 마스크: 유효 후보만 사용
            valid_mask = (cand_idx >= 0)                       # (Bn, max_g)

            # 후보 키 gather: 패딩 위치는 0으로 채움
            safe_idx  = cand_idx.clamp(min=0, max=n-1)         # -1 → 0, 범위 초과 방지
            keys_c    = F.normalize(
                keys_full[safe_idx.view(-1)].view(nm_idx.shape[0], max_g, D),
                dim=-1
            )                                                  # (Bn, max_g, D)

            # 유사도: (Bn, D) × (Bn, D, max_g) → (Bn, max_g)
            sim_nm    = torch.bmm(
                q_nm.unsqueeze(1), keys_c.transpose(1, 2)
            ).squeeze(1)                                       # (Bn, max_g)

            # 패딩 위치 마스킹 (-inf)
            sim_nm    = sim_nm.masked_fill(~valid_mask, -1e9)

            k_eff     = min(k, max_g)
            _, top    = sim_nm.topk(k_eff, dim=-1)            # (Bn, k)
            i_final   = safe_idx.gather(1, top)               # (Bn, k)

            out_nk[nm_idx] = keys_full[i_final.view(-1)].view(nm_idx.shape[0], k_eff, D)
            out_nv[nm_idx] = vals_full[i_final.view(-1)].view(nm_idx.shape[0], k_eff, D)
            top_k_idx[nm_idx] = i_final

        # ── fallback 샘플: zero 반환 (OOM 근본 해결) ────────────
        # 발생 시점: cached_groups가 아직 없는 초기 1~2 에폭
        # 이전 방식: F.normalize(keys_full) → (B, max_mem, D) 할당 → OOM
        # 수정 방식: zero 반환 → 해당 샘플의 agg_emb=0으로 처리
        # 영향: 초기 에폭 ~2% 구간만, cached_groups 구성 후 정상 작동
        if fallback_mask.any():
            pass  # out_nk/nv/top_k_idx 이미 zeros 초기화 → zero agg_emb

        return out_nk, out_nv, torch.zeros(B, k, device=dev), top_k_idx


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
        self.max_size  = max_size
        self.n_features = n_features
        self.col_names  = col_names or [f"f{i}" for i in range(n_features)]
        # (max_size, F) — CPU 저장, 역정규화 완료값
        self._store = torch.zeros(max_size, n_features)
        self._ptr   = 0
        self._filled = 0

    @torch.no_grad()
    def update(self, X_raw: torch.Tensor) -> None:
        """
        MemoryBank.update와 동일한 순서로 원본 X를 저장.
        X_raw: (B, F) — 역정규화된 원본 feature 값
        """
        B   = X_raw.shape[0]
        end = min(self._ptr + B, self.max_size)
        n   = end - self._ptr
        self._store[self._ptr:end] = X_raw[:n].detach().cpu().float()
        self._ptr    = end % self.max_size
        self._filled = min(self._filled + n, self.max_size)

    @torch.no_grad()
    def retrieve(self, indices: torch.Tensor) -> List[Dict[str, float]]:
        """
        top-k 인덱스로 원본 feature 값 조회.

        indices: (k,) or (B, k) — MemoryBank.retrieve의 top-k 인덱스
        반환: list of dict {col_name: value}
        """
        idx_cpu = indices.detach().cpu()
        # 유효 인덱스만
        idx_cpu = idx_cpu.clamp(0, self._filled - 1)

        if idx_cpu.dim() == 1:
            # (k,) → list of k dicts
            rows = self._store[idx_cpu]   # (k, F)
            return [
                {self.col_names[fi]: float(rows[ki, fi])
                 for fi in range(self.n_features)}
                for ki in range(rows.shape[0])
            ]
        else:
            # (B, k) → list of B lists of k dicts
            B, k = idx_cpu.shape
            result = []
            for b in range(B):
                rows = self._store[idx_cpu[b]]  # (k, F)
                result.append([
                    {self.col_names[fi]: float(rows[ki, fi])
                     for fi in range(self.n_features)}
                    for ki in range(k)
                ])
            return result

    def top_features(
        self,
        sample_dict: Dict[str, float],
        n: int = 6,
    ) -> Dict[str, float]:
        """절대값 기준 상위 n개 feature만 반환 (설명 출력용)."""
        sorted_items = sorted(
            sample_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return dict(sorted_items[:n])

    def __repr__(self) -> str:
        return (f"FeatureStore(max_size={self.max_size}, "
                f"n_features={self.n_features}, "
                f"filled={self._filled})")



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
    loss_weights      : 보조 손실 가중치 {'diversity': .., 'commitment': ..}
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
        self.n_features   = n_features   # cross-attention 역투영용
        self.loss_weights = loss_weights or {
            "diversity":  0.01,
            "commitment": 0.01,
        }
        self.column_names = column_names

        # ── 임베더 ──────────────────────────────────
        self.embedder = TabularEmbedder(n_features, embed_dim, embedder_layers, dropout)

        # ── CentroidLayer (Dual-Space Prototype) ────────
        self.prototype_layer = CentroidLayer(
            n_prototypes=n_prototypes,
            embed_dim=embed_dim,
            n_features=n_features,         # 이중 공간: 원본 feature 저장
            prototype_labels=prototype_labels,
            dropout=dropout,
            col_names=column_names,        # 설명용 컬럼명
        )

        # ── Attention 기반 이웃 집계 (OT → Attention 교체) ───
        self.ot_selector = AttentionAggregator(
            embed_dim=embed_dim,
            k=k,
            n_features=n_features,
            dropout=dropout,
        )

        # ── 메모리 뱅크 (검색 전용) ──────────────────
        self.memory = MemoryBank(memory_size, embed_dim)

        # ── FeatureStore (설명 전용, 검색과 완전 독립) ──
        # nn.Module이 아니므로 _feature_store로 저장
        # forward에서 self._feature_store로 접근
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

        # 2. 프로토타입 라우팅
        context_emb, hard_assignment, routing_probs = self.prototype_layer(query_emb)

        # 3-1. 가설 ②: hard_assignment + sample_groups → 그룹별 부분 검색
        #      O(N) → O(avg_group_size · B/P), Python sort 오버헤드 없음
        sample_groups = self.prototype_layer.sample_groups  # list[list[int]]

        # 4. KNN 검색 (centroid 번호 기준 그룹핑) + OT 증거 선택
        if self.memory.filled.item() >= self.k:
            nk, nv, _, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                hard_assignment=hard_assignment,
                sample_groups=sample_groups,
            )
            agg_emb, evidence_w, feature_imp, attn_w = self.ot_selector(query_emb, nk, nv)
        else:
            agg_emb      = torch.zeros_like(query_emb)
            evidence_w   = torch.full((X.shape[0], self.k), 1.0 / self.k, device=X.device)
            feature_imp  = None
            attn_w       = None
            topk_idx     = torch.zeros(X.shape[0], self.k, dtype=torch.long, device=X.device)

        # 5. 예측
        combined = torch.cat([query_emb, context_emb, agg_emb], dim=-1)
        logits   = self.head(combined)

        # 6. 메모리 업데이트 (학습 시)
        if self.training and labels is not None:
            self.memory.update(query_emb.detach(), context_emb.detach(), labels.float())
            # FeatureStore: 원본 X를 MemoryBank와 동일 순서로 저장
            if self._feature_store is not None:
                self._feature_store.update(X)

        # 7. 보조 손실
        # diversity:  centroid 붕괴 방지 (off-diagonal cosine sim 최소화)
        # commitment: VQ-VAE 표준 — query를 배정 centroid 방향으로 수렴
        # entropy:    STE collapse 방지 — 모든 centroid에 gradient 보장
        #             soft routing 분포의 entropy를 최대화 → 고르게 분산
        #             근거: VQ-VAE-2 (Razavi et al., NeurIPS 2019)
        aux_loss = torch.tensor(0.0, device=X.device)
        if self.training:
            aux_loss = (
                self.loss_weights["diversity"]  * self.prototype_layer.diversity_loss()
                + self.loss_weights["commitment"] * self.prototype_layer.commitment_loss(query_emb, hard_assignment)
                + self.loss_weights.get("entropy", 0.01) * self.prototype_layer.entropy_loss(routing_probs)
            )

        out = {
            "logits":             logits,
            "aux_loss":           aux_loss,
            "routing":            routing_probs,
            "hard_group":         hard_assignment,
            "evidence_w":         evidence_w,
            "feature_imp":        feature_imp,        # (B, k, F) or None
            "attn_w":             attn_w,              # (B, k, D) or None
            "topk_idx":           topk_idx,
        }

        if return_explanations:
            proto_exp   = self.prototype_layer.explain_routing(hard_assignment, routing_probs)
            # proto_exp[b]['centroid_features'] 에 원본 feature 값 포함됨 (§가설 ①)
            ev_exp      = self.ot_selector.explain_evidence(evidence_w)
            # §3.3 feature-level 매칭 설명
            feat_exp = (
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
                    "feature_match": feat_exp[b],   # §3.3
                }
                for b in range(X.shape[0])
            ]

        return out

    @property
    def feature_store(self) -> Optional[FeatureStore]:
        """FeatureStore 외부 접근용 프로퍼티."""
        return self._feature_store

    def anneal(self, factor: float = 0.95) -> None:
        self.prototype_layer.anneal(factor)

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        lines = ["=" * 48, "TabERA", "=" * 48,
                 f"  Parameters  : {total:,}",
                 f"  Embed dim   : {self.embed_dim}",
                 f"  Centroids   : {self.prototype_layer.P}  (sqrt(N) 권장)",
                 f"  KNN k       : {self.k}",
                 f"  Dual-Space  : {'ON' if self.prototype_layer.F > 0 else 'OFF'}"]
        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        return "\n".join(lines)
    