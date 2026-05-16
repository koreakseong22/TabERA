"""
libs/tabera.py
============
TabERA — Dual-Space Prototype Explainable TabR Model.

MultiTab의 tabr.py를 대체하는 파일입니다.

  CentroidLayer        : Dual-Space Prototype + STE Routing + EMA
  AttentionAggregator  : TabR 방식 similarity 기반 이웃 집계
                          + Gated Feature Fusion (faithfulness 보장)
  MemoryBank           : Cross-group fallback (인접 centroid 확장 검색)

Forward 흐름
────────────
  X_query
    ↓  TabularEmbedder
  query_emb (B, D)
    ├─ CentroidLayer(query_emb) → context_emb (B, D) + routing + FAISS mask
    └─ MemoryBank.retrieve → k neighbours (cross-group fallback 포함)
         ↓ AttentionAggregator(query_emb, nk, nv, labels)
         │  └─ Gated fusion: fused = g·feat_emb + (1-g)·agg
       fused_agg (B, D) + evidence_w (B, k) + feature_imp (B, k, F)
    ↓
  [query_emb ‖ context_emb ‖ fused_agg] → PredictionHead → ŷ

[Gated Fusion 효과]
  - 설명에 쓰이는 feature_imp가 prediction 경로에도 참여
  - faithfulness(설명-예측 일관성) architectural 보장
  - gate 값으로 sample-wise feature/neighbor path 사용 비율 진단 가능

[Cross-group fallback 효과]
  - 그룹 크기 < K인 소규모 그룹에서 인접 centroid 그룹까지 확장 검색
  - 전체 검색 fallback 대비 centroid 분할 의미 유지
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.prototypes   import CentroidLayer, PrototypeLayer  # PrototypeLayer = CentroidLayer alias
from libs.evidence     import AttentionAggregator


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
        centroid_emb: "Optional[torch.Tensor]" = None,  # (P, D) — cross-group용
    ) -> None:
        """
        sample_groups를 GPU 텐서로 미리 변환·캐시.
        EMA 업데이트 후 에폭당 1번만 호출 → retrieve 내부 변환 비용 제거.
        패딩: 가장 큰 그룹 크기에 맞춰 -1로 채움.

        [Cross-group fallback]
        centroid_emb가 주어지면, 각 centroid에 대해 가장 가까운 centroid를
        미리 계산하여 캐시합니다. 그룹 크기 < k인 경우 인접 centroid 그룹까지
        확장하여 검색합니다 (전체 검색 대신).
        """
        P   = len(sample_groups)
        max_g = max((len(g) for g in sample_groups), default=0)
        if max_g == 0:
            self._cached_groups      = None
            self._cached_group_sizes = None
            self._cached_extended    = None
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

        # ── Cross-group: 인접 centroid 그룹까지 합친 확장 그룹 캐시 ──
        # 각 centroid p에 대해, 자기 그룹 + 가장 가까운 centroid 그룹을 합침
        if centroid_emb is not None and P > 1:
            c = F.normalize(centroid_emb.detach(), dim=-1)    # (P, D)
            sim = c @ c.T                                     # (P, P)
            sim.fill_diagonal_(-1.0)  # 자기 자신 제외
            nearest = sim.argmax(dim=-1)                      # (P,) 각 centroid의 최인접

            extended_groups = []
            for p in range(P):
                # 자기 그룹 + 최인접 그룹 합침
                own    = sample_groups[p]
                near_p = nearest[p].item()
                neighbor = sample_groups[near_p]
                merged = own + neighbor
                extended_groups.append(merged)

            # 확장 그룹도 패딩 텐서로 캐시
            max_eg = max((len(g) for g in extended_groups), default=0)
            if max_eg > 0:
                padded_ext = torch.full((P, max_eg), -1, dtype=torch.long)
                for p, g in enumerate(extended_groups):
                    if g:
                        padded_ext[p, :len(g)] = torch.tensor(g, dtype=torch.long)
                self._cached_extended      = padded_ext.to(device)    # (P, max_eg)
                self._cached_extended_sizes = torch.tensor(
                    [len(g) for g in extended_groups], dtype=torch.long, device=device
                )
            else:
                self._cached_extended = None
        else:
            self._cached_extended = None

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

        # fallback 샘플: 인접 centroid 그룹까지 확장하여 검색 (cross-group)
        # 기존: zeros 유지 (사실상 전체 검색 포기)
        # 개선: 자기 그룹 + 최인접 centroid 그룹에서 검색
        if fallback_mask.any():
            fb_idx = fallback_mask.nonzero(as_tuple=True)[0]   # (Bf,)
            ha_fb  = ha[fb_idx]                                 # (Bf,)
            q_fb   = q_norm[fb_idx]                             # (Bf, D)
            Bf     = fb_idx.shape[0]

            # cross-group 확장 캐시가 있으면 사용, 없으면 전체 검색
            ext = getattr(self, '_cached_extended', None)
            if ext is not None:
                cand_ext   = ext[ha_fb]                         # (Bf, max_eg)
                ext_sizes  = self._cached_extended_sizes[ha_fb] # (Bf,)
                valid_ext  = (cand_ext >= 0)
                safe_ext   = cand_ext.clamp(min=0, max=n - 1)

                # 확장 그룹도 K보다 작으면 → 진짜 전체 검색 fallback
                still_small = ext_sizes < k
                use_ext     = ~still_small

                if use_ext.any():
                    ext_idx   = use_ext.nonzero(as_tuple=True)[0]
                    max_eg    = safe_ext.shape[1]
                    k_eff_ext = min(k, max_eg)

                    for i in ext_idx:
                        i = i.item()
                        b_pos = fb_idx[i]
                        si_e  = safe_ext[i]                     # (max_eg,)
                        vm_e  = valid_ext[i]                    # (max_eg,)
                        q_e   = q_fb[i:i+1]                     # (1, D)

                        keys_e = F.normalize(keys_full[si_e[vm_e]], dim=-1)  # (valid, D)
                        if keys_e.shape[0] < k:
                            # 그래도 부족하면 전체 검색
                            keys_all = F.normalize(keys_full, dim=-1)
                            sim_all  = q_e @ keys_all.T
                            _, idx_all = sim_all.topk(min(k, n), dim=-1)
                            idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                            out_nk[b_pos]     = keys_full[idx_all]
                            out_nv[b_pos]     = vals_full[idx_all]
                            out_labels[b_pos] = labels_full[idx_all]
                            top_k_idx[b_pos]  = idx_all
                        else:
                            sim_e = q_e @ keys_e.T                           # (1, valid)
                            _, top_e = sim_e.topk(min(k, keys_e.shape[0]), dim=-1)
                            real_idx = si_e[vm_e][top_e.squeeze(0)]          # (k,)
                            real_idx = real_idx.clamp(0, n - 1)
                            kk = real_idx.shape[0]
                            out_nk[b_pos, :kk]     = keys_full[real_idx]
                            out_nv[b_pos, :kk]     = vals_full[real_idx]
                            out_labels[b_pos, :kk] = labels_full[real_idx]
                            top_k_idx[b_pos, :kk]  = real_idx

                # 확장도 부족한 샘플 → 전체 검색
                if still_small.any():
                    ss_idx = still_small.nonzero(as_tuple=True)[0]
                    for i in ss_idx:
                        i = i.item()
                        b_pos = fb_idx[i]
                        q_s   = q_fb[i:i+1]
                        keys_all = F.normalize(keys_full, dim=-1)
                        sim_all  = q_s @ keys_all.T
                        _, idx_all = sim_all.topk(min(k, n), dim=-1)
                        idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                        out_nk[b_pos]     = keys_full[idx_all]
                        out_nv[b_pos]     = vals_full[idx_all]
                        out_labels[b_pos] = labels_full[idx_all]
                        top_k_idx[b_pos]  = idx_all
            else:
                # 확장 캐시 없으면 전체 검색 (기존 동작)
                keys_all = F.normalize(keys_full, dim=-1)
                for i in range(Bf):
                    b_pos = fb_idx[i]
                    q_s   = q_fb[i:i+1]
                    sim_all = q_s @ keys_all.T
                    _, idx_all = sim_all.topk(min(k, n), dim=-1)
                    idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                    out_nk[b_pos]     = keys_full[idx_all]
                    out_nv[b_pos]     = vals_full[idx_all]
                    out_labels[b_pos] = labels_full[idx_all]
                    top_k_idx[b_pos]  = idx_all

        return out_nk, out_nv, out_labels, top_k_idx

    @torch.no_grad()
    def retrieve_hierarchical(
        self,
        query: torch.Tensor,             # (B, D)
        k: int,                           # 그룹당 이웃 수
        topM_idx: torch.Tensor,          # (B, M) — top-M centroid 인덱스
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-M centroid 각각의 그룹에서 k개 이웃을 검색.

        기존 retrieve를 M번 호출 (M=2~3이므로 부담 없음).

        Returns
        ───────
        nk_hier     : (B, M, k, D)  — M개 그룹 × k개 이웃 key
        nv_hier     : (B, M, k, D)  — M개 그룹 × k개 이웃 value
        labels_hier : (B, M, k)
        idx_hier    : (B, M, k)     — MemoryBank 인덱스 (FeatureStore용)
        """
        B, M = topM_idx.shape

        nk_list, nv_list, lbl_list, idx_list = [], [], [], []
        for m in range(M):
            ha_m = topM_idx[:, m]  # (B,)
            nk_m, nv_m, lbl_m, idx_m = self.retrieve(
                query, k, hard_assignment=ha_m,
            )
            nk_list.append(nk_m)
            nv_list.append(nv_m)
            lbl_list.append(lbl_m)
            idx_list.append(idx_m)

        nk_hier  = torch.stack(nk_list,  dim=1)   # (B, M, k, D)
        nv_hier  = torch.stack(nv_list,  dim=1)
        lbl_hier = torch.stack(lbl_list, dim=1)
        idx_hier = torch.stack(idx_list, dim=1)

        return nk_hier, nv_hier, lbl_hier, idx_hier


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
    loss_weights      : 보조 손실 가중치 {'diversity': .., 'commitment': .., 'entropy': ..}
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
            "diversity":    0.01,
            "commitment":   0.01,
            "entropy":      0.01,
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

        # ── TabR 방식 이웃 집계 + Gated Fusion (faithfulness) ──
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

        # ── 예측 헤드: [query ‖ context ‖ fused_agg] → ŷ ──
        # head 입력 차원은 그대로 3D (gated fusion 덕분에 차원 유지)
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
        # prototypes.py가 5개 반환 (hierarchical 호환) → 필요한 3개만 사용
        context_emb, hard_assignment, routing_probs, _, _ = \
            self.prototype_layer(query_emb)

        # 3. KNN 검색 + Gated Fusion
        if self.memory.filled.item() >= self.k:
            nk, nv, neighbour_labels, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                hard_assignment=hard_assignment,
            )
            fused_agg, evidence_w, feature_imp, attn_w, gate, agg_emb_pure = \
                self.ot_selector(query_emb, nk, nv, neighbour_labels)
        else:
            # Memory 미충족 fallback
            fused_agg      = torch.zeros_like(query_emb)
            agg_emb_pure   = torch.zeros_like(query_emb)
            evidence_w     = torch.full((X.shape[0], self.k), 1.0 / self.k, device=X.device)
            feature_imp    = None
            attn_w         = None
            topk_idx       = torch.zeros(X.shape[0], self.k, dtype=torch.long, device=X.device)
            gate           = torch.full_like(query_emb, 0.5)

        # 4. 예측
        combined = torch.cat([query_emb, context_emb, fused_agg], dim=-1)
        logits   = self.head(combined)

        # 5. 메모리 업데이트 (학습 시)
        if self.training and labels is not None:
            self.memory.update(query_emb.detach(), context_emb.detach(), labels.float())
            if self._feature_store is not None:
                self._feature_store.update(X)

        # 6. 보조 손실
        aux_loss = torch.tensor(0.0, device=X.device)
        if self.training:
            aux_loss = (
                self.loss_weights["diversity"]  * self.prototype_layer.diversity_loss()
                + self.loss_weights["commitment"] * self.prototype_layer.commitment_loss(query_emb, hard_assignment)
                + self.loss_weights.get("entropy", 0.01) * self.prototype_layer.entropy_loss(routing_probs)
            )

        out = {
            "logits":      logits,
            "aux_loss":    aux_loss,
            "routing":     routing_probs,
            "hard_group":  hard_assignment,
            "evidence_w":  evidence_w,
            "feature_imp": feature_imp,
            "attn_w":      attn_w,
            "topk_idx":    topk_idx,
            # Gated Fusion 진단용
            "gate":          gate,
            "agg_emb_pure":  agg_emb_pure,
            "fused_agg":     fused_agg,
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
                    "prototype":      proto_exp[b],
                    "evidence":       ev_exp[b],
                    "feature_match":  feat_exp[b],
                    "gate_mean":      float(gate[b].mean().item()),
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
                 f"  Parameters     : {total:,}",
                 f"  Embed dim      : {self.embed_dim}",
                 f"  Centroids      : {self.prototype_layer.P}",
                 f"  KNN k          : {self.k}",
                 f"  Dual-Space     : {'ON' if self.prototype_layer.F > 0 else 'OFF'}",
                 f"  Gated Fusion   : ON (faithfulness)",
                 f"  Cross-group    : ON (adjacent centroid fallback)"]
        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        return "\n".join(lines)