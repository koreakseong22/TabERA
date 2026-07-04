"""
libs/tabera.py
============
TabERA — Dual-Space Prototype Explainable TabR Model.

  CentroidLayer        : Dual-Space Prototype + STE Routing + EMA
  AttentionAggregator  : TabR 방식 similarity 기반 이웃 집계
  MemoryBank           : Cross-group fallback (인접 centroid 확장 검색)

Forward 흐름
────────────
  X_query
    ↓  TabularEmbedder
  query_emb (B, D)
    ├─ CentroidLayer(query_emb) → context_emb (B, D) + routing
    └─ MemoryBank.retrieve → k neighbours (cross-group fallback 포함)
         ↓ AttentionAggregator(query_emb, nk, nv, labels)
       agg_emb (B, D) + evidence_w (B, k)
    ↓
  [query_emb ‖ context_emb ‖ agg_emb] → PredictionHead → ŷ

[경량화: Gated Fusion 제거]
  실험 근거 (4개 데이터셋, seed=8):
    - gate_mean ≈ 0.5, std ≈ 0.01~0.08 → gate_net 미학습
    - feature_imp: ρ≈Random → IG로 대체 완료
  결과: evidence_w가 prediction에 직접 기여 → 설명 ② faithfulness 향상

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
    def __init__(self, max_size: int, embed_dim: int, n_size_buckets: int = 4):
        super().__init__()
        self.max_size = max_size
        # 그룹-크기 버킷 개수 (retrieve()의 normal_mask 처리용, 상수 반복 횟수)
        self.n_size_buckets = n_size_buckets
        self.register_buffer("keys",   torch.zeros(max_size, embed_dim))
        self.register_buffer("vals",   torch.zeros(max_size, embed_dim))
        self.register_buffer("labels", torch.zeros(max_size))
        self.register_buffer("ptr",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        # ── 정규화 캐시: retrieve() 내부의 반복 F.normalize 제거용 ──────
        # update() 시 O(B)로 증분 갱신, retrieve()에서는 그대로 gather만 함.
        self.register_buffer("_keys_norm", torch.zeros(max_size, embed_dim))

        # [출처 명확화] retrieve()가 "그룹 하나가 이례적으로 커서 나머지
        # 모두를 그 폭에 맞춰 패딩하는 게 낭비인지"를 판단하는 임계값.
        # 기본값 4096은 어떤 계산/문헌 근거도 없는 값 — 초기화 시점(에폭 0,
        # GPU 메모리 조회 전)이나 CPU 환경에서만 쓰이는 안전한 폴백일 뿐.
        # 학습 중에는 매 epoch update_outlier_threshold()가 실제 GPU 여유
        # 메모리를 반영해서 이 값을 다시 계산해 덮어씀 — retrieve()가 매
        # 배치 GPU를 조회하면(동기화 오버헤드) 예전에 없앤 문제가 재발하므로,
        # 조회는 epoch당 1회(supervised.py)로 제한.
        self._outlier_threshold = 4096

    def update_outlier_threshold(
        self,
        n_prototypes: int,
        free_bytes: "Optional[int]" = None,
        device: "Optional[torch.device]" = None,
        safety_fraction: float = 0.3,
    ) -> None:
        """
        retrieve()의 "정상 경로"(단일 텐서)가 만들 것으로 예상되는 텐서 크기가
        현재 남은 GPU 메모리의 safety_fraction을 넘지 않도록, local_max_g의
        임계값을 역산한다. 근거 없는 고정 상수(4096) 대신 실제 자원 제약에
        직접 결부시키기 위함 — supervised.py의 collapse 안전장치와 동일한
        원칙. epoch당 1회만 호출할 것을 전제로 함(배치마다 부르면 GPU 조회로
        인한 동기화 오버헤드가 재발함).

        Parameters
        ──────────
        n_prototypes : 전체 centroid 수(P) — 한 배치에 등장 가능한 unique
                       centroid 수(U)의 최악의 경우 상한으로 사용.
        free_bytes   : 이미 조회한 남은 GPU 메모리(바이트). None이면 함수
                       내부에서 직접 조회(추가 동기화 1회 발생).
        device       : free_bytes를 안 넘겼을 때 조회에 쓸 device.
        safety_fraction : 정상 경로 텐서가 남은 메모리의 이 비율을 넘으면
                       위험하다고 판단 (기본 0.3 — keys_u 외 Q_pad/sim_u
                       등 부수 텐서도 있어 여유를 둠).
        """
        if free_bytes is None:
            if device is None or not torch.cuda.is_available() or not str(device).startswith("cuda"):
                return  # CPU 등 GPU 메모리 개념이 없는 환경 → 폴백(4096) 유지
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
            except Exception:
                return  # 조회 실패 → 폴백 유지

        D = self.keys.shape[1]
        U_pad_worst = ((n_prototypes + 7) // 8) * 8  # 배치 내 unique centroid 수의 최악의 경우 상한
        # keys_u + Q_pad + sim_u 등 부수 텐서를 대략 3배로 어림 (정확한
        # 수치가 아니라 "이 정도면 위험하다"는 자릿수 판단용 — 이 3배율도
        # 검증된 상수는 아니고 supervised.py의 안전장치에서 쓴 것과 같은
        # 수준의 어림값임을 명시)
        denom = U_pad_worst * D * 4 * 3
        if denom <= 0:
            return
        new_threshold = int((free_bytes * safety_fraction) / denom)
        # 256배수로 내림 (retrieve()의 라운딩 단위와 맞춤), 너무 작아지지
        # 않도록 최소 k*4 이상은 보장
        new_threshold = max((new_threshold // 256) * 256, 512)
        self._outlier_threshold = new_threshold

    @torch.no_grad()
    def update(self, keys, vals, labels):
        B   = keys.shape[0]
        ptr = self.ptr.item()
        end = min(ptr + B, self.max_size)
        n   = end - ptr
        self.keys[ptr:end]   = keys[:n].detach()
        self.vals[ptr:end]   = vals[:n].detach()
        self.labels[ptr:end] = labels[:n].float().detach()
        # 정규화도 여기서 O(B)로 한 번만 계산 (retrieve 매 배치 재계산 제거)
        self._keys_norm[ptr:end] = F.normalize(keys[:n].detach(), dim=-1)
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

        # [임시 진단] 하이브리드(이례적) 경로가 실제로 발동하는지 배치 단위로
        # 카운트 — 41150 검증 완료되면 제거할 것. supervised.py에서 epoch마다
        # 읽어서 로그로 보여줌.
        if not hasattr(self, "_normal_path_count"):
            self._normal_path_count = 0
            self._hybrid_path_count = 0

        keys_full   = self.keys[:n]    # (n, D)
        vals_full   = self.vals[:n]    # (n, D)
        labels_full = self.labels[:n]  # (n,)
        q_norm      = F.normalize(query, dim=-1)  # (B, D)

        # ── 캐시 없거나 초기화 전 → 전체 검색 fallback ──────────
        cached = getattr(self, '_cached_groups', None)
        if hard_assignment is None or cached is None or n < k:
            keys_all = self._keys_norm[:n]  # 정규화 캐시 재사용
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

        # ── 정상 샘플: centroid dedup(전체 1회, 티어링 없음) ──────────────
        # [프로파일링 결과 반영] N=35,855 / D≤256 / B=256 규모에서는 실제
        # GPU 연산(Self CUDA)이 1ms도 안 될 만큼 작은데, 이전 버전(사이즈-티어
        # + dedup)은 티어마다 argsort/unique_consecutive/remap/nonzero/
        # repeat_interleave를 반복 호출해 CPU 25.4ms 중 대부분을
        # aten::index(71회)/aten::nonzero(21회)/aten::index_put_(30회)/
        # aten::repeat_interleave(콜당 728us) 같은 "부기(bookkeeping)"
        # 연산에 소모했음 (실제 topk/bmm은 933us 중 260us뿐).
        #
        # 이 규모에서는 FLOP을 아끼는 것보다 커널 발사 횟수를 줄이는 게
        # 훨씬 중요 → 티어링을 제거하고 dedup만 남김:
        #   - centroid dedup은 여전히 유지 (같은 centroid를 가리키는 쿼리가
        #     후보를 중복 gather하는 것만 막아도 가장 큰 낭비는 해결됨)
        #   - 폭(local_max_g)은 "이 배치에 실제로 등장한 centroid들 중
        #     최댓값"으로만 제한 (여전히 전역 max_g보다 훨씬 작음)
        #   - repeat_interleave → bucketize로 교체 (동일 결과, 훨씬 가벼움)
        #   - nonzero 호출을 배치 전체에서 1회로 축소 (티어당 반복 없음)
        if normal_mask.any():
            nm_idx = normal_mask.nonzero(as_tuple=True)[0]  # (Bn,)
            ha_nm  = ha[nm_idx]                              # (Bn,)
            q_nm   = q_norm[nm_idx]                          # (Bn, D)
            Bn = nm_idx.shape[0]                              # python int, 동기화 없음

            # ── centroid dedup: 배치 전체 1회 ──
            csort_idx   = torch.argsort(ha_nm)                # (Bn,)
            ha_c_sorted = ha_nm[csort_idx]
            q_c_sorted  = q_nm[csort_idx]

            uniq, counts = torch.unique_consecutive(ha_c_sorted, return_counts=True)  # (U,)
            U = uniq.shape[0]

            offsets = counts.cumsum(0)                          # (U,) 각 그룹의 끝 위치(배타적 경계)
            group_id = torch.bucketize(
                torch.arange(Bn, device=dev), offsets, right=True
            )                                                    # (Bn,) 0..U-1
            rank = torch.arange(Bn, device=dev) - (offsets[group_id] - counts[group_id])  # (Bn,) centroid 내 0-index

            grp_sizes_u = self._cached_group_sizes[uniq]         # (U,) 이 배치에 등장한 centroid들의 진짜 그룹 크기
            local_max_g_raw = max(int(grp_sizes_u.max()), k)

            # [하이브리드 대응] local_max_g_raw가 이례적으로 크면(예: centroid
            # 하나가 데이터의 상당 부분을 흡수한 상태), 모든 U개 centroid를
            # 이 큰 폭에 맞춰 패딩하는 게 메모리/연산량을 폭증시킴 (id=41150,
            # N=104,050에서 max_cluster_size가 3,526→34,195까지 커지며 실측
            # 됨 — 대부분 centroid는 건강한데 소수만 비대해지는 경우라
            # active_ratio 기반 collapse 감지로는 못 잡음).
            # [출처 명확화] 이 임계값은 self._outlier_threshold — 근거 없는
            # 고정 상수가 아니라 update_outlier_threshold()가 실제 GPU 여유
            # 메모리 기준으로 계산해 넣어둔 값 (supervised.py가 epoch당 1회
            # 갱신). 아직 한 번도 갱신 안 됐거나 CPU 환경이면 __init__의
            # 폴백값(4096, 이것도 근거 없는 값)이 쓰임 — 이 경우는 문서화된
            # 한계로 남겨둠.
            _OUTLIER_THRESHOLD = self._outlier_threshold

            if local_max_g_raw <= _OUTLIER_THRESHOLD:
                self._normal_path_count += 1   # [임시 진단]
                # ── 정상 경로: 기존과 완전히 동일 (오버헤드 없음) ──────
                _round_u = 8
                U_pad = ((U + _round_u - 1) // _round_u) * _round_u
                if U_pad > U:
                    pad_ids = uniq[:1].expand(U_pad - U)
                    uniq_p  = torch.cat([uniq, pad_ids], dim=0)
                else:
                    uniq_p = uniq

                max_q_raw = int(counts.max())
                max_q = ((max_q_raw + 15) // 16) * 16
                max_q = min(max_q, Bn)

                local_max_g = ((local_max_g_raw + 255) // 256) * 256
                local_max_g = min(local_max_g, self._cached_groups.shape[1])

                Q_pad = torch.zeros(U_pad, max_q, D, device=dev)
                Q_pad[group_id, rank] = q_c_sorted

                cand_u  = self._cached_groups[uniq_p, :local_max_g]
                valid_u = cand_u >= 0
                safe_u  = cand_u.clamp(min=0, max=n - 1)

                keys_u = self._keys_norm[:n][safe_u.reshape(-1)].view(U_pad, local_max_g, D)

                sim_u = torch.bmm(Q_pad, keys_u.transpose(1, 2))
                sim_u = sim_u.masked_fill(~valid_u.unsqueeze(1), -1e9)

                k_eff = min(k, local_max_g)
                _, top_u  = sim_u.topk(k_eff, dim=-1)
                i_final_u = safe_u.unsqueeze(1).expand(-1, max_q, -1).gather(2, top_u)
                i_final_c_sorted = i_final_u[group_id, rank]

                final_pos = nm_idx[csort_idx]
                top_k_idx[final_pos, :k_eff]  = i_final_c_sorted
                out_nk[final_pos, :k_eff]     = keys_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff, D)
                out_nv[final_pos, :k_eff]     = vals_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff, D)
                out_labels[final_pos, :k_eff] = labels_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff)

            else:
                self._hybrid_path_count += 1   # [임시 진단]
                # ── 이례적 경로: 큰 그룹 / 작은 그룹 분리 (드문 경우만) ──
                big_mask = grp_sizes_u > _OUTLIER_THRESHOLD          # (U,) bool
                for tier_mask in (~big_mask, big_mask):
                    if not tier_mask.any():
                        continue
                    query_in_tier = tier_mask[group_id]              # (Bn,) bool
                    if not query_in_tier.any():
                        continue
                    sel_pos = query_in_tier.nonzero(as_tuple=True)[0]  # (Bt,) csorted 좌표계 위치

                    tier_uniq_local = tier_mask.nonzero(as_tuple=True)[0]  # (Ut,) 0..U-1 인덱스
                    Ut = tier_uniq_local.shape[0]
                    remap = torch.full((U,), -1, dtype=torch.long, device=dev)
                    remap[tier_uniq_local] = torch.arange(Ut, device=dev)

                    local_gid  = remap[group_id[sel_pos]]            # (Bt,) 0..Ut-1
                    local_rank = rank[sel_pos]                        # (Bt,)
                    q_sel      = q_c_sorted[sel_pos]                  # (Bt, D)

                    tier_centroid_ids = uniq[tier_uniq_local]         # (Ut,) 실제 centroid id
                    tier_counts       = counts[tier_uniq_local]       # (Ut,)

                    Ut_pad = ((Ut + 7) // 8) * 8
                    if Ut_pad > Ut:
                        pad_ids2 = tier_centroid_ids[:1].expand(Ut_pad - Ut)
                        tier_centroid_ids_p = torch.cat([tier_centroid_ids, pad_ids2], dim=0)
                    else:
                        tier_centroid_ids_p = tier_centroid_ids

                    max_q_tier_raw = int(tier_counts.max())
                    max_q_tier = ((max_q_tier_raw + 15) // 16) * 16
                    max_q_tier = min(max_q_tier, Bn)

                    local_max_g_tier_raw = max(
                        int(self._cached_group_sizes[tier_centroid_ids].max()), k
                    )
                    local_max_g_tier = ((local_max_g_tier_raw + 255) // 256) * 256
                    local_max_g_tier = min(local_max_g_tier, self._cached_groups.shape[1])

                    Q_pad_t = torch.zeros(Ut_pad, max_q_tier, D, device=dev)
                    Q_pad_t[local_gid, local_rank] = q_sel

                    cand_t  = self._cached_groups[tier_centroid_ids_p, :local_max_g_tier]
                    valid_t = cand_t >= 0
                    safe_t  = cand_t.clamp(min=0, max=n - 1)
                    keys_t  = self._keys_norm[:n][safe_t.reshape(-1)].view(Ut_pad, local_max_g_tier, D)

                    sim_t = torch.bmm(Q_pad_t, keys_t.transpose(1, 2))
                    sim_t = sim_t.masked_fill(~valid_t.unsqueeze(1), -1e9)

                    k_eff_t = min(k, local_max_g_tier)
                    _, top_t  = sim_t.topk(k_eff_t, dim=-1)
                    i_final_t = safe_t.unsqueeze(1).expand(-1, max_q_tier, -1).gather(2, top_t)
                    i_final_sel = i_final_t[local_gid, local_rank]      # (Bt, k_eff_t)

                    final_pos_t = nm_idx[csort_idx[sel_pos]]            # (Bt,) 원래 배치(B) 내 최종 위치
                    top_k_idx[final_pos_t, :k_eff_t]  = i_final_sel
                    out_nk[final_pos_t, :k_eff_t]     = keys_full[i_final_sel.reshape(-1)].view(-1, k_eff_t, D)
                    out_nv[final_pos_t, :k_eff_t]     = vals_full[i_final_sel.reshape(-1)].view(-1, k_eff_t, D)
                    out_labels[final_pos_t, :k_eff_t] = labels_full[i_final_sel.reshape(-1)].view(-1, k_eff_t)

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

                        keys_e = self._keys_norm[:n][si_e[vm_e]]  # (valid, D) 정규화 캐시 재사용
                        if keys_e.shape[0] < k:
                            # 그래도 부족하면 전체 검색
                            keys_all = self._keys_norm[:n]
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
                        keys_all = self._keys_norm[:n]
                        sim_all  = q_s @ keys_all.T
                        _, idx_all = sim_all.topk(min(k, n), dim=-1)
                        idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                        out_nk[b_pos]     = keys_full[idx_all]
                        out_nv[b_pos]     = vals_full[idx_all]
                        out_labels[b_pos] = labels_full[idx_all]
                        top_k_idx[b_pos]  = idx_all
            else:
                # 확장 캐시 없으면 전체 검색 (기존 동작)
                keys_all = self._keys_norm[:n]
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
        use_offset_correction: bool = True,
        global_retrieve: bool = False,
        use_context_emb: bool = True,
        detach_context_grad: bool = False,
        use_context_projection: bool = False,
    ) -> None:
        super().__init__()
        self.k            = k
        self.embed_dim    = embed_dim
        self.n_output     = n_output
        self.n_features   = n_features
        self.loss_weights = loss_weights or {
            "diversity":    0.01,
            "commitment":   0.01,
        }
        self.column_names = column_names
        self.use_offset_correction = use_offset_correction
        # [진단용] True면 retrieve()에서 그룹 제약을 끄고 전체 검색(순수
        # TabR 스타일 전역 KNN)을 함. context_emb(설명①)는 정상적으로
        # n_prototypes개 그룹 정보를 그대로 유지 — retrieve()만 영향받음.
        # "그룹-제약 KNN이 정확도에 요구하는 대가"를 격리해서 재기 위한
        # 일회성 진단용이며, 본 실험에는 기본값(False)을 씀.
        self.global_retrieve = global_retrieve
        # [진단용] context_emb(설명①의 신호)를 head 입력에서 아예 제외.
        # STE 라우팅/centroid 학습(diversity_loss, commitment_loss)은 그대로
        # 유지됨 — 이 둘은 centroid_emb를 직접 학습시키지 context_emb가
        # head에 들어가는지와 무관하기 때문. "그룹 신호 자체가 예측에
        # 기여하는가"만 깨끗하게 격리해서 재기 위함. T()처럼 꺼지면
        # 파라미터 자체가 안 생김(head 입력 차원이 줄어듦).
        self.use_context_emb = use_context_emb
        # [진단용] centroid_emb는 diversity_loss(흩어뜨림)와 task_loss
        # (head를 거친 예측 손실, context_emb 경유)라는 서로 다른 목적의
        # gradient를 동시에 받음 (commitment_loss는 이미 assigned.detach()라
        # centroid_emb를 안 건드림 — 확인됨). 이 두 목적이 충돌해 centroid_emb가
        # 어느 쪽에도 최적이 아닌 타협점에 머물 가능성을 검증하기 위해,
        # True면 head로 가는 context_emb만 detach — forward 값은 그대로
        # 전달되지만 task_loss가 centroid_emb로 역전파되지 않음
        # (centroid_emb는 diversity_loss만으로 학습됨).
        self.detach_context_grad = detach_context_grad
        # [구조 조정] context_emb를 head로 보내기 전 학습 가능한 Linear를
        # 하나 거치게 함. detach_context_grad(gradient 완전 차단)와 달리
        # task_loss의 gradient가 여전히 centroid_emb까지 도달하되, 이
        # 프로젝션 행렬이 "예측에 유리하게 바꾸는 일"의 일부를 대신 떠맡아
        # centroid_emb 자체가 덜 왜곡되길 기대하는 절충안. raw centroid_emb
        # 를 직접 쓰는 설명①(hard_assignment, centroid_x medoid, confidence)
        # 계산에는 전혀 관여하지 않음 — head 직전에만 끼움.
        self.use_context_projection = use_context_projection
        self.context_proj = (
            nn.Linear(embed_dim, embed_dim)
            if (use_context_projection and use_context_emb) else None
        )

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

        # ── TabR 방식 이웃 집계 ──────────────────────────
        # use_offset_correction=False → value=label_emb만 사용 (T() ablation)
        self.ot_selector = AttentionAggregator(
            embed_dim=embed_dim,
            k=k,
            n_features=n_features,
            n_output=n_output,
            dropout=dropout,
            use_offset_correction=use_offset_correction,
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

        # ── 예측 헤드: [query ‖ context ‖ agg_emb] → ŷ ──
        # use_context_emb=False면 context_emb를 아예 제외 (head 입력 차원도
        # 그만큼 줄어듦 — T()처럼 "안 쓰는 파라미터가 남아있는" 상태가 아니라
        # 진짜로 없앤 상태로 비교하기 위함)
        _head_in = embed_dim * (3 if use_context_emb else 2)
        self.head = nn.Sequential(
            nn.LayerNorm(_head_in),
            nn.Linear(_head_in, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_output),
        )

    # ── Forward ────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,                          # (B, F)
        labels: Optional[torch.Tensor] = None,    # (B,) 학습 시 메모리 업데이트용
        return_explanations: bool = False,
        ablation_mode: str = "none",              # ablation 모드 (학습 시 "none" 유지)
    ) -> Dict[str, torch.Tensor]:
        # 1. 임베딩
        query_emb = self.embedder(X)               # (B, D)

        # 2. 프로토타입 라우팅 (CentroidLayer)
        # prototypes.py가 5개 반환 (hierarchical 호환) → 필요한 3개만 사용
        context_emb, hard_assignment, routing_probs, _, _ = \
            self.prototype_layer(query_emb)

        # 3. KNN 검색 + Attention 집계
        if self.memory.filled.item() >= self.k:
            nk, nv, neighbour_labels, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                # [진단용] global_retrieve=True면 그룹 무시하고 전체 검색.
                # context_emb(위 2번)는 hard_assignment와 무관하게 이미
                # 정상 계산됨 — 설명①은 그대로, 검색만 전역으로 바뀜.
                hard_assignment=(None if self.global_retrieve else hard_assignment),
            )

            # ── Ablation: random_neighbor ────────────────────────
            if ablation_mode == "random_neighbor":
                B_abl = nk.shape[0]
                nk              = F.normalize(torch.randn_like(nk), dim=-1)
                nv              = torch.randn_like(nv)
                rand_perm        = torch.randperm(B_abl, device=X.device)
                neighbour_labels = neighbour_labels[rand_perm]

            agg_emb, evidence_w = self.ot_selector(query_emb, nk, nv, neighbour_labels)

        else:
            # Memory 미충족 fallback
            agg_emb    = torch.zeros_like(query_emb)
            evidence_w = torch.full((X.shape[0], self.k), 1.0 / self.k, device=X.device)
            topk_idx   = torch.zeros(X.shape[0], self.k, dtype=torch.long, device=X.device)

        # 4. 예측
        # use_context_emb=False면 context_emb를 head 입력에서 제외
        # (STE 라우팅/centroid 학습 자체는 그대로 — aux_loss가 별도로 학습시킴)
        if self.use_context_emb:
            _ctx_for_head = context_emb
            if self.context_proj is not None:
                _ctx_for_head = self.context_proj(_ctx_for_head)
            if self.detach_context_grad:
                _ctx_for_head = _ctx_for_head.detach()
            combined = torch.cat([query_emb, _ctx_for_head, agg_emb], dim=-1)
        else:
            combined = torch.cat([query_emb, agg_emb], dim=-1)
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
            )

        out = {
            "logits":      logits,
            "aux_loss":    aux_loss,
            "routing":     routing_probs,
            "hard_group":  hard_assignment,
            "evidence_w":  evidence_w,
            "topk_idx":    topk_idx,
            "agg_emb":     agg_emb,
            "ablation_mode": ablation_mode,
        }

        if return_explanations:
            # 설명용 softmax — temperature scaling 적용
            _temperature = 0.1
            with torch.no_grad():
                q_norm     = F.normalize(query_emb.detach(), dim=-1)       # (B, D)
                c_norm     = F.normalize(
                    self.prototype_layer.centroid_emb.detach(), dim=-1)    # (P, D)
                soft_probs = F.softmax(
                    (q_norm @ c_norm.T) / _temperature, dim=-1)            # (B, P)

            proto_exp = self.prototype_layer.explain_routing(hard_assignment, soft_probs)
            ev_exp    = self.ot_selector.explain_evidence(evidence_w)
            out["explanations"] = [
                {
                    "prototype": proto_exp[b],
                    "evidence":  ev_exp[b],
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
                 f"  Cross-group    : ON (adjacent centroid fallback)",
                 f"  Offset T()     : {'ON' if self.use_offset_correction else 'OFF (ablation)'}",
                 f"  Retrieve       : {'GROUP-CONSTRAINED' if not self.global_retrieve else 'GLOBAL (진단용, ablation)'}",
                 f"  context_emb    : {'IN head input' if self.use_context_emb else 'EXCLUDED (진단용, ablation)'}",
                 f"  context grad   : {'STOP (진단용, ablation)' if self.detach_context_grad else 'flows to centroid_emb'}",
                 f"  context proj   : {'Linear projection (구조 조정)' if self.context_proj is not None else 'none (raw concat)'}"]
        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        return "\n".join(lines)