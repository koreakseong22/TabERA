"""
evidence.py — Attention 기반 이웃 집계 및 설명 모듈

설계 원칙
─────────
- 미분 가능성: Scaled dot-product attention → 역전파 가능
- 설명가능성: attention weight = evidence_w (이웃별 기여도)
- 속도:       Sinkhorn 20 iter 제거 → 단순 softmax (~15배 빠름)
- 근거:       TabR 원본 방식과 동일한 attention 집계

[경량화: Gated Fusion 제거]
──────────────────────────────
실험 근거 (4개 데이터셋, seed=8):
  - gate_mean ≈ 0.5, std ≈ 0.01~0.08 → gate_net이 학습되지 않음
  - feature_imp: ρ≈Random → 설명으로 쓸모없음 (IG로 대체 완료)
  - 설명 ①②③ 모두 Gated Fusion과 무관

제거 대상: FeatureCrossAttention, feat_to_emb, feat_proj_d, gate_net
결과: agg_emb_pure가 직접 head 입력으로 사용됨
      → evidence_w가 prediction에 직접 기여 → 설명 ② faithfulness 향상
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AttentionAggregator(nn.Module):
    """
    TabR 방식 이웃 집계 (순수 retrieval-augmented prediction).

    Forward 출력
    ────────────
    agg_emb    : (B, D)   — attention-weighted aggregation (head 입력용)
    evidence_w : (B, k)   — 이웃별 attention weight (설명 ② 용)
    """

    def __init__(self, embed_dim, k, n_features, n_output, dropout=0.0,
                 use_offset_correction: bool = True):
        """
        use_offset_correction : True(기본값)면 TabR 원본 그대로
            value = label_emb + T(query - neighbour).
            False면 T()를 아예 생성하지 않고 value = label_emb만 사용
            (ablation용 — "T()가 실제로 기여하는가"를 재학습으로 검증하기 위함,
            evidence_w/retrieve()의 동작에는 영향 없음).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.use_offset_correction = use_offset_correction

        # TabR 원본: label 임베딩
        self.label_encoder = nn.Linear(1, embed_dim)

        # TabR 원본: T() — query-neighbour 차이 변환 MLP
        # use_offset_correction=False면 아예 생성하지 않음 (파라미터 수
        # 비교 등에서 "T()가 정말 없는" 상태와 정확히 대응하도록)
        if self.use_offset_correction:
            d_block = embed_dim * 2
            self.T = nn.Sequential(
                nn.Linear(embed_dim, d_block),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_block, embed_dim, bias=False),
            )
        else:
            self.T = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, query_emb, nk, nv, neighbour_labels):
        """
        Parameters
        ──────────
        query_emb        : (B, D)   — 임베더 출력
        nk               : (B, k, D)— 이웃 key 임베딩
        nv               : (B, k, D)— 이웃 value 임베딩 (현재 미사용, TabR 호환)
        neighbour_labels : (B, k)   — 이웃 레이블
        """
        # 1. TabR 방식 similarity: -||q||² + 2(q·k) - ||k||²
        similarities = (
            -query_emb.square().sum(-1, keepdim=True)
            + 2 * (query_emb.unsqueeze(1) @ nk.transpose(-1, -2)).squeeze(1)
            - nk.square().sum(-1)
        )
        evidence_w = F.softmax(similarities, dim=-1)    # (B, k)
        evidence_w = self.dropout(evidence_w)

        # 2. TabR 방식 value = label_emb + T(query - neighbour)
        #    (use_offset_correction=False면 T() 항 없이 label_emb만 사용)
        label_emb = self.label_encoder(
            neighbour_labels.unsqueeze(-1).float()
        )                                                # (B, k, D)
        if self.use_offset_correction:
            values = label_emb + self.T(
                query_emb.unsqueeze(1) - nk
            )
        else:
            values = label_emb

        # 3. 가중합 → head 입력
        agg_emb = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        return agg_emb, evidence_w

    # ── 진단 인터페이스 ────────────────────────────────────

    @torch.no_grad()
    def diagnose_value_components(
        self,
        query_emb: torch.Tensor,          # (B, D)
        nk: torch.Tensor,                 # (B, k, D)
        neighbour_labels: torch.Tensor,   # (B, k)
    ) -> Dict[str, float]:
        """
        value = label_emb + T(query - neighbour) 에서 두 항의 상대적 크기 진단.

        Gated Fusion 제거 때와 동일한 방식(재학습 없이, 학습된 가중치로
        구성 요소의 실제 크기를 직접 측정)의 T() 버전. T()가 항상
        label_emb 대비 무시할 만큼 작은 값을 낸다면, T가 학습 중
        유의미한 걸 배우지 못했다는 정황 증거가 됨 (gate_net이
        ≈0.5로 고정됐던 것과 같은 패턴). 다만 이건 "필요성"에 대한
        확정적 증거는 아니고 — 확실히 하려면 T() 없는 아키텍처를
        재학습해서 비교해야 함 (이 함수는 그 재학습이 필요한지
        먼저 가늠하기 위한 저비용 사전 진단용).

        Returns
        ───────
        label_emb_norm_mean/std : label_emb 항의 L2 norm 평균/표준편차
        offset_norm_mean/std    : T(query-neighbour) 항의 L2 norm 평균/표준편차
        ratio_mean/std          : offset_norm / label_emb_norm (샘플·이웃별)
        """
        if not self.use_offset_correction or self.T is None:
            raise RuntimeError(
                "이 모델은 use_offset_correction=False로 생성되어 T()가 없습니다 — "
                "value_diagnosis는 T() 있는 모델에서만 의미가 있습니다."
            )

        label_emb = self.label_encoder(
            neighbour_labels.unsqueeze(-1).float()
        )                                                # (B, k, D)
        offset_term = self.T(query_emb.unsqueeze(1) - nk)  # (B, k, D)

        label_norm  = label_emb.norm(dim=-1)              # (B, k)
        offset_norm = offset_term.norm(dim=-1)            # (B, k)
        ratio       = offset_norm / (label_norm + 1e-8)   # (B, k)

        return {
            "label_emb_norm_mean": float(label_norm.mean()),
            "label_emb_norm_std":  float(label_norm.std()),
            "offset_norm_mean":    float(offset_norm.mean()),
            "offset_norm_std":     float(offset_norm.std()),
            "ratio_mean":          float(ratio.mean()),
            "ratio_std":           float(ratio.std()),
        }

    # ── 설명 인터페이스 ────────────────────────────────────

    def explain_evidence(
        self,
        evidence_w: torch.Tensor,   # (B, k)
        top_n: int = 3,
    ) -> List[Dict]:
        """이웃별 기여도 설명."""
        ew_np = evidence_w.detach().cpu().numpy()
        B, k  = ew_np.shape
        out   = []
        for b in range(B):
            w          = ew_np[b]
            sorted_idx = np.argsort(w)[::-1]
            top_n_list = [(int(i), float(w[i])) for i in sorted_idx[:top_n]]
            out.append({
                "top_neighbours":  top_n_list,
                "dominant_weight": float(w.max()),
                "ignored_ratio":   float((w < 0.05).mean()),
                "entropy":         float(-(w * np.log(w + 1e-8)).sum()),
            })
        return out