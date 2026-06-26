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

    def __init__(self, embed_dim, k, n_features, n_output, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k

        # TabR 원본: label 임베딩
        self.label_encoder = nn.Linear(1, embed_dim)

        # TabR 원본: T() — query-neighbour 차이 변환 MLP
        d_block = embed_dim * 2
        self.T = nn.Sequential(
            nn.Linear(embed_dim, d_block),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_block, embed_dim, bias=False),
        )

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
        label_emb = self.label_encoder(
            neighbour_labels.unsqueeze(-1).float()
        )                                                # (B, k, D)
        values = label_emb + self.T(
            query_emb.unsqueeze(1) - nk
        )

        # 3. 가중합 → head 입력
        agg_emb = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        return agg_emb, evidence_w

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