"""
evidence.py — Attention 기반 이웃 집계 및 설명 모듈

설계 원칙
─────────
- 미분 가능성: Scaled dot-product attention → 역전파 가능
- 설명가능성: attention weight = evidence_w (이웃별 기여도)
- 속도:       Sinkhorn 20 iter 제거 → 단순 softmax (~15배 빠름)

[Gated Fusion 제거]
───────────────────
feat_to_emb, gate_net 제거.
feature_imp는 순수 설명 경로로만 사용 (prediction path 비참여).
head 입력은 [query_emb ‖ context_emb ‖ agg_emb] 유지.

Faithfulness 주장 범위:
- evidence_w 가중합 agg_emb가 prediction에 직접 참여
  → "이웃 #i가 evidence_w[i] 비율로 예측에 기여"는 구조적으로 성립
- feature_imp는 설명 참고용, faithfulness 주장 대상 아님
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureCrossAttention(nn.Module):
    """
    어떤 feature 때문에 유사한지 계산하는 모듈.
    OT 교체와 무관하게 유지 — 설명 경로 계층 3.
    """

    def __init__(self, embed_dim: int, n_features: int) -> None:
        super().__init__()
        self.feat_proj = nn.Linear(embed_dim, n_features, bias=False)

    def forward(
        self,
        query_emb: torch.Tensor,          # (B, D)
        neighbour_emb: torch.Tensor,      # (B, k, D)
        evidence_weights: torch.Tensor,   # (B, k)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환
        ────
        feature_importance : (B, k, F) — 이웃별 원본 feature 기여도
        attn_weights       : (B, k, D) — 차원별 attention (진단용)
        """
        B, k, D = neighbour_emb.shape

        q = query_emb.unsqueeze(1)                       # (B, 1, D)
        # 차원별 유사도
        dim_match = q * neighbour_emb                    # (B, k, D)

        # evidence_weights로 dominant 이웃 강조
        ew = evidence_weights.unsqueeze(-1)              # (B, k, 1)
        weighted = dim_match * ew                        # (B, k, D)

        # D → F 투영
        feat_imp_raw    = self.feat_proj(weighted)       # (B, k, F)
        feature_importance = feat_imp_raw.abs()
        attn_weights       = dim_match.abs()             # (B, k, D)

        return feature_importance, attn_weights


class AttentionAggregator(nn.Module):
    """
    TabR 방식 이웃 집계.

    Forward 출력
    ────────────
    agg_emb    : (B, D)    — evidence_w 가중합 aggregation (head 입력용)
    evidence_w : (B, k)    — 이웃별 attention weight (설명 + faithfulness)
    feature_imp: (B, k, F) — feature attribution (설명용, prediction 경로 비참여)
    attn_w     : (B, k, D) — 차원별 attention (진단용)
    """

    def __init__(self, embed_dim, k, n_features, n_output, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.n_features = n_features

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
        self.feat_cross = FeatureCrossAttention(embed_dim, n_features)

    def forward(self, query_emb, nk, nv, neighbour_labels):
        """
        Parameters
        ──────────
        query_emb        : (B, D)   — 임베더 출력
        nk               : (B, k, D)— 이웃 key 임베딩
        nv               : (B, k, D)— 이웃 value 임베딩 (현재 미사용, TabR 호환)
        neighbour_labels : (B, k)   — 이웃 레이블
        """
        # 1. TabR 방식 similarity
        # -||q||² + 2(q·k) - ||k||²
        similarities = (
            -query_emb.square().sum(-1, keepdim=True)              # (B, 1)
            + 2 * (query_emb.unsqueeze(1) @ nk.transpose(-1,-2)).squeeze(1)  # (B, k)
            - nk.square().sum(-1)                                  # (B, k)
        )
        evidence_w = F.softmax(similarities, dim=-1)               # (B, k)
        evidence_w = self.dropout(evidence_w)

        # 2. TabR 방식 value = label_emb + T(query - neighbour)
        label_emb = self.label_encoder(
            neighbour_labels.unsqueeze(-1).float()                 # (B, k, 1)
        )                                                           # (B, k, D)
        values = label_emb + self.T(
            query_emb.unsqueeze(1) - nk                            # (B, k, D)
        )

        # 3. 가중합 → agg_emb (prediction path)
        agg_emb = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        # 4. feature attribution (설명 전용)
        feature_imp, attn_w = self.feat_cross(query_emb, nk, evidence_w)

        return agg_emb, evidence_w, feature_imp, attn_w

    # ── 설명 인터페이스 (OTEvidenceSelector와 완전 동일) ────────

    def explain_evidence(
        self,
        evidence_w: torch.Tensor,   # (B, k)
        top_n: int = 3,
    ) -> List[Dict]:
        """
        이웃별 기여도 설명.
        OTEvidenceSelector.explain_evidence와 동일 인터페이스.
        """
        ew_np = evidence_w.detach().cpu().numpy()
        B, k  = ew_np.shape
        out   = []
        for b in range(B):
            w      = ew_np[b]
            sorted_idx = np.argsort(w)[::-1]
            top_n_list = [(int(i), float(w[i])) for i in sorted_idx[:top_n]]
            out.append({
                "top_neighbours":    top_n_list,
                "dominant_weight":   float(w.max()),
                "ignored_ratio":     float((w < 0.05).mean()),
                "entropy":           float(-(w * np.log(w + 1e-8)).sum()),
            })
        return out

    def explain_feature_match(
        self,
        feature_importance: torch.Tensor,   # (B, k, F)
        evidence_weights:   torch.Tensor,   # (B, k)
        col_names:          List[str],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        feature 수준 기여도 설명.
        OTEvidenceSelector.explain_feature_match와 동일 인터페이스.
        """
        fi_np = feature_importance.detach().cpu().numpy()
        ew_np = evidence_weights.detach().cpu().numpy()
        B, k, F = fi_np.shape
        out = []
        for b in range(B):
            # evidence_weight로 가중 평균 feature 중요도
            w_agg = (fi_np[b] * ew_np[b, :, None]).sum(0)  # (F,)
            total = w_agg.sum() + 1e-8
            ranked = sorted(
                enumerate(w_agg), key=lambda x: x[1], reverse=True
            )[:top_n]
            out.append({
                "top_features": [
                    {"feature": col_names[i] if i < len(col_names) else f"f{i}",
                     "importance": float(v / total)}
                    for i, v in ranked
                ]
            })
        return out