"""
evidence.py — Attention 기반 이웃 집계 및 설명 모듈

OTEvidenceSelector(Sinkhorn OT)를 AttentionAggregator로 교체.

설계 원칙
─────────
- 미분 가능성: Scaled dot-product attention → 역전파 가능
- 설명가능성: attention weight = evidence_w (이웃별 기여도)
              인터페이스는 기존 OT와 완전 동일 유지
- 속도:       Sinkhorn 20 iter 제거 → 단순 softmax (~15배 빠름)
- 근거:       TabERA 원본 방식과 동일한 attention 집계
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
    Scaled Dot-Product Attention 기반 이웃 집계.

    OTEvidenceSelector를 대체합니다.

    작동 방식
    ─────────
    1. query와 k개 이웃 key 간 scaled dot-product attention
       → attention weight = evidence_w (B, k)
       → 이웃별 기여도를 자연스럽게 제공 (TabR 원본 방식)

    2. attention weight로 이웃 value를 가중 집계
       → agg_emb (B, D)

    3. FeatureCrossAttention으로 feature 수준 기여도 계산
       → feature_imp (B, k, F)

    기존 OTEvidenceSelector와 완전히 동일한 인터페이스 유지.
    """

    def __init__(
        self,
        embed_dim: int,
        k: int,
        n_features: int,
        n_heads: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim  = embed_dim
        self.k          = k
        self.n_features = n_features
        self.scale      = math.sqrt(embed_dim)

        # query/key/value 투영 (optional — TabR 원본은 투영 없이 raw 사용)
        # 여기서는 학습 가능한 투영 추가로 표현력 향상
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # feature 수준 기여도 (설명 계층 3)
        self.feat_cross = FeatureCrossAttention(embed_dim, n_features)

    def forward(
        self,
        query_emb: torch.Tensor,     # (B, D)
        nk:        torch.Tensor,     # (B, k, D) — 이웃 key 임베딩
        nv:        torch.Tensor,     # (B, k, D) — 이웃 value 임베딩
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        반환 (OTEvidenceSelector와 동일 인터페이스)
        ────
        agg_emb      : (B, D)    — 집계된 이웃 표현
        evidence_w   : (B, k)   — 이웃별 기여도 (attention weight)
        feature_imp  : (B, k, F) — feature 수준 기여도
        attn_w       : (B, k, D) — 차원별 attention (진단용)
        """
        B = query_emb.shape[0]

        # 투영
        q = self.q_proj(query_emb)   # (B, D)
        k = self.k_proj(nk)          # (B, k, D)
        v = self.v_proj(nv)          # (B, k, D)

        # Scaled dot-product attention
        # (B, D) → (B, 1, D) @ (B, D, k) → (B, 1, k) → (B, k)
        attn_logits = torch.bmm(
            q.unsqueeze(1), k.transpose(1, 2)
        ).squeeze(1) / self.scale                     # (B, k)

        evidence_w = F.softmax(attn_logits, dim=-1)   # (B, k) — 합=1
        evidence_w = self.dropout(evidence_w)

        # 가중 집계
        agg_emb = torch.bmm(
            evidence_w.unsqueeze(1), v
        ).squeeze(1)                                   # (B, D)

        # feature 수준 기여도 (설명 계층 3)
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
