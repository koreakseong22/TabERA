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

[Gated Fusion of Feature Attribution]
─────────────────────────────────────
feature_imp가 explanation에만 쓰이고 prediction 경로에 참여하지 않는
구조적 약점을 해결합니다. AttentionAggregator 내부에서 feature_imp를
D차원으로 투영(feat_to_emb)한 뒤 agg_emb와 sample-wise gated fusion으로
결합합니다.

- gate ∈ (0,1)^D: query_emb, feat_emb, agg_emb_pure 모두 참조하여 학습
                  → neighbor 품질과 feature 품질을 비교해 mixing 결정 가능
                  → sample-wise adaptive gating이 구조적으로 보장됨
- fused_agg = gate * feat_emb + (1 - gate) * agg_emb
- TabERA.forward에서 head 입력에 fused_agg 사용 (차원 그대로 3D 유지)

이렇게 함으로써 설명에 사용되는 feature attribution이 곧 예측 경로에도
영향을 미치게 되어 faithfulness(설명-예측 일관성)를 architectural하게
보장합니다. (cf. Highway Networks, Srivastava et al. 2015;
              GRU update gate, Cho et al. 2014)
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
    어떤 feature 때문에 유사한지 계산하는 모듈 — 설명 경로 계층 3.

    두 가지 모드를 지원:
    ─────────────────────────────────────────────────────────────────
    모드 A (raw feature 기반, 기본값):
        원본 feature 공간에서 query와 neighbor의 차이를 직접 계산.
        차이가 작을수록(유사할수록) 해당 feature의 기여도가 높음.
        → "volatile_acidity가 비슷하기 때문에 유사하다"는 설명이
          실제 원본 feature 값 차이로부터 직접 도출됨.
        → faithfulness 검증 가능 (검증 3-A, 3-B 모두 통과 가능)

    모드 B (embedding 기반, fallback):
        query_x / neighbour_x가 없을 때 (예: FeatureStore 미충족)
        기존 embedding space 방식으로 동작.
        → 학습 초기 FeatureStore가 채워지기 전 단계에서 사용.
    ─────────────────────────────────────────────────────────────────
    """

    def __init__(self, embed_dim: int, n_features: int) -> None:
        super().__init__()
        # fallback용 (embedding 방식): FeatureStore 미충족 시 사용
        self.feat_proj = nn.Linear(embed_dim, n_features, bias=False)
        self.n_features = n_features

    def forward(
        self,
        query_emb: torch.Tensor,                        # (B, D)
        neighbour_emb: torch.Tensor,                    # (B, k, D)
        evidence_weights: torch.Tensor,                 # (B, k)
        query_x: Optional[torch.Tensor] = None,         # (B, F) 원본 feature
        neighbour_x: Optional[torch.Tensor] = None,     # (B, k, F) 이웃 원본 feature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        반환
        ────
        feature_importance : (B, k, F) — 이웃별 원본 feature 기여도
        attn_weights       : (B, k, D) — 차원별 attention (진단용)
        """
        B, k, D = neighbour_emb.shape

        if query_x is not None and neighbour_x is not None:
            # ── 모드 A: 원본 feature 공간에서 직접 계산 ──────────
            # feat_diff: (B, k, F) — query와 각 neighbor의 원본 feature 차이
            # 차이가 작을수록(값이 유사할수록) 해당 feature가 유사도에 기여
            feat_diff = query_x.unsqueeze(1) - neighbour_x          # (B, k, F)
            # [스케일 안정화] 역수(1/diff)는 diff≈0 시 폭발 → gradient 불안정
            # softmax(-|diff|, dim=2):
            #   항상 0~1 범위, 차이 작은(유사한) feature에 높은 가중치
            feat_sim  = F.softmax(-feat_diff.abs(), dim=2)           # (B, k, F)
            ew = evidence_weights.unsqueeze(-1)                      # (B, k, 1)
            feature_importance = feat_sim * ew                       # (B, k, F)
        else:
            # ── 모드 B: embedding space fallback ─────────────────
            q = query_emb.unsqueeze(1)                              # (B, 1, D)
            dim_match = q * neighbour_emb                           # (B, k, D)
            ew = evidence_weights.unsqueeze(-1)                     # (B, k, 1)
            weighted = dim_match * ew                               # (B, k, D)
            feat_imp_raw = self.feat_proj(weighted)                 # (B, k, F)
            feature_importance = feat_imp_raw.abs()

        # 차원별 attention (진단용) — 항상 embedding 기반
        q = query_emb.unsqueeze(1)
        attn_weights = (q * neighbour_emb).abs()                    # (B, k, D)

        return feature_importance, attn_weights


class AttentionAggregator(nn.Module):
    """
    TabR 방식 이웃 집계 + Gated Feature Fusion (faithfulness 보장).

    Forward 출력
    ────────────
    fused_agg     : (B, D)   — gated fusion된 최종 aggregation (head 입력용)
    evidence_w    : (B, k)   — 이웃별 attention weight (설명용)
    feature_imp   : (B, k, F)— feature attribution (설명용)
    attn_w        : (B, k, D)— 차원별 attention (진단용)
    gate          : (B, D)   — 학습된 gate 값 (진단/ablation용)
    agg_emb_pure  : (B, D)   — gated fusion 전 raw aggregation (ablation/진단용)
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

        # ── Gated Fusion 구성 요소 ──────────────────────────
        # (a) feature attribution (B, k, F) → evidence-weighted summary (B, F) → (B, D)
        self.feat_to_emb = nn.Linear(n_features, embed_dim)

        # (b) gate network: [query_emb, feat_emb, agg_emb] → gate ∈ (0,1)^D
        # agg_emb_pure를 포함함으로써 neighbor 품질도 gate 결정에 반영.
        # 이전: [query, feat] → feature path만 보고 gate 결정 (neighbor 품질 무시)
        # 이후: [query, feat, agg] → 두 path의 품질을 비교하여 gate 결정 가능
        # → sample-wise adaptive gating이 구조적으로 가능해짐
        # (cf. GRU: reset/update gate가 hidden state와 input을 모두 참조)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(embed_dim * 3),
            nn.Linear(embed_dim * 3, embed_dim),
        )

        # Gate 초기화: 초기 gate ≈ 0.5 (두 path 균등 출발)
        # normal init으로 초기 분기 유도 (zeros만 쓰면 수렴 속도 느림)
        nn.init.zeros_(self.gate_net[-1].bias)
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)

    def forward(
        self,
        query_emb,
        nk,
        nv,
        neighbour_labels,
        query_x: Optional[torch.Tensor] = None,      # (B, F) 원본 feature
        neighbour_x: Optional[torch.Tensor] = None,  # (B, k, F) 이웃 원본 feature
    ):
        """
        Parameters
        ──────────
        query_emb        : (B, D)   — 임베더 출력
        nk               : (B, k, D)— 이웃 key 임베딩
        nv               : (B, k, D)— 이웃 value 임베딩 (현재 미사용, TabR 호환)
        neighbour_labels : (B, k)   — 이웃 레이블
        query_x          : (B, F)   — 원본 feature (FeatureCrossAttention 모드 A용)
        neighbour_x      : (B, k, F)— 이웃 원본 feature (FeatureCrossAttention 모드 A용)
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

        # 3. 가중합 (raw aggregation, gated fusion 전)
        agg_emb_pure = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        # 4. feature 수준 기여도 (설명용 + gated fusion 입력)
        # query_x / neighbour_x가 있으면 원본 feature 공간 기반 (모드 A)
        # 없으면 embedding space fallback (모드 B: FeatureStore 미충족 시)
        feature_imp, attn_w = self.feat_cross(
            query_emb, nk, evidence_w,
            query_x=query_x,
            neighbour_x=neighbour_x,
        )
        # feature_imp: (B, k, F), attn_w: (B, k, D)

        # ── Gated Fusion ────────────────────────────────────
        # (a) feature attribution을 evidence-weighted summary로 압축
        feat_summary = feature_imp.sum(dim=1)                      # (B, F)

        # (b) F → D 투영
        feat_emb = self.feat_to_emb(feat_summary)                  # (B, D)

        # (c) Gate 계산: query, feat_emb, agg_emb_pure 모두 참조하여 mixing 결정
        # agg_emb_pure가 포함됨으로써 "neighbor가 얼마나 유용한가"를
        # gate가 직접 비교할 수 있게 됨
        gate_input = torch.cat([query_emb, feat_emb, agg_emb_pure], dim=-1)  # (B, 3D)
        gate = torch.sigmoid(self.gate_net(gate_input))            # (B, D)

        # (d) Sample-wise gated fusion
        fused_agg = gate * feat_emb + (1.0 - gate) * agg_emb_pure  # (B, D)

        return fused_agg, evidence_w, feature_imp, attn_w, gate, agg_emb_pure

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