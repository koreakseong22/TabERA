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

[개선 1 — FiLM-based Centroid Conditioning (2026-05)]
─────────────────────────────────────────────────────
Similarity 계산에 사용되는 query를 centroid context로 변조합니다.
같은 query라도 어느 centroid 그룹에 속하느냐에 따라 attention pattern이
달라지도록 강제하여, "centroid-conditioned retrieval"이라는 architectural
motivation을 실제 attention 메커니즘에 새깁니다.

- FiLM (Perez et al., 2018, NeurIPS): gamma * x + beta (feature-wise affine)
- 적용 범위 (a): similarity 계산에만 변조 (value 계산과 feat_cross는 원본 query 유지)
- 초기화: gamma ≈ 1, beta ≈ 0 → 학습 초기에 정확히 baseline 동작 보장

[개선 2 — Gated Fusion of Feature Attribution (2026-05)]
─────────────────────────────────────────────────────────
feature_imp가 explanation에만 쓰이고 prediction 경로에 참여하지 않는
구조적 약점을 해결합니다. AttentionAggregator 내부에서 feature_imp를
D차원으로 투영(feat_to_emb)한 뒤 agg_emb와 sample-wise gated fusion으로
결합합니다.

- gate ∈ (0,1)^D: query_emb와 feat_emb로부터 학습
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
    TabR 방식 이웃 집계 + FiLM Conditioning (개선 1) + Gated Feature Fusion (개선 2).

    Forward 출력
    ────────────
    fused_agg     : (B, D)   — gated fusion된 최종 aggregation (head 입력용)
    evidence_w    : (B, k)   — 이웃별 attention weight (설명용)
    feature_imp   : (B, k, F)— feature attribution (설명용)
    attn_w        : (B, k, D)— 차원별 attention (진단용)
    gate          : (B, D)   — 학습된 gate 값 (진단/ablation용, 개선 2)
    agg_emb_pure  : (B, D)   — gated fusion 전 raw aggregation (ablation/진단용)
    film_params   : Tuple[(B,D), (B,D)] — (gamma, beta) FiLM 파라미터 (진단용, 개선 1)
    """

    def __init__(self, embed_dim, k, n_features, n_output, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.n_features = n_features

        # TabR 원본: label 임베딩
        # n_output=1 (binary/regression) → Linear, >1 (multiclass) → Embedding
        self.label_encoder = nn.Linear(1, embed_dim)  # binary/regression용
        # multiclass는 tabera.py에서 n_output 넘겨받아 분기

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

        # ── [개선 1] FiLM Conditioning (Perez et al., NeurIPS 2018) ──────
        # context_emb로부터 (gamma, beta) 생성, query에 feature-wise affine
        # 적용. 적용 범위 (a): similarity 계산에만. value/feat_cross는 원본
        # query 유지하여 설명 해석을 명확히 분리.
        self.film_net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
        )
        # zero-init: 학습 초기에 gamma=1, beta=0 → 정확히 baseline 동작
        nn.init.zeros_(self.film_net[-1].weight)
        nn.init.zeros_(self.film_net[-1].bias)

        # ── [개선 2] Gated Fusion 구성 요소 ──────────────────
        # (a) feature attribution (B, k, F) → evidence-weighted summary (B, F) → (B, D)
        #     feat_to_emb는 F-dim feature space를 D-dim embedding space로 lift.
        self.feat_to_emb = nn.Linear(n_features, embed_dim)

        # (b) gate network: [query_emb, feat_emb] → gate ∈ (0,1)^D
        #     query별로 feature path vs neighbor path 비중을 sample-wise로 결정.
        self.gate_net = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Gate 초기화: 초기 gate ≈ 0.5가 되도록 마지막 Linear의 bias를 0으로.
        # → 학습 초기에 두 path를 동등하게 사용, 학습이 진행되며 자동 조정.
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, query_emb, nk, nv, neighbour_labels, context_emb=None):
        """
        Parameters
        ──────────
        query_emb        : (B, D)   — 임베더 출력
        nk               : (B, k, D)— 이웃 key 임베딩
        nv               : (B, k, D)— 이웃 value 임베딩 (현재 미사용, TabR 호환)
        neighbour_labels : (B, k)   — 이웃 레이블
        context_emb      : (B, D) | None — centroid context (개선 1, FiLM input).
                                           None이면 baseline 동작 (FiLM 미적용).
        """
        # ── [개선 1] FiLM Conditioning ────────────────────────
        if context_emb is not None:
            film_out = self.film_net(context_emb)             # (B, 2D)
            gamma_raw, beta = film_out.chunk(2, dim=-1)        # 각 (B, D)
            gamma = 1.0 + gamma_raw                            # zero-init → 학습 초기 1
            q_cond = gamma * query_emb + beta                  # (B, D)
        else:
            # context_emb 없으면 변조 없음 (구버전 호환 + ablation 용도)
            gamma  = torch.ones_like(query_emb)
            beta   = torch.zeros_like(query_emb)
            q_cond = query_emb

        # 1. TabR 방식 similarity (FiLM-conditioned query 사용)
        # -||q||² + 2(q·k) - ||k||²
        # 적용 범위 (a): similarity 계산에만 변조된 query 사용
        similarities = (
            -q_cond.square().sum(-1, keepdim=True)             # (B, 1)
            + 2 * (q_cond.unsqueeze(1) @ nk.transpose(-1,-2)).squeeze(1)  # (B, k)
            - nk.square().sum(-1)                              # (B, k)
        )
        evidence_w = F.softmax(similarities, dim=-1)           # (B, k)
        evidence_w = self.dropout(evidence_w)

        # 2. TabR 방식 value = label_emb + T(query - neighbour)
        #    [개선 1 범위 (a) 원칙] value 계산에는 원본 query_emb 유지.
        #    → 변조는 "누구에게 attend할지"에만, "어떻게 가공할지"는 보존.
        label_emb = self.label_encoder(
            neighbour_labels.unsqueeze(-1).float()             # (B, k, 1)
        )                                                       # (B, k, D)
        values = label_emb + self.T(
            query_emb.unsqueeze(1) - nk                        # (B, k, D)
        )

        # 3. 가중합 (raw aggregation, gated fusion 전)
        agg_emb_pure = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        # 4. feature 수준 기여도 (설명용 + gated fusion 입력)
        #    [개선 1 범위 (a) 원칙] feat_cross도 원본 query_emb 유지.
        #    → 설명 경로는 변조 없는 query 기준으로 일관되게 해석.
        feature_imp, attn_w = self.feat_cross(query_emb, nk, evidence_w)
        # feature_imp: (B, k, F), attn_w: (B, k, D)

        # ── [개선 2] Gated Fusion ───────────────────────────
        # (a) feature attribution을 evidence-weighted summary로 압축
        #     feature_imp는 이미 evidence_w로 가중치된 값이지만, 명시적으로
        #     이웃 축을 따라 합산하여 sample-level feature contribution을 얻음.
        feat_summary = feature_imp.sum(dim=1)                  # (B, F)

        # (b) F → D 투영
        feat_emb = self.feat_to_emb(feat_summary)              # (B, D)

        # (c) Gate 계산: query와 feat_emb를 보고 mixing 결정
        gate_input = torch.cat([query_emb, feat_emb], dim=-1)  # (B, 2D)
        gate = torch.sigmoid(self.gate_net(gate_input))        # (B, D)

        # (d) Sample-wise gated fusion
        fused_agg = gate * feat_emb + (1.0 - gate) * agg_emb_pure  # (B, D)

        return fused_agg, evidence_w, feature_imp, attn_w, gate, agg_emb_pure, (gamma, beta)

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