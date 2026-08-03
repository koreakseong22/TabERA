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


def _evidence_hyperspherical_scale(k: int) -> float:
    """evidence retrieval의 cosine softmax용 sharpness scale.

    CentroidLayer의 routing_scale(search_space.py의 adacos_fixed_scale,
    Zhang et al. 2019 AdaCos fixed-scale 공식 s=√2·log(C-1), C→centroid
    수 P로 치환)과 "정규화된 hyperspherical 공간에서 softmax sharpness를
    후보 개수로부터 계산한다"는 같은 원리를 쓴다 — 다만 routing_scale
    값 자체를 재사용하는 게 아니라(그건 P=centroid 수 기준으로 계산된
    값이라 여기 후보 개수(k=검색된 이웃 수)와 다른 축), evidence 쪽
    후보 개수(k)에 맞춰 같은 공식을 독립적으로 다시 적용한 것이다.
    P→C, k→C로 치환 근거의 성격 자체는 동일(검증된 이론이 아니라 유비
    기반 실용적 근사 — A/B로 반드시 확인 필요).
    """
    return max(1.0, math.sqrt(2) * math.log(max(k - 1, 1)))


class NeighborInteractionBlock(nn.Module):
    """[v2 후보 A] pooling(evidence_w 가중합) 이전에 k개 이웃 values
    (label_emb + T(query-neighbour))끼리 self-attention으로 섞는 블록.

    query token 없음(이웃끼리만 attend) — T()가 이미 query에 강하게
    의존한다는 게 diagnose_value_components로 확인된 바 있어서, query
    token까지 넣으면 "이웃 상호작용 효과"와 "query 의존 강화 효과"가
    섞여 원인 분리가 안 됨. FFN 없음(self-attn + residual + LN만) —
    mixing과 capacity 증가를 한번에 넣지 않기 위한 최소 개입.

    evidence_w는 이 블록과 완전히 무관하게 그대로 계산됨(AttentionAggregator.
    forward에서 유지) — 설명②로 노출되는 값이 이 블록의 존재 여부와
    상관없이 forward pass에서 실제 쓰인 것과 항상 일치해야 하므로.

    [스모크 테스트 확인 완료, 2026-07] shape/gradient/NaN 정상,
    neighbor_interaction_mode=None 경로는 v1과 수치적으로 100% 동일
    (bit-for-bit), neighbor 간 정보가 실제로 섞이는 것을 직접 교란
    실험으로 확인(neighbor 3 교란 시 neighbor 0 출력 변화량 0.29,
    interaction_free_baseline은 0.000000).
    """

    def __init__(self, embed_dim: int, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """values: (B, k, D) -> (B, k, D). k=1이면 self-attention이
        항등에 가깝게 수렴(이웃이 자기 자신만 봄) — 에러는 안 남."""
        attn_out, _ = self.mha(values, values, values, need_weights=False)
        return self.ln(values + attn_out)


class NeighborCapacityBaseline(nn.Module):
    """[v2 대조군 b] "파라미터/비선형성만 늘어서 좋아진 것 아니냐"를
    배제하기 위한 느슨한 baseline. 이웃끼리 섞지 않고(mixing 없음),
    각 이웃의 value를 독립적으로 넓은 MLP에 통과시켜 대략 비슷한
    파라미터 수를 태움. 파라미터 수를 attn 블록과 정확히 맞추지는
    않음(대략적 capacity 참고용) — 엄밀한 대조군은
    NeighborInteractionFreeBaseline(아래) 쪽.
    """

    def __init__(self, embed_dim: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = embed_dim * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.ln(values + self.mlp(values))


class NeighborInteractionFreeBaseline(nn.Module):
    """[v2 대조군 c, 핵심 necessity ablation] NeighborInteractionBlock과
    정확히 같은 파라미터 구성(같은 nn.MultiheadAttention 클래스/설정,
    파라미터 수 100% 동일 — 스모크 테스트로 확인)이되, attention을
    이웃별 identity(자기 자신에게만 attend)로 강제해서 이웃 간 정보
    교환을 구조적으로 차단. "attention이라는 연산 형태" 자체는 그대로
    두고 "이웃끼리 실제로 섞이는가"만 켜고 끄는 대조군.

    해석:
      attn > interaction_free ≈ v1        → mixing 자체가 원인
      attn ≈ interaction_free (둘 다 > v1) → capacity/projection 증가가 원인
      attn ≈ interaction_free ≈ v1         → 이 경로 전체가 병목이 아님
                                              (pooling bottleneck 가설 반증)

    [주의] 이 실험이 검증하는 건 "pooling bottleneck 존재 여부"이지,
    "Aggregator vs Head 전체 문제"의 완전한 답은 아님 — attn도 null이면
    후보 B(head-level cross-attention)로 넘어가야 함.
    """

    def __init__(self, embed_dim: int, n_heads: int = 2, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        k = values.shape[1]
        # 대각선만 0, 나머지는 -inf인 additive attn_mask → softmax 후 각
        # 위치가 자기 자신에게만 가중치 1을 줌(이웃 간 mixing 완전 차단).
        # in_proj/out_proj 파라미터는 NeighborInteractionBlock과 완전히
        # 동일하게 다 사용됨(파라미터 수 차이 0 — 스모크 테스트로 확인).
        mask = torch.full((k, k), float("-inf"), device=values.device, dtype=values.dtype)
        mask.fill_diagonal_(0.0)
        attn_out, _ = self.mha(values, values, values, attn_mask=mask, need_weights=False)
        return self.ln(values + attn_out)


class HeadCrossAttention(nn.Module):
    """[v2, head 재설계 최종안] AttentionAggregator의 고정 weighted-sum
    pooling을 완전히 대체. retrieve()(어떤 k개를 찾을지)와 value 구성
    (label_emb + T(query-neighbour), TabR 원본)은 그대로 유지하되, 그
    k개를 어떻게 예측에 반영할지를 head 내부의 단일 cross-attention이
    직접 맡음 — "무엇을 검색할지는 그대로, 사용 방식만 바꾼다".

    Q = query_emb, K = nk(이웃 key 임베딩, retrieval과 같은 공간),
    V = value(label_emb + T(query-neighbour)). n_heads=1, layer 1개
    (최소 개입 — capacity/optimization/regularization 변수를 한번에
    늘리지 않기 위함. 성공하면 그때 multi-head/multi-layer 고려).

    updated_query = query_emb + alpha * attn_out   (residual, CLS-token 방식)
      alpha는 학습 가능한 스칼라(기본) — alpha_override로 고정 가능.
      0으로 고정하면 updated_query = query_emb가 되어 자동으로
      query-only baseline이 재현됨(necessity ablation).

    evidence_w = 이 cross-attention의 실제 softmax weight(B,k) — attn_out을
    만드는 데 실제로 쓰인 값 그 자체이므로, 설명②가 다시 causal claim이
    될 수 있음(v1은 head가 agg_emb를 안 써서 descriptive claim으로
    재정의해야 했던 것과 대비되는 v2의 핵심 이득).

    head 쪽 배선: TabERA에서 agg_emb 자리에 updated_query를 넣고
    use_query_emb_in_head=False로 두면(기존에 이미 있던 플래그) head
    입력이 정확히 [updated_query ‖ context_emb]가 됨(2-branch, B안) —
    query_emb를 별도 branch로 안 넣는 이유는 residual 덕에 이미
    updated_query 안에 들어있기 때문(중복 방지).

    neighbor_source : [necessity/capacity 대조군, 생성자 시점 고정 —
      재학습 필요]
      "real"(기본값) — 실제 검색된 이웃.
      "learned_const" — K/V를 검색 결과 대신 학습 가능한 상수 토큰
        (nn.Parameter, (k,D))으로 완전히 대체. attention 모듈 자체
        (W_q/W_k/W_v/W_out) 파라미터 수는 "real"과 100% 동일 — 늘어나는
        건 상수 토큰 자체(k*embed_dim)뿐. "실제 검색 결과 없이도
        cross-attention이라는 형태/capacity만으로 좋아지는가"를 격리.
      "shuffled" — 매 forward마다(학습 중 포함) 배치 내에서 K/V를
        무작위로 섞음. "learned_const"와 다른 점: 매 배치 다른 진짜
        이웃 벡터 분포를 보되(그래서 "batch-level 통계"는 학습 가능),
        "이 query와 이 이웃의 실제 대응"만 학습 내내 원천적으로 불가능
        하게 만듦. "학습 후 post-hoc으로 섞는 것"(forward()의
        shuffle_neighbors 인자, 아래)과 다름 — 그건 "real"로 다 학습된
        모델에 대한 necessity 검증이고, 이건 애초에 대응을 본 적 없는
        상태로 재학습해야 의미 있는 별도 baseline.
    """

    def __init__(self, embed_dim: int, k: int, tasktype: str = "regression",
                 n_classes: Optional[int] = None, dropout: float = 0.0,
                 use_offset_correction: bool = True,
                 neighbor_source: str = "real",
                 alpha_override: Optional[float] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.tasktype = tasktype
        self.use_offset_correction = use_offset_correction

        if neighbor_source not in ("real", "learned_const", "shuffled"):
            raise ValueError(
                f"neighbor_source은 'real'/'learned_const'/'shuffled' 중 하나여야 합니다: {neighbor_source}"
            )
        self.neighbor_source = neighbor_source

        # value 구성 — AttentionAggregator(value_mode="default")와 동일 로직
        # (TabR 원본 유지, 여기서는 value_mode 변형은 안 둠 — 최소 개입).
        if tasktype in ("binclass", "multiclass"):
            if n_classes is None or n_classes < 2:
                raise ValueError(f"tasktype='{tasktype}'면 n_classes(2 이상) 필요")
            self.n_classes = n_classes
            self.label_encoder = nn.Embedding(n_classes, embed_dim)
        else:
            self.n_classes = None
            self.label_encoder = nn.Linear(1, embed_dim)

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

        # n_heads=1 고정 — 클래스 docstring 참고(설명②를 모호하게 만들지
        # 않기 위한 의도적 선택, capacity가 부족하면 그때 multi-head 검토).
        self.mha = nn.MultiheadAttention(embed_dim, num_heads=1, dropout=dropout, batch_first=True)

        if alpha_override is not None:
            self.register_buffer("alpha", torch.tensor(float(alpha_override)))
            self._alpha_is_learnable = False
        else:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self._alpha_is_learnable = True

        if neighbor_source == "learned_const":
            self.const_tokens = nn.Parameter(torch.randn(k, embed_dim) * 0.02)
        else:
            self.const_tokens = None

    def _encode_labels(self, neighbour_labels: torch.Tensor) -> torch.Tensor:
        if self.tasktype in ("binclass", "multiclass"):
            idx = neighbour_labels.round().long().clamp(0, self.n_classes - 1)
            return self.label_encoder(idx)
        else:
            return self.label_encoder(neighbour_labels.unsqueeze(-1).float())

    def forward(self, query_emb: torch.Tensor, nk: torch.Tensor,
                neighbour_labels: torch.Tensor,
                shuffle_neighbors: bool = False):
        """
        query_emb        : (B, D)
        nk                : (B, k, D) — 이웃 key 임베딩 (K로 사용)
        neighbour_labels  : (B, k)
        shuffle_neighbors : [post-hoc ablation, 재학습 불필요] True면 배치
          내에서 K/V를 무작위로 섞어(다른 샘플의 이웃을 붙여) query-이웃
          대응을 깨뜨림 — cross-attention이 "이 query의 진짜 이웃"에
          의존하는지 직접 검증(necessity). 학습 시에는 항상 False.
        반환: updated_query(B,D), evidence_w(B,k) — 실제 예측에 쓰인
          attention weight 그 자체, evidence_diag(dict).
        """
        B = query_emb.shape[0]

        if self.neighbor_source == "learned_const":
            # [capacity-only 대조군] 검색 결과 완전 무시 — label/T()로 만든
            # 실제 값은 여기서 아예 안 씀(섞이면 real 정보가 새어들어감).
            keys = self.const_tokens.unsqueeze(0).expand(B, -1, -1)
            vals = self.const_tokens.unsqueeze(0).expand(B, -1, -1)
        else:
            label_emb = self._encode_labels(neighbour_labels)      # (B,k,D)
            if self.use_offset_correction:
                offset_term = self.T(query_emb.unsqueeze(1) - nk)
                values = label_emb + offset_term
            else:
                values = label_emb
            keys, vals = nk, values
            # [수정] "shuffled"는 생성자 시점 고정 baseline(매 forward마다,
            # 학습 중에도 항상 섞음) — forward() 인자 shuffle_neighbors는
            # "real"로 다 학습된 모델에 대한 post-hoc necessity 검증이라
            # 서로 다른 것. 둘 중 하나라도 True/설정돼 있으면 섞음(중첩
            # 호출 시에도 안전하게 한 번만 섞임 — 매번 새 permutation).
            if self.neighbor_source == "shuffled" or shuffle_neighbors:
                perm = torch.randperm(B, device=query_emb.device)
                keys = keys[perm]
                vals = vals[perm]

        attn_out, attn_w = self.mha(
            query_emb.unsqueeze(1), keys, vals,
            need_weights=True, average_attn_weights=True,
        )
        attn_out = attn_out.squeeze(1)     # (B, D)
        evidence_w = attn_w.squeeze(1)     # (B, k) — 실제 예측에 쓰인 weight

        updated_query = query_emb + self.alpha * attn_out
        # [추가] attention entropy — Explanation②/faithfulness 분석,
        # v1 evidence_w와의 entropy 비교, neighbor utilization 분석에
        # 공통으로 쓸 수 있게 미리 계산해서 diag에 포함(호출부마다
        # 따로 계산 안 해도 되게).
        with torch.no_grad():
            _attn_entropy = float(
                (-(evidence_w * (evidence_w + 1e-8).log()).sum(-1)).mean().item()
            )
        evidence_diag = {
            "alpha": float(self.alpha.detach().item()),
            "attn_entropy_mean": _attn_entropy,
        }
        return updated_query, evidence_w, evidence_diag

    def explain_evidence(
        self,
        evidence_w: torch.Tensor,   # (B, k)
        top_n: int = 3,
    ) -> List[Dict]:
        """이웃별 attention weight 요약 — AttentionAggregator.explain_evidence와
        인터페이스는 동일하지만, 여기서는 evidence_w가 실제로 updated_query
        (= 예측에 쓰인 표현)를 만드는 데 쓰인 weight 그 자체이므로 "이
        이웃 때문에 이렇게 예측했다"는 causal claim으로 취급해도 됨(v1의
        AttentionAggregator.explain_evidence는 head가 agg_emb를 안 써서
        descriptive claim으로만 제한해야 했음 — 그 caveat이 여기서는
        구조적으로(head_attn_alpha_override로 강제 0이 아닌 한) 해소됨).
        neighbor_source="learned_const"일 때는 evidence_w가 실제 검색
        결과가 아닌 학습된 상수 토큰에 대한 가중치이므로, 이 경우
        explain_evidence 결과를 "검색된 이웃"으로 표시하면 안 됨 — 호출부
        (reproduce.py --explain)에서 self.neighbor_source를 확인해 문구를
        분기할 것.
        """
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


class AttentionAggregator(nn.Module):
    """
    TabR 방식 이웃 집계 (순수 retrieval-augmented prediction).

    Forward 출력
    ────────────
    agg_emb    : (B, D)   — attention-weighted aggregation (head 입력용)
    evidence_w : (B, k)   — 이웃별 attention weight (설명 ② 용)
    """

    def __init__(self, embed_dim, k, n_features, n_output, dropout=0.0,
                 use_offset_correction: bool = True,
                 tasktype: str = "regression", n_classes: Optional[int] = None,
                 evidence_temperature: float = 1.0,
                 learn_evidence_temperature: bool = False,
                 evidence_metric: str = "euclidean",
                 value_mode: str = "default",
                 neighbor_interaction_mode: Optional[str] = None,  # [v2, 추가]
                     # None(기본값, 기존과 100% 동일 — 하위 호환)/"attn"(후보 A)/
                     # "capacity_baseline"/"interaction_free_baseline"(대조군).
                     # pooling(evidence_w 가중합) 전에 k개 이웃 values끼리
                     # 상호작용시킬지 여부. evidence_w 계산 자체에는 관여 안 함
                     # — 설명②로 노출되는 값은 이 플래그와 무관하게 항상 forward
                     # pass에서 실제 쓰인 것과 일치. 클래스 docstring 참고.
                 interaction_n_heads: int = 2):  # [v2, 추가] neighbor_interaction_mode
                     # 가 "attn"/"interaction_free_baseline"일 때만 의미 있음.
            # [value_mode 설명, 이어짐] use_offset_correction=True일 때만
            # 의미 있음(use_offset_correction=False면 value=label_emb만 — 이게
            # "label_only" ablation, 이 파라미터와 별개로 이미 존재하던 플래그로
            # 커버됨). "default"(기존과 100% 동일, 하위호환): value = label_emb +
            # T(query-neighbour), 정규화 없이 그대로 더함. 동기: 실측(
            # diagnose_value_components)으로 T(query-neighbour) 항의 norm이
            # label_emb 항보다 평균 4.9배 크다는 게 확인됨(mfeat-zernike) —
            # concat 시절 embed_dim 메커니즘(query_emb가 CosFace-정규화된
            # context_emb를 norm 격차로 짓눌렀던 것)과 구조적으로 같은 패턴이
            # AttentionAggregator 내부 value 구성 단계에서 재현된 것으로 보임.
            # "offset_only": value = T(query-neighbour)만(label_emb 항 자체를
            # 뺌) — "지금 모델이 사실상 라벨 정보를 거의 안 쓰고 이것만 쓰고
            # 있는가"를 직접 검증. "balanced": value = LN(label_emb) +
            # LN(T(query-neighbour)) — 두 항을 각각 unit-scale로 맞춘 뒤 더함,
            # 스케일 불균형만 제거한 최소 개입.
        """
        use_offset_correction : True(기본값)면 TabR 원본 그대로
            value = label_emb + T(query - neighbour).
            False면 T()를 아예 생성하지 않고 value = label_emb만 사용
            (ablation용 — "T()가 실제로 기여하는가"를 재학습으로 검증하기 위함,
            evidence_w/retrieve()의 동작에는 영향 없음).

        tasktype / n_classes : [수정] TabR 원본(yandex-research/tabular-dl-tabr,
            bin/tabr.py)은 label_encoder를 조건부로 만듦 —
                nn.Linear(1, d_main) if n_classes is None else nn.Embedding(...)
            즉 regression(라벨이 진짜 연속형)일 때만 Linear를 쓰고, classification
            (라벨이 명목형 클래스 인덱스)일 때는 Embedding을 씀. 기존 TabERA
            구현은 이 분기 없이 항상 nn.Linear(1, embed_dim)을 썼음 — 오늘
            categorical feature에서 고친 것과 정확히 같은 문제(순서 없는
            명목형을 raw 정수로 취급)가 이웃 라벨에도 그대로 남아있었음.
            tasktype="regression"(기본값, 하위 호환)이면 이전과 동일하게
            nn.Linear(1, embed_dim). "binclass"/"multiclass"면 n_classes가
            반드시 필요하고 nn.Embedding(n_classes, embed_dim)을 씀.

            [하위 호환 범위] classification 모델의 label_encoder 파라미터
            구조가 바뀜(Linear(1,D) → Embedding(n_classes,D)) — 기존
            classification 체크포인트는 --from_saved_state로 로드 불가
            (regression 체크포인트는 영향 없음, 오늘 categorical embedding
            변경 때와 동일한 성격의 하위 호환 범위).

        evidence_temperature : [추가, 진단용] evidence_w = softmax(similarities
            / evidence_temperature). 기본값 1.0 = 기존과 100% 동일(하위 호환).
            similarities = -‖q-k‖²가 정규화 안 된 raw 유클리드 거리라, 학습
            중 query_emb norm이 커지면(실측 확인됨, 데이터셋별 3.7~213배)
            softmax가 포화돼 evidence_w가 사실상 1-NN으로 붕괴하는 현상이
            관찰됨(jasmine/credit-g 둘 다, entropy가 학습 초반부터 이미
            ln(k)=2.77 대비 크게 낮고 계속 더 낮아짐). CentroidLayer의
            routing_scale(AdaCos 공식, search_space.py의 adacos_fixed_scale)
            과 달리, 여기 similarities는 정규화 안 된 공간이라 같은 공식을
            바로 못 씀 — 그래서 우선 자유 파라미터로 두고 수동 스윕
            (reproduce.py --evidence_temperature_override)으로 sharpness
            자체가 원인인지부터 격리 검증. 값을 검증하면 HPO나 (q,k를
            먼저 정규화한 뒤 AdaCos류 공식을 쓰는) 더 근본적인 방식으로
            나중에 대체 검토.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.k = k
        self.use_offset_correction = use_offset_correction
        self.tasktype = tasktype
        # [2026-07] learn_evidence_temperature=True면 T를 학습 파라미터로 둥다.
        # [배경] 기본값 T=1.0은 HPO 탐색 대상도 아니어서 한 번도 조정된 적이 없었다.
        # 그런데 관측된 similarity margin이 0.03~0.68이라 softmax(sim/1.0)은
        # 산술적으로 거의 균등분포가 된다(n_eff/k≈1, 10/10 데이터셋).
        # 또 최적 sharpness가 k에 따라 다르다는 증거도 있다(k=4에선 sharp가
        # +1.4%p 우세, k=48에선 -14.1%p). k는 HPO가 정하고 T는 고정이니 둘이
        # 맞을 이유가 없다 — 모델이 직접 찾게 하는 것이 이 옵션의 목적.
        # softplus로 T>0을 보장하고, log_evidence_temperature=0으로 초기화하면
        # softplus(0)=0.693이므로 오프셋을 두어 시작값이 정확히 evidence_temperature
        # 와 같게 맞췄다(기존 실험과 동일 출발점).
        self.learn_evidence_temperature = learn_evidence_temperature
        if learn_evidence_temperature:
            import math as _math
            _t0 = float(evidence_temperature)
            # softplus(x)=t0 를 만족하는 x = log(exp(t0)-1)
            _x0 = _math.log(_math.expm1(_t0)) if _t0 < 20 else _t0
            self.log_evidence_temperature = nn.Parameter(torch.tensor(float(_x0)))
            self.evidence_temperature = _t0   # 참고용 스칼라(실제 사용은 property)
        else:
            self.evidence_temperature = evidence_temperature
        if value_mode not in ("default", "offset_only", "balanced", "offset_normalized",
                              "sum_normalized", "neighbor_only", "neighbor_offset",
                              "interaction", "interaction_only",
                              "interaction_ln"):
            raise ValueError(
                f"value_mode은 'default'/'offset_only'/'balanced'/'offset_normalized'/"
                f"'sum_normalized' 중 하나: {value_mode}"
            )
        if value_mode != "default" and not use_offset_correction:
            raise ValueError(
                "value_mode='offset_only'/'balanced'/'offset_normalized'/'sum_normalized'는 "
                "T()가 있어야(use_offset_correction=True) "
                "의미가 있습니다."
            )
        self.value_mode = value_mode

        # evidence_metric : [추가, 진단용] evidence_w를 계산할 유사도 공간.
        #   "euclidean"(기본값, 기존과 100% 동일 — 하위 호환): -‖q-k‖²,
        #     정규화 안 된 raw 유클리드 거리. 학습 중 query_emb norm이
        #     커지면(실측: epoch당 query_norm-distance_mean 상관 rho=0.998)
        #     softmax가 포화돼 evidence_w가 사실상 1-NN으로 붕괴하는 게
        #     확인됨(jasmine, evidence_temperature 스윕으로도 해결 안 됨 —
        #     고정 스칼라로는 학습 중 계속 커지는 norm을 못 따라잡음).
        #   "cosine": q,k를 CentroidLayer 라우팅과 동일하게 unit-norm
        #     정규화한 뒤 2·cos(q,k) — norm 자체가 유사도 계산에서 빠지므로
        #     원리적으로 이 collapse 메커니즘이 발생할 수 없음. softmax
        #     상수항(-2)은 어차피 없어지므로 생략(2·cos만 사용).
        #   "cosine_scaled": 위에 _evidence_hyperspherical_scale(k)를 곱함 —
        #     CentroidLayer의 routing_scale과 같은 원리(AdaCos fixed-scale,
        #     Zhang et al. 2019)를 evidence 후보 개수(k)에 독립적으로 적용
        #     (routing_scale 값 자체를 재사용하는 게 아님 — 축이 다름).
        if evidence_metric not in ("euclidean", "cosine", "cosine_scaled"):
            raise ValueError(
                f"evidence_metric은 'euclidean'/'cosine'/'cosine_scaled' 중 "
                f"하나여야 합니다: {evidence_metric}"
            )
        self.evidence_metric = evidence_metric

        # TabR 원본: label 임베딩 — regression(연속형)은 Linear,
        # classification(명목형 클래스)은 Embedding
        if tasktype in ("binclass", "multiclass"):
            if n_classes is None or n_classes < 2:
                raise ValueError(
                    f"tasktype='{tasktype}'면 n_classes(2 이상)를 반드시 줘야 합니다."
                )
            self.n_classes = n_classes
            self.label_encoder = nn.Embedding(n_classes, embed_dim)
        else:
            self.n_classes = None
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

        # ── [V2] Evidence Constructor ────────────────────────────────
        # [배경] 지금까지의 value는 전부 **집계 대상 자체**를 바꾸는 시도였다:
        #   label_only     그룹 라벨 요약        auroc 0.9799 (기준)
        #   offset_only    T(q − n_i)           차이 항 하나만
        #   neighbor_only  n_i (이웃 표현 자체)  auroc 0.9767 (−0.0032, p=0.01) ← 유의하게 악화
        # 즉 "무엇을 retrieval 하는가"는 여러 번 바꿔봤으나 효과가 없었다.
        #
        # [가설] 필요한 것은 이웃의 **표현**이 아니라 query–neighbor **관계**다.
        #   retrieve → (query와 비교) → reason → evidence
        # 현재는 가운데 두 단계가 없고 바로 평균을 낸다.
        #
        # v_i = n_i + MLP([q, n_i, n_i−q, |n_i−q|, q⊙n_i])
        #
        # ⚠ residual 형태를 쓴다. MLP가 0을 출력하면 `neighbor_only`와 동일해져
        #   학습 초기에 알려진 동작에서 출발한다(최적화가 쉽고, 실패해도
        #   기존보다 나쁘지 않은 하한이 있다).
        # ⚠ |n−q|는 부호 없는 거리 성분 — n−q만 쓰면 방향만 남아
        #   "얼마나 먼가"가 표현되지 않는다.
        # ⚠ q⊙n은 곱 상호작용 — 합/차만으로는 표현 못 하는 항이다.
        #   `offset_only`가 실패한 이유 중 하나가 이것일 수 있다(합성 검증에서
        #   offset의 이웃 간 pairwise cos가 +0.47로 가장 높았다 — 모든 이웃이
        #   q라는 공통 성분을 공유해 방향이 몰린다).
        #
        # ⚠ state_dict에 키가 추가되므로, 이 모듈이 없는 기존 체크포인트를
        #   strict=True로 로드하면 깨진다. value_mode='interaction'일 때만 생성.
        self.evidence_constructor = None
        if value_mode in ("interaction", "interaction_only", "interaction_ln"):
            d_in = embed_dim * 5
            d_hid = embed_dim * 2
            self.evidence_constructor = nn.Sequential(
                nn.Linear(d_in, d_hid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hid, embed_dim, bias=False),
            )
            # ⚠ zero-init은 `interaction`(α 없음)에서만 쓴다.
            #   `interaction_ln`은 r = tanh(α)·LN(raw)인데, MLP를 zero-init하면
            #     raw = 0  →  ∂loss/∂α = <grad, LN(raw)> = 0  →  α 고정
            #     α = 0    →  ∂loss/∂raw 에 α가 곱해짐        →  MLP도 고정
            #   서로가 서로를 0으로 묶는 **데드락**이 된다.
            #   실측: 5 seed 전부 α=0, ‖raw‖=0으로 학습 내내 정지.
            #   → α 게이트가 있으면 MLP는 일반 초기화를 쓴다. α=0이 이미
            #     "시작 시 v=n" 안전장치를 제공하므로 zero-init이 불필요하다.
            if value_mode == "interaction":
                nn.init.zeros_(self.evidence_constructor[-1].weight)

        # ── residual 안정화 (value_mode='interaction_ln') ─────────────
        #   v = n + α · LN(r)
        #
        # [배경] `interaction`(제어 없는 residual) 실측:
        #   proj비 0.109 / perp비 0.994  → residual의 99%가 n에 **직교**.
        #     즉 MLP는 neighbor amplifier가 아니라 실제로 새 방향을 만든다.
        #   그런데 ‖r‖이 19 → 4,953으로 40배 폭주하고 seed 간 100배 편차.
        #   → 실패 원인은 **표현력 부족이 아니라 크기 제어 실패**다.
        #
        # ⚠ `LayerNorm(n + r)`을 쓰면 안 된다 — neighbor의 절대 크기와
        #   evidence의 "강도"까지 함께 정규화해버린다. 지금 문제는
        #   **방향은 맞고 크기만 틀린 것**이므로 residual만 정규화한다.
        #
        # α는 스칼라 파라미터, tanh로 감싸 |α| < 1을 보장하고 0에서 시작한다.
        # → 학습 초기 v = n (neighbor_only와 동일), residual 폭주는 구조적으로 불가.
        self.ec_residual_ln = None
        self.ec_residual_alpha = None
        if value_mode == "interaction_ln":
            self.ec_residual_ln = nn.LayerNorm(embed_dim)
            # ⚠ α=0으로 시작하면 ∂loss/∂raw 에 tanh(0)=0이 곱해져 MLP gradient가
            #   0이 된다. 작은 양수에서 시작해 양방향 gradient를 살린다.
            #   tanh(0.1) ≈ 0.0997 이므로 residual 초기 기여는 ‖n‖의 10% 이하다.
            self.ec_residual_alpha = nn.Parameter(torch.full((1,), 0.1))

        self.dropout = nn.Dropout(dropout)

        # [v2, 추가] neighbor_interaction_mode : None(기본값, 기존과 100%
        # 동일 — 하위 호환)이면 pooling 전 어떤 이웃 간 상호작용도 없음
        # (v1 그대로). "attn"이면 NeighborInteractionBlock(후보 A, self-
        # attention among neighbours, query token 없음, FFN 없음).
        # "capacity_baseline"/"interaction_free_baseline"은 "attn"과
        # 비교하기 위한 대조군(각 클래스 docstring 참고) — 이웃 간 mixing이
        # 구조적으로 불가능하면서 attn과 비슷한 자리에 비슷한 성격의 학습
        # 가능한 변환을 넣음. 셋 다 evidence_w 계산에는 관여하지 않음 —
        # evidence_w는 항상 원래 similarities에서만 계산됨(위 2번 단계).
        valid_modes = (None, "attn", "capacity_baseline", "interaction_free_baseline")
        if neighbor_interaction_mode not in valid_modes:
            raise ValueError(
                f"neighbor_interaction_mode은 {valid_modes} 중 하나여야 합니다: "
                f"{neighbor_interaction_mode}"
            )
        self.neighbor_interaction_mode = neighbor_interaction_mode
        if neighbor_interaction_mode == "attn":
            self.neighbor_interaction = NeighborInteractionBlock(
                embed_dim, n_heads=interaction_n_heads, dropout=dropout
            )
        elif neighbor_interaction_mode == "capacity_baseline":
            self.neighbor_interaction = NeighborCapacityBaseline(embed_dim, dropout=dropout)
        elif neighbor_interaction_mode == "interaction_free_baseline":
            self.neighbor_interaction = NeighborInteractionFreeBaseline(
                embed_dim, n_heads=interaction_n_heads, dropout=dropout
            )
        else:
            self.neighbor_interaction = None

    def _encode_labels(self, neighbour_labels: torch.Tensor) -> torch.Tensor:
        """neighbour_labels (B, k) → label_emb (B, k, D).
        classification이면 반올림 후 long 캐스팅(부동소수점 오차 대비 —
        memory.update()가 labels.float()로 저장하므로 클래스 인덱스도
        float32로 들어와 있음, categorical feature 인코딩과 동일한 이유)
        해서 Embedding lookup, regression이면 기존처럼 Linear."""
        if self.tasktype in ("binclass", "multiclass"):
            idx = neighbour_labels.round().long().clamp(0, self.n_classes - 1)
            return self.label_encoder(idx)                     # (B, k, D)
        else:
            return self.label_encoder(neighbour_labels.unsqueeze(-1).float())  # (B, k, D)

    def forward(self, query_emb, nk, neighbour_labels):
        """
        Parameters
        ──────────
        query_emb        : (B, D)   — 임베더 출력
        nk               : (B, k, D)— 이웃 key 임베딩
        neighbour_labels : (B, k)   — 이웃 레이블

        [변경 이력] nv(이웃 value 임베딩) 파라미터를 제거함. 본문 로직은
        원래도 nv를 쓰지 않았음(TabR 원본 그대로 value=label_emb+T(...)만
        사용) — nv_utility_probe로 실측한 결과(mfeat-zernike/vehicle/
        credit-approval 3개 데이터셋), nv가 nk/label로 이미 설명되는
        잔차 이상의 추가 정보를 noise 대조군과 통계적으로 구분되게
        보여주지 못해, MemoryBank/TabERA에서 nv 자체를 완전히 제거하며
        여기 시그니처에서도 제거함. [v3 → 제거] context_emb 인자도 같은
        이유로 제거 — ContextOffsetFiLM/ContextEvidenceTemperature(centroid
        context로 offset representation/evidence temperature를 조건화하는
        실험)가 통제된 비교(dataset 1043, 동일 hyperparameter, 3-seed)에서
        전부 baseline보다 나쁘거나(FiLM) 뚜렷한 이득이 없어(Temperature,
        margin↔N_eff가 이미 -0.98로 자연 발생) 채택되지 않음 — 관련 실험
        기록은 프로젝트 문서 참고.
        """
        # 1. 유사도 계산 — evidence_metric에 따라 분기
        if self.evidence_metric == "euclidean":
            # TabR 원본 방식: -||q||² + 2(q·k) - ||k||²  (= -‖q-k‖², raw)
            similarities = (
                -query_emb.square().sum(-1, keepdim=True)
                + 2 * (query_emb.unsqueeze(1) @ nk.transpose(-1, -2)).squeeze(1)
                - nk.square().sum(-1)
            )
        else:
            # cosine / cosine_scaled: CentroidLayer 라우팅과 동일하게
            # q,k를 먼저 unit-norm 정규화 — norm이 유사도 계산에서
            # 완전히 빠지므로, query_emb norm이 학습 중 얼마나 커지든
            # (실측: epoch당 최대 89배) 이 계산 자체는 영향을 안 받음.
            q_n = F.normalize(query_emb, dim=-1)                     # (B, D)
            k_n = F.normalize(nk, dim=-1)                            # (B, k, D)
            cosine = (q_n.unsqueeze(1) @ k_n.transpose(-1, -2)).squeeze(1)  # (B, k)
            if self.evidence_metric == "cosine":
                # -‖q̂-k̂‖² = 2·cos(q̂,k̂) - 2 인데, softmax는 상수항(-2)에
                # 불변이라 생략 — 2·cos만 사용(리뷰 지적 반영).
                similarities = 2 * cosine
            else:  # "cosine_scaled"
                similarities = _evidence_hyperspherical_scale(self.k) * cosine

        # [진단용, 추가] "distance/score 값 자체가 큰가"(temperature로 해결)
        # vs "query/key norm이 학습 중 커져서"(normalization으로 해결) 원인을
        # 구분하려면 이 넷을 같이 봐야 함 — 하나만 보면 두 원인을 못 가른다는
        # 리뷰 지적 반영. metric이 뭐든 "-similarities"를 "distance"라고
        # 부르는 건 euclidean 모드 한정으로만 문자 그대로 정확하지만
        # (cosine 모드에선 부호 반전된 유사도일 뿐), 세 모드 다 "값이 작을
        # 수록 가깝다"는 방향은 같아서 추세 비교에는 그대로 씀. gradient
        # 불필요한 순수 통계라 no_grad.
        with torch.no_grad():
            evidence_diag = {
                "query_norm":    float(query_emb.detach().norm(dim=-1).mean().item()),
                "key_norm":      float(nk.detach().norm(dim=-1).mean().item()),
                "distance_mean": float((-similarities).detach().mean().item()),
                "distance_std":  float((-similarities).detach().std().item()),
            }
            # [Local Retriever 진단, 추가] similarity geometry — temperature와
            # 독립적인 원인 분리용. evidence_w(softmax 이후)가 균등해 보여도,
            # 그 원인이 (a) similarity 자체가 이미 거의 동률(sim1≈sim_k, margin
            # 작음 — retrieval이 애초에 비슷한 이웃만 찾음)인지, (b) similarity는
            # 벌어져 있는데 temperature가 커서 softmax가 눌린 것인지 구분
            # 불가능함 — margin을 직접 저장해서 이 둘을 분리. similarities는
            # 이미 계산된 텐서라 추가 forward 연산 없음(정렬만 추가).
            _sim_sorted = similarities.detach().sort(dim=-1, descending=True).values  # (B, k)
            evidence_diag["similarity_top1_per_sample"]   = _sim_sorted[:, 0]           # (B,) — 가장 가까운 이웃
            evidence_diag["similarity_bottomk_per_sample"] = _sim_sorted[:, -1]          # (B,) — 검색된 k개 중 가장 먼 이웃
            evidence_diag["similarity_margin_per_sample"] = _sim_sorted[:, 0] - _sim_sorted[:, -1]  # top1-topk
            # [Local Retriever 진단, 추가] margin만으로는 shape을 구분 못함 —
            # [0.95,0.94,...,0.91]과 [0.95,0.90,...,0.75]는 margin이 비슷해도
            # (전자는 완만한 감소, 후자는 급격한 감소) 완전히 다른 retrieval
            # geometry. std로 이 둘을 구분.
            evidence_diag["similarity_std_per_sample"] = similarities.detach().std(dim=-1)  # (B,)

        # [2026-07] 학습 모드면 softplus로 양수 보장된 T를 쓴다.
        _T = (F.softplus(self.log_evidence_temperature)
              if self.learn_evidence_temperature else self.evidence_temperature)
        evidence_w = F.softmax(similarities / _T, dim=-1)    # (B, k)
        if self.learn_evidence_temperature:
            evidence_diag["evidence_temperature_value"] = float(_T.detach())

        # dropout은 softmax 직후 적용.
        evidence_w = self.dropout(evidence_w)

        # 2. TabR 방식 value = label_emb + T(query - neighbour)
        #    (use_offset_correction=False면 T() 항 없이 label_emb만 사용 — "label_only" ablation)
        label_emb = self._encode_labels(neighbour_labels)      # (B, k, D)

        # ── [V2 재설계] neighbor 표현을 value로 쓰는 모드 ────────────
        # [배경] 현재 value는 label_emb(또는 +offset)이라 agg가
        #   "이 그룹에 어떤 label이 많은가"만 전달한다. 그런데 centroid
        #   배정도 query의 함수이므로 context_emb가 이미 같은 것을 말한다
        #   (실측: agg←context R² 0.75~0.97, agg의 query 대비 gain ≈ 0).
        #   즉 retrieval이 **local evidence extractor**가 아니라
        #   **centroid-conditioned label summary extractor**로 작동하고 있다.
        #
        # nk = 이웃의 key 임베딩(= 그 이웃의 query_emb, retrieval과 같은 공간).
        #   이걸 value로 쓰면 agg가 "주변 실제 샘플들이 어떤 표현인가"를
        #   담게 되어, label 축이 아니라 **manifold 기하** 축이 된다.
        #
        #   neighbor_only    values = nk                  이웃의 절대 표현
        #   neighbor_offset  values = nk + T(q − nk)      절대 + query 상대
        #
        # ⚠ 차원은 D로 유지한다(concat 아님) — agg_emb가 head의 branch
        #   슬라이스 폭과 맞아야 하므로. cat([nk, offset])은 2D가 되어
        #   head 구조까지 바꿔야 한다(별도 실험).
        # ⚠ 이것도 여전히 query의 결정론적 함수다. 목적은 "새 정보 추가"가
        #   아니라 **query가 표현하기 어려운 local structure를 명시적으로
        #   제공하는 inductive bias**다.
        # ⚠ offset_term은 아래 분기에서만 정의된다 — 진단 코드가
        #   use_offset_correction만 보고 참조하면 UnboundLocalError가 난다
        #   (neighbor_only는 use_offset_correction=True인데 T()를 안 씀).
        #   여기서 None으로 초기화하고, 진단은 `is not None`으로 판정한다.
        offset_term = None
        if self.value_mode == "interaction_only":
            # v_i = MLP([q, n_i, n_i−q, |n_i−q|, q⊙n_i])   ← residual 없음
            # [배경] `interaction`(residual)에서 MLP가 새 방향을 만들지 않고
            #   neighbor를 rescale하는 데 그칠 수 있다(ec_r_proj_ratio로 확인).
            #   residual 형태는 "n을 그대로 복사"가 가능해서 그 지름길이 있다.
            #   이 모드는 n을 보존하지 않으므로 **query–neighbor 관계를
            #   표현하지 않으면 아무것도 못 만든다.**
            # ⚠ 대신 zero-init 안전장치가 없어 학습 초기가 불안정할 수 있다.
            _q = query_emb.unsqueeze(1).expand_as(nk)
            _d = nk - _q
            values = self.evidence_constructor(
                torch.cat([_q, nk, _d, _d.abs(), _q * nk], dim=-1))
        elif self.value_mode == "interaction_ln":
            # v_i = n_i + α · LN(MLP([q, n_i, n_i−q, |n_i−q|, q⊙n_i]))
            _q = query_emb.unsqueeze(1).expand_as(nk)
            _d = nk - _q
            _raw = self.evidence_constructor(
                torch.cat([_q, nk, _d, _d.abs(), _q * nk], dim=-1))
            _alpha = torch.tanh(self.ec_residual_alpha)
            _r = _alpha * self.ec_residual_ln(_raw)
            values = nk + _r
            with torch.no_grad():
                _rn = _r.norm(dim=-1)
                _nn = nk.norm(dim=-1).clamp_min(1e-8)
                _nhat = nk / _nn.unsqueeze(-1)
                _proj = (_r * _nhat).sum(-1)
                _perp = (_r - _proj.unsqueeze(-1) * _nhat).norm(dim=-1)
                _rd = _rn.clamp_min(1e-8)
                evidence_diag["ec_residual_norm_mean"] = float(_rn.mean())
                evidence_diag["ec_residual_norm_std"]  = float(_rn.std())
                evidence_diag["ec_residual_rel_mean"]  = float((_rn / _nn).mean())
                evidence_diag["ec_cos_value_neighbor"] = float(
                    F.cosine_similarity(values, nk, dim=-1).mean())
                evidence_diag["ec_r_proj_ratio"] = float((_proj.abs() / _rd).mean())
                evidence_diag["ec_r_perp_ratio"] = float((_perp / _rd).mean())
                evidence_diag["ec_cos_r_neighbor"] = float(
                    F.cosine_similarity(_r, nk, dim=-1).mean())
                evidence_diag["ec_cos_r_query"] = float(
                    F.cosine_similarity(_r, _q, dim=-1).mean())
                evidence_diag["ec_alpha"] = float(_alpha)
                evidence_diag["ec_raw_norm_mean"] = float(_raw.norm(dim=-1).mean())
        elif self.value_mode == "interaction":
            # v_i = n_i + MLP([q, n_i, n_i−q, |n_i−q|, q⊙n_i])
            _q = query_emb.unsqueeze(1).expand_as(nk)          # (B, k, D)
            _d = nk - _q
            _feat = torch.cat([_q, nk, _d, _d.abs(), _q * nk], dim=-1)   # (B, k, 5D)
            _r = self.evidence_constructor(_feat)              # (B, k, D) residual
            values = nk + _r
            # ── Evidence Constructor 진단 ──────────────────────────
            # ⚠ 이 두 지표는 **게이트**다. ‖r‖ ≈ 0이면 MLP가 neighbor_only에서
            #   전혀 움직이지 않았다는 뜻이고, 그러면 성능·구조 지표를 해석할
            #   근거가 없다(zero-init residual이라 시작점이 정확히 0이다).
            #
            #   ‖r‖ ≈ 0            MLP가 안 배움 → 나머지 해석 무의미
            #   ‖r‖ 큰데 성능 그대로  배우긴 했으나 유용한 evidence가 아님
            #   ‖r‖ 크고 성능/구조 개선  Evidence Constructor 작동
            #
            # cos(v_i, n_i)는 "이웃 표현을 얼마나 재해석했는가"다.
            #   시작 1.0 → 학습이 진행되며 감소하면 재구성이 일어난 것.
            # ⚠ ‖r‖은 스케일 의존이므로 ‖r‖/‖n‖(상대값)도 함께 남긴다 —
            #   절대값만으로는 데이터셋/embed_dim 간 비교가 안 된다.
            with torch.no_grad():
                _rn = _r.norm(dim=-1)                          # (B, k)
                _nn = nk.norm(dim=-1).clamp_min(1e-8)
                evidence_diag["ec_residual_norm_mean"] = float(_rn.mean())
                evidence_diag["ec_residual_norm_std"]  = float(_rn.std())
                evidence_diag["ec_residual_rel_mean"]  = float((_rn / _nn).mean())
                evidence_diag["ec_cos_value_neighbor"] = float(
                    F.cosine_similarity(values, nk, dim=-1).mean())
                # ── residual 분해 — "scaling인가 새 방향인가" ──────────
                # ⚠ cos(v, n)만 보면 오해할 수 있다. v = n + r 에서 ‖n‖이 크면
                #   r이 새 방향이어도 cos(v,n)은 1에 가깝게 나온다.
                #   따라서 **r 자체**를 n 방향과 직교 방향으로 분해해야 한다.
                #
                #   r = proj_n(r) + perp(r)
                #     proj 비중 ≈ 1  →  r ≈ c·n. MLP가 interaction이 아니라
                #                        **neighbor amplifier**를 배운 것
                #     perp 비중 큼   →  실제로 새 방향을 만들고 있음
                #
                # cos(r, q)도 같이 본다 — query 방향으로만 몰리면
                # (offset_only에서 관측된 문제) 이웃별 구별이 사라진다.
                _nhat = nk / _nn.unsqueeze(-1)
                _proj = (_r * _nhat).sum(-1)                    # (B, k) 부호 있는 투영
                _perp = (_r - _proj.unsqueeze(-1) * _nhat).norm(dim=-1)
                _rd = _rn.clamp_min(1e-8)
                evidence_diag["ec_r_proj_ratio"] = float((_proj.abs() / _rd).mean())
                evidence_diag["ec_r_perp_ratio"] = float((_perp / _rd).mean())
                evidence_diag["ec_cos_r_neighbor"] = float(
                    F.cosine_similarity(_r, nk, dim=-1).mean())
                evidence_diag["ec_cos_r_query"] = float(
                    F.cosine_similarity(_r, _q, dim=-1).mean())
        elif self.value_mode in ("neighbor_only", "neighbor_offset"):
            if self.value_mode == "neighbor_only":
                values = nk
            else:
                if not self.use_offset_correction:
                    raise RuntimeError(
                        "value_mode='neighbor_offset'은 use_offset_correction=True가 "
                        "필요합니다(T()가 있어야 함).")
                offset_term = self.T(query_emb.unsqueeze(1) - nk)
                values = nk + offset_term
        elif self.use_offset_correction:
            offset_term = self.T(query_emb.unsqueeze(1) - nk)
            if self.value_mode == "offset_only":
                values = offset_term
            elif self.value_mode == "balanced":
                values = F.normalize(label_emb, dim=-1) + F.normalize(offset_term, dim=-1)
            elif self.value_mode == "offset_normalized":
                values = label_emb + F.normalize(offset_term, dim=-1)
            elif self.value_mode == "sum_normalized":
                values = F.normalize(label_emb + offset_term, dim=-1)
            else:  # "default"
                values = label_emb + offset_term
        else:
            values = label_emb

        # [2026-07, 추가] value 구성요소별 이웃 간 다양성 진단 —
        # "attention이 uniform이다"(n_eff/k≈1)를 고쳐도 aggregation 결과가
        # 바뀌지 않을 수 있다는 가설(value collapse) 검증용. 각 샘플의 k개
        # 이웃 벡터끼리의 평균 pairwise cosine을 label_emb / offset_term /
        # values 각각에 대해 따로 잼 — 이 셋을 분리해야
        #   (a) label_emb만 붕괴  → 그룹이 라벨 순수해서 생기는 문제
        #   (b) offset_term만 붕괴 → T(query-neighbour)가 이웃을 구분 못하는 문제
        #   (c) 둘 다 붕괴        → retrieval 후보 자체가 동질적
        # 를 구분할 수 있다. 값이 1에 가까우면 k개 이웃이 사실상 같은 벡터라,
        # evidence_w를 아무리 sharpen해도 Σw·v가 거의 안 변한다는 뜻.
        # no_grad + 순수 forward 통계라 학습에 영향 없음.
        with torch.no_grad():
            def _mean_pairwise_cos(x: torch.Tensor) -> Optional[torch.Tensor]:
                """x: (B, k, D) → (B,) 샘플별 k개 벡터 간 평균 pairwise cosine
                (대각 제외). k<2면 정의되지 않아 None."""
                _k = x.shape[1]
                if _k < 2:
                    return None
                _xn = F.normalize(x.detach(), dim=-1)
                _gram = torch.bmm(_xn, _xn.transpose(1, 2))          # (B, k, k)
                _off = _gram.sum(dim=(1, 2)) - _gram.diagonal(dim1=1, dim2=2).sum(dim=1)
                return _off / (_k * (_k - 1))

            def _rel_dispersion(x: torch.Tensor) -> Optional[torch.Tensor]:
                """x: (B, k, D) → (B,) 평균벡터 대비 상대적 퍼짐
                    mean_i ‖v_i - v̄‖ / ‖v̄‖
                [왜 pairwise cosine만으로는 부족한가] cosine은 **방향만** 본다.
                value = label_emb + T(query - neighbour)에서 T() 항은 모든 이웃에
                대해 query라는 공통 성분을 갖기 때문에, query 성분이 크면 k개
                벡터가 전부 비슷한 방향을 향해 pairwise cosine이 1에 가까워진다 —
                그래도 크기나 특정 좌표에서는 이웃별 차이가 남아 있을 수 있고,
                head는 그 차이를 쓸 수 있다. 이 지표는 크기 차이까지 포함해
                "실제로 얼마나 퍼져 있는가"를 스케일 무관하게 잰다.
                (0에 가까우면 k개가 사실상 같은 벡터, 1이면 평균 크기만큼 퍼짐)"""
                _k = x.shape[1]
                if _k < 2:
                    return None
                _x = x.detach()
                _mean = _x.mean(dim=1, keepdim=True)                  # (B, 1, D)
                _num = (_x - _mean).norm(dim=-1).mean(dim=1)          # (B,)
                _den = _mean.squeeze(1).norm(dim=-1).clamp_min(1e-12)  # (B,)
                return _num / _den

            evidence_diag["value_pairwise_cos_per_sample"] = _mean_pairwise_cos(values)
            evidence_diag["label_pairwise_cos_per_sample"] = _mean_pairwise_cos(label_emb)
            evidence_diag["offset_pairwise_cos_per_sample"] = (
                _mean_pairwise_cos(offset_term) if offset_term is not None else None
            )
            evidence_diag["value_rel_dispersion_per_sample"] = _rel_dispersion(values)
            evidence_diag["label_rel_dispersion_per_sample"] = _rel_dispersion(label_emb)
            evidence_diag["offset_rel_dispersion_per_sample"] = (
                _rel_dispersion(offset_term) if offset_term is not None else None
            )

        # [v2, 추가] pooling(evidence_w 가중합) 전에 이웃 간 상호작용/
        # 대조군 변환 적용. evidence_w는 위에서 이미 계산 완료 —
        # 이 블록과 무관. neighbor_interaction_mode=None이면 values는
        # 그대로 통과(기존 v1과 수치적으로 100% 동일 — 스모크 테스트 확인).
        if self.neighbor_interaction is not None:
            values = self.neighbor_interaction(values)

        # ── [Step 2 진단] T 사후 스윕용 원본 ──────────────────
        # agg_emb = Σ evidence_w_i · value_i 이므로, similarities와 (최종)
        # values가 있으면 재학습 없이 임의의 T에 대해
        #   agg(T) = softmax(similarities/T) @ values
        # 를 **정확히** 계산할 수 있다(근사 아님).
        # ⚠ 반드시 neighbor_interaction 적용 **후**의 values를 저장해야
        #   재계산 결과가 실제 agg_emb와 일치한다.
        # 이것으로 오프라인에서 훑을 수 있는 것:
        #   · within-centroid variance(T)  ← "T를 낮추면 agg가 centroid 평균에서
        #                                     벗어나는가" — 이후 가설들의 전제
        #   · CCA(query, agg(T)) / Δlogloss(T)
        # ⚠ 이는 "추론 시 T만 바꾼 것"이라 "T를 바꿔 학습한 것"과는 다르다.
        evidence_diag["similarities_raw"] = similarities.detach()   # (B, k)
        evidence_diag["values_raw"]       = values.detach()         # (B, k, D)
        evidence_diag["evidence_w_raw"]   = evidence_w.detach()     # (B, k)
        # ⚠ 0차원 스칼라면 export 쪽 torch.cat(dim=0)이 실패한다 —
        #   배치 크기만큼 펼쳐 (B,)로 내보낸다.
        _T_val = float(_T.detach()) if torch.is_tensor(_T) else float(_T)
        evidence_diag["evidence_T"] = torch.full(
            (similarities.shape[0],), _T_val,
            device=similarities.device, dtype=torch.float32)

        # 3. 가중합 → head 입력
        agg_emb = (evidence_w.unsqueeze(1) @ values).squeeze(1)  # (B, D)

        return agg_emb, evidence_w, evidence_diag

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

        label_emb = self._encode_labels(neighbour_labels)      # (B, k, D)
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
        """이웃별 attention weight(retrieval 결과) 요약.

        [명명 정정] 예전 docstring은 "이웃별 기여도 설명"이었으나, 이 세션의
        검증(agg_emb 제거 시 accuracy 거의 불변 — 4데이터셋×5seed necessity
        test)에서 이 weight가 실제로 prediction을 좌우한다는 근거가 부족함이
        확인됨. "기여도"라는 표현 자체가 causal claim("이 이웃 때문에 이렇게
        예측했다")을 함의하므로 부적절 — 이 반환값은 "모델이 무엇을 검색해서
        어떤 가중치를 줬는가"(retrieval evidence, descriptive)로만 취급할 것.
        "왜 이렇게 예측했는가"(predictive faithfulness, causal)의 근거로
        제시하지 말 것 — 그 claim은 현재 데이터로 뒷받침되지 않음.
        """
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

# ─────────────────────────────────────────────────────────────
# [2026-07, S-1A] Retrieval objective — candidate surrogate
# ─────────────────────────────────────────────────────────────

def soft_nearest_neighbor_loss(
    z: torch.Tensor,               # (B, D) query_retr (정규화 전)
    labels: torch.Tensor,          # (B,)
    group: "Optional[torch.Tensor]" = None,   # (B,) hard_assignment
    tau: float = 0.1,
) -> torch.Tensor:
    """
    Soft Nearest Neighbor loss — "가까울수록 같은 라벨일 확률이 높게".

    [왜 이것인가]
    최종 평가 목표는 rank informativeness(= acc(1위 이웃)/acc(평균 이웃))인데,
    그것은 순서 통계량이라 미분 가능한 학습 목표가 아니다. SNN은 "거리에 따른
    라벨 일치 확률의 감쇠"를 최적화하므로 목표와 형태가 가장 가깝다.
    ⚠ **등가는 아니다.** SNN은 확률 밀도, rank informativeness는 순서 통계량이다.
    문서·논문에서 끝까지 "candidate surrogate"로 표기하고, 학습 중 두 값의
    궤적을 함께 기록해 대응이 실제로 성립하는지 확인해야 한다.

    [왜 group 안에서만 계산하는가]
    전역으로 계산하면 단순 클래스 분리 문제로 퇴화한다. 그건 CE가 이미 잘 하는
    일이고(실측 그룹 purity 0.89~0.94), 그 방향으로 더 밀면 오히려 그룹 내부
    순위 정보가 사라진다(purity ↑ ⇒ rank informativeness ↓ 가 4/4에서 관찰됨).
    retrieval이 실제로 쓰는 후보 풀 = centroid 그룹이므로 그 안에서만 잰다.

    [붕괴 위험]
    tau가 너무 낮거나 오래 학습하면 클래스가 한 점으로 뭉쳐 purity=1,
    rank informativeness=1.00(평탄)이 된다 — V1과 같은 실패로 되돌아간다.
    uniformity_loss를 함께 쓰고, 학습 중 within-class variance를 감시할 것.
    """
    B = z.shape[0]
    if B < 3:
        return z.sum() * 0.0
    dev = z.device
    zn = torch.nn.functional.normalize(z, dim=-1)
    sim = (zn @ zn.T) / max(tau, 1e-6)                       # (B, B)

    eye = torch.eye(B, dtype=torch.bool, device=dev)
    same_label = labels.view(-1, 1) == labels.view(1, -1)
    if group is not None:
        same_group = group.view(-1, 1) == group.view(1, -1)
    else:
        same_group = torch.ones_like(eye)

    mask_all = same_group & ~eye                              # 분모
    mask_pos = mask_all & same_label                          # 분자
    valid = mask_pos.any(dim=-1) & mask_all.any(dim=-1)
    if not bool(valid.any()):
        return z.sum() * 0.0

    NEG = torch.finfo(sim.dtype).min / 4
    log_den = torch.logsumexp(sim.masked_fill(~mask_all, NEG), dim=-1)
    log_num = torch.logsumexp(sim.masked_fill(~mask_pos, NEG), dim=-1)
    return -(log_num - log_den)[valid].mean()


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang & Isola(2020) 스타일 uniformity — 표현이 구 위에 퍼지도록 유지.
    SNN 단독으로는 클래스가 한 점으로 붕괴할 수 있어 그 반대 압력을 준다.
    """
    if z.shape[0] < 2:
        return z.sum() * 0.0
    zn = torch.nn.functional.normalize(z, dim=-1)
    sq = torch.cdist(zn, zn).pow(2)
    iu = torch.triu_indices(zn.shape[0], zn.shape[0], offset=1)
    return sq[iu[0], iu[1]].mul(-t).exp().mean().clamp_min(1e-12).log()


def retrieval_space_diag(query_pred: torch.Tensor, query_retr: torch.Tensor) -> dict:
    """분기 거리 진단 — 분리가 실제로 일어났는가(§3.5)."""
    with torch.no_grad():
        a = torch.nn.functional.normalize(query_pred.float(), dim=-1)
        b = torch.nn.functional.normalize(query_retr.float(), dim=-1)
        cos = float((a * b).sum(-1).mean()) if a.shape == b.shape else float("nan")
        # linear CKA (회전·스케일 불변) — 차원이 달라도 계산 가능
        X = query_pred.float() - query_pred.float().mean(0)
        Y = query_retr.float() - query_retr.float().mean(0)
        xy = float((X.T @ Y).pow(2).sum())
        xx = float((X.T @ X).pow(2).sum()); yy = float((Y.T @ Y).pow(2).sum())
        cka = xy / max((xx * yy) ** 0.5, 1e-12)
    return {"cos_pred_retr": cos, "cka_pred_retr": cka}