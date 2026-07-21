"""
libs/tabera.py
============
TabERA — Dual-Space Prototype Explainable TabR Model.

  CentroidLayer        : Dual-Space Prototype + STE Routing + 주기적 Regroup
  AttentionAggregator  : TabR 방식 similarity 기반 이웃 집계
  MemoryBank           : Cross-group fallback (인접 centroid 확장 검색)

Forward 흐름
────────────
  X_query
    ↓  TabularEmbedder
  query_emb (B, D)
    ├─ CentroidLayer(query_emb) → context_emb (B, D) + routing
    └─ MemoryBank.retrieve → k neighbours (cross-group fallback 포함)
         ↓ AttentionAggregator(query_emb, nk, labels)
       agg_emb (B, D) + evidence_w (B, k)
    ↓
  [query_emb ‖ context_emb ‖ agg_emb] → PredictionHead → ŷ

[경량화: Gated Fusion 제거]
  실험 근거 (4개 데이터셋, seed=8):
    - gate_mean ≈ 0.5, std ≈ 0.01~0.08 → gate_net 미학습
    - feature_imp: ρ≈Random → post-hoc attribution(③, 현재 SHAP)으로 대체 완료
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
from libs.evidence     import AttentionAggregator, HeadCrossAttention


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
    """수치형/범주형 특성을 공유 임베딩 공간으로 투영.

    [수정 이력 1] 기존에는 numeric/categorical 구분 없이 전체 feature
    벡터를 LayerNorm+Linear 하나로 투영했음 — categorical feature가
    LabelEncoder 정수 코드(순서 없는 명목형)인데도 연속형 스칼라로
    취급돼, "카테고리 0과 3이 0과 1보다 멀다"는 우연한 인코딩 순서를
    실제 거리로 잘못 해석하는 문제가 있었음. TabZilla 29개 데이터셋
    baseline 비교에서 categorical 비중과 AUROC gap(baseline 대비)의
    상관관계가 견고하게 확인됨(Spearman rho=-0.63, p=0.0003).

    [수정 이력 2 — 속도] categorical 컬럼마다 별도 nn.Embedding + 파이썬
    for-loop 대신, 카디널리티를 이어붙인 단일 테이블 + offset으로
    벡터화(nomao에서 4.17배 속도 개선 실측).

    [수정 이력 3 — sum vs concat] Guo & Berkhahn(2016) 원 논문은 concat을
    쓰는데 처음엔 sum으로 결합했음 — `cat_combine="concat"`으로 전환
    가능 (컬럼별 embedding을 이어붙인 뒤 최종 Linear로 embed_dim 투영).

    [수정 이력 4 — TabM/ModernNCA 방식] 실제 최신 모델(TabM, ModernNCA,
    TabR)을 조사해보니 학습형 embedding(sum/concat) 대신 categorical엔
    **순수 one-hot**(학습 파라미터 없음), numeric엔 **PLE**(Piecewise
    Linear Encoding, Gorishniy et al. 2022 — quantile 기반 구간별
    선형 인코딩)를 쓰는 게 오히려 더 흔한 조합이었음. `cat_combine=
    "onehot"` + `num_embedding="ple"`로 이 조합 재현 가능. TabM(one-hot+
    PLE-계열)과 ModernNCA(one-hot+PLR)가 사실상 같은 전처리 조합이라
    하나로 둘 다 테스트됨.

    PLE 공식(Gorishniy et al. 2022): 컬럼별 quantile 기반 구간 경계
    b_0<...<b_T에 대해 z_t = clamp((x-b_{t-1})/(b_t-b_{t-1}), 0, 1) —
    x가 구간 t 아래면 0, 위면 1, 안이면 구간 내 상대 위치. 구간 경계는
    학습 데이터에서 미리 계산해 `num_bin_edges`로 전달받음(카테고리의
    cat_cardinalities와 같은 패턴).

    [수정 이력 5 — PLR(lite) 채택 확정, 기본값 변경] TabR(Gorishniy et al.
    2024)/ModernNCA(Ye et al. 2024)가 실제로 쓰는 PLR(lite) — 구간이 아니라
    학습 가능한 주기(periodic) 함수 + 전체 컬럼이 공유하는 Linear+ReLU.
    공식 구현(yandex-research/rtdl-num-embeddings)의 수식과 대조 검증 완료:
    ReLU(Linear(CosSin(2π·Linear(x,bias=False)))), lite는 바깥 Linear만 공유.

    데이터로는 방식마다 승패가 갈렸음(profb 같은 numeric feature가 아주
    적은 데이터셋에서는 PLR이 오히려 불안정 — auroc가 무작위 수준까지
    떨어진 사례 있음). 그럼에도 **"TabR/ModernNCA 계보를 잇는 retrieval
    기반 모델"이라는 정체성을 우선해 `cat_combine="onehot"` +
    `num_embedding="plr_lite"`를 기본값으로 확정**함 — 특정 데이터셋
    성능 최적화보다 아키텍처 일관성을 택한 결정. sum/concat/PLE는 여전히
    옵션으로 남아있어 필요시 비교 가능.

    cat_col_idx=None(기본값)이면 위 옵션들과 무관하게 이전(raw numeric
    전용) 경로로 100% 동일하게 동작 — raw-encoding 체크포인트 하위 호환.
    """
    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        cat_col_idx: Optional[List[int]] = None,
        num_col_idx: Optional[List[int]] = None,
        cat_cardinalities: Optional[List[int]] = None,
        cat_combine: str = "onehot",
        cat_embed_dim: int = 16,
        num_embedding: str = "plr_lite",
        num_bin_edges: Optional[torch.Tensor] = None,
        ple_d_embedding: int = 12,   # [추가] PiecewiseLinearEmbeddings의 feature별 embedding
                                      # 차원. rtdl_num_embeddings 공식 문서 권장 시작값
                                      # (activation=False 기준 d_embedding=12) — Optuna 탐색
                                      # 대상에 넣지 않고 고정. "컴팩트한 탐색 공간"이 목표인
                                      # 이 작업 취지상, PLR 3종처럼 검증 없이 탐색 차원을
                                      # 늘리는 실수를 반복하지 않기 위함 — 필요성이 실측되면
                                      # 그때 탐색 대상 추가 검토.
        plr_n_frequencies: int = 16,
        plr_freq_scale: float = 0.01,
        plr_out_dim: int = 8,
    ):
        super().__init__()
        self.cat_col_idx = list(cat_col_idx) if cat_col_idx else []
        self.num_col_idx = list(num_col_idx) if num_col_idx is not None else None
        self.cat_combine = cat_combine
        self.num_embedding = num_embedding
        n_num = len(self.num_col_idx) if self.num_col_idx is not None else 0

        if self.cat_col_idx or (n_num > 0 and num_embedding in ("ple", "plr_lite")):
            if cat_combine not in ("sum", "concat", "onehot", "none"):
                raise ValueError(f"cat_combine은 'sum'/'concat'/'onehot'/'none'이어야 합니다: {cat_combine}")
            if num_embedding not in ("linear", "ple", "plr_lite"):
                raise ValueError(f"num_embedding은 'linear'/'ple'/'plr_lite'여야 합니다: {num_embedding}")

            # ── categorical 처리 준비 ──
            self.cat_embeddings = None  # 구버전(파이썬 루프) 자리 표시(사용 안 함)
            self.cat_embed_table = None
            if self.cat_col_idx:
                if cat_cardinalities is None or len(cat_cardinalities) != len(self.cat_col_idx):
                    raise ValueError(
                        "cat_col_idx가 주어지면 cat_cardinalities도 같은 길이로 "
                        "줘야 합니다 (컬럼별 categorical 카디널리티)."
                    )
                cardinalities = [int(c) for c in cat_cardinalities]
                offsets = torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(cardinalities[:-1]), dim=0).tolist())
                    if len(cardinalities) > 1 else [0],
                    dtype=torch.long,
                )
                self.register_buffer("_cat_offsets", offsets, persistent=True)
                self.register_buffer(
                    "_cat_cardinalities", torch.tensor(cardinalities, dtype=torch.long), persistent=True
                )
                total_vocab = sum(cardinalities)
                if cat_combine == "sum":
                    self.cat_embed_table = nn.Embedding(total_vocab, embed_dim)
                elif cat_combine == "concat":
                    self.cat_embed_table = nn.Embedding(total_vocab, cat_embed_dim)
                elif cat_combine == "onehot":
                    self._onehot_total_vocab = total_vocab  # 학습 파라미터 없음

            # ── numeric 처리 준비 ──
            self.num_proj = None
            self.ple_n_bins = 0
            self.ple_d_embedding = 0
            self.plr_out_dim = 0
            if n_num > 0:
                if num_embedding == "ple":
                    if num_bin_edges is None:
                        raise ValueError(
                            "num_embedding='ple'면 num_bin_edges(학습 데이터에서 미리 계산한 "
                            "컬럼별 quantile 구간 경계, shape=(n_num, n_bins+1))를 줘야 합니다."
                        )
                    self.register_buffer("ple_edges", num_bin_edges.clone(), persistent=True)
                    self.ple_n_bins = num_bin_edges.shape[1] - 1
                    # [수정 — TabM 정합성] 기존엔 raw bin 벡터(z)를 그대로
                    # concat→final_proj(전체 feature 공유 Linear)에 넘겼음 —
                    # 이건 rtdl_num_embeddings의 PiecewiseLinearEncoding이지,
                    # TabM이 기본값으로 권장하는 PiecewiseLinearEmbeddings
                    # (activation=False)가 아님. 후자는 "Linear(PLE(x_i))"를
                    # feature마다 독립적으로 적용한다 — 즉 feature i, bin t마다
                    # 별도로 학습되는 (n_bins, d_embedding) 가중치가 있어야 함.
                    # 여기서 그 가중치를 직접 만들고, forward에서 z와 가중합
                    # (einsum)해서 feature별 embedding을 얻는다.
                    # [주의] TabM 논문(subsection A.3)의 "version B" 초기화는
                    # 약간 다른 디테일이 있을 수 있음 — 여기서는 nn.Linear
                    # 표준 초기화(Kaiming uniform 계열)에 준하는 일반적인
                    # 방식을 씀. bit-exact 재현이 필요하면 TabM 공식 구현체
                    # (rtdl_num_embeddings.PiecewiseLinearEmbeddings) 소스를
                    # 직접 대조해야 함.
                    self.ple_d_embedding = ple_d_embedding
                    self.ple_emb_weight = nn.Parameter(torch.empty(n_num, self.ple_n_bins, ple_d_embedding))
                    self.ple_emb_bias   = nn.Parameter(torch.zeros(n_num, ple_d_embedding))
                    bound = 1.0 / (self.ple_n_bins ** 0.5)
                    nn.init.uniform_(self.ple_emb_weight, -bound, bound)
                elif num_embedding == "plr_lite":
                    # PLR(lite) — TabR(Gorishniy et al. 2024): periodic embedding
                    # (컬럼별 학습 가능한 주파수) → 모든 컬럼이 공유하는 Linear
                    # → ReLU. PLE(구간 기반)와 완전히 다른 메커니즘 — 여기선
                    # "구간"이 아니라 "주기 함수"로 값을 표현함.
                    # 주파수(c)는 컬럼별로 따로 학습(자연스러운 스케일이 컬럼마다
                    # 다르므로), 그 뒤의 Linear+ReLU만 전체 컬럼이 공유 — 이게
                    # "lite"의 핵심(컬럼마다 별도 Linear를 두는 원래 PLR보다
                    # 파라미터 훨씬 적음, TabR 논문에서 "성능 손실 없이 가벼워짐"
                    # 이라고 보고).
                    self.plr_freq = nn.Parameter(torch.randn(n_num, plr_n_frequencies) * plr_freq_scale)
                    self.plr_linear = nn.Linear(2 * plr_n_frequencies, plr_out_dim)  # 컬럼 간 공유
                    self.plr_out_dim = plr_out_dim
                elif cat_combine == "sum":
                    self.num_proj = nn.Sequential(nn.LayerNorm(n_num), nn.Linear(n_num, embed_dim))
                # concat/onehot/none + linear numeric: raw x_num을 그대로 이어붙임 (num_proj=None)

            # ── 최종 결합 방식 결정 ──
            if cat_combine == "sum" and num_embedding == "linear":
                # 기존 sum 경로 그대로
                self.final_proj = None
            else:
                # concat/onehot/PLE는 전부 "이어붙인 뒤 최종 Linear" 패턴
                concat_dim = 0
                if n_num > 0:
                    if num_embedding == "ple":
                        concat_dim += self.ple_d_embedding * n_num
                    elif num_embedding == "plr_lite":
                        concat_dim += self.plr_out_dim * n_num
                    else:
                        concat_dim += n_num
                if self.cat_col_idx:
                    if cat_combine == "concat":
                        concat_dim += len(self.cat_col_idx) * cat_embed_dim
                    elif cat_combine == "onehot":
                        concat_dim += self._onehot_total_vocab
                    elif cat_combine == "sum":
                        concat_dim += embed_dim  # sum 결과 자체를 하나의 블록으로 이어붙임
                self.final_proj = nn.Sequential(nn.LayerNorm(concat_dim), nn.Linear(concat_dim, embed_dim))
        else:
            self.cat_embed_table = None
            self.cat_embeddings = None
            self.final_proj = None
            self.num_proj = None
            self.ple_edges = None
            # [하위 호환 필수] 기존 체크포인트의 state_dict 키가
            # "embedder.proj.0.weight" 형태로 저장돼 있음 — 속성 이름을
            # num_proj가 아니라 반드시 proj로 유지해야 --from_saved_state
            # 로딩이 깨지지 않는다.
            self.proj = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, embed_dim))

        self.blocks = nn.Sequential(*[ResidualMLP(embed_dim, embed_dim * 2, dropout) for _ in range(n_layers)])

    def _encode_categorical(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """categorical 컬럼들을 cat_combine 방식에 맞게 인코딩. (B, cat_dim) 반환."""
        if not self.cat_col_idx:
            return None
        x_cat = x[:, self.cat_col_idx].round().long()
        x_cat = torch.clamp(x_cat, min=torch.zeros_like(self._cat_cardinalities),
                             max=self._cat_cardinalities - 1)
        x_cat_global = x_cat + self._cat_offsets            # (B, n_cat) — 컬럼별 offset 적용

        if self.cat_combine == "onehot":
            # 학습 파라미터 없음. offset 덕분에 컬럼별 one-hot 구간이
            # 서로 안 겹치므로, sum이 곧 "컬럼별 one-hot을 이어붙인 것"과
            # 수학적으로 동일함 (block-diagonal 구조).
            onehot = F.one_hot(x_cat_global, num_classes=self._onehot_total_vocab).sum(dim=1)
            return onehot.float()
        else:
            cat_embs = self.cat_embed_table(x_cat_global)    # (B, n_cat, D) — 한 번의 gather
            if self.cat_combine == "sum":
                return cat_embs.sum(dim=1)                    # (B, embed_dim)
            else:  # concat
                B = x.shape[0]
                return cat_embs.reshape(B, -1)                # (B, n_cat * cat_embed_dim)

    def _encode_numeric(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """numeric 컬럼들을 num_embedding 방식에 맞게 인코딩."""
        if self.num_col_idx is None or len(self.num_col_idx) == 0:
            return None
        x_num = x[:, self.num_col_idx]
        if self.num_embedding == "ple":
            # PiecewiseLinearEmbeddings(activation=False) — TabM(Gorishniy et
            # al. 2024)이 기본값으로 권장하는 버전. bin 경계로 raw encoding
            # z를 만드는 것까지는 기존과 동일하지만, 그 z를 그대로 내보내는
            # 대신 feature별 학습 가중치(ple_emb_weight)와 가중합해서 각
            # feature마다 (ple_d_embedding,)짜리 벡터를 만든다 — "Linear(PLE
            # (x_i))"를 feature마다 독립적으로 적용하는 것과 동일.
            lo = self.ple_edges[:, :-1]                       # (n_num, n_bins)
            hi = self.ple_edges[:, 1:]                         # (n_num, n_bins)
            x_expand = x_num.unsqueeze(-1)                     # (B, n_num, 1)
            frac = (x_expand - lo) / (hi - lo + 1e-8)           # (B, n_num, n_bins)
            z = torch.clamp(frac, 0.0, 1.0)                     # (B, n_num, n_bins)
            # (B, n_num, n_bins) x (n_num, n_bins, d) → (B, n_num, d)
            emb = torch.einsum("bnk,nkd->bnd", z, self.ple_emb_weight) + self.ple_emb_bias
            return emb.reshape(x_num.shape[0], -1)              # (B, n_num*ple_d_embedding)
        elif self.num_embedding == "plr_lite":
            # PLR(lite) (TabR, Gorishniy et al. 2024): periodic embedding
            # (컬럼별 학습 주파수) → 전체 컬럼이 공유하는 Linear → ReLU.
            # 파이썬 루프 없음 — nn.Linear가 (B, n_num, 2k) 텐서의 마지막
            # 차원에 자동으로 브로드캐스팅되므로 "공유 Linear"가 자연스럽게
            # 구현됨.
            x_expand = x_num.unsqueeze(-1)                       # (B, n_num, 1)
            v = 2 * torch.pi * self.plr_freq * x_expand           # (B, n_num, k)
            periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, n_num, 2k)
            out = F.relu(self.plr_linear(periodic))               # (B, n_num, plr_out_dim) — Linear 공유
            return out.reshape(x_num.shape[0], -1)                # (B, n_num*plr_out_dim)
        else:
            return x_num  # linear 모드는 raw 값 그대로 (num_proj 또는 final_proj가 처리)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_proj is not None:
            parts = []
            num_repr = self._encode_numeric(x)
            if num_repr is not None:
                if self.num_proj is not None:
                    parts.append(self.num_proj(num_repr))
                else:
                    parts.append(num_repr)
            cat_repr = self._encode_categorical(x)
            if cat_repr is not None:
                parts.append(cat_repr)
            combined = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            return self.blocks(self.final_proj(combined))
        elif self.cat_embed_table is not None:
            # 기존 sum(+linear numeric) 경로 — final_proj 없이 바로 embed_dim
            emb = None
            if self.num_proj is not None:
                emb = self.num_proj(x[:, self.num_col_idx])
            cat_repr = self._encode_categorical(x)
            emb = cat_repr if emb is None else emb + cat_repr
            return self.blocks(emb)
        else:
            return self.blocks(self.proj(x))


class MemoryBank(nn.Module):
    """학습 임베딩 저장소 (KNN 검색용)."""
    def __init__(self, max_size: int, embed_dim: int, n_size_buckets: int = 4,
                 group_round_unit: int = 256, vectorized_fallback: bool = True):
        super().__init__()
        self.max_size = max_size
        # 그룹-크기 버킷 개수 (retrieve()의 normal_mask 처리용, 상수 반복 횟수)
        self.n_size_buckets = n_size_buckets
        # [실험용 파라미터화] retrieve()가 local_max_g를 반올림하는 단위.
        # 기존 코드는 이 값이 256으로 하드코딩되어 있었음 — 근거 주석 없음.
        # 기본값을 256으로 유지해 기존 동작과 100% 동일하게 시작하고,
        # 실측 실험(round_unit sweep)을 위해서만 외부에서 바꿀 수 있게 함.
        self._group_round_unit = group_round_unit
        # [실험용 파라미터화] cross-group fallback 경로를 샘플 단위
        # Python for-loop(+ .item() 동기화) 대신 배치 텐서 연산(bmm+topk)으로
        # 처리할지 여부. 실측(torch.profiler)으로 원래 경로가 cudaStreamSynchronize
        # 에 51%, GPU 실연산은 2%만 쓴다는 게 확인된 뒤 추가한 대안 경로.
        # 기본값 True = vectorized fallback 사용 
        self._vectorized_fallback = vectorized_fallback
        self.register_buffer("keys",   torch.zeros(max_size, embed_dim))
        self.register_buffer("labels", torch.zeros(max_size))
        self.register_buffer("ptr",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        # ── 정규화 캐시: retrieve() 내부의 반복 F.normalize 제거용 ──────
        # update() 시 O(B)로 증분 갱신, retrieve()에서는 그대로 gather만 함.
        self.register_buffer("_keys_norm", torch.zeros(max_size, embed_dim))
        # [추가 — 설명가능성/재현성] 슬롯 i가 X_train의 몇 번째 행인지 저장.
        # FeatureStore.sample_ids와 비교하면 "MemoryBank 슬롯과 FeatureStore
        # 슬롯이 같은 샘플을 가리키는가"를 통계적 근사(사전검증 1.5의 기존
        # percentile 비교) 없이 정확한 등식으로 확정할 수 있음. -1은
        # "아직 채워지지 않은 슬롯" 표시(정상적인 X_train 행 번호는 항상
        # 0 이상이므로 구분됨). state_dict()에 buffer로 잡히므로 best_state
        # 스냅샷/복원(libs/supervised.py) 시 자동으로 같이 따라감.
        self.register_buffer("sample_ids", torch.full((max_size,), -1, dtype=torch.long))

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
        # retrieve()의 라운딩 단위(self._group_round_unit)와 맞춰 내림.
        # 이전에는 여기 독립적으로 256이 하드코딩되어 있었음 — round_unit을
        # 바꾸는 실험을 할 때 이 값도 같이 따라가도록 결합.
        _ru = self._group_round_unit
        new_threshold = max((new_threshold // _ru) * _ru, max(2 * _ru, 512))
        self._outlier_threshold = new_threshold

    @torch.no_grad()
    def update(self, keys, labels, sample_ids=None):
        B   = keys.shape[0]
        ptr = self.ptr.item()
        end = min(ptr + B, self.max_size)
        n   = end - ptr
        self.keys[ptr:end]   = keys[:n].detach()
        self.labels[ptr:end] = labels[:n].float().detach()
        # 정규화도 여기서 O(B)로 한 번만 계산 (retrieve 매 배치 재계산 제거)
        self._keys_norm[ptr:end] = F.normalize(keys[:n].detach(), dim=-1)
        # [추가] sample_ids가 없으면(하위 호환 — 예전 호출부) -1로 남겨둠.
        # 새 학습 루프(libs/supervised.py)는 항상 X_train 행 번호(perm 슬라이스)를
        # 넘기도록 수정됨.
        if sample_ids is not None:
            self.sample_ids[ptr:end] = sample_ids[:n].detach().to(self.sample_ids.device)
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
        regroup_update 후 에폭당 1번만 호출 → retrieve 내부 변환 비용 제거.
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
        exclude_ids: "Optional[torch.Tensor]" = None,  # (B,) — 이 sample_id와 일치하는
            # MemoryBank 슬롯은 후보에서 제외(self-retrieval 방지). None이면 기존과
            # 동일(제외 없음, 하위호환). [주의] "이례적 경로"(그룹 하나가
            # self._outlier_threshold를 넘는 드문 경우, 아래 else 분기)는 이 인자를
            # 아직 반영하지 않음 — 실제로 트리거하는 조건을 이 세션에서 재현/검증할
            # 여유가 없어서 미구현으로 남김. 그 분기를 탈 만큼 큰 centroid가 있는
            # 데이터셋에서는 self-retrieval이 여전히 가능할 수 있음.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        완전 벡터화 k-NN 검색.

        반환: (nk, neighbour_labels, top_k_idx)
          nk              : (B, k, D) — 이웃 key 임베딩
          neighbour_labels: (B, k)    — 이웃 레이블 (TabR 방식 value 구성용)
          top_k_idx       : (B, k)    — MemoryBank 내 실제 인덱스 (FeatureStore 조회용)

        [변경 이력] 이전에는 nv(이웃이 저장 당시 가졌던 context_emb)도
        반환했으나, AttentionAggregator가 이를 전혀 사용하지 않았고
        (value = label_emb + T(query-neighbour)만 계산), nv_utility_probe로
        "nv가 nk/label로 설명되는 잔차 이상의 추가 정보를 갖는지" 실측한
        결과 3개 데이터셋(mfeat-zernike/vehicle/credit-approval) 모두에서
        noise 대조군과 통계적으로 구분되지 않았음 — 실질적 근거 없이
        저장/검색 비용만 발생시키고 있어 제거함.
        """
        n   = self.filled.item()
        B   = query.shape[0]
        D   = query.shape[1]
        dev = query.device

        keys_full   = self.keys[:n]    # (n, D)
        labels_full = self.labels[:n]  # (n,)
        q_norm      = F.normalize(query, dim=-1)  # (B, D)

        # ── 캐시 없거나 초기화 전 → 전체 검색 fallback ──────────
        cached = getattr(self, '_cached_groups', None)
        if hard_assignment is None or cached is None or n < k:
            keys_all = self._keys_norm[:n]  # 정규화 캐시 재사용
            sim      = q_norm @ keys_all.T
            if exclude_ids is not None:
                # (B, n) — 이 슬롯의 sample_id가 쿼리 자신의 sample_id와 같으면
                # 유사도를 -inf로 눌러 topk 후보에서 배제
                _self_mask = self.sample_ids[:n].unsqueeze(0) == exclude_ids.unsqueeze(1).to(dev)
                sim = sim.masked_fill(_self_mask, -1e9)
            _, idx   = sim.topk(min(k, n), dim=-1)
            idx      = idx.clamp(0, n - 1)
            neighbour_labels = labels_full[idx]              # (B, k)
            return keys_full[idx], neighbour_labels, idx

        # ── 완전 벡터화 (for loop 없음) ────────────────────────
        ha        = hard_assignment.to(dev)             # (B,)
        grp_sizes = self._cached_group_sizes[ha]        # (B,)

        # fallback 여부 판단: 그룹 크기 < k 인 샘플
        fallback_mask = grp_sizes < k                   # (B,) bool
        normal_mask   = ~fallback_mask

        # 결과 버퍼 (fallback 샘플은 zeros 유지)
        out_nk    = torch.zeros(B, k, D,          device=dev)
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

                _ru = self._group_round_unit
                local_max_g = ((local_max_g_raw + _ru - 1) // _ru) * _ru
                local_max_g = min(local_max_g, self._cached_groups.shape[1])

                Q_pad = torch.zeros(U_pad, max_q, D, device=dev)
                Q_pad[group_id, rank] = q_c_sorted

                cand_u  = self._cached_groups[uniq_p, :local_max_g]
                valid_u = cand_u >= 0
                safe_u  = cand_u.clamp(min=0, max=n - 1)

                keys_u = self._keys_norm[:n][safe_u.reshape(-1)].view(U_pad, local_max_g, D)

                sim_u = torch.bmm(Q_pad, keys_u.transpose(1, 2))
                sim_u = sim_u.masked_fill(~valid_u.unsqueeze(1), -1e9)

                if exclude_ids is not None:
                    # exclude_ids를 Q_pad와 같은 (U_pad, max_q) 레이아웃으로 재배치.
                    # 패딩 위치(실제 쿼리가 없는 (group_id,rank) 밖 자리)는 sentinel
                    # -1(실제 sample_id로 절대 안 나오는 값 — sample_ids 버퍼 초기값과
                    # 동일한 convention)로 남겨 어떤 후보와도 매칭 안 되게 함. 이
                    # 자리는 결과 사용 시 i_final_u[group_id, rank]로 걸러지므로
                    # 어차피 안 쓰이지만, 방어적으로 sentinel 처리.
                    ids_nm       = exclude_ids[nm_idx].to(dev)         # (Bn,)
                    ids_c_sorted = ids_nm[csort_idx]                   # (Bn,)
                    Ids_pad = torch.full((U_pad, max_q), -1, dtype=ids_c_sorted.dtype, device=dev)
                    Ids_pad[group_id, rank] = ids_c_sorted
                    cand_ids_u = self.sample_ids[safe_u.reshape(-1)].view(U_pad, local_max_g)  # (U_pad, local_max_g)
                    _self_mask_u = cand_ids_u.unsqueeze(1) == Ids_pad.unsqueeze(-1)  # (U_pad, max_q, local_max_g)
                    sim_u = sim_u.masked_fill(_self_mask_u, -1e9)

                k_eff = min(k, local_max_g)
                _, top_u  = sim_u.topk(k_eff, dim=-1)
                i_final_u = safe_u.unsqueeze(1).expand(-1, max_q, -1).gather(2, top_u)
                i_final_c_sorted = i_final_u[group_id, rank]

                final_pos = nm_idx[csort_idx]
                top_k_idx[final_pos, :k_eff]  = i_final_c_sorted
                out_nk[final_pos, :k_eff]     = keys_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff, D)
                out_labels[final_pos, :k_eff] = labels_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff)

            else:
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
                    _ru_t = self._group_round_unit
                    local_max_g_tier = ((local_max_g_tier_raw + _ru_t - 1) // _ru_t) * _ru_t
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

            if self._vectorized_fallback:
                # ── 벡터화 경로 (실측으로 확인된 병목 제거용) ──────────
                # 원래 경로는 fallback 샘플마다 Python for-loop + .item()
                # 동기화로 처리했음 (torch.profiler 실측: cudaStreamSynchronize
                # 가 self CPU 시간의 51%, 실제 GPU 연산은 2%뿐). 여기서는
                # 동일한 후보 집합·동일한 topk 선택을 배치 텐서 연산
                # (bmm/gather + masked topk)으로 한 번에 처리한다.
                # [주의] 이 경로가 원래 경로와 "완전히 동일한 결과"를 내는지는
                # 아직 벤치마크 스크립트의 정확성 검증으로만 확인된 가설임 —
                # 이 주석만으로 정확성이 보장되는 것은 아님.
                if ext is not None:
                    cand_ext  = ext[ha_fb]                          # (Bf, max_eg)
                    ext_sizes = self._cached_extended_sizes[ha_fb]  # (Bf,)
                    valid_ext = (cand_ext >= 0)
                    safe_ext  = cand_ext.clamp(min=0, max=n - 1)

                    still_small = ext_sizes < k
                    use_ext     = ~still_small

                    if use_ext.any():
                        ext_idx   = use_ext.nonzero(as_tuple=True)[0]      # (Bs,)
                        max_eg    = safe_ext.shape[1]
                        k_eff_ext = min(k, max_eg)

                        q_sel     = q_fb[ext_idx]                          # (Bs, D)
                        safe_sel  = safe_ext[ext_idx]                      # (Bs, max_eg)
                        valid_sel = valid_ext[ext_idx]                     # (Bs, max_eg)

                        keys_sel = self._keys_norm[:n][safe_sel.reshape(-1)] \
                                       .view(ext_idx.shape[0], max_eg, D)
                        sim_sel  = torch.bmm(
                            q_sel.unsqueeze(1), keys_sel.transpose(1, 2)
                        ).squeeze(1)                                      # (Bs, max_eg)
                        sim_sel  = sim_sel.masked_fill(~valid_sel, -1e9)

                        _, top_sel   = sim_sel.topk(k_eff_ext, dim=-1)     # (Bs, k_eff_ext)
                        real_idx_sel = safe_sel.gather(1, top_sel).clamp(0, n - 1)

                        final_pos_sel = fb_idx[ext_idx]
                        top_k_idx[final_pos_sel, :k_eff_ext]  = real_idx_sel
                        out_nk[final_pos_sel, :k_eff_ext]     = keys_full[real_idx_sel.reshape(-1)] \
                            .view(-1, k_eff_ext, D)
                        out_labels[final_pos_sel, :k_eff_ext] = labels_full[real_idx_sel.reshape(-1)] \
                            .view(-1, k_eff_ext)

                    if still_small.any():
                        ss_idx   = still_small.nonzero(as_tuple=True)[0]   # (Bt,)
                        q_ss     = q_fb[ss_idx]                            # (Bt, D)
                        keys_all = self._keys_norm[:n]
                        k_eff_ss = min(k, n)

                        sim_all    = q_ss @ keys_all.T                     # (Bt, n)
                        _, idx_all = sim_all.topk(k_eff_ss, dim=-1)
                        idx_all    = idx_all.clamp(0, n - 1)

                        final_pos_ss = fb_idx[ss_idx]
                        top_k_idx[final_pos_ss, :k_eff_ss]  = idx_all
                        out_nk[final_pos_ss, :k_eff_ss]     = keys_full[idx_all.reshape(-1)] \
                            .view(-1, k_eff_ss, D)
                        out_labels[final_pos_ss, :k_eff_ss] = labels_full[idx_all.reshape(-1)] \
                            .view(-1, k_eff_ss)
                else:
                    keys_all  = self._keys_norm[:n]
                    k_eff_all = min(k, n)
                    sim_all    = q_fb @ keys_all.T                         # (Bf, n)
                    _, idx_all = sim_all.topk(k_eff_all, dim=-1)
                    idx_all    = idx_all.clamp(0, n - 1)

                    top_k_idx[fb_idx, :k_eff_all]  = idx_all
                    out_nk[fb_idx, :k_eff_all]     = keys_full[idx_all.reshape(-1)] \
                        .view(-1, k_eff_all, D)
                    out_labels[fb_idx, :k_eff_all] = labels_full[idx_all.reshape(-1)] \
                        .view(-1, k_eff_all)

            elif ext is not None:
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
                            out_labels[b_pos] = labels_full[idx_all]
                            top_k_idx[b_pos]  = idx_all
                        else:
                            sim_e = q_e @ keys_e.T                           # (1, valid)
                            _, top_e = sim_e.topk(min(k, keys_e.shape[0]), dim=-1)
                            real_idx = si_e[vm_e][top_e.squeeze(0)]          # (k,)
                            real_idx = real_idx.clamp(0, n - 1)
                            kk = real_idx.shape[0]
                            out_nk[b_pos, :kk]     = keys_full[real_idx]
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
                    out_labels[b_pos] = labels_full[idx_all]
                    top_k_idx[b_pos]  = idx_all

        return out_nk, out_labels, top_k_idx

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
    - forward()가 embedder(X)에 넘기는 것과 동일한 X를 그대로 저장
      [정정] 이전 문서엔 "역정규화 완료값"이라 적혀 있었으나, 실제
      forward()의 `self._feature_store.update(X)` 호출은 embedder(X)에
      쓰인 것과 같은 텐서 X를 그대로 넘긴다 — 별도 역정규화 과정 없음.
      즉 _store[i]를 다시 embedder()에 통과시키면(refresh_memory_keys)
      추가 전처리 없이 그 자체로 재현 가능한 값이 나옴. 사람이 읽는
      원 스케일 표시(예: credit_amount=3050)는 저장 시점이 아니라
      출력 시점(print_explanation의 quantile_transformer 역변환)에
      이루어짐.
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
        # [추가 — 설명가능성/재현성] MemoryBank.sample_ids와 짝을 이루는
        # X_train 행 번호. nn.Module이 아니라 buffer로는 못 잡으니, 저장/
        # 복원 시 feature_store_state 튜플에 같이 실어 날라야 함
        # (reproduce.py checkpoint 저장/--from_saved_state 복원,
        # libs/supervised.py의 best_feature_store 스냅샷 세 군데 모두).
        self._sample_ids = torch.full((max_size,), -1, dtype=torch.long)

    @torch.no_grad()
    def update(self, X_raw: torch.Tensor, sample_ids: Optional[torch.Tensor] = None) -> None:
        B   = X_raw.shape[0]
        end = min(self._ptr + B, self.max_size)
        n   = end - self._ptr
        self._store[self._ptr:end] = X_raw[:n].detach().cpu().float()
        if sample_ids is not None:
            self._sample_ids[self._ptr:end] = sample_ids[:n].detach().cpu()
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
    loss_weights      : 보조 손실 가중치 {'diversity': .., 'commitment': .., 'codebook': ..(선택, 없으면 0)}
    column_names      : 특성 컬럼명 (설명 출력용)
    cat_col_idx       : categorical 컬럼의 인덱스 목록 (None이면 전체를
                        수치형으로 취급 — 기존 동작과 동일, 하위 호환)
    num_col_idx       : numeric 컬럼의 인덱스 목록 (cat_col_idx와 함께 줘야 함)
    cat_cardinalities : cat_col_idx와 같은 순서의 카디널리티 목록
                        (nn.Embedding 테이블 크기 결정용)
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
        exclude_self_retrieval: bool = False,  # [추가] True면 MemoryBank 검색 시
            # 쿼리 자신과 sample_id가 같은 슬롯(이전 epoch에 저장해둔 자기 자신)을
            # 후보에서 제외. 기본 False(하위호환, 기존 체크포인트/재현 결과와 동일
            # 동작). 동기: self-retrieval이 실측 결과 데이터셋마다 크게 다름
            # (jasmine≈0.1~0.3%, mfeat-zernike≈20~27%) — MemoryBank가 label을 그대로
            # 저장/반환하므로(TabR 방식 value 구성) self-retrieval 시 그 슬롯의
            # neighbour_label은 자기 자신의 진짜 정답. 다만 agg_emb의 predictive
            # null 결과와는 뚜렷한 상관을 못 찾음(self-retrieval 최고인 데이터셋이
            # agg-only 성능도 최고이거나 최저이지 않았음) — 그래서 이 옵션은 "결론을
            # 바꾸기 위해서"가 아니라 순수하게 구현 정확성(retrieval에서 자기 자신을
            # 배제하는 게 일반적인 관례) 차원에서 추가. [주의] "이례적 경로"
            # (centroid 그룹 하나가 매우 큰 드문 경우, MemoryBank.retrieve() 참고)는
            # 이 옵션을 아직 반영 안 함 — 검증 없이 손대지 않기로 함.
        # [수정] AttentionAggregator의 이웃 라벨 인코딩(label_encoder)이
        # TabR 원본처럼 classification(nn.Embedding)/regression(nn.Linear)을
        # 구분하려면 tasktype이 필요함 — 기존엔 n_output만 받아서 binclass
        # (n_output=1)와 regression(n_output=1)을 구분할 방법이 없었음.
        # tasktype="regression"(기본값)이면 이전과 100% 동일하게 동작
        # (하위 호환 — regression 체크포인트는 영향 없음).
        tasktype: str = "regression",
        n_classes: Optional[int] = None,
        routing_scale: float = 1.0,
        use_context_emb: bool = True,
        use_query_emb_in_head: bool = True,
        detach_context_grad: bool = False,
        use_ema_codebook: bool = False,
        ema_decay: float = 0.99,
        blockwise_layernorm: bool = False,
        head_branch_l2norm: bool = False,  # [v1.1, 추가] head 입력 직전(concat
            # 전) query/context/agg 각 branch를 sample-wise unit-L2-norm으로
            # 정규화. 기본값 False(기존과 100% 동일 — 하위호환). 동기: linear
            # probe 실측(1043/31)에서 concat(q+c+a)가 최고 단일 branch보다도
            # 낮게 나오는 현상이 branch별 L2-normalize만으로 상당 부분(1043)~
            # 거의 완전히(31) 회복됨을 확인 — StandardScaler(LayerNorm과 유사,
            # 차원별 z-score)는 오히려 31에서 더 악화시켜서, "차원별 분산"이
            # 아니라 "branch 전체 크기(norm) 격차"가 관련 있다는 쪽을 가리킴.
            # 다만 이건 probe(고정된 표현에 대한 사후 선형 분류기) 수준의
            # 관찰이라 "L2 정규화가 도움이 된다"까지만 보여주지 "scale
            # imbalance가 원인이다"를 증명하진 않음 — end-to-end 재학습으로
            # 실제 정확도가 따라오는지 확인 필요(이 플래그가 그 검증용).
            # blockwise_layernorm(학습되는 affine 포함 LayerNorm)과는 별개 —
            # 원 probe 실험 그대로 재현하려면 이것만 켜고 blockwise_layernorm은
            # 기본값(False) 유지 권장. 둘 다 켜면 LN 적용 후 L2-normalize(아래
            # forward() 참고) — 그 조합 자체는 검증 안 된 추가 실험임.
        fusion_mode: str = "concat",   # [추가] head가 [query,context,agg]를 합치는 방식.
            # "concat"(기본값, 기존과 100% 동일 — 하위 호환): [query‖context‖agg]를
            #   이어붙여 공유 MLP에 통과. freeze_encoder_retrain_head 5-seed 실험
            #   (mfeat-zernike, embed_dim=256, evM_cosine, sharedLN/blockLN 둘 다)에서
            #   인코더 고정 + head 백지 재학습을 해도 원래 공동학습 head와 통계적으로
            #   구분 안 되는 정확도로 수렴함(양쪽 다 paired p>0.4, d<0.2) — "정보는
            #   linear probe로 확인될 만큼 있는데 concat+공유 MLP 구조 자체가 그
            #   정보를 못 끌어쓴다"는 가설(시나리오 A)을 뒷받침. "residual"은 그
            #   가설에 대한 직접 대응책: concat 대신 z = LN(q) + α·LN(c) + β·LN(a)
            #   (α,β는 학습 가능한 스칼라)로 합쳐서 embed_dim 크기의 z 하나만
            #   MLP에 넣음 — 공유 MLP의 첫 Linear가 특정 브랜치의 concat 구간
            #   column을 슬라이스로 무시하는 경로 자체가 없어지고, α/β 값 자체가
            #   "이 브랜치를 얼마나 쓰는가"의 직접 해석 가능한 지표가 됨. branch별
            #   LayerNorm은 residual 모드에서 항상 켜짐(blockwise_layernorm 플래그와
            #   무관) — LN 없이 그냥 더하면 embed_dim 메커니즘(사실 #5: context는
            #   CosFace 재투영으로 norm≈1 고정, query/agg는 학습 중 무제한으로
            #   커짐 — jasmine embed_dim=256에서 query_act≈130,000까지 실측됨)이
            #   재현되어 사실상 query/agg만 더한 것과 같아짐. α,β는 1.0으로
            #   초기화(각 branch가 LN으로 이미 비슷한 스케일이라 "처음엔 동등하게
            #   더한다"는 것 외에 다른 근거 없는 값 — 학습이 알아서 줄이거나
            #   키우게 둠). "gated"(attention 방식) 등 더 복잡한 fusion은 이걸로도
            #   시나리오 A가 안 풀리면 다음 단계로 검토.
        use_context_projection: bool = False,
        fusion_alpha_override: Optional[float] = None,  # [추가] residual fusion에서
            # α를 학습 가능한 파라미터 대신 고정 스칼라로 못박음(nn.Buffer,
            # requires_grad=False). 목적: "학습이 α≈1을 선택했다"와 "α=1로
            # 고정해도 실제로 비슷한 성능이 나온다"는 다른 주장이다 — 전자는
            # optimizer의 선택일 뿐이고, 후자라야 그 값에서 실제로 무슨 일이
            # 벌어지는지(‖αc‖가 ‖q‖에 비해 크게/작게 기여하는지) 원인적으로
            # 확인 가능. {0, 0.5, 1, 2} 등으로 스윕해서 accuracy/shuffle drop이
            # 어떻게 변하는지 보는 게 목적. fusion_mode="residual"이 아니거나
            # use_context_emb=False면 무의미하므로 검증에서 걸러냄. None(기본값)
            # 이면 기존과 동일하게 학습 가능한 파라미터로 생성(하위 호환).
        fusion_beta_override: Optional[float] = None,   # [추가] 위와 대칭, agg 쪽.

        use_confidence_scaling: bool = False,      # [진단용, 추가] context_emb를
            # head에 넣기 전 top1_confidence로 스케일 — 라우팅/검색은 안 건드림
        confidence_scaling_detach: bool = False,   # [진단용, 추가] True면 스케일
            # 값 자체는 쓰되 그 경로로 gradient는 안 흐름 (Variant B).
            # use_confidence_scaling=False면 무효과.
        evidence_temperature: float = 1.0,   # [진단용, 추가] AttentionAggregator의
            # evidence_w = softmax(-‖q-k‖² / evidence_temperature). 기본값 1.0 =
            # 기존과 동일(하위 호환). jasmine/credit-g 실측: 학습 초반부터
            # entropy가 ln(k) 대비 크게 낮고(사실상 1-NN 붕괴) 학습 중 더 낮아짐 —
            # 이게 raw(정규화 안 됨) 유클리드 거리 softmax에 temperature가 없어서
            # 생기는 calibration 문제인지 검증하기 위한 자유 파라미터.
            # [결론, 실측 완료] jasmine T=0.5~10 스윕 — 초기화 시점 sharpness는
            # T로 정확히 조절되지만, 학습 후에는 T와 무관하게 전부 n_eff≈1.0으로
            # 붕괴(query_norm이 학습 중 계속 커져 distance scale 자체가 시간에
            # 따라 변하므로 고정 스칼라로는 못 따라잡음 — epoch vs query_norm
            # rho=0.986, query_norm vs distance_mean rho=0.998). 그래서
            # evidence_metric="cosine"/"cosine_scaled" 도입(아래) — temperature
            # 자체는 기각됐지만 하위 호환을 위해 파라미터는 유지.
        evidence_metric: str = "euclidean",   # [추가] "euclidean"(기본값, 기존과
            # 동일)/"cosine"/"cosine_scaled". CentroidLayer 라우팅처럼 q,k를
            # 먼저 정규화해 evidence_temperature로도 못 잡은 collapse(query_emb
            # norm 성장에 유사도 계산이 그대로 종속되는 문제)를 원천 제거하려는
            # 개입. evidence.py의 AttentionAggregator.__init__ docstring 참고.
        value_mode: str = "default",   # [추가] AttentionAggregator의 value 구성
            # 방식. "default"(기존과 100% 동일, 하위호환)/"offset_only"/"balanced".
            # evidence.py의 AttentionAggregator.__init__ docstring 참고 — 동기는
            # diagnose_value_components 실측(T(query-neighbour) 항이 label_emb
            # 항보다 평균 4.9배 크다는 게 mfeat-zernike에서 확인됨)에서 나옴.
        neighbor_interaction_mode: Optional[str] = None,  # [v2, 추가] pooling
            # (evidence_w 가중합) 전에 k개 이웃 values끼리 상호작용시킬지.
            # None(기본값, 기존과 100% 동일 — 하위호환)/"attn"(v2 후보 A,
            # self-attention among neighbours)/"capacity_baseline"/
            # "interaction_free_baseline"(대조군, evidence.py의
            # NeighborInteractionFreeBaseline 참고 — attn과 파라미터 수
            # 정확히 동일하면서 이웃 간 mixing만 구조적으로 차단).
            # fusion_mode/value_mode와 같은 성격의 구조적 선택(HPO 탐색
            # 대상 아님) — optimize.py에는 threading 안 함, reproduce.py의
            # 진단/ablation 전용 CLI 플래그로만 노출.
        interaction_n_heads: int = 2,  # [v2, 추가] neighbor_interaction_mode
            # 가 "attn"/"interaction_free_baseline"일 때만 의미 있음.
        aggregator_mode: str = "pooling",  # [v2 최종안, 추가] "pooling"
            # (기본값, 기존과 100% 동일 — 하위호환): AttentionAggregator의
            # 고정 weighted-sum. "cross_attention": AttentionAggregator를
            # 안 쓰고 HeadCrossAttention(evidence.py)이 agg_emb 자리를
            # 대체 — retrieve()/value 구성은 그대로, pooling만 head 내부
            # cross-attention으로 교체. 이 모드를 쓸 때는 B안(2-branch,
            # [updated_query‖context_emb])을 쓰려면 use_query_emb_in_head=
            # False(--no_query_emb)도 같이 줘야 함 — updated_query에 이미
            # query_emb가 residual로 들어있어 중복 방지. 기존 fusion_mode/
            # ablation_mode(agg_emb_shuffle 등) 인프라는 agg_emb 자리에
            # updated_query가 들어가는 것뿐이라 그대로 재사용됨.
        head_attn_alpha_override: Optional[float] = None,  # [v2, 추가]
            # aggregator_mode="cross_attention"일 때만 의미 있음.
            # HeadCrossAttention의 residual scale alpha를 학습 대신 이
            # 값으로 고정. 0.0으로 주면 updated_query=query_emb가 되어
            # retrieval 분기를 완전히 끈 necessity baseline이 재현됨.
        head_neighbor_source: str = "real",  # [v2, 추가] aggregator_mode=
            # "cross_attention"일 때만 의미 있음. "real"(기본값)/
            # "learned_const"(capacity-only 대조군 — evidence.py의
            # HeadCrossAttention docstring 참고, 재학습 필요).
        vectorized_fallback: bool = True,
        cat_col_idx: Optional[List[int]] = None,
        num_col_idx: Optional[List[int]] = None,
        cat_cardinalities: Optional[List[int]] = None,
        cat_combine: str = "onehot",
        cat_embed_dim: int = 16,
        num_embedding: str = "plr_lite",
        num_bin_edges: Optional[torch.Tensor] = None,
        ple_d_embedding: int = 12,   # [추가] TabularEmbedder와 동일 — 고정 기본값,
                                      # Optuna 탐색 대상 아님 (rtdl_num_embeddings 권장값).
        plr_n_frequencies: int = 16,
        plr_freq_scale: float = 0.01,
        plr_out_dim: int = 8,
        regroup_warmup_epochs: int = 0,   # [추가] CentroidLayer로 배선 — 지금까지는
                                            # 이 값이 TabERA 생성자에 아예 없어서
                                            # CentroidLayer 자체 기본값(0=즉시 활성화)이
                                            # 무조건 쓰이고 있었음. 학습 초반 STE+
                                            # dead-centroid reinit이 불안정한 시기에
                                            # regroup을 미루면 그 노이즈가 줄어드는지
                                            # 확인하기 위해 조정 가능하게 함.
        dead_reinit_patience: int = 5,     # [추가] 마찬가지로 지금까지 CentroidLayer로
                                            # 배선이 안 돼서 항상 기본값(5)만 쓰였음.
                                            # 검증 안 된 값이라 스윕 가능하게 노출.
        dead_reinit_noise_scale: float = 0.01,  # [추가] 재초기화 시 노이즈 상대 크기,
                                            # 마찬가지로 검증 안 된 값이라 노출.
        log_branch_gradients: bool = False,  # [진단용, 추가] head concat 직전
                                            # query/context/agg 활성값에
                                            # retain_grad()를 걸어 branch별
                                            # gradient를 잴 수 있게 함. 값 자체는
                                            # 안 바꾸므로(순수 .grad 보존) 학습
                                            # 결과(가중치/예측)에 영향 없음 —
                                            # 활성값을 추가로 붙들고 있어 메모리만
                                            # 약간 증가. 그래서 옵트인(기본 False).
                                            # state_dict/체크포인트와 무관한 순수
                                            # 런타임 계측 플래그라 model_kwargs/
                                            # _save_tag/--from_saved_state 하위
                                            # 호환 체계에는 안 태움.
    ) -> None:
        super().__init__()
        self.k            = k
        self.embed_dim    = embed_dim
        self.n_output     = n_output
        self.tasktype = tasktype
        if tasktype in ("binclass", "multiclass"):
            # multiclass는 n_output이 곧 n_classes. binclass는 n_output=1
            # 이지만 라벨은 2개 클래스(0/1)라 명시적으로 n_classes를 받거나
            # 기본값 2를 씀.
            self._n_classes_for_labels = n_classes if n_classes is not None else (
                n_output if tasktype == "multiclass" else 2
            )
        else:
            self._n_classes_for_labels = None
        self.n_features   = n_features
        self.log_branch_gradients = log_branch_gradients
        self._branch_grad_tensors = {}  # [v2, 안전장치] cross_attention 모드에서
            # log_branch_gradients=True를 켜도 AttributeError 안 나게 기본값을
            # 미리 만들어둠 — 실제 채우기는 forward()에서 매 배치 갱신.
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
        self.exclude_self_retrieval = exclude_self_retrieval
        # [진단용] context_emb(설명①의 신호)를 head 입력에서 아예 제외.
        # STE 라우팅/centroid 학습(diversity_loss, commitment_loss)은 그대로
        # 유지됨 — 이 둘은 centroid_emb를 직접 학습시키지 context_emb가
        # head에 들어가는지와 무관하기 때문. "그룹 신호 자체가 예측에
        # 기여하는가"만 깨끗하게 격리해서 재기 위함. T()처럼 꺼지면
        # 파라미터 자체가 안 생김(head 입력 차원이 줄어듦).
        self.use_context_emb = use_context_emb
        # [진단용, 신규] query_emb(양자화 안 된 원본 임베더 출력)를 head
        # 입력에서 제외 — use_context_emb와 대칭인 ablation. 지금까지는
        # query_emb가 항상 무조건 head에 들어갔고(옵션으로 뺄 수 있던 적
        # 없음), --no_context_emb는 "context_emb를 빼도 되는가"만 검증했지
        # "query_emb와 context_emb를 같이 넣는 조합 자체가 최선인가"는 검증
        # 공백이었음. True(기본, 기존 동작)면 지금과 동일. False면 head가
        # agg_emb(+context_emb, use_context_emb=True인 경우)만으로 예측 —
        # 즉 예측이 양자화된 신호만 통과하는 순수 VQ-VAE식 bottleneck에
        # 가까워짐. 이걸로 (a) query_emb의 raw 값이 실제로 얼마나
        # 필요한지, (b) 그만큼 Explanation①(prototype 배정)이 예측을
        # 얼마나 "진짜로" 설명하는지(vs 그냥 곁다리 신호였는지)를 같이 잰다.
        self.use_query_emb_in_head = use_query_emb_in_head
        # [진단용] centroid_emb는 diversity_loss(흩어뜨림)·codebook_loss
        # (배정된 쿼리 쪽으로 당김)·task_loss(head를 거친 예측 손실,
        # context_emb 경유)라는 서로 다른 목적의 gradient를 동시에 받음
        # (commitment_loss는 정규화된 centroid 쪽을 detach하므로
        # centroid_emb를 안 건드림 — 확인됨). 이 목적들이 충돌해
        # centroid_emb가 어느 쪽에도 최적이 아닌 타협점에 머물 가능성을
        # 검증하기 위해, True면 head로 가는 context_emb만 detach — forward
        # 값은 그대로 전달되지만 task_loss가 centroid_emb로 역전파되지 않음
        # (centroid_emb는 diversity_loss만으로 학습됨).
        self.detach_context_grad = detach_context_grad
        # [구조 조정] context_emb를 head로 보내기 전 학습 가능한 Linear를
        # 하나 거치게 함. detach_context_grad(gradient 완전 차단)와 달리
        # task_loss의 gradient가 여전히 centroid_emb까지 도달하되, 이
        # 프로젝션 행렬이 "예측에 유리하게 바꾸는 일"의 일부를 대신 떠맡아
        # centroid_emb 자체가 덜 왜곡되길 기대하는 절충안. raw centroid_emb를
        # 직접 쓰는 설명①(hard_assignment, 그룹 텍스트 라벨, confidence)
        # 계산에는 전혀 관여하지 않음 — head 직전에만 끼움.
        self.use_context_projection = use_context_projection
        self.use_confidence_scaling = use_confidence_scaling
        self.confidence_scaling_detach = confidence_scaling_detach
        if fusion_mode not in ("concat", "residual"):
            raise ValueError(f"fusion_mode은 'concat'/'residual' 중 하나여야 합니다: {fusion_mode}")
        if fusion_mode == "residual" and not use_query_emb_in_head:
            raise ValueError(
                "fusion_mode='residual'은 query_emb를 베이스(잔차의 기준점)로 쓰므로 "
                "use_query_emb_in_head=False와 같이 쓸 수 없습니다."
            )
        if fusion_alpha_override is not None and (fusion_mode != "residual" or not use_context_emb):
            raise ValueError(
                "fusion_alpha_override는 fusion_mode='residual'이고 use_context_emb=True일 "
                "때만 의미가 있습니다."
            )
        if fusion_beta_override is not None and fusion_mode != "residual":
            raise ValueError("fusion_beta_override는 fusion_mode='residual'일 때만 의미가 있습니다.")
        self.fusion_mode = fusion_mode
        self.fusion_alpha_is_fixed = fusion_alpha_override is not None
        self.fusion_beta_is_fixed  = fusion_beta_override is not None
        self.context_proj = (
            nn.Linear(embed_dim, embed_dim)
            if (use_context_projection and use_context_emb) else None
        )

        # ── 임베더 ──────────────────────────────────
        # cat_col_idx가 주어지면 categorical feature를 nn.Embedding으로
        # 처리 (① 후보 검증 결과 반영 — 안 주면 이전과 100% 동일 동작,
        # 기존 체크포인트 하위 호환 유지)
        self.embedder = TabularEmbedder(
            n_features, embed_dim, embedder_layers, dropout,
            cat_col_idx=cat_col_idx, num_col_idx=num_col_idx,
            cat_cardinalities=cat_cardinalities,
            cat_combine=cat_combine, cat_embed_dim=cat_embed_dim,
            num_embedding=num_embedding, num_bin_edges=num_bin_edges,
            ple_d_embedding=ple_d_embedding,
            plr_n_frequencies=plr_n_frequencies, plr_freq_scale=plr_freq_scale,
            plr_out_dim=plr_out_dim,
        )

        # ── CentroidLayer (Dual-Space Prototype) ────────
        self.prototype_layer = CentroidLayer(
            n_prototypes=n_prototypes,
            embed_dim=embed_dim,
            n_features=n_features,
            prototype_labels=prototype_labels,
            dropout=dropout,
            col_names=column_names,
            routing_scale=routing_scale,
            regroup_warmup_epochs=regroup_warmup_epochs,
            dead_reinit_patience=dead_reinit_patience,
            dead_reinit_noise_scale=dead_reinit_noise_scale,
            use_ema_codebook=use_ema_codebook,
            ema_decay=ema_decay,
        )

        # ── TabR 방식 이웃 집계 ──────────────────────────
        # use_offset_correction=False → value=label_emb만 사용 (T() ablation)
        # [수정] tasktype/n_classes 전달 — classification이면 label_encoder가
        # nn.Embedding(명목형 클래스에 정확한 표현), regression이면 기존과
        # 동일한 nn.Linear(연속형 값 그대로).
        # [v2, 추가] aggregator_mode: "pooling"(기존 v1, 하위호환)이면
        # AttentionAggregator를, "cross_attention"(v2)이면 HeadCrossAttention을
        # 만듦 — 둘 다 만들지 않음(파라미터/메모리 중복 방지, 그리고 어느
        # 쪽을 쓰는지 self.ot_selector/self.head_cross_attn 존재 여부로
        # 명확히 구분되게).
        valid_aggregator_modes = ("pooling", "cross_attention")
        if aggregator_mode not in valid_aggregator_modes:
            raise ValueError(
                f"aggregator_mode은 {valid_aggregator_modes} 중 하나여야 합니다: {aggregator_mode}"
            )
        self.aggregator_mode = aggregator_mode

        if aggregator_mode == "pooling":
            self.ot_selector = AttentionAggregator(
                embed_dim=embed_dim,
                k=k,
                n_features=n_features,
                n_output=n_output,
                dropout=dropout,
                use_offset_correction=use_offset_correction,
                tasktype=tasktype,
                n_classes=self._n_classes_for_labels,
                evidence_temperature=evidence_temperature,
                evidence_metric=evidence_metric,
                value_mode=value_mode,
                neighbor_interaction_mode=neighbor_interaction_mode,
                interaction_n_heads=interaction_n_heads,
            )
            self.head_cross_attn = None
        else:  # "cross_attention"
            self.ot_selector = None
            self.head_cross_attn = HeadCrossAttention(
                embed_dim=embed_dim,
                k=k,
                tasktype=tasktype,
                n_classes=self._n_classes_for_labels,
                dropout=dropout,
                use_offset_correction=use_offset_correction,
                neighbor_source=head_neighbor_source,
                alpha_override=head_attn_alpha_override,
            )

        # ── 메모리 뱅크 (검색 전용) ──────────────────
        # [실측 확인됨] cross-group fallback 경로의 기존 구현(샘플 단위
        # Python for-loop + .item() 동기화)은 fallback 비율이 높을수록
        # (vehicle 데이터셋 기준 실측 75%) retrieve() 자체를 최대 184배까지
        # 느리게 만드는 것으로 확인됨 (torch.profiler: cudaStreamSynchronize
        # 가 self CPU 시간의 59%, 실제 GPU 연산은 4%뿐). vectorized_fallback=True
        # 는 이 경로를 배치 텐서 연산(bmm+gather+masked topk)으로 대체한
        # 버전이며, 정확성은 다양한 스케일에서 기존 경로와 bit-identical함이
        # 확인됨 (bench_fallback_cost.py). 기본값은 False로 유지 —
        # 실제 4개 데이터셋에서 최종 확인 전까지는 명시적으로 켜야 함.
        self.memory = MemoryBank(memory_size, embed_dim,
                                  vectorized_fallback=vectorized_fallback)

        # ── FeatureStore (설명 전용) ──────────────────
        self._feature_store: Optional[FeatureStore] = None
        if column_names and n_features > 0:
            self._feature_store = FeatureStore(
                max_size=memory_size,
                n_features=n_features,
                col_names=column_names,
            )

        # ── 예측 헤드: [query ‖ context ‖ agg_emb] → ŷ ──
        # use_context_emb=False면 context_emb를, use_query_emb_in_head=False면
        # query_emb를 아예 제외 (head 입력 차원도 그만큼 줄어듦 — T()처럼
        # "안 쓰는 파라미터가 남아있는" 상태가 아니라 진짜로 없앤 상태로
        # 비교하기 위함). agg_emb는 항상 포함(최소 1개는 남음).
        _n_head_parts = int(use_query_emb_in_head) + int(use_context_emb) + 1
        if _n_head_parts == 1:
            print("  ⚠️  use_query_emb_in_head=False, use_context_emb=False — "
                  "head가 agg_emb만 보고 예측합니다(진단용 극단 케이스).")
        _head_in = embed_dim * _n_head_parts

        # [추가] blockwise_layernorm=False(기본)면 기존과 완전히 동일 —
        # [query‖context‖agg]를 하나로 묶어 nn.LayerNorm(_head_in) 하나로
        # 정규화. True면 각 블록을 따로 정규화한 뒤 concat — context_emb/
        # agg_emb(둘 다 routing/retrieval에 딸려 있어 배치마다 값이
        # 흔들릴 수 있음)의 통계 변화가 query_emb 쪽 정규화에 새어드는
        # 경로 자체를 없앤다. 기존 체크포인트의 head[0]이 단일 LayerNorm
        # (_head_in,) 모양이라 --from_saved_state 하위 호환을 위해 반드시
        # 옵트인으로 둠(기본 False = state_dict 모양 그대로 유지).
        #
        # [추가] fusion_mode="residual"이면 애초에 concat을 안 하므로
        # blockwise_layernorm 플래그와 무관하게 branch별 LayerNorm이 항상
        # 필요함(합산 전 스케일을 맞춰야 함 — 위 fusion_mode docstring 참고).
        self.blockwise_layernorm = blockwise_layernorm
        self.head_branch_l2norm = head_branch_l2norm
        # [v2, 추가] aggregator_mode="cross_attention"이면 위 fusion_mode/
        # use_query_emb_in_head 기반 head 구성 전체를 안 씀 — updated_query
        # 안에 query_emb가 이미 residual로 흡수돼 있어(retrieval branch가
        # updated_query로 흡수된 것이지 agg_emb를 대체하는 게 아님) "query
        # branch를 안 쓴다"는 의미의 use_query_emb_in_head=False와는
        # 개념이 다름 — 그 플래그를 재사용하면 나중에 코드 읽는 사람이
        # 헷갈림. 대신 전용 2-branch head를 명시적으로 따로 만듦:
        # head_input = [updated_query ‖ context_emb] 고정(다른 조합 없음).
        if self.aggregator_mode == "pooling":
            self._per_branch_ln = blockwise_layernorm or (fusion_mode == "residual")
            if self._per_branch_ln:
                self.head_query_ln   = nn.LayerNorm(embed_dim) if use_query_emb_in_head else None
                self.head_context_ln = nn.LayerNorm(embed_dim) if use_context_emb else None
                self.head_agg_ln     = nn.LayerNorm(embed_dim)  # agg_emb는 항상 포함됨
            else:
                self.head_query_ln = self.head_context_ln = self.head_agg_ln = None

            if fusion_mode == "residual":
                # z = LN(q) + α·LN(c) + β·LN(a) → embed_dim 크기 하나만 MLP에 통과.
                # α는 use_context_emb=False면 애초에 안 만듦(T()/context_proj와
                # 같은 패턴 — "안 쓰는 파라미터가 남아있는" 상태가 아니라 진짜로
                # 없앤 상태로 비교하기 위함).
                # [추가] override가 주어지면 nn.Parameter 대신 register_buffer로
                # 고정값을 등록 — optimizer.parameters()에 안 잡히므로 학습 중
                # 절대 안 바뀜(디버깅으로 실수로 update되는 일 방지). state_dict에는
                # 여전히 저장/복원됨(buffer도 state_dict에 포함되므로 --from_saved_state
                # 호환 유지).
                if use_context_emb:
                    if self.fusion_alpha_is_fixed:
                        self.register_buffer("fusion_alpha", torch.tensor(float(fusion_alpha_override)))
                    else:
                        self.fusion_alpha = nn.Parameter(torch.tensor(1.0))
                else:
                    self.fusion_alpha = None
                if self.fusion_beta_is_fixed:
                    self.register_buffer("fusion_beta", torch.tensor(float(fusion_beta_override)))
                else:
                    self.fusion_beta = nn.Parameter(torch.tensor(1.0))
                self.head = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(embed_dim, n_output),
                )
            elif blockwise_layernorm or head_branch_l2norm:
                # [수정] head_branch_l2norm=True면 global LayerNorm(_head_in)을
                # 건너뜀 — 그 LN이 concat된 벡터 전체를 다시 joint하게
                # 재정규화해서, 바로 앞에서 만든 branch별 unit-L2-norm을
                # 지워버리는 문제가 스모크 테스트로 확인됨(hook으로 첫 Linear
                # 입력을 직접 봤더니 branch norm이 1이 아니었음). probe가
                # 검증한 건 "L2-normalize 후 아무 추가 정규화 없이 바로
                # 분류기"였으므로, 실제 모델에서도 그 조건을 정확히 재현.
                self.head = nn.Sequential(
                    nn.Linear(_head_in, embed_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(embed_dim, n_output),
                )
            else:
                self.head = nn.Sequential(
                    nn.LayerNorm(_head_in),
                    nn.Linear(_head_in, embed_dim), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(embed_dim, n_output),
                )

            # [진단용] log_branch_gradients가 참조하는 head 첫 Linear layer와,
            # 그 in_features 축에서 각 브랜치가 차지하는 column 구간. combined
            # 순서(_parts 조립 순서, forward() 참고)는 항상
            # [query(있으면) → context(있으면) → agg(항상)] 이므로 여기서도
            # 그 순서를 그대로 따른다 — forward()의 조립 순서가 바뀌면 이
            # 매핑도 같이 바뀌어야 함.
            # [주의] fusion_mode="residual"이면 concat 자체가 없어 "column 구간"
            # 개념이 없음 — 그 대신 self.fusion_alpha/self.fusion_beta 값 자체가
            # 이미 "브랜치별 기여"의 직접적이고 더 해석 가능한 지표이므로
            # _head_block_slices는 빈 채로 둔다(log_branch_gradients의 weight-norm
            # 진단은 concat 계열 모드 전용).
            self._head_first_linear = self.head[0] if (blockwise_layernorm or head_branch_l2norm or fusion_mode == "residual") else self.head[1]
            self._head_block_slices: Dict[str, Tuple[int, int]] = {}
            if fusion_mode == "concat":
                _off = 0
                if use_query_emb_in_head:
                    self._head_block_slices["query"] = (_off, _off + embed_dim)
                    _off += embed_dim
                if use_context_emb:
                    self._head_block_slices["context"] = (_off, _off + embed_dim)
                    _off += embed_dim
                self._head_block_slices["agg"] = (_off, _off + embed_dim)

            self.head_v2 = None
        else:  # "cross_attention"
            if not use_query_emb_in_head:
                print("  ⚠️  aggregator_mode='cross_attention'에서는 use_query_emb_in_head/"
                      "--no_query_emb가 아무 효과가 없습니다(무시됨) — head 입력은 항상 "
                      "[updated_query‖context_emb] 2-branch로 고정입니다. updated_query에 "
                      "query_emb가 이미 residual로 흡수돼 있어 별도 query branch 개념이 "
                      "없습니다.")
            self.head_v2 = nn.Sequential(
                nn.LayerNorm(2 * embed_dim),
                nn.Linear(2 * embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, n_output),
            )
            self.head = None
            self._head_first_linear = self.head_v2[1]
            self._head_block_slices = {
                "updated_query": (0, embed_dim),
                "context":       (embed_dim, 2 * embed_dim),
            }
            self.head_query_ln = self.head_context_ln = self.head_agg_ln = None
            self._per_branch_ln = False
            # [안전장치] fusion_mode="residual"과 aggregator_mode="cross_attention"을
            # (의미는 없지만) 같이 줘도 out dict 구성부에서 AttributeError가
            # 안 나게 기본값을 잡아둠 — cross_attention에서는 애초에 fusion_mode
            # 자체를 안 씀(head_v2가 이미 [updated_query‖context]로 고정).
            self.fusion_alpha = None
            self.fusion_beta = None

    # ── Forward ────────────────────────────────────────

    def forward(
        self,
        X: torch.Tensor,                          # (B, F)
        labels: Optional[torch.Tensor] = None,    # (B,) 학습 시 메모리 업데이트용
        sample_ids: Optional[torch.Tensor] = None, # (B,) 학습 시 X_train 행 번호 — MemoryBank/
                                                    # FeatureStore 슬롯 대응을 확정적으로 증명하기 위함
        return_explanations: bool = False,
        ablation_mode: str = "none",              # ablation 모드 (학습 시 "none" 유지)
    ) -> Dict[str, torch.Tensor]:
        # 1. 임베딩
        query_emb = self.embedder(X)               # (B, D)

        # 2. 프로토타입 라우팅 (CentroidLayer)
        # prototypes.py가 6개 반환 (hierarchical 호환 + top1_confidence) →
        # 필요한 4개만 사용. top1_confidence: routing_probs(STE, forward=
        # one-hot이라 confidence로 못 씀)와 달리 샘플마다 실제로 다른 값 —
        # ①의 group_confidence 표시와 confidence scaling(옵션) 양쪽에 씀.
        context_emb, hard_assignment, routing_probs, _, _, top1_confidence = \
            self.prototype_layer(query_emb)

        # 3. KNN 검색 + Attention 집계
        if self.memory.filled.item() >= self.k:
            nk, neighbour_labels, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                # [진단용] global_retrieve=True면 그룹 무시하고 전체 검색.
                # context_emb(위 2번)는 hard_assignment와 무관하게 이미
                # 정상 계산됨 — 설명①은 그대로, 검색만 전역으로 바뀜.
                hard_assignment=(None if self.global_retrieve else hard_assignment),
                exclude_ids=(sample_ids if self.exclude_self_retrieval else None),
            )

            # [진단용, 추가] self-retrieval 검증 — MemoryBank는 epoch를 넘어
            # 유지되는 circular buffer라, retrieve()가 이번 배치 자기 자신을
            # (이전 epoch에 저장해둔 자기 자신을) 이웃으로 돌려줄 가능성이
            # 있음. MemoryBank가 label도 그대로 저장/반환하므로(update()의
            # `labels` 인자, retrieve()의 `neighbour_labels` 반환값 — TabR
            # 방식 value 구성에 직접 쓰임) self-retrieval이 실제로 일어나면
            # 그 슬롯의 neighbour_label은 이 샘플의 진짜 정답 그 자체임.
            # retrieve()가 update()보다 먼저 호출되므로(바로 아래 4번 항목,
            # memory.update는 이 forward 마지막에 호출됨) **같은 iteration
            # 안에서는 self-retrieval이 원천적으로 불가능** — 여기서 재는 건
            # "이전 epoch에 저장된 자기 자신"과의 매칭만 해당.
            _self_retrieval_top1_rate = None
            _self_retrieval_topk_rate = None
            if sample_ids is not None:
                _stored_ids = self.memory.sample_ids[topk_idx]        # (B, k)
                _is_self    = _stored_ids == sample_ids.unsqueeze(-1).to(_stored_ids.device)  # (B, k)
                _self_retrieval_top1_rate = float(_is_self[:, 0].float().mean().item())
                _self_retrieval_topk_rate = float(_is_self.any(dim=-1).float().mean().item())


            # ── Ablation: random_neighbor / neighbor_noise ───────
            # [수정 이력] 기존 random_neighbor는 nk/nv는 순수 노이즈로,
            # neighbour_labels는 배치 내 셔플로 서로 "다른 방식"으로
            # 조작했음 — 그러면 성능 하락이 "이웃이 틀려서"인지 "이웃
            # 정보 자체가 없어서"인지 구분이 안 됨 (두 가설이 뒤섞임).
            # 이제 두 가설을 분리된 모드로 나눔 (nv는 제거되어 더는
            # 등장하지 않음 — nv_utility_probe로 잔차 설명력이 noise
            # 대조군과 구분 안 됨을 실측한 뒤 삭제):
            #   random_neighbor : nk/labels를 "같은" permutation으로
            #                     통째로 셔플 → 배치 내 다른 쿼리의
            #                     진짜(실존하는) 이웃 세트로 통째로
            #                     바꿔치기. retrieval이 "맞는 이웃"을
            #                     찾았는지만 순수하게 검증 (이웃 정보
            #                     자체는 여전히 real).
            #   neighbor_noise  : nk/labels 전부 실제 데이터와 무관한
            #                     것으로 교체. "이웃 정보가 조금이라도
            #                     존재하는가" 자체를 검증.
            if ablation_mode == "random_neighbor":
                B_abl = nk.shape[0]
                rand_perm = torch.randperm(B_abl, device=X.device)
                nk               = nk[rand_perm]
                neighbour_labels = neighbour_labels[rand_perm]

            elif ablation_mode == "neighbor_noise":
                nk = F.normalize(torch.randn_like(nk), dim=-1)
                # neighbour_labels: uniform random class가 아니라, 이번
                # 배치에서 실제로 검색된 라벨들의 empirical pool에서
                # 재추출 — 실제 클래스 비율(marginal)은 유지한 채
                # 쿼리-라벨 연결만 완전히 끊음.
                B_abl, K_abl = neighbour_labels.shape[0], neighbour_labels.shape[1]
                label_pool = neighbour_labels.reshape(-1)
                rand_pos = torch.randint(
                    0, label_pool.numel(), (B_abl, K_abl), device=X.device
                )
                neighbour_labels = label_pool[rand_pos]

            if self.aggregator_mode == "cross_attention":
                # [v2] AttentionAggregator 대신 head 내부 cross-attention.
                # updated_query가 agg_emb 자리를 대체 — 아래 기존 fusion/
                # ablation 코드는 전혀 안 바뀜(agg_emb라는 이름의 텐서가
                # 뭘로 계산됐는지만 다름). shuffle_neighbors는 evidence.py의
                # HeadCrossAttention 전용 necessity ablation — 기존
                # ablation_mode="agg_emb_shuffle"(아래에서 agg_emb 자체를
                # 배치 내에서 섞음)과는 다른 걸 검증함(전자는 "이 query의
                # 진짜 이웃에 의존하는가", 후자는 "다른 브랜치와 이 값의
                # 대응이 중요한가").
                agg_emb, evidence_w, evidence_diag = self.head_cross_attn(
                    query_emb, nk, neighbour_labels,
                    shuffle_neighbors=(ablation_mode == "shuffle_neighbors"),
                )
            else:
                agg_emb, evidence_w, evidence_diag = self.ot_selector(query_emb, nk, neighbour_labels)

        else:
            # Memory 미충족 fallback
            agg_emb    = torch.zeros_like(query_emb)
            evidence_w = torch.full((X.shape[0], self.k), 1.0 / self.k, device=X.device)
            topk_idx   = torch.zeros(X.shape[0], self.k, dtype=torch.long, device=X.device)
            # [추가] fallback 경로는 진짜 검색이 아니라 균등분포를 그냥
            # 채운 것이라 query/key norm·distance 자체가 없음 — None으로
            # 표시해 supervised.py가 이 배치를 통계에서 자연스럽게 제외
            # 하게 함(evidence_w의 uniform 값 자체는 의미 있는 신호가
            # 아니므로 여기서 온 entropy=ln(k)를 "붕괴 안 됨"으로 잘못
            # 해석하지 않도록).
            evidence_diag = None
            _self_retrieval_top1_rate = None
            _self_retrieval_topk_rate = None

        # 4. 예측
        # use_context_emb=False면 context_emb를, use_query_emb_in_head=False면
        # query_emb를 head 입력에서 제외 (STE 라우팅/centroid 학습 자체는
        # 그대로 — aux_loss가 별도로 학습시킴, head 입력 여부와 무관)
        if self.aggregator_mode == "pooling":
            _parts = []
            if self.use_query_emb_in_head:
                _query_for_head = query_emb
                # [추가] eval-time perturbation — 이미 학습이 끝난 모델에 그대로
                # 적용해서 "head가 query_emb를 실제로 얼마나 쓰는가"를 재학습
                # 없이 재는 용도. --no_query_emb(처음부터 빼고 재학습)는 학습
                # 자체가 가능한지를 보여줬을 뿐(붕괴함), 정상 학습된 모델이
                # 이 슬롯에 얼마나 의존하는지는 안 보여줌 — 이 둘은 다른 질문.
                #   query_emb_zero    : 이 슬롯을 0으로 — head가 학습 때 한 번도
                #                       못 본 입력이라 "정보 제거"와 "분포 이탈"
                #                       효과가 섞임(zero-ablation의 알려진 한계).
                #   query_emb_shuffle : 배치 내에서 셔플 — 각 슬롯 자체의 값
                #                       분포(LayerNorm이 보는 통계)는 그대로
                #                       유지한 채 "이 샘플의 진짜 query_emb"라는
                #                       연결만 끊음 — 표준 permutation
                #                       importance와 같은 방식이라 분포 이탈
                #                       효과가 zero보다 작음. 기본으로는 이쪽을
                #                       더 신뢰할 것.
                if ablation_mode == "query_emb_zero":
                    _query_for_head = torch.zeros_like(query_emb)
                elif ablation_mode == "query_emb_shuffle":
                    _perm = torch.randperm(query_emb.shape[0], device=query_emb.device)
                    _query_for_head = query_emb[_perm]
                if self._per_branch_ln:
                    _query_for_head = self.head_query_ln(_query_for_head)
                if self.head_branch_l2norm:
                    _query_for_head = F.normalize(_query_for_head, dim=-1)
                _parts.append(_query_for_head)
            if self.use_context_emb:
                _ctx_for_head = context_emb
                if self.context_proj is not None:
                    _ctx_for_head = self.context_proj(_ctx_for_head)
                # [진단용, 추가] confidence scaling — "라우팅은 그대로, head가
                # 받는 신호의 크기만 assignment confidence로 조절"하는 실험용
                # 개입. context_emb는 M=1(기본값)에서 항상 unit-norm centroid
                # 하나 그대로라 norm이 샘플마다 다를 수 없었음(실측 확인:
                # context_act_norm이 학습 내내 ~1.40, 변동 0.3% 수준) — 이
                # 스케일링은 그 "norm에 정보가 없는" 상태를 의도적으로 깨고,
                # 라우팅이 애매했던 샘플(top1_confidence 낮음)의 context_emb를
                # 작게, 확신 있던 샘플은 크게 만들어 head에 전달한다.
                # confidence_scaling_detach=False(Variant A)면 gradient가
                # top1_confidence를 거쳐 centroid_emb/embedder까지 흐름 —
                # "애매한 샘플일수록 gradient도 같이 작아진다"는 부작용이
                # 있을 수 있음(리뷰 지적). True(Variant B)면 스케일 값 자체는
                # forward에 반영하되 gradient는 그 경로로 안 흐름(순수 크기
                # 조절만). 둘 다 실험해서 비교할 것 — 어느 쪽이 나은지 아직
                # 검증 안 됨.
                if self.use_confidence_scaling:
                    _conf = top1_confidence.detach() if self.confidence_scaling_detach else top1_confidence
                    _ctx_for_head = _ctx_for_head * _conf.unsqueeze(-1)
                if self.detach_context_grad:
                    _ctx_for_head = _ctx_for_head.detach()
                # [추가] query_emb_zero/shuffle과 대칭 — context_emb 쪽도 같은
                # 방식으로 eval-time에 얼마나 의존하는지 잴 수 있게 함.
                if ablation_mode == "context_emb_zero":
                    _ctx_for_head = torch.zeros_like(_ctx_for_head)
                elif ablation_mode == "context_emb_shuffle":
                    _perm = torch.randperm(_ctx_for_head.shape[0], device=_ctx_for_head.device)
                    _ctx_for_head = _ctx_for_head[_perm]
                if self._per_branch_ln:
                    _ctx_for_head = self.head_context_ln(_ctx_for_head)
                if self.head_branch_l2norm:
                    _ctx_for_head = F.normalize(_ctx_for_head, dim=-1)
                _parts.append(_ctx_for_head)
            # [추가] query_emb_zero/shuffle, context_emb_zero/shuffle과 대칭 —
            # agg_emb(검색·attention 집계 결과)도 같은 방식으로 잰다. query_emb_
            # shuffle이 성능을 랜덤 수준까지 무너뜨린 게 "agg_emb는 원래 기여가
            # 없다"인지, "head가 query_emb와 agg_emb를 짝(pair)으로 해석하도록
            # 학습돼서 짝이 어긋나면 더 헷갈린다"인지 구분이 안 됐음 — 이 두
            # 모드로 그 모호함을 직접 검증. agg_emb만 셔플하고 query_emb는
            # 그대로 두면(반대로 query_emb_shuffle은 agg_emb를 그대로 두고
            # query_emb만 섞었음), "agg_emb 단독의 기여도"와 "짝 어긋남의
            # 대가"를 분리해서 볼 수 있음.
            _agg_for_head = agg_emb
            if ablation_mode == "agg_emb_zero":
                _agg_for_head = torch.zeros_like(agg_emb)
            elif ablation_mode == "agg_emb_shuffle":
                _perm = torch.randperm(agg_emb.shape[0], device=agg_emb.device)
                _agg_for_head = agg_emb[_perm]
            if self._per_branch_ln:
                _agg_for_head = self.head_agg_ln(_agg_for_head)
            if self.head_branch_l2norm:
                _agg_for_head = F.normalize(_agg_for_head, dim=-1)
            _parts.append(_agg_for_head)
    
            # [진단용] log_branch_gradients=True면 head가 실제로 보는 시점(concat
            # 직전)의 브랜치별 활성값에 retain_grad()를 건다. supervised.py가
            # loss.backward() 직후 self._branch_grad_tensors[name].grad를 읽어
            # "head가 각 브랜치에 얼마나 gradient를 돌려주는가"를 잰다.
            # retain_grad()는 값을 바꾸지 않으므로(순수 .grad 보존) 이 진단을
            # 켜도 학습 결과에는 영향이 없다 — 활성값을 추가로 붙들고 있어
            # 메모리만 약간 증가(그래서 기본 False, train 중에만 동작).
            #
            # [주의 — gradient ≠ importance] 이게 재는 건 "학습 신호의 흐름"
            # 이지 "예측이 실제로 그 브랜치를 쓰는가"가 아니다. 이미 잘 학습된
            # 브랜치는 gradient가 작아도 여전히 예측에 크게 기여할 수 있다 —
            # 반드시 --ablation *_shuffle/zero 결과, 그리고 head 첫 Linear의
            # block별 weight norm(supervised.py가 epoch마다 같이 기록)과 함께
            # 해석할 것. gradient가 작다는 것만으로 "head가 이 브랜치를 안
            # 쓴다"고 결론 내리지 말 것.
            if self.training and self.log_branch_gradients:
                self._branch_grad_tensors = {}
                if self.use_query_emb_in_head and _query_for_head.requires_grad:
                    _query_for_head.retain_grad()
                    self._branch_grad_tensors["query"] = _query_for_head
                if self.use_context_emb and _ctx_for_head.requires_grad:
                    _ctx_for_head.retain_grad()
                    self._branch_grad_tensors["context"] = _ctx_for_head
                # [주의] memory.filled < k인 극초반 배치는 agg_emb=torch.zeros_like(...)
                # fallback이라 계산 그래프에 안 걸려 requires_grad=False임 — 이때는
                # retain_grad()가 에러를 내므로 건너뛴다. 이 자체가 "메모리가 아직
                # 안 채워진 동안은 agg 브랜치에 애초에 gradient가 흐를 수 없다"는
                # 진단적으로 유의미한 사실이라, 억지로 걸지 않고 그 배치만 누락시킨다
                # (supervised.py의 _branch_grad_sum이 dict.get() 기반이라 브랜치별로
                # 등장 배치 수가 달라도 문제없이 누적됨).
                if _agg_for_head.requires_grad:
                    _agg_for_head.retain_grad()
                    self._branch_grad_tensors["agg"] = _agg_for_head
    
            # [진단용, 추가] LN 적용 후, combine(그리고 있다면 α/β 곱셈) 이전의
            # branch별 평균 L2 norm(배치 평균). "α≈1"이라는 숫자 하나만으로는
            # residual 합산에서 그 branch가 실제로 얼마나 기여하는지 알 수 없음
            # (예: ‖LN(q)‖=100, ‖LN(c)‖=5면 α=1이어도 q+αc≈q) — 이 norm과
            # out["fusion_alpha"]/out["fusion_beta"](스칼라)를 곱하면 ‖αc‖/‖βa‖를
            # 사후에 그대로 계산할 수 있음(스칼라 곱은 norm과 교환되므로 여기서
            # 미리 곱해두지 않고 원재료만 남김 — α/β가 음수일 수도 있어 부호
            # 정보를 norm 계산 전에 지우지 않기 위함).
            _head_query_norm_mean = (
                float(_query_for_head.detach().norm(dim=-1).mean().item())
                if self.use_query_emb_in_head else None
            )
            _head_context_norm_mean = (
                float(_ctx_for_head.detach().norm(dim=-1).mean().item())
                if self.use_context_emb else None
            )
            _head_agg_norm_mean = float(_agg_for_head.detach().norm(dim=-1).mean().item())
    
            # [진단용, 추가] branch 간 cosine similarity (배치 평균). norm만으로는
            # "합쳐졌을 때 실제로 서로 강화하는지 상쇄하는지"를 알 수 없음 —
            # ‖q‖=10, ‖αc‖=10이어도 cos(q,c)=1이면 ‖q+αc‖≈20, cos(q,c)=-1이면
            # ‖q+αc‖≈0. fusion_mode와 무관하게 항상 계산(concat도 "합치는 방식과
            # 무관하게 세 표현이 서로 얼마나 다른 정보를 담고 있는가"를 보여주는
            # 값이라 비교 기준으로 유용).
            _head_cos_qc_mean = (
                float(F.cosine_similarity(_query_for_head.detach(), _ctx_for_head.detach(), dim=-1).mean().item())
                if (self.use_query_emb_in_head and self.use_context_emb) else None
            )
            _head_cos_qa_mean = (
                float(F.cosine_similarity(_query_for_head.detach(), _agg_for_head.detach(), dim=-1).mean().item())
                if self.use_query_emb_in_head else None
            )
            _head_cos_ca_mean = (
                float(F.cosine_similarity(_ctx_for_head.detach(), _agg_for_head.detach(), dim=-1).mean().item())
                if self.use_context_emb else None
            )
    
            if self.fusion_mode == "residual":
                # z = LN(q) + α·LN(c) + β·LN(a) — LN은 위에서 이미 branch별로
                # 적용됨(self._per_branch_ln이 residual 모드에서 항상 True).
                combined = _query_for_head + self.fusion_beta * _agg_for_head
                if self.use_context_emb:
                    combined = combined + self.fusion_alpha * _ctx_for_head
            else:
                combined = torch.cat(_parts, dim=-1)
            # [진단용, 추가] ‖q+αc+βa‖ 배치 평균 — 벡터 합의 norm은 개별 norm의
            # 단순 합이 아니므로(위 cosine 참고) 실제로 합을 한 번 계산해야만
            # 나옴. concat 모드에서는 이 양 자체가 존재하지 않으므로(combined가
            # 다른 표현 공간) None.
            _head_combined_norm_mean = (
                float(combined.detach().norm(dim=-1).mean().item())
                if self.fusion_mode == "residual" else None
            )
            logits = self.head(combined)
        else:
            # [v2] retrieval branch가 updated_query(=agg_emb 변수명 그대로
            # 재사용 중이지만 개념은 "agg_emb 대체"가 아니라 "retrieval
            # branch가 updated_query 안으로 흡수된 것") 안으로 흡수됨.
            # head 입력은 항상 [updated_query‖context_emb] 2-branch —
            # use_query_emb_in_head/fusion_mode 등 pooling 전용 스위치는
            # 여기서 아예 참조하지 않음(생성자에서 head_v2를 이미 이 구조로
            # 고정해서 만들어뒀음 — self.head_v2 참고).
            _query_for_head = query_emb          # 진단용 원본(아래 norm 등)
            _ctx_for_head   = context_emb
            _agg_for_head   = agg_emb             # = updated_query
            combined = torch.cat([agg_emb, context_emb], dim=-1)
            logits = self.head_v2(combined)

            # [진단용] pooling 모드와 같은 log_branch_gradients 지원 —
            # 브랜치가 2개(updated_query/context)뿐이라는 것만 다름.
            if self.training and self.log_branch_gradients:
                self._branch_grad_tensors = {}
                if _query_for_head.requires_grad:
                    _query_for_head.retain_grad()
                    self._branch_grad_tensors["updated_query"] = _query_for_head
                if _ctx_for_head.requires_grad:
                    _ctx_for_head.retain_grad()
                    self._branch_grad_tensors["context"] = _ctx_for_head

            with torch.no_grad():
                _head_query_norm_mean   = float(query_emb.detach().norm(dim=-1).mean().item())
                _head_context_norm_mean = float(context_emb.detach().norm(dim=-1).mean().item())
                _head_agg_norm_mean     = float(agg_emb.detach().norm(dim=-1).mean().item())
                _head_combined_norm_mean = None  # concat이라 residual 모드의 그 의미가 없음
                _head_cos_qc_mean = float(
                    F.cosine_similarity(query_emb.detach(), context_emb.detach(), dim=-1).mean().item()
                )
                # [주의] v2는 query/context/agg 3-branch가 아니라 updated_query/
                # context 2-branch라 "query-agg", "context-agg" cosine이라는
                # 개념 자체가 없음(agg가 독립 branch가 아니라 query에 흡수된
                # 것) — pooling 모드 전용 진단이라 여기서는 None.
                _head_cos_qa_mean = None
                _head_cos_ca_mean = None

        # 5. 메모리 업데이트 (학습 시)
        if self.training and labels is not None:
            self.memory.update(query_emb.detach(), labels.float(), sample_ids)
            if self._feature_store is not None:
                self._feature_store.update(X, sample_ids)

        # 5.5 [추가] EMA codebook 업데이트 (학습 시, use_ema_codebook일 때만)
        # — 매 배치마다 non-gradient로 centroid_emb를 직접 갱신. memory.update()
        # 바로 다음(hard_assignment가 이미 이 배치에 대해 계산돼 있는 시점)에
        # 두는 이유는 딱히 순서 의존성이 있어서가 아니라 같은 "배치 단위
        # 부기(bookkeeping)" 묶음이라 여기 두는 게 자연스러워서.
        if self.training and self.prototype_layer.use_ema_codebook:
            self.prototype_layer.ema_update(query_emb.detach(), hard_assignment)

        # 6. 보조 손실
        aux_loss = torch.tensor(0.0, device=X.device)
        if self.training:
            if self.prototype_layer.use_ema_codebook:
                # [중요] EMA를 쓰면 diversity_loss/codebook_loss 둘 다 뺀다.
                # codebook_loss는 EMA가 대체(당연히 중복 적용 안 함).
                # diversity_loss는 centroid_emb.requires_grad=False라 여기서
                # 계산해도 gradient가 갈 곳이 없어 계산 자체가 낭비이고,
                # 혹시라도 남겨두면 "centroid_emb에 gradient가 없다"는 걸
                # 모르는 채 loss 값만 보고 "diversity가 반영되고 있다"고
                # 착각하기 쉬움 — 아예 호출을 안 해서 그 착각의 여지를
                # 없앤다. commitment_loss는 embedder 쪽 gradient라 그대로 둠.
                aux_loss = (
                    self.loss_weights["commitment"] * self.prototype_layer.commitment_loss(query_emb, hard_assignment)
                )
            else:
                aux_loss = (
                    self.loss_weights["diversity"]  * self.prototype_layer.diversity_loss()
                    + self.loss_weights["commitment"] * self.prototype_layer.commitment_loss(query_emb, hard_assignment)
                    # [추가] codebook_loss — commitment_loss와 방향만 반대인 짝.
                    # .get()으로 하위 호환: 옛 체크포인트의 loss_weights에는
                    # "codebook" 키가 없어 --from_saved_state 로드 시 이 항은
                    # 0으로 처리됨(codebook_loss 자체가 아예 없던 상태와 동일).
                    + self.loss_weights.get("codebook", 0.0) * self.prototype_layer.codebook_loss(query_emb, hard_assignment)
                )

        out = {
            "logits":      logits,
            "aux_loss":    aux_loss,
            "routing":     routing_probs,
            "hard_group":  hard_assignment,
            "evidence_w":  evidence_w,
            "evidence_diag": evidence_diag,
            "topk_idx":    topk_idx,
            "agg_emb":     agg_emb,
            # [추가, 진단용] linear probe(query_emb/context_emb/agg_emb 각각에 별도
            # 선형 분류기를 붙여 "정보가 없어서 head가 무시하는가(A) vs 정보는
            # 있는데 concat+공유 MLP가 못/안 쓰는가(B)"를 구분)를 위해 head에
            # 들어가기 전(정규화/스케일링/ablation 적용 전) raw 값을 그대로 노출.
            # head가 실제로 보는 값(_query_for_head 등, confidence_scaling·
            # detach_context_grad·ablation_mode가 적용된 뒤)과는 다를 수 있음 —
            # probe는 "이 표현 자체에 정보가 있는가"를 보는 거라 원본을 쓰는 게 맞음.
            "query_emb":   query_emb,
            "context_emb": context_emb,
            "ablation_mode": ablation_mode,
            # [추가, 진단용] fusion_mode="residual"일 때 학습된 α(context 가중치)/
            # β(agg 가중치) 현재값 — concat 모드의 head 첫 Linear weight-norm
            # 진단을 대신하는, residual 모드 전용의 직접 해석 가능한 지표.
            # concat 모드에서는 항상 None.
            "fusion_alpha": (
                float(self.fusion_alpha.detach().item())
                if (self.fusion_mode == "residual" and self.fusion_alpha is not None) else None
            ),
            "fusion_beta": (
                float(self.fusion_beta.detach().item())
                if self.fusion_mode == "residual" else None
            ),
            # [추가] ||LN(q)||, ||LN(c)||, ||LN(a)|| 배치 평균 — fusion_mode와
            # 무관하게 항상 계산됨(concat 모드에서도 self._per_branch_ln이면
            # 의미 있음). residual 모드가 아니면 branch별 LN을 안 켰을 수
            # 있어(blockwise_layernorm=False) 이 경우 LN 전 raw activation norm임 —
            # self._per_branch_ln 값을 같이 확인해서 해석할 것.
            "head_query_norm_mean":   _head_query_norm_mean,
            "head_context_norm_mean": _head_context_norm_mean,
            "head_agg_norm_mean":     _head_agg_norm_mean,
            "head_combined_norm_mean": _head_combined_norm_mean,
            "head_cos_qc_mean": _head_cos_qc_mean,
            "head_cos_qa_mean": _head_cos_qa_mean,
            "head_cos_ca_mean": _head_cos_ca_mean,
            # [추가, 진단용] self-retrieval 비율. sample_ids가 None이면(추론
            # 시 등) None — supervised.py가 자연스럽게 통계에서 제외함.
            "self_retrieval_top1_rate": _self_retrieval_top1_rate,
            "self_retrieval_topk_rate": _self_retrieval_topk_rate,
        }

        if return_explanations:
            # [수정] 이전에는 고정 temperature=0.1을 별도로 써서, 실제
            # 라우팅(CentroidLayer.forward()의 soft — routing_scale 사용,
            # HPO가 데이터셋마다 다르게 찾은 값, jasmine 기준 5.44)과 다른
            # 분포를 보여주고 있었음 — ①의 routing_confidence가 "실제 모델이
            # 이 배정에 얼마나 확신하는가"가 아니라 "0.1이라는 임의의
            # 온도로 다시 계산한 별개의 숫자"였다는 뜻(faithfulness 문제).
            # 실제 라우팅과 정확히 같은 공식(routing_scale 사용)으로 바꿔
            # 설명과 실제 라우팅 분포가 항상 일치하게 함. .detach()는
            # 그대로 유지(설명은 어차피 no_grad라 gradient 불필요).
            with torch.no_grad():
                q_norm     = F.normalize(query_emb.detach(), dim=-1)       # (B, D)
                c_norm     = F.normalize(
                    self.prototype_layer.centroid_emb.detach(), dim=-1)    # (P, D)
                cos_sim    = q_norm @ c_norm.T                              # (B, P), scale 적용 전
                soft_probs = F.softmax(
                    cos_sim * self.prototype_layer.routing_scale,
                    dim=-1)                                                # (B, P)

            proto_exp = self.prototype_layer.explain_routing(hard_assignment, soft_probs, cos_sim=cos_sim)
            ev_exp    = (
                self.ot_selector.explain_evidence(evidence_w)
                if self.aggregator_mode == "pooling"
                else self.head_cross_attn.explain_evidence(evidence_w)
            )
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

    @torch.no_grad()
    def refresh_memory_keys(self, batch_size: int = 1024) -> Optional[Dict[str, float]]:
        """학습 종료 후(best_state/feature_store 복원 직후) 1회 호출.

        feature_store에 저장된 raw feature를 지금(frozen) 가중치로 다시
        embedder에 통과시켜 memory.keys/_keys_norm을 덮어쓴다. 그 결과
        memory.keys[i]는 "학습 도중 특정 시점의 dropout mask + 그 시점의
        가중치로 계산된 1회성 스냅샷"이 아니라 "현재 가중치에서 raw
        feature의 순수 결정론적 함수"가 된다.

        [설계 근거] 원본 TabR(libs/tabr.py)의 candidate 처리 방식과 같은
        원칙 — TabR도 candidate embedding을 영구 저장하지 않고, epoch마다
        `self.model.cached_candidate_k = self._encode(...)`로 통째로 다시
        인코딩하며, predict()에서는 self.model.eval() 이후 candidate를
        처음부터 다시 계산한다. 즉 "추론/설명 단계에는 학습 노이즈가
        새어나가지 않는다"는 원본 설계를 TabERA의 저장 구조(ring buffer +
        group cache)에 맞게 적용한 것.

        [학습 중엔 호출 안 함 — 의도적] 학습 스텝마다 이걸 부르지 않는
        이유: (1) 매 스텝 embedder를 한 번 더 통과시키는 건 학습 전체
        기간(E epoch) 동안 E×N_train번의 추가 forward가 필요해 비쌈,
        이 방식은 학습 전체에 걸쳐 N_train번(최선의 경우 그보다 적게,
        best가 갱신될 때만) 한 번만 계산하면 됨. (2) TabR도 학습 중엔
        현재 배치의 noisy key를 candidate에 섞어 쓰므로(`candidate_k =
        torch.cat([k, cached_candidate_k])`), 학습 중 retrieval 대상에
        약간의 노이즈가 섞이는 것 자체는 TabR 계열에서 드문 일이 아님.

        Returns
        ───────
        None이면 feature_store가 없어 건너뜀. 있으면 진단용 dict
        {"n_refreshed": int} — 몇 개 슬롯을 갱신했는지.
        """
        if self._feature_store is None:
            return None

        n_mem  = int(self.memory.filled.item())
        n_feat = self._feature_store._filled
        # [방어적 확인] 두 저장소는 forward()에서 항상 같은 호출, 같은
        # 배치로 붙어서 update되므로 filled가 같아야 하는 구조적 불변조건.
        # 어긋나면 memory와 feature_store가 서로 다른 시점(예: best_state는
        # 복원됐는데 feature_store는 최신 상태로 남은 경우)을 가리키고
        # 있다는 뜻이라, 조용히 잘못된 슬롯을 짝짓는 대신 바로 알아채도록
        # assert로 막는다 — 리뷰에서 지적된 "memory/feature_store 복원
        # 순서" 문제를 코드가 스스로 검증하게 하는 안전장치.
        assert n_mem == n_feat, (
            f"refresh_memory_keys(): memory.filled({n_mem}) != "
            f"feature_store._filled({n_feat}) — memory와 feature_store가 "
            f"서로 다른 시점으로 복원된 상태일 수 있습니다. best_state/"
            f"feature_store 복원이 refresh_memory_keys() 호출보다 먼저 "
            f"끝났는지 확인하세요."
        )
        if n_mem == 0:
            return {"n_refreshed": 0}

        was_training = self.training
        self.eval()
        device = self.memory.keys.device
        for start in range(0, n_mem, batch_size):
            end = min(start + batch_size, n_mem)
            raw   = self._feature_store._store[start:end].to(device)
            clean = self.embedder(raw)
            self.memory.keys[start:end]      = clean
            self.memory._keys_norm[start:end] = F.normalize(clean, dim=-1)
        if was_training:
            self.train()
        return {"n_refreshed": n_mem}

    # [제거됨] forward_soft_for_ig / get_fixed_neighbors_for_ig /
    # forward_fixed_neighbors_for_ig — 전부 IG의 completeness axiom을
    # STE 불연속 아래서도 성립시키려고 만든 IG 전용 우회 장치였음(연속
    # forward, causal 검증용 fixed-neighbor forward). ③을 SHAP으로
    # 통일하면서 제거됨: SHAP은 gradient/연속 경로가 필요 없는 black-box
    # perturbation 방법이라 이 우회 자체가 무의미해짐. 참고로 이 3개
    # 메서드는 reproduce.py에서도 forward_soft_for_ig 외에는 실제로
    # 호출되지 않던 코드였다(get_fixed_neighbors_for_ig/
    # forward_fixed_neighbors_for_ig는 사전 인과 검증 실험에 1회성으로
    # 쓰이고 메인 흐름에는 연결되지 않았음).

    def anneal(self, factor: float = 0.95) -> None:
        self.prototype_layer.anneal(factor)

    def summary(self, n_train: Optional[int] = None) -> str:
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
                 f"  query_emb      : {'IN head input' if self.use_query_emb_in_head else 'EXCLUDED (진단용, ablation)'}",
                 f"  context_emb    : {'IN head input' if self.use_context_emb else 'EXCLUDED (진단용, ablation)'}",
                 f"  context grad   : {'STOP (진단용, ablation)' if self.detach_context_grad else 'flows to centroid_emb'}",
                 f"  context proj   : {'Linear projection (구조 조정)' if self.context_proj is not None else 'none (raw concat)'}",
                 f"  codebook update: {'EMA (decay=' + str(self.prototype_layer.ema_decay) + '), diversity_loss OFF' if self.prototype_layer.use_ema_codebook else 'gradient (codebook_loss + diversity_loss)'}",
                 f"  head LayerNorm : {'블록별(query/context/agg 따로)' if self._per_branch_ln else '결합(하나로 묶어서, 기존 방식)'}",
                 f"  head fusion    : {'concat([q‖c‖a] → MLP, 기존 방식)' if self.fusion_mode == 'concat' else f'residual(z=LN(q)+α·LN(c)+β·LN(a) → MLP, α/β 학습됨)'}",
                 f"  branch L2norm  : {'ON (v1.1, concat 전 branch별 unit-L2-norm)' if getattr(self, 'head_branch_l2norm', False) else 'OFF (기존과 동일)'}",
                 f"  neighbor mixing: {(self.ot_selector.neighbor_interaction_mode or 'OFF (v1 그대로, pooling 전 이웃 상호작용 없음)') if self.aggregator_mode == 'pooling' else 'N/A (aggregator_mode=cross_attention — pooling 자체가 없음)'}",
                 f"  aggregator     : {'pooling(AttentionAggregator, 고정 weighted-sum)' if self.aggregator_mode == 'pooling' else f'cross_attention(HeadCrossAttention, n_heads=1, alpha={self.head_cross_attn.alpha.detach().item():.3f}, neighbor_source={self.head_cross_attn.neighbor_source})'}" ]

        # [Limitation 진단] k가 평균 그룹 크기(N_train/P)보다 크면
        # cross-group fallback이 상시 발동할 위험이 있음 — 4개 데이터셋
        # 진단(dataset_profile)에서 실측: vehicle(k=64, 평균 그룹=28.2)은
        # fallback 75.3%, ada/qsar/wine(k ≤ 평균 그룹 크기)은 7~14%.
        # "group-constrained KNN"이라는 설명②의 클레임이 데이터셋 크기와
        # k 선택에 따라 실제로 지켜지는 정도가 다름을 알리기 위한 경고.
        if n_train is not None and self.prototype_layer.P > 0:
            avg_group_size = n_train / self.prototype_layer.P
            ratio = self.k / avg_group_size
            lines.append(f"  Avg group size : {avg_group_size:.1f}  (N_train={n_train:,} / P={self.prototype_layer.P})")
            if ratio > 1.0:
                lines.append(
                    f"  ⚠️  k({self.k}) > 평균 그룹 크기({avg_group_size:.1f}) "
                    f"— cross-group fallback이 상시 발동할 가능성 높음"
                )
                lines.append(
                    f"     (설명②의 'group-constrained' 클레임이 이 설정에서는 "
                    f"약화될 수 있음 — 참고: reproduce.py --ablation dataset_profile)"
                )

        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        return "\n".join(lines)