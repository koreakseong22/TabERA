"""
libs/prototypes.py
==================
CentroidLayer — Dual-Space Prototype Representation
    (이중 공간 프로토타입 기반 계층적 설명가능 검색 레이어)

가설 핵심 구현
──────────────
(1) 임베딩 공간 centroid
    - centroid_emb  (P, D) : 임베딩 공간 — STE routing + FAISS 마스킹용

(2) Straight-Through Estimator (STE) Hard Routing + FAISS 범위 제한
    - O(N) → O(P + k·log k) 복잡도 개선
    - 배정된 centroid 그룹 내 샘플 인덱스만 FAISS 검색 후보로 제한
    - train: Straight-Through Estimator (STE) / eval: argmax hard
    - STE: forward=argmax(hard), backward=softmax gradient 통과
    - 근거: VQ-VAE(van den Oord, 2017) 표준 설계 + commitment loss

(3) 그룹 텍스트 라벨 — ①의 그룹 설명용
    - ①의 주 콘텐츠: 이 그룹이 어떤 target(클래스)에 해당하는지
      (label_groups_by_target) — ②/③ 어디에도 없는 ①만의 고유 정보
    - 보조 정보: 그룹을 가장 잘 특징짓는 feature들의 실제 그룹 평균값
      (label_all_groups) — 정성적 밴드("매우 높음" 등)가 아니라 원값
    - 매 epoch ema_update() 직후 supervised.py에서 계산·캐싱
    - [설계 변경 이력] 이전에는 centroid_x(medoid)를 buffer로 저장해
      ①에서 대표 데이터 1개를 그대로 보여줬으나, (a) 표본이 적은
      그룹에서 outlier 1개가 그대로 대표값이 될 위험, (b) "실제 존재
      하는 샘플을 보여준다"는 medoid의 장점은 이미 ②(MemoryBank 이웃
      k개 + evidence_w)가 더 강한 근거로 제공하고 있어 역할이 중복됨
      — 두 이유로 ①에서는 제거하고 텍스트 요약으로 대체함. 처음엔
      별도 파일(libs/group_labels.py)이었으나, CentroidLayer에만
      쓰이는 전용 헬퍼라 파일을 분리해 둘 이유가 없어 이 파일로 합침.

[제거됨] ig_baseline — 이전엔 Integrated Gradients(③) 전용 medoid를 여기서
    관리했음. ③을 SHAP으로 통일하면서 제거됨 — SHAP은 IG처럼 baseline→input
    연속 경로가 필요 없고(gradient 기반이 아니라 black-box perturbation
    기반), background로 medoid 1개가 아니라 여러 대표 샘플의 "분포"가
    필요해 이 buffer의 용도 자체가 없어짐(SHAP background는 reproduce.py
    에서 X_train 랜덤 샘플링으로 별도 처리).
    [하위 호환 주의] ig_baseline / ig_baseline_initialized buffer가 state_dict
    에서 빠짐 — 이전에 저장된 체크포인트를 --from_saved_state로 로드하면
    strict=True 기준으로는 실패함 (그 checkpoint를 쓸 일이 있다면 로딩 시
    strict=False 필요).

이론적 근거
───────────
- Dual-Space Prototype Representation (본 가설)
- Straight-Through Estimator (Bengio et al. 2013)
- VQ-VAE hard assignment trick (van den Oord et al. 2017)

하위 호환성
───────────
PrototypeLayer = CentroidLayer  (alias 유지, 기존 tabr.py 수정 불필요)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 그룹 텍스트 라벨링 헬퍼 (구 libs/group_labels.py)
# ─────────────────────────────────────────────────────────────
# CentroidLayer.sample_groups/target_labels/group_labels 캐싱에만
# 쓰이는 전용 함수들이라, CentroidLayer와 같은 파일에 둔다.
# supervised.py에서 `from libs.prototypes import label_all_groups,
# label_groups_by_target`로 가져다 쓴다.
#
# 두 부분으로 구성된다:
# 1) label_groups_by_target()  — ①의 주 콘텐츠. "이 그룹이 어떤
#    target(클래스)에 해당하는가". ②/③ 어디에도 없는, ①만의 고유 정보.
# 2) label_all_groups()        — 그 그룹을 가장 잘 특징짓는("두드러진")
#    feature들의 실제 그룹 평균값(원값, 정성적 밴드 아님).
#
# 랭킹 기준: 그룹 간 대비(cross-group distinctiveness)
# ─────────────────────────────────────────────
# 목표는 "각 centroid의 서로 구별되는 특징"을 보여주는 것이다. "이 그룹이
# 전체 데이터셋 대비 얼마나 극단적인가"만 보면, 여러 그룹에 걸쳐 비슷하게
# 튀는 feature가 계속 1등을 차지해서 서로 다른 centroid들이 같은 feature로
# 도배되는 문제가 있었다. 대신 "이 그룹의 값이 다른 그룹들의 값 분포에서
# 얼마나 벗어나는가"(다른 그룹들 대비 robust z-score)로 랭킹한다.
#
# numeric은 그룹 median(표본이 작을 때(흔히 1~10개) outlier에 안 휘둘림),
# categorical은 최빈 카테고리 + 그 비율을 그대로 값으로 보여준다 — "매우
# 높음/보통" 같은 구간 라벨은 안 쓴다(원값이 이미 직접 해석 가능하고,
# 구간 경계값에 대한 근거를 따로 만들 필요가 없어짐).

@dataclass
class FeatureLabel:
    feature_idx:  int
    feature_name: str
    kind:         str    # "numeric" | "categorical"
    label:        str    # 그룹의 실제 값 (예: "10.4" 또는 "Category 2 (65%)")
    detail: dict          # 사람이 검증/디버그할 때 참고할 원값


def _group_stats_numeric(
    X_train: np.ndarray,                    # (N, F)
    valid_groups: Sequence[int],
    sample_groups: Sequence[Sequence[int]],
    feature_idx: int,
) -> Dict[int, float]:
    """그룹별 median (raw 값). {group_idx: group_median}"""
    col = X_train[:, feature_idx]
    return {p: float(np.median(col[sample_groups[p]])) for p in valid_groups}


def _group_stats_categorical(
    X_train: np.ndarray,
    valid_groups: Sequence[int],
    sample_groups: Sequence[Sequence[int]],
    feature_idx: int,
    eps: float = 1e-6,
) -> Dict[int, dict]:
    """그룹별 (최빈 카테고리, 그 비율, lift). {group_idx: {...}}"""
    col = np.rint(X_train[:, feature_idx]).astype(int)
    out = {}
    for p in valid_groups:
        group_vals = col[sample_groups[p]]
        if len(group_vals) == 0:
            continue
        values, counts = np.unique(group_vals, return_counts=True)
        top_cat = int(values[np.argmax(counts)])
        group_prop   = float((group_vals == top_cat).mean())
        overall_prop = float((col == top_cat).mean())
        out[p] = {
            "top_category": top_cat,
            "group_prop":   group_prop,
            "lift":         group_prop / (overall_prop + eps),
        }
    return out


def _cross_group_distinctiveness(this_value: float, other_values: Sequence[float]) -> Optional[float]:
    """
    other_values(다른 그룹들의 같은 feature 값) 분포에서 this_value가
    얼마나 벗어나는지 robust z-score로 계산.
    median/MAD(median absolute deviation) 사용 — 그룹 몇 개가 극단치여도
    std보다 덜 흔들림. 비교할 다른 그룹이 2개 미만이면(=P가 아주 작은
    경우) 계산 불가하므로 None 반환 → 호출부에서 fallback.
    """
    if len(other_values) < 2:
        return None
    others = np.asarray(other_values, dtype=float)
    med = np.median(others)
    mad = np.median(np.abs(others - med)) * 1.4826 + 1e-6  # 정규분포 가정 시 std와 동일 스케일
    return float(abs(this_value - med) / mad)


def inverse_transform_numeric(qt, num_cols: Sequence[int], feature_idx: int, value: float) -> Optional[float]:
    """
    numeric feature는 libs/data.py의 prep_data()에서 QuantileTransformer로
    [0,1] uniform 값으로 바뀐 채 저장된다 — "0.328"이 실제로 몇 단위인지
    (예: credit_amount가 몇 마르크인지) 알 방법이 없었던 원인. qt(fit된
    QuantileTransformer)가 주어지면 실제 단위로 역변환한다.

    QuantileTransformer는 각 컬럼을 독립적으로 처리하므로(fit 시 컬럼별로
    별도 분위수 매핑을 학습함), 다른 컬럼에 아무 값이나 채워 넣어도
    feature_idx 위치의 역변환 결과에는 영향이 없다 — 그래서 그룹 값 하나만
    바꿔 넣은 더미 행으로 역변환해도 안전하다(검증됨).

    qt가 None이거나 feature_idx가 num_cols에 없으면 None 반환 →
    호출부에서 [0,1] 값 그대로 표시하는 걸로 fallback.
    """
    if qt is None:
        return None
    try:
        col_pos = list(num_cols).index(feature_idx)
    except ValueError:
        return None
    dummy = np.full((1, len(num_cols)), 0.5)
    dummy[0, col_pos] = value
    try:
        return float(qt.inverse_transform(dummy)[0, col_pos])
    except Exception:
        return None


def label_all_groups(
    X_train: np.ndarray,
    sample_groups: Sequence[Sequence[int]],
    cat_cols: Sequence[int],
    num_cols: Sequence[int],
    col_names: Sequence[str],
    top_k: int = 5,
    min_group_size: int = 2,
    cat_category_names: Optional[Dict[str, Sequence[str]]] = None,
    quantile_transformer=None,
) -> Dict[int, List[FeatureLabel]]:
    """
    ema_update() 직후 호출해서 캐싱해두는 용도.
    반환값: {group_index: [FeatureLabel, ...]}  (top_k개, 그룹 간
    대비(distinctiveness) 내림차순 — "이 그룹만 유별난" feature가 위로)

    cat_category_names: {col_name: [원본 카테고리 문자열, ...]}가 주어지면
    (libs/data.py의 load_data()가 반환하는 것), categorical 라벨을
    "Category 0" 대신 실제 이름("male single" 등)으로 표시한다. 없으면
    "Category N"으로 fallback (하위 호환 — 이 인자 없이 부르던 기존 코드도
    그대로 동작).

    quantile_transformer: libs/data.py의 prep_data()가 반환하는 fit된
    QuantileTransformer가 주어지면, numeric 라벨을 [0,1] uniform 값
    대신 실제 단위(예: credit_amount=3271)로 역변환해 보여준다. 없으면
    [0,1] 값 그대로 표시 (하위 호환).
    """
    valid_groups = [p for p, g in enumerate(sample_groups)
                     if g is not None and len(g) >= min_group_size]
    if not valid_groups:
        return {p: [] for p in range(len(sample_groups))}

    num_stats: Dict[int, Dict[int, float]] = {
        fi: _group_stats_numeric(X_train, valid_groups, sample_groups, fi)
        for fi in num_cols
    }
    cat_stats: Dict[int, Dict[int, dict]] = {
        fi: _group_stats_categorical(X_train, valid_groups, sample_groups, fi)
        for fi in cat_cols
    }

    result: Dict[int, List[FeatureLabel]] = {p: [] for p in range(len(sample_groups))}

    for p in valid_groups:
        candidates: List[FeatureLabel] = []

        for fi in num_cols:
            stats = num_stats[fi]
            if p not in stats:
                continue
            this_val = stats[p]
            others   = [v for q, v in stats.items() if q != p]
            dist = _cross_group_distinctiveness(this_val, others)
            if dist is None:
                dist = abs(this_val - float(np.median(list(stats.values()))))  # fallback

            real_val = inverse_transform_numeric(quantile_transformer, num_cols, fi, this_val)
            display_val = real_val if real_val is not None else this_val

            candidates.append(FeatureLabel(
                feature_idx=fi,
                feature_name=col_names[fi] if fi < len(col_names) else f"f{fi}",
                kind="numeric",
                label=f"{display_val:.3g}",
                detail={"group_value_uniform": this_val, "group_value_real": real_val,
                        "distinctiveness": dist},
            ))

        for fi in cat_cols:
            stats = cat_stats[fi]
            if p not in stats:
                continue
            top_cat, group_prop, lift = (stats[p]["top_category"], stats[p]["group_prop"], stats[p]["lift"])
            this_log = float(np.log2(lift + 1e-6))
            others_log = [float(np.log2(s["lift"] + 1e-6)) for q, s in stats.items() if q != p]
            dist = _cross_group_distinctiveness(this_log, others_log)
            if dist is None:
                dist = abs(this_log)  # fallback: lift=1(log=0)에서 얼마나 떨어졌는지

            fname = col_names[fi] if fi < len(col_names) else f"f{fi}"
            names_for_col = cat_category_names.get(fname) if cat_category_names else None
            cat_display = (str(names_for_col[top_cat])
                            if names_for_col is not None and top_cat < len(names_for_col)
                            else f"Category {top_cat}")

            candidates.append(FeatureLabel(
                feature_idx=fi,
                feature_name=fname,
                kind="categorical",
                label=f"{cat_display} ({group_prop:.0%})",
                detail={"top_category": top_cat, "group_prop": group_prop, "lift": lift, "distinctiveness": dist},
            ))

        candidates.sort(key=lambda fl: fl.detail["distinctiveness"], reverse=True)
        result[p] = candidates[:top_k]

    return result


def format_group_labels(labels: List[FeatureLabel]) -> str:
    if not labels:
        return "(그룹 크기가 작아 특징을 요약할 수 없습니다)"
    lines = [f"  - {fl.feature_name}: {fl.label}" for fl in labels]
    return "\n".join(lines)


def label_groups_by_target(
    labels: np.ndarray,                      # (N,) MemoryBank 라벨(class index as float, 또는 regression target)
    sample_groups: Sequence[Sequence[int]],
    tasktype: str,                            # "multiclass" | "binclass" | "regression"
    class_names: Optional[Sequence[str]] = None,
    min_group_size: int = 2,
    second_class_threshold: float = 0.2,      # 2등 클래스가 이 비율 이상이면 같이 표시
) -> Dict[int, Optional[dict]]:
    """
    각 그룹이 실제로 어떤 target에 해당하는지 요약 — ①의 주 콘텐츠.
    - classification(multiclass/binclass): 최다 클래스 + 비율. 2등
      클래스가 second_class_threshold 이상이면 같이 반환(그룹이 두
      클래스에 걸쳐 있다는 걸 숨기지 않기 위함).
    - regression: 그룹 target 평균이 전체 분포에서 몇 percentile인지.

    반환값: {group_idx: {...} or None}  (그룹 크기 미달 시 None)
    """
    labels = np.asarray(labels)
    result: Dict[int, Optional[dict]] = {}

    for p, grp in enumerate(sample_groups):
        if grp is None or len(grp) < min_group_size:
            result[p] = None
            continue
        y_grp = labels[grp]

        if tasktype in ("multiclass", "binclass"):
            y_int = np.rint(y_grp).astype(int)
            vals, counts = np.unique(y_int, return_counts=True)
            order = np.argsort(-counts)
            top_cls   = int(vals[order[0]])
            top_count = int(counts[order[0]])
            top_prop  = float(top_count / len(y_int))
            top_name  = (class_names[top_cls]
                         if class_names is not None and top_cls < len(class_names)
                         else f"Class {top_cls}")

            second = None
            if len(order) > 1:
                second_cls   = int(vals[order[1]])
                second_count = int(counts[order[1]])
                second_prop  = float(second_count / len(y_int))
                if second_prop >= second_class_threshold:
                    second_name = (class_names[second_cls]
                                   if class_names is not None and second_cls < len(class_names)
                                   else f"Class {second_cls}")
                    second = {"class": second_cls, "name": second_name,
                              "prop": second_prop, "count": second_count}

            result[p] = {
                "kind": "classification",
                "top_class": top_cls, "top_class_name": top_name,
                "top_prop": top_prop, "top_count": top_count,
                "second": second, "n": len(y_int),
            }
        else:  # regression
            grp_mean   = float(np.mean(y_grp))
            percentile = float((labels <= grp_mean).mean()) * 100.0
            result[p] = {
                "kind": "regression",
                "group_mean": grp_mean, "percentile": percentile, "n": len(y_grp),
            }

    return result


def format_target_label(info: Optional[dict]) -> str:
    """label_groups_by_target()의 그룹 하나 결과를 사람이 읽을 텍스트로."""
    if info is None:
        return "(그룹 크기가 작아 요약할 수 없습니다)"

    if info["kind"] == "classification":
        s = f"주로 \"{info['top_class_name']}\" {info['top_count']}/{info['n']} ({info['top_prop']:.0%})"
        if info["second"] is not None:
            s += (f" — \"{info['second']['name']}\"도 "
                  f"{info['second']['count']}/{info['n']} ({info['second']['prop']:.0%}) 포함")
        return s
    else:
        s = (f"target 평균 {info['group_mean']:.3g} "
             f"(전체 분포 기준 백분위 {info['percentile']:.0f}, n={info['n']})")
        return s


class CentroidLayer(nn.Module):
    """
    이중 공간 프로토타입 표현 레이어.

    Parameters
    ──────────
    n_prototypes      : centroid 수 P
    embed_dim         : 임베딩 차원 D
    n_features        : 원본 feature 수 F (이중 공간 저장용)
    prototype_labels  : centroid 의미론적 이름 (없으면 "Centroid_i" 자동 생성)
    ema_momentum      : EMA 업데이트 모멘텀 (0.9~0.99 권장)
    ema_warmup_epochs : 이 에폭 이후부터 EMA 활성화 (기본 0 = 즉시)
    dropout           : 컨텍스트 벡터 드롭아웃
    col_names         : 원본 feature 컬럼명 (설명 출력용)
    """

    def __init__(
        self,
        n_prototypes: int,
        embed_dim: int,
        n_features: int = 0,
        prototype_labels: Optional[List[str]] = None,
        ema_momentum: float = 0.95,
        ema_warmup_epochs: int = 0,   # 즉시 활성화 (warmup 없음)
        dropout: float = 0.0,
        col_names: Optional[List[str]] = None,
        routing_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.P                 = n_prototypes
        self.D                 = embed_dim
        self.F                 = n_features
        self.ema_momentum      = ema_momentum
        self.ema_warmup_epochs = ema_warmup_epochs
        self.col_names         = col_names or [f"f{i}" for i in range(n_features)]
        # [추가] routing softmax의 scale factor. ArcFace/CosFace/AdaCos/
        # von Mises-Fisher Loss 등 코사인 유사도 기반 softmax를 쓰는
        # 문헌에서 공통적으로 지적하는 문제 — cos유사도가 [-1,1]이라는
        # 좁은 범위에 갇혀 있어 스케일링 없이 그대로 softmax에 넣으면
        # 분포가 평평(flat)해지고, 그 결과 STE backward의 gradient
        # 신호가 약해짐. 학습 파라미터가 아니라 고정 배수(기본 1.0 =
        # 기존과 완전 동일, state_dict도 안 바뀜 — 새 학습 파라미터가
        # 아니라 plain 속성이라 하위 호환 유지) — HPO로 데이터셋마다
        # 탐색 가능하게 둠 (AdaCos 논문: 최적 scale이 클래스/그룹 수에
        # 따라 달라짐 — 고정 상수 하나로는 부족).
        self.routing_scale     = routing_scale

        # ── 온도 (register_buffer: 저장되지만 gradient 없음) ──
        self.register_buffer("current_epoch", torch.tensor(0, dtype=torch.long))

        # ── centroid 임베딩 (학습 가능 파라미터: routing + FAISS 마스킹) ──
        self.centroid_emb = nn.Parameter(torch.empty(n_prototypes, embed_dim))
        nn.init.orthogonal_(self.centroid_emb)

        # [제거됨] IG(③) 전용 baseline buffer — ③이 SHAP으로 통일되면서
        # 더 이상 필요 없어짐 (모듈 docstring 참고). n_features 파라미터
        # 자체는 다른 용도(생성자 시그니처 하위 호환)로 유지.

        # ── centroid별 샘플 인덱스 그룹 (FAISS 범위 제한용) ──
        # list of lists: sample_groups[p] = [idx, idx, ...]
        self.sample_groups: Optional[List[List[int]]] = None

        # ── centroid별 텍스트 라벨 캐시 (libs/group_labels.py 결과) ──
        # {p: [FeatureLabel, ...]} — ema_update() 직후 supervised.py에서
        # 채워짐. ①의 그룹 특징 설명은 이 캐시가 담당한다.
        self.group_labels: Optional[Dict[int, list]] = None

        # ── centroid별 target(클래스) 분포 캐시 (①의 주 콘텐츠) ──
        # {p: {"kind": ..., "top_class_name": ..., ...} or None} —
        # ema_update() 직후 supervised.py에서 label_groups_by_target()로 채움.
        self.target_labels: Optional[Dict[int, Optional[dict]]] = None

        # ── centroid별 평균 레이블 (Rank-Consistency Loss용) ──
        self.register_buffer(
            'centroid_labels',
            torch.full((n_prototypes,), float('nan'))
        )

        # ── 레이블 ────────────────────────────────────────────
        self.labels = prototype_labels or [f"Centroid_{i}" for i in range(n_prototypes)]

        self.dropout = nn.Dropout(dropout)

    # ─────────────────────────────────────────────────────────
    # 초기화: 훈련 데이터로 centroid 설정
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def initialize_from_data(
        self,
        X_emb: torch.Tensor,             # (N, D) 훈련 임베딩
        X_raw: Optional[torch.Tensor] = None,   # (N, F) 원본 feature — [미사용, 호출부 하위 호환용 시그니처만 유지]
        y_labels: Optional[torch.Tensor] = None, # (N,) 레이블 (소수 클래스 보장용)
    ) -> None:
        """
        KMeans++ 스타일 초기화.

        Arthur & Vassilvitskii (SODA 2007) "k-means++: The Advantages of
        Careful Seeding"의 거리 기반 확률적 시딩을 구현합니다.

        알고리즘
        ────────
        1. 첫 centroid: 균등 무작위 선택
        2. 이후 각 centroid: 기존 centroid와의 최소 거리²에 비례한
           확률로 다음 centroid 선택
           → 멀리 있는 점이 선택될 확률이 높아져 dead centroid 방지

        y_labels가 주어지면 소수 클래스 보장
        ──────────────────────────────────
        P개 선택 후에도 특정 클래스가 대표 centroid를 갖지 못하면
        해당 클래스의 대표 샘플(클래스 평균에 가장 가까운 샘플)로
        마지막 centroid를 대체합니다.
        """
        N   = X_emb.shape[0]
        dev = X_emb.device
        X_n = F.normalize(X_emb.float(), dim=-1)  # (N, D) 정규화

        # ── Step 1: KMeans++ 시딩 ────────────────────────────────
        selected_idx = []

        # 첫 centroid: 균등 무작위
        first = torch.randint(N, (1,), device=dev).item()
        selected_idx.append(first)

        for _ in range(self.P - 1):
            ctrs  = X_n[torch.tensor(selected_idx, device=dev)]
            sims  = X_n @ ctrs.T
            max_sim, _ = sims.max(dim=1)
            dists_sq = (1.0 - max_sim).clamp(min=0.0) ** 2
            dists_sq[torch.tensor(selected_idx, device=dev)] = 0.0
            if dists_sq.sum() < 1e-10:
                nxt = torch.randint(N, (1,), device=dev).item()
            else:
                nxt = torch.multinomial(dists_sq, 1).item()
            selected_idx.append(nxt)

        # ── Step 2: 소수 클래스 보장 (y_labels 있을 때) ────────────
        if y_labels is not None:
            y_cpu = y_labels.cpu()
            unique_cls = y_cpu.unique().tolist()
            sel_labels = y_cpu[torch.tensor(selected_idx)].tolist()

            for cls in unique_cls:
                if cls not in sel_labels:
                    # 해당 클래스 샘플들의 평균 임베딩에 가장 가까운 샘플
                    cls_mask = (y_cpu == cls).nonzero(as_tuple=True)[0]
                    cls_emb  = X_n[cls_mask.to(dev)].mean(0, keepdim=True)
                    dists    = 1.0 - (X_n[cls_mask.to(dev)] @ cls_emb.T).squeeze()
                    rep_idx  = cls_mask[dists.argmin().item()].item()
                    # 가장 큰 클러스터에 해당하는 마지막 selected를 교체
                    sel_t    = torch.tensor(selected_idx)
                    sel_lbl  = y_cpu[sel_t]
                    majority = sel_lbl.long().bincount().argmax().item()
                    maj_pos  = (sel_lbl == majority).nonzero(as_tuple=True)[0]
                    if len(maj_pos) > 1:
                        selected_idx[maj_pos[-1].item()] = rep_idx

        # ── Step 3: centroid 등록 ────────────────────────────────
        idx_t = torch.tensor(selected_idx, device=dev)
        self.centroid_emb.data = X_n[idx_t]

        # [제거됨] 이전엔 여기서 ig_baseline(IG medoid)도 같이 초기화했음.
        # X_raw 파라미터는 supervised.py 호출부 하위 호환을 위해 시그니처는
        # 유지하되, 더 이상 이 함수 내부에서 쓰이지 않음(③=SHAP 통일).
        self.sample_groups = [[] for _ in range(self.P)]

        # 초기화 품질 로그: centroid 간 평균 코사인 거리
        sim_mat  = self.centroid_emb.data @ self.centroid_emb.data.T
        mask     = ~torch.eye(self.P, dtype=torch.bool, device=dev)
        avg_dist = (1.0 - sim_mat[mask]).mean().item()
        print(f"  [CentroidLayer] KMeans++ {self.P} centroids "
              f"from {N} samples. avg_inter_dist={avg_dist:.3f}")

    # ─────────────────────────────────────────────────────────
    # (3) EMA 업데이트 (에폭 종료 후 호출)
    # ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def ema_update(
        self,
        X_emb: torch.Tensor,        # (N, D) 전체 훈련 임베딩 — assignment 계산용
        X_raw: Optional[torch.Tensor] = None,   # (N, F) 원본 feature — [미사용, 호출부 하위 호환용 시그니처만 유지]
        assignments: Optional[torch.Tensor] = None,  # (N,) hard assignment
    ) -> Dict[str, float]:
        """
        에폭 종료 후 호출: sample_groups 갱신.

        [제거됨] 이전엔 여기서 ig_baseline(IG(③) 전용 medoid)도 매 에폭
        갱신했음 — ③이 SHAP으로 통일되면서 제거됨(모듈 docstring 참고).
        ①의 그룹 텍스트 라벨(group_labels)은 이 함수가 sample_groups를
        갱신한 직후, supervised.py의 호출부에서 libs/group_labels.py로
        별도 계산·캐싱한다 (이 변경과 무관하게 그대로 유지).

        유지되는 기능
        ─────────────
        - sample_groups 갱신: KNN 검색 범위 제한 (필수)
        - dead centroid 감지: collapse 조기 종료 (필수)

        Returns
        ───────
        stats: {"active_ratio": float, "min_cluster_size": int, "max_cluster_size": int}
        """
        epoch = self.current_epoch.item()
        if epoch < self.ema_warmup_epochs:
            return {"active_ratio": 0.0, "min_cluster_size": 0, "max_cluster_size": 0}

        if assignments is None:
            # 현재 centroid 기준으로 재배정
            q = F.normalize(X_emb.float(), dim=-1)
            c = F.normalize(self.centroid_emb, dim=-1)
            assignments = (q @ c.T).argmax(dim=-1)

        # [제거됨] 이전엔 여기서 벡터화된 medoid 계산(all_sims/medoid_indices/
        # assigned_mask)으로 ig_baseline을 갱신했음. ig_baseline이 제거되면서
        # 이 계산 블록도 함께 제거(다른 곳에서 재사용되지 않음 — sample_groups
        # 갱신은 아래에서 assignments_cpu 기반의 별도 로직으로 처리됨).
        P = self.P
        assignments_cpu = assignments.cpu()
        new_groups: List[List[int]] = [[] for _ in range(P)]
        sizes = [0] * P
        for p in range(P):
            mask_cpu = (assignments_cpu == p).nonzero(as_tuple=True)[0]
            new_groups[p] = mask_cpu.tolist()
            sizes[p] = len(new_groups[p])

        self.sample_groups = new_groups
        self.current_epoch += 1

        # 통계
        n_assigned = sum(1 for s in sizes if s > 0)

        return {
            "active_ratio":     n_assigned / self.P,
            "active_centroids": int(n_assigned),
            "pruned_this_epoch": 0,
            "min_cluster_size": int(min(s for s in sizes if s > 0)) if any(s > 0 for s in sizes) else 0,
            "max_cluster_size": int(max(s for s in sizes if s > 0)) if any(s > 0 for s in sizes) else 0,
        }

    # ─────────────────────────────────────────────────────────
    # 온도 어닐링 (에폭마다 호출)
    # ─────────────────────────────────────────────────────────

    def anneal(self, factor: Optional[float] = None) -> None:
        """
        (하위 호환) 온도 어닐링 인터페이스.
        factor가 None이면 tau_anneal_rate 기반 지수 감소.
        """
        pass  # STE 전환으로 annealing 불필요

    # ─────────────────────────────────────────────────────────
    # FAISS 마스킹용 인덱스 반환
    # ─────────────────────────────────────────────────────────

    def get_candidate_indices(
        self,
        hard_assignment: torch.Tensor,  # (B,)
        max_candidates: int = 5000,
    ) -> Optional[List[List[int]]]:
        """
        배정된 centroid 그룹의 샘플 인덱스를 반환합니다.
        FAISS 검색 범위를 해당 그룹으로 제한하는 데 사용합니다.
        O(N) → O(P + k·log k) 복잡도 개선의 핵심.

        Returns None if sample_groups not yet initialized.
        """
        if self.sample_groups is None:
            return None

        B = hard_assignment.shape[0]
        result = []
        for b in range(B):
            p = hard_assignment[b].item()
            grp = self.sample_groups[p]
            if len(grp) == 0:
                result.append(None)  # 빈 그룹: 전체 검색으로 fallback
            else:
                result.append(grp[:max_candidates])
        return result

    # ─────────────────────────────────────────────────────────
    # Forward (Hierarchical-extended)
    # ─────────────────────────────────────────────────────────

    def forward(
        self,
        query_emb: torch.Tensor,                          # (B, D)
        top_m: int = 1,                                    # 신규: 사용할 centroid 수
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """
        Hierarchical-extended forward.

        Parameters
        ──────────
        query_emb : (B, D) 임베더 출력
        top_m     : 사용할 top-M centroid 수 (default 1 = 기존 hard routing)
                    M=1이면 기존 STE 동작과 완전 동일 (backward-compat)
                    M=2~3이면 hierarchical soft routing

        Returns
        ───────
        context_emb     : (B, D)   — top-M centroid 가중 혼합 (gradient 흐름)
        hard_assignment : (B,)     — Top-1 centroid (FAISS 마스킹 + 설명용)
        routing_probs   : (B, P)   — 전체 분포 (entropy loss용, STE-style for M=1)
        topM_idx        : (B, M)   — top-M centroid 인덱스 (retrieve용)
        topM_weights    : (B, M)   — top-M softmax 가중치 (differentiable)

        Backward-compatibility
        ──────────────────────
        top_m=1일 때:
          - hard_assignment, routing_probs 동작은 기존과 동일 (STE 유지)
          - context_emb = centroid_emb[hard_assignment] (기존과 동일)
          - topM_idx = hard_assignment.unsqueeze(1) — (B, 1)
          - topM_weights = ones(B, 1) — 단일 centroid이므로 1.0
        """
        # 코사인 유사도 로짓
        q = F.normalize(query_emb, dim=-1)               # (B, D)
        c = F.normalize(self.centroid_emb, dim=-1)        # (P, D)
        logits = (q @ c.T) * self.routing_scale           # (B, P) — scale 적용

        # ── 전체 softmax (STE backward gradient용, M=1에서 STE 적용) ─
        soft = F.softmax(logits, dim=-1)                  # (B, P)

        # ── Top-M soft routing (핵심 변경) ──────────────
        top_m_eff = min(top_m, self.P)  # P 초과 방지
        topM_logits, topM_idx = logits.topk(top_m_eff, dim=-1)  # (B, M)

        # ── hard_assignment: Top-1 (FAISS 마스킹 + 설명용) ──
        hard_assignment = topM_idx[:, 0]                  # (B,)

        # ── routing_probs 분기 ──────────────────────────
        # M=1: 기존 STE 동작 유지 (backward gradient가 soft를 거쳐 흐르게 함)
        # M>1: 일반 softmax (hierarchical 경로에서는 STE 의미 없음)
        if top_m_eff == 1:
            # STE: forward는 hard (argmax), backward는 soft gradient
            # training/eval 무관하게 항상 STE 유지.
            #
            # [근거]
            # VQ-VAE (van den Oord et al. 2017)와 Bengio et al. 2013의 STE 원 정의는
            # training/eval 구분 없이 forward=hard, backward=soft를 유지한다.
            # eval에서 STE를 끄면 ∂context_emb/∂query_emb = 0이 됨.
            # [갱신] 이 gradient 경로는 원래 ③(IG)이 eval 모드에서 context_emb
            # 기여를 측정하려고 필요했던 것인데, ③이 SHAP(gradient 불필요한
            # black-box 방법)으로 통일되면서 그 필요성 자체는 없어짐. 다만
            # eval에서도 STE를 유지하는 이 코드는 forward 값 기준으로 hard
            # argmax와 완전히 동일하므로(∵ soft + (hard-soft).detach() = hard,
            # 값 기준) 굳이 되돌릴 이유도 없어 그대로 둠 — 순수 gradient
            # 경로 하나가 이제 아무도 안 쓰는 채로 남아있을 뿐, 예측 결과에는
            # 영향 없음.
            hard_one_hot = F.one_hot(hard_assignment, self.P).float()  # (B, P)
            routing_probs = soft + (hard_one_hot - soft).detach()      # STE (always)

            # topM_weights = 1.0 (단일 centroid)
            topM_weights = torch.ones_like(topM_logits)   # (B, 1)

            # context_emb: 기존과 동일 (routing_probs @ centroid_emb)
            context_emb = self.dropout(routing_probs @ self.centroid_emb)
        else:
            # Hierarchical 경로
            # routing_probs는 forward() 반환값(diagnose/설명 등 호출부에서
            # 전체 분포를 참조할 수 있게) — soft 그대로, gradient 흐름
            routing_probs = soft

            # topM_weights: top-M에 대한 softmax (differentiable!)
            topM_weights = F.softmax(topM_logits, dim=-1) # (B, M)

            # context_emb: top-M centroid 가중 혼합
            # ★ topM_weights가 differentiable → centroid 선택에 gradient 흐름
            topM_centroids = self.centroid_emb[topM_idx]  # (B, M, D)
            context_emb = (
                topM_weights.unsqueeze(-1) * topM_centroids
            ).sum(dim=1)                                   # (B, D)
            context_emb = self.dropout(context_emb)

        return context_emb, hard_assignment, routing_probs, topM_idx, topM_weights

    # ─────────────────────────────────────────────────────────
    # Auxiliary Losses (기존 tabr.py 호환)
    # ─────────────────────────────────────────────────────────

    def diversity_loss(self) -> torch.Tensor:
        """Centroid 붕괴 방지: off-diagonal cosine similarity 최소화.
        clamp(max=1e4): STE collapse 시 nan 전파 방지
        """
        c = F.normalize(self.centroid_emb, dim=-1)
        sim = c @ c.T
        mask = 1.0 - torch.eye(self.P, device=sim.device)
        loss = (sim.pow(2) * mask).sum() / (self.P * (self.P - 1))
        return loss.clamp(max=1e4)  # nan 방지

    # [제거됨] entropy_loss — 정의만 되어 있고 tabera.py의 aux_loss 조합
    # (diversity + commitment)에 실제로는 한 번도 연결된 적 없는 죽은
    # 코드였음. 참고로 이 손실은 "배치 평균 라우팅 분포"의 entropy를
    # 최대화하는 것(codebook utilization/dead-centroid 방지, VQ-VAE-2
    # 방식)이라 "샘플 하나하나의 라우팅을 confident하게 만드는" 것과는
    # 다른 목적이었음 — 그 목적(샘플별 confidence)을 위해서는 이것과
    # 다른 손실(샘플별 entropy 최소화, entropy minimization 계열)이
    # 필요하며, 별도 검토 없이 지금 다시 넣지 않기로 함.

    def cosine_similarity_matrix(self) -> torch.Tensor:
        """진단용: centroid 간 cosine similarity 행렬 반환 (P, P)."""
        c = F.normalize(self.centroid_emb.detach(), dim=-1)
        return (c @ c.T).cpu()

    def commitment_loss(
        self, query_emb: torch.Tensor, hard_assignment: torch.Tensor
    ) -> torch.Tensor:
        """쿼리를 배정된 centroid 방향으로."""
        assigned = self.centroid_emb[hard_assignment]
        return F.mse_loss(query_emb, assigned.detach())

    # ─────────────────────────────────────────────────────────
    # 설명 헬퍼 (기존 tabr.py 호환 + 원본 feature 값 추가)
    # ─────────────────────────────────────────────────────────

    def explain_routing(
        self,
        hard_assignment: torch.Tensor,   # (B,)
        routing_probs: torch.Tensor,     # (B, P)
        norm_mean: Optional[np.ndarray] = None,  # 더 이상 안 씀 (centroid_x
                                                  # 역정규화용이었음) — 호출부
                                                  # 시그니처 호환을 위해 유지
        norm_std:  Optional[np.ndarray] = None,  # 위와 동일
    ) -> List[dict]:
        """
        샘플별 centroid 배정 설명.

        ①의 주 콘텐츠는 target_labels(label_groups_by_target() 결과)
        — "이 그룹은 대체로 어떤 target(클래스)인가". 배정된 그룹뿐
        아니라 runner-up 그룹들도 각자의 target_info를 같이 반환한다
        — runner-up도 결국 "이 샘플이 속할 뻔한 다른 그룹"이라, 그
        그룹이 어떤 target인지도 같이 봐야 맥락이 온전해진다.
        group_feature_labels는 그 그룹을 다른 그룹들과 가장 뚜렷이
        구별시키는 feature들의 실제 그룹 평균값(원값) — 보조 정보로
        같이 반환한다. 캐싱 전(supervised.py 연결 안 된 경우)이면
        각각 None/빈 리스트.
        """
        pa   = hard_assignment.detach().cpu().numpy()
        pr   = routing_probs.detach().cpu().numpy()

        out  = []

        for b in range(pa.shape[0]):
            p     = int(pa[b])
            label = self.labels[p]
            conf  = float(pr[b, p])

            runner_idx = sorted(
                [i for i in range(self.P) if i != p],
                key=lambda i: -float(pr[b, i]),
            )[:2]
            runners = [
                {
                    "label":       self.labels[i],
                    "confidence":  float(pr[b, i]),
                    "target_info": (self.target_labels.get(i)
                                     if self.target_labels is not None else None),
                }
                for i in runner_idx
            ]

            # ①의 주 콘텐츠: 이 그룹이 어떤 target(클래스)에 해당하는가
            target_info = (
                self.target_labels.get(p)
                if self.target_labels is not None else None
            )

            # 보조 정보: feature별 "매우 높음/높음/보통/..." 요약
            group_feature_labels = (
                self.group_labels.get(p, [])
                if self.group_labels is not None else []
            )

            out.append({
                "assigned_group":       label,
                "centroid_idx":         p,
                "group_confidence":     conf,
                "runners_up":           runners,
                "target_info":          target_info,          # ← ①의 주 콘텐츠
                "group_feature_labels": group_feature_labels,  # ← 보조 정보
            })
        return out

    def centroid_summary(self, top_n: int = 3) -> str:
        """
        전체 centroid의 그룹 크기 + target 분포 + 두드러진 feature
        평균값 요약 출력.
        top_n: 그룹당 보여줄 feature 개수 상한.
        """
        lines = [f"CentroidLayer — {self.P} centroids", "─" * 44]

        for p in range(self.P):
            grp_size = (len(self.sample_groups[p])
                        if self.sample_groups else "?")
            line = f"  [{self.labels[p]}]  n={grp_size}"

            # ①의 콘텐츠: target 분포
            tinfo = self.target_labels.get(p) if self.target_labels else None
            if tinfo is not None:
                if tinfo["kind"] == "classification":
                    line += f"  → {tinfo['top_class_name']} {tinfo['top_count']}/{tinfo['n']} ({tinfo['top_prop']:.0%})"
                else:
                    line += f"  → target≈{tinfo['group_mean']:.3g}(p{tinfo['percentile']:.0f})"

            # 두드러진 feature의 그룹 평균값
            labels_p = self.group_labels.get(p) if self.group_labels else None
            if labels_p:
                vals = ", ".join(f"{fl.feature_name}={fl.label}" for fl in labels_p[:top_n])
                line += f"  [{vals}]"
            lines.append(line)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 하위 호환 alias (tabera.py import 수정 불필요)
# ─────────────────────────────────────────────────────────────
PrototypeLayer = CentroidLayer