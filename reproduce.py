## Main file for reproducing the best TabERA configuration.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import sys, os, argparse

# ── CUDA_VISIBLE_DEVICES: torch import 전 설정 ──────────────
_parser_pre = argparse.ArgumentParser(add_help=False)
_parser_pre.add_argument("--gpu_id", type=int, default=0)
_pre, _ = _parser_pre.parse_known_args()
if _pre.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.gpu_id)

import joblib, json, pickle, datetime
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from libs.data         import TabularDataset
from libs.search_space import params_to_model_kwargs
from libs.supervised   import TabERAWrapper
from libs.tabera         import TabERA
from libs.prototypes     import inverse_transform_numeric
from libs.eval         import calculate_metric
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────
# 설명 출력 (①② architectural + ③ SHAP post-hoc)
# ─────────────────────────────────────────────────────────────

def _fmt_signed(x: float, decimals: int = 4) -> str:
    """
    부호 있는 소수 포맷팅 전용 — 아주 작은 음수(예: -0.00003)가 반올림되면
    파이썬이 "-0.0000"으로 찍어서, 실제로는 0에 불과한 값이 마치 의미 있는
    음의 값처럼 오해를 살 수 있다(rank_correlation의 random null 평균
    corr_rand가 대표적 — 무작위 순위끼리의 기대 상관은 0이라 이런 미세
    음수가 흔히 나옴). round() 후 +0.0을 더해 음의 0을 양의 0으로
    정규화한 뒤 포맷한다(IEEE754에서 -0.0 + 0.0 == 0.0).
    """
    v = round(x, decimals) + 0.0
    return f"{v:.{decimals}f}"


def _fmt_pval(p: float, n_draws: int) -> str:
    """
    Bootstrap/permutation 기반 경험적 p-value 포맷팅 전용.

    n_draws번 무작위 재표본추출 중 관측값을 한 번도 못 넘으면(count=0)
    p=0.0000으로 그대로 찍기 쉬운데, 이건 "확률이 정확히 0"이라는 뜻이
    아니라 "n_draws번 중 한 번도 못 봤다"는 관측 해상도의 한계일 뿐이다
    (실제 p-value는 1/n_draws보다 작다는 것만 알 수 있음 — 0이라는 뜻은
    아님). rank_correlation의 p_shap_vs_null, interaction_check의
    p_vs_null 둘 다 이 문제를 갖고 있어 공용 헬퍼로 분리함.
    """
    if p <= 0.0:
        return f"<{1.0 / n_draws:.4g}"
    return f"{p:.4f}"


def _fmt_class(name: str, count: int, n: int, prop: float) -> str:
    """하나의 클래스를 "name" count/n (prop%) 형식으로. top/second 어디서
    부르든 항상 이 하나의 함수만 거치게 해서, 포맷이 서로 어긋나는 걸 막는다
    (이전에 top은 "(count/n, prop%)", second는 "count/n (prop%)"로 서로
    다른 괄호 스타일을 쓰던 문제가 있었음 — 데이터셋과 무관하게 항상 이
    함수 하나로 통일)."""
    return f"\"{name}\" {count}/{n} ({prop:.0%})"


def _format_target_info(tinfo) -> str:
    """target_info(label_groups_by_target() result) as a short string."""
    if tinfo is None:
        return "(no target info)"
    if tinfo["kind"] == "classification":
        s = _fmt_class(tinfo['top_class_name'], tinfo['top_count'], tinfo['n'], tinfo['top_prop'])
        if tinfo["second"] is not None:
            s += ", " + _fmt_class(tinfo['second']['name'], tinfo['second']['count'],
                                    tinfo['n'], tinfo['second']['prop'])
        return s
    else:
        return f"target≈{tinfo['group_mean']:.3g}(p{tinfo['percentile']:.0f})"


def _select_query_similar_features(
    query: dict, neighbour: dict, cat_names: set,
    max_n: int = 4, max_gap: float = 0.15,
) -> list:
    """
    "이 이웃의 값이 원래 크다"가 아니라 "query와 이 이웃이 이 feature에서
    얼마나 가까운가"로 feature를 고른다 — query도 안 보여주고 이웃 혼자
    값이 큰 feature만 나열하면 "그래서 왜 비슷한 이웃인지" 설명이 안 됨.

    numeric은 |query-neighbour| (이미 [0,1] 정규화됨), categorical은
    같으면 0/다르면 1 (Gower distance와 동일한 방식 — LabelEncoder 정수
    코드에 순서가 없어 그냥 뺄셈하면 안 됨). gap이 작을수록(=가까울수록)
    상위로 정렬하고, max_gap을 넘는 건 애초에 후보에서 제외한다 — 그래서
    정말 비슷한 feature가 몇 개 없는 이웃은 개수가 max_n보다 적게 나올
    수 있다(숫자 채우기용으로 안 비슷한 feature를 억지로 넣지 않음).
    비슷한 feature가 하나도 없으면(전부 max_gap 초과) 그래도 가장 가까운
    1개는 보여준다 — 완전히 빈 설명보다는 "그나마 제일 가까운 게 이거"가 낫다.

    반환값: [(name, value, kind), ...] — kind는 "numeric"|"categorical".
    호출부에서 kind별로 나눠 보여줄 수 있게 dict 대신 list로 반환한다.
    """
    diffs = []
    for k, v in neighbour.items():
        if k not in query:
            continue
        is_cat = k in cat_names
        gap = (0.0 if query[k] == v else 1.0) if is_cat else abs(query[k] - v)
        diffs.append((k, v, gap, "categorical" if is_cat else "numeric"))
    if not diffs:
        return []
    diffs.sort(key=lambda x: x[2])
    selected = [(k, v, kind) for k, v, gap, kind in diffs if gap <= max_gap][:max_n]
    if not selected:
        k, v, gap, kind = diffs[0]
        selected = [(k, v, kind)]
    return selected


def _split_by_kind(labels, get_kind, get_str):
    """items를 kind별(numeric/categorical)로 나눠 두 개의 문자열 리스트로."""
    num_strs, cat_strs = [], []
    for item in labels:
        (num_strs if get_kind(item) == "numeric" else cat_strs).append(get_str(item))
    return num_strs, cat_strs


def print_explanation(explanations: list, sample_idx: int, col_names: list,
                       cat_category_names: dict = None,
                       quantile_transformer=None, num_cols: list = None) -> None:
    e = explanations[sample_idx]

    print(f"\n{'━'*52}")
    print(f"  TabERA Explanation — Sample #{sample_idx}")
    print(f"{'━'*52}")

    # ① Prototype group (target distribution — which class does this group represent?)
    proto = e["prototype"]
    print(f"\n  ① Prototype Group")

    # 이 그룹의 target(클래스) 분포 — ①의 주 콘텐츠 (label_groups_by_target(),
    # ema_update() 직후 캐싱됨). ②(실제 이웃의 raw feature 값)와 정보 종류가
    # 겹치지 않도록, feature 요약이 아니라 "이 그룹이 어떤 부류인가"만 보여준다.
    # [배치] 배정된 그룹 이름/confidence와 같은 줄에 붙여서, "이 그룹이 뭔지"를
    # 한눈에 읽을 수 있게 함 (Runner-up은 부가 정보라 그 다음 줄로 내림).
    tinfo = proto.get("target_info")
    if tinfo is not None:
        if tinfo["kind"] == "classification":
            target_str = _fmt_class(tinfo['top_class_name'], tinfo['top_count'], tinfo['n'], tinfo['top_prop'])
            if tinfo["second"] is not None:
                target_str += ", also " + _fmt_class(tinfo['second']['name'], tinfo['second']['count'],
                                                       tinfo['n'], tinfo['second']['prop'])
        else:
            target_str = (f"target mean {tinfo['group_mean']:.3g} "
                           f"(percentile {tinfo['percentile']:.0f}, n={tinfo['n']})")
    else:
        target_str = "(no group target info — target_labels may not have been cached during training)"

    print(f"     → \"{proto['assigned_group']}\"  (confidence={proto['group_confidence']:.1%})  —  {target_str}")

    if proto["runners_up"]:
        ru = ", ".join(
            f"\"{r['label']}\"({r['confidence']:.1%}, {_format_target_info(r['target_info'])})"
            for r in proto["runners_up"]
        )
        print(f"     Runner-up: {ru}")

    # 이 그룹을 다른 그룹들과 가장 뚜렷이 구별시키는 feature의 실제
    # 그룹 평균값(label_all_groups, 그룹 간 대비(distinctiveness) 상위 K개).
    # numeric/categorical을 나눠서 보여줌 — 섞어서 나열하면 스케일이 전혀
    # 다른 값(원시 비율 vs 카테고리 코드+비율)을 한 줄로 읽어야 해서 헷갈림.
    labels = proto.get("group_feature_labels", [])
    if labels:
        num_strs, cat_strs = _split_by_kind(
            labels, get_kind=lambda fl: fl.kind,
            get_str=lambda fl: f"{fl.feature_name}={fl.label}",
        )
        print(f"     Distinctive features:")
        if num_strs:
            print(f"       numeric:     {',  '.join(num_strs)}")
        if cat_strs:
            print(f"       categorical: {',  '.join(cat_strs)}")

    # ② Neighbor evidence (Attention weight)
    ev = e["evidence"]
    print(f"\n  ② Neighbor Evidence (Attention)")
    print(f"     dominant={ev['dominant_weight']:.1%},  entropy={ev['entropy']:.3f}")

    # 기여도가 사실상 0인 이웃은 생략 (반올림하면 0.0%로 보이는 것도 포함) —
    # 예측에 아무 영향을 안 준 이웃까지 보여주는 건 정보가 아니라 소음이다.
    _WEIGHT_EPS = 1e-3
    shown = [(rank, idx, w) for rank, (idx, w) in enumerate(ev["top_neighbours"])
              if w > _WEIGHT_EPS]

    if not shown:
        print(f"     (no neighbor contributed meaningfully)")

    nf = e.get("neighbour_features")
    name_to_idx = {name: i for i, name in enumerate(col_names)} if col_names else {}

    def _fmt_cat_value(name: str, code_val: float) -> str:
        # cat_category_names(libs/data.py의 load_data() 결과)가 있으면
        # 실제 카테고리 문자열 + 원래 코드 번호를 같이, 없으면 코드만.
        names_for_col = cat_category_names.get(name) if cat_category_names else None
        code = int(code_val)
        if names_for_col is not None and 0 <= code < len(names_for_col):
            return f"{name}={names_for_col[code]} [{code}]"
        return f"{name}=Category {code}"

    def _fmt_num_value(name: str, uniform_val: float) -> str:
        # quantile_transformer(libs/data.py의 prep_data() 결과)가 있으면
        # [0,1] uniform 값을 실제 단위로 역변환 — ①의 Distinctive features와
        # 같은 처리를 ②의 이웃 feature 값에도 동일하게 적용.
        if quantile_transformer is not None and num_cols is not None and name in name_to_idx:
            real_val = inverse_transform_numeric(quantile_transformer, num_cols, name_to_idx[name], uniform_val)
            if real_val is not None:
                return f"{name}={real_val:.3g}"
        return f"{name}={uniform_val:.3f}"

    for rank, idx, w in shown:
        print(f"     #{rank+1} Neighbor {idx}: {w:.1%}")
        if nf and idx < len(nf) and nf[idx]:
            num_strs, cat_strs = _split_by_kind(
                nf[idx], get_kind=lambda item: item[2],
                get_str=lambda item: (_fmt_cat_value(item[0], item[1])
                                       if item[2] == "categorical" else _fmt_num_value(item[0], item[1])),
            )
            if num_strs:
                print(f"        → numeric:     {', '.join(num_strs)}")
            if cat_strs:
                print(f"        → categorical: {', '.join(cat_strs)}")

    print(f"{'━'*52}")


# ─────────────────────────────────────────────────────────────
# [제거됨] Integrated Gradients (Sundararajan et al. 2017, ICML)
# ─────────────────────────────────────────────────────────────
# compute_integrated_gradients / make_logit_target_fn 두 함수를 여기서
# 제거함. ③(Feature Attribution)을 SHAP으로 통일하기로 확정한 이유:
#   1. IG는 categorical feature에서 근본적으로 깨짐 — libs/tabera.py의
#      _encode_categorical()이 x.round().long()으로 정수 캐스팅하는
#      순간 autograd 그래프가 끊겨, categorical column의 gradient가
#      항상 정확히 0이 됨(토이 예제로 재현 확인됨). 전부 categorical인
#      데이터셋(splice 등)에서는 아예 RuntimeError로 크래시.
#   2. IG는 연속 경로 적분(baseline→input)을 전제하는 방법이라 이산
#      입력에 원천적으로 안 맞음 — 문헌에서도 "모델이 미분가능해야
#      하며, 이는 비미분 요소나 workaround 없는 이산 입력에 직접
#      적용하는 것을 제한한다"고 명시적으로 분류됨(Turing Institute
#      TEA Techniques 등).
#   3. SHAP(Shapley value)은 gradient가 아니라 함수를 여러 번 평가하는
#      black-box perturbation 방법이라 이 문제 자체가 없고, 게다가
#      efficiency/symmetry/dummy/additivity 네 공리를 만족하는 유일한
#      배분 규칙이라는 이론적 근거도 있음(Lundberg & Lee 2017).
# SHAP 계산은 rank_correlation ablation 내부(model_predict 클로저 +
# shap.KernelExplainer)에서 직접 이뤄짐 — 별도 top-level 함수로 뺄
# 만큼 여러 곳에서 재사용되지 않아 그대로 inline.


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TabERA Reproduce Best Config")
    parser.add_argument("--gpu_id",    type=int, default=0)
    parser.add_argument("--openml_id", type=int, required=True)
    parser.add_argument("--savepath",  type=str, default=".",
                        help="optim_logs가 있는 상위 경로")
    parser.add_argument("--seed",      type=int, default=1,
                        help="optimize.py와 동일한 seed 사용")
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--patience",  type=int, default=30)
    parser.add_argument("--n_explain", type=int, default=3,
                        help="설명 출력할 테스트 샘플 수")
    parser.add_argument("--explain",   action="store_true",
                        help="학습 후 feature 기여도 설명 출력")
    parser.add_argument("--from_saved_state", type=str, default=None,
                        help=(
                            "이전 실행이 저장한 *_model_state.pt 경로를 넘기면 "
                            "재학습을 완전히 건너뛰고 그 상태를 그대로 복원해서 "
                            "--explain / --ablation만 다시 돌린다. optimize.py의 "
                            "study 파일도 필요 없음(model_kwargs를 이 파일에서 "
                            "직접 읽음). --n_explain/--ablation 등 다른 인자는 "
                            "그대로 같이 쓰면 됨. seed/openml_id는 저장 당시와 "
                            "일치해야 dataset 분할이 같아짐 — 지금 CLI에 준 값을 "
                            "그대로 쓰므로 저장했을 때와 동일하게 넘길 것."
                        ))
    parser.add_argument("--ablation",  type=str, default="none",
                        choices=["none", "random_neighbor", "neighbor_noise",
                                 "rank_correlation", "dual_space_faithfulness",
                                 "interaction_check",
                                 "dataset_profile"],
                        help=(
                            "ablation 모드 선택 (학습된 모델에 inference 단계에서 적용):\n"
                            "  none                  : full model 기준 (기본값)\n"
                            "  random_neighbor       : nk/labels를 같은 permutation으로\n"
                            "                         통째로 셔플 — 배치 내 다른 쿼리의 진짜\n"
                            "                         (real) 이웃 세트로 통째로 교체.\n"
                            "                         retrieval이 '맞는 이웃'을 찾았는지만\n"
                            "                         순수하게 검증 (이웃 정보 자체는 real).\n"
                            "  neighbor_noise        : nk/labels 전부 실제 데이터와 무관한\n"
                            "                         노이즈/재추출 라벨로 교체. '이웃 정보가\n"
                            "                         조금이라도 존재하는가' 자체를 검증.\n"
                            "                         (random_neighbor와 함께 봐야 함 —\n"
                            "                         이 둘의 성능 하락 차이가 '틀린 이웃'과\n"
                            "                         '이웃 없음'의 영향을 분리해서 보여줌)\n"
                            "  rank_correlation      : ③(SHAP) feature 순위가 Delta(단순\n"
                            "                         1차 perturbation) 순위와 어느 정도\n"
                            "                         정합하는지 보는 실험. [주의] Delta는\n"
                            "                         ground truth가 아니라 low-fidelity\n"
                            "                         baseline(feature 상호작용을 못 봄) —\n"
                            "                         '정합도가 낮다'가 곧 'SHAP이 틀렸다'는\n"
                            "                         뜻이 아님. random null과 SHAP MC 노이즈\n"
                            "                         까지 같이 봐야 해석 가능.\n"
                            "  dual_space_faithfulness : sample_groups 인덱스 정합성 +\n"
                            "                         그룹 분리도(F-test/χ²) 검증\n"
                            "  interaction_check      : 두 feature를 동시에 perturb했을 때의\n"
                            "                         변화 vs 개별 perturb 합의 차이로,\n"
                            "                         '이 데이터셋에 SHAP이 잡아야 할 만큼\n"
                            "                         유의미한 feature 상호작용이 실제로\n"
                            "                         있는가'를 rank_correlation과 별개로\n"
                            "                         직접 확인. rank_correlation에서\n"
                            "                         SHAP-Delta 불일치가 나왔을 때, 그게\n"
                            "                         상호작용 때문인지 SHAP 추정 오차\n"
                            "                         때문인지 구분하는 데 씀.\n"
                            "  dataset_profile        : 예측 확신도, fallback 비율 등 빠른\n"
                            "                         데이터셋 진단(예전엔 IG completeness/\n"
                            "                         deletion_auc 포함했으나 ③=SHAP 통일로\n"
                            "                         해당 부분은 제거 — rank_correlation이\n"
                            "                         그 역할을 대신함)."
                        ))
    parser.add_argument("--detach_context_grad", action="store_true",
                        help=(
                            "[진단용] context_emb는 head 입력으로 그대로 전달하되, "
                            "그쪽에서 오는 gradient만 centroid_emb로 안 흐르게 끊음 "
                            "(commitment_loss는 원래도 detach라 영향 없음, diversity_loss "
                            "gradient는 그대로 흐름). 'task_loss와 diversity_loss가 "
                            "centroid_emb를 두고 서로 다른 방향으로 당기며 충돌하고 있는지' "
                            "검증용."
                        ))
    parser.add_argument("--cat_combine", type=str, default="onehot", choices=["sum", "concat", "onehot"],
                        help=(
                            "categorical embedding 결합 방식. 'onehot'(기본값, 채택 확정)은 "
                            "TabR/ModernNCA 계보를 따름 — 학습 파라미터 없는 순수 one-hot(컬럼별 "
                            "자리 보장, 정보 섞임 없음). 'sum'은 컬럼별 embedding(embed_dim 폭)을 "
                            "더함 — 초기 구현, 기존 sum 체크포인트와 하위 호환용으로 남겨둠. "
                            "'concat'은 Guo & Berkhahn(2016) 원 논문 방식 — 컬럼별로 작은 "
                            "embedding(--cat_embed_dim)을 만들어 이어붙인 뒤 최종 Linear로 "
                            "embed_dim에 투영."
                        ))
    parser.add_argument("--cat_embed_dim", type=int, default=16,
                        help="cat_combine=concat일 때 컬럼별 embedding 차원 (기본 16).")
    parser.add_argument("--num_embedding", type=str, default="plr_lite",
                        choices=["linear", "ple", "plr_lite"],
                        help=(
                            "numeric feature 인코딩 방식. 'plr_lite'(기본값, 채택 확정)는 "
                            "TabR(Gorishniy et al. 2024)/ModernNCA가 실제로 쓰는 방식 — 학습"
                            "가능한 주기(periodic) 함수 + 전체 컬럼이 공유하는 Linear+ReLU "
                            "(공식 구현과 수식 대조 검증됨). 'linear'는 raw 값을 그대로 Linear에 "
                            "투영 — 기존 동작, 하위 호환용. 'ple'는 Piecewise Linear Encoding "
                            "(구간/bin 기반, Gorishniy et al. 2022) — plr_lite와 완전히 다른 "
                            "메커니즘. numeric feature가 매우 적은 데이터셋(예: 1~5개)에서는 "
                            "plr_lite가 오히려 불안정할 수 있음(실측: profb에서 auroc가 무작위 "
                            "수준까지 하락) — 그런 경우 --num_embedding linear/ple 고려."
                        ))
    parser.add_argument("--num_bins", type=int, default=8,
                        help="num_embedding=ple일 때 컬럼당 구간(bin) 개수 (기본 8 — 48보다 "
                             "여러 데이터셋에서 더 나은 calibration 확인 후 기본값 변경).")
    parser.add_argument("--plr_n_frequencies", type=int, default=16,
                        help="num_embedding=plr_lite일 때 컬럼별 주기 함수 주파수 개수 (기본 16).")
    parser.add_argument("--plr_freq_scale", type=float, default=0.01,
                        help="num_embedding=plr_lite일 때 주파수 초기화 스케일 (기본 0.01, "
                             "TabR 논문 권장 탐색 범위: LogUniform[0.01, 100.0]).")
    parser.add_argument("--plr_out_dim", type=int, default=8,
                        help="num_embedding=plr_lite일 때 컬럼당 최종 출력 차원 (기본 8).")
    parser.add_argument("--context_projection", action="store_true",
                        help=(
                            "[구조 조정] context_emb를 head로 보내기 전 학습 가능한 "
                            "Linear를 하나 거치게 함. detach_context_grad와 달리 "
                            "gradient가 여전히 centroid_emb까지 도달함. optimize.py "
                            "--context_projection으로 학습한 study가 있으면 그 "
                            "best_params를 쓰는 게 이상적이지만, 없으면 기존 study "
                            "best_params 위에 이 구조만 얹어 1회 재학습(별도 study 불필요)."
                        ))
    parser.add_argument("--shap_background", type=int, default=50,
                        help=(
                            "rank_correlation의 SHAP KernelExplainer background 샘플 수. "
                            "기본 50. [실측 확인됨] nsamples가 F 대비 부족한 상태에서 이 값만 "
                            "늘리면 오히려 정합도가 떨어질 수 있음(jasmine, F=144: "
                            "background 50→200 단독으로 올렸더니 ρ 0.53→0.36으로 악화) — "
                            "nsamples가 --shap_nsamples(기본 auto)로 충분히 확보된 상태에서만 "
                            "이 값을 올리는 걸 권장."
                        ))
    parser.add_argument("--shap_nsamples", type=int, default=None,
                        help=(
                            "rank_correlation의 SHAP KernelExplainer nsamples(perturbation 표본 "
                            "수). 기본값 None → SHAP 라이브러리 자체의 'auto' 공식을 그대로 씀 "
                            "(nsamples = 2*n_features + 2048, shap 공식 문서 기준). [실측 확인됨] "
                            "이전엔 비용 절감 목적으로 n_features와 무관하게 100으로 고정했었는데, "
                            "jasmine(F=144)에서 nsamples 100→500만으로도 ρ가 0.53→0.63로 뚜렷이 "
                            "올랐음 — F 대비 nsamples가 부족하면(KernelSHAP이 내부에서 푸는 가중"
                            "회귀가 사실상 미지수>관측치인 underdetermined 상태가 되어) 추정치가 "
                            "체계적으로 편향됨. auto 공식은 이 문제를 원천적으로 피하도록 "
                            "설계되어 있어(라이브러리 자체가 F에 비례해 표본을 늘림), 임의로 "
                            "고정값을 주는 것보다 기본값으로 더 적합함. 정수를 명시하면 auto 대신 "
                            "그 값을 그대로 씀(--shap_repeats로 MC 노이즈 진단 시 등 실험 목적)."
                        ))
    parser.add_argument("--shap_repeats", type=int, default=1,
                        help=(
                            "rank_correlation에서 SHAP KernelExplainer 자체의 몬테카를로 "
                            "노이즈(같은 샘플에 대해서도 background/nsamples 표본추출에 "
                            "따라 값이 흔들리는 정도)를 진단하기 위해 SHAP 계산을 몇 번 "
                            "반복할지. 기본값 1 = 반복 안 함(기존 동작과 동일, 추가 비용 "
                            "없음). 2 이상이면 매번 다른 random background로 SHAP을 "
                            "다시 계산해 corr_shap의 반복 간 표준편차를 보고함 — feature 수가 "
                            "많은 데이터셋에서는 그만큼 배로 느려지므로 필요할 때만 켤 것."
                        ))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── 데이터 로드 ────────────────────────────────────────
    with open(args.json, "r") as f:
        data_info = json.load(f)

    openml_id    = str(args.openml_id)
    dataset_info = data_info[openml_id]
    tasktype     = dataset_info["tasktype"]
    print(f"[TabERA Reproduce] {dataset_info['fullname']} (id={openml_id}, task={tasktype})")

    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std      = dataset.y_std
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")

    # ── best params 로드 ───────────────────────────────────
    # optimize.py 저장 경로와 동일하게 맞춤
    if not args.savepath.endswith("optim_logs"):
        log_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
    else:
        log_dir = args.savepath

    _save_tag = ("..detach_ctx" if args.detach_context_grad else "") \
              + ("..ctx_proj" if args.context_projection else "") \
              + ("..cat_concat" if args.cat_combine == "concat" else "") \
              + ("..cat_onehot" if args.cat_combine == "onehot" else "") \
              + ("..num_ple" if args.num_embedding == "ple" else "") \
              + ("..num_plr" if args.num_embedding == "plr_lite" else "")

    _saved_state = None
    if args.from_saved_state:
        # ── --from_saved_state: study 파일 불필요, 저장된 model_kwargs를
        # 그대로 씀. 재학습을 건너뛰므로 --epochs/--patience는 무시됨.
        print(f"  [--from_saved_state] {args.from_saved_state} 로드 중 (재학습 생략)")
        # [수정] PyTorch 2.6부터 torch.load()의 기본값이 weights_only=True로
        # 바뀌어서, sample_groups/group_labels에 들어있는 커스텀 클래스
        # (FeatureLabel 등)를 안전 목록에 없다는 이유로 거부한다. 이 파일은
        # 우리가 방금 위에서 직접 저장한 신뢰 가능한 파일이라(외부에서
        # 받은 게 아님) weights_only=False로 명시.
        _saved_state = torch.load(args.from_saved_state, map_location=device, weights_only=False)
        model_kwargs = _saved_state["model_kwargs"]
        best_params  = _saved_state.get("best_params", {})
        if best_params:
            print(f"  Params(저장된 값): {best_params}")
        # [하위 호환] 이번 --from_saved_state 지원 이전에 저장된 파일은
        # model_kwargs에 memory_size가 안 들어있어서(예전엔 TabERA(...)
        # 호출 시 별도 kwarg로만 넘기고 model_kwargs 딕셔너리 자체에는
        # 안 합쳐졌음), 새로 모델을 만들면 TabERA 기본값(10000)으로
        # 만들어져 체크포인트의 실제 크기(n_train)와 안 맞아 로딩이
        # 깨진다. n_train은 예전 포맷에도 있었으니 그걸로 대체.
        if "memory_size" not in model_kwargs:
            fallback_size = _saved_state.get("n_train")
            if fallback_size is not None:
                model_kwargs = {**model_kwargs, "memory_size": fallback_size}
                print(f"  ⚠️  옛날 포맷 파일(memory_size 없음) — n_train={fallback_size}로 대체."
                      f" sample_groups 등도 없을 수 있으니 아래 경고를 확인하세요.")
    else:
        fname = os.path.join(log_dir, f"data={openml_id}..model=tabera.pkl")
        if not os.path.exists(fname):
            _hint_cmd = f"optimize.py --openml_id {openml_id} --seed {args.seed}"
            raise FileNotFoundError(
                f"최적화 로그 없음: {fname}\n"
                f"먼저 {_hint_cmd} 를 실행하세요."
            )

        study       = joblib.load(fname)
        best_params = study.best_params
        print(f"  Best trial #{study.best_trial.number}  val={study.best_value:.4f}")

        # optimize.py가 실제 사용한 n_prototypes 그대로 복원
        best_params["n_prototypes"] = study.best_trial.user_attrs["n_prototypes_actual"]
        print(f"  n_prototypes (from optimize.py): {best_params['n_prototypes']}")
        print(f"  Params: {best_params}")

        # ── PLE(Piecewise Linear Encoding) 구간 경계 계산 ───────
        # 학습 데이터의 quantile로 컬럼별 구간 경계를 미리 계산해서 넘김
        # (cat_cardinalities와 같은 패턴 — 모델 생성 전에 데이터에서 파생).
        num_bin_edges = None
        if args.num_embedding == "ple" and len(dataset.X_num) > 0:
            X_num_train = X_train[:, dataset.X_num]  # (n_train, n_num)
            q = torch.linspace(0.0, 1.0, args.num_bins + 1, device=X_num_train.device)
            # torch.quantile(input, q, dim=0) → (n_bins+1, n_num), transpose해서 (n_num, n_bins+1)
            num_bin_edges = torch.quantile(X_num_train, q, dim=0).T.contiguous()
            # 동일 quantile 값이 반복되면(예: 이산적인 numeric 컬럼) 구간 폭이 0이
            # 될 수 있음 — PLE의 (hi-lo) 분모에 1e-8을 더해 안전하게 처리하지만,
            # 완전히 동일한 경계가 연속되면 그 구간은 항상 z=0 또는 1로 사실상
            # 죽은 구간이 됨(오류는 아니지만 표현력 낭비). 필요시 --num_bins를
            # 줄이거나 나중에 unique-based binning으로 개선 가능.

        # ── 모델 구성 ──────────────────────────────────────────
        model_kwargs = params_to_model_kwargs(best_params, dataset.n_features, output_dim)
        model_kwargs.update(dict(
            # [수정] optimize.py와 동일하게 캡 제거 (memory_size가 다르면
            # HPO 때 찾은 best_params가 이 재현 실행에서 재현되지 않음)
            memory_size=len(y_train),
            # 채택된 아키텍처: offset correction 사용, group-constrained
            # retrieval, context_emb를 head 입력에 포함 (각각
            # --no_offset_correction/--global_retrieve/--no_context_emb
            # ablation으로 이미 검증 완료된 결정 — 더 이상 옵션으로 안 둠)
            use_offset_correction=True,
            global_retrieve=False,
            use_context_emb=True,
            # [진단용] context_emb는 head에 그대로 전달하되 gradient만 끊음
            detach_context_grad=args.detach_context_grad,
            # [구조 조정] context_emb를 head 직전 Linear 프로젝션에 통과시킴
            use_context_projection=args.context_projection,
            # [후보 A 구현 → 채택 확정] categorical feature를 raw 정수 대신
            # 별도 처리 — TabZilla 29개 baseline 비교에서 cat_ratio와
            # AUROC gap의 견고한 상관관계(Spearman rho=-0.63, p=0.0003)
            # 확인 후 적용.
            cat_col_idx=list(dataset.X_cat),
            num_col_idx=list(dataset.X_num),
            cat_cardinalities=list(dataset.X_cat_cardinality),
            # [채택 확정 — TabR/ModernNCA 계보] categorical=one-hot(학습
            # 파라미터 없음), numeric=PLR(lite)(주기함수+공유 Linear+ReLU,
            # Gorishniy et al. 2024). sum/concat/PLE도 데이터셋에 따라
            # 이겼다 졌다 했지만(특히 numeric feature가 아주 적은 데이터셋
            # 에서 PLR이 불안정한 사례 있었음 — profb), "TabR/ModernNCA를
            # 잇는 retrieval 기반 모델"이라는 아키텍처 정체성을 성능
            # 최적화보다 우선해 이걸로 확정. 필요시 --cat_combine/
            # --num_embedding으로 다른 방식도 여전히 선택 가능.
            cat_combine=args.cat_combine,
            cat_embed_dim=args.cat_embed_dim,
            num_embedding=args.num_embedding,
            num_bin_edges=num_bin_edges,
        ))
        # [수정] plr_freq_scale/plr_n_frequencies/plr_out_dim은 이제
        # search_space.py가 num_embedding="plr_lite"일 때 trial마다 직접
        # 탐색한다(Gorishniy et al. 2022 권장 방식 — 이전엔 optimize.py가
        # 이 값들을 전체 실행에 고정해서, mfeat-fourier/vehicle 같은
        # numeric-only 데이터셋에서 완전 붕괴 trial이 반복 관찰됨).
        # best_params가 새 study(이 값들을 이미 탐색한)면 params_to_model_
        # kwargs()가 이미 model_kwargs에 넣어놨으니 그대로 두고, 구버전
        # study(이 값들을 모르는)라면 CLI 고정값으로 fallback한다 — 무조건
        # .update()로 덮어쓰면 Optuna가 찾은 값을 고정값이 지워버리는
        # 버그가 생기므로 "없을 때만 채움" 방식으로 처리.
        for _key, _default in [
            ("plr_n_frequencies", args.plr_n_frequencies if hasattr(args, "plr_n_frequencies") else 16),
            ("plr_freq_scale",    args.plr_freq_scale if hasattr(args, "plr_freq_scale") else 0.01),
            ("plr_out_dim",       args.plr_out_dim if hasattr(args, "plr_out_dim") else 8),
        ]:
            model_kwargs.setdefault(_key, _default)

    # [필수 수정] AttentionAggregator의 이웃 라벨 인코딩 — classification
    # (nn.Embedding)/regression(nn.Linear) 구분에 필요. model_kwargs 안에
    # 넣어야 --from_saved_state로 저장/재로드할 때도 유지됨 (plr_* 값들과
    # 같은 이유로 setdefault 사용 — 이미 저장된 새 체크포인트를 다시
    # --from_saved_state로 불러올 때 model_kwargs에 이미 들어있는 값을
    # 덮어쓰면 안 됨).
    model_kwargs.setdefault("tasktype", tasktype)
    model_kwargs.setdefault(
        "n_classes",
        output_dim if tasktype == "multiclass" else (2 if tasktype == "binclass" else None),
    )

    model = TabERA(**model_kwargs, column_names=dataset.col_names)

    # ── 학습 (--from_saved_state면 건너뛰고 바로 복원) ───────
    wrapper = TabERAWrapper(
        model, best_params, tasktype,
        device=str(device), epochs=args.epochs, patience=args.patience,
        # 그룹 텍스트 라벨링에 필요 — ①의 그룹 특징 설명은 텍스트
        # 요약(medoid 아님)으로 대체됐고, 이 캐시가 그 역할을 함
        cat_cols=list(dataset.X_cat), num_cols=list(dataset.X_num),
        col_names=dataset.col_names,
        cat_category_names=dataset.cat_category_names,
        target_class_names=dataset.target_class_names,
        quantile_transformer=dataset.quantile_transformer,
    )
    wrapper._data_id = args.openml_id
    if _saved_state is not None:
        # ── 재학습 생략, 저장된 상태 그대로 복원 ──────────────
        model.load_state_dict(_saved_state["state_dict"])
        # state_dict에 안 잡히는 것들(plain Python 속성이라 buffer가 아님)
        # — sample_groups는 group-constrained 검색에 필수라 이게 없으면
        # retrieve()가 제대로 동작 안 함. group_labels/target_labels는
        # ①의 텍스트 라벨. feature_store._store는 ②의 원본 feature 값.
        model.prototype_layer.sample_groups = _saved_state.get("sample_groups")
        model.prototype_layer.group_labels  = _saved_state.get("group_labels")
        model.prototype_layer.target_labels = _saved_state.get("target_labels")
        fs_state = _saved_state.get("feature_store_state")
        if fs_state is not None and model.feature_store is not None:
            store, ptr, filled = fs_state
            model.feature_store._store  = store.to(device)
            model.feature_store._ptr    = ptr
            model.feature_store._filled = filled
        if model.prototype_layer.sample_groups is None:
            print(f"  ⚠️  저장된 state에 sample_groups가 없습니다 — 이 파일은 이번"
                  f" --from_saved_state 지원 이전 버전으로 저장된 것 같습니다."
                  f" group-constrained 검색/①②가 제대로 안 나올 수 있습니다.")
        print(f"  [--from_saved_state] 복원 완료 (epoch 0부터 재학습 안 함)")
    else:
        wrapper.fit(X_train, y_train, X_val, y_val)

    # ── 평가 ──────────────────────────────────────────────
    preds_val  = wrapper.predict(X_val)
    preds_test = wrapper.predict(X_test)
    probs_val  = wrapper.predict_proba(X_val)  if tasktype != "regression" else None
    probs_test = wrapper.predict_proba(X_test) if tasktype != "regression" else None

    if tasktype == "regression":
        val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, None, tasktype, "val")
        test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, None, tasktype, "test")
    else:
        val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
        test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

    print(f"\n  {env_info}  {openml_id}  {dataset_info['name']}  tabera  {log_dir}")
    print(f"  val  : {val_metrics}")
    print(f"  test : {test_metrics}")

    # ── Ablation 평가 ──────────────────────────────────────────
    # 학습된 모델 가중치는 고정한 채, inference 단계에서만 ablation 적용.
    # 따라서 별도 재학습 없이 동일 가중치로 3가지 ablation을 빠르게 비교 가능.
    if args.ablation != "none":
        print(f"\n{'='*60}")
        print(f"  Ablation Mode: {args.ablation}")
        print(f"{'='*60}")

        model.eval()

        # ── rank_correlation: SHAP(③) 순위 vs Delta(1차 신호) 순위 정합성 체크 ──
        if args.ablation == "rank_correlation":
            import shap
            from scipy.stats import spearmanr

            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            # 샘플 수 제한 (SHAP KernelExplainer가 느림)
            n_rc       = min(100, X_test.shape[0])
            _rc_perm   = np.random.RandomState(args.seed).permutation(X_test.shape[0])[:n_rc]
            X_rc       = X_test[_rc_perm]
            X_rc_np    = X_rc.detach().cpu().numpy()
            X_train_np = X_train.detach().cpu().numpy()

            print(f"\n  Rank Correlation — SHAP(③) vs Delta(1차 신호) 정합성 체크 (n={n_rc})")
            print(f"  {'─'*60}")
            print(f"  [주의] 이 실험은 'SHAP이 정확하다'를 증명하는 게 아니라, ")
            print(f"  'SHAP 순위가 단순 1차 perturbation(Delta) 순위와 어느 정도")
            print(f"  일치하는가'를 보는 정합성 체크임. Delta는 feature를 하나씩만")
            print(f"  독립적으로 perturb하는 low-fidelity 방법(Occlusion-1)이라")
            print(f"  고차 feature 상호작용을 못 봄 — SHAP과 Delta가 불일치할 때,")
            print(f"  그게 'SHAP이 틀려서'가 아니라 'SHAP이 Delta는 못 보는 상호작용을")
            print(f"  반영해서'일 수 있음(--ablation interaction_check로 별도 확인 권장).")

            with torch.no_grad():
                logits_orig = model(X_rc)["logits"]           # (N, C) or (N, 1)
                _target_class = (
                    logits_orig.argmax(dim=-1).cpu().numpy()
                    if tasktype == "multiclass" else None
                )

            def _pick_target(logits: torch.Tensor) -> torch.Tensor:
                if tasktype == "multiclass":
                    idx = torch.as_tensor(_target_class, device=logits.device, dtype=torch.long)
                    return logits[torch.arange(logits.shape[0], device=logits.device), idx]
                return logits.squeeze(-1)

            print(f"  [1/3] Delta 순위 계산 중 (feature {n_features}개)...")
            with torch.no_grad():
                train_mean   = X_train.mean(dim=0)             # (F,)
                orig_target  = _pick_target(logits_orig)       # (N,)

                delta_samples = np.zeros((n_rc, n_features))   # (N, F)
                for f in range(n_features):
                    X_masked       = X_rc.clone()
                    X_masked[:, f] = train_mean[f]
                    logits_masked  = model(X_masked)["logits"]
                    masked_target  = _pick_target(logits_masked)
                    delta_samples[:, f] = (orig_target - masked_target).abs().cpu().numpy()

            delta_arr  = delta_samples.mean(axis=0)            # (F,) 점추정치
            delta_rank = np.argsort(np.argsort(-delta_arr))   # 0-based, 낮을수록 중요

            # [SHAP 공식 그대로 사용] --shap_nsamples를 안 주면(None) SHAP
            # 라이브러리 자체의 'auto' 공식(nsamples = 2*n_features + 2048,
            # shap 공식 문서 기준)을 그대로 계산해서 씀. 예전엔 비용 절감
            # 목적으로 n_features와 무관하게 100 고정값을 썼었는데, jasmine
            # (F=144) 실측에서 nsamples 부족이 SHAP 추정치를 체계적으로
            # 편향시키는 게 확인됨(100→500만으로 ρ 0.53→0.63) — 임의
            # 고정값보다 F에 비례해 커지는 auto 공식이 원칙적으로 더 맞고,
            # 상한(cap)은 일부러 두지 않음: cap을 걸면 결국 예전과 같은
            # "F가 큰 데이터셋에서 표본이 F 대비 부족해지는" 문제가 다시
            # 생기기 때문. 비용이 부담되면 --shap_nsamples로 직접 낮은 값을
            # 줘서 의도적으로 근사 정밀도를 낮추는 쪽을 선택할 것.
            _shap_nsamples = (
                args.shap_nsamples if args.shap_nsamples is not None
                else 2 * n_features + 2048
            )
            print(f"  [2/3] SHAP KernelExplainer 실행 중 "
                  f"(background={args.shap_background}, nsamples={_shap_nsamples}"
                  f"{' [auto]' if args.shap_nsamples is None else ''})...")

            def model_predict(x_np):
                # [실측 확인된 OOM 방지] SHAP은 explain 대상 샘플 1개당
                # nsamples×background(auto 기준으로도 수천 단위)행짜리 합성
                # 배치를 model()에 한 번에 통째로 넣으려 한다. 학습 때 배치
                # 크기(보통 128~512)의 수십 배라, group 크기가 큰 데이터셋
                # (예: SpeedDating, 일부 centroid 그룹 크기 2000+)에서는
                # MemoryBank.retrieve()의 "정상 경로" 중간 텐서가 이 배치
                # 크기에 비례해 커져 CUDA OOM으로 죽는 게 실측으로 확인됨
                # (_outlier_threshold는 학습 중 epoch마다만 GPU 여유 메모리
                # 기준으로 재보정되고, 추론/ablation 단계에서는 갱신되지
                # 않아 이 큰 배치에 대응하지 못함). random_neighbor/
                # neighbor_noise ablation과 동일하게 고정 mini-batch로
                # 잘라서 순차 forward — 예측값은 배치 분할과 무관하게 동일.
                _predict_batch = 256
                x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                logits_chunks = []
                with torch.no_grad():
                    for start in range(0, x_t.shape[0], _predict_batch):
                        chunk = x_t[start:start + _predict_batch]
                        logits_chunks.append(model(chunk)["logits"].cpu())
                logits_np = torch.cat(logits_chunks, dim=0).numpy()
                if tasktype == "multiclass":
                    exp_l = np.exp(logits_np - logits_np.max(-1, keepdims=True))
                    return exp_l / exp_l.sum(-1, keepdims=True)
                elif tasktype == "binclass":
                    return 1 / (1 + np.exp(-logits_np))
                else:
                    return logits_np

            def _run_shap_once(bg_rng: np.random.RandomState):
                """SHAP 1회 실행 → (shap_arr, shap_mean, shap_rank)."""
                bg_n        = min(args.shap_background, len(X_train_np))
                bg_idx      = bg_rng.choice(len(X_train_np), size=bg_n, replace=False)
                bg_data     = X_train_np[bg_idx]
                explainer   = shap.KernelExplainer(model_predict, bg_data)
                shap_values = explainer.shap_values(X_rc_np, nsamples=_shap_nsamples, silent=True)

                if isinstance(shap_values, list):
                    arrays = [np.abs(np.array(sv, dtype=float)) for sv in shap_values]
                    valid = [a for a in arrays if a.ndim == 2 and a.shape[1] == n_features]
                    if valid and _target_class is not None:
                        n_valid = len(valid)
                        shap_arr_ = np.stack([
                            valid[min(int(_target_class[i]), n_valid - 1)][i]
                            for i in range(n_rc)
                        ])                                          # (N, F)
                    elif valid:
                        shap_arr_ = np.mean(valid, axis=0)           # (N, F)
                    else:
                        shap_arr_ = arrays[0]
                else:
                    shap_values = np.array(shap_values, dtype=float)
                    if shap_values.ndim == 3:
                        shape3 = shap_values.shape
                        sample_axis, feat_axis = None, None
                        for ax, sz in enumerate(shape3):
                            if sz == n_rc and sample_axis is None:
                                sample_axis = ax
                        for ax, sz in enumerate(shape3):
                            if ax != sample_axis and sz == n_features and feat_axis is None:
                                feat_axis = ax

                        if sample_axis is not None and feat_axis is not None:
                            class_axis = [a for a in range(3) if a not in (sample_axis, feat_axis)][0]
                            shap_moved = np.moveaxis(shap_values, [sample_axis, feat_axis, class_axis], [0, 1, 2])
                            if _target_class is not None:
                                shap_arr_ = np.abs(np.stack([
                                    shap_moved[i, :, int(_target_class[i])] for i in range(n_rc)
                                ]))                                       # (N, F)
                            else:
                                shap_arr_ = np.abs(shap_moved).mean(axis=2)  # (N, F)
                        else:
                            shap_arr_ = np.abs(shap_values).mean(axis=-1)
                            if shap_arr_.shape[0] != n_rc:
                                shap_arr_ = shap_arr_.T
                    else:
                        shap_arr_ = np.abs(shap_values)             # (N, F)

                assert shap_arr_.shape[0] == n_rc, (
                    f"shap_arr의 첫 축이 샘플 수(n_rc={n_rc})와 안 맞습니다: "
                    f"shap_arr.shape={shap_arr_.shape}. shap_values의 반환 형태가 "
                    f"예상과 다를 수 있습니다 (shap 버전 확인 필요)."
                )
                shap_mean_raw_ = np.array(shap_arr_.mean(axis=0), dtype=float)
                if shap_mean_raw_.shape[0] != n_features:
                    shap_mean_raw_ = shap_arr_.mean(axis=0)
                    if shap_mean_raw_.ndim > 1:
                        shap_mean_raw_ = shap_mean_raw_.mean(axis=-1)
                    shap_mean_raw_ = shap_mean_raw_[:n_features]
                shap_mean_ = np.array(shap_mean_raw_, dtype=float).flatten()[:n_features]
                assert shap_mean_.shape[0] == n_features, f"shap_mean shape {shap_mean_.shape} != {n_features}"
                shap_rank_ = np.argsort(np.argsort(-shap_mean_)).astype(int)
                return shap_arr_, shap_mean_, shap_rank_

            shap_arr, shap_mean, shap_rank = _run_shap_once(np.random.RandomState(args.seed))

            shap_mc_std = None
            if args.shap_repeats > 1:
                print(f"  [SHAP MC 노이즈 진단] {args.shap_repeats}회 반복 재계산 중"
                      f"(매번 다른 background)...")
                # [해석 우선순위] 이 노이즈부터 확인해야 함 — corr_shap이 corr_rand와
                # 별 차이 없어 보여도, 그게 'SHAP이 Delta와 안 맞아서'인지 'SHAP 추정
                # 자체가 이 정도로 흔들려서'인지 이 진단 없이는 구분 불가능함.
                shap_mc_corrs = [spearmanr(shap_rank, delta_rank)[0]]
                for _r in range(1, args.shap_repeats):
                    _, _, shap_rank_r = _run_shap_once(np.random.RandomState(args.seed * 1000 + _r))
                    corr_r, _ = spearmanr(shap_rank_r, delta_rank)
                    shap_mc_corrs.append(corr_r)
                shap_mc_corrs = np.array(shap_mc_corrs)
                shap_mc_std = float(shap_mc_corrs.std())
                print(f"    corr_shap (반복 {args.shap_repeats}회): "
                      f"{shap_mc_corrs.mean():.4f} ± {shap_mc_std:.4f}  "
                      f"(min={shap_mc_corrs.min():.4f}, max={shap_mc_corrs.max():.4f})")
                if shap_mc_std > 0.02:
                    print(f"    ⚠️  SHAP 자체 노이즈(±{shap_mc_std:.4f})가 꽤 큽니다 — "
                          f"아래 bootstrap CI 폭의 일부는 샘플 선택이 아니라 이 노이즈")
                    print(f"       때문일 수 있습니다. --shap_nsamples/--shap_background를 "
                          f"늘리는 걸 고려하세요.")

            print(f"  [3/3] Random attribution baseline 계산 중 (1000회 반복)...")
            rng_rc = np.random.RandomState(args.seed)
            n_rand_draws = 1000
            rand_corrs = np.empty(n_rand_draws)
            for r in range(n_rand_draws):
                rand_mean_r = rng_rc.rand(n_features)
                rand_rank_r = np.argsort(np.argsort(-rand_mean_r))
                rand_corrs[r], _ = spearmanr(rand_rank_r, delta_rank)

            corr_rand      = float(rand_corrs.mean())
            corr_rand_std  = float(rand_corrs.std())

            delta_rank = np.array(delta_rank,  dtype=int)
            shap_rank  = np.array(shap_rank,   dtype=int)

            corr_shap, p_shap = spearmanr(shap_rank, delta_rank)
            p_shap_vs_null     = float((rand_corrs >= corr_shap).mean())

            print(f"\n  {'─'*60}")
            print(f"  {'Method':<20} {'Spearman ρ':>12}  {'p-value':>12}")
            print(f"  {'─'*60}")
            print(f"  {'SHAP (③)':<20} {corr_shap:>12.4f}  {p_shap:>12.4f}")
            print(f"  {'Random (1000회)':<20} {_fmt_signed(corr_rand):>12}  {'±' + f'{corr_rand_std:.4f}':>12}")
            print(f"  {'─'*60}")
            print(f"  랜덤 귀무분포 대비 경험적 p-value:")
            print(f"    P(random ρ ≥ SHAP ρ) = {_fmt_pval(p_shap_vs_null, n_rand_draws)}")

            print(f"\n  [Bootstrap] SHAP-Delta 정합도 안정성 검정 (200회 재표본추출)...")
            n_boot = 200
            rng_boot = np.random.RandomState(args.seed + 1)
            boot_corrs = np.empty(n_boot)
            for b in range(n_boot):
                idx_b = rng_boot.randint(0, n_rc, size=n_rc)  # 복원추출
                delta_b = delta_samples[idx_b].mean(axis=0)
                shap_b  = shap_arr[idx_b].mean(axis=0)

                delta_rank_b = np.argsort(np.argsort(-delta_b))
                shap_rank_b  = np.argsort(np.argsort(-shap_b))
                boot_corrs[b], _ = spearmanr(shap_rank_b, delta_rank_b)

            boot_ci_low, boot_ci_high = np.percentile(boot_corrs, [2.5, 97.5])

            print(f"    corr_shap 재표본 분포: mean={boot_corrs.mean():+.4f}  "
                  f"(95% CI: [{boot_ci_low:+.4f}, {boot_ci_high:+.4f}])")
            if boot_ci_low > corr_rand + 2 * corr_rand_std:
                print(f"    → CI가 random 수준을 안정적으로 넘음: SHAP 순위가 Delta와")
                print(f"      우연 이상으로 정합함")
            else:
                print(f"    → CI가 random 수준과 겹칠 수 있음: 이 데이터셋에서 SHAP-Delta")
                print(f"      정합도를 '우연보다 유의하게 낫다'고 단정하기엔 이름")

            print(f"\n  [Delta 상위 5개 feature — SHAP 순위 비교]")
            top5_delta = np.argsort(delta_arr)[::-1][:5]
            print(f"  {'Feature':<25} {'Delta순위':>8}  {'SHAP순위':>8}")
            print(f"  {'─'*45}")
            for fi in top5_delta:
                fn = col_names[fi] if fi < len(col_names) else f"f{fi}"
                print(
                    f"  {fn:<25} "
                    f"  #{int(delta_rank[fi])+1:>4}    "
                    f"  #{int(shap_rank[fi])+1:>4}"
                )

            # [추가] 위 표는 Delta 기준 상위만 보여줘서, "SHAP은 상위로 보는데
            # Delta는 안 중요하게 보는" 반대 방향 불일치는 사각지대였음(예:
            # 순수 상호작용으로만 작동해서 Delta 개별-perturb로는 안 잡히는
            # feature). SHAP 상위 5개 중 위 표에 이미 나온 feature는 빼고
            # 마저 보여줌 — 두 표를 합치면 양방향 불일치를 다 볼 수 있음.
            top5_delta_set = set(int(fi) for fi in top5_delta)
            top_shap_sorted = np.argsort(shap_mean)[::-1]
            top5_shap_only = [fi for fi in top_shap_sorted if int(fi) not in top5_delta_set][:5]
            if top5_shap_only:
                print(f"\n  [SHAP 상위 중 위 표에 없던 feature — Delta 순위 비교]")
                print(f"  {'Feature':<25} {'SHAP순위':>8}  {'Delta순위':>8}")
                print(f"  {'─'*45}")
                for fi in top5_shap_only:
                    fi = int(fi)
                    fn = col_names[fi] if fi < len(col_names) else f"f{fi}"
                    print(
                        f"  {fn:<25} "
                        f"  #{int(shap_rank[fi])+1:>4}    "
                        f"  #{int(delta_rank[fi])+1:>4}"
                    )

            print(f"\n  [해석]")
            print(f"  SHAP-Delta Spearman ρ={corr_shap:.3f} (random 기준 {_fmt_signed(corr_rand, 3)}±{corr_rand_std:.3f})")
            if p_shap_vs_null < 0.05:
                print(f"  → 우연(random)보다 유의하게 나은 정합도 (p={_fmt_pval(p_shap_vs_null, n_rand_draws)}).")
                print(f"    다만 이는 'SHAP이 정확하다'는 증명이 아니라, SHAP 순위가")
                print(f"    단순 1차 신호(Delta)와도 어느 정도 통하는 합리적인 순위라는")
                print(f"    정도의 정합성 체크임.")
            else:
                print(f"  ⚠️  우연(random)과 유의하게 다르다고 말하기 어려움 (p={_fmt_pval(p_shap_vs_null, n_rand_draws)}).")
                print(f"    SHAP이 틀렸다는 뜻일 수도 있지만, (a) SHAP MC 노이즈가 크거나")
                print(f"    (b) 이 데이터셋에 Delta로는 못 보는 상호작용이 많아서일 수도")
                print(f"    있음 — --shap_repeats로 (a)를, --ablation interaction_check로")
                print(f"    (b)를 먼저 배제한 뒤 재해석할 것.")
            print(f"     + explanation이 prediction path 안에 있다는 구조적 차별성(①②)은")
            print(f"       이 ρ 값과 무관하게 항상 성립함 — ③(SHAP)은 그 옆의 보조 장치.")

            rc_save = {
                "corr_shap":         corr_shap,
                "corr_random_mean":  corr_rand,
                "corr_random_std":   corr_rand_std,
                "p_shap":            p_shap,
                "p_shap_vs_null":    p_shap_vs_null,
                "boot_corr_mean":    float(boot_corrs.mean()),
                "boot_corr_ci":      [float(boot_ci_low), float(boot_ci_high)],
                "shap_mc_std":       shap_mc_std,
                "delta_arr":    delta_arr.tolist(),
                "shap_mean":    shap_mean.tolist(),
                "col_names":    col_names,
                "openml_id":    openml_id,
                "seed":         args.seed,
            }
            rc_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_rank_correlation.pkl"
            )
            with open(rc_path, "wb") as f:
                pickle.dump(rc_save, f)
            print(f"\n  저장: {rc_path}")

        # ── interaction_check: feature 상호작용이 실제로 존재하는지 직접 확인 ──
        # (SHAP-Delta 불일치가 '상호작용 때문'이라는 주장을 뒷받침하려면, 그 전에
        # 데이터에 상호작용이 실제로 있는지부터 데이터로 확인해야 함. 여기서는
        # interaction(i,j) = |perturb(i,j 동시)| - [|perturb(i)| + |perturb(j)|] 로 정의—
        # 0보다 유의하게 크면 i,j가 예측에 super-additive하게 같이 작동한다는 뜻.)
        elif args.ablation == "interaction_check":
            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            n_ic = min(100, X_test.shape[0])
            _ic_perm = np.random.RandomState(args.seed).permutation(X_test.shape[0])[:n_ic]
            X_ic = X_test[_ic_perm]

            print(f"\n  Feature Interaction Check (n={n_ic})")
            print(f"  {'─'*60}")
            print(f"  '두 feature를 동시에 perturb했을 때의 변화'와 '개별 perturb 변화의")
            print(f"  합' 사이의 차이로, 상호작용이 실제로 존재하는지 먼저 확인합니다.")
            print(f"  (SHAP interaction values가 아니라, 모델 구조에 안 얽매이는 직접")
            print(f"  perturbation 방식 — TabERA처럼 hard-routing 등 불연속을 가진")
            print(f"  구조에도 안전하게 적용됨.)")

            with torch.no_grad():
                logits_orig = model(X_ic)["logits"]
                _target_class = (
                    logits_orig.argmax(dim=-1).cpu().numpy()
                    if tasktype == "multiclass" else None
                )

            def _pick_target_ic(logits: torch.Tensor) -> torch.Tensor:
                if tasktype == "multiclass":
                    idx = torch.as_tensor(_target_class, device=logits.device, dtype=torch.long)
                    return logits[torch.arange(logits.shape[0], device=logits.device), idx]
                return logits.squeeze(-1)

            with torch.no_grad():
                train_mean  = X_train.mean(dim=0)
                orig_target = _pick_target_ic(logits_orig)     # (N,)

                print(f"  [1/3] 개별 Delta 계산 중 (feature {n_features}개)...")
                delta_1d = np.zeros((n_ic, n_features))
                for f in range(n_features):
                    X_m = X_ic.clone()
                    X_m[:, f] = train_mean[f]
                    delta_1d[:, f] = (orig_target - _pick_target_ic(model(X_m)["logits"])).abs().cpu().numpy()

            # 상위 K개 Delta-important feature 쌍만 확인 (O(K^2)로 비용 통제)
            top_k = min(12, n_features)
            top_feats = np.argsort(-delta_1d.mean(axis=0))[:top_k]
            n_pairs = top_k * (top_k - 1) // 2
            print(f"  [2/3] 상위 {top_k}개 feature(O(K^2)={n_pairs}쌍)에 대해 "
                  f"쌍별 상호작용 계산 중...")

            with torch.no_grad():
                pair_interactions = []   # [(i, j, mean_abs_interaction), ...]
                for a in range(top_k):
                    for b in range(a + 1, top_k):
                        fi, fj = int(top_feats[a]), int(top_feats[b])
                        X_pair = X_ic.clone()
                        X_pair[:, fi] = train_mean[fi]
                        X_pair[:, fj] = train_mean[fj]
                        delta_pair = (orig_target - _pick_target_ic(model(X_pair)["logits"])).abs().cpu().numpy()
                        # super-additive면 양수, sub-additive(중복 신호)면 음수
                        interaction = delta_pair - (delta_1d[:, fi] + delta_1d[:, fj])
                        pair_interactions.append((fi, fj, float(np.abs(interaction).mean()),
                                                   float(interaction.mean())))

            pair_interactions.sort(key=lambda t: -t[2])

            print(f"  [3/3] Random 쌍 대조군 계산 중 (동일 개수, 무작위 feature 쌍)...")
            rng_ic = np.random.RandomState(args.seed)
            rand_abs_interactions = []
            with torch.no_grad():
                for _ in range(n_pairs):
                    fi, fj = rng_ic.choice(n_features, size=2, replace=False)
                    X_pair = X_ic.clone()
                    X_pair[:, int(fi)] = train_mean[int(fi)]
                    X_pair[:, int(fj)] = train_mean[int(fj)]
                    delta_pair = (orig_target - _pick_target_ic(model(X_pair)["logits"])).abs().cpu().numpy()
                    interaction = delta_pair - (delta_1d[:, int(fi)] + delta_1d[:, int(fj)])
                    rand_abs_interactions.append(float(np.abs(interaction).mean()))

            top_abs_mean  = float(np.mean([t[2] for t in pair_interactions]))
            rand_abs_mean = float(np.mean(rand_abs_interactions))

            # [통계적 엄밀성 추가] 기존엔 "top_abs_mean > rand_abs_mean * 1.5"라는
            # 임의 배수 임계값으로만 판단했음 — rank_correlation처럼 경험적 null
            # 분포와 p-value로 바꾼다. 다만 random 쌍 자체를 1000번 다시 뽑아
            # model forward를 또 도는 건 비용이 n_pairs배로 늘어나므로, 이미
            # 계산해둔 rand_abs_interactions 풀(n_pairs개, 실제 model forward로
            # 얻은 값)에서 크기 n_pairs로 복원추출(bootstrap)해 "무작위 K쌍의
            # 평균 |상호작용|"의 null 분포를 근사한다 — rank_correlation의
            # bootstrap 재표본추출과 같은 원칙(이미 계산된 데이터를 재표본추출해
            # 추가 forward 비용 없이 분포를 얻음).
            print(f"  [Null 분포] random 쌍 풀에서 1000회 bootstrap 재표본추출 중...")
            rand_pool = np.array(rand_abs_interactions)
            rng_null  = np.random.RandomState(args.seed + 1)
            n_null_draws = 1000
            null_means = np.empty(n_null_draws)
            for r in range(n_null_draws):
                sample = rng_null.choice(rand_pool, size=len(rand_pool), replace=True)
                null_means[r] = sample.mean()

            null_mean = float(null_means.mean())
            null_std  = float(null_means.std())
            p_vs_null = float((null_means >= top_abs_mean).mean())

            print(f"\n  {'─'*60}")
            print(f"  Delta-important 상위 {top_k}개 쌍의 |상호작용| 평균: {top_abs_mean:.4f}")
            print(f"  무작위 feature 쌍의 |상호작용| 평균:              {rand_abs_mean:.4f}")
            print(f"  Random null 분포 (bootstrap 1000회):              {null_mean:.4f} ± {null_std:.4f}")
            print(f"  P(random null 평균 ≥ top_abs_mean) = {_fmt_pval(p_vs_null, n_null_draws)}")
            print(f"  {'─'*60}")

            print(f"\n  [상위 상호작용 5쌍]")
            print(f"  {'Feature i':<20} {'Feature j':<20} {'|interaction|':>14} {'부호':>6}")
            print(f"  {'─'*64}")
            for fi, fj, abs_int, signed_int in pair_interactions[:5]:
                ni = col_names[fi] if fi < len(col_names) else f"f{fi}"
                nj = col_names[fj] if fj < len(col_names) else f"f{fj}"
                sign = "super+" if signed_int > 0 else "sub-"
                print(f"  {ni:<20} {nj:<20} {abs_int:>14.4f} {sign:>6}")

            print(f"\n  [해석]")
            if p_vs_null < 0.05:
                print(f"  ✅ Delta-important feature 쌍에서 상호작용이 무작위 null(p={_fmt_pval(p_vs_null, n_null_draws)})")
                print(f"    보다 유의하게 큼 ({top_abs_mean:.4f} vs null {null_mean:.4f}±{null_std:.4f}) —")
                print(f"    이 데이터셋에는 SHAP이 잡아낼 가치가 있는 feature 상호작용이")
                print(f"    실제로 존재함. rank_correlation에서 SHAP-Delta가 불일치했다면,")
                print(f"    상호작용 반영 때문일 가능성을 무게 있게 고려할 수 있음.")
            else:
                print(f"  ⚠️  Delta-important 쌍의 상호작용이 무작위 null과 유의하게 다르다고")
                print(f"    말하기 어려움 (p={_fmt_pval(p_vs_null, n_null_draws)}, {top_abs_mean:.4f} vs null "
                      f"{null_mean:.4f}±{null_std:.4f}). rank_correlation에서 SHAP-Delta 불일치가")
                print(f"    나온다면, 상호작용보다는 SHAP 추정 자체의 노이즈(--shap_repeats로")
                print(f"    확인)일 가능성이 더 큼.")
            print(f"     (주의: 이 null은 random pair 풀 {n_pairs}개의 재표본추출로 근사한 것 —")
            print(f"      pool 자체가 작으면(top_k가 작은 데이터셋) null 분포도 거칠어질 수 있음.)")

            ic_save = {
                "top_feats":            [int(f) for f in top_feats],
                "pair_interactions":    pair_interactions,
                "top_abs_mean":         top_abs_mean,
                "rand_abs_mean":        rand_abs_mean,
                "null_mean":            null_mean,
                "null_std":             null_std,
                "p_vs_null":            p_vs_null,
                "col_names":            col_names,
                "openml_id":            openml_id,
                "seed":                 args.seed,
            }
            ic_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_interaction_check.pkl"
            )
            with open(ic_path, "wb") as f:
                pickle.dump(ic_save, f)
            print(f"\n  저장: {ic_path}")

        elif args.ablation == "dual_space_faithfulness":
            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features
            n_val      = min(512, X_test.shape[0])
            X_val_sub  = X_test[:n_val]

            print(f"\n  Dual-Space Faithfulness Analysis")
            print(f"  {'─'*58}")

            with torch.no_grad():
                out_ds        = model(X_val_sub)
                hard_assign   = out_ds["hard_group"].cpu()
                evidence_w_ds = out_ds.get("evidence_w")
                topk_idx_ds   = out_ds.get("topk_idx")

            sample_groups = model.prototype_layer.sample_groups
            X_val_cpu     = X_val_sub.detach().cpu()

            n_mem = model.memory.filled.item()
            ref_emb = model.memory.keys[:n_mem].detach().cpu()          # (n_mem, D)
            ref_raw = (
                model.feature_store._store[:n_mem].detach().cpu()
                if model.feature_store is not None else None
            )                                                            # (n_mem, F)

            cat_cols = list(dataset.X_cat)
            num_cols = list(dataset.X_num)

            valid_p_all = [p for p in range(model.prototype_layer.P)
                           if sample_groups and len(sample_groups[p]) >= 2]
            valid_p = valid_p_all
            if ref_raw is None:
                print(f"    ⚠️  model.feature_store가 없어 원본 feature 공간 비교를 할 수 없습니다 —")
                print(f"       검증 2를 건너뜁니다 (인덱스 정합성 확인만 아래에서 진행).")
                valid_p = []

            centroid_emb_cpu = model.prototype_layer.centroid_emb.detach().cpu()  # (P, D)

            with torch.no_grad():
                q_check = F.normalize(ref_emb, dim=-1)
                c_check = F.normalize(centroid_emb_cpu, dim=-1)
                assign_check = (q_check @ c_check.T).argmax(dim=-1).numpy()  # (n_mem,)

            match_count, total_count = 0, 0
            for p in valid_p_all:
                grp = sample_groups[p]
                total_count += len(grp)
                match_count += int((assign_check[grp] == p).sum())
            chance_rate = 1.0 / model.prototype_layer.P

            print(f"  [사전 검증] sample_groups 인덱스 정합성 확인 (MemoryBank 슬롯 기준)")
            if total_count == 0:
                print(f"    ⚠️  검증 가능한 그룹(크기≥2)이 없어 일치율을 계산할 수 없습니다.")
                index_ok = False
            else:
                match_rate = match_count / total_count
                print(f"    재배정 일치율: {match_rate:.1%}  (무작위 기대치: {chance_rate:.1%})")
                index_ok = match_rate >= 0.99
                if not index_ok:
                    print(f"    ❌ 검증 시점에는 추가 학습이 없어 EMA 지연으로 설명될 수 없습니다 —")
                    print(f"       sample_groups가 가리키는 소스(MemoryBank/FeatureStore)와 지금")
                    print(f"       비교에 쓴 소스가 여전히 어긋나 있을 가능성이 높습니다.")
                    print(f"       아래 검증 2 결과는 재확인 전까지 신뢰할 수 없습니다.")
                else:
                    print(f"    ✅ 인덱스 정합성 확인됨 (MemoryBank 슬롯 기준) — 아래 결과를 신뢰할 수 있습니다.")

            if not index_ok:
                valid_p = []

            print(f"\n  [검증 2] Between-Group Feature Separation")
            print(f"  (numeric: One-way ANOVA F-test / categorical: Chi-square 독립성 검정)")

            if valid_p:
                from scipy.stats import f as f_dist, chi2_contingency

                group_sizes = np.array([len(sample_groups[p]) for p in valid_p])
                P_valid     = len(valid_p)

                stat_arr  = np.full(n_features, np.nan)
                p_arr     = np.full(n_features, np.nan)
                test_type = np.array(["-"] * n_features, dtype=object)

                if num_cols:
                    group_means_num = np.array([
                        ref_raw[sample_groups[p]].numpy()[:, num_cols].mean(axis=0)
                        for p in valid_p
                    ])                                                    # (P_valid, F_num)
                    ss_within = np.zeros(len(num_cols))
                    total_n   = 0
                    for p in valid_p:
                        grp_data = ref_raw[sample_groups[p]].numpy()[:, num_cols]
                        grp_mean = grp_data.mean(axis=0)
                        ss_within += ((grp_data - grp_mean) ** 2).sum(axis=0)
                        total_n   += grp_data.shape[0]
                    df_within = max(total_n - P_valid, 1)
                    msw       = ss_within / df_within

                    grand_mean = np.average(group_means_num, axis=0, weights=group_sizes)
                    ssb        = np.sum(group_sizes[:, None] * (group_means_num - grand_mean) ** 2, axis=0)
                    df_between = max(P_valid - 1, 1)
                    msb        = ssb / df_between

                    F_stat_num = msb / (msw + 1e-8)
                    p_num      = f_dist.sf(F_stat_num, df_between, df_within)

                    for j, fi in enumerate(num_cols):
                        stat_arr[fi]  = F_stat_num[j]
                        p_arr[fi]     = p_num[j]
                        test_type[fi] = "F"

                if cat_cols:
                    for fi in cat_cols:
                        cats_per_group = [
                            np.rint(ref_raw[sample_groups[p]].numpy()[:, fi]).astype(int)
                            for p in valid_p
                        ]
                        all_cats = np.unique(np.concatenate(cats_per_group))
                        table = np.zeros((P_valid, len(all_cats)), dtype=int)
                        for gi, vals in enumerate(cats_per_group):
                            for c in vals:
                                table[gi, np.searchsorted(all_cats, c)] += 1
                        if table.shape[1] >= 2 and (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
                            try:
                                chi2, p, dof, _ = chi2_contingency(table)
                                stat_arr[fi]  = chi2
                                p_arr[fi]     = p
                                test_type[fi] = "χ²"
                            except ValueError:
                                pass   # 검정 불가(예: 기대빈도 문제) → NaN 유지

                valid_mask       = ~np.isnan(p_arr)
                bonferroni_alpha = 0.05 / n_features   # 다중비교 보정 (전체 feature 수 기준)
                n_significant    = int((p_arr[valid_mask] < bonferroni_alpha).sum())

                neglogp = np.full(n_features, -1.0)
                neglogp[valid_mask] = -np.log10(np.clip(p_arr[valid_mask], 1e-300, 1.0))
                top_sep_idx = np.argsort(neglogp)[::-1][:5]

                print(f"  (유효 그룹 {P_valid}개 / numeric {len(num_cols)}개 F-test / "
                      f"categorical {len(cat_cols)}개 χ²-test, 검정 가능 {int(valid_mask.sum())}/{n_features})")
                print(f"  {'Feature':<20} {'Test':>6} {'Stat':>10}  {'p-value':>12}")
                print(f"  {'─'*52}")
                for fi in top_sep_idx:
                    fname = col_names[fi] if fi < len(col_names) else f"f{fi}"
                    if np.isnan(p_arr[fi]):
                        print(f"  {fname:<20} {'-':>6} {'(검정 불가)':>21}")
                        continue
                    sig_mark = "*" if p_arr[fi] < bonferroni_alpha else " "
                    print(f"  {fname:<20} {test_type[fi]:>6} {stat_arr[fi]:>10.3f}  {p_arr[fi]:>10.4f}{sig_mark}")

                print(f"\n  Bonferroni 보정(α={bonferroni_alpha:.2e}) 후 유의한 feature 수: "
                      f"{n_significant}/{n_features}")
                if n_significant > 0:
                    print(f"  → centroid가 최소 {n_significant}개 feature에서 통계적으로 "
                          f"유의하게 그룹을 구분함")
                else:
                    print(f"  ⚠️  다중비교 보정 후 유의한 feature가 하나도 없음 — "
                          f"'이 그룹은 X, Y 특성이 다르다'는 설명의 통계적 근거가 약함")

                F_stat, p_values = stat_arr, p_arr   # 저장용 변수명 유지(하위 호환)
            else:
                F_stat, p_values, test_type = None, None, None

            dsf_save = {
                "anova_F_stat":     F_stat.tolist() if F_stat is not None else None,
                "anova_p_values":   p_values.tolist() if p_values is not None else None,
                "anova_test_type":  test_type.tolist() if test_type is not None else None,
                "openml_id":       openml_id,
                "seed":            args.seed,
            }
            dsf_path = (
                Path(log_dir)
                / f"data={openml_id}{_save_tag}..seed{args.seed}_dual_space_faithfulness.pkl"
            )
            with open(dsf_path, "wb") as f:
                pickle.dump(dsf_save, f)
            print(f"\n  저장: {dsf_path}")

        # ── dataset_profile: 예측 확신도/fallback 비율 빠른 진단 ──
        elif args.ablation == "dataset_profile":
            model.eval()
            n_test = min(100, X_test.shape[0])
            X_dp   = X_test[:n_test].clone()

            print(f"\n  Dataset Profile — 빠른 진단 (n={n_test})")
            print(f"  {'='*70}")
            # [변경 이력] 이전엔 여기서 IG의 mean/medoid baseline completeness
            # error, deletion_auc 샘플별 분산까지 계산해 A/B/C로 자동 분류했음.
            # ③이 SHAP으로 통일되면서 그 진단들은 의미가 없어져 제거함 —
            # SHAP의 faithfulness/노이즈 진단은 --ablation rank_correlation
            # (특히 --shap_repeats)이 대신 담당한다. 여기 남은 두 진단(예측
            # 확신도, fallback 비율)은 ③과 무관하게 여전히 유효한 정보라 유지.

            with torch.no_grad():
                logits_dp = model(X_dp)["logits"]
                if tasktype == "regression":
                    max_prob_dp = None
                elif tasktype == "multiclass":
                    probs_dp = torch.softmax(logits_dp, dim=-1)
                    max_prob_dp = probs_dp.max(dim=-1).values.cpu().numpy()
                else:
                    probs_dp = torch.sigmoid(logits_dp.squeeze(-1))
                    max_prob_dp = torch.where(probs_dp >= 0.5, probs_dp, 1 - probs_dp).cpu().numpy()

            print(f"\n  [1. 예측 확신도]")
            if max_prob_dp is not None:
                print(f"    mean={max_prob_dp.mean():.4f}  median={np.median(max_prob_dp):.4f}  "
                      f"std={max_prob_dp.std():.4f}")
                if np.median(max_prob_dp) > 0.9:
                    print(f"    ⚠️  median > 0.9 — overconfident, perturbation 기반 신호(Delta/SHAP) "
                          f"둔감 위험 (rank_correlation 해석 시 참고)")

            cached_sizes_dp = getattr(model.memory, "_cached_group_sizes", None)
            print(f"\n  [2. Fallback 비율]")
            if cached_sizes_dp is not None:
                with torch.no_grad():
                    q_dp = F.normalize(model.embedder(X_dp), dim=-1)
                    c_dp = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    ha_dp = (q_dp @ c_dp.T).argmax(dim=-1)
                    grp_sizes_dp = cached_sizes_dp[ha_dp]
                    fallback_rate_dp = (grp_sizes_dp < model.k).float().mean().item()
                    avg_group_size_dp = cached_sizes_dp[cached_sizes_dp > 0].float().mean().item()
                print(f"    k={model.k}, 평균 alive 그룹 크기={avg_group_size_dp:.1f}, "
                      f"fallback 비율={fallback_rate_dp*100:.1f}%")
                if model.k > avg_group_size_dp:
                    print(f"    ⚠️  k({model.k}) > 평균 그룹 크기({avg_group_size_dp:.1f}) "
                          f"— cross-group fallback이 상시 발동할 가능성 높음 (설명②의 "
                          f"'group-constrained' 클레임이 이 설정에서는 약화될 수 있음)")
            else:
                print(f"    _cached_group_sizes 없음 — skip")

        # ── random_neighbor / neighbor_noise: 성능 비교 ─────────────
        else:
            with torch.no_grad():
                abl_logits_list, abl_labels_list = [], []
                full_evw_list, abl_evw_list = [], []
                batch_size = 256
                n_test     = X_test.shape[0]

                for start in range(0, n_test, batch_size):
                    X_batch = X_test[start:start + batch_size]
                    out_batch      = model(X_batch, ablation_mode=args.ablation)
                    out_batch_full = model(X_batch, ablation_mode="none")
                    abl_logits_list.append(out_batch["logits"].cpu())
                    if out_batch.get("evidence_w") is not None:
                        abl_evw_list.append(out_batch["evidence_w"].cpu())
                    if out_batch_full.get("evidence_w") is not None:
                        full_evw_list.append(out_batch_full["evidence_w"].cpu())

                abl_logits = torch.cat(abl_logits_list, dim=0)
                abl_evw    = torch.cat(abl_evw_list, dim=0) if abl_evw_list else None
                full_evw   = torch.cat(full_evw_list, dim=0) if full_evw_list else None

            if tasktype == "regression":
                abl_preds   = abl_logits.squeeze(-1).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy() * y_std,
                    abl_preds * y_std,
                    None, tasktype, "test"
                )
                abl_probs = None
            elif tasktype == "multiclass":
                abl_preds   = abl_logits.argmax(-1).numpy()
                abl_probs   = torch.softmax(abl_logits, dim=-1).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy(), abl_preds, abl_probs, tasktype, "test"
                )
            else:  # binary
                abl_preds   = (abl_logits.squeeze(-1) > 0).long().numpy()
                abl_probs   = torch.sigmoid(abl_logits.squeeze(-1)).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy(), abl_preds, abl_probs, tasktype, "test"
                )

            print(f"\n  {'Metric':<20} {'Full Model':>12}  {'Ablation':>12}  {'Δ':>10}")
            print(f"  {'-'*58}")
            for k_name, v_full in test_metrics.items():
                v_abl = abl_metrics.get(k_name, float("nan"))
                delta = v_abl - v_full
                arrow = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "─")
                print(f"  {k_name:<20} {v_full:>12.4f}  {v_abl:>12.4f}  {delta:>+9.4f} {arrow}")

            evw_stats = {}
            if full_evw is not None and abl_evw is not None:
                k_dim = full_evw.shape[-1]

                def _norm_entropy(w):
                    ent = -(w * (w + 1e-8).log()).sum(dim=-1)   # (N,)
                    return (ent / torch.log(torch.tensor(float(k_dim)))).numpy()

                full_ent = _norm_entropy(full_evw)
                abl_ent  = _norm_entropy(abl_evw)
                full_max = full_evw.max(dim=-1).values.numpy()
                abl_max  = abl_evw.max(dim=-1).values.numpy()

                print(f"\n  evidence_w 엔트로피 (0=한 이웃에 완전 집중, 1=완전 uniform, k={k_dim})")
                print(f"  {'-'*58}")
                print(f"  {'':<20} {'Full Model':>12}  {'Ablation':>12}")
                print(f"  {'정규화 엔트로피 평균':<18} {full_ent.mean():>12.4f}  {abl_ent.mean():>12.4f}")
                print(f"  {'최대 가중치 평균':<18} {full_max.mean():>12.4f}  {abl_max.mean():>12.4f}")

                evw_stats = {
                    "full_entropy_mean": float(full_ent.mean()),
                    "abl_entropy_mean":  float(abl_ent.mean()),
                    "full_max_w_mean":   float(full_max.mean()),
                    "abl_max_w_mean":    float(abl_max.mean()),
                }

            print(f"\n  해석:")
            if args.ablation == "random_neighbor":
                print(f"  → 성능 하락 = '검색이 틀린 이웃을 찾았을 때'의 대가")
                print(f"    (이웃 정보 자체는 여전히 real data — retrieval 정확도의 가치)")
            elif args.ablation == "neighbor_noise":
                print(f"  → 성능 하락 = '이웃 정보가 조금이라도 있는가'의 대가")
                print(f"    (real이든 아니든 neighbor evidence 자체의 존재 가치)")
                print(f"  참고: random_neighbor보다 여기서 하락폭이 훨씬 커야 정상")
                print(f"    (같은 배치 크기지만 '틀린 진짜 이웃' < '이웃 자체 없음'이 더")
                print(f"    나쁜 상황이어야 두 ablation이 일관된 이야기를 함)")
                if evw_stats and evw_stats["abl_entropy_mean"] > evw_stats["full_entropy_mean"] + 0.1:
                    print(f"  → evidence_w가 실제로 uniform 쪽으로 이동함 "
                          f"(엔트로피 {evw_stats['full_entropy_mean']:.3f} → "
                          f"{evw_stats['abl_entropy_mean']:.3f}). nk가 노이즈가 되면서")
                    print(f"    attention이 '누구를 볼지 못 정하는' 상태가 됐다는 뜻 —")
                    print(f"    성능이 덜 떨어진 건 uniform 평균이 이 데이터셋에서")
                    print(f"    우연히 나쁘지 않은 예측이기 때문일 수 있음.")

            abl_save = {
                "ablation_mode":  args.ablation,
                "full_metrics":   test_metrics,
                "abl_metrics":    abl_metrics,
                "evidence_w_stats": evw_stats,
                "openml_id":      openml_id,
                "seed":           args.seed,
            }
            abl_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_ablation_{args.ablation}.pkl"
            with open(abl_path, "wb") as f:
                pickle.dump(abl_save, f)
            print(f"\n  저장: {abl_path}")



    # ── 결과 저장 ──────────────────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_meta.pkl"

    model.eval()
    with torch.no_grad():
        logits = model(X_test)["logits"].cpu().numpy()
    np.save(str(pred_path), logits)

    meta = {
        "openml_id":   openml_id,
        "tasktype":    tasktype,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "seed":        args.seed,
        "use_offset_correction": True,
        "global_retrieve": False,
        "use_context_emb": True,
        "detach_context_grad": args.detach_context_grad,
        "use_context_projection": args.context_projection,
        "cat_embedding": True,  # [후보 A] categorical nn.Embedding 적용 여부 기록
        "cat_combine": args.cat_combine,
        "cat_embed_dim": args.cat_embed_dim if args.cat_combine == "concat" else None,
        "num_embedding": args.num_embedding,
        "num_bins": args.num_bins if args.num_embedding == "ple" else None,
        "plr_n_frequencies": args.plr_n_frequencies if args.num_embedding == "plr_lite" else None,
        "plr_freq_scale": args.plr_freq_scale if args.num_embedding == "plr_lite" else None,
        "plr_out_dim": args.plr_out_dim if args.num_embedding == "plr_lite" else None,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  저장: {pred_path}")

    # ── model state 저장 (--from_saved_state 용) ──────────────
    # model_kwargs에 이미 use_offset_correction 등 아키텍처 플래그가
    # 다 병합돼 있음(위에서 model_kwargs.update()로 처리) — best_params
    # (Optuna 탐색 대상)에는 없는 값이라, 이게 없으면 --from_saved_state로
    # 복원할 때 기본값으로 되돌아가 버려 재현이 어긋남.
    #
    # [수정] state_dict()에 안 잡히는 것들(sample_groups/group_labels/
    # target_labels — plain Python 속성이라 buffer가 아님, feature_store
    # — nn.Module이 아니라 model.state_dict()에 안 잡힘)을 여기서도
    # 놓치고 있었음 — best-checkpoint 스냅샷 때(libs/supervised.py)와
    # 정확히 같은 문제. 이것들이 없으면 --from_saved_state로 복원해도
    # ①②가 제대로 안 나옴(특히 sample_groups 없으면 group-constrained
    # 검색 자체가 깨짐).
    state_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_model_state.pt"
    fs = model.feature_store
    torch.save({
        "state_dict":     model.state_dict(),
        "model_kwargs":   model_kwargs,
        "best_params":    best_params,
        "sample_groups":  model.prototype_layer.sample_groups,
        "group_labels":   model.prototype_layer.group_labels,
        "target_labels":  model.prototype_layer.target_labels,
        "feature_store_state": (
            (fs._store.detach().cpu(), fs._ptr, fs._filled) if fs is not None else None
        ),
        "col_names":    dataset.col_names,
        "n_train":      len(X_train),
        "tasktype":     tasktype,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "seed":         args.seed,
    }, str(state_path))
    print(f"  저장: {state_path}")

    # ── Feature 기여도 설명 출력 ─────────────────────────
    if args.explain:
        print(f"\n{'='*52}")
        print(f"  TabERA Explanations (--explain)")
        print(f"{'='*52}")

        model.eval()
        n_show = min(args.n_explain, len(y_test))
        X_show = X_test[:n_show]

        with torch.no_grad():
            out = model(X_show, return_explanations=True)

        explanations = out.get("explanations", [])

        topk_idx = out.get("topk_idx")
        if model.feature_store is not None and topk_idx is not None:
            cat_names = {dataset.col_names[i] for i in dataset.X_cat}
            X_show_cpu = X_show.detach().cpu().numpy()
            neighbour_feats = model.feature_store.retrieve(topk_idx)  # list[list[dict]]
            for b, exp in enumerate(explanations):
                if b < len(neighbour_feats):
                    query_dict = {name: float(X_show_cpu[b, i])
                                  for i, name in enumerate(dataset.col_names)}
                    exp["neighbour_features"] = [
                        _select_query_similar_features(query_dict, nd, cat_names)
                        for nd in neighbour_feats[b]
                    ]
        if not explanations:
            print("  (no explanations — memory bank has not been filled yet)")
            print("  → try increasing epochs or n_trials.")
        else:
            for i in range(n_show):
                print_explanation(explanations, i, dataset.col_names,
                                   cat_category_names=dataset.cat_category_names,
                                   quantile_transformer=dataset.quantile_transformer,
                                   num_cols=list(dataset.X_num))


if __name__ == "__main__":
    main()