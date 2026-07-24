## Main file for reproducing the best TabERA configuration.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import os, argparse, time

# ── CUDA_VISIBLE_DEVICES: torch import 전 설정 ──────────────
_parser_pre = argparse.ArgumentParser(add_help=False)
_parser_pre.add_argument("--gpu_id", type=int, default=0)
_parser_pre.add_argument("--deterministic", action="store_true")
_pre, _ = _parser_pre.parse_known_args()
if _pre.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.gpu_id)
if _pre.deterministic:
    # torch.use_deterministic_algorithms(True)가 CUDA >=10.2에서 일부 cuBLAS
    # 연산(예: 특정 matmul/conv 백워드)을 결정적으로 돌리려면 이 환경변수가
    # CUDA 컨텍스트 생성(=torch import 시점) *이전*에 설정돼 있어야 함 —
    # torch import 뒤에 os.environ으로 설정하면 이미 늦어서 조용히 무시됨.
    # 그래서 --gpu_id와 같은 자리(pre-parser)에서 처리.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from libs.data         import TabularDataset
from libs.search_space import params_to_model_kwargs, study_pkl_tag, HPO_TRAINING_SCHEDULE
from libs.supervised   import TabERAWrapper
from libs.tabera         import TabERA
from libs.prototypes     import inverse_transform_numeric
from libs.eval         import calculate_metric, get_preds_and_probs, get_criterion
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
                       quantile_transformer=None, num_cols: list = None,
                       pred_info: dict = None) -> None:
    e = explanations[sample_idx]

    print(f"\n{'━'*52}")
    print(f"  TabERA Explanation — Sample #{sample_idx}")
    print(f"{'━'*52}")

    # [추가] Prediction confidence(classifier softmax) — Routing confidence
    # (아래 ①)와 절대 같은 값이 아님을 처음부터 분리해서 보여줌.
    # query→routing→context→retrieval→fusion→classifier 파이프라인에서
    # classifier는 routing 외의 정보(retrieval evidence 등)도 다 쓰므로,
    # routing이 애매해도(confidence 낮음) 최종 예측은 확신할 수 있고 그
    # 반대도 가능함 — 이 둘을 한 화면에 나란히 보여줘서 혼동을 막는다.
    if pred_info is not None:
        print(f"\n  Prediction")
        print(f"     → {pred_info['pred_label']}")
        if pred_info.get("pred_confidence") is not None:
            print(f"     Prediction confidence: {pred_info['pred_confidence']:.1%}  "
                  f"(classifier output — separate from routing confidence below)")

    # ① Prototype routing (target distribution — which class does this group represent?)
    proto = e["prototype"]
    print(f"\n  ① Prototype Assignment")

    # 이 그룹의 target(클래스) 분포 — ①의 주 콘텐츠 (label_groups_by_target(),
    # regroup_update() 직후 캐싱됨). ②(실제 이웃의 raw feature 값)와 정보 종류가
    # 겹치지 않도록, feature 요약이 아니라 "이 그룹이 어떤 부류인가"만 보여준다.
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

    # [명명 정정] "confidence" 단독 표기는 이 값을 classifier의 예측 확신도로
    # 오해하게 만듦 — 실제로는 prototype routing 단계에서 "이 query가 다른
    # centroid 대비 배정된 centroid에 상대적으로 얼마나 우세한가"이고, 최종
    # 예측 확신도(Prediction confidence, 아래 call site에서 별도 출력)와는
    # 다른 값. margin/others/cosine을 같이 보여줘서 이 숫자 하나만으로 판단
    # 안 하고 맥락과 함께 읽게 함.
    print(f"     Assigned prototype: \"{proto['assigned_group']}\"")
    cos_str = f"  |  cosine similarity={proto['cosine_similarity']:.3f}" if proto.get('cosine_similarity') is not None else ""
    print(f"     Routing confidence: {proto['routing_confidence']:.1%}"
          f"  (relative preference among all prototypes, not a prediction probability)")
    print(f"     Margin over runner-up: {proto['margin']:+.1%}{cos_str}")
    print(f"     Prototype label distribution: {target_str}")

    if proto["runners_up"]:
        print(f"     Routing distribution:")
        print(f"       • {proto['assigned_group']:<20s} {proto['routing_confidence']:>6.1%}  (assigned)")
        for r in proto["runners_up"]:
            print(f"       • {r['label']:<20s} {r['routing_confidence']:>6.1%}  "
                  f"({_format_target_info(r['target_info'])})")
        print(f"       • {'Others':<20s} {proto['others_mass']:>6.1%}")

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
    # [명명 정정] "Neighbor Evidence"는 causal claim("이 이웃 때문에 예측했다")을
    # 함의함 — 이 세션의 necessity 검증(agg_emb 제거해도 accuracy 거의 불변,
    # 4데이터셋×5seed)에서 이 weight가 실제로 prediction을 좌우한다는 근거가
    # 부족함이 확인됨. "Retrieved Neighbors"(무엇을 검색했는가, descriptive)로
    # 표현을 낮추고, 그 아래 한 줄로 한계를 명시. context/agg branch 자체나
    # 검색 메커니즘은 그대로 유지 — retrieval inspection/error analysis
    # 용도로는 여전히 유효함.
    print(f"\n  ② Retrieved Neighbors (Similarity)")
    print(f"     (attention weight — 실제 예측 결정과의 인과관계는 검증되지 않음)")
    print(f"     dominant={ev['dominant_weight']:.1%},  entropy={ev['entropy']:.3f}")

    # attention weight가 사실상 0인 이웃은 생략 (반올림하면 0.0%로 보이는 것도
    # 포함) — weight가 거의 없는 이웃까지 보여주는 건 정보가 아니라 소음이다.
    # ["기여도"라는 표현은 안 씀 — 위 causal claim 이슈와 같은 이유]
    _WEIGHT_EPS = 1e-3
    shown = [(rank, idx, w) for rank, (idx, w) in enumerate(ev["top_neighbours"])
              if w > _WEIGHT_EPS]

    if not shown:
        print(f"     (no neighbor received meaningful attention weight)")

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

    # Level 3: Retrieval signal magnitude — [추가]
    # "기여도(contribution)"라고 안 부름 — head가 비선형 함수(예: residual
    # 모드의 Head(q+βa))라 ‖βa‖가 prediction에 미치는 실제 영향과 정확히
    # 비례한다는 보장이 없음(위 ②의 "기여도" 명명 정정과 같은 이유).
    # 여기서 주는 건 순수 magnitude 정보 — causal attribution 아님.
    rs = e.get("retrieval_signal")
    if rs is not None:
        print(f"\n  Level 3 — Retrieval Signal Magnitude")
        print(f"     (representation 크기 비교 — prediction에 대한 causal 기여도가 아님)")
        beta_str = f"{rs['beta']:.4f}" if rs.get("beta") is not None else "N/A (fusion_mode != residual)"
        print(f"     ‖query_emb‖={rs['query_norm']:.3f}   ‖agg_emb‖={rs['agg_norm']:.3f}   β={beta_str}")

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
# Calibration 분석 — routing confidence vs prediction confidence
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Centroid 단위 통계 (size/purity/cohesion) — train set 기준
# ─────────────────────────────────────────────────────────────

def compute_centroid_train_stats(model, X_train, y_train, tasktype: str,
                                  class_names=None, batch_size: int = 256):
    """centroid별 size(그룹 크기)/purity(그룹 내 최다 target 비율)/
    cohesion(그룹 내 샘플들이 자기 centroid에 얼마나 모여있는지)를 train
    set 기준으로 계산.

    [배경] --ablation centroid_representativeness와 같은 계산(size/purity/
    gap/cohesion)이지만, 그건 출력 전용(print/pickle 저장)이라 다른 곳에서
    반환값을 재사용하기 어려움 — 그 로직을 여기 독립 함수로 다시 구현해서
    run_calibration_analysis()가 "큰 centroid일수록 순도가 낮은가"를
    correlation으로 직접 계산할 수 있게 함(사용자 가설:
    큰 centroid → 순도 낮음 → 예측 실패, 를 직접 검증하기 위함).
    기존 ablation 코드는 검증된 대로 그대로 두고 건드리지 않음 — 중복은
    있지만 회귀 위험을 줄이는 쪽을 택함.

    반환: {centroid_idx: {"size": int, "purity": float|None,
                           "cohesion": float, "gap": float|None}}
      purity/gap은 classification에서만(regression은 None) — gap은
      purity - 전역 baseline(최다 target 비율).
    """
    model.eval()
    P = model.prototype_layer.P
    sample_groups = model.prototype_layer.sample_groups
    target_labels = model.prototype_layer.target_labels
    if sample_groups is None:
        return {}

    y_train_np = y_train.detach().cpu().numpy()
    global_majority_prop = None
    if tasktype in ("multiclass", "binclass"):
        y_int = np.rint(y_train_np).astype(int)
        _, counts = np.unique(y_int, return_counts=True)
        global_majority_prop = float(counts.max() / counts.sum())

    with torch.no_grad():
        c_norm = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
        q_chunks = []
        for start in range(0, X_train.shape[0], batch_size):
            q_chunks.append(
                F.normalize(model.embedder(X_train[start:start + batch_size]), dim=-1).cpu()
            )
        q_all = torch.cat(q_chunks)
    c_norm_cpu = c_norm.cpu()

    stats = {}
    for p in range(P):
        grp = sample_groups[p] if sample_groups is not None else None
        size = len(grp) if grp else 0
        if size == 0:
            continue
        idx_t = torch.as_tensor(grp, dtype=torch.long)
        q_grp = q_all[idx_t]
        cohesion = float((q_grp @ c_norm_cpu[p]).mean())

        tl = target_labels.get(p) if target_labels is not None else None
        purity, gap = None, None
        if tl is not None and tl.get("kind") == "classification":
            purity = tl["top_prop"]
            gap = purity - global_majority_prop if global_majority_prop is not None else None

        # [추가] label entropy H(y|c) = -Σ p(y|c) log p(y|c) — purity(최다
        # 클래스 비율 하나만 봄)와 달리 그룹 내 클래스 분포 전체를 반영.
        # 예: 3-class에서 (0.5, 0.5, 0.0)과 (0.5, 0.25, 0.25)는 purity가
        # 같아도(0.5) entropy는 다름(전자가 더 낮음, 2개 클래스에만 걸쳐
        # 있으므로) — purity가 못 보는 "얼마나 여러 클래스에 흩어져
        # 있는가"를 추가로 잡아냄. classification에서만 의미 있음.
        entropy = None
        if tasktype in ("multiclass", "binclass"):
            y_grp_int = np.rint(y_train_np[grp]).astype(int)
            _, grp_counts = np.unique(y_grp_int, return_counts=True)
            p_y = grp_counts / grp_counts.sum()
            entropy = float(-(p_y * np.log(p_y + 1e-12)).sum())

        stats[p] = {"size": size, "purity": purity, "cohesion": cohesion,
                     "gap": gap, "entropy": entropy}

    return stats


# ─────────────────────────────────────────────────────────────
# ECE 계산(재사용 가능한 standalone 버전)
# ─────────────────────────────────────────────────────────────

def compute_ece(pred_confidence: np.ndarray, corrects: np.ndarray, n_bins: int = 5) -> float:
    """표준 ECE(Guo et al. 2017) — run_calibration_analysis 내부에서 쓰는 것과
    같은 정의(bin별 |accuracy - mean_confidence|를 bin 크기로 가중평균)를
    독립 함수로 뺌. agg_emb_shuffle 같은 ablation 후 확률 자체가 무너졌는지
    (calibration 문제) vs accuracy만 유지된 채 확률 분포가 다른 이유로
    흔들렸는지(logit scale 등)를 가르는 데 씀 — logloss 폭증이 반드시
    calibration 악화를 의미하진 않으므로, 이 둘을 분리해서 봐야 함.
    """
    pred_confidence = np.asarray(pred_confidence)
    corrects = np.asarray(corrects)
    n_total = len(corrects)
    if n_total == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (pred_confidence >= lo) & (pred_confidence <= hi)
        else:
            mask = (pred_confidence >= lo) & (pred_confidence < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float(corrects[mask].mean())
        mean_conf = float(pred_confidence[mask].mean())
        ece += (n / n_total) * abs(acc - mean_conf)
    return float(ece)


# ─────────────────────────────────────────────────────────────
# Branch별 선형 기여도(||W_i x_i||) — activation norm이 아니라
# head가 실제로 보는 지점에서 측정
# ─────────────────────────────────────────────────────────────

def compute_branch_linear_contribution(model, X, batch_size: int = 512):
    """head의 첫 Linear(model._head_first_linear)가 **실제로 받는 입력**
    (LayerNorm이 있으면 그걸 통과한 뒤)을 forward hook으로 붙잡아서,
    branch별(query/context/agg 등, model._head_block_slices 기준)로
    ||W_i @ x_i||(그 branch가 head의 첫 hidden layer에 실제로 기여하는
    선형 성분의 크기)를 계산.

    [배경] activation norm(raw, concat 전 값)만 보고 "이 branch가 크니까
    지배적이다"라고 결론 내리면 안 됨 — Linear(Wx+b)는 x가 100배 커도
    W가 100배 작으면 출력은 똑같음(activation-weight trade-off). 게다가
    raw activation은 head 내부 LayerNorm(있는 경우)을 거치기 **전** 값이라
    classifier가 실제로 보는 것과 다를 수 있음. 이 함수는 그 두 문제를
    모두 피해서, "실제로 head 입력에 도달한 뒤 그 branch의 weight까지
    곱한 값"을 직접 잼 — causal intervention(--ablation *_shuffle 등)
    만큼 강한 증거는 아니지만, activation norm보다는 훨씬 head가 실제로
    보는 것에 가까운 관찰(observation)임.

    재학습 불필요 — forward pass만 필요해서 --from_saved_state와 같이
    쓸 수 있음(--log_branch_gradients는 학습 중 gradient가 필요해서
    재학습이 있어야 했던 것과 대비).

    반환: {branch_name: {"contribution_norm_mean": float,
                          "share_of_total": float}}  # share는 branch별
      norm 합 대비 비율(벡터 합이 아니라 norm의 합이라 상쇄를 무시한 대략적
      지표 — 정확한 분해는 아니지만 "이 branch가 대략 몇 %를 차지하는가"의
      직관적 요약으로는 유효).
    """
    if not hasattr(model, "_head_first_linear") or not hasattr(model, "_head_block_slices"):
        raise ValueError("이 모델에는 _head_first_linear/_head_block_slices가 없습니다 "
                          "(구버전 체크포인트이거나 예상 밖의 head 구조).")
    if not model._head_block_slices:
        raise ValueError("_head_block_slices가 비어 있습니다 — fusion_mode='residual'/'gated_sum'/'anchor_gate'/'context_gated_beta'이면 "
                          "concat 자체가 없어 이 진단이 적용 안 됩니다(residual은 fusion_alpha/beta, "
                          "gated_sum/anchor_gate/context_gated_beta는 head_gate_mean/var/entropy가 이미 branch별 기여도 지표임).")

    model.eval()
    W = model._head_first_linear.weight.detach()  # (out, in)
    slices = model._head_block_slices              # {name: (start, end)}

    captured = {}
    def _hook(module, inp, out):
        captured["x"] = inp[0].detach()
    handle = model._head_first_linear.register_forward_hook(_hook)

    per_branch_norms = {name: [] for name in slices}
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                model(X[start:start + batch_size])
                x = captured["x"]  # (B, in) — head 첫 Linear가 실제로 받은 입력
                for name, (s, e) in slices.items():
                    contrib = x[:, s:e] @ W[:, s:e].T   # (B, out) — 이 branch만의 선형 기여
                    per_branch_norms[name].append(contrib.norm(dim=-1).cpu())
    finally:
        handle.remove()

    result = {}
    means = {}
    per_sample_arrays = {}
    for name, chunks in per_branch_norms.items():
        arr = torch.cat(chunks).numpy()
        per_sample_arrays[name] = arr
        means[name] = float(arr.mean())
    total = sum(means.values())
    for name, m in means.items():
        result[name] = {
            "contribution_norm_mean": m,
            "share_of_total": (m / total) if total > 0 else float("nan"),
            "contribution_norm_per_sample": per_sample_arrays[name],  # [추가] 샘플별 원본 —
                # 분산/상관 분석용(analyze_branch_information에서 재사용).
        }
    return result


def print_branch_linear_contribution(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Branch별 선형 기여도 (||W_i x_i||, head 첫 Linear 입력 기준)")
    print(f"{'='*60}")
    print(f"  {'branch':<14}{'contribution_norm':>20}{'share(대략)':>14}")
    for name, r in sorted(result.items(), key=lambda kv: -kv[1]["contribution_norm_mean"]):
        print(f"  {name:<14}{r['contribution_norm_mean']:>20.4f}{r['share_of_total']:>13.1%}")
    print(f"  (share는 벡터 합이 아니라 norm의 합 기준 근사치 — branch끼리 상쇄되는")
    print(f"   부분은 못 잡음, '대략 몇 % 비중인가' 정도의 참고용. activation norm이나")
    print(f"   gradient norm과 다르게 이건 head가 실제로 계산에 쓰는 선형 성분 크기라")
    print(f"   'classifier가 이 branch를 얼마나 반영하는가'에 더 가까운 지표.)")


def analyze_branch_information(model, X, tasktype: str, batch_size: int = 512):
    """"agg_emb가 크게 기여하지만 정보가 없을 수도 있다"는 가설(사용자 제안,
    시나리오 1/2/3)을 직접 검증. norm(크기)이 아니라 정보량을 잼:

    1. contribution 분산(CV=std/mean) — 샘플마다 거의 똑같은 값이면(CV
       작음) "bias처럼 작동"(시나리오 2)일 가능성.
    2. raw embedding(query_emb/context_emb/agg_emb, W 곱하기 전)의 PCA —
       첫 PC가 분산 대부분을 설명하면(예: 90%+) 사실상 거의 한 방향으로만
       움직이는 저정보 표현(시나리오 1/2와 정합).
    3. redundancy — agg_emb를 query_emb로 선형회귀했을 때의 R² — 높으면
       agg_emb가 query_emb에서 선형적으로 복원 가능한 중복 정보라는 뜻
       (시나리오 3, "가장 가능성 높다"고 지목된 것). context_emb도 같이 봄.

    재학습 불필요(forward pass만) — --from_saved_state와 같이 쓸 수 있음.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    model.eval()
    embs = {"query": [], "context": [], "agg": []}
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            out = model(X[start:start + batch_size])
            embs["query"].append(out["query_emb"].cpu())
            embs["context"].append(out["context_emb"].cpu())
            embs["agg"].append(out["agg_emb"].cpu())
    for k in embs:
        embs[k] = torch.cat(embs[k], dim=0).numpy()

    # contribution(||W_i x_i||)의 샘플별 분산 — 위 compute_branch_linear_
    # contribution()을 그대로 재사용(중복 계산 피함).
    contrib = compute_branch_linear_contribution(model, X, batch_size=batch_size)

    info = {}
    for name in ("query", "context", "agg"):
        c = contrib[name]["contribution_norm_per_sample"]
        cv = float(c.std() / c.mean()) if c.mean() > 0 else float("nan")

        emb = embs[name]
        n_comp = min(20, emb.shape[0], emb.shape[1])
        pca = PCA(n_components=n_comp)
        pca.fit(emb)
        evr = pca.explained_variance_ratio_
        # 90% 분산을 설명하는 데 필요한 PC 개수 — 작을수록(예: 1~2개)
        # "거의 한 방향"이라는 뜻, 클수록 다양한 정보를 담고 있다는 뜻.
        cum = np.cumsum(evr)
        n90 = int(np.searchsorted(cum, 0.9) + 1)

        # [추가] PCA는 기본적으로 평균을 뺀(centering) 뒤 계산하므로,
        # "샘플마다 거의 똑같은 상수 벡터"(시나리오 2)라면 그 상수 성분
        # 자체가 평균이라 centering 과정에서 사라지고, 남은 노이즈의
        # rank만 보게 됨(실측: 상수+노이즈 벡터를 PCA하면 노이즈가
        # isotropic이라 PC1_ratio가 오히려 낮게 나옴 — PCA만으론 "거의
        # 상수인가"를 못 잡는다는 뜻). 그래서 "평균 대비 샘플 간 변동
        # 크기"를 별도로 직접 잼 — 작으면(예: <0.05) 진짜 "거의 상수
        # 벡터"(bias처럼 작동), 크면 샘플마다 실질적으로 다른 값.
        mean_vec = emb.mean(axis=0)
        deviation = emb - mean_vec
        relative_variation = float(deviation.std() / (np.linalg.norm(mean_vec) + 1e-8))

        # [추가] pairwise cosine similarity — rel_var/R²는 각각 "평균 대비
        # 변동 크기"와 "query로 설명되는 비율"을 보는 거라, "진짜 거의 다
        # 같은 방향을 가리키는가"(가설A: agg_i·agg_j 방향이 다 비슷) vs
        # "방향은 다양한데 classifier가 그 다양성을 안 쓰는가"(가설B)를
        # 직접 가르진 못함 — 방향 자체의 유사도를 재는 이게 그 둘을
        # 가르는 가장 직접적인 지표. 평균 cosine이 1에 가까우면(예: >0.9)
        # 가설A(embedding 자체가 거의 한 방향)가 강하게 지지되고, 낮으면
        # (예: <0.5) 가설B(다양한데 활용을 안 함) 쪽. 표본이 크면(n>2000)
        # O(n²) 메모리 부담이 커서 서브샘플링.
        n_sample_for_cos = min(2000, emb.shape[0])
        if emb.shape[0] > n_sample_for_cos:
            _idx = np.random.RandomState(0).choice(emb.shape[0], n_sample_for_cos, replace=False)
            emb_for_cos = emb[_idx]
        else:
            emb_for_cos = emb
        emb_norm = emb_for_cos / (np.linalg.norm(emb_for_cos, axis=1, keepdims=True) + 1e-8)
        sim_matrix = emb_norm @ emb_norm.T
        n_c = sim_matrix.shape[0]
        off_diag_mask = ~np.eye(n_c, dtype=bool)
        pairwise_cosine_mean = float(sim_matrix[off_diag_mask].mean())
        pairwise_cosine_std  = float(sim_matrix[off_diag_mask].std())

        info[name] = {
            "contribution_mean": float(c.mean()),
            "contribution_std":  float(c.std()),
            "contribution_cv":   cv,   # 작을수록(예: <0.1) "거의 상수" 의심
            "pca_top1_ratio":    float(evr[0]),   # 변동 내부의 집중도(상수 성분과는 별개)
            "pca_n90":           n90,
            "relative_variation": relative_variation,  # 작을수록(예: <0.05) 진짜 "거의 상수"
            "pairwise_cosine_mean": pairwise_cosine_mean,  # 클수록(예: >0.9) 가설A(거의 한 방향)
            "pairwise_cosine_std":  pairwise_cosine_std,
            "embed_dim":         emb.shape[1],
        }


    # redundancy: agg_emb/context_emb를 query_emb로 선형회귀했을 때 R²
    def _linreg_r2(target, source):
        reg = LinearRegression().fit(source, target)
        r2 = reg.score(source, target)  # sklearn 기본 R²(다중 출력이면 각
        # 출력의 R²를 평균 — multioutput='uniform_average'가 기본값)
        return float(r2)

    redundancy = {
        "agg_from_query_r2":     _linreg_r2(embs["agg"], embs["query"]),
        "context_from_query_r2": _linreg_r2(embs["context"], embs["query"]),
    }

    return {"branch_info": info, "redundancy": redundancy}


def print_branch_information(result: dict) -> None:
    info = result["branch_info"]
    red  = result["redundancy"]
    print(f"\n{'='*60}")
    print(f"  Branch별 정보량 진단 (norm이 아니라 '샘플마다 다른가')")
    print(f"{'='*60}")
    print(f"  {'branch':<10}{'contrib_mean':>13}{'contrib_CV':>12}{'rel_var':>10}{'cos_sim':>10}{'PC1_ratio':>11}{'n_PC(90%)':>11}{'dim':>6}")
    for name, r in info.items():
        print(f"  {name:<10}{r['contribution_mean']:>13.3f}{r['contribution_cv']:>12.3f}"
              f"{r['relative_variation']:>10.3f}{r['pairwise_cosine_mean']:>10.3f}"
              f"{r['pca_top1_ratio']:>11.1%}{r['pca_n90']:>11d}{r['embed_dim']:>6d}")
    print(f"  (rel_var가 낮으면(대략 <0.05) embedding이 샘플과 거의 무관한 '거의 상수")
    print(f"   벡터'라는 뜻 — bias처럼 작동해서 shuffle해도 별 차이가 없는 이유가 설명됨.")
    print(f"   cos_sim(pairwise cosine similarity 평균)은 rel_var/R²가 못 가르는 두 가설을")
    print(f"   직접 구분함 — 높으면(예: >0.9) '가설A: embedding 자체가 거의 한 방향'(그래서")
    print(f"   shuffle해도 비슷한 값끼리 바뀌는 것), 낮으면(예: <0.5) '가설B: embedding은")
    print(f"   다양한데 classifier가 그 다양성을 활용하지 않는다'는 쪽이 더 유력해짐.")
    print(f"   PC1_ratio/n_PC(90%)는 그 '변동이 있는 부분 안에서' 얼마나 다양한 방향으로")
    print(f"   퍼져 있는지를 보는 것 — rel_var가 이미 작으면 이 둘은 노이즈의 형태를")
    print(f"   보는 것뿐이라 별 의미 없음(PCA는 평균을 빼고 계산해서 '거의 상수'라는")
    print(f"   신호 자체는 못 잡음 — 그래서 rel_var를 따로 둠). contrib_CV가 낮은 것도")
    print(f"   비슷한 신호(head에 도달하는 선형 기여도 자체가 샘플마다 안 변함).)")


    print(f"\n  Redundancy(query_emb로부터 선형 복원 가능한 정도, R²):")
    print(f"    agg_emb     ~ f(query_emb) : R²={red['agg_from_query_r2']:.3f}")
    print(f"    context_emb ~ f(query_emb) : R²={red['context_from_query_r2']:.3f}")
    print(f"  (R²가 높으면(예: >0.7) 그 branch가 query_emb에서 선형적으로 거의")
    print(f"   복원 가능한 중복 정보라는 뜻 — agg_emb_shuffle이 안 먹히는 이유가")
    print(f"   '정보가 없어서'가 아니라 'query_emb에 이미 있는 정보라서'일 수 있음.)")


def compute_branch_gradient_attribution(model, X, y, tasktype: str, batch_size: int = 512):
    """재학습 없이(가중치 고정) 한 번의 forward+backward만으로, 실제 loss가
    각 branch(query/context/agg)에 얼마나 gradient를 보내는지 측정.

    [배경] --log_branch_gradients는 학습 도중 epoch마다 기록하는 거라
    재학습이 필요했음 — 이건 이미 학습된 모델(--from_saved_state)에 test/
    eval 데이터를 한 번 흘려서 gradient만 재는, 훨씬 가벼운 one-shot 측정.
    head의 첫 Linear 입력(LayerNorm 통과 후 — compute_branch_linear_
    contribution과 같은 지점)에 retain_grad를 걸어 backward 후 grad norm을
    branch별로 분리.
    """
    if not hasattr(model, "_head_first_linear") or not hasattr(model, "_head_block_slices"):
        raise ValueError("_head_first_linear/_head_block_slices가 없습니다.")
    if not model._head_block_slices:
        raise ValueError("fusion_mode='residual'/'gated_sum'/'anchor_gate'/'context_gated_beta'에서는 이 진단이 적용 안 됩니다.")

    model.eval()  # dropout 등은 끄되, gradient 계산 자체는 정상적으로 됨
    criterion = get_criterion(tasktype)
    slices = model._head_block_slices

    captured = {}
    def _hook(module, inp, out):
        x = inp[0]
        x.retain_grad()
        captured["x"] = x
    handle = model._head_first_linear.register_forward_hook(_hook)

    grad_norms = {name: [] for name in slices}
    act_norms  = {name: [] for name in slices}
    try:
        for start in range(0, len(X), batch_size):
            model.zero_grad(set_to_none=True)
            X_batch = X[start:start + batch_size]
            y_batch = y[start:start + batch_size]
            out = model(X_batch)
            logits = out["logits"]
            if tasktype == "regression":
                loss = criterion(logits.squeeze(-1), y_batch.float())
            elif tasktype == "binclass":
                loss = criterion(logits.squeeze(-1), y_batch.float())
            else:
                loss = criterion(logits, y_batch.long())
            loss.backward()

            x = captured["x"]
            if x.grad is None:
                continue  # 이 배치는 head까지 gradient가 안 흐름(극히 드묾) — 스킵
            for name, (s, e) in slices.items():
                grad_norms[name].append(x.grad[:, s:e].norm(dim=-1).detach().cpu())
                act_norms[name].append(x[:, s:e].detach().norm(dim=-1).detach().cpu())
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    result = {}
    for name in slices:
        if not grad_norms[name]:
            continue
        g = torch.cat(grad_norms[name])
        a = torch.cat(act_norms[name])
        result[name] = {
            "grad_norm_mean": float(g.mean()),
            "grad_norm_std":  float(g.std()),
            "act_norm_mean":  float(a.mean()),
        }
    total_grad = sum(r["grad_norm_mean"] for r in result.values())
    for name, r in result.items():
        r["grad_share"] = r["grad_norm_mean"] / total_grad if total_grad > 0 else float("nan")
    return result


def print_branch_gradient_attribution(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Branch별 Gradient Attribution (one-shot, 재학습 불필요)")
    print(f"{'='*60}")
    print(f"  {'branch':<10}{'grad_norm':>14}{'grad_share':>12}{'act_norm':>14}")
    for name, r in sorted(result.items(), key=lambda kv: -kv[1]["grad_norm_mean"]):
        print(f"  {name:<10}{r['grad_norm_mean']:>14.6e}{r['grad_share']:>12.1%}{r['act_norm_mean']:>14.4f}")
    print(f"  (grad_share가 낮으면(예: agg가 query 대비 100배 이상 작으면) loss가")
    print(f"   그 branch를 거의 안 거쳐 흐른다는 뜻 — head가 실제로 그 branch에")
    print(f"   맞춰 업데이트되고 있지 않다는 직접 증거.)")


def compute_pre_fusion_gradient_attribution(model, X, y, tasktype: str, batch_size: int = 512):
    """residual fusion(z = query_emb + β·agg_emb, 필요시 + α·context_emb)
    전용 one-shot gradient attribution.

    [배경] compute_branch_gradient_attribution은 head 첫 Linear의 입력을
    branch별 slice(_head_block_slices)로 나눠서 재는데, 이건 fusion_mode=
    "concat"([q|c|a] → Linear로 branch별 W가 분리되는 경우)에만 성립하는
    개념이다. residual에서는 head에 들어가기 전에 이미 branch들이 하나의
    벡터 z로 합쳐져 있어("classifier는 W_z 하나만 본다") _head_block_slices
    자체가 없고(tabera.py, concat일 때만 채워짐), 그 함수를 그대로 가져다
    쓰면 구조적으로 안 맞는 질문을 하는 셈이 된다 — 그래서 이 함수를 새로
    만들지 않고 concat 전용으로 남겨두기로 함(사용자 결정).

    residual에서 자연스러운 질문은 "loss가 fusion **이전** 원본 표현
    (query_emb/agg_emb/context_emb, out dict에 fusion_mode와 무관하게 항상
    raw로 노출됨)에 얼마나 gradient를 돌려보내는가"이다. 여기에 직접
    retain_grad()를 걸어 backward 한 번으로 측정한다 — head가 fusion 이후
    실제로 어느 branch에 "맞춰 업데이트"되고 있는지 그 근원(query_emb vs
    agg_emb, 분석계획 4번)을 residual 구조에 맞는 지점에서 본다.

    재학습 불필요(가중치 고정, forward+backward 한 번) — --from_saved_state와
    같이 쓸 수 있음. --log_branch_gradients(supervised.py)는 학습 도중 매
    epoch 이 정보를 기록하지만 재학습이 필요했음 — 이건 그 one-shot 버전
    (같은 raw query_emb/agg_emb/context_emb 지점을 잼, tabera.py:1994-1996
    의 학습 중 hook과 동일한 대상).
    """
    model.eval()
    criterion = get_criterion(tasktype)
    branch_names = ("query_emb", "agg_emb", "context_emb")

    grad_norms = {name: [] for name in branch_names}
    act_norms  = {name: [] for name in branch_names}
    for start in range(0, len(X), batch_size):
        model.zero_grad(set_to_none=True)
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]
        out = model(X_batch)  # no_grad 밖 — autograd 정상 추적

        tensors = {}
        for name in branch_names:
            t = out.get(name)
            if t is None or not t.requires_grad:
                continue  # context_emb가 detach_context_grad로 끊긴 설정 등은 자연히 스킵
            t.retain_grad()
            tensors[name] = t
        if not tensors:
            continue

        logits = out["logits"]
        if tasktype == "regression":
            loss = criterion(logits.squeeze(-1), y_batch.float())
        elif tasktype == "binclass":
            loss = criterion(logits.squeeze(-1), y_batch.float())
        else:
            loss = criterion(logits, y_batch.long())
        loss.backward()

        for name, t in tensors.items():
            if t.grad is None:
                continue  # 이 배치에서 이 branch까지 gradient가 안 흐름 — 진단적으로 유의미하니 스킵만
            grad_norms[name].append(t.grad.norm(dim=-1).detach().cpu())
            act_norms[name].append(t.detach().norm(dim=-1).cpu())
    model.zero_grad(set_to_none=True)

    result = {}
    for name in branch_names:
        if not grad_norms[name]:
            continue
        g = torch.cat(grad_norms[name])
        a = torch.cat(act_norms[name])
        result[name] = {
            "grad_norm_mean": float(g.mean()),
            "grad_norm_std":  float(g.std()),
            "act_norm_mean":  float(a.mean()),
        }
    total_grad = sum(r["grad_norm_mean"] for r in result.values())
    for name, r in result.items():
        r["grad_share"] = r["grad_norm_mean"] / total_grad if total_grad > 0 else float("nan")
    return result


def print_pre_fusion_gradient_attribution(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Pre-fusion Gradient Attribution (residual 전용, one-shot)")
    print(f"{'='*60}")
    print(f"  {'branch':<10}{'grad_norm':>14}{'grad_share':>12}{'act_norm':>14}")
    for name, r in sorted(result.items(), key=lambda kv: -kv[1]["grad_norm_mean"]):
        print(f"  {name:<10}{r['grad_norm_mean']:>14.6e}{r['grad_share']:>12.1%}{r['act_norm_mean']:>14.4f}")
    print(f"  (raw query_emb/agg_emb/context_emb — fusion(z=q+βa) **이전**, head 진입 전")
    print(f"   지점에서 잰 gradient. agg의 grad_share가 낮으면(예: query 대비 100배 이상")
    print(f"   작으면) loss가 agg_emb 쪽으로 거의 안 흐른다는 뜻 — 단, gradient는 '학습")
    print(f"   신호의 흐름'이지 '예측이 실제로 그 branch를 쓰는가'가 아니므로(이미 잘")
    print(f"   학습된 branch는 gradient가 작아도 여전히 예측에 크게 기여할 수 있음),")
    print(f"   --ablation agg_emb_zero/shuffle 결과와 반드시 같이 해석할 것.)")


def compute_head_input_cancellation(model, X, batch_size: int = 512):
    """residual fusion(z = LN(q) + β·LN(a)) 전용. representation은 크게
    움직이는데(‖z-q‖=‖β·LN(a)‖ 큼) accuracy는 거의 안 변한다는 관찰(사용자
    지적)에 대해 "head 진입 직후(첫 hidden layer)에서 이미 상쇄되는가"를
    직접 검증.

    [원리] head의 첫 Linear(W, bias b)는 선형이므로,
        W@z + b = (W@LN(q) + b) + β·(W@LN(a))
    이 항등식이 **항상** 정확히 성립한다(근사가 아님 — 아래서 실제 forward
    출력과 비교해 부동소수점 오차 수준인지 sanity check까지 함). 좌변 두
    항을 각각
        h_q = W@LN(q) + b   (agg_emb_zero ablation이 만드는 값과 정확히 동일)
        h_a = β·(W@LN(a))   (bias 없는 순수 agg 기여분)
    로 부르면, cos(h_q, h_a)와 ‖h_q+h_a‖ vs ‖h_q‖+‖h_a‖를 비교해서 두 기여가
    "반대 방향으로 상쇄"되는지 "직교라 서로 안 건드리는지" "같은 방향으로
    강화"되는지 직접 구분할 수 있다. _head_block_slices(concat 전용)가 전혀
    필요 없음 — "합 다음에 선형 레이어"라는 구조 자체가 이 분해를 보장하는
    residual 고유의 성질.

    raw query_emb/agg_emb(out dict에 항상 노출)에 model.head_query_ln/
    model.head_agg_ln(forward 내부와 정확히 같은 모듈)을 그대로 적용해서
    재현 — 새 파라미터 없음, 순수 관찰. head_branch_l2norm=True인 체크포인트도
    forward와 동일한 순서(LN → L2norm)로 재현.

    재학습 불필요(forward만) — --from_saved_state와 같이 쓸 수 있음.
    """
    if getattr(model, "fusion_mode", None) != "residual":
        raise ValueError(f"이 진단은 fusion_mode='residual' 전용입니다 "
                          f"(현재: {getattr(model, 'fusion_mode', None)}) — "
                          f"'합 다음에 선형 레이어'라는 분해가 β가 스칼라인 "
                          f"residual에서 가장 단순하게 성립함.")
    if not hasattr(model, "_head_first_linear"):
        raise ValueError("이 모델에는 _head_first_linear가 없습니다(구버전 체크포인트).")

    model.eval()
    W = model._head_first_linear.weight.detach()
    b = model._head_first_linear.bias.detach() if model._head_first_linear.bias is not None else None

    cos_chunks, hq_norm_chunks, ha_norm_chunks, hfull_norm_chunks = [], [], [], []
    sanity_err_chunks = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            out = model(X[start:start + batch_size])
            q, a = out["query_emb"], out["agg_emb"]
            beta = float(out["fusion_beta"])

            q_ln = model.head_query_ln(q) if model._per_branch_ln else q
            a_ln = model.head_agg_ln(a) if model._per_branch_ln else a
            if model.head_branch_l2norm:
                q_ln = F.normalize(q_ln, dim=-1)
                a_ln = F.normalize(a_ln, dim=-1)

            h_q = F.linear(q_ln, W, b)              # W@LN(q) + b  (agg_emb_zero와 동일 지점)
            h_a = beta * F.linear(a_ln, W, None)     # β·(W@LN(a))  (bias 없음)

            # sanity check: 실제 forward가 만드는 첫-레이어 출력과 정확히 일치해야 함
            z = q_ln + beta * a_ln
            h_full = F.linear(z, W, b)
            sanity_err_chunks.append((h_full - (h_q + h_a)).norm(dim=-1).cpu())

            cos_chunks.append(F.cosine_similarity(h_q, h_a, dim=-1).cpu())
            hq_norm_chunks.append(h_q.norm(dim=-1).cpu())
            ha_norm_chunks.append(h_a.norm(dim=-1).cpu())
            hfull_norm_chunks.append(h_full.norm(dim=-1).cpu())

    cos      = torch.cat(cos_chunks).numpy()
    hq_norm  = torch.cat(hq_norm_chunks).numpy()
    ha_norm  = torch.cat(ha_norm_chunks).numpy()
    hfull_norm = torch.cat(hfull_norm_chunks).numpy()
    sanity_err = torch.cat(sanity_err_chunks).numpy()

    # <1이면 h_q/h_a가 서로 상쇄(반대 방향 성분이 겹쳐서 합의 norm이 줄어듦),
    # ≈1이면 대략 직교(서로 거의 안 건드림), >1이면 오히려 강화(같은 방향
    # 정렬 — 두 unit vector가 완전히 같은 방향이면 최대 (‖hq‖+‖ha‖)/‖hq+ha‖=1
    # 이 그대로 유지되므로 이 비율이 1을 넘는 건 사실 불가능하지 않고, 두
    # norm이 서로 다를 때 삼각부등식 여유 안에서 소폭 발생 가능 — 1 근처가
    # "간섭이 거의 없다"의 기준선).
    cancellation_ratio = hfull_norm / (hq_norm + ha_norm + 1e-8)

    return {
        "cos_hq_ha_mean":   float(cos.mean()),
        "cos_hq_ha_median": float(np.median(cos)),
        "cos_hq_ha_p5":     float(np.percentile(cos, 5)),
        "cos_hq_ha_p95":    float(np.percentile(cos, 95)),
        "hq_norm_mean":     float(hq_norm.mean()),
        "ha_norm_mean":     float(ha_norm.mean()),
        "hfull_norm_mean":  float(hfull_norm.mean()),
        "cancellation_ratio_mean":   float(cancellation_ratio.mean()),
        "cancellation_ratio_median": float(np.median(cancellation_ratio)),
        "sanity_max_reconstruction_error": float(sanity_err.max()),  # ≈0(부동소수점 오차 수준)이어야 정상 — 크면 코드 버그 의심
        "cos_hq_ha_per_sample":            cos,
        "cancellation_ratio_per_sample":   cancellation_ratio,
    }


def print_head_input_cancellation(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Head Input Cancellation (residual 전용, one-shot)")
    print(f"{'='*60}")
    print(f"  cos(h_q, h_a):       mean={result['cos_hq_ha_mean']:+.4f}  median={result['cos_hq_ha_median']:+.4f}  "
          f"p5={result['cos_hq_ha_p5']:+.4f}  p95={result['cos_hq_ha_p95']:+.4f}")
    print(f"  ‖h_q‖ mean={result['hq_norm_mean']:.4f}   ‖h_a‖ mean={result['ha_norm_mean']:.4f}   "
          f"‖h_q+h_a‖ mean={result['hfull_norm_mean']:.4f}")
    print(f"  cancellation_ratio(=‖h_q+h_a‖/(‖h_q‖+‖h_a‖)): mean={result['cancellation_ratio_mean']:.4f}  "
          f"median={result['cancellation_ratio_median']:.4f}")
    print(f"  sanity check(재구성 오차, ≈0이어야 정상): {result['sanity_max_reconstruction_error']:.2e}")
    print(f"  (cos(h_q,h_a)가 음수이고 cancellation_ratio가 1보다 뚜렷이 작으면 — agg의")
    print(f"   영향이 head 진입 직후(첫 hidden layer)에서 이미 상당 부분 상쇄된다는 뜻.")
    print(f"   representation(‖z-q‖)은 크게 움직이는데 accuracy는 거의 안 변하는 현상에")
    print(f"   대한 구조적 설명 후보 하나 — cos(query_emb, agg_emb)의 raw embedding 레벨")
    print(f"   음수 부호가 head를 거치며 사라지는지/유지되는지/더 강해지는지 비교해서 볼 것.)")


def compute_head_sensitivity(model, X, batch_size: int = 512, scale_factor: float = 10.0):
    """agg_emb(및 다른 branch)를 head 입력 지점에서 직접 zero/random(배치 내
    셔플)/scale(×10)로 바꿨을 때, 최종 logits가 얼마나 변하는지 직접 측정.

    [배경] --ablation agg_emb_shuffle은 '다른 real 샘플의 값으로 바꿔치기'라
    그 값이 우연히 비슷하면 효과가 작게 나올 수 있음(값 자체가 collapse돼
    있으면 특히 그럼). 이건 head 입력에서 직접 조작해서 head가 그 branch에
    '얼마나 민감한가'를 재는, shuffle보다 더 통제된 측정 — zero(정보를 아예
    지움)/scale(크기를 10배로 키움, 정보는 유지)까지 같이 봐서, "정보가
    없어서 안 변하는가" vs "있어도 head가 그 크기 변화에도 무감각한가"를
    구분.

    재학습 불필요 — forward pass만 필요해서 --from_saved_state 가능.
    """
    if not hasattr(model, "_head_first_linear") or not hasattr(model, "_head_block_slices"):
        raise ValueError("_head_first_linear/_head_block_slices가 없습니다.")
    if not model._head_block_slices:
        raise ValueError("fusion_mode='residual'/'gated_sum'/'anchor_gate'/'context_gated_beta'에서는 이 진단이 적용 안 됩니다.")

    model.eval()
    head = model.head
    first = model._head_first_linear
    idx = next(i for i, layer in enumerate(head) if layer is first)
    rest_of_head = torch.nn.Sequential(*list(head.children())[idx + 1:])
    slices = model._head_block_slices

    captured = {}
    def _hook(module, inp, out):
        captured["x"] = inp[0].detach()
    handle = first.register_forward_hook(_hook)

    deltas = {name: {"zero": [], "random": [], "scaled": []} for name in slices}
    logit_ref_norms = []
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                X_batch = X[start:start + batch_size]
                out = model(X_batch)
                logits_full = out["logits"]
                logit_ref_norms.append(logits_full.norm(dim=-1).cpu())
                x = captured["x"]
                B = x.shape[0]
                if B < 2:
                    continue  # random(셔플)은 배치 크기 2 이상 필요
                perm = torch.randperm(B, device=x.device)
                for name, (s, e) in slices.items():
                    x_zero = x.clone(); x_zero[:, s:e] = 0.0
                    x_rand = x.clone(); x_rand[:, s:e] = x[perm, s:e]
                    x_scaled = x.clone(); x_scaled[:, s:e] = x[:, s:e] * scale_factor

                    for key, x_pert in [("zero", x_zero), ("random", x_rand), ("scaled", x_scaled)]:
                        logits_pert = rest_of_head(first(x_pert))
                        delta = (logits_pert - logits_full).norm(dim=-1)
                        deltas[name][key].append(delta.cpu())
    finally:
        handle.remove()

    ref_norm_mean = float(torch.cat(logit_ref_norms).mean())
    result = {"logit_ref_norm_mean": ref_norm_mean, "branches": {}}
    for name in slices:
        result["branches"][name] = {}
        for key in ("zero", "random", "scaled"):
            if not deltas[name][key]:
                continue
            arr = torch.cat(deltas[name][key])
            result["branches"][name][key] = {
                "mean_logit_delta": float(arr.mean()),
                "relative_delta": float(arr.mean()) / (ref_norm_mean + 1e-8),  # logit 크기 대비 상대 변화
            }
    return result


def print_head_sensitivity(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Head Sensitivity (branch를 직접 조작했을 때 logit 변화량)")
    print(f"{'='*60}")
    print(f"  기준 logit norm 평균 = {result['logit_ref_norm_mean']:.4f}")
    print(f"  {'branch':<10}{'perturbation':<12}{'mean_logit_delta':>18}{'relative_delta':>16}")
    for name, r in result["branches"].items():
        for key in ("zero", "random", "scaled"):
            if key in r:
                print(f"  {name:<10}{key:<12}{r[key]['mean_logit_delta']:>18.4f}{r[key]['relative_delta']:>16.1%}")
    print(f"  (relative_delta가 낮으면(예: <5%) 그 조작이 logit을 거의 안 바꾼다는 뜻.")
    print(f"   zero도 낮고 scaled(×10, 정보는 그대로 두고 크기만 키움)도 낮으면 —")
    print(f"   head가 그 branch의 존재 여부/크기 둘 다에 무감각하다는 강한 증거.")
    print(f"   zero는 낮은데 scaled는 높으면 head가 '크기'에는 반응하지만 '내용'에는")
    print(f"   안 반응한다는(bias처럼 취급) 뜻일 수 있음.)")


def run_calibration_analysis(model, X_test, y_test, tasktype: str,
                              batch_size: int = 512, n_bins: int = 5,
                              X_train=None, y_train=None, class_names=None):
    """전체 test set에서 routing_confidence(①)와 prediction_confidence(②)
    각각을 실제 정확도와 대조.

    --explain은 n_explain개(기본 3~10개) 샘플만 텍스트로 자세히 보여주는
    반면, 이건 "routing이 애매해도 최종 예측이 믿을 만한가"를 개별 샘플
    하나가 아니라 **test set 전체 통계**로 답하기 위한 것 — 그래서 feature
    요약/neighbour 텍스트 같은 비싼 걸 다 걷어내고 숫자(routing_confidence,
    prediction_confidence, correct 여부)만 뽑는다.

    [수정] routing_confidence를 prediction_confidence와 똑같이 고정
    0/20/40/60/80/100% 구간으로 나눴던 게 잘못이었음 — 실측(adult, P=190)에서
    test set 4523개 전부가 0-20% 구간 하나에 몰리는 결과가 나왔는데, 이걸
    "routing이 무너졌다"고 바로 해석하면 안 됨. routing_confidence =
    softmax(cos(q,c)*routing_scale)의 절대 스케일은 n_prototypes(P)에
    구조적으로 종속적임 — 균등분포 기준선이 1/P이므로(P=190이면 0.53%),
    prediction_confidence(항상 0~100%가 보편적으로 의미 있는 실제 확률)와
    달리 고정 percent 구간이 P가 다른 데이터셋끼리도, 심지어 같은
    데이터셋 안에서도 "이게 낮은 게 맞는지" 판단할 기준이 없음. 그래서:
    (a) 분포 자체(mean/median/std/min/max/p90/p99)를 먼저 보여주고,
    (b) 구간은 절대 confidence % 대신 **percentile**(하위 20%/20-40%/.../
    상위 20%) 기준으로 나눔 — 이러면 P나 routing_scale이 뭐든 "이 test set
    안에서 상대적으로 routing이 애매했던 샘플들과 확신했던 샘플들 간에
    accuracy 차이가 있는가"라는, 원래 하려던 질문에 실제로 답이 됨.
    prediction_confidence는 실제 확률이라 고정 구간을 그대로 유지.

    반환: {
      "routing_stats": {"mean":.., "median":.., "std":.., "min":.., "max":..,
                         "p90":.., "p99":.., "n_prototypes":.., "uniform_baseline":..},
      "routing_bins": [(lo_pct, hi_pct, lo_conf, hi_conf, n, acc), ...]  # percentile 기준
      "prediction_bins": [(lo, hi, n, acc, mean_conf), ...],   # 고정 confidence % 기준
      "prediction_ece": float,   # Expected Calibration Error
      "n_total": int,
      "overall_acc": float,
    }
    """
    if tasktype == "regression":
        raise ValueError("calibration_analysis는 classification(binclass/multiclass) 전용입니다 "
                          "— regression엔 'routing/prediction confidence' 개념이 없습니다.")

    model.eval()
    routing_confs, pred_confs, corrects, assigned_centroids, margins = [], [], [], [], []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            X_batch = X_test[start:start + batch_size]
            y_batch = y_test[start:start + batch_size]
            out = model(X_batch, return_explanations=True)

            explanations = out.get("explanations", [])
            if not explanations:
                # memory bank가 아직 안 찼거나 하는 초반 배치 — 스킵
                # (--explain의 "no explanations" 케이스와 동일 사유)
                continue

            pred_idx, pred_probs = get_preds_and_probs(out["logits"][:len(explanations)], tasktype)

            for i, exp in enumerate(explanations):
                routing_confs.append(exp["prototype"]["routing_confidence"])
                assigned_centroids.append(int(exp["prototype"]["centroid_idx"]))
                margins.append(exp["prototype"]["margin"])  # top1 - runner-up1 routing 확신도 격차
                idx = int(pred_idx[i].item())
                pred_confs.append(float(pred_probs[i, idx].item()))
                y_i = int(y_batch[i].item()) if tasktype == "multiclass" else int(y_batch[i].item())
                corrects.append(int(idx == y_i))

    routing_confs      = np.array(routing_confs)
    pred_confs         = np.array(pred_confs)
    corrects           = np.array(corrects)
    assigned_centroids = np.array(assigned_centroids)
    margins             = np.array(margins)
    n_total             = len(corrects)

    if n_total == 0:
        raise RuntimeError("calibration_analysis: 유효한 샘플이 하나도 없습니다 "
                            "(memory bank가 test set 전체에서 한 번도 안 찼을 수 있음).")

    # [추가] centroid_size — 각 샘플이 배정된 centroid에 몇 개의 (train/memory)
    # 샘플이 속해 있는지. "routing confidence는 높은데 accuracy는 낮은 구간이
    # 있다"를 그 자체로 결론 내리지 않고, 그게 특정(거대) centroid에 쏠린
    # 현상인지 직접 대조하기 위한 최소 정보 — assigned_centroid만으로는 안
    # 보이던 것(centroid 크기)까지 같이 저장.
    sample_groups = getattr(getattr(model, "prototype_layer", None), "sample_groups", None)
    if sample_groups is not None:
        centroid_sizes = np.array([len(sample_groups[c]) for c in assigned_centroids])
    else:
        centroid_sizes = np.full(n_total, -1)  # sample_groups 캐싱 전(비정상 케이스) — -1로 표시

    n_prototypes = getattr(getattr(model, "prototype_layer", None), "P", None)

    # [추가] N_eff = exp(H(assignment distribution)) — "alive centroid 수"와
    # "실제로 traffic이 고르게 퍼진 centroid 수"는 다른 개념이라는 게 실측으로
    # 반복 확인됨(예: alive=139인데 상위 몇 개가 test traffic 절반 이상을
    # 담당). 균등분포면 N_eff=P(190), 완전히 한 centroid로만 쏠리면
    # N_eff=1 — "실질적으로 몇 개의 prototype이 일하고 있는가"를 단일 숫자로
    # 요약. test_n_eff는 이번 run의 test 4523개가 실제로 도달한 분포 기준,
    # train_n_eff는 sample_groups(전체 36177개 train) 크기 분포 기준 — 후자가
    # 표본이 훨씬 커서 더 안정적인 지표.
    def _n_eff(counts: np.ndarray) -> float:
        counts = counts[counts > 0]
        if counts.sum() == 0:
            return 0.0
        p = counts / counts.sum()
        h = -(p * np.log(p + 1e-12)).sum()
        return float(np.exp(h))

    _, test_counts = np.unique(assigned_centroids, return_counts=True)
    test_n_eff = _n_eff(test_counts)
    train_n_eff = None
    if sample_groups is not None:
        train_counts = np.array([len(g) for g in sample_groups if g])
        train_n_eff = _n_eff(train_counts)

    routing_stats = {
        "mean":   float(routing_confs.mean()),
        "median": float(np.median(routing_confs)),
        "std":    float(routing_confs.std()),
        "min":    float(routing_confs.min()),
        "max":    float(routing_confs.max()),
        "p90":    float(np.percentile(routing_confs, 90)),
        "p99":    float(np.percentile(routing_confs, 99)),
        "n_prototypes": n_prototypes,
        "uniform_baseline": (1.0 / n_prototypes) if n_prototypes else None,
        "test_n_eff":  test_n_eff,
        "train_n_eff": train_n_eff,
    }

    # [추가] routing/prediction confidence 간, 그리고 centroid_size/accuracy
    # 간 Spearman 상관 — "특정 가설(예: 큰 centroid일수록 부정확)"을 말로만
    # 주장하지 않고 숫자로 같이 내보냄. Spearman을 쓰는 이유: correct는
    # 0/1 binary라 Pearson보다 순위 기반이 덜 왜곡됨(point-biserial과
    # 유사한 해석), 그리고 confidence들끼리도 비선형 단조관계만 있어도
    # 잡아냄.
    from scipy.stats import spearmanr
    corr_routing_vs_pred, _      = spearmanr(routing_confs, pred_confs)
    corr_routing_vs_correct, _   = spearmanr(routing_confs, corrects)
    corr_margin_vs_correct, _    = spearmanr(margins, corrects)
    corr_centroidsize_vs_correct, _ = (
        spearmanr(centroid_sizes, corrects) if sample_groups is not None else (float("nan"), None)
    )
    correlations = {
        "routing_vs_prediction_confidence": float(corr_routing_vs_pred),
        "routing_vs_correct":               float(corr_routing_vs_correct),
        "routing_margin_vs_correct":        float(corr_margin_vs_correct),
        "centroid_size_vs_correct":         float(corr_centroidsize_vs_correct),
    }

    # [추가] centroid_purity/cohesion(train set 기준, compute_centroid_train_stats)를
    # sample 단위로 join + centroid 단위 correlation. X_train/y_train이 없으면
    # (하위호환 — 이 값들 없이 부르는 기존 코드도 있을 수 있음) 이 블록 전체를
    # 건너뜀. 사용자 가설("큰 centroid → 순도 낮음 → 예측 실패")을 sample
    # 단위(centroid_purity vs correct)와 centroid 단위(size vs purity,
    # purity vs test_accuracy) 양쪽에서 직접 검증하기 위함 — 세 번째
    # 세션에서 제안된 3단계 분석(centroid 통계 → centroid 단위 상관 →
    # sample 단위 상관)을 그대로 구현.
    centroid_train_stats = {}
    centroid_level_correlations = {}
    centroid_table = []
    centroid_purities  = np.full(n_total, np.nan)
    centroid_cohesions = np.full(n_total, np.nan)

    if X_train is not None and y_train is not None:
        centroid_train_stats = compute_centroid_train_stats(
            model, X_train, y_train, tasktype, class_names=class_names
        )
        for i, c in enumerate(assigned_centroids):
            st = centroid_train_stats.get(int(c))
            if st is not None:
                if st["purity"] is not None:
                    centroid_purities[i] = st["purity"]
                centroid_cohesions[i] = st["cohesion"]

        _valid_purity = ~np.isnan(centroid_purities)
        if _valid_purity.sum() >= 2:
            corr_purity_vs_correct, _ = spearmanr(centroid_purities[_valid_purity], corrects[_valid_purity])
            correlations["centroid_purity_vs_correct"] = float(corr_purity_vs_correct)
        _valid_cohesion = ~np.isnan(centroid_cohesions)
        if _valid_cohesion.sum() >= 2:
            corr_cohesion_vs_correct, _ = spearmanr(centroid_cohesions[_valid_cohesion], corrects[_valid_cohesion])
            correlations["centroid_cohesion_vs_correct"] = float(corr_cohesion_vs_correct)

        # centroid 단위(sample 단위가 아니라 centroid 하나당 값 하나) 상관 —
        # test set에서 그 centroid에 배정된 샘플들의 평균 accuracy를 test_accuracy로 씀.
        _centroid_ids  = sorted(centroid_train_stats.keys())

        # [추가] train_count vs test_count 전체 표 — "test에서 몇 개 centroid만
        # 쓰였다"는 게 진짜 활용도 문제인지, 아니면 이 태스크가 원래 소수
        # 영역에 자연스럽게 집중되는 구조인지 구분하기 위해 test_n=0인
        # centroid까지 전부 포함해서 남김(필터링 없음 — 아래 correlation
        # 계산용 리스트와 달리 이 표는 test_n=0도 그대로 보여줌).
        centroid_table = []
        for c in _centroid_ids:
            mask = (assigned_centroids == c)
            st = centroid_train_stats[c]
            centroid_table.append({
                "centroid": c, "train_count": st["size"], "test_count": int(mask.sum()),
                "purity": st["purity"], "entropy": st["entropy"], "cohesion": st["cohesion"],
                "test_accuracy": float(corrects[mask].mean()) if mask.sum() > 0 else None,
            })

        _sizes, _purities, _cohesions, _test_accs = [], [], [], []
        for c in _centroid_ids:
            mask = (assigned_centroids == c)
            if mask.sum() == 0:
                continue  # 이 centroid로 배정된 test 샘플이 없으면 test_accuracy 계산 불가
            st = centroid_train_stats[c]
            _sizes.append(st["size"])
            _purities.append(st["purity"] if st["purity"] is not None else np.nan)
            _cohesions.append(st["cohesion"])
            _test_accs.append(float(corrects[mask].mean()))
        _sizes, _purities, _cohesions, _test_accs = map(np.array, (_sizes, _purities, _cohesions, _test_accs))

        if len(_sizes) >= 2:
            _valid = ~np.isnan(_purities)
            if _valid.sum() >= 2:
                r, _ = spearmanr(_sizes[_valid], _purities[_valid])
                centroid_level_correlations["size_vs_purity"] = float(r)
                r, _ = spearmanr(_purities[_valid], _test_accs[_valid])
                centroid_level_correlations["purity_vs_test_accuracy"] = float(r)
            r, _ = spearmanr(_cohesions, _test_accs)
            centroid_level_correlations["cohesion_vs_test_accuracy"] = float(r)
            centroid_level_correlations["n_centroids"] = int(len(_sizes))


    def _fixed_bin_stats(confs, edges):
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            # 마지막 구간만 hi를 포함(<=), 나머지는 [lo, hi) — 100%가 마지막
            # 구간에서 누락되지 않게.
            if hi == edges[-1]:
                mask = (confs >= lo) & (confs <= hi)
            else:
                mask = (confs >= lo) & (confs < hi)
            n = int(mask.sum())
            acc = float(corrects[mask].mean()) if n > 0 else float("nan")
            mean_conf = float(confs[mask].mean()) if n > 0 else float("nan")
            rows.append({"lo": lo, "hi": hi, "n": n, "acc": acc, "mean_conf": mean_conf})
        return rows

    def _percentile_bin_stats(confs, n_bins):
        # 경계를 percentile로 계산(같은 값이 몰려있으면 경계가 겹칠 수 있음
        # — 그 경우 일부 구간 n=0이 될 수 있고, 이 자체도 "분포가 얼마나
        # 뭉쳐있는가"를 보여주는 정보라 별도 보정 없이 그대로 둠).
        pct_edges = np.linspace(0, 100, n_bins + 1)
        conf_edges = np.percentile(confs, pct_edges)
        rows = []
        for i in range(n_bins):
            lo_pct, hi_pct = pct_edges[i], pct_edges[i + 1]
            lo_conf, hi_conf = conf_edges[i], conf_edges[i + 1]
            if i == n_bins - 1:
                mask = (confs >= lo_conf) & (confs <= hi_conf)
            else:
                mask = (confs >= lo_conf) & (confs < hi_conf)
            n = int(mask.sum())
            acc = float(corrects[mask].mean()) if n > 0 else float("nan")
            # [추가] 이 구간 샘플들이 배정된 centroid의 평균 크기 — "routing
            # confidence가 높은/낮은 구간이 큰 centroid에 쏠려있는가"를 accuracy
            # 표와 나란히 바로 볼 수 있게(별도로 산점도를 그릴 필요 없이 1차 확인용).
            mean_centroid_size = (
                float(centroid_sizes[mask].mean())
                if n > 0 and sample_groups is not None else None
            )
            rows.append({"lo_pct": lo_pct, "hi_pct": hi_pct,
                         "lo_conf": float(lo_conf), "hi_conf": float(hi_conf),
                         "n": n, "acc": acc, "mean_centroid_size": mean_centroid_size})
        return rows

    routing_bins    = _percentile_bin_stats(routing_confs, n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prediction_bins = _fixed_bin_stats(pred_confs, edges)

    # ECE: 각 bin의 |accuracy - mean_confidence|를 bin 크기로 가중평균
    # (Guo et al. 2017, "On Calibration of Modern Neural Networks" 표준 정의)
    ece = sum(
        (b["n"] / n_total) * abs(b["acc"] - b["mean_conf"])
        for b in prediction_bins if b["n"] > 0
    )

    return {
        "routing_stats":    routing_stats,
        "routing_bins":     routing_bins,
        "prediction_bins":  prediction_bins,
        "prediction_ece":   float(ece),
        "n_total":          n_total,
        "overall_acc":      float(corrects.mean()),
        "correlations":     correlations,
        "centroid_train_stats":         centroid_train_stats,        # {centroid_idx: {size,purity,cohesion,gap,entropy}}
        "centroid_table":               centroid_table,  # [{centroid,train_count,test_count,purity,entropy,cohesion,test_accuracy}, ...] test_count=0 포함 전체
        "centroid_level_correlations":  centroid_level_correlations,  # size_vs_purity 등, centroid 하나당 값 하나 기준
        # [추가] 샘플 단위 원본 배열 — scatter plot이나 추가 상관분석을
        # 직접 해보고 싶을 때 재계산 없이 바로 쓸 수 있게. bin 통계로는 안
        # 보이는 패턴(예: 특정 몇 개 centroid만 문제인지 vs 전반적 현상인지)
        # 확인용.
        "per_sample": {
            "routing_confidence":    routing_confs.tolist(),
            "routing_margin":        margins.tolist(),
            "prediction_confidence": pred_confs.tolist(),
            "assigned_centroid":     assigned_centroids.tolist(),
            "centroid_size":         centroid_sizes.tolist(),
            "centroid_purity":       centroid_purities.tolist(),   # train 기준, X_train 없으면 전부 NaN
            "centroid_cohesion":     centroid_cohesions.tolist(),  # 위와 동일 조건
            "correct":               corrects.tolist(),
        },
    }


def print_calibration_analysis(result: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Calibration Analysis (test set, n={result['n_total']})")
    print(f"{'='*60}")
    print(f"  Overall accuracy: {result['overall_acc']:.1%}")
    print(f"  Prediction ECE (Expected Calibration Error): {result['prediction_ece']:.4f}")
    print(f"  (ECE가 낮을수록 'confidence만큼 실제로 맞는다'는 뜻 — 0에 가까울수록 잘 보정됨,")
    print(f"   보통 0.05 미만이면 양호, 0.15 이상이면 상당히 overconfident/underconfident로 봄)")

    rs = result["routing_stats"]
    print(f"\n  Routing confidence 분포 (n_prototypes={rs['n_prototypes']}, "
          f"균등분포 기준선={rs['uniform_baseline']:.2%} — 이보다 훨씬 높으면 routing이 실제로 "
          f"특정 centroid에 쏠려 있다는 뜻, 비슷하면 사실상 균등에 가깝다는 뜻):")
    print(f"    mean={rs['mean']:.2%}  median={rs['median']:.2%}  std={rs['std']:.2%}  "
          f"min={rs['min']:.2%}  max={rs['max']:.2%}  p90={rs['p90']:.2%}  p99={rs['p99']:.2%}")
    print(f"  (절대 % 구간이 아니라 percentile로 나눔 — routing_confidence의 유의미한 스케일이")
    print(f"   n_prototypes에 구조적으로 종속적이라, 고정 % 구간은 P가 다르면 비교가 안 됨)")

    print(f"\n  Effective prototype 수 (N_eff = exp(entropy) — 균등분포면 {rs['n_prototypes']}, "
          f"한 centroid로만 쏠리면 1):")
    if rs.get("train_n_eff") is not None:
        print(f"    train 기준 N_eff = {rs['train_n_eff']:.1f}  (전체 train 샘플의 실제 분포 기준)")
    print(f"    test 기준  N_eff = {rs['test_n_eff']:.1f}  (이번 run의 test set이 실제로 도달한 분포)")
    print(f"  ('alive' centroid 수(위 [Regroup] 로그)와 다른 개념 — alive는 '죽지 않은' 것,")
    print(f"   N_eff는 '실제로 traffic을 고르게 나눠 받는 정도'. alive는 큰데 N_eff가 훨씬 작으면")
    print(f"   '살아는 있지만 소수만 일하고 있다'는 뜻.)")

    print(f"\n  {'Routing confidence (percentile)':<34s}{'conf range':<20s}{'n':>6s}{'accuracy':>10s}{'mean centroid_size':>20s}")
    for b in result["routing_bins"]:
        acc_str = f"{b['acc']:.1%}" if b["n"] > 0 else "  n/a"
        range_str = f"{b['lo_conf']:.2%}-{b['hi_conf']:.2%}"
        size_str = f"{b['mean_centroid_size']:.0f}" if b["n"] > 0 and b["mean_centroid_size"] is not None else "  n/a"
        print(f"  {b['lo_pct']:>3.0f}–{b['hi_pct']:>3.0f}pct{'':<20s}{range_str:<20s}{b['n']:>6d}{acc_str:>10s}{size_str:>20s}")

    print(f"\n  {'Prediction confidence':<24s}{'n':>8s}{'accuracy':>12s}{'mean conf':>12s}")
    for b in result["prediction_bins"]:
        lo_pct, hi_pct = int(b["lo"] * 100), int(b["hi"] * 100)
        acc_str  = f"{b['acc']:.1%}" if b["n"] > 0 else "  n/a"
        conf_str = f"{b['mean_conf']:.1%}" if b["n"] > 0 else "  n/a"
        print(f"  {lo_pct:>3d}–{hi_pct:>3d}%{'':<16s}{b['n']:>8d}{acc_str:>12s}{conf_str:>12s}")

    corr = result["correlations"]
    print(f"\n  Spearman 상관 (전체 test set 기준, sample 단위):")
    print(f"    routing_confidence vs prediction_confidence : {corr['routing_vs_prediction_confidence']:+.3f}")
    print(f"    routing_confidence vs correct(0/1)          : {corr['routing_vs_correct']:+.3f}")
    print(f"    routing_margin(top1-runnerup1) vs correct   : {corr['routing_margin_vs_correct']:+.3f}")
    print(f"    centroid_size vs correct(0/1)               : {corr['centroid_size_vs_correct']:+.3f}")
    if "centroid_purity_vs_correct" in corr:
        print(f"    centroid_purity(train) vs correct(0/1)      : {corr['centroid_purity_vs_correct']:+.3f}")
    if "centroid_cohesion_vs_correct" in corr:
        print(f"    centroid_cohesion(train) vs correct(0/1)    : {corr['centroid_cohesion_vs_correct']:+.3f}")
    print(f"  (routing_confidence vs correct가 뚜렷이 음수면 'routing이 확신할수록 오히려 더 틀린다'는")
    print(f"   뜻이고, centroid_size/purity/cohesion vs correct가 뚜렷하면 그 centroid 속성이 예측")
    print(f"   실패와 관련 있다는 뜻 — 다만 이 상관계수 하나로 인과를 단정할 수 없음, per_sample")
    print(f"   원본 배열로 직접 산점도를 그려보는 걸 권장.)")

    ct = result.get("centroid_table", [])
    if ct:
        n_zero_test = sum(1 for r in ct if r["test_count"] == 0)
        print(f"\n  Centroid별 train/test 활용 비교 ({len(ct)}개 centroid에 train 샘플이 있음, "
              f"그중 {n_zero_test}개는 test 샘플이 0개 배정됨):")
        print(f"  [주의] test_count=0인 centroid가 많다고 바로 '죽었다'고 단정하지 말 것 — 이 태스크")
        print(f"   자체가 소수 영역에 자연스럽게 집중되는 구조일 수도 있음. train_count도 같이 작은지")
        print(f"   (즉 애초에 학습 때도 거의 안 쓰였는지) 비교해서 판단할 것.")
        _top = sorted(ct, key=lambda r: -r["train_count"])[:15]
        print(f"\n  {'Centroid':<10}{'train_n':>9}{'test_n':>8}{'purity':>9}{'entropy':>9}{'cohesion':>10}{'test_acc':>10}")
        for r in _top:
            purity_str = f"{r['purity']:.1%}" if r['purity'] is not None else "  n/a"
            entropy_str = f"{r['entropy']:.3f}" if r['entropy'] is not None else "  n/a"
            acc_str = f"{r['test_accuracy']:.1%}" if r['test_accuracy'] is not None else "  n/a"
            print(f"  Centroid_{r['centroid']:<4}{r['train_count']:>9}{r['test_count']:>8}"
                  f"{purity_str:>9}{entropy_str:>9}{r['cohesion']:>10.4f}{acc_str:>10}")
        print(f"  (train_count 기준 상위 15개만 표시 — 전체는 result['centroid_table']에 있음)")

    clc = result.get("centroid_level_correlations", {})
    if clc:
        print(f"\n  Spearman 상관 (centroid 단위 — centroid 하나당 값 하나, n_centroids={clc.get('n_centroids', '?')}):")
        print(f"  [주의] n_centroids가 작으면(예: 10개 미만) 아래 상관계수는 표본이 매우 작아 신뢰구간이")
        print(f"   넓음 — 극단적인 값이 나와도 과대 해석하지 말 것.")
        if "size_vs_purity" in clc:
            print(f"    size vs purity            : {clc['size_vs_purity']:+.3f}  "
                  f"(음수면 '클수록 순도가 낮다' — 사용자 가설의 첫 단계)")
        if "purity_vs_test_accuracy" in clc:
            print(f"    purity vs test_accuracy   : {clc['purity_vs_test_accuracy']:+.3f}  "
                  f"(양수면 '순도 높은 centroid일수록 실제로 test에서도 잘 맞는다')")
        if "cohesion_vs_test_accuracy" in clc:
            print(f"    cohesion vs test_accuracy : {clc['cohesion_vs_test_accuracy']:+.3f}")
        print(f"  (이 셋이 전부 예상 방향(size↔purity 음수, purity/cohesion↔accuracy 양수)이면")
        print(f"   '큰 centroid → 순도 낮음 → 예측 실패' 경로가 centroid 단위에서도 일관되게 지지됨)")

    # [수정] "평평하면 좋다"고 무조건 단정하지 않음 — 실제로 accuracy가
    # percentile에 따라 단조롭지 않은(특히 상위 percentile에서 급락하는)
    # 경우가 실측으로 확인된 바 있어서, 그 경우에는 원인을 안다고 주장하지
    # 않고 다음에 뭘 봐야 하는지만 안내.
    accs = [b["acc"] for b in result["routing_bins"] if b["n"] > 0 and not np.isnan(b["acc"])]
    is_monotonic_nondecreasing = all(a <= b + 0.03 for a, b in zip(accs, accs[1:]))  # 3%p 여유
    max_drop = max((accs[i] - accs[i+1] for i in range(len(accs)-1)), default=0.0)

    print(f"\n  해석:")
    if is_monotonic_nondecreasing and max_drop < 0.05:
        print(f"    - Routing confidence percentile 구간별 accuracy가 대체로 평평하거나 단조 증가 —")
        print(f"      retrieval/fusion이 routing의 상대적 불확실성을 실제로 보완하고 있다는 근거와")
        print(f"      일관됨(다만 이 지표 하나로 인과를 증명하는 건 아님).")
    else:
        print(f"    - Routing confidence percentile 구간별 accuracy가 단조롭지 않음(최대 낙폭 "
              f"{max_drop:.1%}p). 이것만으로는 원인을 알 수 없음 — 다음 중 하나 이상일 수 있음:")
        print(f"        1) 특정(주로 크고 순도 낮은) centroid에 상위 percentile 샘플이 몰려있음")
        print(f"           → mean centroid_size 열과 centroid_size vs correct 상관 확인")
        print(f"        2) routing과 최종 예측이 서로 다른 정보를 봐서 일관되지 않음")
        print(f"           → routing_confidence vs prediction_confidence 상관 확인")
        print(f"        3) 이 test set/seed 하나의 우연(표본 크기·학습 불안정성 등)")
        print(f"           → 다른 --train_seed로 재현되는지 확인")
        print(f"      섣불리 하나로 단정하지 말 것 — per_sample 배열로 직접 파봐야 함.")
    print(f"    - Prediction confidence 구간의 accuracy가 mean_conf보다 뚜렷이 낮다")
    print(f"      (특히 80-100% 구간) → overconfidence, calibration이 나쁘다는 뜻.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────



def run_single_seed(
    dataset, X_train, y_train, X_val, y_val, X_test, y_test, y_std,
    output_dim, tasktype, openml_id, dataset_info, device, log_dir, env_info,
    args, train_seed, do_analysis,
):
    """dataset/HPO study(둘 다 train_seed와 무관 — main()에서 한 번만 로드해서
    넘겨받음)를 갖고 이 train_seed 하나로 학습·평가·(선택)분석까지 수행.

    [배경] optimize.py는 dataset을 한 번만 로드해서 100개 trial이 재사용하는
    구조인데(objective() 밖에서 로드), reproduce.py는 예전엔 매 프로세스 실행마다
    (=seed 하나 돌릴 때마다) dataset을 처음부터 다시 로드했음 — openml fetch/
    NaN 전처리/StratifiedKFold/QuantileTransformer 비용이 --train_seeds로 5번
    돌리면 5번 다 실렸음. 이 함수로 그 로직(원래 main() 안에 인라인으로 있던
    ~2400줄)을 그대로 옮겨서, main()이 dataset/study를 한 번만 로드하고 이
    함수를 seed 개수만큼 호출하는 구조로 바꿈 — optimize.py와 같은 패턴.

    do_analysis : --explain/--calibration_analysis/--linear_probe(켜져 있는
      것들)를 이 seed에서 실제로 실행할지. --train_seeds 여러 개를 돌 때 전부
      켜두면 로그가 seed 수만큼 불어나므로, main()이 --explain_seed(기본값:
      마지막 seed)와 비교해서 이 값을 결정해 넘겨준다.

    반환: {"train_seed": train_seed, "val_metrics": dict, "test_metrics": dict}
      — main()이 --train_seeds가 2개 이상이면 이 반환값들을 모아 mean±std로
      요약 출력함.
    """
    # [이동됨] 예전엔 main()에서 dataset 로딩 전에(train_seed 계산 직후) 호출됐음 —
    # dataset 로딩 자체는 train_seed와 무관해서 여기(함수 진입 시점)로 옮겨도
    # 동작은 완전히 동일함(오히려 "이 함수가 호출될 때마다 이 seed로 다시 씨드한다"는
    # 의미가 더 명확해짐).
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if len(getattr(args, '_train_seed_list', [train_seed])) > 1 or train_seed != args.seed:
        print(f"  [train_seed={train_seed}] 학습 초기화/배치 순서 seed (데이터 분할은 --seed={args.seed} 그대로)")

    _save_tag = ("..detach_ctx" if args.detach_context_grad else "") \
              + (f"..qDetachWarmupE{args.query_detach_warmup_epochs}" if args.query_detach_warmup_epochs > 0 else "") \
              + (f"..qDetachWarmupS{args.query_detach_warmup_steps}" if args.query_detach_warmup_steps > 0 else "") \
              + ("..confscale" if args.confidence_scaling else "") \
              + ("..confscale_detach" if (args.confidence_scaling and args.confidence_scaling_detach) else "") \
              + ("..no_query_emb" if args.no_query_emb else "") \
              + ("..no_context_emb" if args.no_context_emb else "") \
              + ("..ema_codebook" if args.ema_codebook else "") \
              + (f"..ema_decay{args.ema_decay_override:g}" if args.ema_decay_override is not None else "") \
              + ("..blockLN" if args.blockwise_layernorm else "") \
              + ("..branchL2norm" if args.head_branch_l2norm else "") \
              + ("..fusion_residual" if args.fusion_mode == "residual" else "") \
              + ("..fusion_concat" if args.fusion_mode == "concat" else "") \
              + ("..fusion_gatedsum" if args.fusion_mode == "gated_sum" else "") \
              + ("..fusion_anchorgate" if args.fusion_mode == "anchor_gate" else "") \
              + ("..fusion_ctxgatedbeta" if args.fusion_mode == "context_gated_beta" else "") \
              + ("..no_retrieval" if args.disable_retrieval_branch else "") \
              + (f"..gateT{args.fusion_gate_temperature:g}" if args.fusion_gate_temperature != 1.0 else "") \
              + ("..allowSelfRet" if args.allow_self_retrieval else "") \
              + (f"..valMode_{args.value_mode}" if args.value_mode != "default" else "") \
              + (f"..nbrInt_{args.neighbor_interaction_mode}" if args.neighbor_interaction_mode is not None else "") \
              + (f"..nbrHeads{args.interaction_n_heads}" if args.interaction_n_heads != 2 else "") \
              + (f"..aggMode_{args.aggregator_mode}" if args.aggregator_mode != "pooling" else "") \
              + (f"..headAlpha{args.head_attn_alpha_override}" if args.head_attn_alpha_override is not None else "") \
              + (f"..headNbrSrc_{args.head_neighbor_source}" if args.head_neighbor_source != "real" else "") \
              + (f"..fa{args.fusion_alpha_override:g}" if args.fusion_alpha_override is not None else "") \
              + (f"..fb{args.fusion_beta_override:g}" if args.fusion_beta_override is not None else "") \
              + ("..freezeHead" if args.freeze_encoder_retrain_head else "") \
              + ("..ctx_proj" if args.context_projection else "") \
              + ("..cat_concat" if args.cat_combine == "concat" else "") \
              + ("..cat_onehot" if args.cat_combine == "onehot" else "") \
              + ("..num_ple" if args.num_embedding == "ple" else "") \
              + ("..num_plr" if args.num_embedding == "plr_lite" else "") \
              + (f"..lcb{args.loss_codebook_override:g}" if args.loss_codebook_override is not None else "") \
              + (f"..lcm{args.loss_commitment_override:g}" if args.loss_commitment_override is not None else "") \
              + (f"..ldv{args.loss_diversity_override:g}" if args.loss_diversity_override is not None else "") \
              + (f"..ed{args.embed_dim_override}" if args.embed_dim_override is not None else "") \
              + (f"..do{args.dropout_override:g}" if args.dropout_override is not None else "") \
              + (f"..evT{args.evidence_temperature_override:g}" if args.evidence_temperature_override is not None else "") \
              + (f"..evM_{args.evidence_metric_override}" if args.evidence_metric_override is not None
                 else (f"..evM_{args.evidence_metric}" if args.evidence_metric != "euclidean" else "")) \
              + (f"..bs{args.batch_size_override}" if args.batch_size_override is not None else "") \
              + (f"..rwe{args.regroup_warmup_epochs_override}" if args.regroup_warmup_epochs_override is not None else "") \
              + (f"..drp{args.dead_reinit_patience_override}" if args.dead_reinit_patience_override is not None else "") \
              + (f"..drn{args.dead_reinit_noise_scale_override:g}" if args.dead_reinit_noise_scale_override is not None else "") \
              + (f"..trainseed{train_seed}" if train_seed != args.seed else "") \
              + ("..deterministic" if args.deterministic else "") \
              + (f"..{args.run_tag}" if args.run_tag is not None else "")

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
        if args.loss_codebook_override is not None:
            print(f"  ⚠️  --loss_codebook_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.batch_size_override is not None:
            print(f"  ⚠️  --batch_size_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.regroup_warmup_epochs_override is not None:
            print(f"  ⚠️  --regroup_warmup_epochs_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.dead_reinit_patience_override is not None:
            print(f"  ⚠️  --dead_reinit_patience_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.dead_reinit_noise_scale_override is not None:
            print(f"  ⚠️  --dead_reinit_noise_scale_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.dropout_override is not None:
            print(f"  ⚠️  --dropout_override는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.train_seed is not None:
            print(f"  ⚠️  --train_seed는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.deterministic:
            print(f"  ⚠️  --deterministic은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 재학습을 안 하므로 이 플래그를 무시합니다.")
        if args.no_query_emb:
            print(f"  ⚠️  --no_query_emb는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(head 입력 차원 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다.")
        if args.no_context_emb:
            print(f"  ⚠️  --no_context_emb는 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(head 입력 차원 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다.")
        if args.ema_codebook:
            print(f"  ⚠️  --ema_codebook은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(EMA 사용 여부 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다(체크포인트 자체가 EMA로 "
                  f"학습됐다면 자동으로 EMA 구조로 복원됩니다).")
        if args.blockwise_layernorm:
            print(f"  ⚠️  --blockwise_layernorm은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(head LayerNorm 구조 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다(체크포인트가 이 구조로 학습됐다면 "
                  f"자동으로 복원됩니다. 반대로 결합형 LayerNorm으로 저장된 체크포인트에 "
                  f"이 플래그를 켜도 state_dict 모양이 달라 로드 자체는 저장된 구조를 "
                  f"따르므로 문제없음).")
        if args.head_branch_l2norm:
            print(f"  ⚠️  --head_branch_l2norm은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(head 구조 포함)를 그대로 "
                  f"쓰므로 이 플래그를 무시합니다 — blockwise_layernorm과 같은 이유.")
        if args.fusion_mode in ("residual", "gated_sum", "anchor_gate", "context_gated_beta"):
            print(f"  ⚠️  --fusion_mode {args.fusion_mode}은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(head fusion 구조 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다(체크포인트가 이 모드로 학습됐다면 "
                  f"자동으로 복원됩니다).")
    else:
        # [수정] optimize.py가 실제로 저장한 파일명과 일치시키기 위해
        # study_pkl_tag()를 그대로 재사용 — 예전엔 여기서 태그 없이
        # "data={id}..model=tabera.pkl"로 고정해뒀는데, optimize.py의
        # --num_embedding 기본값이 ple로 바뀌면서 실제 저장 파일명엔
        # "..num_ple"이 붙어 조용히 어긋나는 사고가 났음(FileNotFoundError).
        # no_offset_correction/global_retrieve는 reproduce.py에 CLI 플래그
        # 자체가 없음(이미 "채택 확정"돼 하드코딩된 값 — 아래 meta 저장부의
        # use_offset_correction=True/global_retrieve=False와 동일) — 그래서
        # 여기도 같은 고정값(False, False)으로 명시.
        _study_tag = study_pkl_tag(
            no_offset_correction=False,
            global_retrieve=False,
            detach_context_grad=args.detach_context_grad,
            context_projection=args.context_projection,
            cat_combine=args.cat_combine,
            num_embedding=args.num_embedding,
            evidence_metric=args.evidence_metric,
            fusion_mode=args.fusion_mode,
            use_context_emb=not args.no_context_emb,
            disable_retrieval_branch=args.disable_retrieval_branch,
        )
        fname = os.path.join(log_dir, f"data={openml_id}{_study_tag}..model=tabera.pkl")
        if not os.path.exists(fname):
            _hint_flags = ""
            if args.num_embedding != "ple":
                _hint_flags += f" --num_embedding {args.num_embedding}"
            if args.cat_combine != "onehot":
                _hint_flags += f" --cat_combine {args.cat_combine}"
            if args.detach_context_grad:
                _hint_flags += " --detach_context_grad"
            if args.context_projection:
                _hint_flags += " --context_projection"
            if args.fusion_mode != "concat":
                _hint_flags += f" --fusion_mode {args.fusion_mode}"
            if args.no_context_emb:
                _hint_flags += " --no_context_emb"
            _hint_cmd = f"optimize.py --openml_id {openml_id} --seed {args.seed}{_hint_flags}"
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
        # [추가] batch_size가 더 이상 trial.suggest_*가 아니라 상수(256)라
        # study.best_params에 이 키 자체가 없음 — k/routing_scale과 같은
        # 문제. .setdefault()로 채움: 구버전 study(batch_size가 실제로
        # 탐색된 경우)는 이미 키가 있으니 그 값 그대로 보존, 신규 study는
        # 여기서 256으로 채워짐.
        best_params.setdefault("batch_size", 256)
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
        # [추가] cat_combine/num_embedding과 같은 성격 — best_params에는 없는
        # 구조 선택이라 여기서 명시적으로 채움. --evidence_metric_override가
        # 뒤에서 더 우선시되도록 이 대입이 먼저 와야 함(순서 중요).
        model_kwargs["evidence_metric"] = args.evidence_metric
        if args.evidence_metric != "euclidean":
            print(f"  [--evidence_metric] evidence_metric={args.evidence_metric} "
                  f"(이 값으로 HPO된 study를 불러옴 — study_pkl_tag 참고)")
        if args.loss_commitment_override is not None:
            _old_commitment_w = model_kwargs.get("loss_weights", {}).get("commitment", 0.0)
            model_kwargs.setdefault("loss_weights", {})["commitment"] = args.loss_commitment_override
            best_params["loss_commitment"] = args.loss_commitment_override
            print(f"  [--loss_commitment_override] loss_weights['commitment']: "
                  f"{_old_commitment_w:.4g} → {args.loss_commitment_override:.4g} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.loss_diversity_override is not None:
            _old_diversity_w = model_kwargs.get("loss_weights", {}).get("diversity", 0.0)
            model_kwargs.setdefault("loss_weights", {})["diversity"] = args.loss_diversity_override
            best_params["loss_diversity"] = args.loss_diversity_override
            print(f"  [--loss_diversity_override] loss_weights['diversity']: "
                  f"{_old_diversity_w:.4g} → {args.loss_diversity_override:.4g} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.loss_codebook_override is not None:
            # [통제 실험용] best_params가 찾은 loss_codebook 값(있다면)을
            # 무시하고 이 값으로 강제 — 나머지 하이퍼파라미터는 best_params
            # 그대로라, 이 값 하나만 바꿔가며 재학습해서 codebook_loss
            # 자체의 효과와 "HPO가 다른 조합에 우연히 정착한 것"을 분리
            # 검증할 수 있음.
            _old_codebook_w = model_kwargs.get("loss_weights", {}).get("codebook", 0.0)
            model_kwargs.setdefault("loss_weights", {})["codebook"] = args.loss_codebook_override
            best_params["loss_codebook"] = args.loss_codebook_override  # 저장/재출력 시 실제 학습값과 일치하도록
            print(f"  [--loss_codebook_override] loss_weights['codebook']: "
                  f"{_old_codebook_w:.4g} → {args.loss_codebook_override:.4g} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.embed_dim_override is not None:
            # [통제 실험용] embed_dim만 격리해서 바꿈 — cosine HPO가 embed_dim과
            # 동시에 바꾼 dropout/lr/layers/loss weight는 best_params 그대로 둠.
            # 모델 구조(가중치 shape)가 바뀌므로 재학습이 반드시 필요(로드 불가).
            _old_embed_dim = model_kwargs.get("embed_dim")
            model_kwargs["embed_dim"] = args.embed_dim_override
            best_params["embed_dim"] = args.embed_dim_override
            print(f"  [--embed_dim_override] embed_dim: {_old_embed_dim} → {args.embed_dim_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.dropout_override is not None:
            # [통제 실험용] dropout은 TabularEmbedder(ResidualMLP) 내부에서
            # query_emb 자체를 매 forward마다 흔드는 유일한 확률적 요소라,
            # 라우팅 churn(연속 dead/reinit)의 원인 후보로 지목됨 — 검증
            # 안 된 가설이라 나머지는 그대로 두고 이 값 하나만 바꿔 재학습.
            _old_dropout = model_kwargs.get("dropout")
            model_kwargs["dropout"] = args.dropout_override
            best_params["dropout"] = args.dropout_override  # 저장/재출력 시 실제 학습값과 일치하도록
            print(f"  [--dropout_override] dropout: {_old_dropout} → {args.dropout_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.evidence_temperature_override is not None:
            # [통제 실험용] AttentionAggregator의 evidence_w softmax temperature.
            # best_params에는 애초에 없는 값(HPO 탐색 대상 아님, 기본 1.0)이라
            # dropout_override와 달리 "덮어쓸 기존 값"이 없음 — model_kwargs에
            # 직접 새로 설정.
            model_kwargs["evidence_temperature"] = args.evidence_temperature_override
            print(f"  [--evidence_temperature_override] evidence_temperature: "
                  f"1.0(기본값) → {args.evidence_temperature_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.evidence_metric_override is not None:
            model_kwargs["evidence_metric"] = args.evidence_metric_override
            print(f"  [--evidence_metric_override] evidence_metric: "
                  f"euclidean(기본값) → {args.evidence_metric_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.batch_size_override is not None:
            # [통제 실험용] batch_size는 model_kwargs가 아니라 best_params
            # (=TabERAWrapper.params, 학습 루프의 self.params["batch_size"])
            # 로만 흘러가므로 model_kwargs는 안 건드림 — 모델 구조는 그대로.
            _old_batch_size = best_params.get("batch_size")
            best_params["batch_size"] = args.batch_size_override
            print(f"  [--batch_size_override] batch_size: {_old_batch_size} → "
                  f"{args.batch_size_override} (나머지 파라미터는 best_params 그대로)")
        if args.regroup_warmup_epochs_override is not None:
            # [통제 실험용] CentroidLayer 생성자 파라미터라 model_kwargs에
            # 반영 — dropout_override와 같은 위치(모델 구조 파라미터).
            _old_warmup = model_kwargs.get("regroup_warmup_epochs", 0)
            model_kwargs["regroup_warmup_epochs"] = args.regroup_warmup_epochs_override
            print(f"  [--regroup_warmup_epochs_override] regroup_warmup_epochs: "
                  f"{_old_warmup} → {args.regroup_warmup_epochs_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.dead_reinit_patience_override is not None:
            _old_patience = model_kwargs.get("dead_reinit_patience", 5)
            model_kwargs["dead_reinit_patience"] = args.dead_reinit_patience_override
            print(f"  [--dead_reinit_patience_override] dead_reinit_patience: "
                  f"{_old_patience} → {args.dead_reinit_patience_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        if args.dead_reinit_noise_scale_override is not None:
            _old_noise_scale = model_kwargs.get("dead_reinit_noise_scale", 0.01)
            model_kwargs["dead_reinit_noise_scale"] = args.dead_reinit_noise_scale_override
            print(f"  [--dead_reinit_noise_scale_override] dead_reinit_noise_scale: "
                  f"{_old_noise_scale} → {args.dead_reinit_noise_scale_override} "
                  f"(나머지 파라미터는 best_params 그대로)")
        model_kwargs.update(dict(
            # [수정] optimize.py와 동일하게 캡 제거 (memory_size가 다르면
            # HPO 때 찾은 best_params가 이 재현 실행에서 재현되지 않음)
            memory_size=len(y_train),
            # [재개] --no_offset_correction ablation으로 한 번 검증 완료돼
            # "더 이상 옵션으로 안 둔다"고 닫았던 결정을, 이번 value ablation
            # 실험(diagnose_value_components 실측 — T(query-neighbour) 항이
            # label_emb보다 평균 4.9배 크다는 게 확인됨)을 위해 의식적으로
            # 다시 연다. --value_mode로 통제.
            use_offset_correction=(args.value_mode != "label_only"),
            global_retrieve=False,
            use_context_emb=not args.no_context_emb,
            use_query_emb_in_head=not args.no_query_emb,
            use_ema_codebook=args.ema_codebook,
            ema_decay=args.ema_decay_override if args.ema_decay_override is not None else 0.99,
            value_mode=("default" if args.value_mode in ("default", "label_only") else args.value_mode),
            neighbor_interaction_mode=args.neighbor_interaction_mode,
            interaction_n_heads=args.interaction_n_heads,
            aggregator_mode=args.aggregator_mode,
            head_attn_alpha_override=args.head_attn_alpha_override,
            head_neighbor_source=args.head_neighbor_source,
            blockwise_layernorm=args.blockwise_layernorm,
            head_branch_l2norm=args.head_branch_l2norm,
            fusion_mode=args.fusion_mode,
            disable_retrieval_branch=args.disable_retrieval_branch,
            exclude_self_retrieval=(not args.allow_self_retrieval),
            fusion_alpha_override=args.fusion_alpha_override,
            fusion_beta_override=args.fusion_beta_override,
            fusion_gate_temperature=args.fusion_gate_temperature,
            detach_context_grad=args.detach_context_grad,
            # [구조 조정] context_emb를 head 직전 Linear 프로젝션에 통과시킴
            use_context_projection=args.context_projection,
            # [진단용] head concat 직전 브랜치별 gradient 계측 — state_dict
            # 구조는 안 바꾸지만(detach_context_grad와 같은 이유로) 다른
            # TabERA 동작 플래그들과 같은 자리에 모아두는 게 일관적이라
            # 여기 합류시킴.
            log_branch_gradients=args.log_branch_gradients,
            # [진단용] context_emb를 head에 넣기 전 assignment confidence로
            # 스케일 — 라우팅/검색은 안 건드림.
            use_confidence_scaling=args.confidence_scaling,
            confidence_scaling_detach=args.confidence_scaling_detach,
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
        regroup_log_every=args.regroup_log_every,
        refresh_on_best=args.refresh_on_best,
        log_branch_gradients=args.log_branch_gradients,
        log_branch_gradients_first_n_epochs=args.log_branch_gradients_first_n_epochs,
        log_evidence_stats=args.log_evidence_stats,
        log_fusion_trajectory=args.log_fusion_trajectory,
        log_centroid_label_mi_trajectory=args.log_centroid_label_mi_trajectory,
        log_shuffle_ablation_trajectory=args.log_shuffle_ablation_trajectory,
        log_representation_drift_trajectory=args.log_representation_drift_trajectory,
        query_detach_warmup_epochs=args.query_detach_warmup_epochs,
        query_detach_warmup_steps=args.query_detach_warmup_steps,
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
            # [하위 호환] 예전 체크포인트는 (store, ptr, filled) 3-tuple —
            # sample_ids가 없으면 전부 -1(미확인)로 채움. 이 경우
            # dual_space_faithfulness의 ID 비교는 "확인 불가"로 표시됨.
            if len(fs_state) == 4:
                store, ptr, filled, sample_ids = fs_state
            else:
                store, ptr, filled = fs_state
                sample_ids = torch.full((model.feature_store.max_size,), -1, dtype=torch.long)
                print(f"  ⚠️  저장된 feature_store_state에 sample_ids가 없습니다 — "
                      f"이전 버전 체크포인트로 보입니다. ID 기반 검증은 건너뜁니다.")
            model.feature_store._store       = store.to(device)
            model.feature_store._ptr         = ptr
            model.feature_store._filled      = filled
            model.feature_store._sample_ids  = sample_ids.to(device)
        if model.prototype_layer.sample_groups is None:
            print(f"  ⚠️  저장된 state에 sample_groups가 없습니다 — 이 파일은 이번"
                  f" --from_saved_state 지원 이전 버전으로 저장된 것 같습니다."
                  f" group-constrained 검색/①②가 제대로 안 나올 수 있습니다.")
        # [방법2 fallback] 저장 당시 --refresh_on_best가 꺼져 있었거나(기본값)
        # 이전 버전 체크포인트라 memory.keys가 여전히 noisy할 수 있음 —
        # 이번 실행에서 --refresh_on_best를 켰다면 로드 직후 여기서 한 번
        # 실행. 저장 당시 이미 refresh된 상태였다면 keys를 다시 같은 값으로
        # 덮어쓸 뿐이라 안전(no-op에 가까움).
        if args.refresh_on_best:
            refresh_stats = model.refresh_memory_keys()
            if refresh_stats is not None:
                print(f"  [--refresh_on_best] memory.keys {refresh_stats['n_refreshed']}개 "
                      f"슬롯을 frozen weight로 재계산 완료")
                regroup_stats = wrapper._resync_groups_after_refresh()
                if regroup_stats is not None:
                    print(f"  [--refresh_on_best] clean 임베딩 기준으로 sample_groups 재동기화 "
                          f"완료 (active={regroup_stats.get('active_ratio', 0)*100:.0f}%, "
                          f"reinit={regroup_stats.get('reinit_count', 0)})")
        if args.freeze_encoder_retrain_head:
            # ── 인코더 고정 + head만 재학습 ──────────────────────
            HEAD_MODULE_NAMES = ("head", "head_query_ln", "head_context_ln",
                                  "head_agg_ln", "context_proj")
            n_frozen, n_trainable = 0, 0
            for _name, _p in model.named_parameters():
                _top = _name.split(".")[0]
                if _top in HEAD_MODULE_NAMES:
                    _p.requires_grad = True
                    n_trainable += _p.numel()
                else:
                    _p.requires_grad = False
                    n_frozen += _p.numel()
            # head를 백지로 재초기화 — "기존 head를 이어서 미세조정"이 아니라
            # "고정된 인코더 표현 위에서 head가 처음부터 그 정보를 쓰는 법을
            # 배울 수 있는가"를 순수하게 보기 위함(기존 head 가중치가 이미
            # query-only 지역 최적점에 있으면 거기서 못 벗어날 수 있으므로).
            for _mod_name in HEAD_MODULE_NAMES:
                _mod = getattr(model, _mod_name, None)
                if _mod is None:
                    continue
                for _m in _mod.modules():
                    if hasattr(_m, "reset_parameters"):
                        _m.reset_parameters()
            print(f"  [--freeze_encoder_retrain_head] 인코더 고정(파라미터 {n_frozen:,}개, "
                  f"gradient 차단) — head 계열만 재초기화 후 재학습(파라미터 {n_trainable:,}개, "
                  f"{args.freeze_head_epochs} epoch)")
            wrapper.epochs = args.freeze_head_epochs
            wrapper.fit(X_train, y_train, X_val, y_val, skip_centroid_init=True)
        else:
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

        # ── centroid_geometry: cosine_similarity_matrix()를 실제로 노출 ──
        # (지금까지 정의만 되고 아무 데서도 안 쓰이던 진단 메서드)
        #
        # [설계 의도 반영] centroid끼리 가까운 것 자체는 버그가 아닐 수
        # 있음 — 하나의 매니폴드/자연 군집을 여러 centroid가 나눠서
        # 대표하도록(다중 커버리지) 의도적으로 설계된 것이라는 전제가
        # 있음. 그래서 이 진단은 "가까운 쌍 = 나쁨"으로 단정하지 않고,
        # 가까운 쌍을 찾은 뒤 그 둘의 target 구성(어떤 클래스/값을
        # 대표하는가)이 서로 같은지 다른지로 한 번 더 나눠서 본다:
        #   - 가깝고 target도 비슷함 → 의도한 대로 같은 영역을 일관되게
        #     나눠 대표하는 것(다중 커버리지, 정상)
        #   - 가깝지만 target이 다름 → 같은 embedding 위치에서 서로 다른
        #     이야기를 하는 centroid들이 경합 중이라는 뜻 — 그 경계에
        #     있는 쿼리의 confidence가 낮게 나오는 게 여기서 비롯될 수
        #     있고, 이 경우가 실제로 살펴볼 가치가 있는 케이스.
        elif args.ablation == "centroid_geometry":
            model.eval()
            P = model.prototype_layer.P
            sim_matrix = model.prototype_layer.cosine_similarity_matrix()  # (P, P), CPU

            print(f"\n  Centroid Geometry — cosine_similarity_matrix() 진단 (P={P})")
            print(f"  {'─'*60}")
            print(f"  centroid끼리 가까운 것 자체는 버그가 아닐 수 있음(하나의 매니폴드를")
            print(f"  여러 centroid가 나눠 대표하도록 설계됨) — 여기서는 '가까운 쌍'을 찾은 뒤,")
            print(f"  그 쌍의 target(대표 클래스/값) 구성이 같은지 다른지로 한 번 더 나눠서 봄.")

            sim_np = sim_matrix.numpy()
            off_diag_mask = ~np.eye(P, dtype=bool)
            off_diag_vals = sim_np[off_diag_mask]

            print(f"\n  [Off-diagonal 유사도 분포] (자기 자신 제외, {len(off_diag_vals)}개 쌍)")
            print(f"    mean={off_diag_vals.mean():.4f}  std={off_diag_vals.std():.4f}  "
                  f"median={np.median(off_diag_vals):.4f}  max={off_diag_vals.max():.4f}")

            # 상위 top_n_pairs개 가장 가까운 쌍 (i<j로 중복 제거)
            top_n_pairs = min(10, P * (P - 1) // 2)
            iu = np.triu_indices(P, k=1)
            pair_sims = sim_np[iu]
            top_idx = np.argsort(-pair_sims)[:top_n_pairs]

            target_labels = model.prototype_layer.target_labels
            labels_list    = model.prototype_layer.labels

            print(f"\n  [가장 가까운 centroid 쌍 top {top_n_pairs}]")
            print(f"  {'Pair':<20} {'cos_sim':>8}  {'같은 target?':<14}  {'세부'}")
            print(f"  {'─'*90}")

            same_target_count = 0
            diff_target_count = 0
            unknown_count      = 0

            for idx in top_idx:
                i, j = int(iu[0][idx]), int(iu[1][idx])
                s = float(pair_sims[idx])
                pair_name = f"{labels_list[i]}-{labels_list[j]}"

                ti = target_labels.get(i) if target_labels is not None else None
                tj = target_labels.get(j) if target_labels is not None else None

                if ti is None or tj is None:
                    verdict = "?(그룹 too small)"
                    detail  = ""
                    unknown_count += 1
                elif ti["kind"] == "classification":
                    if ti["top_class"] == tj["top_class"]:
                        verdict = "같음"
                        same_target_count += 1
                        detail = (f"둘 다 '{ti['top_class_name']}' "
                                  f"({ti['top_prop']:.0%} vs {tj['top_prop']:.0%})")
                    else:
                        verdict = "⚠️ 다름"
                        diff_target_count += 1
                        detail = (f"'{ti['top_class_name']}'({ti['top_prop']:.0%}) vs "
                                  f"'{tj['top_class_name']}'({tj['top_prop']:.0%})")
                else:  # regression
                    pdiff = abs(ti["percentile"] - tj["percentile"])
                    if pdiff < 20.0:
                        verdict = "비슷함"
                        same_target_count += 1
                    else:
                        verdict = "⚠️ 다름"
                        diff_target_count += 1
                    detail = (f"percentile {ti['percentile']:.0f} vs {tj['percentile']:.0f} "
                              f"(Δ{pdiff:.0f})")

                print(f"  {pair_name:<20} {s:>8.4f}  {verdict:<14}  {detail}")

            print(f"\n  [요약] 가까운 top {top_n_pairs}쌍 중: "
                  f"같은/비슷한 target {same_target_count}쌍, "
                  f"⚠️ 다른 target {diff_target_count}쌍, "
                  f"판단불가 {unknown_count}쌍")

            print(f"\n  [해석]")
            if diff_target_count == 0:
                print(f"  ✅ 가까운 centroid 쌍은 전부 같은/비슷한 target을 대표함 — ")
                print(f"    의도한 대로 하나의 영역을 여러 centroid가 일관되게 나눠 대표하는")
                print(f"    '다중 커버리지'로 보임. 이 경우 confidence가 낮게 나오는 건 버그가")
                print(f"    아니라, 애초에 여러 centroid가 같은 이야기를 하도록 설계된 결과일")
                print(f"    가능성이 큼.")
            else:
                print(f"  ⚠️  가까운 centroid 쌍 중 {diff_target_count}개가 서로 다른 target을")
                print(f"    대표함 — 이 쌍들 근처에 있는 쿼리는 'confidence는 낮은데 서사도")
                print(f"    갈리는' 진짜 애매한 케이스일 수 있음. 위 표에서 ⚠️ 표시된 쌍을 눈여겨")
                print(f"    볼 것 — 위 표에 나열된 ⚠️ 쌍들이 그 후보입니다.")

            # ── Query-Centroid 유사도: centroid끼리의 유사도와 나란히 비교 ──
            # centroid-centroid 유사도가 이미 압축돼 있다면(위 off-diagonal
            # 분포), 그게 이 embed_dim 공간 자체의 특성(고차원에서 cosine
            # similarity가 0 근처로 몰리는 현상)인지, 아니면 정말 query 쪽만
            # 특별히 애매한 것인지는 query-centroid 유사도를 직접 봐야
            # 구분됨. "가장 확실한 매칭"조차 이 공간에서 어디까지 올라가는지
            # 확인하는 게 핵심.
            print(f"\n  {'='*60}")
            print(f"  [Query-Centroid 유사도] — 위 centroid-centroid 유사도와 비교용")
            print(f"  {'='*60}")

            n_qc = X_test.shape[0]  # 전체 테스트셋 (비용이 forward 1회뿐이라 샘플링 불필요)
            _qc_batch = 256
            top1_sims  = []
            margins    = []  # top1 - top2 (라우팅이 얼마나 여유있게 갈렸는지)
            with torch.no_grad():
                c_norm_qc = F.normalize(model.prototype_layer.centroid_emb, dim=-1)  # (P, D)
                for start in range(0, n_qc, _qc_batch):
                    X_batch = X_test[start:start + _qc_batch]
                    q_norm_qc = F.normalize(model.embedder(X_batch), dim=-1)          # (b, D)
                    sim_qc = q_norm_qc @ c_norm_qc.T                                   # (b, P)
                    top2 = sim_qc.topk(min(2, P), dim=-1).values                       # (b, ≤2)
                    top1_sims.append(top2[:, 0].cpu())
                    if top2.shape[1] > 1:
                        margins.append((top2[:, 0] - top2[:, 1]).cpu())

            top1_sims = torch.cat(top1_sims).numpy()
            margins   = torch.cat(margins).numpy() if margins else np.array([])

            print(f"\n  [Top-1 query-centroid 유사도 분포] (n={n_qc}, raw cosine, scale/temperature 적용 전)")
            print(f"    mean={top1_sims.mean():.4f}  std={top1_sims.std():.4f}  "
                  f"median={np.median(top1_sims):.4f}")
            print(f"    min={top1_sims.min():.4f}  max={top1_sims.max():.4f}")

            print(f"\n  [Top1-Top2 margin 분포] (라우팅이 2등과 얼마나 벌어져 있는지)")
            print(f"    mean={margins.mean():.4f}  std={margins.std():.4f}  "
                  f"median={np.median(margins):.4f}  min={margins.min():.4f}")
            narrow_margin_ratio = float((margins < 0.01).mean())
            print(f"    margin<0.01인 샘플 비율: {narrow_margin_ratio:.1%} "
                  f"(1등·2등이 사실상 구분 안 되는 쿼리)")

            print(f"\n  {'─'*60}")
            print(f"  [Null 베이스라인] 완전 무작위(학습 전혀 안 된) centroid/query 벡터를")
            print(f"  같은 D/P/N 조건으로 50회 시뮬레이션 — '이 정도 구조는 학습 없이도")
            print(f"  나오는가'를 z-score로 직접 검정. (3배 임계값 같은 임의 배수 대신")
            print(f"  이 방식을 씀 — 실측으로 그 배수 판정이 SpeedDating에서 틀렸던 걸 확인함.)")

            D = model.prototype_layer.centroid_emb.shape[1]
            n_null_trials = 50
            null_top1_medians = np.empty(n_null_trials)
            null_margin_means = np.empty(n_null_trials)
            for _t in range(n_null_trials):
                _g = torch.Generator().manual_seed(args.seed * 1000 + _t)
                _q_null = F.normalize(torch.randn(n_qc, D, generator=_g), dim=-1)
                _c_null = F.normalize(torch.randn(P, D, generator=_g), dim=-1)
                _sim_null = _q_null @ _c_null.T
                _top2_null = _sim_null.topk(min(2, P), dim=-1).values
                null_top1_medians[_t] = _top2_null[:, 0].median().item()
                if _top2_null.shape[1] > 1:
                    null_margin_means[_t] = (_top2_null[:, 0] - _top2_null[:, 1]).mean().item()
                else:
                    null_margin_means[_t] = float("nan")

            null_top1_mean, null_top1_std = float(null_top1_medians.mean()), float(null_top1_medians.std())
            null_margin_mean, null_margin_std = float(np.nanmean(null_margin_means)), float(np.nanstd(null_margin_means))

            z_top1   = (float(np.median(top1_sims)) - null_top1_mean) / (null_top1_std + 1e-8)
            z_margin = (float(margins.mean()) - null_margin_mean) / (null_margin_std + 1e-8)

            print(f"\n  {'':<28} {'null(50회)':>16}  {'실측':>10}  {'z-score':>8}")
            print(f"  {'top1 유사도 median':<28} {null_top1_mean:>9.4f}±{null_top1_std:<5.4f}  "
                  f"{np.median(top1_sims):>10.4f}  {z_top1:>8.2f}")
            print(f"  {'margin(top1-top2) mean':<28} {null_margin_mean:>9.4f}±{null_margin_std:<5.4f}  "
                  f"{margins.mean():>10.4f}  {z_margin:>8.2f}")

            print(f"\n  [해석]")
            if z_margin < -2.0:
                print(f"  🔴 margin이 무작위 null보다 유의하게 '더 좁습니다'(z={z_margin:.2f}) —")
                print(f"    이건 단순히 '학습이 구조를 못 만들었다'가 아니라, 학습 과정이")
                print(f"    top1·top2를 오히려 무작위보다 더 가깝게 만들고 있다는 뜻입니다.")
                print(f"    (참고: 이번 실행의 routing_scale={model.prototype_layer.routing_scale:.2f}.")
                print(f"    routing_scale이 낮을 때 이 현상이 나온 사례가 있었지만, routing_scale이")
                print(f"    낮지 않은 데이터셋에서도 같은 현상이 재현된 바 있어 — 원인을 이것")
                print(f"    하나로 단정할 근거는 없습니다. 원인 미확정 — dropout, loss_diversity/")
                print(f"    commitment/codebook 배합 등 다른 요인이 섞여 있을 수 있습니다.)")
                print(f"    ⚠️ 이건 reproduce.py(추론 전용)에서 post-hoc으로 못 고칩니다 —")
                print(f"    이미 학습된 embedding을 다시 정렬시키려면 재학습이 필요합니다.")
                print(f"    --ablation centroid_representativeness로 그룹별 대표성까지 같이")
                print(f"    보거나, --regroup_log_every로 학습 과정 자체가 수렴했는지부터")
                print(f"    확인해보는 걸 권합니다.")
            elif z_top1 < 2.0 and z_margin < 2.0:
                print(f"  ⚠️  top1 유사도·margin 둘 다 무작위 null과 통계적으로 구분되지")
                print(f"    않습니다(z_top1={z_top1:.2f}, z_margin={z_margin:.2f}) — 이 데이터셋의")
                print(f"    centroid 라우팅이 학습을 통해 유의미한 구조를 갖췄다고 보기")
                print(f"    어렵습니다. ①의 confidence·runner-up 정보가 '진짜 기하학적")
                print(f"    신호'라기보다 노이즈에 가까울 수 있음.")
            else:
                print(f"  ✅ 무작위 null보다 유의하게 큼(z_top1={z_top1:.2f}, z_margin={z_margin:.2f}) —")
                print(f"    이 데이터셋의 centroid 라우팅은 학습을 통해 실제로 유의미한 구조를")
                print(f"    갖췄다고 볼 수 있음. 이 공간 안에서 confidence가 낮게 나오는 샘플은")
                print(f"    '노이즈'가 아니라 상대적으로 정말 애매한 축에 속하는 케이스로 봐도 됨.")

            cg_save = {
                "sim_matrix":         sim_np.tolist(),
                "off_diag_mean":      float(off_diag_vals.mean()),
                "off_diag_std":       float(off_diag_vals.std()),
                "top_pairs":          [(int(iu[0][idx]), int(iu[1][idx]), float(pair_sims[idx]))
                                          for idx in top_idx],
                "same_target_count":  same_target_count,
                "diff_target_count":  diff_target_count,
                "qc_top1_mean":       float(top1_sims.mean()),
                "qc_top1_median":     float(np.median(top1_sims)),
                "qc_top1_max":        float(top1_sims.max()),
                "qc_margin_mean":     float(margins.mean()) if len(margins) else None,
                "qc_margin_narrow_ratio": narrow_margin_ratio,
                "null_top1_mean":     null_top1_mean,
                "null_top1_std":      null_top1_std,
                "null_margin_mean":   null_margin_mean,
                "null_margin_std":    null_margin_std,
                "z_top1":             float(z_top1),
                "z_margin":           float(z_margin),
                "openml_id":          openml_id,
                "seed":               args.seed,
            }
            cg_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_centroid_geometry.pkl"
            )
            with open(cg_path, "wb") as f:
                pickle.dump(cg_save, f)
            print(f"\n  저장: {cg_path}")

        # ── centroid_representativeness: 크기가 아니라 대표성(purity·cohesion) ──
        # [배경] centroid_geometry는 "가까운 centroid 쌍이 서로 다른 target을
        # 대표하는가"를 봤는데, 이건 쌍(pair) 단위 진단이라 "이 centroid
        # 하나가 자기 그룹을 얼마나 잘 대표하는가"는 안 봄. 크기가 크다고
        # 나쁜 게 아니고(데이터가 밀집된 영역이면 자연스럽게 큼), 작다고
        # 나쁜 것도 아님(outlier 영역이면 작은 게 정상) — 유일하게 문제인
        # 경우는 "크든 작든, 그 그룹 내부가 실제로 하나의 이야기로
        # 수렴하지 않는" 경우. 그래서 크기 대신 순도(purity, 그룹 내
        # 최다 target 비율)와 응집도(cohesion, 그룹 내 실제 샘플들이
        # 자기 centroid 주변에 얼마나 모여있는지)로 정렬해서 본다.
        elif args.ablation == "centroid_representativeness":
            model.eval()
            P = model.prototype_layer.P
            sample_groups = model.prototype_layer.sample_groups
            target_labels = model.prototype_layer.target_labels
            class_names = getattr(dataset, "target_class_names", None)

            print(f"\n  Centroid Representativeness (P={P})")
            print(f"  {'─'*60}")
            print(f"  크기가 아니라 대표성을 봄 — 크더라도 순도가 높으면 정상(밀집")
            print(f"  지역), 작더라도 순도가 100%면 정당한 outlier 그룹. 순도가")
            print(f"  baseline(전역 최다 target 비율)과 다를 바 없는 centroid만 문제.")

            y_train_np = y_train.detach().cpu().numpy()

            if tasktype in ("multiclass", "binclass"):
                y_int = np.rint(y_train_np).astype(int)
                vals, counts = np.unique(y_int, return_counts=True)
                global_majority_prop = float(counts.max() / counts.sum())
                global_majority_cls  = int(vals[counts.argmax()])
                global_majority_name = (
                    class_names[global_majority_cls]
                    if class_names is not None and global_majority_cls < len(class_names)
                    else f"Class {global_majority_cls}"
                )
                print(f"\n  [전역 baseline] 최다 target '{global_majority_name}' = "
                      f"{global_majority_prop:.1%} (n_classes={len(vals)})")
                print(f"  → 순도가 이 값보다 안 높으면, centroid가 굳이 있을 필요 없이")
                print(f"    '그냥 전체 다수결로 찍는 것'과 다를 바 없다는 뜻.")
            else:
                global_std = float(y_train_np.std())
                print(f"\n  [전역 baseline] y_train std = {global_std:.4f}")
                print(f"  → 그룹 내 std가 이 값과 다를 바 없으면, centroid가 굳이 있을")
                print(f"    필요 없이 '전체 평균'과 다를 바 없다는 뜻.")

            print(f"\n  [1/2] cohesion 계산 중 (train set 전체 embedding, feature {model.n_features}개)...")
            with torch.no_grad():
                c_norm = F.normalize(model.prototype_layer.centroid_emb, dim=-1)  # (P, D)
                q_chunks = []
                _batch = 256
                for start in range(0, X_train.shape[0], _batch):
                    q_chunks.append(
                        F.normalize(model.embedder(X_train[start:start + _batch]), dim=-1).cpu()
                    )
                q_all = torch.cat(q_chunks)  # (N_train, D), CPU
            c_norm_cpu = c_norm.cpu()

            print(f"  [2/2] centroid별 purity·cohesion 집계 중...")
            rows = []  # (p, size, purity_or_None, gap_or_None, cohesion, label_str)
            for p in range(P):
                grp = sample_groups[p] if sample_groups is not None else None
                size = len(grp) if grp else 0
                if size == 0:
                    continue

                idx_t   = torch.as_tensor(grp, dtype=torch.long)
                q_grp   = q_all[idx_t]                               # (size, D)
                cohesion = float((q_grp @ c_norm_cpu[p]).mean())

                tl = target_labels.get(p) if target_labels is not None else None
                if tl is None:
                    rows.append((p, size, None, None, cohesion, "N/A(그룹<2)"))
                    continue

                if tl["kind"] == "classification":
                    purity = tl["top_prop"]
                    gap    = purity - global_majority_prop
                    label_str = f"{tl['top_class_name']} {purity:.0%}"
                else:
                    y_grp     = y_train_np[grp]
                    group_std = float(np.std(y_grp))
                    purity    = 1.0 - (group_std / (global_std + 1e-8))
                    gap       = purity  # baseline은 정의상 0
                    label_str = f"mean={tl['group_mean']:.3g}, 집중도={purity:.0%}"

                rows.append((p, size, purity, gap, cohesion, label_str))

            # cohesion의 전체(centroid 간) percentile — 다른 centroid 대비 상대 순위
            cohesion_vals = np.array([r[4] for r in rows])
            cohesion_ranks = {
                r[0]: float((cohesion_vals < r[4]).mean()) for r in rows
            }

            rows_known   = sorted([r for r in rows if r[2] is not None], key=lambda r: r[2])
            rows_unknown = [r for r in rows if r[2] is None]

            print(f"\n  {'Centroid':<12} {'크기':>5}  {'대표':<20} {'gap vs baseline':>16}  "
                  f"{'cohesion':>9}  {'cohesion 순위':>12}")
            print(f"  {'─'*90}")
            for p, size, purity, gap, cohesion, label_str in rows_known:
                gap_str = f"{gap:+.1%}" if gap is not None else "-"
                crank = cohesion_ranks[p]
                flag = " ⚠️" if gap is not None and gap <= 0 else ""
                print(f"  Centroid_{p:<4} {size:>5}  {label_str:<20} {gap_str:>16}  "
                      f"{cohesion:>9.4f}  {crank:>11.0%}{flag}")
            for p, size, purity, gap, cohesion, label_str in rows_unknown:
                crank = cohesion_ranks[p]
                print(f"  Centroid_{p:<4} {size:>5}  {label_str:<20} {'-':>16}  "
                      f"{cohesion:>9.4f}  {crank:>11.0%}")

            n_below_baseline = sum(1 for r in rows_known if r[3] is not None and r[3] <= 0)
            eval_ratio = len(rows_known) / P if P > 0 else 0.0
            print(f"\n  [요약] {len(rows_known)}/{P}개 centroid({eval_ratio:.0%})가 평가 가능 — "
                  f"그중 {n_below_baseline}개가 baseline")
            print(f"  이하(⚠️ 표시) — '있으나 마나 한' centroid 후보. {len(rows_unknown)}개는")
            print(f"  그룹이 너무 작아(<2) 판단 불가.")
            if eval_ratio < 0.5:
                print(f"  ⚠️  평가 가능 비율 자체가 절반 미만입니다 — 대부분의 centroid가")
                print(f"    너무 작아 판단 불가 상태라는 뜻이고, '⚠️ 0개'만으로 안심할 수")
                print(f"    없는 상황입니다. 아래 요약과 별개로 이 비율 자체를 문제로")
                print(f"    보는 게 맞을 수 있습니다.")

            print(f"\n  [해석]")
            print(f"  이 표는 purity 오름차순(대표성 낮은 것부터)이라, 위쪽에 있는")
            print(f"  centroid일수록 자기 그룹을 잘 못 대표함. cohesion 순위가 같이")
            print(f"  낮으면(예: 하위 20% 안) '경계가 애매한 것'을 넘어 '애초에 이")
            print(f"  centroid 주변에 실제로 모인 게 없다'는 더 근본적인 신호일 수 있음")
            print(f"  — purity는 낮은데 cohesion은 높다면 '여러 target이 섞여있지만")
            print(f"  그 섞인 형태 자체는 일관됨'이라 해석이 다름. 100% 순도·응집도를")
            print(f"  기대할 필요는 없음 — baseline 대비 나은지가 실질적인 기준.")

            rep_save = {
                "rows": [
                    {"centroid": p, "size": size, "purity": purity, "gap": gap,
                     "cohesion": cohesion, "cohesion_percentile": cohesion_ranks[p],
                     "label": label_str}
                    for p, size, purity, gap, cohesion, label_str in rows
                ],
                "global_majority_prop": (global_majority_prop
                                          if tasktype in ("multiclass", "binclass") else None),
                "global_std": (global_std if tasktype == "regression" else None),
                "eval_ratio": eval_ratio,
                "n_below_baseline": n_below_baseline,
                "openml_id": openml_id,
                "seed": args.seed,
            }
            rep_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_centroid_representativeness.pkl"
            )
            with open(rep_path, "wb") as f:
                pickle.dump(rep_save, f)
            print(f"\n  저장: {rep_path}")

        # ── evidence_compensation: "①이 흐릿한 곳을 ②가 메워주는가" 직접 검증 ──
        # [배경] centroid_representativeness에서 purity가 낮아도(baseline
        # 이하) cohesion은 높은 centroid(예: credit-g의 Centroid_27, 26)가
        # 발견됨 — embedding은 일관되게 뭉쳐있는데 그 안의 target은 거의
        # 반반으로 섞인 경우. ①(그룹)만 보면 "애매하다"고 하지만, ②는
        # 실제 이웃 개별 샘플을 보여주는 방식이라 이 coarse-graining
        # 문제가 덜할 수 있음 — 이걸 실측으로 확인한다.
        elif args.ablation == "evidence_compensation":
            from scipy.stats import mannwhitneyu

            model.eval()
            P = model.prototype_layer.P
            sample_groups = model.prototype_layer.sample_groups
            target_labels = model.prototype_layer.target_labels

            print(f"\n  Evidence Compensation — '①이 흐릿한 곳을 ②가 메워주는가' (P={P})")
            print(f"  {'─'*60}")
            print(f"  centroid_representativeness와 같은 기준(purity vs baseline,")
            print(f"  cohesion은 이 run의 중앙값 기준 이분)으로 centroid를 3종으로 나눔:")
            print(f"    type A(진짜 문제)  : purity<=baseline, cohesion<=중앙값")
            print(f"    type B(①만 흐릿함) : purity<=baseline, cohesion>중앙값  ← 여기가 관심 대상")
            print(f"    normal            : purity>baseline")

            y_train_np = y_train.detach().cpu().numpy()
            if tasktype in ("multiclass", "binclass"):
                y_int = np.rint(y_train_np).astype(int)
                vals, counts = np.unique(y_int, return_counts=True)
                global_majority_prop = float(counts.max() / counts.sum())
            else:
                global_std = float(y_train_np.std())

            print(f"\n  [1/3] centroid별 purity·cohesion 재계산 중...")
            with torch.no_grad():
                c_norm = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                q_chunks = []
                _batch = 256
                for start in range(0, X_train.shape[0], _batch):
                    q_chunks.append(
                        F.normalize(model.embedder(X_train[start:start + _batch]), dim=-1).cpu()
                    )
                q_all = torch.cat(q_chunks)
            c_norm_cpu = c_norm.cpu()

            gaps = {}
            cohesions = {}
            for p in range(P):
                grp = sample_groups[p] if sample_groups is not None else None
                size = len(grp) if grp else 0
                if size == 0:
                    continue
                idx_t = torch.as_tensor(grp, dtype=torch.long)
                cohesions[p] = float((q_all[idx_t] @ c_norm_cpu[p]).mean())

                tl = target_labels.get(p) if target_labels is not None else None
                if tl is None:
                    continue
                if tl["kind"] == "classification":
                    gaps[p] = tl["top_prop"] - global_majority_prop
                else:
                    y_grp = y_train_np[grp]
                    group_std = float(np.std(y_grp))
                    gaps[p] = (1.0 - group_std / (global_std + 1e-8))  # baseline=0

            cohesion_vals = np.array(list(cohesions.values()))
            cohesion_median = float(np.median(cohesion_vals)) if len(cohesion_vals) else 0.0

            type_of = {}  # centroid_idx -> 'A' | 'B' | 'normal' | None(판단불가)
            for p in range(P):
                if p not in cohesions:
                    continue
                if p not in gaps:
                    type_of[p] = None   # 그룹 너무 작아 purity 판단 불가
                    continue
                if gaps[p] > 0:
                    type_of[p] = "normal"
                elif cohesions[p] > cohesion_median:
                    type_of[p] = "B"
                else:
                    type_of[p] = "A"

            n_a = sum(1 for v in type_of.values() if v == "A")
            n_b = sum(1 for v in type_of.values() if v == "B")
            n_normal = sum(1 for v in type_of.values() if v == "normal")
            print(f"  centroid 분류: type A={n_a}개, type B={n_b}개, normal={n_normal}개")

            if n_b == 0:
                print(f"\n  ⚠️  type B centroid가 하나도 없어 이 진단을 진행할 수 없습니다")
                print(f"    (이 데이터셋/모델에서는 'purity 낮지만 cohesion 높은' centroid가")
                print(f"    발견되지 않음 — centroid_representativeness로 먼저 확인해볼 것).")
            else:
                print(f"\n  [2/3] test set forward — ②(evidence_w) 수집 중...")
                n_test = X_test.shape[0]
                dominant_list, entropy_list, hard_group_list = [], [], []
                with torch.no_grad():
                    for start in range(0, n_test, 256):
                        X_batch = X_test[start:start + 256]
                        out_batch = model(X_batch)
                        evw = out_batch.get("evidence_w")
                        hg  = out_batch.get("hard_group")
                        if evw is None or hg is None:
                            continue
                        dom = evw.max(dim=-1).values
                        ent = -(evw * torch.log(evw + 1e-8)).sum(dim=-1)
                        dominant_list.append(dom.cpu())
                        entropy_list.append(ent.cpu())
                        hard_group_list.append(hg.cpu())

                if not dominant_list:
                    print(f"  ⚠️  evidence_w를 얻을 수 없습니다(fallback 등으로 이웃이 없는 경우일 수 있음).")
                else:
                    dominant  = torch.cat(dominant_list).numpy()
                    entropy   = torch.cat(entropy_list).numpy()
                    hard_group = torch.cat(hard_group_list).numpy()

                    sample_type = np.array([type_of.get(int(g), None) for g in hard_group])

                    mask_b      = sample_type == "B"
                    mask_a      = sample_type == "A"
                    mask_rest_b = ~mask_b  # type B 아닌 전부(A+normal+판단불가)
                    mask_rest_a = ~mask_a  # type A 아닌 전부(B+normal+판단불가) — 대조군용

                    print(f"\n  [3/3] Mann-Whitney U 검정 중 (test n={n_test})...")
                    print(f"\n  {'그룹':<12} {'n':>5}  {'dominant_weight':>16}  {'entropy':>10}")
                    print(f"  {'─'*50}")
                    for name, mask in [("type B", mask_b), ("type A", mask_a),
                                        ("나머지(전체)", np.ones_like(mask_b, dtype=bool))]:
                        if mask.sum() == 0:
                            print(f"  {name:<12} {'0':>5}  {'-':>16}  {'-':>10}")
                            continue
                        print(f"  {name:<12} {int(mask.sum()):>5}  "
                              f"{dominant[mask].mean():>16.4f}  {entropy[mask].mean():>10.4f}")

                    print(f"\n  [type B vs 나머지] — 핵심 비교")
                    if mask_b.sum() >= 3 and mask_rest_b.sum() >= 3:
                        u_dom, p_dom = mannwhitneyu(dominant[mask_b], dominant[mask_rest_b],
                                                     alternative="greater")
                        u_ent, p_ent = mannwhitneyu(entropy[mask_b], entropy[mask_rest_b],
                                                     alternative="less")
                        print(f"    dominant_weight: type B가 더 큼? Mann-Whitney p={p_dom:.4f}")
                        print(f"    entropy:         type B가 더 작음(뾰족함)? Mann-Whitney p={p_ent:.4f}")
                    else:
                        p_dom = p_ent = None
                        print(f"    표본 부족(type B n={mask_b.sum()}) — 검정 생략")

                    print(f"\n  [type A vs 나머지] — 대조군 (여기서는 유의하지 않아야 A/B 구분이 의미있음)")
                    if mask_a.sum() >= 3 and mask_rest_a.sum() >= 3:
                        u_dom_a, p_dom_a = mannwhitneyu(dominant[mask_a], dominant[mask_rest_a],
                                                         alternative="greater")
                        u_ent_a, p_ent_a = mannwhitneyu(entropy[mask_a], entropy[mask_rest_a],
                                                         alternative="less")
                        print(f"    dominant_weight: type A가 더 큼? Mann-Whitney p={p_dom_a:.4f}")
                        print(f"    entropy:         type A가 더 작음(뾰족함)? Mann-Whitney p={p_ent_a:.4f}")
                    else:
                        p_dom_a = p_ent_a = None
                        print(f"    표본 부족(type A n={mask_a.sum()}) — 검정 생략")

                    print(f"\n  [해석]")
                    b_significant = (p_dom is not None and p_dom < 0.05) or \
                                     (p_ent is not None and p_ent < 0.05)
                    a_significant = (p_dom_a is not None and p_dom_a < 0.05) or \
                                     (p_ent_a is not None and p_ent_a < 0.05)
                    if b_significant and not a_significant:
                        print(f"  ✅ type B는 ②가 유의하게 더 결정적이고, type A는 그렇지 않음 —")
                        print(f"    '①이 흐릿한 곳(순도는 낮지만 일관된 곳)을 ②가 실제로 메워준다'는")
                        print(f"    가설이 뒷받침됨. ①②를 나눠 설계한 근거가 이 데이터셋에서 실측으로")
                        print(f"    확인된 것으로 볼 수 있음.")
                    elif b_significant and a_significant:
                        print(f"  ⚠️  type A·B 둘 다 ②가 유의하게 결정적임 — ②가 '①이 흐릿한 곳만")
                        print(f"    선택적으로' 메워준다기보다, 그냥 전반적으로 ①보다 결정적인")
                        print(f"    경향이 있을 수 있음(①②의 역할 분담이 이 특정 형태로는 뚜렷이")
                        print(f"    드러나지 않음). 다른 데이터셋에서도 이 패턴이 반복되는지 볼 것.")
                    else:
                        print(f"  type B에서 ②가 유의하게 더 결정적이라고 하기 어려움. 표본이 적거나")
                        print(f"    (n_b={mask_b.sum()}), 이 데이터셋에서는 ①이 흐릿한 곳에서 ②도")
                        print(f"    같이 흐릿할 수 있음 — 데이터셋마다 다를 수 있는 부분이라 여러")
                        print(f"    데이터셋에서 반복 확인이 필요함.")

                    ec_save = {
                        "type_of":        {int(k): v for k, v in type_of.items()},
                        "n_a": n_a, "n_b": n_b, "n_normal": n_normal,
                        "dominant_mean_B": float(dominant[mask_b].mean()) if mask_b.sum() else None,
                        "dominant_mean_A": float(dominant[mask_a].mean()) if mask_a.sum() else None,
                        "dominant_mean_rest": float(dominant.mean()),
                        "entropy_mean_B": float(entropy[mask_b].mean()) if mask_b.sum() else None,
                        "entropy_mean_A": float(entropy[mask_a].mean()) if mask_a.sum() else None,
                        "p_dom_B": p_dom, "p_ent_B": p_ent,
                        "p_dom_A": p_dom_a, "p_ent_A": p_ent_a,
                        "openml_id": openml_id, "seed": args.seed,
                    }
                    ec_path = (
                        Path(log_dir)
                        / f"data={openml_id}..seed{args.seed}_evidence_compensation.pkl"
                    )
                    with open(ec_path, "wb") as f:
                        pickle.dump(ec_save, f)
                    print(f"\n  저장: {ec_path}")

        elif args.ablation == "dual_space_faithfulness":
            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            print(f"\n  Dual-Space Faithfulness Analysis")
            print(f"  {'─'*58}")

            sample_groups = model.prototype_layer.sample_groups

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
                    print(f"    ❌ 검증 시점에는 추가 학습이 없어 regroup 지연으로 설명될 수 없습니다 —")
                    print(f"       sample_groups가 가리키는 소스(MemoryBank/FeatureStore)와 지금")
                    print(f"       비교에 쓴 소스가 여전히 어긋나 있을 가능성이 높습니다.")
                    print(f"       아래 검증 2 결과는 재확인 전까지 신뢰할 수 없습니다.")
                else:
                    print(f"    ✅ 인덱스 정합성 확인됨 (MemoryBank 슬롯 기준) — 아래 결과를 신뢰할 수 있습니다.")

            if not index_ok:
                valid_p = []

            # ── [추가] 스토어 간 슬롯 대응 직접 검증 ──────────────────
            # [배경] 위 검증은 "sample_groups(캐시)가 지금 라우팅과 맞는가"만
            # 봄 — MemoryBank와 FeatureStore가 애초에 같은 슬롯에 같은
            # 샘플을 담고 있는지는 통계적 정황에만 의존하고 있었음.
            #
            # [갱신] sample_ids(MemoryBank.sample_ids / FeatureStore._sample_ids)
            # 도입 이후에는 이걸 percentile 비교가 아니라 정확한 등식으로
            # 확정할 수 있음. sample_ids가 아직 채워지지 않은 구버전
            # 체크포인트(전부 -1)라면, 예전 방식(무작위 셔플 대비 percentile)
            # 으로 자동 fallback.
            print(f"\n  [사전 검증 1.5] 스토어 간 슬롯 대응 확인 (MemoryBank ↔ FeatureStore)")
            if ref_raw is None or n_mem < 2:
                print(f"    ⚠️  feature_store가 없거나 데이터가 부족해 이 검증을 건너뜁니다.")
                store_ok = None
            else:
                mem_ids  = model.memory.sample_ids[:n_mem].detach().cpu()
                feat_ids = model.feature_store._sample_ids[:n_mem].detach().cpu()
                has_ids  = bool((mem_ids >= 0).any()) and bool((feat_ids >= 0).any())

                if has_ids:
                    # ── 1.5-a: 인덱스 대응 — 통계 아니라 정확한 등식 ──────
                    id_match = (mem_ids == feat_ids)
                    id_match_rate = float(id_match.float().mean())
                    print(f"    [1.5-a] sample_id 일치율: {id_match_rate:.1%}  "
                          f"(100%가 아니면 즉시 확정적 버그 — 통계적 여지 없음)")
                    id_ok = id_match_rate >= 0.999  # 부동소수점 아닌 정수 비교라 사실상 100% 기대
                    if id_ok:
                        print(f"    ✅ 두 스토어의 슬롯이 같은 샘플을 가리키는 것으로 확정됨.")
                    else:
                        print(f"    ❌ sample_id가 어긋나는 슬롯이 있습니다 — "
                              f"MemoryBank/FeatureStore가 서로 다른 시점 또는 순서로 "
                              f"복원됐을 가능성이 높습니다 (예: best_state/feature_store "
                              f"복원 순서 확인).")

                    # ── 1.5-b: 값 재현성 — refresh_on_best 여부에 따라
                    # 기대치가 다름. refresh했다면 부동소수점 오차 수준(≈1.0)
                    # 까지 기대할 수 있고, 안 했다면(기본값) 여전히 dropout
                    # 노이즈가 섞여 있어 1.0보다 뚜렷이 낮은 게 정상.
                    n_check   = min(n_mem, 300)
                    check_idx = torch.randperm(n_mem)[:n_check]
                    with torch.no_grad():
                        recomputed = model.embedder(ref_raw[check_idx].to(device)).cpu()
                    recomputed_n = F.normalize(recomputed, dim=-1)
                    stored_n     = F.normalize(ref_emb[check_idx], dim=-1)
                    matched_sim  = (recomputed_n * stored_n).sum(dim=-1)
                    print(f"    [1.5-b] 재계산 코사인 유사도: "
                          f"mean={matched_sim.mean():.6f}  min={matched_sim.min():.6f}")
                    if getattr(args, "refresh_on_best", False):
                        # refresh 이후엔 거의 정확히 1.0이어야 함 — 부동소수점
                        # 오차(비결정적 GPU 커널 포함) 감안해 0.999를 기준으로.
                        value_ok = float(matched_sim.min()) > 0.999
                        print(f"       (--refresh_on_best 켜짐 → ≈1.0 기대) "
                              f"{'✅ 재현됨' if value_ok else '❌ 기대에 못 미침 — refresh 로직 확인 필요'}")
                    else:
                        print(f"       (--refresh_on_best 꺼짐 → dropout 노이즈로 1.0보다 "
                              f"뚜렷이 낮은 게 정상. 재현성이 필요하면 --refresh_on_best로 재학습)")
                    store_ok = id_ok
                    if not id_ok:
                        valid_p = []
                else:
                    # ── 하위 호환: sample_ids 없는 구버전 체크포인트 → 기존 percentile 방식
                    print(f"    ⚠️  sample_ids가 없는 체크포인트입니다 — 기존 percentile 기반")
                    print(f"       방식으로 대신 확인합니다(확정적 증명 아님, 통계적 근사).")
                    n_check    = min(n_mem, 300)
                    check_idx  = torch.randperm(n_mem)[:n_check]
                    with torch.no_grad():
                        recomputed = model.embedder(ref_raw[check_idx].to(device)).cpu()
                    recomputed_n = F.normalize(recomputed, dim=-1)
                    stored_n     = F.normalize(ref_emb[check_idx], dim=-1)
                    matched_sim  = (recomputed_n * stored_n).sum(dim=-1)
                    shuffled_idx = torch.randperm(n_check)
                    shuffled_sim = (recomputed_n * stored_n[shuffled_idx]).sum(dim=-1)

                    print(f"    매칭된 슬롯끼리 코사인 유사도:      "
                          f"{matched_sim.mean():.4f} ± {matched_sim.std():.4f}")
                    print(f"    무작위로 섞은 슬롯끼리 코사인 유사도: "
                          f"{shuffled_sim.mean():.4f} ± {shuffled_sim.std():.4f}")
                    shuffled_p99 = float(np.percentile(shuffled_sim.numpy(), 99))
                    matched_median = float(matched_sim.median())
                    print(f"    셔플 분포의 99th percentile: {shuffled_p99:.4f}  "
                          f"vs  매칭 중앙값: {matched_median:.4f}")
                    store_ok = matched_median > shuffled_p99
                    if store_ok:
                        print(f"    ✅ 매칭된 슬롯이 무작위 분포의 상위 1%보다도 확실히 유사함")
                        print(f"       — 두 스토어의 슬롯이 같은 샘플을 가리키는 것으로 확인됨.")
                    else:
                        print(f"    ❌ 매칭된 슬롯의 유사도가 무작위로 섞은 분포의 상위 1%")
                        print(f"       수준을 못 넘습니다 — MemoryBank/FeatureStore 슬롯이")
                        print(f"       서로 다른 샘플을 가리키고 있을 가능성이 있습니다.")
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
                abl_logits_list = []
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

            # [추가] accuracy delta의 paired bootstrap CI — "이 Δ가 표본 크기
            # 때문에 노이즈로도 나올 수 있는 수준인가"를 바로 판단하기 위함.
            # (test set이 작은 데이터셋(예: N=100~300)에서 Δ가 몇 %p 안 되면
            # 실제로 유의미한지 눈으로 판단하기 어려움 — 특히 934처럼 "완전히
            # 0으로 안 돌아온다"는 주장을 하려면 이 CI가 0을 포함하는지가
            # 핵심.) full/ablation 예측을 같은 샘플 인덱스로 페어링해서
            # resampling — 독립 2-sample이 아니라 paired인 이유는 같은 테스트
            # 샘플에 대한 두 조건(원본/ablation) 비교라서, 샘플별 난이도 차이가
            # 상쇄되어 CI가 더 타이트하고 정확해짐(독립으로 재면 과도하게
            # 넓어짐). forward pass 재실행 없이 이미 계산된 preds만 재표본
            # 추출하므로 사실상 비용이 0에 가까움.
            if tasktype != "regression":
                _y_np = y_test.cpu().numpy() if torch.is_tensor(y_test) else np.asarray(y_test)
                _pf_np = preds_test.cpu().numpy() if torch.is_tensor(preds_test) else np.asarray(preds_test)
                _pa_np = abl_preds if isinstance(abl_preds, np.ndarray) else np.asarray(abl_preds)
                _rng = np.random.default_rng(0)
                _n = len(_y_np)
                _n_boot = 2000
                _correct_full = (_pf_np == _y_np).astype(np.float64)
                _correct_abl  = (_pa_np == _y_np).astype(np.float64)
                _boot_deltas = np.empty(_n_boot)
                for _bi in range(_n_boot):
                    _idx = _rng.integers(0, _n, size=_n)
                    _boot_deltas[_bi] = _correct_abl[_idx].mean() - _correct_full[_idx].mean()
                _ci_lo, _ci_hi = np.percentile(_boot_deltas, [2.5, 97.5])
                _point_delta = _correct_abl.mean() - _correct_full.mean()
                _sig = "0을 포함 안 함 → 유의미" if (_ci_lo > 0 or _ci_hi < 0) else "0을 포함 → 노이즈와 구분 안 됨"
                print(f"\n  [Bootstrap CI, paired, n_boot=2000] Δaccuracy = {_point_delta:+.4f}  "
                      f"95% CI [{_ci_lo:+.4f}, {_ci_hi:+.4f}]  (N_test={_n}) — {_sig}")

            # [추가] ECE(full vs ablation) — logloss가 크게 튀었을 때 그게
            # "calibration 자체가 망가진 것"인지 "accuracy는 그대로인 채 확률
            # 분포/logit scale만 흔들린 것"인지 구분하기 위함. 이 둘은 다른
            # 결론으로 이어짐 — ECE까지 같이 나빠지면 calibration 문제라고
            # 말할 수 있고, ECE는 그대로인데 logloss만 크면 소수 샘플의 극단적
            # 오배정(예: 정답에 0.999→0.000001) 같은 다른 메커니즘을 의심해야 함.
            if tasktype != "regression":
                y_test_np = y_test.cpu().numpy()
                preds_test_np = (preds_test.cpu().numpy() if torch.is_tensor(preds_test)
                                  else np.asarray(preds_test))
                probs_test_np = (probs_test.cpu().numpy() if torch.is_tensor(probs_test)
                                   else np.asarray(probs_test))

                def _pred_confidence(preds_np, probs_np):
                    # multiclass는 항상 (N,C) stacked. binclass는 두 형태가
                    # 섞여 있음 — probs_test(get_preds_and_probs 경유)는
                    # (N,2) stacked인데, abl_probs(이 블록 위에서 raw
                    # sigmoid로 직접 계산)는 (N,) 스칼라(P(class=1))라서
                    # ndim으로 분기해서 둘 다 처리.
                    probs_np = np.asarray(probs_np)
                    if probs_np.ndim == 2:
                        return probs_np[np.arange(len(preds_np)), preds_np]
                    else:  # (N,) — P(class=1)
                        return np.where(preds_np == 1, probs_np, 1.0 - probs_np)

                full_correct = (preds_test_np == y_test_np).astype(int)
                abl_correct  = (abl_preds == y_test_np).astype(int)
                full_conf = _pred_confidence(preds_test_np, probs_test_np)
                abl_conf  = _pred_confidence(abl_preds, abl_probs)

                full_ece = compute_ece(full_conf, full_correct)
                abl_ece  = compute_ece(abl_conf, abl_correct)
                ece_delta = abl_ece - full_ece
                arrow = "▼(악화)" if ece_delta > 0.01 else ("▲(개선)" if ece_delta < -0.01 else "─(거의 동일)")
                print(f"\n  ECE(Expected Calibration Error)")
                print(f"  {'-'*58}")
                print(f"  {'Full Model':>12}  {'Ablation':>12}  {'Δ':>10}")
                print(f"  {full_ece:>12.4f}  {abl_ece:>12.4f}  {ece_delta:>+9.4f} {arrow}")
                print(f"  (ECE도 같이 나빠지면(Δ 크게 양수) '{args.ablation}'가 진짜 calibration을")
                print(f"   해친다는 뜻 — logloss만 보고 그렇게 결론 내리면 안 됨. ECE는 그대로인데")
                print(f"   logloss만 폭증하면, accuracy에 영향 없는 소수 샘플에서 예측 확률이")
                print(f"   극단적으로(예: 0.999→0.000001) 무너졌을 가능성 쪽을 봐야 함 — 그 경우")
                print(f"   per-sample logloss 상위 몇 개를 직접 찍어보는 걸 권장.)")

                # [추가] per-sample logloss 증가량 상위 K개 — ECE는 그대로인데
                # logloss만 폭증했을 때, 정확히 몇 개 샘플이 그 폭증을 만들었는지
                # 직접 확인. -log(p_true_class) 기준.
                # [주의] "1-confidence로 정답 클래스 확률을 역산"하는 방식은
                # 클래스가 2개일 때만 성립하고 다중클래스에서는 예측이 틀린
                # 샘플에 대해 틀린 값을 줌 — probs 배열에서 정답 클래스 확률을
                # 직접 인덱싱해서 계산(포맷이 stacked(N,C)든 scalar(N,)든
                # 대응, binary/multiclass 공통).
                def _prob_of_true_class(probs_np, y_np):
                    probs_np = np.asarray(probs_np)
                    # [수정] y_test는 float32로 저장돼 있어서(binclass 특히)
                    # 정수 인덱싱에 바로 못 씀 — np.rint로 반올림 후 int 캐스팅
                    # (그냥 astype(int)는 0.999999 같은 부동소수점 오차를
                    # 0으로 잘라버릴 수 있어 위험, 다른 곳(run_calibration_
                    # analysis 등)에서도 이미 이 패턴을 씀).
                    y_int = np.rint(np.asarray(y_np)).astype(int)
                    if probs_np.ndim == 2:
                        return probs_np[np.arange(len(y_int)), y_int]
                    else:  # (N,) — P(class=1), binclass 전용
                        return np.where(y_int == 1, probs_np, 1.0 - probs_np)

                eps = 1e-12
                full_p_true = _prob_of_true_class(probs_test_np, y_test_np)
                abl_p_true  = _prob_of_true_class(abl_probs, y_test_np)
                full_ll_per = -np.log(np.clip(full_p_true, eps, 1.0))
                abl_ll_per  = -np.log(np.clip(abl_p_true, eps, 1.0))
                ll_increase = abl_ll_per - full_ll_per

                # [진단용, 추가] per-sample 재구성이 실제 집계 logloss(위 표의
                # test_metrics/abl_metrics, sklearn log_loss 기준)와 일치하는지
                # 직접 대조 — 어긋나면(예: 아래 두 줄이 크게 다르면) 이 블록의
                # 재구성 로직 자체에 버그가 있다는 뜻이고, 일치하면 재구성은
                # 맞고 다른 데(집계 쪽)를 봐야 한다는 뜻. total_increase가
                # 음수로 나오는 게 실측됐는데 평균 logloss는 크게 늘었다고
                # 보고돼서, 이 둘이 모순이라 직접 찍어서 확인.
                print(f"\n  [진단] per-sample 재구성 vs 공식 집계 logloss 대조:")
                print(f"    mean(full_ll_per)={full_ll_per.mean():.4f}  "
                      f"vs  test_metrics['logloss_test']={test_metrics.get('logloss_test', float('nan')):.4f}")
                print(f"    mean(abl_ll_per) ={abl_ll_per.mean():.4f}  "
                      f"vs  abl_metrics['logloss_test'] ={abl_metrics.get('logloss_test', float('nan')):.4f}")
                print(f"    (위 두 쌍이 각각 비슷해야 정상 — 다르면 재구성 로직 버그, 같으면 다른 원인)")


                total_increase = ll_increase.sum()
                order = np.argsort(-ll_increase)
                n_samples = len(ll_increase)

                print(f"\n  logloss 증가량 집중도 (전체 {n_samples}개 샘플의 총 증가량 {total_increase:+.2f} 기준):")
                if total_increase <= 1e-6:
                    print(f"    총 증가량이 0 이하 — 나빠진 샘플과 좋아진 샘플이 서로 상쇄되어")
                    print(f"    순효과가 거의 없다는 뜻(집중도 %는 이 경우 의미가 없어 생략).")
                    print(f"    참고로 Δlogloss>0(나빠짐)인 샘플만 {int((ll_increase > 0).sum())}개, "
                          f"그 합={ll_increase[ll_increase > 0].sum():.2f} / "
                          f"Δlogloss<0(좋아짐)인 샘플 {int((ll_increase < 0).sum())}개, "
                          f"그 합={ll_increase[ll_increase < 0].sum():.2f}")
                else:
                    for k in (20, 50, 100):
                        k_eff = min(k, n_samples)
                        share = ll_increase[order[:k_eff]].sum() / total_increase
                        print(f"    Top {k_eff:>3d}개가 전체 증가량의 {share:>6.1%} 차지")

                top_k = 20
                print(f"\n  per-sample 상위 {top_k}개 상세 (p(correct) = 정답 클래스에 준 확률):")
                print(f"  {'idx':>6}  {'full_p(correct)':>16}  {'abl_p(correct)':>16}  {'Δlogloss':>10}  {'correct(full→abl)':>18}")
                for i in order[:top_k]:
                    print(f"  {i:>6}  {full_p_true[i]:>16.6f}  {abl_p_true[i]:>16.6f}  "
                          f"{ll_increase[i]:>+10.4f}  {full_correct[i]}→{abl_correct[i]}")
                print(f"  (correct 열이 1→1인데 logloss가 크게 늘었으면 '여전히 맞았지만 확신을")
                print(f"   잃은' 경우, 1→0/0→1이면 예측 자체가 뒤집힌 경우 — 전자가 많으면")
                print(f"   accuracy에는 영향 없이 확률만 붕괴하는 이번 현상의 전형적인 모습.)")

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
            elif args.ablation in ("query_emb_zero", "query_emb_shuffle"):
                print(f"  → 성능 하락 = '정상 학습된 head가 query_emb 슬롯에 실제로")
                print(f"    얼마나 의존하는가'의 대가. --no_query_emb(처음부터 빼고")
                print(f"    재학습 — 학습 붕괴로 확인됨)와는 다른 질문: 이건 재학습 없이")
                print(f"    이미 학습된 가중치 그대로 이 슬롯 하나만 지워본 것.")
                print(f"    하락폭이 크면 → head가 이 슬롯에 실제로 크게 의존 (예상대로).")
                print(f"    하락폭이 작으면 → head가 이 슬롯을 거의 안 쓰면서도 학습")
                print(f"    자체는 이 슬롯 없이는 불가능했다는 뜻 — 즉 query_emb의 역할이")
                print(f"    '최종 예측 재료'가 아니라 '학습 중 gradient 경로 안정화'였을")
                print(f"    가능성. shuffle이 zero보다 분포 이탈이 적어 더 신뢰할 것.")
            elif args.ablation in ("context_emb_zero", "context_emb_shuffle"):
                print(f"  → 성능 하락 = '정상 학습된 head가 context_emb(=Explanation①의")
                print(f"    prototype 신호) 슬롯에 실제로 얼마나 의존하는가'의 대가.")
                print(f"    하락폭이 query_emb_zero/shuffle보다 훨씬 작으면 → Explanation①이")
                print(f"    보여주는 'Centroid_X로 배정됨'이라는 서사가 실제 예측 근거로서는")
                print(f"    약하다는 뜻(그래도 그룹 자체의 존재는 검색 속도/해석에 별도로")
                print(f"    유효 — 이 실험은 '예측 기여도'만 봄, ②retrieval 마스킹 가치나")
                print(f"    그룹의 데이터 구조 반영 여부는 안 봄).")
            elif args.ablation in ("agg_emb_zero", "agg_emb_shuffle"):
                print(f"  → query_emb는 정상, agg_emb만 섞음 — query_emb_shuffle(정반대 조합,")
                print(f"    agg_emb는 정상·query_emb만 섞음)과 나란히 놓고 비교할 것.")
                print(f"    이번 하락폭이 크면 → agg_emb 자체가 예측에 기여(검색이 실제로")
                print(f"    유용한 정보를 나름). 이번 하락폭이 작으면(query_emb_shuffle의")
                print(f"    거의 랜덤 수준 붕괴와 대비된다면) → 성능은 사실상 query_emb")
                print(f"    슬롯 하나가 거의 다 담당하고 있고, agg_emb는 '짝이 맞을 때만")
                print(f"    의미 있는 보조 신호'이거나 거의 장식적인 슬롯일 가능성.")
                print(f"    두 실험을 합치면 '짝 어긋남 자체의 대가'와 'agg_emb 단독 정보량'을")
                print(f"    분리해서 볼 수 있음.")

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

    # ── Linear Probe ───────────────────────────────────────────
    # [추가] query_emb/context_emb/agg_emb 각각에 별도 선형 분류기(또는
    # 회귀, regression이면 Ridge)를 붙여서 "이 표현 자체에 타겟 정보가
    # 있는가"를 직접 측정. shuffle ablation(예측 결과 관점)과 다른 질문 —
    # shuffle이 0에 가까웠던 게 "정보가 없어서"(A)인지 "정보는 있는데
    # concat+공유 MLP가 못 쓰는 것"(B)인지 구분하려는 용도. 재학습 없이
    # --from_saved_state로 불러온 모델에서 임베딩만 뽑아 sklearn으로
    # 별도 학습(TabERA 자체는 안 건드림).
    if args.linear_probe and do_analysis:
        print(f"\n{'='*60}")
        print(f"  Linear Probe: query_emb / context_emb / agg_emb 정보량 확인")
        print(f"{'='*60}")
        model.eval()

        def _extract_embeddings(X, batch_size=512):
            qs, cs, ags = [], [], []
            with torch.no_grad():
                for start in range(0, len(X), batch_size):
                    _out = model(X[start:start + batch_size])
                    qs.append(_out["query_emb"].cpu())
                    cs.append(_out["context_emb"].cpu())
                    ags.append(_out["agg_emb"].cpu())
            return (torch.cat(qs).numpy(), torch.cat(cs).numpy(), torch.cat(ags).numpy())

        q_tr, c_tr, a_tr = _extract_embeddings(X_train)
        q_te, c_te, a_te = _extract_embeddings(X_test)

        import numpy as _np
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import accuracy_score, r2_score

        if tasktype == "regression":
            y_tr_np = (y_train * y_std).cpu().numpy()
            y_te_np = (y_test * y_std).cpu().numpy()
        else:
            y_tr_np = y_train.cpu().numpy().astype(int)
            y_te_np = y_test.cpu().numpy().astype(int)

        probe_results = {}
        reprs = {
            "query_emb":     (q_tr, q_te),
            "context_emb":   (c_tr, c_te),
            "agg_emb":       (a_tr, a_te),
            "concat(q+c+a)": (_np.concatenate([q_tr, c_tr, a_tr], axis=1),
                              _np.concatenate([q_te, c_te, a_te], axis=1)),
        }

        # [추가] scale 불균형 가설 검증용 — concat 전에 branch별로 정규화한
        # 두 가지 버전. "정규화하면 agg_emb 단독 수준까지 회복되는가"를 보려는
        # 것 — 회복되면 scale이 실제 원인일 가능성을 지지, 안 되면(가능성2/3:
        # multicollinearity, representation geometry 붕괴) scale 하나로는
        # 설명 안 된다는 뜻.
        def _l2_normalize_blocks(*blocks_tr_te):
            """(tr, te) 쌍들을 각각 L2-normalize(샘플별 unit norm)한 뒤 concat."""
            tr_parts, te_parts = [], []
            for tr, te in blocks_tr_te:
                tr_n = tr / (_np.linalg.norm(tr, axis=1, keepdims=True) + 1e-8)
                te_n = te / (_np.linalg.norm(te, axis=1, keepdims=True) + 1e-8)
                tr_parts.append(tr_n)
                te_parts.append(te_n)
            return _np.concatenate(tr_parts, axis=1), _np.concatenate(te_parts, axis=1)

        def _standardize_blocks(*blocks_tr_te):
            """(tr, te) 쌍들을 각각 StandardScaler(train 기준 fit, LayerNorm과
            유사하게 차원별 zero-mean/unit-variance)한 뒤 concat."""
            from sklearn.preprocessing import StandardScaler
            tr_parts, te_parts = [], []
            for tr, te in blocks_tr_te:
                _scaler = StandardScaler()
                tr_parts.append(_scaler.fit_transform(tr))
                te_parts.append(_scaler.transform(te))
            return _np.concatenate(tr_parts, axis=1), _np.concatenate(te_parts, axis=1)

        reprs["concat(q+c+a)_l2norm"] = _l2_normalize_blocks((q_tr, q_te), (c_tr, c_te), (a_tr, a_te))
        reprs["concat(q+c+a)_standardized"] = _standardize_blocks((q_tr, q_te), (c_tr, c_te), (a_tr, a_te))

        for _name, (_tr, _te) in reprs.items():
            if tasktype == "regression":
                _clf = Ridge(alpha=1.0)
                _clf.fit(_tr, y_tr_np)
                _score = float(r2_score(y_te_np, _clf.predict(_te)))
                _metric_name = "R2"
            else:
                _clf = LogisticRegression(max_iter=2000)
                _clf.fit(_tr, y_tr_np)
                _score = float(accuracy_score(y_te_np, _clf.predict(_te)))
                _metric_name = "acc"
            probe_results[_name] = _score
            print(f"  {_name:28s} linear probe {_metric_name}={_score:.4f}")

        # [추가] representation similarity — "agg_emb가 새로운 정보인가,
        # query_emb와 거의 같은 방향인가"를 직접 측정. cosine은 샘플별
        # 방향 유사도(직관적), linear CKA는 전체 표현 공간 정렬도(scale/
        # rotation-invariant, 더 엄밀한 multivariate 지표) — 두 지표가
        # 다르게 나올 수 있어(예: 개별 샘플 cosine은 낮은데 CKA는 높을 수
        # 있음, 그 반대도 가능) 같이 봄.
        def _linear_cka(X: "_np.ndarray", Y: "_np.ndarray") -> float:
            Xc = X - X.mean(axis=0, keepdims=True)
            Yc = Y - Y.mean(axis=0, keepdims=True)
            hsic = _np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
            norm_x = _np.linalg.norm(Xc.T @ Xc, ord="fro")
            norm_y = _np.linalg.norm(Yc.T @ Yc, ord="fro")
            return float(hsic / (norm_x * norm_y + 1e-12))

        def _mean_cosine(X: "_np.ndarray", Y: "_np.ndarray"):
            xn = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            yn = Y / (_np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
            cos = (xn * yn).sum(axis=1)
            return {"mean": float(cos.mean()), "std": float(cos.std())}

        similarity_results = {}
        for _pair_name, _X, _Y in [
            ("query_vs_agg",     q_te, a_te),
            ("query_vs_context", q_te, c_te),
            ("context_vs_agg",   c_te, a_te),
        ]:
            _cka = _linear_cka(_X, _Y)
            _cos = _mean_cosine(_X, _Y)
            similarity_results[_pair_name] = {"cka": _cka, "cosine_mean": _cos["mean"], "cosine_std": _cos["std"]}
            print(f"  [similarity] {_pair_name:18s} CKA={_cka:.4f}  "
                  f"cosine={_cos['mean']:+.4f}±{_cos['std']:.4f}")

        probe_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_linear_probe.pkl"
        with open(probe_path, "wb") as f:
            pickle.dump({
                "probe_results": probe_results,
                "similarity_results": similarity_results,
                "openml_id": openml_id, "seed": args.seed, "tasktype": tasktype,
            }, f)
        print(f"\n  저장: {probe_path}")

    if args.calibration_analysis and do_analysis:
        if tasktype == "regression":
            print(f"\n  ⚠️  --calibration_analysis는 classification 전용입니다 — "
                  f"이 데이터셋({tasktype})에서는 건너뜁니다.")
        else:
            calib_result = run_calibration_analysis(
                model, X_test, y_test, tasktype,
                X_train=X_train, y_train=y_train,
                class_names=getattr(dataset, "target_class_names", None),
            )
            print_calibration_analysis(calib_result)
            calib_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_calibration.pkl"
            with open(calib_path, "wb") as f:
                pickle.dump({**calib_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {calib_path}")

    if args.branch_contribution and do_analysis:
        if not hasattr(model, "_head_block_slices") or not model._head_block_slices:
            print(f"\n  ⚠️  --branch_contribution은 fusion_mode='concat'에서만 의미가 있습니다 "
                  f"(residual은 fusion_alpha/beta가 이미 같은 역할) — 건너뜁니다.")
        else:
            contrib_result = compute_branch_linear_contribution(model, X_test)
            print_branch_linear_contribution(contrib_result)
            contrib_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_branch_contribution.pkl"
            with open(contrib_path, "wb") as f:
                pickle.dump({**contrib_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {contrib_path}")

    if args.branch_information and do_analysis:
        if not hasattr(model, "_head_block_slices") or not model._head_block_slices:
            print(f"\n  ⚠️  --branch_information은 fusion_mode='concat'에서만 의미가 있습니다 — "
                  f"건너뜁니다.")
        else:
            info_result = analyze_branch_information(model, X_test, tasktype)
            print_branch_information(info_result)
            info_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_branch_information.pkl"
            with open(info_path, "wb") as f:
                pickle.dump({**info_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {info_path}")

    if args.gradient_attribution and do_analysis:
        if not hasattr(model, "_head_block_slices") or not model._head_block_slices:
            print(f"\n  ⚠️  --gradient_attribution은 fusion_mode='concat'에서만 됩니다 — "
                  f"건너뜁니다.")
        else:
            grad_result = compute_branch_gradient_attribution(model, X_test, y_test, tasktype)
            print_branch_gradient_attribution(grad_result)
            grad_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_gradient_attribution.pkl"
            with open(grad_path, "wb") as f:
                pickle.dump({**grad_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {grad_path}")

    if args.head_sensitivity and do_analysis:
        if not hasattr(model, "_head_block_slices") or not model._head_block_slices:
            print(f"\n  ⚠️  --head_sensitivity는 fusion_mode='concat'에서만 됩니다 — "
                  f"건너뜁니다.")
        else:
            sens_result = compute_head_sensitivity(model, X_test)
            print_head_sensitivity(sens_result)
            sens_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_head_sensitivity.pkl"
            with open(sens_path, "wb") as f:
                pickle.dump({**sens_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {sens_path}")

    if args.head_input_cancellation and do_analysis:
        if getattr(model, "fusion_mode", None) != "residual":
            print(f"\n  ⚠️  --head_input_cancellation은 fusion_mode='residual'인 모델에서만 "
                  f"의미가 있습니다 — 이 체크포인트는 fusion_mode='{getattr(model, 'fusion_mode', None)}' "
                  f"라 건너뜁니다.")
        else:
            hic_result = compute_head_input_cancellation(model, X_test)
            print_head_input_cancellation(hic_result)
            hic_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_head_input_cancellation.pkl"
            with open(hic_path, "wb") as f:
                pickle.dump({**hic_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {hic_path}")
    if args.pre_fusion_gradient_attribution and do_analysis:
        # [수정] args.fusion_mode가 아니라 실제 로드된 model.fusion_mode를 봐야
        # 함 — --from_saved_state는 architecture 관련 CLI 플래그(--fusion_mode
        # 포함)를 전부 무시하고 저장된 model_kwargs로 모델을 재구성하므로
        # (위 1527-1531행 경고 참고), args.fusion_mode는 사용자가 --fusion_mode를
        # 안 줬을 때의 기본값("residual")일 뿐 실제 로드된 체크포인트의 구조를
        # 반영하지 않을 수 있다. --gradient_attribution/--head_sensitivity가
        # model._head_block_slices(실제 모델 속성)로 판단하는 것과 같은 이유로
        # 여기도 model.fusion_mode(실제 속성)로 판단.
        if getattr(model, "fusion_mode", None) != "residual":
            print(f"\n  ⚠️  --pre_fusion_gradient_attribution은 fusion_mode='residual'인 "
                  f"모델에서만 의미가 있습니다(β가 있어야 raw agg_emb 항의 크기를 해석할 "
                  f"기준이 생김) — 이 체크포인트는 fusion_mode='{getattr(model, 'fusion_mode', None)}' "
                  f"라 건너뜁니다. concat 모드는 --gradient_attribution을 쓰세요.")
        else:
            pfg_result = compute_pre_fusion_gradient_attribution(model, X_test, y_test, tasktype)
            print_pre_fusion_gradient_attribution(pfg_result)
            pfg_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_pre_fusion_gradient_attribution.pkl"
            with open(pfg_path, "wb") as f:
                pickle.dump({**pfg_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype}, f)
            print(f"\n  저장: {pfg_path}")



    # ── 결과 저장 ──────────────────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_meta.pkl"

    model.eval()
    # [수정] 이전엔 X_test 전체(수천~수만 샘플)를 한 번에 forward했는데,
    # centroid 쏠림이 심한 데이터셋(예: adult, max_cluster_size가 수천~수만)
    # 에서 retrieve()가 그 큰 클러스터를 배치 전체 크기만큼 한꺼번에 처리하려다
    # 메모리 요구량이 폭발(실측: 25GB 요청, adult 데이터셋)해서 CUDA OOM으로
    # 죽는 문제가 있었음 — --calibration_analysis(배치 512개씩 처리)는 같은
    # 모델·같은 데이터로 문제없이 끝났는데, 바로 이 지점만 배치가 안 걸려있어서
    # 발생. run_calibration_analysis()와 같은 패턴으로 배치 처리하도록 수정.
    _pred_batch_size = 512
    _logits_chunks = []
    # [추가, v2 Phase 2] fusion_mode="gated_sum"이면 이 루프에서 이미 도는
    # forward pass의 out["head_gate_*"]를 배치 크기 가중평균으로 같이
    # 누적 — 별도 forward pass를 새로 만들 필요 없음. concat/residual
    # 모드에서는 out["head_gate_mean"]이 항상 빈 dict/None이라 자동으로
    # 아무것도 안 쌓임(아래 if 조건이 자연히 False).
    _gate_mean_sum = {}
    _gate_var_sum = {}
    _gate_entropy_sum = 0.0
    _gate_n_samples = 0
    _gate_logit_mean_sum = {}
    _gate_logit_gap_sum = 0.0
    # [추가, v2, context_gated_beta 전용] centroid별 β 상관관계 사후분석용 —
    # 배치 평균이 아니라 X_test 전체에 대한 (centroid_id, β) 샘플별 쌍을
    # 그대로 모음. 다른 fusion_mode에서는 계속 빈 리스트로 남아 저장 자체를
    # 스킵함.
    _centroid_id_chunks = []
    _agg_beta_chunks = []
    _rb_centroid_id_chunks = []
    _rb_routing_confidence_chunks = []
    _rb_topk_idx_chunks = []
    _rb_entropy_chunks = []
    _rb_n_eff_chunks = []
    _rb_top1_weight_chunks = []
    _rb_purity_chunks = []            # [추가] top-k 중 query와 같은 라벨인 비율 (unweighted)
    _rb_weighted_purity_chunks = []   # [추가] evidence_w로 가중한 same-label 비율
    # [Local Retriever 진단, 추가] similarity geometry — temperature와 원인
    # 분리용(사용자 요청). evidence.py가 새 모듈 없이 항상 계산.
    _rb_sim_top1_chunks = []
    _rb_sim_bottomk_chunks = []
    _rb_sim_margin_chunks = []
    _rb_sim_std_chunks = []
    # [추가, evidence utilization 진단] "agg_emb가 query_emb와 실질적으로
    # 다른 정보를 담고 있는가"를 raw(head 진입 전, LN 적용 전) 표현 기준
    # 샘플별로 직접 잼 — head_cos_qa_mean(--log_fusion_trajectory)은 LN
    # 적용 후 배치 평균 스칼라 하나만 epoch별로 남기므로, 여기서는 (1) LN
    # 없는 원본 표현 기준, (2) 샘플별 분포(퍼센타일 계산 가능) 두 가지
    # 점에서 보완적. fusion_mode="residual"에서만 fusion_beta가 스칼라로
    # 채워지므로(그 외 모드는 None) 이 블록은 residual 전용.
    _rb_cos_qa_chunks = []
    _rb_qnorm_chunks = []
    _rb_anorm_chunks = []
    _rb_beta_ratio_chunks = []   # β·‖agg_emb‖/‖query_emb‖ (샘플별)
    _rb_shift_norm_chunks = []   # ‖z-q‖ = ‖β·agg_emb‖ (representation shift, 샘플별)
    with torch.no_grad():
        for _start in range(0, len(X_test), _pred_batch_size):
            _out = model(X_test[_start:_start + _pred_batch_size])
            _logits_chunks.append(_out["logits"].cpu())
            if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") and _out.get("head_gate_mean"):
                _bsz = min(_pred_batch_size, len(X_test) - _start)
                for _name, _val in _out["head_gate_mean"].items():
                    _gate_mean_sum[_name] = _gate_mean_sum.get(_name, 0.0) + _val * _bsz
                for _name, _val in _out["head_gate_var"].items():
                    _gate_var_sum[_name] = _gate_var_sum.get(_name, 0.0) + _val * _bsz
                if _out.get("head_gate_entropy_mean") is not None:
                    _gate_entropy_sum += _out["head_gate_entropy_mean"] * _bsz
                _gate_n_samples += _bsz
                for _name, _val in _out.get("head_gate_logit_mean", {}).items():
                    _gate_logit_mean_sum[_name] = _gate_logit_mean_sum.get(_name, 0.0) + _val * _bsz
                if _out.get("head_gate_logit_gap_mean") is not None:
                    _gate_logit_gap_sum += _out["head_gate_logit_gap_mean"] * _bsz
            if args.fusion_mode == "context_gated_beta" and _out.get("agg_beta_per_sample") is not None:
                _centroid_id_chunks.append(_out["centroid_id"].cpu())
                _agg_beta_chunks.append(_out["agg_beta_per_sample"].cpu())
            if args.export_centroid_retrieval_behavior and _out.get("evidence_w") is not None and _out.get("centroid_id") is not None:
                # model.eval() 상태(이 루프 진입 전에 이미 model.eval() 호출됨)라
                # dropout이 no-op — evidence_w가 이미 유효한 확률분포이므로
                # log_evidence_stats의 재정규화 없이 그대로 써도 안전함.
                _ew = _out["evidence_w"].cpu()
                _rb_centroid_id_chunks.append(_out["centroid_id"].cpu())
                _rb_routing_confidence_chunks.append(_out["routing_confidence"].cpu())
                _rb_topk_idx_chunks.append(_out["topk_idx"].cpu())
                _rb_entropy_chunks.append(
                    -(_ew.clamp_min(1e-12) * _ew.clamp_min(1e-12).log()).sum(-1)
                )
                _rb_n_eff_chunks.append(1.0 / _ew.square().sum(-1).clamp_min(1e-12))
                _rb_top1_weight_chunks.append(_ew.max(dim=-1).values)
                # [추가, 사용자 요청] retrieval label purity — "무엇을 가져왔는가"를
                # 직접 잼. topk_idx는 memory bank(학습셋) 인덱스라 model.memory.labels
                # 로 바로 라벨을 찾을 수 있음(새 forward 불필요, 이미 계산된 topk_idx/
                # evidence_w 재사용). regression은 label purity 개념이 없어 스킵.
                if tasktype != "regression":
                    _batch_y = y_test[_start:_start + _pred_batch_size].cpu()
                    _batch_y_int = torch.round(_batch_y).long()
                    _neighbor_labels = model.memory.labels[_out["topk_idx"]].cpu().long()  # (B, k)
                    _same_label = (_neighbor_labels == _batch_y_int.unsqueeze(-1)).float()  # (B, k)
                    _rb_purity_chunks.append(_same_label.mean(dim=-1))            # unweighted: 단순 top-k 중 동일 라벨 비율
                    _rb_weighted_purity_chunks.append((_ew * _same_label).sum(dim=-1))  # evidence_w-weighted: 실제 aggregation에 반영되는 비중까지 고려
                if _out.get("similarity_top1_per_sample") is not None:
                    _rb_sim_top1_chunks.append(_out["similarity_top1_per_sample"].cpu())
                    _rb_sim_bottomk_chunks.append(_out["similarity_bottomk_per_sample"].cpu())
                    _rb_sim_margin_chunks.append(_out["similarity_margin_per_sample"].cpu())
                    _rb_sim_std_chunks.append(_out["similarity_std_per_sample"].cpu())
                # [추가, evidence utilization 진단] fusion_mode="residual"일
                # 때만 의미 있음(β가 스칼라로 존재하는 유일한 모드). q/a는
                # raw(query_emb/agg_emb, LN 적용 전) — head가 실제로 보는
                # LN(q)/LN(a)와는 다를 수 있지만, "이 두 표현 자체가 얼마나
                # 다른 정보인가"를 보는 질문(분석계획 1번)엔 원본이 맞는
                # 기준임. 0-division 방어로 clamp_min 사용.
                if _out.get("fusion_beta") is not None:
                    _q = _out["query_emb"].cpu()
                    _a = _out["agg_emb"].cpu()
                    _beta = float(_out["fusion_beta"])
                    _q_norm = _q.norm(dim=-1)
                    _a_norm = _a.norm(dim=-1)
                    _rb_cos_qa_chunks.append(F.cosine_similarity(_q, _a, dim=-1))
                    _rb_qnorm_chunks.append(_q_norm)
                    _rb_anorm_chunks.append(_a_norm)
                    _rb_beta_ratio_chunks.append(
                        (abs(_beta) * _a_norm) / _q_norm.clamp_min(1e-12)
                    )
                    # z - q = β·agg_emb (+ α·context_emb, use_context_emb=True일
                    # 때만) — 현재 기본값(use_context_emb=False)에서는 뒤 항이
                    # 없어 β·agg_emb와 정확히 같지만, use_context_emb=True 비교
                    # 실행에서도 정확하도록 alpha 항을 조건부로 더함.
                    _shift = _beta * _a
                    if _out.get("fusion_alpha") is not None and _out.get("context_emb") is not None:
                        _shift = _shift + float(_out["fusion_alpha"]) * _out["context_emb"].cpu()
                    _rb_shift_norm_chunks.append(_shift.norm(dim=-1))
    logits = torch.cat(_logits_chunks, dim=0).numpy()
    np.save(str(pred_path), logits)

    # [추가, v2, context_gated_beta 전용] (centroid_id, β) 샘플별 쌍 저장 —
    # X_test 기준(다른 진단들과 일관성 유지). 파일명은 preds.npy와 같은
    # _save_tag를 공유해서 어느 run 결과인지 바로 알 수 있게.
    if args.fusion_mode == "context_gated_beta" and _centroid_id_chunks:
        _centroid_ids_all = torch.cat(_centroid_id_chunks, dim=0).numpy()
        _agg_betas_all = torch.cat(_agg_beta_chunks, dim=0).numpy()
        _cb_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_centroid_beta.npz"
        np.savez(str(_cb_path), centroid_id=_centroid_ids_all, agg_beta=_agg_betas_all)
        print(f"  [context_gated_beta] centroid_id/β 샘플별 쌍 저장: {_cb_path}"
              f" ({len(_centroid_ids_all)}개, centroid_id는 test set 기준)")

    # [Centroid Retrieval Behavior Analysis, 신규] baseline/V2 포함 어떤
    # 모델에서도 계산 가능(evidence_w/centroid_id/topk_idx/routing_confidence
    # 는 항상 존재) — 특정 모듈을 정당화하기 위한 진단이 아니라 TabERA의
    # retrieval 특성 자체(group마다 evidence distribution이 다른가, routing
    # confidence와 entropy가 상관관계를 갖는가, 같은 centroid 안에서 retrieval
    # 이 안정적인가)를 이해하기 위한 독립적 진단.
    if args.export_centroid_retrieval_behavior and _rb_centroid_id_chunks:
        _rb_centroid_ids_all = torch.cat(_rb_centroid_id_chunks, dim=0).numpy()
        _rb_routing_confidences_all = torch.cat(_rb_routing_confidence_chunks, dim=0).numpy()
        _rb_topk_idx_all = torch.cat(_rb_topk_idx_chunks, dim=0).numpy()  # (N, k)
        _rb_entropies_all = torch.cat(_rb_entropy_chunks, dim=0).numpy()
        _rb_n_effs_all = torch.cat(_rb_n_eff_chunks, dim=0).numpy()
        _rb_top1_weights_all = torch.cat(_rb_top1_weight_chunks, dim=0).numpy()
        _rb_sample_ids_all = np.arange(len(_rb_centroid_ids_all))
        _rb_savez_kwargs = dict(
            sample_id=_rb_sample_ids_all,
            centroid_id=_rb_centroid_ids_all,
            routing_confidence=_rb_routing_confidences_all,
            topk_idx=_rb_topk_idx_all,           # (N, k) — memory-side index, neighbor 재구성/label 조회용
            entropy=_rb_entropies_all,
            n_eff=_rb_n_effs_all,
            top1_weight=_rb_top1_weights_all,
        )
        if _rb_purity_chunks:
            # [추가, 사용자 요청] retrieval label purity — "무엇을 가져왔는가".
            # purity: top-k 이웃 중 query와 같은 라벨 비율(단순 카운트).
            # weighted_purity: evidence_w로 가중한 버전 — 실제 agg_emb 계산에
            # 반영되는 비중까지 고려(top1 하나가 정답이고 나머지가 오답이어도
            # top1의 evidence_w가 압도적이면 weighted_purity는 높게 나옴 —
            # purity와 weighted_purity의 차이 자체가 "attention이 정답 쪽에
            # 잘 집중하는가"의 지표가 됨). tasktype="regression"이면 label
            # purity 개념이 없어 둘 다 빈 상태로 남음(위 export 루프에서
            # 애초에 append 안 함).
            _rb_savez_kwargs["retrieval_label_purity"] = torch.cat(_rb_purity_chunks, dim=0).numpy()
            _rb_savez_kwargs["retrieval_weighted_label_purity"] = torch.cat(_rb_weighted_purity_chunks, dim=0).numpy()
        if _rb_sim_top1_chunks:
            _rb_savez_kwargs["similarity_top1"] = torch.cat(_rb_sim_top1_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_bottomk"] = torch.cat(_rb_sim_bottomk_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_margin"] = torch.cat(_rb_sim_margin_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_std"] = torch.cat(_rb_sim_std_chunks, dim=0).numpy()
        # [추가, evidence utilization 진단 — 분석계획 1번] fusion_mode="residual"
        # 일 때만 채워짐(그 외 모드는 fusion_beta가 None이라 위 루프에서
        # 애초에 안 쌓임). cos_qa/q_norm/a_norm은 raw(LN 적용 전) query_emb/
        # agg_emb 기준 — "두 표현이 실제로 다른 정보인가"를 head 내부
        # 정규화와 무관하게 직접 보기 위함. beta_agg_ratio는 β·‖agg‖/‖query‖,
        # representation_shift_norm은 ‖z-q‖(=‖β·agg_emb(+α·context_emb)‖).
        if _rb_cos_qa_chunks:
            _rb_savez_kwargs["cos_qa"] = torch.cat(_rb_cos_qa_chunks, dim=0).numpy()
            _rb_savez_kwargs["query_emb_norm"] = torch.cat(_rb_qnorm_chunks, dim=0).numpy()
            _rb_savez_kwargs["agg_emb_norm"] = torch.cat(_rb_anorm_chunks, dim=0).numpy()
            _rb_savez_kwargs["beta_agg_ratio"] = torch.cat(_rb_beta_ratio_chunks, dim=0).numpy()
            _rb_savez_kwargs["representation_shift_norm"] = torch.cat(_rb_shift_norm_chunks, dim=0).numpy()
        # [Local Retriever 진단, 추가] centroid별 "실제 예측 품질"을 보려면
        # 정답과 맞대조가 필요함 — sample count/margin/N_eff만으로는 "이
        # centroid가 좋은 local expert인가"를 못 봄(사용자 지적). logits는
        # 이미 위에서 계산돼 있으므로(np.save(pred_path,...) 직전) 추가
        # forward 없이 get_preds_and_probs()만 재사용.
        with torch.no_grad():
            _rb_preds_t, _rb_probs_t = get_preds_and_probs(torch.from_numpy(logits), tasktype)
        _rb_y_test_np = y_test.cpu().numpy()
        if tasktype == "regression":
            _rb_savez_kwargs["y_true"] = _rb_y_test_np
            _rb_savez_kwargs["error"] = (_rb_preds_t.numpy() - _rb_y_test_np) ** 2  # squared error
        else:
            _rb_y_int = np.rint(_rb_y_test_np).astype(int)
            _rb_preds_np = _rb_preds_t.numpy()
            _rb_probs_np = _rb_probs_t.numpy()
            _rb_savez_kwargs["y_true"] = _rb_y_int
            _rb_savez_kwargs["correct"] = (_rb_preds_np == _rb_y_int).astype(int)
            # per-sample logloss(-log p_true) — accuracy만으로 안 보이는 "얼마나
            # 확신 있게 맞았는지/틀렸는지"까지 centroid별로 볼 수 있게.
            if _rb_probs_np.ndim == 2:
                _rb_p_true = _rb_probs_np[np.arange(len(_rb_y_int)), _rb_y_int]
            else:  # (N,) — P(class=1), binclass 전용
                _rb_p_true = np.where(_rb_y_int == 1, _rb_probs_np, 1.0 - _rb_probs_np)
            _rb_savez_kwargs["error"] = -np.log(np.clip(_rb_p_true, 1e-12, 1.0))  # per-sample logloss
        _rb_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_centroid_retrieval_behavior.npz"
        np.savez(str(_rb_path), **_rb_savez_kwargs)
        print(f"  [export_centroid_retrieval_behavior] sample_id/centroid_id/routing_confidence/topk_idx/entropy/n_eff/top1_weight"
              f"{'/retrieval_label_purity/retrieval_weighted_label_purity' if _rb_purity_chunks else ''}"
              f"{'/similarity_top1/bottomk/margin/std' if _rb_sim_top1_chunks else ''}"
              f"{'/cos_qa/query_emb_norm/agg_emb_norm/beta_agg_ratio/representation_shift_norm' if _rb_cos_qa_chunks else ''}"
              f"/y_true/{'error' if tasktype=='regression' else 'correct/error'} 샘플별 쌍 저장: {_rb_path}"
              f" ({len(_rb_centroid_ids_all)}개, centroid_id/sample_id는 test set 기준)"
              + ("" if _rb_cos_qa_chunks else "\n  [주의] fusion_mode≠'residual'이라 cos_qa/beta_agg_ratio/"
                 "representation_shift_norm은 저장되지 않았습니다(β가 없는 fusion_mode에서는 정의되지 않는 값)."))


    # [추가, v2 Phase 2] 위에서 누적한 gate 통계를 배치 가중평균으로 확정 —
    # meta dict 구성 시 fusion_gate_*_final 필드가 참조함. gated_sum이
    # 아니거나 X_test가 비어있으면(있을 수 없지만 방어적으로) 빈 값 유지.
    _final_gate_stats = {"mean": {}, "var": {}, "entropy": None,
                          "logit_mean": {}, "logit_gap": None}
    if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") and _gate_n_samples > 0:
        _final_gate_stats["mean"] = {k: v / _gate_n_samples for k, v in _gate_mean_sum.items()}
        _final_gate_stats["var"]  = {k: v / _gate_n_samples for k, v in _gate_var_sum.items()}
        _final_gate_stats["entropy"] = _gate_entropy_sum / _gate_n_samples
        _final_gate_stats["logit_mean"] = {k: v / _gate_n_samples for k, v in _gate_logit_mean_sum.items()}
        _final_gate_stats["logit_gap"] = _gate_logit_gap_sum / _gate_n_samples

    meta = {
        "openml_id":   openml_id,
        "tasktype":    tasktype,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "seed":        args.seed,
        "train_seed":  train_seed,
        # [추가] optimize.py의 HPO trial들은 이미 trial.set_user_attr()로
        # reinit_per_epoch/active_ratio_std를 study.pkl에 저장하고 있었음
        # (몰랐던 게 아니라 이미 있었음) — 근데 이 최종 재학습(reproduce.py)
        # 쪽 meta.pkl에는 안 담겨서, "채택된 모델 1개"의 학습 안정성
        # 지표를 study.pkl까지 다시 뒤지지 않고는 못 봤음. wrapper가 이미
        # 계산해둔 걸 그대로 옮겨 담기만 함 — 계산 로직 변경 없음.
        "centroid_geometry_diag": wrapper.centroid_geometry_diag,
        # [진단용] --log_branch_gradients=False면 둘 다 빈 리스트(학습을
        # 안 했거나 --from_saved_state로 건너뛴 경우도 마찬가지) — 항상
        # 키 자체는 존재하게 해서 다운스트림 분석 코드가 .get() 없이도
        # 안전하게 접근 가능.
        "branch_gradient_history": wrapper.branch_gradient_history,
        "branch_gradient_batch_history": wrapper.branch_gradient_batch_history,
        # [추가] epoch별 active_ratio 등 라우팅 안정성 전체 시계열 — 지금까지는
        # centroid_geometry_diag(마지막 스냅샷 하나)만 저장돼서, "active_ratio가
        # 낮은 epoch에 context/agg gradient도 같이 낮은가"처럼 branch_gradient_
        # history와 시점을 맞춰 보는 분석이 불가능했다. 둘 다 epoch 키로
        # zip 가능 (regroup_history는 매 epoch, branch_gradient_history는
        # log_branch_gradients=True일 때만 매 epoch — 둘 다 켰으면 길이가 같음).
        "regroup_history": wrapper.regroup_history,
        "evidence_stats_history": wrapper.evidence_stats_history,
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only if args.deterministic else None,
        "use_offset_correction": True,
        "global_retrieve": False,
        "use_context_emb": not args.no_context_emb,
        "use_query_emb_in_head": not args.no_query_emb,
        "use_ema_codebook": args.ema_codebook,
        "ema_decay": (args.ema_decay_override if args.ema_decay_override is not None else 0.99) if args.ema_codebook else None,
        "blockwise_layernorm": args.blockwise_layernorm,
        "head_branch_l2norm": args.head_branch_l2norm,
        "fusion_mode": args.fusion_mode,
        "disable_retrieval_branch": args.disable_retrieval_branch,
        "exclude_self_retrieval": (not args.allow_self_retrieval),
        "value_mode": args.value_mode,
        "neighbor_interaction_mode": args.neighbor_interaction_mode,
        "interaction_n_heads": args.interaction_n_heads,
        "aggregator_mode": args.aggregator_mode,
        "head_attn_alpha_override": args.head_attn_alpha_override,
        "head_neighbor_source": args.head_neighbor_source,
        # [v2, 진단용] cross_attention 모드의 학습된 alpha 최종값 — 전체
        # 모델을 다시 로드하지 않고도 meta.pkl만으로 "이 run에서 head가
        # retrieval 정보를 얼마나 크게 반영하기로 했는가"를 바로 볼 수
        # 있게(fusion_alpha_final과 같은 성격). pooling 모드에서는 None.
        "head_attn_alpha_final": (
            float(model.head_cross_attn.alpha.detach().item())
            if args.aggregator_mode == "cross_attention" else None
        ),
        # [추가, 진단용] residual fusion의 학습된 α/β 최종값 — 전체 모델을
        # 다시 로드하지 않고도 meta.pkl만으로 "이 run에서 head가 context/agg를
        # 어느 정도 크기로 쓰기로 했는가"를 바로 볼 수 있게. concat 모드에서는
        # 둘 다 None.
        "fusion_alpha_final": (
            float(model.fusion_alpha.detach().item())
            if (args.fusion_mode == "residual" and model.fusion_alpha is not None) else None
        ),
        "fusion_beta_final": (
            float(model.fusion_beta.detach().item())
            if args.fusion_mode == "residual" else None
        ),
        # [추가, v2 Phase 2, 진단용] gated_sum의 gate 최종 통계 — meta.pkl만
        # 봐도 "이 run에서 head가 branch별로 평균 얼마씩 가져갔는가"를 바로
        # 알 수 있게. fusion_alpha_final/beta_final과 같은 성격이지만
        # (1) 샘플별로 다른 값의 "배치 가중평균"이라는 점, (2) branch가
        # 3개(또는 use_context_emb=False면 2개)라 dict라는 점이 다름. 위
        # preds.npy를 만드는 X_test 배치 순회 루프에서 같이 누적한 값 —
        # 학습 종료 후 eval 모드에서의 test set 전체 평균이라 필드 이름을
        # "final"로 함(단일 배치나 학습 중간 값이 아님). concat/residual
        # 모드에서는 둘 다 빈 dict/None.
        "fusion_gate_mean_final": (
            _final_gate_stats.get("mean", {}) if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") else {}
        ),
        "fusion_gate_var_final": (
            _final_gate_stats.get("var", {}) if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") else {}
        ),
        "fusion_gate_entropy_final": (
            _final_gate_stats.get("entropy") if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") else None
        ),
        # [추가, v2 Phase 2 후속] temperature 값 자체(재현성 확인용, 기본
        # 1.0이면 기존과 동일 동작) + pre-softmax logit 최종 통계.
        "fusion_gate_temperature": args.fusion_gate_temperature,
        "fusion_gate_logit_mean_final": (
            _final_gate_stats.get("logit_mean", {}) if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") else {}
        ),
        "fusion_gate_logit_gap_final": (
            _final_gate_stats.get("logit_gap") if args.fusion_mode in ("gated_sum", "anchor_gate", "context_gated_beta") else None
        ),
        # [추가] 이번 run에서 α/β가 학습됐는지(None) 아니면 고정됐는지(값) —
        # fusion_alpha_final/beta_final만 보면 "학습해서 이 값이 됐다"와
        # "애초에 이 값으로 고정해놨다"를 구분할 수 없어서 별도로 남김.
        "fusion_alpha_override": args.fusion_alpha_override,
        "fusion_beta_override": args.fusion_beta_override,
        # [추가, 진단용] --log_fusion_trajectory로 기록한 epoch별 α/β·branch
        # norm 궤적. 기본은 빈 리스트(플래그 안 켰으면).
        "fusion_trajectory_history": getattr(wrapper, "fusion_trajectory_history", []),
        "centroid_label_mi_history": getattr(wrapper, "centroid_label_mi_history", []),
        "shuffle_ablation_trajectory_history": getattr(wrapper, "shuffle_ablation_trajectory_history", []),
        "representation_drift_history": getattr(wrapper, "representation_drift_history", []),
        "detach_context_grad": args.detach_context_grad,
        "query_detach_warmup_epochs": args.query_detach_warmup_epochs,
        "query_detach_warmup_steps": args.query_detach_warmup_steps,
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
    if wrapper.centroid_geometry_diag is not None:
        _diag = wrapper.centroid_geometry_diag
        print(f"  centroid_geometry_diag: "
              f"reinit_per_epoch={_diag.get('reinit_per_epoch', float('nan')):.3f}  "
              f"active_ratio_std={_diag.get('active_ratio_std', float('nan')):.4f}  "
              f"margin_percentile={_diag.get('margin_percentile', float('nan')):.3f}  "
              f"avg_inter_dist_final={_diag.get('avg_inter_dist_final', float('nan')):.3f} "
              f"(← 위 '[CentroidLayer] KMeans++ ... avg_inter_dist=' 값과 비교 — "
              f"학습 끝에서 뚜렷이 작아졌으면 centroid들이 서로 뭉쳤다는 뜻)")
    if wrapper.branch_gradient_history:
        _first, _last = wrapper.branch_gradient_history[0], wrapper.branch_gradient_history[-1]
        _names = [k[:-len("_grad_norm")] for k in _first if k.endswith("_grad_norm")]
        print(f"  branch_gradient_history: epoch {int(_first['epoch'])} → {int(_last['epoch'])}")
        for _n in _names:
            print(f"    {_n:8s}: grad {_first.get(f'{_n}_grad_norm', float('nan')):.4f} → "
                  f"{_last.get(f'{_n}_grad_norm', float('nan')):.4f}   "
                  f"W {_first.get(f'{_n}_weight_norm', float('nan')):.4f} → "
                  f"{_last.get(f'{_n}_weight_norm', float('nan')):.4f}")
        print(f"    (전체 곡선은 meta.pkl의 branch_gradient_history/"
              f"branch_gradient_batch_history 참고 — 이 요약은 첫/끝 epoch만 비교)")
    if getattr(wrapper, "shuffle_ablation_trajectory_history", None):
        _sfirst = wrapper.shuffle_ablation_trajectory_history[0]
        _slast  = wrapper.shuffle_ablation_trajectory_history[-1]
        print(f"  shuffle_ablation_trajectory: epoch {int(_sfirst['epoch'])} → {int(_slast['epoch'])}")
        print(f"    Δquery_shuffle: {_sfirst['delta_query_shuffle']:+.4f} → {_slast['delta_query_shuffle']:+.4f}")
        print(f"    Δagg_shuffle  : {_sfirst['delta_agg_shuffle']:+.4f} → {_slast['delta_agg_shuffle']:+.4f}")
        print(f"    (agg 쪽 delta가 학습 초반보다 후반에 0에 더 가까워지면 — 'retrieval은 "
              f"optimization scaffold' 가설과 정합. 전체 곡선은 meta.pkl의 "
              f"shuffle_ablation_trajectory_history 참고.)")
    if getattr(wrapper, "representation_drift_history", None) and len(wrapper.representation_drift_history) > 1:
        _dfirst = wrapper.representation_drift_history[1]  # [0]은 anchor 스냅샷 자체(항상 0)라 skip
        _dlast  = wrapper.representation_drift_history[-1]
        print(f"  representation_drift_trajectory: epoch {int(_dfirst['epoch'])} → {int(_dlast['epoch'])}")
        print(f"    cos(q_t-q_0, a_0): {_dfirst['cos_drift_vs_agg0']:+.3f} → {_dlast['cos_drift_vs_agg0']:+.3f}  "
              f"(증가 추세면 'query가 초기 retrieval 방향으로 이동'=흡수 신호)")
        print(f"    cos(q_t, a_0)    : {_dfirst['cos_query_t_vs_agg0']:+.3f} → {_dlast['cos_query_t_vs_agg0']:+.3f}")
        print(f"    centroid_stability_vs_epoch0: {_dfirst['centroid_stability_vs_epoch0']:.1%} → "
              f"{_dlast['centroid_stability_vs_epoch0']:.1%}")
        print(f"    (전체 곡선은 meta.pkl의 representation_drift_history 참고.)")

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
            (fs._store.detach().cpu(), fs._ptr, fs._filled, fs._sample_ids.detach().cpu())
            if fs is not None else None
        ),
        "col_names":    dataset.col_names,
        "n_train":      len(X_train),
        "tasktype":     tasktype,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "seed":         args.seed,
        "train_seed":   train_seed,
        "deterministic": args.deterministic,
    }, str(state_path))
    print(f"  저장: {state_path}")

    # ── Feature 기여도 설명 출력 ─────────────────────────
    if args.explain and do_analysis:
        print(f"\n{'='*52}")
        print(f"  TabERA Explanations (--explain)")
        print(f"{'='*52}")

        model.eval()
        n_show = min(args.n_explain, len(y_test))
        X_show = X_test[:n_show]

        with torch.no_grad():
            out = model(X_show, return_explanations=True)

        explanations = out.get("explanations", [])

        # [추가] Prediction confidence — classifier softmax(다중/이진분류) 또는
        # 예측값(회귀). Routing confidence(①)와 별개의 값임을 화면에서부터
        # 분리해서 보여주기 위해 여기서 미리 계산해둠 — get_preds_and_probs는
        # eval.py의 metric 계산과 동일한 로직이라 test_metrics와 정의가
        # 어긋나지 않음.
        pred_idx, pred_probs = get_preds_and_probs(out["logits"], tasktype)
        pred_infos = []
        for b in range(n_show):
            if tasktype == "regression":
                pred_val = float(pred_idx[b].item()) * y_std
                pred_infos.append({"pred_label": f"{pred_val:.4g}", "pred_confidence": None})
            else:
                idx = int(pred_idx[b].item())
                conf = float(pred_probs[b, idx].item())
                label = (dataset.target_class_names[idx]
                         if getattr(dataset, "target_class_names", None) else str(idx))
                pred_infos.append({"pred_label": label, "pred_confidence": conf})

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
                                   num_cols=list(dataset.X_num),
                                   pred_info=pred_infos[i])

    return {"train_seed": train_seed, "val_metrics": val_metrics, "test_metrics": test_metrics}


def main():

    parser = argparse.ArgumentParser(description="TabERA Reproduce Best Config")
    parser.add_argument("--gpu_id",    type=int, default=0)
    parser.add_argument("--openml_id", type=int, required=True)
    parser.add_argument("--savepath",  type=str, default=".",
                        help="optim_logs가 있는 상위 경로")
    parser.add_argument("--seed",      type=int, default=1,
                        help="optimize.py와 동일한 seed 사용 (데이터 분할=fold 선택 전용, "
                             "libs/data.py의 KFold(random_state=42 고정)에서 몇 번째 fold를 "
                             "test로 쓸지만 결정함 — 학습 초기화/배치 순서와는 무관)")
    parser.add_argument("--train_seed", type=int, default=None,
                        help=(
                            "[통제 실험용] 학습 초기화·배치 순서 전용 seed — torch.manual_seed/"
                            "np.random.seed에 --seed 대신 이 값을 씀. 데이터 분할(--seed, "
                            "TabularDataset의 fold 선택)에는 영향을 주지 않는다. 지정 안 하면 "
                            "기존 동작과 동일하게 --seed를 그대로 씀(하위 호환). "
                            "run-to-run variance를 재려면 --seed(데이터 분할)는 고정하고 이 값만 "
                            "바꿔가며 N번 반복 실행 — 같은 train/val/test split에서 가중치 초기화, "
                            "DataLoader 배치 순서(supervised.py의 torch.randperm), "
                            "dead-centroid reinit(tabera.py의 torch.randint/torch.randn)이 "
                            "모두 이 seed 하나로 결정되는 torch 전역 RNG에서 나오기 때문에 이 "
                            "값만 바꾸면 나머지는 고정한 채로 학습 궤적만 흔들 수 있다. "
                            "--from_saved_state와 같이 쓰면 재학습을 안 하므로 무효과."
                        ))
    parser.add_argument("--train_seeds", type=int, nargs="+", default=None,
                        help=(
                            "[v1.1, 추가] --train_seed(단수)의 복수형 — 여러 개를 한 번에 "
                            "돈다. 예: --train_seeds 1 2 3 4 5. optimize.py처럼 dataset/HPO "
                            "study를 한 번만 로드하고 그 안에서 seed마다 학습만 반복(run_single_seed()) "
                            "— 예전처럼 shell에서 seed마다 프로세스를 새로 띄우면 매번 dataset "
                            "로딩 비용을 냈던 문제를 없앰. 주어지면 --train_seed(단수)는 무시됨. "
                            "--from_saved_state와는 같이 못 씀(특정 seed로 저장된 체크포인트 "
                            "하나를 불러오는 거라 여러 seed를 도는 것 자체가 의미가 없음 — 같이 "
                            "주면 에러). 2개 이상이면 끝에 val/test metric의 seed 간 mean±std "
                            "요약이 추가로 출력됨."
                        ))
    parser.add_argument("--explain_seed", type=int, default=None,
                        help=(
                            "[v1.1, 추가] --train_seeds로 여러 seed를 돌 때, --explain/"
                            "--calibration_analysis/--linear_probe(켜져 있는 것들)를 어느 "
                            "seed에서만 실행할지. 기본값(None)이면 --train_seeds의 마지막 "
                            "seed. 모든 seed마다 --explain 텍스트가 다 나오면 로그가 지나치게 "
                            "길어지므로, 상세 분석은 대표 seed 하나로 제한하고 나머지는 "
                            "val/test metric만 남긴다. --train_seeds에 없는 값을 주면 에러."
                        ))
    parser.add_argument("--deterministic", action="store_true",
                        help=(
                            "[통제 실험용] torch.use_deterministic_algorithms(True) + "
                            "cudnn.deterministic=True + cudnn.benchmark=False를 켜고 재학습. "
                            "지금까지 측정한 --train_seed 간 변동성(test 성능/active_ratio_std/"
                            "reinit count/조기종료 epoch)이 GPU 비결정성 때문인지, 아키텍처 "
                            "자체의 chaotic sensitivity 때문인지 분리하기 위한 용도 — 이 플래그를 "
                            "켠 채로 같은 --seed에 --train_seed만 바꿔가며 N번 반복했을 때 "
                            "변동성이 (a) 거의 사라지면 GPU 비결정성이 주 원인, (b) 그대로 "
                            "남으면 아키텍처의 chaotic sensitivity가 주 원인이라는 뜻. "
                            "CUBLAS_WORKSPACE_CONFIG 환경변수는 이 플래그가 켜져 있으면 "
                            "torch import 전에(--gpu_id와 같은 자리) 자동으로 설정됨. "
                            "일부 연산이 결정적 구현이 없으면 RuntimeError로 즉시 중단되는데, "
                            "이건 버그가 아니라 '어떤 연산이 비결정성의 소스인지'를 알려주는 "
                            "유용한 정보이므로 에러 메시지의 연산 이름을 그대로 보고할 것 — "
                            "--deterministic_warn_only로 우회하지 말고 먼저 보고."
                        ))
    parser.add_argument("--deterministic_warn_only", action="store_true",
                        help=(
                            "--deterministic가 RuntimeError로 중단될 때만 우회용으로 사용. "
                            "결정적 구현이 없는 연산을 에러 대신 경고만 내고 그냥(비결정적으로) "
                            "실행 — 즉 이 옵션을 켜면 '완전한 결정성 보장'이 깨지므로, 어떤 "
                            "연산이 남아있는 비결정성의 원인인지 콘솔 경고를 확인하고 결과 "
                            "해석 시 감안할 것. --deterministic 없이는 아무 효과 없음."
                        ))
    parser.add_argument("--run_tag", type=str, default=None,
                        help=(
                            "[통제 실험용] 파일명에 붙는 임의의 태그(예: 'r1', 'r2'). "
                            "--seed/--train_seed/--deterministic가 전부 동일한 조합을 "
                            "N번 반복 실행할 때(=순수 GPU 비결정성 크기 측정) 그냥 두면 "
                            "매번 같은 파일명이라 이전 결과를 덮어쓰게 됨 — 이럴 때만 "
                            "지정. 기본값 None이면 기존 동작과 동일(태그 없음)."
                        ))
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    parser.add_argument("--epochs",    type=int, default=HPO_TRAINING_SCHEDULE["epochs"],
                        help=(
                            "[수정] 기본값을 optimize.py의 HPO trial과 동일한 값으로 "
                            "맞춤(libs/search_space.py의 HPO_TRAINING_SCHEDULE 참고) — "
                            "예전엔 이 기본값이 200(HPO는 100)이라 'best config를 "
                            "재현한다'는 이름의 스크립트가 실제로는 HPO 때와 다른 학습 "
                            "스케줄로 돌아가는 불일치가 있었음(실측: adult(1590)에서 "
                            "reproduce.py가 더 오래 학습했는데도 val acc가 HPO best "
                            "trial보다 낮고, centroid 쏠림도 더 심하게 진행됨). HPO와 "
                            "다른 스케줄로 일부러 실험하고 싶으면 이 값을 명시적으로 "
                            "override하면 됨 — 그때는 'best config 재현'이 아니라 "
                            "별도 실험이라는 걸 인지하고 쓸 것."
                        ))
    parser.add_argument("--patience",  type=int, default=HPO_TRAINING_SCHEDULE["patience"],
                        help="[수정] 기본값을 HPO_TRAINING_SCHEDULE에서 가져옴 — 위 --epochs 참고.")
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
    parser.add_argument("--freeze_encoder_retrain_head", action="store_true",
                        help=(
                            "[통제 실험용] --from_saved_state로 불러온 모델에서 embedder/"
                            "prototype_layer/ot_selector/context_proj를 전부 얼리고(gradient "
                            "차단, KMeans++ 재초기화·regroup_update도 건너뜀 — centroid를 "
                            "완전히 고정) head 계열(head/head_query_ln/head_context_ln/"
                            "head_agg_ln)만 백지에서(reset_parameters) 다시 학습. linear_probe "
                            "가 '새 분류기를 새로 학습'해서 context/agg_emb에 정보가 있다는 "
                            "것만 보여준 것과 달리, 이건 '지금 이 head 구조가 그 정보를 실제로 "
                            "쓰는 법을 배울 수 있는가'를 직접 검증 — 이렇게 재학습한 head가 "
                            "원래(공동학습) 모델보다 test 성능이 오르면 원래 학습이 optimization "
                            "dynamics(query path가 먼저 수렴해 나머지를 밀어냄) 때문에 그 정보를 "
                            "못 썼다는 뜻이고, 그래도 안 오르면 head 구조 자체("
                            "concat+MLP)가 구조적으로 그 정보를 못 쓴다는 뜻. "
                            "--freeze_head_epochs로 재학습 epoch 수 조절. "
                            "--from_saved_state 없이는 무효과."
                        ))
    parser.add_argument("--freeze_head_epochs", type=int, default=50,
                        help=(
                            "--freeze_encoder_retrain_head일 때 head만 재학습할 epoch 수. "
                            "인코더가 고정돼 있어 head 혼자 수렴하는 데 필요한 epoch은 "
                            "원래 공동학습보다 짧을 수 있음 — 검증 안 된 기본값(50), "
                            "필요시 조정."
                        ))
    parser.add_argument("--linear_probe", action="store_true",
                        help=(
                            "[통제 실험용] --from_saved_state로 불러온 모델에서 "
                            "query_emb/context_emb/agg_emb(+concat)를 각각 뽑아 sklearn "
                            "LogisticRegression(분류)/Ridge(회귀)로 별도 학습해 test 성능을 "
                            "비교. --ablation *_shuffle 결과(정확도 하락 없음)가 "
                            "'context/agg emb에 애초에 정보가 없어서'인지 'concat+공유 "
                            "MLP head가 그 정보를 못/안 쓰는 것뿐'인지 구분하려는 용도 — "
                            "전자면 context/agg emb의 단독 probe 성능도 query_emb보다 "
                            "훨씬 낮게 나오고, 후자면 context/agg emb 단독으로도 "
                            "query_emb에 준하는 성능이 나옴. TabERA 자체는 재학습하지 "
                            "않음(임베딩만 추출, sklearn은 별도로 가볍게 학습) — "
                            "--from_saved_state 필수는 아니지만 없으면 방금 막 학습을 "
                            "마친 모델 그대로 씀."
                        ))
    parser.add_argument("--calibration_analysis", action="store_true",
                        help=(
                            "[진단용] test set 전체에서 routing confidence(①, prototype "
                            "공간에서의 상대적 우세)와 prediction confidence(classifier "
                            "softmax) 각각을 실제 정확도와 대조 — 개별 샘플(--explain)이 "
                            "아니라 test set 전체 통계로 'routing이 애매해도 최종 예측이 "
                            "믿을 만한가'에 답하기 위함. routing confidence 구간별 accuracy가 "
                            "평평하면 retrieval/fusion이 routing 불확실성을 실제로 보완한다는 "
                            "근거, prediction confidence의 ECE가 높으면 (특히 고신뢰 구간에서 "
                            "accuracy가 confidence에 못 미치면) overconfidence/calibration "
                            "불량. TabERA 자체는 재학습 안 함(--linear_probe와 같은 성격) — "
                            "--from_saved_state 필수는 아니지만 없으면 방금 학습된 모델 그대로 씀."
                        ))
    parser.add_argument("--branch_contribution", action="store_true",
                        help=(
                            "[진단용] head의 첫 Linear가 실제로 받는 입력(내부 LayerNorm이 "
                            "있으면 그걸 통과한 뒤)에서 branch(query/context/agg)별 ||W_i x_i||"
                            "(그 branch의 실제 선형 기여도)를 측정 — activation norm(--log_"
                            "branch_gradients가 재는 것)과 다르게 이건 head가 실제로 계산에 "
                            "쓰는 값이라 'classifier가 이 branch를 얼마나 반영하는가'에 더 "
                            "가까움. activation norm은 Linear(Wx+b)에서 x가 커도 W가 그만큼 "
                            "작으면 출력엔 영향 없다는 점(activation-weight trade-off) 때문에 "
                            "단독으로는 기여도를 못 보여줌 — 이 진단이 그 문제를 피함. 순수 "
                            "forward pass만 필요해서 재학습 불필요(--from_saved_state와 같이 "
                            "쓸 수 있음, --log_branch_gradients는 학습 중 gradient가 필요해서 "
                            "재학습이 있어야 했던 것과 대비). fusion_mode='residual'이면 "
                            "concat 자체가 없어 이 진단은 스킵됨(그땐 fusion_alpha/beta 값 "
                            "자체가 이미 branch별 기여도 지표)."
                        ))
    parser.add_argument("--branch_information", action="store_true",
                        help=(
                            "[진단용] --branch_contribution이 'norm(크기)'만 보는 것과 달리, "
                            "이건 '정보량'(샘플마다 실제로 다른가)을 봄. (1) 평균 대비 샘플 간 "
                            "변동 크기(rel_var) — 작으면(<0.05) embedding이 사실상 상수 벡터라 "
                            "bias처럼 작동한다는 뜻(agg_emb_shuffle이 안 먹히는 이유가 '정보가 "
                            "없어서'일 수 있음). (2) PCA 유효 차원(PC1_ratio/n_PC(90%%)) — 변동이 "
                            "있는 부분 안에서 얼마나 다양한 방향으로 퍼져 있는지. (3) query_emb "
                            "로부터의 redundancy(선형회귀 R²) — 높으면(>0.7) 그 branch가 "
                            "query_emb의 중복 정보라 새로 주는 게 없다는 뜻. 순수 forward pass만 "
                            "필요해서 재학습 불필요(--from_saved_state와 같이 쓸 수 있음)."
                        ))
    parser.add_argument("--ablation",  type=str, default="none",
                        choices=["none", "random_neighbor", "neighbor_noise",
                                 "query_emb_zero", "query_emb_shuffle",
                                 "context_emb_zero", "context_emb_shuffle",
                                 "agg_emb_zero", "agg_emb_shuffle",
                                 "rank_correlation", "dual_space_faithfulness",
                                 "interaction_check", "centroid_geometry",
                                 "centroid_representativeness", "evidence_compensation",
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
                            "  query_emb_zero/shuffle : [학습된 모델에 eval 시점만 적용,\n"
                            "                         재학습 없음] head 입력의 query_emb\n"
                            "                         슬롯을 0으로 채우거나(zero) 배치 내\n"
                            "                         셔플(shuffle, permutation importance\n"
                            "                         방식이라 분포 이탈 효과가 작아 더\n"
                            "                         신뢰할 만함). '정상 학습된 모델이\n"
                            "                         query_emb에 실제로 얼마나 의존하는가'를\n"
                            "                         --no_query_emb(처음부터 빼고 재학습 —\n"
                            "                         학습 자체가 붕괴하는지만 보여줌)와\n"
                            "                         별개로 측정.\n"
                            "  context_emb_zero/shuffle : 위와 대칭, context_emb 슬롯 대상.\n"
                            "                         query_emb_* 결과와 나란히 놓고 보면\n"
                            "                         'Explanation①(prototype 배정)이 예측을\n"
                            "                         얼마나 진짜로 설명하는가'에 대한 직접\n"
                            "                         증거가 됨.\n"
                            "  agg_emb_zero/shuffle   : 위와 대칭, agg_emb(검색+attention 집계)\n"
                            "                         슬롯 대상. query_emb_shuffle이 성능을\n"
                            "                         무너뜨린 게 'agg_emb 자체가 기여 없음'\n"
                            "                         때문인지 'query_emb와 agg_emb가 서로\n"
                            "                         다른 샘플 것으로 짝이 어긋나 더\n"
                            "                         헷갈리기' 때문인지 구분하기 위함 — 이\n"
                            "                         모드는 agg_emb만 섞고 query_emb는\n"
                            "                         그대로 둠(반대 조합).\n"
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
                            "  centroid_geometry      : centroid끼리 서로 얼마나 가까운지\n"
                            "                         (cosine_similarity_matrix()) 확인.\n"
                            "                         가까운 쌍이 같은 target을 대표하면\n"
                            "                         (①이 하나의 영역을 여러 centroid로\n"
                            "                         나눠 대표하는 의도된 설계) 정상,\n"
                            "                         다른 target을 대표하면 그 경계의\n"
                            "                         샘플들은 confidence는 낮은데 서사도\n"
                            "                         갈리는 진짜 애매한 케이스일 수 있음.\n"
                            "  centroid_representativeness : centroid_geometry가 못 보는 축 —\n"
                            "                         '이 centroid가 자기한테 배정된 실제\n"
                            "                         샘플들을 얼마나 잘 대표하는가'를 크기가\n"
                            "                         아니라 순도(purity)·응집도(cohesion)\n"
                            "                         기준으로 정렬해서 봄. 큰 centroid도 순도가\n"
                            "                         높으면 정상(밀집 지역), 작은 centroid도\n"
                            "                         순도 100%%면 정당한 outlier 그룹 — 문제는\n"
                            "                         '크지만 순도가 baseline(전역 최다 클래스\n"
                            "                         비율)과 다를 바 없는' centroid.\n"
                            "  evidence_compensation  : centroid_representativeness의 'purity\n"
                            "                         낮음+cohesion 높음'(①이 흐릿한) centroid\n"
                            "                         소속 샘플들만 모아서, ②(dominant weight/\n"
                            "                         entropy)가 다른 샘플들보다 유의하게 더\n"
                            "                         결정적인지 Mann-Whitney U 검정. '①이\n"
                            "                         흐릿한 곳을 ②가 메워준다'는 ①②를 나눠\n"
                            "                         설계한 근거를 직접 검증.\n"
                            "  dataset_profile        : 예측 확신도, fallback 비율 등 빠른\n"
                            "                         데이터셋 진단(예전엔 IG completeness/\n"
                            "                         deletion_auc 포함했으나 ③=SHAP 통일로\n"
                            "                         해당 부분은 제거 — rank_correlation이\n"
                            "                         그 역할을 대신함)."
                        ))
    parser.add_argument("--query_detach_warmup_epochs", type=int, default=0,
                        help=(
                            "[v2, Phase 1-1, 진단/개입용] 학습 시작 후 이 값 이하 "
                            "epoch(1-base, epoch<=N) 동안 head가 보는 query_emb 사본만 "
                            "detach — embedder는 context_emb/agg_emb 경로로 계속 "
                            "classification gradient를 받음(detach_context_grad와 "
                            "대칭 위치, TabERA.forward()의 _query_for_head만 끊음). "
                            "Phase 0에서 확인된 'epoch 1~2 사이에 query gradient가 "
                            "급격히 우세해진다'는 관측을 causal intervention으로 "
                            "검증하기 위함(TabERA_retrieval_failure_analysis.md 참고). "
                            "0(기본값)이면 항상 off — 기존 동작과 100%% 동일. "
                            "--query_detach_warmup_steps와 동시에 0이 아니면 안 됨."
                        ))
    parser.add_argument("--query_detach_warmup_steps", type=int, default=0,
                        help=(
                            "[v2, Phase 1-1] 위와 같으나 epoch 대신 전역 optimizer "
                            "step(배치) 기준. Phase 0의 배치 단위 로그에서 collapse가 "
                            "epoch 1 안(약 20~140 배치 사이)에 대부분 끝나는 게 "
                            "확인돼서, 데이터셋마다 epoch당 배치 수가 다르면 epoch "
                            "기준이 너무 거칠 수 있음 — 작은 데이터셋은 epoch=1이 "
                            "몇 배치 안 될 수 있음. 0(기본값)이면 항상 off. "
                            "--query_detach_warmup_epochs와 동시에 0이 아니면 안 됨."
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
    parser.add_argument("--no_query_emb", action="store_true",
                        help=(
                            "[진단용] head 입력에서 query_emb(양자화 안 된 원본 임베더 출력)를 "
                            "제외. --global_retrieve/--no_context_emb(과거 검증 완료)와 대칭인 "
                            "새 ablation — 지금까지 query_emb가 head에서 빠진 조합은 한 번도 "
                            "테스트된 적이 없었음. --no_context_emb와 반대로 켜면(둘 다 켜면 "
                            "agg_emb만 남는 극단 케이스도 가능, 경고 출력됨) head가 순수 "
                            "quantized 신호(context_emb)만으로 예측하는 vanilla VQ-VAE식 "
                            "bottleneck에 가까워짐. 목적: (a) query_emb의 raw 값이 최종 성능에 "
                            "실제로 얼마나 기여하는지, (b) 그 기여도가 클수록 Explanation①이 "
                            "예측을 얼마나 '진짜로' 설명하는지에 대한 신뢰도가 낮아진다는 "
                            "역상관 관계를 실측하기 위함. --detach_context_grad와 함께 쓰면 "
                            "'context_emb의 값 자체 vs gradient 경로' 중 어느 쪽이 더 "
                            "중요한지도 나눠서 볼 수 있음."
                        ))
    parser.add_argument("--use_context_emb", action="store_true",
                        help=(
                            "[2026-07, v2 freeze — 신규] fusion_mode='residual'에서 "
                            "context_emb를 head 입력에 다시 포함시킴(V1식으로 되돌리기, "
                            "ablation/비교 목적). 기본값(플래그 안 줌)은 이제 False — "
                            "v2 채택 구조(query+β·agg만, context_emb는 head에 안 감)가 "
                            "기본. 이 세션 전체의 controlled comparison이 이 기본값으로 "
                            "돌아갔음(FiLM/Temperature 검증, I(C;Y) 분석 등)."
                        ))
    parser.add_argument("--no_context_emb", action="store_true",
                        help=(
                            "[2026-07, deprecated — 하위호환용] use_context_emb=False가 "
                            "이제 기본값이라 이 플래그는 더 이상 아무 효과가 없음(줘도 "
                            "안전 — 어차피 기본 동작). V1식으로 되돌리려면 "
                            "--use_context_emb를 쓸 것. [예전 help, 참고용] head 입력에서 "
                            "context_emb(centroid 라우팅 결과)를 제외 — --no_query_emb와 "
                            "대칭. fusion_mode='residual'과 같이 쓰면 z=LN(q)+β·LN(a)(context "
                            "항 자체가 빠짐) — v2 gated_sum 실험에서 query-only/agg-only 둘 다 "
                            "AUROC≈0.90인데 fusion이 경쟁으로 하나만 골라 쓴다는 게 "
                            "확인된 뒤, '그냥 query+agg를 경쟁 없이 고정 비율로 더하기만 "
                            "해도 좋아지는가'(cooperative sum, gate 없는 최소 baseline)를 "
                            "보기 위한 용도로 추가됐던 플래그."
                        ))
    parser.add_argument("--ema_codebook", action="store_true",
                        help=(
                            "[구조 변경] codebook_loss(gradient 기반, centroid→query 방향)를 "
                            "EMA(exponential moving average) 업데이트로 대체. commitment_loss"
                            "(query→centroid 방향, embedder에 gradient)는 그대로 유지 — Huh et "
                            "al.(2023)이 정리한 VQ-VAE 표준 구도(EMA는 codebook 쪽만 대체, "
                            "commitment는 gradient 기반 유지)를 그대로 따름. [부작용] "
                            "diversity_loss도 이 모드에서는 자동으로 꺼짐 — centroid_emb가 "
                            "requires_grad=False가 되므로(EMA가 매 배치 .data를 통째로 "
                            "덮어써서 gradient 기반 업데이트와 공존이 안 됨) diversity_loss를 "
                            "계산해도 갈 곳이 없어 아예 호출을 생략함. 즉 이 모드는 "
                            "'codebook_loss만 EMA로'가 아니라 'centroid를 EMA 하나로만 "
                            "위치시키고 밀어내기(diversity) 효과는 포기'하는 트레이드오프임."
                        ))
    parser.add_argument("--ema_decay_override", type=float, default=None,
                        help=(
                            "[통제 실험용] --ema_codebook의 EMA decay(문헌 기본값 0.99 — "
                            "van den Oord et al. 2017 Appendix, VQ-VAE-2/Jukebox/SoundStream "
                            "공통. 이 프로젝트 데이터로 검증된 값 아님, 스윕 대상). "
                            "--ema_codebook 없이는 무효과."
                        ))
    parser.add_argument("--blockwise_layernorm", action="store_true",
                        help=(
                            "[구조 변경] head 입력을 [query‖context‖agg] 하나로 묶어 "
                            "nn.LayerNorm(_head_in) 하나로 정규화하던 기존 방식 대신, "
                            "블록마다 따로 LayerNorm을 건 뒤 concat. 동기: context_emb/"
                            "agg_emb_shuffle ablation에서 acc/auroc는 안 흔들리는데 "
                            "logloss만 폭증하는 게 확인됨(routing/retrieval에 딸린 두 "
                            "슬롯의 값이 흔들릴 때 결합 LayerNorm 통계를 통해 query_emb "
                            "쪽 정규화까지 흔든다는 뜻) — 학습 중에도 dead-centroid 재초기화 "
                            "등으로 이 두 슬롯이 흔들릴 때마다 같은 경로로 embedder gradient에 "
                            "노이즈가 새어들었을 가능성. 기존 체크포인트는 head[0]이 단일 "
                            "LayerNorm이라 이 플래그 없이 저장된 --from_saved_state와는 "
                            "구조가 달라 호환 안 됨(옵트인 기본 False로 하위 호환 유지)."
                        ))
    parser.add_argument("--head_branch_l2norm", action="store_true",
                        help=(
                            "[v1.1, 신규] head 입력 직전(concat 전) query/context/agg 각 "
                            "branch를 sample-wise unit-L2-norm으로 정규화. 기본값 False "
                            "(기존과 100%% 동일 — 하위호환). 동기: --linear_probe 실측 "
                            "(1043/31)에서 concat(q+c+a)가 최고 단일 branch보다도 낮게 "
                            "나오는 현상이 branch별 L2-normalize만으로 상당 부분(1043)~ "
                            "거의 완전히(31) 회복됨을 확인 — StandardScaler(LayerNorm과 "
                            "유사한 차원별 z-score)는 오히려 31에서 더 악화시켜서 '차원별 "
                            "분산'이 아니라 'branch 전체 크기(norm) 격차'가 관련 있다는 "
                            "쪽을 가리킴. 다만 그건 probe(사후 선형 분류기) 수준 관찰이라 "
                            "이 플래그로 실제 end-to-end 재학습 시 정확도가 따라오는지 "
                            "확인하는 게 목적 — 'L2 정규화가 도움이 된다'와 'scale "
                            "imbalance가 원인이다'는 다른 주장이므로 여기서 검증. "
                            "--blockwise_layernorm과 같이 쓰면(권장 안 함, 원 probe와 "
                            "다른 조합) LN 적용 후 L2-normalize가 걸림 — 원 probe를 "
                            "정확히 재현하려면 --blockwise_layernorm 없이 이것만 켤 것. "
                            "켜져 있으면 head[0]의 global LayerNorm(_head_in)이 자동으로 "
                            "빠짐(안 그러면 그 LN이 branch별 unit-norm을 다시 지움 — "
                            "스모크 테스트로 확인된 문제라 tabera.py에서 자동 처리됨). "
                            "기존 체크포인트와 head 구조가 달라 --from_saved_state 호환 "
                            "안 됨(blockwise_layernorm과 같은 성격)."
                        ))
    parser.add_argument("--value_mode", type=str, default="default",
                        choices=["default", "label_only", "offset_only", "balanced",
                                 "offset_normalized", "sum_normalized"],
                        help=(
                            "[재개, ablation] AttentionAggregator의 value = "
                            "label_emb + T(query-neighbour) 구성 방식. "
                            "'default'(기존과 동일, 하위호환): 정규화 없이 그대로 더함. "
                            "'label_only': T() 자체를 안 만듦, value=label_emb만 "
                            "(use_offset_correction=False와 동일 — 이웃 label 정보 "
                            "단독의 유용성 검증). 'offset_only': value=T(query-neighbour)만 "
                            "(label_emb 항을 뺌 — 지금 모델이 사실상 이것만 쓰고 있는지 "
                            "검증). 'balanced': value=LN(label_emb)+LN(T(query-neighbour)) "
                            "(두 항을 unit-scale로 맞춘 뒤 더함). 동기: "
                            "diagnose_value_components 실측에서 T(query-neighbour) 항이 "
                            "label_emb보다 평균 4.9배 크다는 게 확인됨(mfeat-zernike) — "
                            "concat 시절 embed_dim 스케일 격차 문제와 구조적으로 같은 "
                            "패턴이 value 구성 단계에서 재현된 것으로 추정. "
                            "[추가] adult(1590) 실측: offset_only의 agg cos_sim(0.984)이 "
                            "default(0.985)와 거의 동일 — offset norm이 label의 5만 배까지 폭주하며 "
                            "collapse의 지배적 원인으로 확인됨, 그런데 label_only accuracy가 "
                            "default보다 오히려 살짝 높았음(0.852 vs 0.847). 다음 질문("
                            "'offset을 완전히 없애야 하나, scale만 통제해도 되나')을 위해 "
                            "두 개 추가: 'offset_normalized': value=label_emb+"
                            "T(query-neighbour)/||T(query-neighbour)|| (T()의 방향은 "
                            "살리고 크기 폭주만 제거). 'sum_normalized': "
                            "value=(label_emb+T(query-neighbour))/||label_emb+"
                            "T(query-neighbour)|| (최종 합 벡터 자체를 unit-norm으로 강제 — "
                            "'balanced'와 다르게 각 항을 따로 정규화하는 게 아니라 더한 "
                            "결과를 한 번에 정규화)."
                        ))
    parser.add_argument("--gradient_attribution", action="store_true",
                        help=(
                            "[진단용, fusion_mode='concat' 전용] --log_branch_gradients(학습 중 "
                            "epoch마다 기록, 재학습 필요)와 달리, 이미 학습된 모델"
                            "(--from_saved_state)에 eval 데이터를 한 번 흘려서(forward+backward "
                            "1회) branch별(query/context/agg) gradient norm을 재는 가벼운 "
                            "one-shot 측정. head 첫 Linear 입력을 branch별 slice"
                            "(_head_block_slices)로 나눠서 재는 방식이라 fusion_mode='concat'"
                            "([q|c|a]→Linear로 branch별 weight가 분리되는 경우)에만 성립 — "
                            "residual은 fusion 전에 이미 하나의 벡터로 합쳐져 있어 이 slice 개념 "
                            "자체가 없음(자동으로 skip됨). residual에서는 대신 "
                            "--pre_fusion_gradient_attribution을 쓸 것. grad_share가 낮으면 "
                            "loss가 그 branch를 거의 안 거쳐 흐른다는 뜻 — head가 실제로 그 "
                            "branch에 맞춰 업데이트되고 있지 않다는 직접 증거. 재학습 불필요."
                        ))
    parser.add_argument("--head_sensitivity", action="store_true",
                        help=(
                            "[진단용, fusion_mode='concat' 전용 — 이유는 --gradient_attribution과 "
                            "동일(_head_block_slices가 concat에서만 채워짐)] --ablation "
                            "agg_emb_shuffle(다른 real 샘플 값으로 바꿔치기 — 그 값이 우연히 "
                            "비슷하면 효과가 작게 나올 수 있음, 특히 collapse된 표현에서)보다 더 "
                            "직접적인 head sensitivity 측정. head 입력 지점에서 branch를 직접 "
                            "zero(정보 제거)/random(배치 내 셔플)/scale(×10, 정보는 유지하고 크기만 "
                            "키움)로 조작한 뒤 최종 logit이 얼마나 변하는지(L2 거리, 기준 logit "
                            "norm 대비 상대값) 잼. zero도 scaled도 둘 다 낮으면 head가 그 branch의 "
                            "존재/크기 모두에 무감각하다는 강한 증거. 재학습 불필요. residual에서는 "
                            "--ablation agg_emb_zero/scaled 조합이 사실상 같은 역할을 함."
                        ))
    parser.add_argument("--pre_fusion_gradient_attribution", action="store_true",
                        help=(
                            "[진단용, fusion_mode='residual' 전용] --gradient_attribution의 "
                            "residual 버전. head 첫 Linear 입력이 아니라, fusion **이전**의 raw "
                            "query_emb/agg_emb/context_emb(out dict에 fusion_mode와 무관하게 항상 "
                            "노출됨)에 직접 retain_grad()를 걸어 backward 1회로 gradient norm을 "
                            "잼 — residual(z=q+βa)은 head 진입 전에 이미 branch들이 하나의 "
                            "벡터로 합쳐져 있어 --gradient_attribution의 slice 기반 접근이 "
                            "구조적으로 성립하지 않기 때문에 별도로 둠(억지로 같은 함수를 고쳐 "
                            "쓰지 않음 — concat 전용 분석은 그대로 남겨둠). 분석계획 4번(∂loss/"
                            "∂query_emb vs ∂loss/∂agg_emb)에 직접 답하는 지표. 재학습 불필요, "
                            "--from_saved_state와 같이 쓸 수 있음."
                        ))
    parser.add_argument("--head_input_cancellation", action="store_true",
                        help=(
                            "[진단용, fusion_mode='residual' 전용] head 첫 Linear(W, bias b)가 "
                            "선형이라는 사실만으로 W@z+b = (W@LN(q)+b) + β·(W@LN(a))가 항상 "
                            "정확히 성립함(_head_block_slices 불필요 — '합 다음에 선형 레이어'"
                            "라는 구조 자체가 보장하는 항등식). 이 h_q=W@LN(q)+b(=agg_emb_zero "
                            "ablation이 만드는 값과 동일)와 h_a=β·(W@LN(a))(bias 없음) 사이의 "
                            "cos/norm을 재서, representation(‖z-q‖=‖β·LN(a)‖)은 크게 움직이는데 "
                            "accuracy는 거의 안 변하는 현상이 head 진입 직후(첫 hidden layer)에서 "
                            "이미 상쇄되기 때문인지 직접 검증. cos(h_q,h_a)<0이고 "
                            "cancellation_ratio(=‖h_q+h_a‖/(‖h_q‖+‖h_a‖))가 1보다 뚜렷이 작으면 "
                            "상쇄 — raw embedding 레벨의 cos(query_emb,agg_emb) 음수 부호가 head를 "
                            "거치며 사라지는지/유지되는지/증폭되는지 비교할 것. 재학습 불필요, "
                            "--from_saved_state와 같이 쓸 수 있음."
                        ))
    parser.add_argument("--neighbor_interaction_mode", type=str, default=None,
                        choices=[None, "attn", "capacity_baseline", "interaction_free_baseline"],
                        help=(
                            "[v2, 신규 ablation] pooling(evidence_w 가중합) 전에 k개 "
                            "이웃 values끼리 상호작용시킬지. None(기본값, 기존과 100%% "
                            "동일 — 하위호환): v1 그대로, 상호작용 없음. 'attn': "
                            "NeighborInteractionBlock(v2 후보 A) — 이웃끼리만 self-"
                            "attention(query token 없음, FFN 없음, 1 layer). "
                            "'interaction_free_baseline': attn과 파라미터 수 정확히 "
                            "동일한 nn.MultiheadAttention을 쓰되 attn_mask로 이웃 간 "
                            "mixing만 구조적으로 차단(핵심 necessity 대조군 — attn과 "
                            "이 값을 나란히 비교해야 'mixing 자체가 원인'과 'capacity/"
                            "projection 증가가 원인'을 가를 수 있음). "
                            "'capacity_baseline': 느슨한 MLP capacity 대조군(파라미터 "
                            "수 정밀 매칭 안 함, 참고용). evidence.py의 각 클래스 "
                            "docstring 참고. fusion_mode/value_mode와 같은 성격의 구조적 "
                            "선택이라 optimize.py에는 threading 안 함 — reproduce.py "
                            "진단/ablation 전용. [주의] 이 ablation이 검증하는 건 "
                            "'single-vector pooling이 병목인가'이지 'Aggregator vs "
                            "Head 전체 문제'의 완전한 답은 아님."
                        ))
    parser.add_argument("--interaction_n_heads", type=int, default=2,
                        help=(
                            "--neighbor_interaction_mode가 'attn' 또는 "
                            "'interaction_free_baseline'일 때만 의미 있음 — "
                            "NeighborInteractionBlock/NeighborInteractionFreeBaseline의 "
                            "multi-head attention head 수."
                        ))
    parser.add_argument("--aggregator_mode", type=str, default="pooling",
                        choices=["pooling", "cross_attention"],
                        help=(
                            "[v2 최종안, 신규] 'pooling'(기본값, 기존과 100%% 동일 — "
                            "하위호환): AttentionAggregator의 고정 weighted-sum. "
                            "'cross_attention': AttentionAggregator를 아예 안 쓰고, "
                            "head 내부 단일 cross-attention(evidence.py의 "
                            "HeadCrossAttention, n_heads=1, layer 1개)이 agg_emb 자리를 "
                            "대체 — retrieve()/value 구성(label_emb+T(query-neighbour))은 "
                            "그대로, pooling만 교체. updated_query = query_emb + "
                            "alpha*attn_out(residual) — 이미 query_emb 정보를 담고 "
                            "있으므로, 설계 의도대로 2-branch([updated_query‖context_emb])로 "
                            "쓰려면 --no_query_emb를 반드시 같이 줄 것(안 주면 query_emb가 "
                            "중복으로 head에 또 들어감, 3-branch가 됨 — 실험 목적에 안 맞음, "
                            "실수 방지용으로 여기서는 자동으로 강제하지 않고 명시적으로 "
                            "같이 주도록 요구). evidence_w가 이 모드에서는 실제 예측에 "
                            "쓰인 attention weight 그 자체라 causal claim으로 취급 가능 "
                            "(v1은 head가 agg_emb를 안 써서 descriptive claim으로만 "
                            "제한해야 했음 — evidence.py의 HeadCrossAttention.explain_evidence "
                            "docstring 참고)."
                        ))
    parser.add_argument("--head_attn_alpha_override", type=float, default=None,
                        help=(
                            "--aggregator_mode cross_attention일 때만 의미 있음. "
                            "HeadCrossAttention의 residual scale alpha를 학습 대신 이 "
                            "값으로 고정. 0.0을 주면 updated_query=query_emb가 되어 "
                            "retrieval 분기를 완전히 끈 necessity baseline이 재현됨 "
                            "(파라미터 수는 그대로 두고 정보 흐름만 차단 — "
                            "fusion_alpha_override=0과 같은 성격의 검증)."
                        ))
    parser.add_argument("--head_neighbor_source", type=str, default="real",
                        choices=["real", "learned_const", "shuffled"],
                        help=(
                            "--aggregator_mode cross_attention일 때만 의미 있음. "
                            "'real'(기본값): 실제 검색된 이웃. 'learned_const': K/V를 "
                            "검색 결과 대신 학습 가능한 상수 토큰(k개)으로 완전히 대체 — "
                            "attention 모듈 파라미터 수는 'real'과 100%% 동일, 늘어나는 "
                            "건 상수 토큰 자체뿐. 'shuffled': 매 forward마다(학습 중 포함) "
                            "배치 내에서 K/V를 무작위로 섞음 — learned_const와 달리 매 "
                            "배치 다른 real 이웃 벡터 분포는 보되 '이 query와 이 이웃의 "
                            "실제 대응'만 학습 내내 원천적으로 차단. 셋 다 attention 모듈 "
                            "파라미터 수는 동일 — '실제 검색 결과 없이도 cross-attention "
                            "형태/capacity만으로 좋아지는가'를 서로 다른 각도로 격리하는 "
                            "capacity-only 대조군(재학습 필요 — 처음부터 이 모드로 학습해야 "
                            "의미 있음, post-hoc 전환 아님)."
                        ))
    parser.add_argument("--allow_self_retrieval", action="store_true",
                        help=(
                            "[기본값 변경] 기본은 이제 self-retrieval 제외(exclude)가 켜져 "
                            "있음 — 이 플래그를 주면 예전 기본 동작(제외 안 함)으로 되돌림. "
                            "MemoryBank 검색 시 쿼리 자신과 sample_id가 같은 슬롯(이전 epoch에 "
                            "저장해둔 자기 자신)을 후보에서 배제하는 게 기본 — MemoryBank가 "
                            "label을 그대로 저장/반환하므로(self-retrieval 시 그 슬롯의 "
                            "neighbour_label은 자기 자신의 진짜 정답) 배제하는 쪽이 구현상 더 "
                            "정확함. 다만 이 옵션은 agg_emb의 predictive null 결과를 바꾸기 "
                            "위한 게 아님 — 사전 분석(self-retrieval 비율과 agg-only 성능 간 "
                            "뚜렷한 상관 없음)에서 이미 그 가설은 기각됨, 순수 구현 정확성 "
                            "차원. '이례적 경로'(초대형 centroid 그룹, 드문 경우)는 기본 켜짐 "
                            "상태에서도 아직 미반영(exclusion 적용 안 됨) — 재현 목적으로 예전 "
                            "결과와 정확히 비교하려면 이 플래그로 예전 동작을 켤 것."
                        ))
    parser.add_argument("--fusion_mode", type=str, default="residual",
                        choices=["concat", "residual", "gated_sum", "anchor_gate", "context_gated_beta"],
                        help=(
                            "[2026-07, v2 freeze — 기본값 변경] TabERA v2 최종 architecture로 "
                            "'residual'이 채택되어 기본값을 이걸로 바꿈 — 이 세션의 모든 "
                            "controlled comparison이 실제로 이 설정으로 돌아갔음. 더 이상 "
                            "이 플래그를 매번 명시할 필요 없음. 'concat'(V1식, 예전 기본값)은 "
                            "이제 ablation/비교 목적으로만 명시적으로 선택. "
                            "head가 [query,context,agg]를 합치는 방식. "
                            "'concat'(V1식): [query‖context‖agg] → 공유 MLP. "
                            "'residual'(v2 기본값): z = LN(q) + α·LN(c) + β·LN(a) (α,β 학습 가능한 "
                            "스칼라) → embed_dim 크기 z 하나만 MLP에 통과. 동기: "
                            "freeze_encoder_retrain_head 5-seed 실험(mfeat-zernike, "
                            "embed_dim=256, evM_cosine, sharedLN/blockLN 둘 다)에서 "
                            "인코더 고정+head 백지 재학습을 해도 원래 공동학습 head와 "
                            "통계적으로 구분 안 되는 정확도(양쪽 paired p>0.4, d<0.2)로 "
                            "수렴 — concat+공유 MLP 구조 자체가 정보를 못 끌어쓴다는 "
                            "가설(시나리오 A)에 대한 직접 대응. residual 모드는 branch별 "
                            "LayerNorm이 blockwise_layernorm 플래그와 무관하게 항상 켜짐. "
                            "'gated_sum'(v2, Phase 2): g_q,g_c,g_a = softmax(MLP([LN(q),"
                            "LN(c),LN(a)])) → h = g_q·LN(q)+g_c·LN(c)+g_a·LN(a) → embed_dim "
                            "크기 h 하나만 MLP에 통과. residual과의 핵심 차이 — (1) g는 "
                            "전체 데이터셋 공통 scalar(α,β)가 아니라 샘플마다 다른 값(gate "
                            "MLP가 세 branch를 다 보고 계산), (2) softmax라 g_q+g_c+g_a=1 "
                            "강제(sigmoid처럼 셋 다 낮게/높게 나오는 scale ambiguity 없음), "
                            "(3) query도 gate 대상(residual은 query 계수가 고정 1). "
                            "동기: residual 3-seed 실험(adult/1590, offset_normalized)에서 "
                            "α≈0.01, β≈0.04~0.07로 수렴 — 학습 가능한 global scalar "
                            "reweighting도 query shortcut을 못 풀고 head가 스스로 "
                            "context/agg를 거의 0으로 억제했음. gate가 sample-dependent로 "
                            "branch 중요도를 조절할 수 있으면 이 문제가 풀리는지 검증하기 "
                            "위함. gated_sum도 branch별 LayerNorm이 항상 켜짐. 기존 "
                            "체크포인트와 파라미터 구조가 달라(신규 fusion_gate_mlp) "
                            "concat/residual 체크포인트로는 --from_saved_state 호환 안 됨. "
                            "'anchor_gate'(v2, Phase 2 후속): h = LN(q) + σ(MLP([LN(q),"
                            "LN(a)]))·LN(a) → MLP. 동기: gated_sum 3-seed 실험에서 "
                            "query-only/agg-only 체크포인트에 각각 query_emb_shuffle/"
                            "agg_emb_shuffle을 돌려보니 둘 다 Δauroc≈-0.38~-0.40(AUROC "
                            "0.90→0.51, 거의 완전 랜덤)로 나옴 — query도 agg도 개별적으로 "
                            "이미 강한 예측 정보를 담고 있는데, gated_sum의 softmax가 "
                            "g_q+g_c+g_a=1을 강제해서 항상 하나만 선택(competition)하고 "
                            "있었다는 게 확인됨. anchor_gate는 그 제약 자체를 제거 — query는 "
                            "항상 계수 1(anchor, gate 대상 아님), agg만 sigmoid gate(g∈(0,1), "
                            "합 제약 없음)로 조절해서 query+agg가 동시에 완전히 반영되는 것도 "
                            "구조적으로 가능하게 함(softmax였으면 불가능). context는 이 "
                            "fusion에 안 들어감(query/agg 개별 강도가 이미 확인된 뒤 우선순위 "
                            "에서 제외 — routing/aux_loss는 use_context_emb에 따라 그대로 "
                            "돌아감, head 입력에만 안 쓰일 뿐). 성공 기준: query-only(~0.90)/"
                            "agg-only(~0.90)보다 anchor_gate의 AUROC가 실제로 더 높아지는가. "
                            "'context_gated_beta'(v2, Phase 2 후속): h = LN(q) + β(context)·"
                            "LN(a), β(context) = σ(MLP(LN(context_emb))). anchor_gate와 "
                            "결정적 차이 — gate 입력이 agg가 아니라 context_emb(centroid "
                            "라우팅 결과). 동기: (1) [q,a] 입력 gate(anchor_gate)도 매 seed "
                            "0 또는 1로 collapse함을 확인 — '이 특정 agg가 좋은가'를 매 샘플 "
                            "새로 판단하게 하는 것 자체가 collapse를 유발할 수 있다는 가설. "
                            "'이 centroid 지역은 retrieval을 얼마나 신뢰할까'라는, 같은 "
                            "centroid의 샘플들끼리 거의 같은 값이 나올 저차원 신호로 gate "
                            "입력을 제한. (2) fixed β sweep(adult/1590)에서 β=1.5가 seed1 "
                            "단독/짧은 스케줄로는 최고였지만 3-seed 정식 스케줄에서는 자유 "
                            "학습 β(0.02~0.06, AUROC 0.9063±0.0006)보다 낮음(0.9029±0.0019) "
                            "— 전체 데이터에 동일한 β를 강제하는 것 자체가 이미 최적이 아닐 "
                            "수 있다는 증거. context는 use_context_emb 설정과 무관하게 항상 "
                            "쓰임(전용 LayerNorm). 성공 기준: (a) AUROC가 cooperative sum "
                            "(0.9063)보다 높아지는가, (b) centroid별 β 평균의 분산이 유의미"
                            "하게 존재하는가(그렇지 않으면 그냥 전역 상수 β를 복잡하게 "
                            "재현한 것에 불과). meta.pkl에 샘플별 (centroid_id, β) 쌍을 "
                            "저장해서 사후 분석 가능."
                        ))
    parser.add_argument("--fusion_gate_temperature", type=float, default=1.0,
                        help=(
                            "[v2, Phase 2 후속, 진단/개입용] fusion_mode='gated_sum' 전용. "
                            "g = softmax(gate_logits / T). 동기: gated_sum 3-seed 실험"
                            "(adult/1590)에서 T=1(기본)이 epoch 14~22 사이 entropy→0으로 "
                            "완전 collapse(seed마다 다른 단일 branch로 winner-take-all — "
                            "seed1/2는 query=1, seed3는 agg=1). 대조 실험(합성 데이터, "
                            "toy 신호)으로 이 collapse가 초기화 시점엔 없고(균등 상태로 "
                            "시작) 실제 예측 신호가 있을 때만 학습 중 progressive하게 "
                            "진행됨을 확인 — gate MLP 자체의 구조적 편향이 아니라 softmax "
                            "의 winner-take-all positive-feedback 학습 동역학(무작위 라벨 "
                            "대조군은 40 step 내내 collapse 없음). T>1로 올리면 같은 "
                            "logit 차이에도 확률분포가 덜 뾰족해짐(T→∞는 균등, T=1.0은 "
                            "기존과 100%% 동일 — 하위호환). 목적은 '이게 최종 해법이다'가 "
                            "아니라 'collapse를 억제하면 necessity가 살아나는가?'를 값싸게 "
                            "먼저 검증하는 것(entropy_regularization/load_balancing/"
                            "Gumbel-softmax보다 구현이 훨씬 단순해서 우선). fusion_mode!="
                            "'gated_sum'이면 무의미(모델 생성 시 ValueError)."
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
    parser.add_argument("--num_embedding", type=str, default="ple",
                        choices=["linear", "ple", "plr_lite"],
                        help=(
                            "numeric feature 인코딩 방식. 'ple'(기본값, 채택 확정 — 2026-07 갱신)은 "
                            "PiecewiseLinearEmbeddings(activation=False, Gorishniy et al. 2022) — "
                            "TabM(Gorishniy et al. 2024)이 기본값으로 권장하는 것과 동일 구조 "
                            "(feature별 학습 가능한 (n_bins, d_embedding) 가중치로 bin 인코딩을 "
                            "가중합 — 예전엔 이 가중치 없이 raw bin 벡터를 그대로 내보내는 "
                            "PiecewiseLinearEncoding이었음, TabM 기본값과 달랐던 걸 이번에 맞춤). "
                            "4개 데이터셋(profb/vehicle/credit-g/jasmine) 실측 근거: PLR 대비 val "
                            "붕괴(무작위 수준 trial)가 0건으로 감소(PLR은 vehicle 2건, credit-g 1건 "
                            "발생) + routing_scale/PLR 3종이 탐색 공간에서 빠져 HPO가 13→9차원으로 "
                            "축소됨. 다만 top5-test 성능은 데이터셋마다 갈렸고(4개 중 1개만 PLE "
                            "우세), centroid margin_percentile은 4개 전부 PLE가 더 낮게 나옴(원인 "
                            "미상) — '성능 우위'가 아니라 '재앙적 실패 방지 + 탐색 단순화'가 채택"
                            "근거임을 분명히 해둠. 'plr_lite'는 이전 기본값(TabR/ModernNCA 계보, "
                            "학습 가능한 주기함수 + 공유 Linear+ReLU) — 필요시 여전히 선택 가능. "
                            "'linear'는 raw 값을 그대로 Linear에 투영 — 기존 동작, 하위 호환용."
                        ))
    parser.add_argument("--evidence_metric", type=str, default="cosine",
                        choices=["euclidean", "cosine", "cosine_scaled"],
                        help=(
                            "AttentionAggregator(evidence_w, 설명②)의 유사도 공간 — cat_combine/"
                            "num_embedding과 같은 성격의 구조 선택(Optuna 탐색 대상 아님). "
                            "[기본값 변경] euclidean → cosine. euclidean은 evidence collapse"
                            "(정규화 안 된 유클리드 거리가 query_emb norm 성장에 종속돼 evidence_w가 "
                            "사실상 1-NN으로 붕괴, n_eff≈1.0)가 4데이터셋×5seed로 확정된 채로 남아있던 "
                            "값이라 기본값으로 두는 게 더 이상 맞지 않음 — cosine이 이미 여러 세션에 "
                            "걸쳐 검증된 해결책(n_eff≈7.5~12, paired t-test 전부 p<0.005). "
                            "[주의] 이 값에 따라 optimize.py가 찾는 HPO study 파일이 달라짐"
                            "(study_pkl_tag가 cosine이면 '..evM_cosine' 태그 추가) — cosine 전용으로 "
                            "HPO를 아직 안 돌린 데이터셋에서는 study를 못 찾을 수 있음. 그 경우 "
                            "'--evidence_metric euclidean'으로 명시하거나 optimize.py를 "
                            "'--evidence_metric cosine'으로 먼저 돌릴 것. "
                            "optimize.py --evidence_metric으로 이 값에 맞춰 HPO를 새로 돌린 뒤, "
                            "여기서도 같은 값을 줘야 그 study를 찾음(study_pkl_tag가 파일명에 "
                            "반영). euclidean이면 기존과 완전히 동일 — 태그 없음. "
                            "--evidence_metric_override(아래)와 다른 점: 이건 '그 metric으로 "
                            "HPO된 study를 불러와서 재학습'이고, override는 '기존 euclidean "
                            "study의 best_params에 이 값만 강제로 바꿔치기해서 재학습'(정식 "
                            "HPO 없이 빠르게 확인하는 용도) — 둘 다 주면 override가 우선."
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
    parser.add_argument("--confidence_scaling", action="store_true",
                        help=(
                            "[진단용] head에 들어가는 context_emb에 assignment "
                            "confidence(top1_confidence — 실제 라우팅 soft에서 선택된 "
                            "centroid의 확률, STE의 routing_probs와 달리 샘플마다 실제로 "
                            "다름)를 곱함. 라우팅/검색 자체는 안 건드리고 head가 받는 "
                            "신호의 크기만 조절 — 'context_emb norm이 지금(M=1, "
                            "unit-norm centroid 그대로라 샘플 간 변동 0.3%% 수준)처럼 "
                            "정보가 없는 상태를 의도적으로 깨서, 애매한 배정은 head가 "
                            "덜 신뢰하게 만들 수 있는가'를 검증. --confidence_scaling_"
                            "detach와 조합해 Variant A(gradient 있음)/B(없음) 비교 가능. "
                            "부작용 가능성: 애매한 샘플(confidence 낮음)은 gradient도 "
                            "같이 작아짐(Variant A에서만) — 학습이 오히려 불안정해질 수 "
                            "있어 검증 안 된 개입."
                        ))
    parser.add_argument("--confidence_scaling_detach", action="store_true",
                        help=(
                            "--confidence_scaling과 함께 쓸 때만 의미 있음(Variant B). "
                            "confidence 값 자체는 곱하되 그 경로로 gradient는 안 흐르게 "
                            "detach — '크기 조절 효과'와 'gradient 흐름 변화'를 분리해서 "
                            "보기 위함."
                        ))
    parser.add_argument("--context_projection", action="store_true",
                        help=(
                            "[구조 조정] context_emb를 head로 보내기 전 학습 가능한 "
                            "Linear를 하나 거치게 함. detach_context_grad와 달리 "
                            "gradient가 여전히 centroid_emb까지 도달함. optimize.py "
                            "--context_projection으로 학습한 study가 있으면 그 "
                            "best_params를 쓰는 게 이상적이지만, 없으면 기존 study "
                            "best_params 위에 이 구조만 얹어 1회 재학습(별도 study 불필요)."
                        ))
    parser.add_argument("--loss_commitment_override", type=float, default=None,
                        help=(
                            "[통제 실험용] best_params의 loss_commitment 값을 이 값으로 "
                            "덮어쓰고 나머지는 그대로 재학습. jasmine(euclidean 0.010 → "
                            "cosine 전용 HPO 0.097, 약 10배)과 mfeat-zernike(0.045 → "
                            "0.071, 약 1.6배) 둘 다 cosine에서 commitment가 커졌는데, "
                            "agg/query gradient는 정반대로 갈렸음(jasmine↓ mfeat-zernike↑) "
                            "— commitment_loss 자체가 이 차이의 원인인지, 아니면 다른 "
                            "하이퍼파라미터(embed_dim 등)와 우연히 같이 바뀐 것뿐인지 "
                            "분리 검증하는 용도. loss_codebook_override와 같은 패턴."
                        ))
    parser.add_argument("--loss_diversity_override", type=float, default=None,
                        help=(
                            "[통제 실험용] best_params의 loss_diversity 값을 이 값으로 "
                            "덮어쓰고 나머지는 그대로 재학습. mfeat-zernike cosine 전용 "
                            "HPO에서 diversity가 크게 줄었음(0.361→0.058, 약 6배) — "
                            "이것도 agg gradient 변화의 후보 원인 중 하나로 같이 확인."
                        ))
    parser.add_argument("--loss_codebook_override", type=float, default=None,
                        help=(
                            "[통제 실험용] best_params의 loss_codebook 값을 이 값으로 "
                            "덮어쓰고 나머지 하이퍼파라미터는 그대로 재학습. codebook_loss "
                            "도입 전후 val_acc/centroid_geometry(z_margin)/"
                            "centroid_representativeness(purity) 변화가 codebook_loss "
                            "자체 때문인지, 아니면 HPO가 다른 조합에 정착한 우연 때문인지 "
                            "(탐색 차원이 하나 늘어난 것 포함) 갈라내려는 용도. 예:\n"
                            "  --loss_codebook_override 0.0   → codebook_loss 끄고 재학습\n"
                            "  --loss_codebook_override 0.044 → best_params가 찾은 값 그대로\n"
                            "(둘을 같은 seed로 각각 돌려서 나머지 파라미터 동일 조건에서 "
                            "비교). --from_saved_state와 같이 쓰면 재학습을 안 하므로 "
                            "아무 효과가 없다 — 경고만 찍고 무시됨."
                        ))
    parser.add_argument("--regroup_log_every", type=int, default=10,
                        help=(
                            "[진단용] [Regroup] 로그를 몇 epoch마다 찍을지. 기본 10(기존과 "
                            "동일). trial의 active_ratio/reinit 추이를 더 촘촘히 보고 싶을 "
                            "때(예: 10epoch 간격으로는 마지막 구간에서 실제로 안정됐는지 "
                            "판단이 안 될 때) 1~2로 낮춰서 재실행. --from_saved_state와 "
                            "같이 쓰면 재학습 자체를 안 하므로 아무 효과가 없다."
                        ))
    parser.add_argument("--log_branch_gradients", action="store_true",
                        help=(
                            "[진단용] head concat 직전(query_emb/context_emb/agg_emb) "
                            "활성값에 retain_grad()를 걸어, epoch마다 브랜치별 "
                            "gradient norm·activation norm·head 첫 Linear의 block별 "
                            "weight norm을 기록(self.branch_gradient_history). "
                            "--ablation *_shuffle/zero(학습 끝난 뒤 정적 진단)와 달리 "
                            "학습 '과정 중' 각 브랜치가 얼마나 학습 신호를 받는지를 "
                            "본다 — 'head가 query_emb에만 의존하도록 학습되는가'(멀티모달 "
                            "학습의 modality imbalance/greedy learning 문헌과 구조적으로 "
                            "유사한 현상) 진단용. retain_grad()는 값 자체를 안 바꾸므로 "
                            "학습 결과(가중치/예측)에는 영향 없음(메모리만 소폭 증가). "
                            "[주의] gradient가 작다는 것과 head가 그 브랜치를 실제로 "
                            "안 쓴다는 것은 다른 얘기다 — 반드시 --ablation "
                            "context_emb_shuffle/agg_emb_shuffle 결과와 같이 해석할 것. "
                            "학습 후 meta.pkl에 branch_gradient_history/"
                            "branch_gradient_batch_history로 저장됨. "
                            "--from_saved_state와 같이 쓰면 재학습 자체를 안 하므로 "
                            "아무 효과가 없다."
                        ))
    parser.add_argument("--log_branch_gradients_first_n_epochs", type=int, default=3,
                        help=(
                            "--log_branch_gradients의 배치 단위 세부 기록"
                            "(branch_gradient_batch_history)을 처음 몇 epoch만 남길지. "
                            "학습 전체에 걸쳐 배치 단위로 남기면 메모리가 계속 쌓이는데, "
                            "OGM 계열 문헌이 강조하는 게 '초기 학습 dynamics'라 초반만 "
                            "촘촘히 보면 충분하다는 판단(검증 안 된 기본값 3, 필요시 조정). "
                            "이후 epoch는 epoch 평균(branch_gradient_history)만 남음."
                        ))
    parser.add_argument("--log_evidence_stats", action="store_true",
                        help=(
                            "[진단용] evidence_w(②의 AttentionAggregator 가중치)의 "
                            "entropy·dominant weight를 epoch마다 기록"
                            "(meta.pkl의 evidence_stats_history). --explain은 학습 "
                            "끝난 뒤 소수 샘플만 보여줘서 '언제부터 evidence가 "
                            "소수 이웃으로 붕괴됐는지'를 알 수 없었는데, 이 진단은 "
                            "학습 전체 epoch에 걸친 추세를 정량적으로 보여줌. "
                            "entropy가 0에 가깝고 dominant_weight가 1에 가까울수록 "
                            "사실상 1개 이웃만 보는 hard 1-NN으로 붕괴했다는 뜻. "
                            "backward/retain_grad 불필요한 순수 forward 통계라 "
                            "--log_branch_gradients보다 오버헤드 적음."
                        ))
    parser.add_argument("--export_centroid_retrieval_behavior", action="store_true",
                        help=(
                            "[Centroid Retrieval Behavior Analysis, 신규] 특정 모듈"
                            "(Temperature 등)을 정당화하기 위한 진단이 아니라, TabERA의 "
                            "retrieval 특성 자체를 이해하기 위한 독립적 진단 — 결과가 "
                            "'새 모듈이 필요하다'로 이어질 수도 '필요 없다'로 이어질 "
                            "수도 있음. --log_evidence_stats가 epoch 전체 평균 하나만 "
                            "주는 것과 달리, X_test 샘플별로 (centroid_id, "
                            "routing_confidence, topk_idx, entropy, N_eff, "
                            "top1_weight)를 저장(*_centroid_retrieval_behavior.npz) — "
                            "centroid_id로 groupby해서 (a) group마다 evidence "
                            "distribution이 실제로 다른가, (b) routing_confidence와 "
                            "entropy 사이 상관관계(확신 있는 group일수록 이미 좁게 "
                            "retrieval하고 있는가), (c) 같은 centroid 안에서 topk_idx/"
                            "top1 neighbor label이 안정적인가(retrieval consistency) "
                            "를 직접 확인하기 위함. 새 모델 파라미터/구조 변경 전혀 "
                            "없음(evidence.py/tabera.py는 topk_idx/routing_confidence "
                            "를 out dict에 노출만 함) — 이미 forward()가 반환하는 "
                            "값들만 사용. test-time(model.eval(), dropout 비활성)에서만 "
                            "계산하므로 --log_evidence_stats가 겪었던 학습 중 dropout "
                            "재정규화 문제와 무관 — raw evidence_w를 그대로 씀. "
                            "N_eff=1/Σw_i²(유효 이웃 수), top1_weight=max(w_i). "
                            "[2026-07, 추가] retrieval_label_purity(top-k 이웃 중 query와 "
                            "같은 라벨 비율, unweighted)/retrieval_weighted_label_purity"
                            "(evidence_w로 가중 — attention이 정답 쪽에 얼마나 잘 집중하는지"
                            "까지 반영, purity와의 차이 자체가 신호). model.memory.labels를 "
                            "topk_idx로 바로 조회(새 forward 없음). tasktype=regression이면 "
                            "label purity 개념이 없어 저장 안 됨. "
                            "baseline/V2 모델을 포함해 항상 계산 가능. [추가] similarity_"
                            "top1/bottomk/margin/std(raw similarity geometry, softmax "
                            "이전)와 y_true/correct(분류)-또는-error(회귀는 squared "
                            "error, 분류는 per-sample logloss)도 같이 저장 — centroid별로 "
                            "'실제 예측 품질'까지 groupby해서 볼 수 있음(단순 표본 수/"
                            "margin만으로는 '좋은 local expert인가'를 알 수 없다는 지적 "
                            "반영). logits는 이미 예측을 위해 계산된 값을 재사용(추가 "
                            "forward 없음). [추가, evidence utilization 진단] "
                            "fusion_mode='residual'일 때만: cos_qa(raw query_emb·agg_emb "
                            "샘플별 cosine — ≈1이면 agg_emb가 사실상 query_emb의 중복 "
                            "복사본, 0.2~0.5대인데 accuracy 효과가 있으면 진짜 새 정보), "
                            "query_emb_norm/agg_emb_norm(샘플별 raw norm), "
                            "beta_agg_ratio(β·‖agg_emb‖/‖query_emb‖, 샘플별 — mean/median/"
                            "5%%/95%% 등 분포로 봐야 함), representation_shift_norm(‖z-q‖, "
                            "agg_emb(+context_emb)가 query_emb를 실제로 얼마나 이동시켰는가) "
                            "도 같이 저장. fusion_mode≠'residual'이면 β가 정의되지 않아 "
                            "이 5개 필드는 저장되지 않음(콘솔에 안내 출력)."
                        ))
    parser.add_argument("--log_fusion_trajectory", action="store_true",
                        help=(
                            "[진단용] fusion_mode=residual일 때 α/β와 branch norm"
                            "(||LN(q)||/||LN(c)||/||LN(a)||)을 epoch마다 기록"
                            "(meta.pkl의 fusion_trajectory_history). 지금까지는 최종값만 "
                            "있어서 '처음부터 거의 안 움직였다'와 '오르내리다 지금 값에 "
                            "안착했다'를 구분 못 했음. norm까지 같이 봐야 'α≈1'이라는 "
                            "숫자 자체가 실제 기여량과 비례하는지 판단 가능 "
                            "(||LN(q)‖≫||αLN(c)||면 α가 1이어도 사실상 안 쓰는 것과 같음)."
                        ))
    parser.add_argument("--log_centroid_label_mi_trajectory", action="store_true",
                        help=(
                            "[2026-07, 신규] I(C;Y)/H(Y) — centroid 배정이 label을 "
                            "얼마나 설명하는가 — 를 epoch마다 검증 세트 기준으로 기록"
                            "(meta.pkl의 centroid_label_mi_history). 새 지표가 아니라 "
                            "--export_centroid_retrieval_behavior 분석에서 이미 검증한 "
                            "지표(cross-dataset corr(I(C;Y)/H(Y), AUROC)≈0.92)를 최종값 "
                            "하나가 아니라 학습 중 궤적으로 보기 위함 — 'prototype이 "
                            "label-aware partition으로 조직되는 과정'을 직접 보여줄 수 "
                            "있는지 확인. embedder+prototype_layer만 거치는 가벼운 "
                            "추가 forward(retrieve/aggregate/head 불필요). "
                            "tasktype=regression이면 label이 연속값이라 무의미 — 그 "
                            "경우 항상 빈 리스트로 남음(에러는 안 남)."
                        ))
    parser.add_argument("--log_shuffle_ablation_trajectory", action="store_true",
                        help=(
                            "[2026-07, 신규] 'retrieval은 optimization scaffold(학습 중엔 "
                            "query representation 형성에 기여하지만, 추론 시점엔 query "
                            "자체가 이미 충분해져서 agg 의존도가 낮다)' 가설 검증용. "
                            "inference-time --ablation query_emb_shuffle/agg_emb_shuffle과 "
                            "정확히 같은 조작(model.forward()의 ablation_mode 그대로 재사용, "
                            "새 model 코드 없음)을 검증 세트에 대해 epoch마다"
                            "(--regroup_log_every 간격) 반복해서 accuracy delta를 기록"
                            "(meta.pkl의 shuffle_ablation_trajectory_history). "
                            "--log_branch_gradients의 agg_grad_share가 학습 내내 "
                            "20~40%%를 유지해도, 이 델타가 학습 후반으로 갈수록 0에 "
                            "가까워진다면 — '학습에는 쓰이지만 추론엔 덜 쓰인다'는 "
                            "가설이 직접 뒷받침됨. retrieve/aggregate/head까지 다 거치는 "
                            "forward를 3회(none/query_emb_shuffle/agg_emb_shuffle) 반복하므로 "
                            "log_centroid_label_mi_trajectory보다 비쌈 — 매 epoch 대신 "
                            "--regroup_log_every 간격으로만 계산. tasktype=regression이면 "
                            "accuracy 개념이 없어 빈 리스트로 남음."
                        ))
    parser.add_argument("--log_representation_drift_trajectory", action="store_true",
                        help=(
                            "[2026-07, 신규/수정] 'encoder가 agg가 주던 방향을 query "
                            "representation 안으로 점점 흡수(internalize)한다'는 가설 "
                            "검증용. --log_shuffle_ablation_trajectory가 retrieval "
                            "contribution이 '감소한다'는 건 보여주지만 '왜' 감소하는지는 "
                            "간접 증거였음 — 이건 representation 자체의 이동을 직접 잼. "
                            "고정 anchor(X_val 앞 256개, 매 epoch 동일 샘플)에 대해 첫 "
                            "로깅 epoch의 query_emb/agg_emb/centroid_id를 스냅샷으로 "
                            "저장해두고, 이후 epoch마다 같은 배치를 다시 흘려서 그 대비 "
                            "query_drift_from_epoch0(‖q_t-q_0‖, 얼마나 움직였는가), "
                            "cos_drift_vs_agg0(cos(q_t-q_0, a_0) — 핵심 지표, 움직인 "
                            "'방향'이 초기 retrieval 방향과 같은가. a_t를 '정답'처럼 쓰면 "
                            "a_t 자체도 매 epoch retrieval이 바뀌며 계속 움직여서 애매해지므로 "
                            "a_0를 고정 기준점으로 쓴 cosine으로 설계), "
                            "cos_query_t_vs_agg0(cos(q_t, a_0) — 최종 query가 초기 retrieval을 "
                            "닮아가는가), cos_query_agg_raw(cos(q_t, a_t), pre-LN/raw), "
                            "cos_query_agg_post_ln(cos(LN(q_t), LN(a_t)) — 같은 epoch·배치), "
                            "cos_query_agg_post_linear(cos(h_q, h_a), h_q=W@LN(q_t)+b, "
                            "h_a=β·W@LN(a_t) — head 첫 Linear를 지난 뒤. 세 값을 나란히 보면 "
                            "raw에서는 안 보이던 방향성이 LN에서 생기는지, 아니면 그 뒤 Linear "
                            "weight까지 가야 벌어지는지 구분됨 — 이전엔 이 비교를 서로 다른 두 "
                            "실행(--log_fusion_trajectory 따로)을 --deterministic 재현성에 기대어 "
                            "겹쳐봤는데, 이제 한 실행 안에서 동시에 나옴. fusion_mode≠residual이면 "
                            "post_ln/post_linear는 None), "
                            "[2026-07, 추가] query_norm_raw/agg_norm_raw/query_norm_post_ln/"
                            "agg_norm_post_ln/hq_norm_post_linear/ha_norm_post_linear — cosine은 "
                            "방향만 보고 크기(head가 실제로 받는 신호 magnitude)는 안 보므로, "
                            "각 단계의 ‖·‖도 같이 기록(특히 hq_norm/ha_norm_post_linear는 head가 "
                            "각 branch에서 실제로 받는 projection 크기 — 이게 학습 초반에 이미 "
                            "안정화되는지, cosine 안정화 시점과 같은지 다른지 비교할 것). "
                            "centroid_stability_vs_epoch0(같은 centroid로 배정된 비율)를 기록"
                            "(meta.pkl의 representation_drift_history). "
                            "embedder+prototype+retrieve+aggregate까지 다 거치는 forward라 "
                            "X_val 전체가 아니라 고정 256개 subset만 씀 — "
                            "--regroup_log_every 간격으로만 계산. tasktype=regression이면 "
                            "빈 리스트로 남음."
                        ))
    parser.add_argument("--fusion_alpha_override", type=float, default=None,
                        help=(
                            "[구조 변경] fusion_mode=residual에서 α를 학습 가능한 "
                            "파라미터 대신 이 값으로 고정(register_buffer, "
                            "requires_grad=False). '학습이 α≈1을 선택했다'와 "
                            "'α=1로 고정해도 비슷한 성능이 나온다'는 다른 주장 — "
                            "{0, 0.5, 1, 2} 등으로 스윕해서 causal하게 확인하기 위함. "
                            "fusion_mode!=residual이거나 --no_context_emb와 같이 쓰면 "
                            "TabERA 생성자가 ValueError."
                        ))
    parser.add_argument("--fusion_beta_override", type=float, default=None,
                        help="fusion_alpha_override와 대칭, agg 쪽(β) 고정값.")
    parser.add_argument("--disable_retrieval_branch", action="store_true",
                        help=(
                            "[2026-07, 추가] '진짜' retrieval-free baseline(Model 2) — "
                            "사용자 지적: --fusion_beta_override 0.0 + --loss_*_override 0.0 "
                            "조합은 head 쪽 fusion만 끊을 뿐, prototype routing이 STE "
                            "(forward=argmax, backward=softmax 통과)라 encoder 쪽으로 "
                            "gradient가 여전히 샐 수 있음 — 그래서 '진짜' 아님. 이 플래그는 "
                            "embedder 직후 곧바로 분기해서 prototype_layer/memory.retrieve/"
                            "aggregator를 아예 호출하지 않음(STE를 포함해 그 무엇도 안 거침). "
                            "z=query_emb를 그대로 head에 넣음 — fusion_mode='residual'+"
                            "aggregator_mode='pooling' 조합만 지원(그 외 조합이면 모델 생성 "
                            "시점에 NotImplementedError). agg_emb/context_emb/centroid_id/"
                            "fusion_beta 등 retrieval 관련 out dict 키는 전부 None — "
                            "ablation_mode나 branch 분석류 진단은 이 모드에서 의미 없음(agg "
                            "자체가 없음). aux_loss=0(commitment/codebook/diversity 전부 "
                            "hard_assignment가 없어서 계산 불가). Model 1(full)/Model 3"
                            "(query_shuffle ablation, 이미 있는 결과)과 나란히 놓고 accuracy "
                            "비교하기 위한 순수 baseline 스위치. HPO(optimize.py)는 그대로 "
                            "retrieval 있는 구조 기준 study를 재사용(k/n_prototypes 등은 이 "
                            "모드에서 안 쓰이므로 무의미하지만 해는 없음) — study_pkl_tag에 "
                            "'..no_retrieval' 태그가 붙어 기존 study와 안 섞임."
                        ))
    parser.add_argument("--evidence_metric_override", type=str, default=None,
                        choices=["euclidean", "cosine", "cosine_scaled"],
                        help=(
                            "[통제 실험용] AttentionAggregator의 evidence_w 유사도 공간. "
                            "기본(None)이면 model_kwargs의 evidence_metric(보통 'euclidean', "
                            "기존과 동일)을 그대로 씀. 'euclidean'은 -‖q-k‖²(raw, 정규화 "
                            "안 됨) — jasmine 실측: query_emb norm이 학습 중 커지면서(최대 "
                            "89배) evidence_w가 사실상 1-NN으로 붕괴, evidence_temperature "
                            "스윕(0.5~10)으로도 해결 안 됨(고정 스칼라로는 계속 커지는 norm을 "
                            "못 따라잡음). 'cosine'은 q,k를 CentroidLayer 라우팅과 동일하게 "
                            "정규화 후 2·cos(q,k) — norm 자체가 계산에서 빠져 이 collapse "
                            "메커니즘을 원천 제거. 'cosine_scaled'는 여기에 hyperspherical "
                            "sharpness scale(√2·log(k-1), routing_scale과 같은 원리를 evidence "
                            "후보 개수 k에 독립 적용 — routing_scale 값 자체를 재사용하는 "
                            "건 아님)을 곱함. --dropout_override와 같은 패턴 — model_kwargs에 "
                            "반영, --from_saved_state와 같이 쓰면 재학습을 안 하므로 무효과."
                        ))
    parser.add_argument("--evidence_temperature_override", type=float, default=None,
                        help=(
                            "[통제 실험용] AttentionAggregator의 evidence_w = "
                            "softmax(-‖q-k‖² / T)에서 T(evidence_temperature)를 이 값으로 "
                            "설정하고 재학습(기본 1.0 = 기존과 동일, 하위 호환). jasmine/"
                            "credit-g 실측: evidence entropy가 학습 초반부터 이미 ln(k) "
                            "대비 크게 낮고(사실상 1-NN 붕괴) 학습 중 더 낮아짐 — 이게 "
                            "raw(정규화 안 됨) 유클리드 거리 softmax의 calibration 문제인지 "
                            "검증하기 위한 수동 스윕용. 추천 스윕값: 0.5/1/2/5/10. T>1이면 "
                            "더 완만하게(여러 이웃), T<1이면 더 뾰족하게(소수 이웃에 집중). "
                            "--dropout_override와 같은 패턴 — model_kwargs에 반영, "
                            "--from_saved_state와 같이 쓰면 재학습을 안 하므로 무효과."
                        ))
    parser.add_argument("--embed_dim_override", type=int, default=None,
                        help=(
                            "[통제 실험용] best_params의 embed_dim 값을 이 값으로 덮어쓰고 "
                            "나머지는 그대로 재학습. jasmine/mfeat-zernike/ada_agnostic 3개 "
                            "데이터셋의 cosine 전용 HPO에서 embed_dim이 커지는 방향(→256)이면 "
                            "agg/query gradient가 줄고, 작아지는 방향(→64)이면 느는 패턴이 "
                            "일관되게 관찰됨(jasmine 64→256/agg↓, mfeat-zernike 256→64/agg↑, "
                            "ada_agnostic 128→256/agg↓) — loss_commitment/diversity는 세 "
                            "데이터셋 다 같은 방향으로 움직여 이 갈림을 설명 못 했지만, "
                            "embed_dim은 방향이 갈려서 정확히 일치함. 다만 HPO가 embed_dim과 "
                            "동시에 dropout/lr/layers/loss weight도 같이 바꿨으므로 상관관계일 "
                            "뿐 인과는 아직 미검증 — 이 플래그로 embed_dim 하나만 격리해서 "
                            "확인. --loss_codebook_override와 같은 패턴. [주의] embed_dim은 "
                            "모델 구조(가중치 shape) 자체를 바꾸므로 --from_saved_state로 "
                            "저장된 다른 embed_dim 체크포인트를 불러올 수 없음(애초에 "
                            "--from_saved_state는 재학습을 안 해서 이 플래그와 같이 못 씀)."
                        ))
    parser.add_argument("--dropout_override", type=float, default=None,
                        help=(
                            "[통제 실험용] best_params의 dropout 값을 이 값으로 덮어쓰고 "
                            "나머지는 그대로 재학습. dropout이 TabularEmbedder(ResidualMLP) "
                            "내부에 있어 query_emb 자체를 매 forward마다 흔드는데, 이게 "
                            "라우팅 churn(연속적인 centroid dead/reinit)의 원인 중 하나인지 "
                            "확인하려는 용도. --loss_codebook_override와 같은 패턴 — "
                            "--from_saved_state와 같이 쓰면 재학습을 안 하므로 무효과."
                        ))
    parser.add_argument("--batch_size_override", type=int, default=None,
                        help=(
                            "[통제 실험용] best_params의 batch_size 값을 이 값으로 덮어쓰고 "
                            "나머지는 그대로 재학습. batch_size를 HPO 탐색 대상에서 빼고 "
                            "데이터셋 크기에 따른 고정값으로 대체할 근거(TabR 계보의 표준 "
                            "관행)를 마련하기 위한 실측용 — 같은 best_params에 batch_size만 "
                            "바꿔가며 여러 값(예: 64/128/256/512)을 스윕해서 val 성능이 얼마나 "
                            "민감한지, 데이터셋 크기와 어떤 관계가 있는지 확인한다. "
                            "--dropout_override/--loss_codebook_override와 같은 패턴 — "
                            "--from_saved_state와 같이 쓰면 재학습을 안 하므로 무효과. "
                            "model_kwargs가 아니라 best_params(=TabERAWrapper.params)에만 "
                            "반영됨 — batch_size는 학습 루프에서만 쓰이고 모델 구조와는 무관."
                        ))
    parser.add_argument("--regroup_warmup_epochs_override", type=int, default=None,
                        help=(
                            "[통제 실험용] CentroidLayer.regroup_warmup_epochs를 이 값으로 "
                            "설정하고 재학습(기본은 0=즉시 활성화 — 지금까지 실제로 쓰인 값). "
                            "학습 초반 STE+dead-centroid reinit이 불안정한 시기에 regroup을 "
                            "미루면 학습 전체의 라우팅 안정성(active_ratio_std, "
                            "reinit_per_epoch)과 최종 성능이 어떻게 바뀌는지 확인하는 용도. "
                            "--dropout_override와 같은 패턴 — model_kwargs에 반영(모델 구조 "
                            "파라미터이므로 --from_saved_state와는 같이 못 씀, 이미 만들어진 "
                            "모델의 CentroidLayer 설정은 재학습 없이 못 바꿈)."
                        ))
    parser.add_argument("--dead_reinit_patience_override", type=int, default=None,
                        help=(
                            "[통제 실험용] CentroidLayer.dead_reinit_patience를 이 값으로 "
                            "설정하고 재학습(기본 5 — 검증 안 된 값, Jukebox/NSVQ 등 원 논문은 "
                            "'연속 N epoch'이 아니라 '사용률이 threshold 아래로 떨어지면'이라는 "
                            "다른 기준을 씀). 값을 늘리면 죽은 centroid가 재초기화(=gradient "
                            "없이 파라미터를 무작위로 덮어쓰는 이벤트)되기까지 더 오래 방치되는 "
                            "대신, 그 무작위 개입 자체의 빈도가 줄어듦 — reinit 빈도와 학습 "
                            "안정성(active_ratio_std) 사이의 트레이드오프를 재기 위한 용도. "
                            "model_kwargs에 반영 — --from_saved_state와는 같이 못 씀."
                        ))
    parser.add_argument("--dead_reinit_noise_scale_override", type=float, default=None,
                        help=(
                            "[통제 실험용] CentroidLayer.dead_reinit_noise_scale을 이 값으로 "
                            "설정하고 재학습(기본 0.01 — 검증 안 된 값. 재초기화 시 anchor "
                            "벡터에 더하는 가우시안 노이즈의 표준편차 = 이 값 × anchor.norm()). "
                            "원 논문은 'small Gaussian noise'라고만 하고 구체적 크기를 안 줌 — "
                            "이 값이 재초기화 직후 그 centroid가 원래 anchor와 얼마나 다른 "
                            "위치에 놓이는지를 결정. 0으로 주면 노이즈 없이 anchor를 그대로 "
                            "복제. model_kwargs에 반영 — --from_saved_state와는 같이 못 씀."
                        ))
    parser.add_argument("--refresh_on_best", action="store_true",
                        help=(
                            "[설명가능성/재현성] best_state(및 feature_store) 복원 직후, "
                            "memory.keys를 raw feature(feature_store._store)로부터 현재 "
                            "(frozen) 가중치로 다시 인코딩해 덮어쓴다. 학습 중 저장된 값은 "
                            "특정 시점의 dropout mask로 계산된 1회성 스냅샷이라 raw feature의 "
                            "결정론적 함수가 아니었는데, 이 플래그를 켜면 memory.keys[i] == "
                            "embedder(feature_store._store[i])가 (부동소수점 오차 수준까지) "
                            "성립하게 됨 — --ablation dual_space_faithfulness의 사전검증 1.5가 "
                            "percentile 비교 대신 정확한 근접도(≈1.0)로 판정 가능해짐. 기본값 "
                            "False — 켜지 않으면 기존 동작과 100%% 동일(HPO best_params도 안전). "
                            "--from_saved_state와 같이 쓰면, 저장된 checkpoint가 이미 refresh된 "
                            "상태가 아닐 경우에만 여기서 다시 refresh를 수행한다(방법2 fallback — "
                            "저장 당시 --refresh_on_best를 켰다면 이미 clean해서 사실상 no-op)."
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

    # [2026-07, v2 freeze] use_context_emb=False가 이제 기본값 — 기존 7군데
    # 흩어진 "not args.no_context_emb" 사용처를 하나하나 안 고치고, 여기서
    # args.no_context_emb 자체를 보정해서 그 아래 로직은 전부 그대로 두는
    # 방식(각 호출부를 개별 수정하는 것보다 훨씬 안전 — 하나 놓쳐서 옛
    # 기본값이 남는 사고 방지). --use_context_emb를 명시적으로 주지 않으면
    # (기본, 대부분의 실행) no_context_emb=True로 취급 — v2 기본 architecture.
    # --use_context_emb를 주면 V1식으로 되돌아감(no_context_emb=False가
    # 되어야 하므로, 이미 --no_context_emb를 실수로 같이 준 경우가 아니면
    # 아래에서 False로 재설정).
    if args.use_context_emb:
        args.no_context_emb = False
    else:
        args.no_context_emb = True

    if args.query_detach_warmup_epochs > 0 and args.query_detach_warmup_steps > 0:
        parser.error(
            "--query_detach_warmup_epochs와 --query_detach_warmup_steps는 "
            "동시에 0이 아닐 수 없습니다 — 하나만 지정하세요."
        )

    if args.no_query_emb and not args.use_context_emb:
        # [2026-07, 수정] use_context_emb=False가 이제 기본값이라, 예전처럼
        # "--no_query_emb와 --no_context_emb를 둘 다 명시적으로 켰을 때만"
        # 경고하면 --no_query_emb 단독 사용(매우 흔해질 조합)마다 항상
        # 스팸성으로 뜸 — 조건을 "--use_context_emb로 V1식을 명시적으로
        # 요청하지 않은 채 --no_query_emb를 켰는가"로 바꿈(사실상 같은
        # 극단 케이스를 가리키지만 새 기본값 기준으로 재정의).
        print(f"  ℹ️  --no_query_emb를 켰고 context_emb는 이미 기본값으로 head에서 제외돼 있습니다 — "
              f"head 입력이 agg_emb 하나만 남는 극단 케이스입니다. 의도한 게 맞는지 "
              f"확인해주세요(예: agg_emb 단독 representation 능력 실측 목적이면 정상).")

    # [v2, 수정] aggregator_mode="cross_attention"은 이제 --no_query_emb와
    # 완전히 무관함 — head_v2가 항상 [updated_query‖context_emb] 2-branch로
    # 고정 생성됨(updated_query에 query_emb가 이미 residual로 흡수돼 있음,
    # "agg_emb를 대체"가 아니라 "retrieval branch가 흡수된 것"). 이전에는
    # --no_query_emb를 같이 줘야 2-branch가 됐지만(agg_emb 슬롯 재사용
    # 방식), 이제 tabera.py가 cross_attention에서 그 플래그를 아예 안 봄.
    if args.aggregator_mode == "cross_attention" and args.no_query_emb:
        print(f"  ℹ️  --aggregator_mode cross_attention에서는 --no_query_emb가 필요 없습니다 "
              f"(항상 자동으로 2-branch) — 준 값은 tabera.py에서 무시됩니다.")





    # [추가] --deterministic: GPU 비결정성 vs 아키텍처 chaotic sensitivity 분리용.
    # cudnn.deterministic/benchmark은 언제나 안전하게 켤 수 있지만,
    # use_deterministic_algorithms(True)는 결정적 구현이 없는 연산을 만나면
    # RuntimeError를 던진다 — 이걸 --deterministic_warn_only 없이 그대로
    # 터뜨리는 게 의도적임: 어떤 연산이 원인인지 에러 메시지에 그대로 찍히므로,
    # 그게 곧 "이 모델의 어느 부분이 비결정적인가"에 대한 직접적인 실측 정보가 됨.
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=args.deterministic_warn_only)
        print(f"  [--deterministic] cudnn.deterministic=True, benchmark=False, "
              f"use_deterministic_algorithms(True, warn_only={args.deterministic_warn_only})"
              + (f" — CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', '(미설정!)')}"
                 if torch.cuda.is_available() else " (CUDA 없음, CPU라 애초에 대부분 결정적)"))
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

    # [진단용, 추가] 데이터 로딩 vs 학습 시간 분리 — optimize.py는 데이터셋을
    # 한 번만 로드해서 100개 trial이 재사용하는 반면(objective() 밖에서
    # 로드), reproduce.py는 매 실행마다(프로세스 단위) 새로 로드함. openml
    # fetch/NaN 전처리/StratifiedKFold/QuantileTransformer 비용이 매번
    # 여기 전부 실림 — "reproduce.py가 optimize.py trial보다 느리다"고
    # 느껴지는 게 학습 자체가 아니라 이 로딩 비용 때문인지 구분하기 위함.
    _t_data_start = time.time()
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    print(f"  [timing] dataset load: {time.time() - _t_data_start:.1f}s")
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

    # [v1.1, 추가] --train_seeds(복수) 지원 — optimize.py처럼 dataset/study를
    # main()에서 한 번만 로드하고, 그 안에서 seed마다 run_single_seed()만
    # 반복 호출(원래 main() 안에 인라인으로 있던 학습·평가·분석 로직 전체가
    # 이제 run_single_seed() 하나로 옮겨감 — 위 함수 정의 참고).
    if args.train_seeds:
        if args.from_saved_state:
            raise ValueError(
                "--train_seeds와 --from_saved_state는 같이 쓸 수 없습니다 — "
                "--from_saved_state는 특정 seed로 저장된 체크포인트 하나를 불러오는 "
                "것이라 여러 seed를 도는 것 자체가 의미가 없습니다. 단일 seed만 "
                "쓰려면 --train_seed(단수)를 쓰세요."
            )
        train_seed_list = args.train_seeds
    else:
        train_seed_list = [args.train_seed if args.train_seed is not None else args.seed]
    # run_single_seed() 안의 로그 문구(단일 실행인지 여러 seed 중 하나인지)
    # 판단용 — args에 임시로 붙여둠(CLI 옵션은 아님).
    args._train_seed_list = train_seed_list

    if args.explain_seed is not None:
        if args.explain_seed not in train_seed_list:
            raise ValueError(
                f"--explain_seed={args.explain_seed}가 --train_seeds({train_seed_list})에 없습니다."
            )
        explain_seed = args.explain_seed
    else:
        explain_seed = train_seed_list[-1]

    results = []
    for _ts in train_seed_list:
        do_analysis = (_ts == explain_seed)
        result = run_single_seed(
            dataset, X_train, y_train, X_val, y_val, X_test, y_test, y_std,
            output_dim, tasktype, openml_id, dataset_info, device, log_dir, env_info,
            args, _ts, do_analysis,
        )
        results.append(result)

    # [v1.1, 추가] seed 2개 이상이면 mean±std 요약 — reproduce.py의 목적을
    # "best config를 여러 초기화로 재확인(robust evaluation)"까지 포함하는
    # 것으로 넓힌 것에 맞춰, 개별 seed 숫자 나열로 끝내지 않고 최종 요약까지
    # 자동으로 낸다.
    if len(results) > 1:
        print(f"\n{'='*60}")
        print(f"  Summary across {len(results)} train_seeds: {train_seed_list}")
        print(f"{'='*60}")
        for split_name, key_dict_name in [("val", "val_metrics"), ("test", "test_metrics")]:
            metric_keys = sorted(results[0][key_dict_name].keys())
            for key in metric_keys:
                vals = np.array([r[key_dict_name][key] for r in results])
                indiv = ', '.join(f"{v:.4f}" for v in vals)
                print(f"  {key:16s} mean={vals.mean():.4f}  std={vals.std():.4f}  (seeds: {indiv})")



if __name__ == "__main__":
    main()