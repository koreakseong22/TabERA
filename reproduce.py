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
from libs import diagnostics as diag
from libs.search_space import params_to_model_kwargs, study_pkl_tag, HPO_TRAINING_SCHEDULE
from libs.supervised   import TabERAWrapper
# L_nbr 의 k/τ/margin 은 튜닝 대상이 아니라 모듈 상수다(libs/tabera.py 참고).
from libs.tabera       import (NBR_K, NBR_TAU, NBR_NEG_MARGIN,
                               strip_legacy_kwargs)
from libs.tabera         import TabERA
from libs.prototypes     import inverse_transform_numeric
from libs                import diagnostics as diag
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


# [제거됨] _select_query_similar_features / _select_query_dissimilar_features
# ─────────────────────────────────────────────────────────────
# 두 함수를 libs/diagnostics.py의 feature_gaps()로 합쳤다.
#
# 제거 이유는 중복이 아니라 **max_gap=0.15라는 임계값**이다. 그 값이
# "gap > 0.15인 feature는 아예 후보에서 뺀다"를 결정했고, 그래서 결과를
# 보여주기 **전에** 정보가 삭제됐다 — "왜 비슷한지"만 보이고 "어디가
# 다른지"는 숨는 확증편향 표시가 된 원인이다. 근거 없는 상수가 무엇을
# 볼지 정한다는 점에서 detector threshold와 같은 문제다.
#
# 지금은 feature_gaps()가 전체 feature의 gap을 그대로 돌려주고, 정렬과
# 상위 N개 절단은 아래 print_explanation(표시 계층)에서만 한다.

def _split_by_kind(labels, get_kind, get_str):
    """items를 kind별(numeric/categorical)로 나눠 두 개의 문자열 리스트로."""
    num_strs, cat_strs = [], []
    for item in labels:
        (num_strs if get_kind(item) == "numeric" else cat_strs).append(get_str(item))
    return num_strs, cat_strs


def print_explanation(explanations: list, sample_idx: int, col_names: list,
                       cat_category_names: dict = None,
                       quantile_transformer=None, num_cols: list = None,
                       pred_info: dict = None,
                       target_class_names: list = None,
                       tasktype: str = None,
                       max_neighbors: int = 3,
                       max_features: int = 4,
                       max_gaps: int = 3,
                       verbose: bool = False,
                       max_dims: int = 5) -> None:
    """
    ⚠ 표시 계층. 아래 max_* 인자는 **콘솔 줄 수 제한**이지 판정 기준이
      아니다 — 어떤 claim도 만들지 않는다. libs/diagnostics.py는 순위
      전체를 반환하고, 자르는 일은 여기서만 한다. 분석 스크립트는
      diagnostics를 직접 불러 전체 분포를 봐야 한다.

    ⚠ 이 함수는 판정하지 않는다. "ambiguous region" 같은 문장을 만들려면
      경험적 분포에서 기준선을 정해야 하는데, 샘플 하나를 찍는 이 자리에는
      참조 분포가 없다. 값만 출력한다.
    """
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

    # 이 of the group target(클래스) 분포 — ①의 주 콘텐츠 (label_groups_by_target(),
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
    # [제거] Routing confidence / margin / cosine similarity
    # 값은 샘플마다 갈리지만 **기준이 없어 읽어도 판단이 안 선다** —
    # P=28이면 균등이 3.6%인데 '16.5%'가 높은 건지 낮은 건지 알 수 없다.
    # 바로 아래 Routing distribution이 같은 정보를 비교 가능한 형태로 준다
    # (2·3위 of the group label distribution까지 붙어서 훨씬 잘 읽힘).
    # 값 자체는 explanations[b]['prototype']에 그대로 남아 있다 — 진단용.
    print(f"     Prototype label distribution: {target_str}")

    if proto["runners_up"]:
        print(f"     Routing distribution:")
        print(f"       • {proto['assigned_group']:<20s} {proto['routing_confidence']:>6.1%}  (assigned)")
        # ⚠ [렌더링 필터] 구성원이 없는 prototype 은 라우팅 확률이 높아도
        #   보여주지 않는다. `(no target info)` 는 설명이 부족한 게 아니라
        #   설명할 대상 자체가 없는 상태다(죽었거나 방금 재초기화됨).
        #   사용자에게는 정보가 아니라 혼란이다.
        #   ⚠ 모델 변경이 아니라 표시 계층의 validity filter 다 — 라우팅
        #     확률 자체는 그대로이고 Others 질량에도 영향을 주지 않는다.
        _shown = 0
        for r in proto["runners_up"]:
            if r.get("target_info") is None:
                continue
            _shown += 1
            print(f"       • {r['label']:<20s} {r['routing_confidence']:>6.1%}  "
                  f"({_format_target_info(r['target_info'])})")
        if _shown < len(proto["runners_up"]) and verbose:
            _hid = len(proto["runners_up"]) - _shown
            print(f"       {_hid} prototype(s) with no members hidden"
                  f" (they have routing mass but no group to describe)")
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
        print(f"     Characteristic features:")
        if num_strs:
            print(f"       numeric:     {',  '.join(num_strs)}")
        if cat_strs:
            print(f"       categorical: {',  '.join(cat_strs)}")

    # ② Local evidence — prototype-conditioned
    ev = e["evidence"]
    le = e.get("local_evidence")
    nbrs = e.get("neighbors")
    _single = False          # 이웃이 단일 클래스인가 (아래 ②에서 결정)
    _skip_contrast = False   # 위 축약 문장이 이미 '반례 없음'을 말했는가
    name_to_idx = {name: i for i, name in enumerate(col_names)} if col_names else {}

    # [Level 2 재정의] 세 가지를 한꺼번에 고친다.
    #
    # (a) 이웃 라벨 분포의 의미.
    #     "가까운 8개 중 6개가 yes" 를 그대로 내보내면 사람은 반드시
    #     "그러니까 yes" 로 읽는다. 그런데 §6-1이 측정으로 못박았다 —
    #     같은 prototype 안에서는 raw feature로도 다수결을 못 넘는다.
    #     그러면 그 6/8은 그룹 분포의 표본 노이즈이지 증거가 아니다.
    #     §4-4b가 지지하는 유일한 해석은 "이 지역이 얼마나 섞여 있는가"
    #     (purity 통제 후에도 entropy가 6/6에서 오분류를 유의하게 예측).
    #     → 항상 그룹 전체 분포와 나란히 찍고, 상대 모호성으로 요약한다.
    #
    # (b) scope 명시. 이 검색은 NN(q, G_p)이지 NN(q, D)가 아니다.
    #     "similar cases"라고만 쓰면 전체 데이터에서 찾은 것으로 읽힌다.
    #
    # (c) 반례(contrasting case). 이전 표시는 query와 **가까운** feature만
    #     골라 보여줬다(_select_query_similar_features는 gap>0.15를 아예
    #     후보에서 제외한다) — "왜 비슷한지"만 보이고 "어디가 다른지"는
    #     숨는 확증편향 표시였다. 예측과 결과가 다른 이웃에는 가장 크게
    #     **어긋난** feature를 같이 보여준다.
    _p = le.get("prototype") if le else None
    _scope = f"NN(q, G_{_p})" if _p is not None else "NN(q, G_p)"
    print(f"\n  ② Similar cases in the same group")
    # ⚠ 아래 5줄은 **내용이 맞지만 매 샘플 반복될 성질이 아니다.**
    #   14건을 읽으면 70줄이 되고, 두 번째 샘플부터는 아무도 안 읽는다.
    #   해석 규칙은 한 번만 읽으면 되는 것이므로 verbose에서만 낸다.
    #   (설명 전체의 규칙이지 이 샘플의 사실이 아니다.)
    if verbose:
        print(f"     Search is restricted to the assigned prototype group, not the whole training set.")
        print(f"     The neighbours' outcome distribution is not evidence for the prediction. Label")
        print(f"       separation inside a prototype is limited, so reading a neighbour majority as")
        print(f"       support would present sampling noise from the group distribution as a reason.")
        print(f"       What this section carries is one quantity: whether the decision was made in")
        print(f"       a typical part of its region or an ambiguous one.")

    # ── 지역 분포 vs 그룹 분포 ────────────────────────────────────
    def _fmt_label(v):
        if v is None:
            return "?"
        if tasktype == "regression":
            return f"{v:.4g}"
        code = int(round(v))
        if target_class_names and 0 <= code < len(target_class_names):
            return str(target_class_names[code])
        return str(code)

    if le is not None:
        if tasktype == "regression":
            if le.get("group_std") == le.get("group_std"):   # not nan
                print(f"\n     neighbourhood (k={le['n_neighbors']})  "
                      f"mean {le['local_mean']:.4g}  std {le['local_std']:.4g}")
                print(f"     whole group (n={le['group_size']})      "
                      f"mean {le['group_mean']:.4g}  std {le['group_std']:.4g}")
                _dr = le.get("dispersion_ratio")
                if _dr is not None and _dr == _dr:
                    print(f"     -> local / group entropy = {_dr:.2f}  "
                          f"(threshold to be set from the empirical distribution)")
        else:
            def _dist_str(counts, total):
                if not total:
                    return "(empty)"
                return ",  ".join(
                    f"{_fmt_label(float(k))} {c}/{total} ({c/total:.0%})"
                    for k, c in sorted(counts.items(), key=lambda x: -x[1]))
            _lc = {int(k): v for k, v in (le.get("label_counts") or {}).items()}
            _gc = {int(k): v for k, v in (le.get("group_label_counts") or {}).items()}
            # [축약] 이웃 k개가 전부 같은 라벨이면 (a) H(label)=0,
            # (b) 상대 모호성 0, (c) Contrasting 없음 — 셋이 **자동으로
            # 동시에** 성립한다. 같은 사실을 세 번 말하게 되므로
            # (credit-g 14건 중 7건이 이 경우) 한 문장으로 합친다.
            # 그룹 분포는 대비가 되므로 함께 남긴다.
            # ⚠ "단일 클래스 = 반례 없음"이 아니다. 이웃이 전부 **예측과
            #   반대** 클래스인 경우가 있고(합성 실측: 예측 yes인데 이웃
            #   6/6이 no), 그때는 오히려 전부가 반례다. 예측 클래스와
            #   일치하는지까지 봐야 축약할 수 있다.
            _one = (len(_lc) == 1 and le["n_neighbors"] > 0)
            _pcode = (pred_info or {}).get("pred_code")
            _same_as_pred = (_one and _pcode is not None
                              and next(iter(_lc)) == int(_pcode))
            _single = _one and _same_as_pred
            # 라벨 분포 entropy임을 표기에도 남긴다 — 화면에 'entropy'만
            # 찍히면 evidence_w entropy와 구분이 안 된다.
            if _single:
                _skip_contrast = True
                print(f"\n     the k={le['n_neighbors']} nearest are all "
                      f"{_fmt_label(float(next(iter(_lc))))} — single-class region, no contrasting case")
            elif _one:
                # 이웃이 전부 예측과 반대 — 축약하지 않고 오히려 강조한다.
                # 아래 Contrasting 목록에 전부 나온다.
                print(f"\n     ⚠ the k={le['n_neighbors']} nearest are all "
                      f"{_fmt_label(float(next(iter(_lc))))} while the prediction is "
                      f"{(pred_info or {}).get('pred_label', '?')} "
                      f"— every neighbour contradicts it")
            else:
                print(f"\n     neighbourhood (k={le['n_neighbors']})   "
                      f"{_dist_str(_lc, le['n_neighbors'])}   H(label) {le['label_entropy']:.3f}")
            print(f"     whole group (n={le['group_size']})       "
                  f"{_dist_str(_gc, le['group_size'])}   H(label) {le['group_label_entropy']:.3f}")
            _ar = le.get("ambiguity_ratio")
            if (not _single) and _ar is not None and _ar == _ar:
                # ⚠ 판정하지 않는다. 이전에는 1.15/0.85를 기준으로 "더 섞인
                #   지역" 같은 문구를 붙였는데, 그 두 값에는 아무 근거가
                #   없었다. 어디부터 높은 것인지는 eval set 전체의 경험적
                #   분포에서 정해야 하고, 그건 분석 스크립트의 일이다.
                print(f"     -> relative ambiguity {_ar:.2f}  "
                      f"(local entropy / group entropy; the threshold for "
                      f"calling a region ambiguous is not fixed here)")

    # ── 사례 목록: supporting / contrasting ───────────────────────
    def _fmt_cat_value(name: str, code_val: float) -> str:
        # ⚠ 이름 조회가 실패해도 **설명 전체가 죽으면 안 된다.** 이름이
        #   없는 것은 표시상의 손실이지만, 여기서 예외가 나면 그 샘플의
        #   설명이 통째로 사라진다. 매핑이 리스트든 dict든, 코드가 범위를
        #   벗어나든, 모두 "Category N"으로 안전하게 떨어뜨린다.
        names_for_col = cat_category_names.get(name) if cat_category_names else None
        code = int(code_val)
        try:
            if names_for_col is not None:
                nm = (names_for_col[code] if not isinstance(names_for_col, dict)
                      else names_for_col.get(code))
                if nm is not None:
                    return f"{name}={nm} [{code}]"
        except (IndexError, KeyError, TypeError):
            pass
        return f"{name}=Category {code}"

    def _fmt_num_value(name: str, uniform_val: float) -> str:
        if quantile_transformer is not None and num_cols is not None and name in name_to_idx:
            real_val = inverse_transform_numeric(quantile_transformer, num_cols,
                                                  name_to_idx[name], uniform_val)
            if real_val is not None:
                return f"{name}={real_val:.3g}"
        return f"{name}={uniform_val:.3f}"

    def _fmt_items(items):
        num_strs, cat_strs = _split_by_kind(
            items, get_kind=lambda it: it[2],
            get_str=lambda it: (_fmt_cat_value(it[0], it[1])
                                 if it[2] == "categorical" else _fmt_num_value(it[0], it[1])),
        )
        return num_strs, cat_strs

    if not nbrs:
        print(f"\n     (no neighbours — the memory bank held fewer than k, or this "
              f"is output from a version without the 'neighbors' key)")
    else:
        # 예측과 같은 결과 / 다른 결과로 나눈다. 반례가 반례로 보여야
        # case-based 설명이 성립한다.
        _pc = (pred_info or {}).get("pred_code")
        if tasktype == "regression" or _pc is None:
            groups = [("Cases (no outcome contrast)", nbrs)]
        else:
            sup = [n for n in nbrs
                   if n["label"] is not None and int(round(n["label"])) == int(_pc)]
            con = [n for n in nbrs
                   if n["label"] is not None and int(round(n["label"])) != int(_pc)]
            groups = [("Outcome-matched cases", sup),
                      ("Outcome-contrasting cases", con)]

        for title, rows in groups:
            if not rows:
                # 위에서 "단일 클래스 영역 (반례 없음)"을 이미 말했으면
                # 같은 사실을 두 번 찍지 않는다.
                if title.startswith("Contrasting") and not _skip_contrast:
                    print(f"\n     Outcome-contrasting cases — none "
                          f"(all {len(nbrs)} nearest share the prediction)")
                continue
            print(f"\n     {title}")
            _is_con = title.startswith("Contrasting")
            for nb in rows[:max_neighbors]:
                sid = nb.get("sample_id")
                sid_str = (f"train #{sid}" if sid is not None and sid >= 0
                           else f"mem #{nb['memory_idx']}")
                print(f"       #{nb['rank']+1}  similarity={nb['similarity']:.3f}"
                      f"   → {_fmt_label(nb.get('label'))}   [{sid_str}]")
                # [임계값 제거] 이전에는 gap<=0.15인 feature만 후보였다.
                # 지금은 전체 gap을 받아 **가장 가까운 것부터** max_features
                # 개를 자를 뿐이다 — 무엇을 보여줄지 상수가 정하지 않는다.
                gp = nb.get("gaps") or []
                if gp:
                    near = sorted(gp, key=lambda g: g["gap"])[:max_features]
                    ns, cs = _fmt_items([(g["name"], g["neighbor_value"], g["kind"])
                                          for g in near])
                    if ns:
                        print(f"            close on numeric:     {', '.join(ns)}")
                    if cs:
                        print(f"            close on categorical: {', '.join(cs)}")
                # 반례에는 "어디가 다른가"를 반드시 같이 보여준다.
                # ⚠ gap이 0이면 "다른 점"이 아니다. 전체를 gap 내림차순으로
                #   자르기만 하면, 모든 gap이 0인 경우(중복 행, 혹은
                #   self-retrieval 같은 이상 상황)에도 "duration 48 → 48"이
                #   찍힌다. 읽는 사람은 "뭐가 다르지?"가 된다.
                #   부동소수점 오차까지 고려해 eps로 거른다.
                _GAP_EPS = 1e-9
                df = ([(g["name"], g["neighbor_value"], g["kind"], g["delta"])
                       for g in sorted(gp, key=lambda g: g["gap"], reverse=True)
                       if g["gap"] > _GAP_EPS][:max_gaps]
                      if gp else [])
                if _is_con and df:
                    # numeric은 quantile 공간([0,1] uniform)의 값이라, 차이는
                    # 그대로 **백분위 차이**로 읽힌다(−0.498 = 약 50 백분위
                    # 낮음). 실제 단위로 역변환한 양끝값을 같이 보여줘서,
                    # ①/②의 다른 numeric 표시(역변환된 실제 단위)와 축이
                    # 어긋나 보이지 않게 한다.
                    for name, nval, kind, delta in df:
                        if kind == "categorical":
                            qv = nval - delta
                            print(f"            differs: {name}  "
                                  f"query {_fmt_cat_value(name, qv).split('=', 1)[1]}"
                                  f" -> neighbour {_fmt_cat_value(name, nval).split('=', 1)[1]}")
                        else:
                            qv = nval - delta
                            print(f"            differs: {name}  "
                                  f"query {_fmt_num_value(name, qv).split('=', 1)[1]}"
                                  f" -> neighbour {_fmt_num_value(name, nval).split('=', 1)[1]}"
                                  f"   ({delta*100:+.0f} pct)")
            if len(rows) > max_neighbors:
                print(f"       ... (+{len(rows) - max_neighbors} more)")

    # evidence_w(attention weight)는 aggregator가 실제로 학습되는 모드에서만
    # 의미가 있다. proto_dev 계열은 균등 상수라 entropy=log(k)로 고정 —
    # 그 값을 "이웃이 고르게 쓰였다"로 읽으면 안 되므로, 균등하지 않을
    # 때만 출력한다(§4-8 ②).
    _ew = ev.get("top_neighbours") or []
    if _ew and (max(w for _, w in _ew) - min(w for _, w in _ew)) > 1e-6:
        # ⚠ 이 entropy는 evidence_w(attention weight) 분포의 것이다.
        #   위 H(label)과 다른 값 — 이름을 화면에서 구분해 찍는다.
        print(f"     [aggregator attention] dominant={ev['dominant_weight']:.1%},  "
              f"H(evidence_w)={ev['entropy']:.3f}")

    # Level 3: Retrieval signal magnitude — [추가]
    # "기여도(contribution)"라고 안 부름 — head가 비선형 함수(예: residual
    # 모드의 Head(q+βa))라 ‖βa‖가 prediction에 미치는 실제 영향과 정확히
    # 비례한다는 보장이 없음(위 ②의 "기여도" 명명 정정과 같은 이유).
    # 여기서 주는 건 순수 magnitude 정보 — causal attribution 아님.
    # ③ Query-direction correction (Level 2.5)
    # §4-2: "argmax는 99.4% 같지만 확률값은 이 항이 결정한다" — 확신도를
    # 설명에 쓰면서 이 항을 안 보여주면 faithfulness 문제다.
    #
    # ⚠ [명칭 정정 §9] 예전엔 "Prototype-relative Deviation" 이라 불렀으나
    #   부정확하다. `‖c‖=1` 고정인데 `‖q‖` 가 7~1197 이라
    #       r = normalize(q − c) ≈ normalize(q) = q̂
    #   이고 실측 `cos(r, q)` 가 0.994~1.000 이다(v3·EMA 양쪽, 전 데이터셋).
    #   c 를 빼는 연산이 방향에 사실상 영향을 주지 않는다. 따라서 이 항은
    #   "prototype 으로부터의 편차" 가 아니라 **query 방향 보정** 이다.
    #   분해 항등식(logits = W·c + W·(β·r))은 그대로 정확하다.
    dv = e.get("prototype_deviation")
    if dv is not None:
        print(f"\n  ③ How this sample differs from its group")
        # ⚠ 이 분해는 근사가 아니다. dev_head가 단일 Linear이므로
        #   logits = (W·c + b) + W·(β·r) 이 항등식이고, 두 항의 합이 항상
        #   실제 logits와 일치한다(스모크에서 오차 0.000e+00로 확인).
        #   SHAP/IG처럼 baseline을 골라 근사하는 것과 성격이 다르다.
        if verbose:
            print(f"     (dev_head is a single Linear, so logits = (W·c + b) + W·(β·r) — an exact decomposition)")
            print(f"     r = normalize(q−c), but ‖q‖ >> ‖c‖ = 1, so it points along q"
                  f" (cos(r, q) ~ 1.00). It is not a deviation from the prototype.")
        # [교체] '편차 비중' + '결정: 그대로' → 확률 이동
        #
        # ⚠ dev_share는 로짓 크기 비율이라 **크기를 과장한다.** credit-g
        #   실측: dev_share 5.6~19.3%로 읽히는데 실제 확신도 이동은
        #   0.2~1.2%p였다. 로짓이 ±0.6 구간이라 sigmoid가 거의 선형이기
        #   때문이다. 사람이 읽는 단위는 확률이므로 확률로 보여준다.
        #
        # ⚠ '결정: 그대로'는 지우지 않는다. credit-g에서는 800/800이
        #   '그대로'라 죽은 줄이지만, P < C인 데이터셋(ds=1493: 35개
        #   prototype으로 100개 클래스)에서는 70.5%가 '바뀜'이고 그때
        #   이 줄이 설명의 핵심이 된다. 조건에 따라 살아나는 줄이다.
        if dv.get("prob_final") is not None:
            _pp, _pf = dv["prob_proto"], dv["prob_final"]
            _lab = (pred_info or {}).get("pred_label", "prediction")
            # ⚠ 이 값은 W·c 에만 의존하므로 **같은 prototype에 배정된 모든
            #   샘플에서 동일하다**(credit-g 실측: Centroid_16의 7개 샘플이
            #   전부 65.1%). "prototype만"이라고 쓰면 샘플별 값처럼 읽히므로
            #   그룹 수준 기준선임을 이름에 드러낸다. 샘플마다 다른 것은
            #   이동폭뿐이다.
            print(f"     prototype-only prediction: {_lab} {_pp:.1%}")
            print(f"     this sample:              {_lab} {_pf:.1%}"
                  f"   ({(_pf - _pp) * 100:+.1f}%p)")
            if dv["argmax_changed"]:
                _pc = dv.get("proto_pred")
                _pn = (target_class_names[_pc]
                       if (target_class_names and _pc is not None
                           and 0 <= _pc < len(target_class_names)) else _pc)
                print(f"     the correction changes the decision — prototype alone gives \"{_pn}\", "
                      f"the query-direction correction flips it to \"{_lab}\"")
        else:
            # 회귀: 확률이 없으므로 로짓 분해 그대로
            print(f"     prototype={dv['logit_proto']:+.4f}"
                  f"   query_dir={dv['logit_dev']:+.4f}"
                  f"   final={dv['logit_proto'] + dv['logit_dev']:+.4f}")

        # [제거] 편차 집중도 / dim_contrib
        # embedding 차원 번호는 사람이 아무것도 할 수 없는 정보다. 25건을
        # 읽는 동안 한 번도 쓸모를 못 느꼈다. 논문 본문 수치로는 가치가
        # 있으므로 diagnostics.prototype_deviation()의 dim_contrib에 전체가
        # 그대로 남아 있다 — 분석 스크립트에서 쓸 것.

    # ③-b feature 공간의 그룹 대비 — 읽을 수 있는 축
    gc = e.get("group_stats")
    if gc and (gc.get("numeric") or gc.get("categorical")):
        print(f"\n     against the group (feature space, n={gc['group_size']})")
        if verbose:
            print(f"     (the group typical value is the inverse transform of a mean taken in quantile space, not an arithmetic mean)")
        # ⚠ 위 ③(embedding 공간의 정확한 logit 분해)과 **다른 축**이다.
        #   이건 기술 통계지 attribution이 아니다 — "이 feature 때문에
        #   예측이 이렇게 나왔다"는 문장을 이 값으로 만들면 안 된다.
        if verbose:
            print(f"     (descriptive statistics against the same group — not attribution; do not read causally with the decomposition above)")
        # [절단 위치] diagnostics는 전체 feature를 |z| 내림차순으로 준다.
        for d in gc.get("numeric", [])[:max_features]:
            # 실단위 역변환. quantile_transformer가 없으면 [0,1] 백분위 그대로.
            _real = (lambda x: inverse_transform_numeric(
                        quantile_transformer, num_cols, d["feature_idx"], x)
                     ) if (quantile_transformer is not None and num_cols is not None) \
                    else (lambda x: None)
            _vr, _mr = _real(d["value"]), _real(d["group_mean"])

            def _fmt(x, fallback):
                # ⚠ 6.19e+03 은 사람이 못 읽는다. 천단위 구분 기호를 쓴다.
                # ⚠ 정수처럼 보이는 값(existing_credits=1, 그룹 평균=1)에
                #   반올림을 걸면 "같은데 왜 z가 −0.74지?"가 된다.
                #   실제로는 1.0 vs 1.4이므로 소수점 한 자리를 남긴다.
                if x is None:
                    return f"{fallback:.3f}"
                ax = abs(x)
                if ax >= 1000:
                    return f"{x:,.0f}"
                if ax >= 10:
                    # ⚠ 허용오차가 1e-9면 역변환 부동소수점 오차(≈1e-5)에
                    #   걸려 392가 "392.0"으로 찍힌다. 값 크기에 비례한
                    #   상대 오차로 판정한다.
                    return (f"{x:,.1f}" if abs(x - round(x)) > 0.01 * max(ax, 1.0)
                            else f"{x:,.0f}")
                return f"{x:.2f}".rstrip("0").rstrip(".")

            _v_s  = _fmt(_vr, d["value"])
            _mu_s = _fmt(_mr, d["group_mean"])
            # ⚠ 정수형 feature(existing_credits, installment_commitment 등)는
            #   둘 다 정수로 반올림되어 "1 (group typical 1, z=-0.79)"처럼
            #   **같아 보이는데 z만 붙는** 상태가 된다(14건 numeric 줄의 약 27%).
            #   읽는 사람은 "같은데 왜 z가 있지?"가 된다. 실제로는 1 vs 1.4다.
            #   "대표값과 같음"으로 쓰면 1.4라는 정보가 사라지므로, 대신
            #   **구분될 때까지 대표값의 정밀도를 올린다.**
            if (_v_s == _mu_s and _vr is not None and _mr is not None
                    and abs(_vr - _mr) > 1e-6):
                for _p in (1, 2, 3):
                    _cand = f"{_mr:,.{_p}f}"
                    if _cand != _v_s:
                        _mu_s = _cand
                        break
            # 배수는 **양수 연속량에서만** 의미가 있다(금액·기간 등).
            # 음수나 0을 지나는 값에서 배수는 해석 불가이므로 생략한다.
            _ratio = ""
            if _vr is not None and _mr is not None and _mr > 0 and _vr > 0:
                _r = _vr / _mr
                if _r >= 1.15 or _r <= 0.87:      # 표시 절단(판정 아님)
                    _ratio = f"{_r:.1f}x, " if _r >= 1 else f"{1/_r:.1f}x lower, "
            # ⚠ "평균"이라고 쓰면 안 된다. 이 값은 quantile 공간 평균을
            #   역변환한 것(inverse_transform(mean(q)))이라 실공간의 산술
            #   평균이 아니다. credit_amount처럼 왜곡된 분포에서는 산술
            #   평균보다 낮게, 중앙값에 가깝게 나온다. quantile 변환이
            #   단조라 **부호는 항상 z와 일치**하므로 비교 자체는 유효하다.
            #   (z는 quantile 공간에서 계산됨 — 축이 다르다는 점도 유의)
            # ⚠ 화면에는 **백분위**를 쓴다. z는 quantile 공간 값이고 여기
            #   표시되는 값/대표값은 실단위 역변환이라 축이 다르다 — 이산형
            #   feature에서 "1 (대표값 1, z=-0.79)"처럼 같아 보이는데 z만
            #   붙는 문제가 생긴다. 백분위는 단조 변환에 불변이라 두 축이
            #   일치한다. z는 diagnostics 반환값에 그대로 남아 있다(분석용).
            _pct = d.get("group_pct")
            if _pct is None:
                _pos = f"z={d['z']:+.2f}"
            elif _pct >= 0.5:
                _pos = f"top {(1 - _pct) * 100:.0f}%"
            else:
                _pos = f"bottom {_pct * 100:.0f}%"
            print(f"       {d['feature_name']}={_v_s}"
                  f"   (group typical {_mu_s},  {_ratio}{_pos})")
        for d in gc.get("categorical", [])[:max_features]:
            # ⚠ 거르지 않는다. 이전에는 "이 값이 곧 최빈값이면 건너뛴다"로
            #   숨겼는데, 무엇을 숨길지 표시부가 정하면 읽는 사람은 그
            #   feature를 확인한 적이 없다는 사실조차 모른다. 이 샘플의
            #   비율과 그룹 최빈을 **항상 같이** 찍어 판단을 넘긴다.
            # 모델은 dataset의 cat_category_names를 모르므로 "Category N"으로
            # 돌려준다 — 실제 이름 매핑은 여기(출력부)에서 한다.
            _v  = _fmt_cat_value(d["feature_name"], d["value"]).split("=", 1)[1]
            _mv = _fmt_cat_value(d["feature_name"], d["group_mode"]).split("=", 1)[1]
            print(f"       {d['feature_name']}={_v} (of the group {d['group_freq']:.0%})"
                  f"   |   group mode {_mv} ({d['group_mode_freq']:.0%})")

    # [제거] Representation Magnitude 블록
    # β는 모델 상수라 25/25 샘플에서 글자 그대로 같았고(β=0.1039),
    # ‖query_emb‖는 절대값이라 해석 기준이 없다. 정보량 0인 블록이었다.
    # 값은 explanations[b]["retrieval_signal"]에 그대로 있다 — 진단용.

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
    # ⚠ hasattr만으로는 부족하다 — proto_residual은 속성이 있되 None이다
    #   (통합 head가 없어 "첫 Linear" 개념 자체가 없음).
    if (getattr(model, "_head_first_linear", None) is None
            or not getattr(model, "_head_block_slices", None)):
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
    _cached_head_inputs = []
    try:
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                model(X[start:start + batch_size])
                x = captured["x"]  # (B, in) — head 첫 Linear가 실제로 받은 입력
                # [Step 2 진단] Jacobian 계산에서 재사용 — model()을 다시 부르지
                # 않기 위해서다. forward를 반복하면 memory/feature_store 경로가
                # 다시 타면서 위험하고, 비용도 두 배가 된다.
                _cached_head_inputs.append(x.cpu())
                for name, (s, e) in slices.items():
                    contrib = x[:, s:e] @ W[:, s:e].T   # (B, out) — 이 branch만의 선형 기여
                    per_branch_norms[name].append(contrib.norm(dim=-1).cpu())
    finally:
        handle.remove()
    model._last_head_inputs = _cached_head_inputs

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


def compute_branch_jacobian(model, X=None, batch_size: int = 256):
    """head **전체**를 통과한 뒤의 ∂logit/∂branch — "head가 이 branch를
    실제로 얼마나 사용하는가"를 비선형까지 포함해서 잰다.

    [왜 필요한가] compute_branch_linear_contribution은 head **첫 Linear**의
    ‖W_i x_i‖만 본다. 그건 "입력이 첫 층에 얼마나 크게 들어가는가"이지
    "최종 출력이 그 branch에 얼마나 민감한가"가 아니다. 첫 층에서 크게
    들어가도 이후 층에서 죽을 수 있고, 반대도 가능하다.
    지금까지의 분석(CKA/CCA/probe/within-variance)은 전부 **representation**
    분석이었고, head가 그 표현을 어떻게 쓰는지는 거의 측정하지 않았다.

    [해석]
      ‖∂logit/∂agg‖ share ≈ 0    head가 agg를 아예 무시 —
                                  value/T/candidate set 무엇을 바꿔도 안 씀
      share 충분히 큼             head는 쓰는데 내용이 나쁨 —
                                  그때 비로소 value/candidate set을 볼 이유가 생김

    ⚠ **모델 forward를 다시 부르지 않는다.** compute_branch_linear_contribution이
      no_grad로 이미 돌면서 캐시해둔 head 입력(model._last_head_inputs)을 쓴다.
      forward를 반복하면 memory/feature_store 경로를 다시 타고, 비용도 두 배다.
      따라서 이 함수는 compute_branch_linear_contribution **뒤에** 호출해야 한다.

    ⚠ `_head_block_slices`는 {name: (start, end)} **튜플**이다. `x[:, sl]`처럼
      쓰면 (start, end)가 인덱스 배열로 해석되어 end == in_features일 때
      CUDA index-out-of-bounds가 난다. 반드시 `x[:, s:e]`로 슬라이싱할 것.

    반환: {branch: {"jac_norm_mean", "share_of_total"}}
    """
    import torch as _t
    if not getattr(model, "_head_block_slices", None):
        raise ValueError("_head_block_slices가 없습니다 "
                         "(fusion_mode='concat'에서만 의미가 있습니다).")
    cached = getattr(model, "_last_head_inputs", None)
    if not cached:
        raise ValueError("model._last_head_inputs가 비어 있습니다 — "
                         "compute_branch_linear_contribution을 먼저 호출하세요.")
    slices = model._head_block_slices
    head = getattr(model, "head", None) or getattr(model, "head_v2", None)
    if head is None:
        raise ValueError("model.head / model.head_v2를 찾을 수 없습니다.")

    # _head_first_linear부터 끝까지가 미분 대상 (그 앞의 LayerNorm 등은 이미
    # 캐시된 입력에 반영돼 있다).
    sub = head
    try:
        mods = list(head)
        for _i, _m in enumerate(mods):
            if _m is model._head_first_linear:
                sub = _t.nn.Sequential(*mods[_i:])
                break
    except TypeError:
        pass

    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    sums = {n: 0.0 for n in slices}
    n_seen = 0
    try:
        for xb in cached:
            x = xb.to(dev).clone().requires_grad_(True)
            logits = sub(x)
            # 클래스 선택에 의존하지 않도록 로짓 노름을 스칼라로 축약
            scalar = (logits ** 2).sum(-1).clamp_min(1e-12).sqrt().sum()
            g = _t.autograd.grad(scalar, x)[0]
            for nm, (s_, e_) in slices.items():
                sums[nm] += float(g[:, s_:e_].norm(dim=-1).sum())
            n_seen += len(x)
    finally:
        if was:
            model.train()
    if n_seen == 0:
        return {}
    means = {nm: v / n_seen for nm, v in sums.items()}
    tot = sum(means.values()) or 1e-12
    return {nm: {"jac_norm_mean": v, "share_of_total": v / tot}
            for nm, v in means.items()}


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


def analyze_branch_information(model, X, tasktype: str, batch_size: int = 512, y=None,
                               n_shuffles: int = 5, residual_null: bool = False):
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


    # ── 행렬 형상 메타 (probe/redundancy 해석 가능 구간 판정용) ──────
    # [2026-07] p>n이면 선형 모델이 무엇이든 완벽히 맞출 수 있다. 그 구간에서
    # 나온 R²/gain은 "중복도"가 아니라 "보간 용량"을 재는 것이므로 해석하면
    # 안 된다. 값만 보고는 구분이 안 되므로 형상을 항상 같이 저장한다.
    def _mat_meta(m):
        m = np.asarray(m, dtype=np.float64)
        n_, p_ = m.shape
        mm = (m - m.mean(0)) / (m.std(0) + 1e-12)
        try:
            s = np.linalg.svd(mm, compute_uv=False)
            tol = s.max() * max(mm.shape) * np.finfo(float).eps
            rank = int((s > tol).sum())
            cond = float(s.max() / s[rank - 1]) if rank > 0 else float("nan")
            # effective rank: 특이값 분포의 엔트로피 지수 — rank가 형식적으로
            # 꽉 차 있어도 실제로 몇 방향이 살아있는지 본다.
            pr = s / (s.sum() + 1e-300)
            eff = float(np.exp(-(pr * np.log(pr + 1e-300)).sum()))
        except Exception:
            rank, cond, eff = -1, float("nan"), float("nan")
        return {"n": int(n_), "p": int(p_), "p_over_n": float(p_ / max(n_, 1)),
                "rank": rank, "effective_rank": eff, "cond": cond}

    # redundancy: agg_emb/context_emb를 query_emb로 회귀했을 때 R²
    #
    # [2026-07 계측기 보정] 이전에는 in-sample LinearRegression의 .score()를
    # 썼다. 1493(n=160, embed_dim=256)에서 agg_from_query_r2 / context_from_
    # query_r2가 **정확히 1.000**으로 나왔는데, 같은 형상에서 query와 완전히
    # 독립인 난수를 타깃으로 넣어도 1.000이 나온다 — p>n이면 OLS가 무엇이든
    # 정확히 맞추기 때문이다. 즉 그 값은 representation redundancy가 아니라
    # linear interpolation capacity를 재고 있었다.
    # → out-of-fold + Ridge(RidgeCV, fold 학습부에서만 alpha 선택)로 교체.
    #   각 샘플의 예측은 그 샘플을 뺀 fold로 학습한 모델에서 나오므로 p>n
    #   에서도 값이 정직하다(맞출 수 없으면 R²가 0 이하로 내려간다).
    # 옛 값은 *_insample로 함께 저장해 인공물 크기를 직접 대조할 수 있게 한다.
    # redundancy_method 키의 유무로 구/신 pkl을 구분할 수 있다.
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict as _cvp
    from sklearn.pipeline import make_pipeline as _mkpipe
    from sklearn.preprocessing import StandardScaler as _SS
    from sklearn.metrics import r2_score as _r2

    _RIDGE_ALPHAS = np.logspace(-2, 4, 13)
    _RED_FOLDS = int(min(5, len(embs["query"])))

    def _linreg_r2(target, source):   # 옛 방식(in-sample OLS) — 대조용으로만
        reg = LinearRegression().fit(source, target)
        return float(reg.score(source, target))

    def _oof_r2(target, source):
        if _RED_FOLDS < 2:
            return float("nan")
        cv = KFold(n_splits=_RED_FOLDS, shuffle=True, random_state=0)
        est = _mkpipe(_SS(), RidgeCV(alphas=_RIDGE_ALPHAS))
        pred = _cvp(est, np.asarray(source, dtype=np.float64),
                    np.asarray(target, dtype=np.float64), cv=cv)
        return float(_r2(target, pred, multioutput="uniform_average"))

    # (target, source) 쌍. agg↔context는 양방향 — evidence_w가 사실상 균등하고
    # 검색이 centroid 그룹 내로 제한되면 agg = 그룹요약 + f(q) 가 되어 context와
    # 같은 층위의 정보를 나를 수 있는데, query로부터의 R²만으로는 "둘 사이의"
    # 중복이 안 잡히기 때문이다. 비대칭이면 포함 관계를 시사한다.
    _RED_PAIRS = {
        "agg_from_query":     ("agg", "query"),
        "context_from_query": ("context", "query"),
        "context_from_agg":   ("context", "agg"),
        "agg_from_context":   ("agg", "context"),
    }
    redundancy = {
        "redundancy_method": "out_of_fold_ridge",
        "redundancy_n_folds": _RED_FOLDS,
        "redundancy_alphas": [float(a) for a in _RIDGE_ALPHAS],
        "shape_meta": {k: _mat_meta(v) for k, v in embs.items()},
    }
    for _nm, (_t, _s) in _RED_PAIRS.items():
        try:
            redundancy[f"{_nm}_r2"] = _oof_r2(embs[_t], embs[_s])
        except Exception as _re:
            redundancy[f"{_nm}_r2"] = float("nan")
            redundancy[f"{_nm}_error"] = f"{type(_re).__name__}: {_re}"
        try:
            redundancy[f"{_nm}_r2_insample"] = _linreg_r2(embs[_t], embs[_s])
        except Exception:
            redundancy[f"{_nm}_r2_insample"] = float("nan")

    # [2026-07, 추가/수정] Information gain — "agg가 라벨에 대해 query보다 새 정보를
    # 주는가"를 직접 잰다.
    # [왜 R²로는 부족한가] agg_from_query R²는 "agg를 query로 **선형 복원**할 수
    # 있는가"만 본다. R²가 낮아도 그 성분이 라벨과 무관하면 쓸모없고, 반대로
    # R²가 높아도 남은 소수 성분이 결정적일 수 있다. 여기서는 같은 선형 분류기를
    # query만 / query+agg / query+context 로 각각 학습해 성능 차이를 본다.
    #
    # [중요 — in-sample 평가는 쓰면 안 된다] 처음엔 probe를 같은 데이터로 학습·
    # 평가했는데, embed_dim이 128~256이고 샘플이 수백 개면 선형 분류기가 완벽히
    # 분리해서 AUROC이 정확히 1.0에 붙어버린다(실측: 10개 중 6개). 그러면 gain을
    # 잴 여지 자체가 없어져 "천장에 안 닿은 데이터셋에서만 gain이 보이는" 착시가
    # 생긴다. 그래서 **out-of-fold 예측**(StratifiedKFold cross_val_predict)으로
    # 바꿨다 — 각 샘플의 예측은 그 샘플을 뺀 fold로 학습한 모델에서 나온다.
    # 그래도 embedding 자체는 test set 전체로 만들어진 것이므로 절대 성능이 아니라
    # 세 입력 간 **상대 비교**용이다.
    information_gain = None
    if y is not None and tasktype != "regression":
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            # y가 torch tensor(CPU/CUDA 모두) 일 수 있음 — np.asarray는 CUDA
            # tensor에서 TypeError를 던진다. detach().cpu()를 거쳐 안전하게 변환.
            if hasattr(y, "detach"):
                _y = y.detach().cpu().numpy().ravel()
            else:
                _y = np.asarray(y).ravel()
            # 분류 라벨은 정수여야 함(float로 저장된 경우 반올림)
            if _y.dtype.kind == "f":
                _y = np.rint(_y).astype(int)
            _cls, _cnt = np.unique(_y, return_counts=True)
            # [2026-07] 희소 클래스 처리.
            # StratifiedKFold는 "가장 작은 클래스의 표본 수 >= n_splits"를 요구한다.
            # 1493(one-hundred-plants, 100클래스)의 test fold는 클래스당 1~2개뿐이라
            # _cnt.min()==1 이 되어 n_splits=2조차 못 잡고 통째로 측정 불가가 됐다
            # (그리고 그 경우 아래 print_branch_information이 KeyError로 죽었다).
            # 여기서 재는 것은 절대 성능이 아니라 query / query+agg / query+context
            # 세 입력 간 **상대 비교**이므로, 표본이 2개 미만인 클래스를 빼고 남은
            # 부분집합에서 세 probe를 모두 '동일하게' 평가하면 비교 자체는 유효하다.
            # 다만 무엇을 얼마나 뺐는지 결과 dict에 남긴다 — 뺀 비율이 크면
            # (n_samples_used / n_samples_total) 해석에 주의해야 한다.
            _keep_cls = _cls[_cnt >= 2]
            _n_dropped_cls = int(len(_cls) - len(_keep_cls))
            if _n_dropped_cls > 0:
                _mask = np.isin(_y, _keep_cls)
            else:
                _mask = np.ones(len(_y), dtype=bool)
            _y_used = _y[_mask]
            if len(_y_used) > 0:
                _cls_u, _cnt_u = np.unique(_y_used, return_counts=True)
            else:
                _cls_u, _cnt_u = np.array([]), np.array([])
            _nfold = int(min(5, _cnt_u.min())) if len(_cls_u) >= 2 else 0
            if len(_cls_u) >= 2 and _nfold >= 2:
                _cv = StratifiedKFold(n_splits=_nfold, shuffle=True, random_state=0)
                # fold 목록을 한 번만 만들어 모든 probe가 재사용한다(동일 분할 보장 +
                # 반복 계산 회피). StandardScaler는 파이프라인 안에 그대로 둔다 —
                # 밖으로 빼서 전체 데이터에 fit하면 fold 간 정보 누출이 된다.
                _folds = list(_cv.split(np.zeros(len(_y_used)), _y_used))
                _y_idx = np.searchsorted(_cls_u, _y_used)

                def _probe(mat):
                    clf = make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000))
                    p = cross_val_predict(clf, mat, _y_used, cv=_folds,
                                          method="predict_proba")
                    if p.shape[1] == 2:
                        auc = float(roc_auc_score(_y_used, p[:, 1]))
                    else:
                        auc = float(roc_auc_score(_y_used, p, multi_class="ovr",
                                                  average="macro"))
                    acc = float((p.argmax(axis=1) == _y_idx).mean())
                    return auc, acc

                # 마스킹된 feature를 한 번만 만들어 재사용
                _q = np.asarray(embs["query"])[_mask]
                _extra = {"agg": np.asarray(embs["agg"])[_mask],
                          "context": np.asarray(embs["context"])[_mask]}
                _dq = _q.shape[1]

                auc_q, acc_q = _probe(_q)

                # ── null 대조 ────────────────────────────────────────────
                # [왜 필요한가] gain = AUROC(q+x) - AUROC(q) 는 우리가 원하는
                # "x가 추가한 정보"와, 원하지 않는 "차원이 늘어난 데서 오는
                # out-of-fold 페널티"의 합이다. synthetic 검증에서 x가 순수
                # 노이즈일 때도 gain이 0이 아니라 음수로 나왔고(16차원/2클래스
                # -0.009, 32차원/50클래스 -0.067), 차원↑·클래스당 표본↓일수록
                # 커졌다. 즉 gain≈0을 기준선으로 쓰면 데이터셋마다 기준이 달라진다.
                #
                # plain null : x의 행 순서를 셔플 → 차원/스케일/fold 동일, 라벨
                #              대응 제거. 해석이 단순해 기본 결과용.
                # resid null : x를 query로 선형회귀한 뒤 **잔차만** 셔플 →
                #              query와의 선형 상관(공선성)까지 보존. plain은
                #              공선성도 같이 깨므로 페널티를 과대추정할 수
                #              있는데(L2 probe에서 기존 feature와 공선인 열은
                #              독립 노이즈 열보다 페널티가 작다), 그 편향은
                #              agg_from_query R²가 클수록 커진다. R-3 해석용
                #              보조 진단이며 기본 비활성(--branch_info_residual_null).
                #
                # delta = gain - null_mean 이 "차원 페널티를 뺀 순수 기여"에
                # 더 가깝다. 다만 어느 쪽도 완전한 통제는 아니므로 두 값을
                # 나란히 저장해 차이 자체를 진단으로 쓴다.
                _n_shuf = max(0, int(n_shuffles))

                # [2026-07] residualization을 OLS → Ridge로 교체.
                # p>n이면 OLS 잔차가 **정확히 0**이 되어 셔플해도 원본과 같아진다
                # (1493 실측: resid_null_std=0.0, delta_resid=0.0 — 이건 "정보가
                # 없다"가 아니라 "대조군이 만들어지지 않았다"였다). Ridge는 축소
                # 추정이라 잔차가 남는다. 그래도 저랭크 branch에서는 여전히 축퇴할
                # 수 있으므로 resid_norm_ratio를 재서 가드한다.
                _resid_cache = {}

                def _resid_parts(x):
                    if id(x) in _resid_cache:
                        return _resid_cache[id(x)]
                    est = _mkpipe(_SS(), RidgeCV(alphas=_RIDGE_ALPHAS))
                    est.fit(_q, x)
                    pred = est.predict(_q)
                    resid = x - pred
                    denom = np.linalg.norm(x - x.mean(0)) + 1e-300
                    ratio = float(np.linalg.norm(resid) / denom)
                    _resid_cache[id(x)] = (pred, resid, ratio)
                    return _resid_cache[id(x)]

                def _null_gain(x, kind, seed):
                    r = np.random.RandomState(seed)
                    perm = r.permutation(len(x))
                    if kind == "plain":
                        x_null = x[perm]
                    else:  # 'resid'
                        pred, resid, _ = _resid_parts(x)
                        x_null = pred + resid[perm]
                    buf = np.empty((len(x), _dq + x.shape[1]), dtype=np.float64)
                    buf[:, :_dq] = _q
                    buf[:, _dq:] = x_null
                    a, c = _probe(buf)
                    return a - auc_q, c - acc_q

                information_gain = {
                    "auroc_query": auc_q, "acc_query": acc_q,
                    "n_folds": _nfold, "eval": "out-of-fold (cross_val_predict)",
                    "n_shuffles": _n_shuf,
                    "residual_null": bool(residual_null),
                    "n_classes_total": int(len(_cls)),
                    "n_classes_used": int(len(_cls_u)),
                    "n_classes_dropped": _n_dropped_cls,
                    "n_samples_total": int(len(_y)),
                    "n_samples_used": int(_mask.sum()),
                    # [필수 확인] query만으로 이미 천장(AUROC≈1)에 닿으면 gain을
                    # 잴 여지가 없다 — 그 경우 이 측정은 무효로 취급해야 한다.
                    "query_auroc_ceiling": bool(auc_q >= 0.999),
                    # [필수 확인] probe 입력의 형상. p/n > 1이면 선형 probe가
                    # 과결정 구간에 있어 gain/null 모두 해석 불가로 봐야 한다.
                    "probe_meta": {
                        "query":         _mat_meta(_q),
                        "query_agg":     {"n": int(len(_q)), "p": int(_dq + _extra["agg"].shape[1]),
                                          "p_over_n": float((_dq + _extra["agg"].shape[1]) / max(len(_q), 1))},
                        "query_context": {"n": int(len(_q)), "p": int(_dq + _extra["context"].shape[1]),
                                          "p_over_n": float((_dq + _extra["context"].shape[1]) / max(len(_q), 1))},
                    },
                }
                information_gain["probe_overdetermined"] = bool(
                    information_gain["probe_meta"]["query_agg"]["p_over_n"] > 1.0)

                for _name, _x in _extra.items():
                    buf = np.empty((len(_x), _dq + _x.shape[1]), dtype=np.float64)
                    buf[:, :_dq] = _q
                    buf[:, _dq:] = _x
                    _auc, _acc = _probe(buf)
                    information_gain[f"auroc_query_{_name}"] = _auc
                    information_gain[f"acc_query_{_name}"]   = _acc
                    information_gain[f"auroc_gain_{_name}"]  = _auc - auc_q
                    information_gain[f"acc_gain_{_name}"]    = _acc - acc_q

                    # null은 base 측정이 dict에 들어간 **뒤에** 계산한다 —
                    # null 계산이 실패해도 base 결과는 남아서, "측정이 안 된
                    # 것"과 "대조가 깨진 것"을 pkl만 보고 구분할 수 있다.
                    _kinds = ["plain"] + (["resid"] if residual_null else [])
                    for _kind in _kinds:
                        _sfx = "shuffled" if _kind == "plain" else "resid_null"
                        _dsfx = "delta" if _kind == "plain" else "delta_resid"
                        try:
                            if _n_shuf <= 0:
                                continue
                            if _kind == "resid":
                                # 잔차가 사실상 0이면 x_null == x 라 대조군이
                                # 성립하지 않는다. 0을 저장하면 "정말 0"과
                                # 구분이 안 되므로 NaN + 플래그로 남긴다.
                                _, _, _ratio = _resid_parts(_x)
                                information_gain[f"resid_norm_ratio_{_name}"] = _ratio
                                if _ratio < 1e-6:
                                    information_gain[f"resid_null_degenerate_{_name}"] = True
                                    for _k in (f"auroc_gain_{_name}_delta_resid",
                                               f"acc_gain_{_name}_delta_resid",
                                               f"auroc_gain_{_name}_resid_null_mean",
                                               f"acc_gain_{_name}_resid_null_mean"):
                                        information_gain[_k] = float("nan")
                                    continue
                                information_gain[f"resid_null_degenerate_{_name}"] = False
                            _ga = [_null_gain(_x, _kind, s) for s in range(_n_shuf)]
                            _aa = np.array([g[0] for g in _ga], dtype=float)
                            _cc = np.array([g[1] for g in _ga], dtype=float)
                            information_gain[f"auroc_gain_{_name}_{_sfx}_mean"] = float(_aa.mean())
                            information_gain[f"auroc_gain_{_name}_{_sfx}_std"]  = float(_aa.std())
                            information_gain[f"acc_gain_{_name}_{_sfx}_mean"]   = float(_cc.mean())
                            information_gain[f"acc_gain_{_name}_{_sfx}_std"]    = float(_cc.std())
                            information_gain[f"auroc_gain_{_name}_{_dsfx}"] = float(
                                information_gain[f"auroc_gain_{_name}"] - _aa.mean())
                            information_gain[f"acc_gain_{_name}_{_dsfx}"] = float(
                                information_gain[f"acc_gain_{_name}"] - _cc.mean())
                        except Exception as _ne:
                            import traceback as _ntb
                            information_gain[f"null_error_{_name}_{_kind}"] = (
                                f"{type(_ne).__name__}: {_ne}")
                            information_gain[f"null_traceback_{_name}_{_kind}"] = (
                                _ntb.format_exc()[-500:])
            else:
                information_gain = {"error": (
                    f"클래스 수/최소 빈도 부족(classes_total={len(_cls)}, "
                    f"classes_with_ge2={len(_cls_u)}, "
                    f"min_count_after_filter={int(_cnt_u.min()) if len(_cnt_u) else 0})")}
        except Exception as _e:
            import traceback as _tb
            _msg = f"{type(_e).__name__}: {_e}"
            print(f"  ⚠️  information gain 계산 실패(무시함): {_msg}")
            # 진단을 위해 예외 내용을 결과에도 남긴다 — 콘솔을 놓쳐도
            # pkl만 보면 원인을 알 수 있게.
            information_gain = {"error": _msg, "traceback": _tb.format_exc()[-800:]}

    return {"branch_info": info, "redundancy": redundancy,
            "information_gain": information_gain}


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


    _oof = red.get("redundancy_method") == "out_of_fold_ridge"
    _lbl = "out-of-fold Ridge" if _oof else "in-sample OLS [구버전]"
    print(f"\n  Redundancy(선형 복원 가능한 정도, R² — {_lbl}):")
    _sm = red.get("shape_meta") or {}
    if _sm:
        print(f"    {'branch':<10}{'n':>7}{'p':>7}{'p/n':>8}{'rank':>7}{'eff_rank':>10}")
        for _b in ("query", "context", "agg"):
            _m = _sm.get(_b)
            if _m:
                print(f"    {_b:<10}{_m['n']:>7}{_m['p']:>7}{_m['p_over_n']:>8.2f}"
                      f"{_m['rank']:>7}{_m['effective_rank']:>10.1f}")
        _pn = max((m["p_over_n"] for m in _sm.values()), default=0.0)
        if _pn > 1.0:
            print(f"    ⚠️  p/n>1 — in-sample 선형회귀라면 R²가 내용과 무관하게 "
                  f"1.000이 되는 구간. out-of-fold 값만 해석할 것.")
    _pairs = [("agg_emb", "query_emb", "agg_from_query"),
              ("context_emb", "query_emb", "context_from_query"),
              ("context_emb", "agg_emb", "context_from_agg"),
              ("agg_emb", "context_emb", "agg_from_context")]
    for _t, _s, _k in _pairs:
        if f"{_k}_r2" not in red:
            continue
        _line = f"    {_t:<11} ~ f({_s:<11}): R²={red[f'{_k}_r2']:+.3f}"
        if f"{_k}_r2_insample" in red:
            _line += f"   (in-sample {red[f'{_k}_r2_insample']:.3f})"
        print(_line)
        if red.get(f"{_k}_error"):
            print(f"      ⚠️  {red[f'{_k}_error']}")
    if "context_from_agg_r2" in red:
        _mx = max(red["context_from_agg_r2"], red["agg_from_context_r2"])
        if _mx > 0.9:
            print("      → 두 branch가 사실상 같은 정보(R²>0.9) — head 입력이 "
                  "[query ‖ 그룹요약 ‖ 그룹요약]에 가까움.")
        elif _mx < 0.5:
            print("      → 두 branch가 서로 다른 정보를 나름(R²<0.5).")
    if _oof:
        print("    (음수 R²는 '평균으로 예측하는 것보다 못하다' = 복원 불가라는 뜻. "
              "in-sample 값과 크게 벌어지면 그 차이가 과적합/보간 용량의 크기.)")
    ig = result.get("information_gain")
    if ig and "error" in ig:
        # [2026-07] 이전에는 error dict도 truthy라 아래 ig['auroc_query']에서
        # KeyError로 죽었다(1493처럼 클래스당 표본이 1개인 경우 재현).
        print("\n  [information gain] 측정 불가 — " + str(ig["error"]))
        if "traceback" in ig:
            print("    (traceback은 결과 pkl의 information_gain['traceback']에 저장됨)")
    elif ig:
        print("\n  [information gain] 동일 선형 probe를 입력만 바꿔 학습 (AUROC / acc)")
        _pm = ig.get("probe_meta") or {}
        if _pm:
            _qa = _pm.get("query_agg", {})
            print(f"    probe 형상: n={_qa.get('n','?')}, "
                  f"p(query)={_pm.get('query',{}).get('p','?')}, "
                  f"p(query+agg)={_qa.get('p','?')}, "
                  f"p/n={_qa.get('p_over_n', float('nan')):.2f}")
            if ig.get("probe_overdetermined"):
                print("    ⚠️  p/n>1 — 선형 probe가 과결정 구간. gain·null·delta "
                      "모두 해석 불가로 볼 것(표본을 늘리거나 이 데이터셋은 제외).")
        print(f"    query only        : {ig['auroc_query']:.4f} / {ig['acc_query']:.4f}")
        print(f"    query + agg       : {ig['auroc_query_agg']:.4f} / {ig['acc_query_agg']:.4f}"
              f"   (gain {ig['auroc_gain_agg']:+.4f} / {ig['acc_gain_agg']:+.4f})")
        print(f"    query + context   : {ig['auroc_query_context']:.4f} / {ig['acc_query_context']:.4f}"
              f"   (gain {ig['auroc_gain_context']:+.4f} / {ig['acc_gain_context']:+.4f})")
        if ig.get("n_shuffles", 0) > 0:
            print(f"\n    [null 대조] shuffle {ig['n_shuffles']}회 — "
                  f"delta = gain − null_mean (차원 페널티 보정)")
            for _n in ("agg", "context"):
                _g = ig.get(f"auroc_gain_{_n}")
                _pm = ig.get(f"auroc_gain_{_n}_shuffled_mean")
                if _g is None or _pm is None:
                    _err = ig.get(f"null_error_{_n}_plain")
                    if _err:
                        print(f"    {_n:<8}: null 계산 실패 — {_err}")
                    continue
                _ps = ig.get(f"auroc_gain_{_n}_shuffled_std", float('nan'))
                print(f"    {_n:<8}: gain {_g:+.4f} | plain null {_pm:+.4f}±{_ps:.4f}"
                      f" | delta {ig[f'auroc_gain_{_n}_delta']:+.4f}")
                if ig.get(f"resid_null_degenerate_{_n}"):
                    print(f"    {'':<8}  resid null 축퇴 — 잔차 비율 "
                          f"{ig.get(f'resid_norm_ratio_{_n}', float('nan')):.2e} "
                          f"< 1e-6 이라 대조군이 원본과 같아짐. delta_resid=NaN.")
                _rm = ig.get(f"auroc_gain_{_n}_resid_null_mean")
                if _rm is not None and _rm == _rm:
                    _rs = ig.get(f"auroc_gain_{_n}_resid_null_std", float('nan'))
                    _dr = ig[f"auroc_gain_{_n}_delta_resid"]
                    print(f"    {'':<8}  resid null {_rm:+.4f}±{_rs:.4f}"
                          f" | delta_resid {_dr:+.4f}  [보조 진단]")
                    if abs(ig[f"auroc_gain_{_n}_delta"] - _dr) > 0.02:
                        print(f"    {'':<8}  → 두 delta 차이가 큼: gain의 상당 부분이 "
                              f"query와의 공선성에서 옴(R-3 관련).")
        print("    (gain이 0에 가까우면 그 branch가 라벨에 대해 query 이상의 정보를 주지 못함. "
              "test set 위에서 probe를 학습·평가하므로 절대값이 아닌 상대 비교용. "
              "선형 probe 기준이므로 '정보가 없다'가 아니라 "
              "'linearly-decodable 정보가 없다'로 서술할 것)")
        if ig.get("n_classes_dropped", 0) > 0:
            print(f"    ⚠️  표본 2개 미만 클래스 {ig['n_classes_dropped']}개 제외 "
                  f"(클래스 {ig['n_classes_used']}/{ig['n_classes_total']}, "
                  f"샘플 {ig['n_samples_used']}/{ig['n_samples_total']}). "
                  f"세 probe 모두 같은 부분집합에서 평가되므로 상대 비교는 유효하지만, "
                  f"제외 비율이 크면 해석에 주의.")
        if ig.get("query_auroc_ceiling"):
            print(f"    ⚠️  auroc_query={ig['auroc_query']:.4f} — 천장 포화. "
                  f"gain을 잴 여지가 없으므로 이 데이터셋의 information gain은 무효로 볼 것.")
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
            # forward는 이제 라우팅 설명만 만든다(이웃 조립/그룹 통계는
            # diagnostics로 빠짐) — 여기서 비용을 따로 끌 이유가 없어졌다.
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

    # ⚠ Any fusion_mode other than the default must appear in the filename.
    #   Listing modes one by one left proto_dev_retr (and proto_dev_vec,
    #   proto_only, ...) untagged, so their checkpoints overwrote the default
    #   run's: the study tag told them apart but the checkpoint name did not.
    _save_tag = (f"..retrproj_{args.retr_proj_mode}" if args.retr_proj_mode != "none" else "") \
              + ("..detachretr" if args.detach_retr_grad else "") \
              + ("..global_retrieve" if args.global_retrieve else "") \
              + ("..detach_ctx" if args.detach_context_grad else "") \
              + (f"..qDetachWarmupE{args.query_detach_warmup_epochs}" if args.query_detach_warmup_epochs > 0 else "") \
              + (f"..qDetachWarmupS{args.query_detach_warmup_steps}" if args.query_detach_warmup_steps > 0 else "") \
              + ("..confscale" if args.confidence_scaling else "") \
              + ("..confscale_detach" if (args.confidence_scaling and args.confidence_scaling_detach) else "") \
              + ("..no_query_emb" if args.no_query_emb else "") \
              + ("..no_context_emb" if args.no_context_emb else "") \
              + ("..grad_cb" if args.gradient_codebook else "") \
              + (f"..ema_decay{args.ema_decay_override:g}" if args.ema_decay_override is not None else "") \
              + ("..blockLN" if args.blockwise_layernorm else "") \
              + ("..branchL2norm" if args.head_branch_l2norm else "") \
              + (f"..fusion_{args.fusion_mode}" if args.fusion_mode != "proto_dev" else "") \
              + ("..no_retrieval" if args.disable_retrieval_branch else "") \
              + (f"..infT{args.inference_evidence_temperature:g}" if args.inference_evidence_temperature is not None else "") \
              + (f"..k{args.k_override}" if args.k_override is not None else "") \
              + (f"..gateT{args.fusion_gate_temperature:g}" if args.fusion_gate_temperature != 1.0 else "") \
              + ("..allowSelfRet" if args.allow_self_retrieval else "") \
              + ("..labelOnly" if args.value_mode == "label_only" else "") \
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
              + ("..nodr" if args.disable_dead_reinit else "") \
              + (f"..nbr{args.nbr_lambda:g}" if args.nbr_lambda > 0 else "") \
              + ("..rvq" if args.residual_vq else "") \
              + (f"..cb{args.loss_codebook_override:g}" if args.loss_codebook_override is not None else "") \
              + (f"..cm{args.loss_commitment_override:g}" if args.loss_commitment_override is not None else "") \
              + (f"..drp{args.dead_reinit_patience_override}" if args.dead_reinit_patience_override is not None else "") \
              + (f"..drn{args.dead_reinit_noise_scale_override:g}" if args.dead_reinit_noise_scale_override is not None else "") \
              + (f"..trainseed{train_seed}" if train_seed != args.seed else "") \
              + ("..deterministic" if args.deterministic else "") \
              + (f"..{args.run_tag}" if args.run_tag is not None else "")

    # [2026-07 가드] run_tag가 파일명에 그대로 들어가므로 검증한다.
    # [왜] PowerShell 변수는 **대소문자를 구분하지 않는다**. `$S`(체크포인트
    # 경로)와 `$s`(seed)를 함께 쓴 스크립트에서 `--run_tag stab_s$s`가
    # `stab_s` + 전체 경로로 확장돼, 학습과 추론이 다 끝난 뒤 np.save 단계에서
    # OSError(Errno 22)로 죽었다. 몇 분치 계산이 통째로 버려진다 —
    # 시작 직후에 걸러야 한다.
    if args.run_tag is not None:
        _bad = [c for c in ('/', '\\', ':', '=', '*', '?', '"', '<', '>', '|') if c in args.run_tag]
        if _bad or len(args.run_tag) > 64:
            raise SystemExit(
                f"--run_tag 값이 파일명으로 쓸 수 없습니다: {args.run_tag!r}\n"
                + (f"  금지 문자 포함: {_bad}\n" if _bad else "")
                + (f"  길이 {len(args.run_tag)} > 64\n" if len(args.run_tag) > 64 else "")
                + "  PowerShell 변수는 대소문자를 구분하지 않습니다 — $S(경로)와 "
                  "$s(seed)를 함께 쓰면\n  같은 변수로 취급되어 경로가 태그에 "
                  "끼어듭니다. 변수명을 서로 다르게 지으세요.")

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
        # ⚠ 구버전 checkpoint 에는 nbr_k/nbr_tau/nbr_neg_margin 이 인자로
        #   들어 있다. 상수로 내리면서 생성자에서 뺐으므로 걷어내야 한다.
        model_kwargs = strip_legacy_kwargs(_saved_state["model_kwargs"])
        # [2026-07] 로드한 체크포인트의 **실제** 구조 설정을 찍는다.
        # [왜] meta.pkl은 args를 기록하므로 --from_saved_state 실행에서는
        # 실제 모델과 어긋난다. 예: P1(retr_proj=linear) 체크포인트를 불러
        # ablation을 돌려도 meta에는 retr_proj_mode='none'(기본값)이 남는다.
        # 이 때문에 정상 결과를 "잘못된 체크포인트"로 오판한 사례가 있었다.
        _key_cfg = {k: model_kwargs.get(k) for k in
                    ("retr_proj_mode", "detach_retr_grad",
                     "global_retrieve", "disable_retrieval_branch",
                     "value_mode", "use_offset_correction", "k", "embed_dim",
                     "n_prototypes", "fusion_mode")
                    if k in model_kwargs}
        print(f"  [from_saved_state] 체크포인트의 실제 설정: "
              + ", ".join(f"{k}={v}" for k, v in _key_cfg.items()))
        if _key_cfg.get("retr_proj_mode", "none") != "none":
            print(f"    ⚠️  이 체크포인트는 retrieval 전용 표현(retr_proj="
                  f"{_key_cfg['retr_proj_mode']})으로 학습됐습니다. "
                  f"meta.pkl의 retr_proj_mode는 args 기준이라 다를 수 있습니다.")
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
        if args.gradient_codebook:
            print(f"  ⚠️  --gradient_codebook은 재학습 시에만 의미가 있습니다 — "
                  f"--from_saved_state는 저장된 model_kwargs(EMA 사용 여부 포함)를 "
                  f"그대로 쓰므로 이 플래그를 무시합니다(체크포인트가 어느 "
                  f"방식으로 학습됐든 그 구조로 복원됩니다).")
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
        # no_offset_correction은 reproduce.py에 CLI 플래그 자체가 없음
        # (이미 "채택 확정"돼 하드코딩) — False 고정.
        # [2026-07] global_retrieve는 S1 실험을 위해 CLI 플래그로 열었다.
        # 여기가 False로 고정돼 있으면 --global_retrieve를 줘도 baseline
        # study(그룹 제약 O)의 best_params를 읽어와서, "전역 검색인데
        # 그룹 제약용으로 튜닝된 하이퍼파라미터"로 학습하게 된다 —
        # 에러 없이 조용히 틀리는 경로라 반드시 args를 따라가야 한다.
        _study_tag = study_pkl_tag(
            no_offset_correction=False,
            retr_proj_mode=args.retr_proj_mode,
            global_retrieve=args.global_retrieve,
            detach_context_grad=args.detach_context_grad,
            context_projection=args.context_projection,
            cat_combine=args.cat_combine,
            num_embedding=args.num_embedding,
            evidence_metric=args.evidence_metric,
            fusion_mode=args.fusion_mode,
            use_context_emb=not args.no_context_emb,
            disable_retrieval_branch=args.disable_retrieval_branch,
            # optimize.py 와 같은 규칙 — 지정했을 때만 ..P{n} 이 붙는다.
            n_prototypes=args.n_prototypes,
            gradient_codebook=args.gradient_codebook,
            no_commitment=not args.commitment,
            disable_dead_reinit=args.disable_dead_reinit,
            nbr_lambda=args.nbr_lambda,
            num_bins=args.num_bins,
            cat_embed_dim=args.cat_embed_dim,
            detach_retr_grad=args.detach_retr_grad,
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
            if args.verbose:
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
        if args.k_override is not None:
            # [통제 실험용] k(검색 이웃 수)만 격리해서 바꿈. evidence_temperature
            # 실험과 짝을 이루기 위한 것 — sharp attention이 해로운 것이
            # (a) sharpening 자체 때문인지 (b) k가 큰 상태에서 소수만 쓰게 만들어
            # 추정 분산이 폭증해서인지 분리하려면 k도 같이 줄여봐야 한다
            # (k=48에서 n_eff≈1.4면 48개 평균 대비 분산이 30배 이상).
            # memory/aggregator의 shape에 영향을 주므로 재학습 필수(로드 불가).
            _old_k = model_kwargs.get("k")
            model_kwargs["k"] = args.k_override
            best_params["k"] = args.k_override
            print(f"  [--k_override] k: {_old_k} → {args.k_override} "
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
        # ⚠ 아래 *_override 들은 **optimize.py 에 대응 플래그가 없다.**
        #   쓰면 HPO 가 탐색한 아키텍처와 재현하는 아키텍처가 달라진다.
        #   진단·ablation 용이며 기본 벤치마크에서는 쓰지 않는다.
        _ovr = [n for n, v in (
            ("evidence_temperature", args.evidence_temperature_override),
            ("regroup_warmup_epochs", args.regroup_warmup_epochs_override),
            ("dead_reinit_patience", args.dead_reinit_patience_override),
            ("dead_reinit_noise_scale", args.dead_reinit_noise_scale_override),
        ) if v is not None]
        if _ovr:
            print(f"  ⚠️  optimize.py 에 대응 플래그가 없는 override 사용: "
                  f"{', '.join(_ovr)} — HPO 와 다른 아키텍처로 재현됩니다.")
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
        if args.residual_vq:
            model_kwargs["residual_vq"] = True
            if args.residual_vq_size is not None:
                model_kwargs["residual_vq_size"] = args.residual_vq_size
            print(f"  [--residual_vq] 2단계 residual VQ 활성 "
                  f"(P2={args.residual_vq_size or 'P1과 동일'})")
        # ⚠ [버그 수정] 예전에는 `if args.nbr_lambda > 0:` 안에서만 주입했다.
        #   λ 기본값이 0.005 이던 시절엔 문제가 없었지만, 기본이 0 으로
        #   바뀌면서 **주입이 통째로 건너뛰어지고 생성자 기본값(0.005)이
        #   쓰였다.** 즉 L_nbr 을 껐는데 켜진 채로 학습됐다.
        #   (증상: 로그에 `[L_nbr] raw kNN graph 생성` 이 뜬다.)
        #   조건 없이 항상 주입한다.
        model_kwargs["nbr_lambda"] = args.nbr_lambda
        # nbr_k / nbr_tau / nbr_neg_margin 은 libs/tabera.py 의 모듈 상수다.
        # 튜닝 대상이 아니므로 주입하지 않는다.
        if args.nbr_lambda > 0:
            print(f"  [--nbr_lambda] L_nbr = {args.nbr_lambda:g} "
                  f"(k={NBR_K}, tau={NBR_TAU:g}, "
                  f"neg_margin={NBR_NEG_MARGIN})  ← 고정 상수, 튜닝 안 함")
        if not args.commitment:
            # HPO 가 탐색한 값을 덮어써서 0 으로 만든다. optimize.py 는
            # 탐색 자체를 안 하므로 저장된 best_params 에도 0 이 들어 있다.
            model_kwargs.setdefault("loss_weights", {})
            model_kwargs["loss_weights"] = {**model_kwargs.get("loss_weights", {}),
                                            "commitment": 0.0}
            if args.verbose:
                print("  commitment_loss = 0 (기본 — §10)")
        if args.disable_dead_reinit:
            # patience를 학습 epoch 수보다 크게 두면 재초기화가 한 번도
            # 발생하지 않는다 — 별도 분기 추가 없이 완전 OFF를 만든다.
            model_kwargs["dead_reinit_patience"] = 10 ** 9
            print(f"  [--disable_dead_reinit] dead_reinit 비활성화 "
                  f"(patience=1e9 — 재초기화 이벤트 없음)")
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
            retr_proj_mode=args.retr_proj_mode,
            detach_retr_grad=args.detach_retr_grad,
            global_retrieve=args.global_retrieve,
            use_context_emb=not args.no_context_emb,
            use_query_emb_in_head=not args.no_query_emb,
            use_ema_codebook=not args.gradient_codebook,
            ema_decay=args.ema_decay_override if args.ema_decay_override is not None else 0.99,
            neighbor_interaction_mode=args.neighbor_interaction_mode,
            interaction_n_heads=args.interaction_n_heads,
            aggregator_mode=args.aggregator_mode,
            head_attn_alpha_override=args.head_attn_alpha_override,
            head_neighbor_source=args.head_neighbor_source,
            blockwise_layernorm=args.blockwise_layernorm,
            head_branch_l2norm=args.head_branch_l2norm,
            fusion_mode=args.fusion_mode,
            disable_retrieval_branch=args.disable_retrieval_branch,
            learn_evidence_temperature=args.learn_evidence_temperature,
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
    # [v3] L_nbr용 raw feature kNN graph — 학습 전 1회 계산.
    #   FeatureStore는 학습 중 채워지므로 첫 epoch에 못 쓴다. offline이어야
    #   seed 간 결과 비교도 가능하다. nbr_lambda=0이면 내부에서 즉시 반환.
    model.set_nbr_graph(X_train)

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
        # Silent unless asked. These lines track how the partition formed,
        # which is a development concern rather than a result.
        # Same default as optimize.py: every 10 epochs. --regroup_log_every 0
        # turns the lines off entirely.
        regroup_log_every=args.regroup_log_every,
        time_epoch=args.time_epoch,
        log_beta=args.log_beta,
        beta_lr_mult=args.beta_lr_mult,
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
        # ⚠ v2 체크포인트(proto_head)를 v3 모델(dev_head, dev_beta_raw)에
        #   strict=True로 로드하면 즉시 실패한다. 그렇다고 strict=False로
        #   그냥 넘기면 오타나 실제 불일치까지 조용히 통과해 "실행은 되는데
        #   가중치가 안 들어간" 상태를 만든다 — 더 위험하다.
        #   그래서 **의도된 missing만** 허용하고 나머지는 즉시 실패시킨다.
        _ALLOWED_MISSING = ("dev_head.", "dev_beta_raw", "dev_gamma_raw",
                            "prototype_layer_2.")
        _miss, _unexp = model.load_state_dict(
            _saved_state["state_dict"], strict=False)
        _bad = [k for k in _miss if not k.startswith(_ALLOWED_MISSING)]
        if _bad or _unexp:
            raise RuntimeError(
                f"체크포인트 불일치 — 예상 못 한 missing {_bad[:5]}, "
                f"unexpected {list(_unexp)[:5]}. 저장 시점과 현재 모델 구조가 "
                f"다릅니다(fusion_mode/residual_vq 설정 확인).")
        if _miss:
            print(f"  [--from_saved_state] v3 신규 파라미터 {len(_miss)}개는 "
                  f"초기값 사용: {[k for k in _miss][:3]}...")
        # [2026-07, 추가] --inference_evidence_temperature — 로드된 가중치는
        # 그대로 두고 **추론 시점의 softmax 온도만** 바꾼다.
        # [왜 별도 플래그가 필요한가] --evidence_temperature_override는 재학습
        # 경로용이라, 그걸로 실험하면 encoder까지 다시 학습되어 agg 변화가
        # "attention이 sharpen돼서"인지 "모델이 다르게 학습돼서"인지 구분이
        # 안 된다. 이 플래그는 동일 가중치에서 T만 바꾸므로
        # cos(agg_before, agg_after)가 순수하게 aggregation 효과만 반영한다.
        # evidence_temperature는 학습 파라미터가 아니라 forward에서 나눗셈에
        # 쓰이는 스칼라 속성이라(evidence.py: softmax(similarities / T)),
        # state_dict 로딩 이후 덮어써도 안전하다.
        if args.inference_evidence_temperature is not None:
            _agg_mod = getattr(model, "ot_selector", None)
            if _agg_mod is None or not hasattr(_agg_mod, "evidence_temperature"):
                print(f"  ⚠️  --inference_evidence_temperature: AttentionAggregator가 "
                      f"없어(aggregator_mode≠'pooling'?) 무시합니다.")
            else:
                _old_T = _agg_mod.evidence_temperature
                _agg_mod.evidence_temperature = args.inference_evidence_temperature
                print(f"  [inference-only] evidence_temperature {_old_T} → "
                      f"{args.inference_evidence_temperature} (가중치는 그대로, 재학습 없음)")
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
        # [2026-07 버그 수정] sample_groups를 복원해도 그것만으로는 부족하다.
        # retrieve()가 실제로 보는 것은 memory._cached_groups이고, 이건
        # register_buffer가 아닌 plain 속성이라 state_dict에 안 들어간다.
        # 따라서 --from_saved_state 실행은 아래 조건에 걸려 **그룹 제약 없이
        # 전체 검색**으로 동작해 왔다:
        #     if hard_assignment is None or cached is None or n < k:  (tabera.py:570)
        # 실측 확인: 검색된 이웃이 자기 그룹 안에 있는 비율이 1493에서 0.235,
        # 46에서 0.528, 1489에서 0.668 (제약이 걸렸다면 1.000이어야 함).
        # 이 때문에 from_saved_state로 산출한 purity/margin/n_eff/topk 계열
        # 진단이 전부 "전역 검색" 기준이었다. 여기서 캐시를 재구성한다.
        if model.prototype_layer.sample_groups is not None:
            try:
                model.memory.cache_sample_groups(
                    model.prototype_layer.sample_groups,
                    device,
                    centroid_emb=model.prototype_layer.centroid_emb.detach(),
                )
                _cg = getattr(model.memory, "_cached_groups", None)
                if _cg is None:
                    print("  ⚠️  cache_sample_groups 호출했으나 캐시가 비었습니다 "
                          "(모든 그룹이 비어 있음) — 전역 검색으로 동작합니다.")
                else:
                    _sz = model.memory._cached_group_sizes
                    print(f"  [group cache] 재구성 완료 — P={_cg.shape[0]}, "
                          f"최대 그룹={_cg.shape[1]}, 중앙값={int(_sz.median().item())}, "
                          f"k={model.k}")
                    _n_fb = int((_sz < model.k).sum().item())
                    if _n_fb:
                        print(f"  ⚠️  그룹 {_n_fb}/{len(_sz)}개가 k({model.k})보다 작습니다 "
                              f"— 해당 샘플은 retrieve()에서 cross-group/전역으로 "
                              f"폴백합니다(tabera.py의 fallback_mask).")
            except Exception as _ce:
                print(f"  ⚠️  group cache 재구성 실패: {type(_ce).__name__}: {_ce} "
                      f"— 전역 검색으로 동작합니다.")
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
            # ── [메타 진단] memory staleness ─────────────────────────
            # memory.keys[i]는 학습 중 특정 시점에, **dropout mask가 걸린 채**
            # 계산된 1회성 스냅샷이다. 반면 inference query는 eval 모드의
            # 결정론적 임베딩이다. 즉 refresh 전에는
            #     memory space = noisy embedding manifold
            #     test  space  = deterministic embedding manifold
            # 로 서로 다른 공간이며, 이 상태에서 train/test routing을 비교하면
            # **encoder drift + dropout noise + distribution shift**가 섞인다.
            #
            # 이 블록은 refresh가 실제로 무엇을 바꿨는지 정량화한다 —
            # 지금까지 계산한 모든 train-side routing 지표
            # (train k-coverage / occupancy 상관 / group size 분포)의
            # 신뢰 구간을 결정하는 메타 진단이다.
            _stale_prev = None
            try:
                _nm0 = int(model.memory.filled.item())
                if _nm0 > 0:
                    _stale_prev = model.memory.keys[:_nm0].detach().float().clone()
                    _stale_prev_assign = None
                    _cN0 = torch.nn.functional.normalize(
                        model.prototype_layer.centroid_emb.detach().float(), dim=-1)
                    _stale_prev_assign = (
                        torch.nn.functional.normalize(_stale_prev, dim=-1) @ _cN0.T
                    ).argmax(-1)
            except Exception:
                _stale_prev = None

            refresh_stats = model.refresh_memory_keys()
            if refresh_stats is not None:
                if args.verbose:
                    print(f"  [--refresh_on_best] memory.keys {refresh_stats['n_refreshed']}개 "
                          f"슬롯을 frozen weight로 재계산 완료")

                if _stale_prev is not None:
                    try:
                        import torch.nn.functional as _sF
                        _new = model.memory.keys[:_stale_prev.shape[0]].detach().float()
                        # ① representation drift — 평균만 보면 안 된다.
                        #    일부 샘플만 크게 어긋나는 경우를 p5로 잡는다.
                        _cos = _sF.cosine_similarity(_stale_prev, _new, dim=-1).cpu().numpy()
                        # ② routing drift — cos이 낮아도 centroid 경계를 안 넘으면
                        #    routing 분석에는 영향이 없다. 둘을 나눠 봐야 한다.
                        #    (전체 회전이면 cos↓ / assignment 유지,
                        #     경계 crossing이면 cos↓ / assignment 변경)
                        _new_assign = (_sF.normalize(_new, dim=-1) @ _cN0.T).argmax(-1)
                        _agree = float((_new_assign == _stale_prev_assign).float().mean())
                        # ③ geometry drift — 배정 centroid까지의 거리 변화
                        _d_old = float((1 - (_sF.normalize(_stale_prev, dim=-1) @ _cN0.T).max(-1).values).mean())
                        _d_new = float((1 - (_sF.normalize(_new, dim=-1) @ _cN0.T).max(-1).values).mean())
                        print(f"  [memory staleness] cos(q_memory, q_refresh): "
                              f"mean={_cos.mean():.4f} std={_cos.std():.4f} "
                              f"p5={np.percentile(_cos,5):.4f} p50={np.percentile(_cos,50):.4f} "
                              f"p95={np.percentile(_cos,95):.4f}")
                        print(f"  [memory staleness] assignment agreement={_agree*100:.1f}%  "
                              f"| centroid dist {_d_old:.4f} → {_d_new:.4f}")
                        # npz에도 남긴다 — 사후에 "이 실행의 진단이 얼마나
                        # 믿을 만한가"를 판정하려면 결과 파일 안에 있어야 한다.
                        globals()["_MEMORY_STALENESS"] = dict(
                            cos_mean=float(_cos.mean()), cos_std=float(_cos.std()),
                            cos_p5=float(np.percentile(_cos, 5)),
                            cos_p50=float(np.percentile(_cos, 50)),
                            cos_p95=float(np.percentile(_cos, 95)),
                            assign_agreement=_agree,
                            centroid_dist_before=_d_old, centroid_dist_after=_d_new,
                        )
                        if _cos.mean() < 0.9 or _agree < 0.9:
                            print(f"  ⚠️  [memory staleness] refresh 전 train-side routing 지표는 "
                                  f"**다른 encoder state**에서 계산된 값입니다 — "
                                  f"기존 train k-coverage / occupancy 상관 / group size 재검증 필요.")
                    except Exception as _se:
                        print(f"  [memory staleness] 진단 실패: {type(_se).__name__}: {_se}")

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
            # [2026-07] agg_emb_constant/centered는 "평가 세트 전체의 agg_emb
            # 평균"을 기준으로 상수/잔차를 가른다. 배치 평균을 쓰면 배치마다
            # 기준이 달라지므로, 먼저 한 번 훑어서 전체 평균을 구해 model에
            # 넣어둔다(=이 진단은 test 통계를 쓰는 사후 분석이라는 뜻).
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
            # [Step 2 진단] head 전체를 통과한 ∂logit/∂branch.
            # 첫 Linear의 ‖W_i x_i‖(위)와 다른 것을 잰다 — 위는 "입력이
            # 얼마나 크게 들어가는가", 이건 "출력이 얼마나 민감한가".
            _jac = {}
            try:
                _jac = compute_branch_jacobian(model, X_test)
                if _jac:
                    print(f"\n  [branch Jacobian] ∂logit/∂branch (head 전체 통과)")
                    for _nm, _v in sorted(_jac.items(),
                                          key=lambda kv: -kv[1]["share_of_total"]):
                        print(f"    {_nm:<12} norm={_v['jac_norm_mean']:.4e}  "
                              f"share={_v['share_of_total']*100:5.1f}%")
            except Exception as _je:
                print(f"  ⚠️  branch Jacobian 계산 실패: {type(_je).__name__}: {_je}")
            contrib_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_branch_contribution.pkl"
            with open(contrib_path, "wb") as f:
                pickle.dump({**contrib_result, "openml_id": openml_id, "seed": args.seed,
                             "tasktype": tasktype, "jacobian": _jac}, f)
            print(f"\n  저장: {contrib_path}")

    if args.branch_information and do_analysis:
        if not hasattr(model, "_head_block_slices") or not model._head_block_slices:
            print(f"\n  ⚠️  --branch_information은 fusion_mode='concat'에서만 의미가 있습니다 — "
                  f"건너뜁니다.")
        else:
            info_result = analyze_branch_information(
                model, X_test, tasktype, y=y_test,
                n_shuffles=getattr(args, "branch_info_shuffles", 5),
                residual_null=getattr(args, "branch_info_residual_null", False))
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
    _rvq_logits_c1_chunks = []   # [v3] c1만의 logits — test 시점 Δ 계산용
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
    _rb_query_retr_chunks = []   # [Step 2 진단] test query_emb (= 라우팅 공간)
    _rb_branch_chunks = {}       # [Step 2 진단] head 3-branch 표현 {name: [chunks]}
    _rb_evraw_chunks  = {}       # [Step 2 진단] T 사후 스윕용 원본 (sim/val/ew)
    _rb_neighbor_label_chunks = []    # [2026-07] 검색된 이웃의 실제 라벨 (N, k)
    _rb_nbr_label_entropy_chunks = []
    _rb_nbr_label_neff_chunks = []
    _rb_neighbor_sid_chunks   = []    # [2026-07] 검색된 이웃의 **원본 train 행 번호** (N, k)
    _rb_purity_chunks = []            # [추가] top-k 중 query와 같은 라벨인 비율 (unweighted)
    _rb_weighted_purity_chunks = []   # [추가] evidence_w로 가중한 same-label 비율
    # [Local Retriever 진단, 추가] similarity geometry — temperature와 원인
    # 분리용(사용자 요청). evidence.py가 새 모듈 없이 항상 계산.
    _rb_sim_top1_chunks = []
    _rb_sim_bottomk_chunks = []
    _rb_sim_margin_chunks = []
    _rb_sim_std_chunks = []
    _rb_val_pwcos_chunks = []    # [추가] value 구성요소별 이웃 간 다양성
    _rb_lbl_pwcos_chunks = []
    _rb_off_pwcos_chunks = []
    _rb_val_disp_chunks = []     # [추가] 상대 퍼짐(크기 차이까지 포함)
    _rb_lbl_disp_chunks = []
    _rb_off_disp_chunks = []
    _rb_agg_emb_chunks = []      # [추가] agg_emb 원본 — 개입 전/후 cos 비교용
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
            if _out.get("rvq_logits_c1") is not None:
                _rvq_logits_c1_chunks.append(_out["rvq_logits_c1"].cpu())
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
                # ⚠ 이것은 **evidence_w(attention weight) 분포**의 entropy이지
                #   검색된 이웃의 라벨 불확실성이 아니다. 이름이 그냥
                #   `entropy`로 나가므로 "retrieval uncertainty"로 오독되기
                #   쉽다 — 그 용도로는 아래 neighbor_label_entropy를 쓸 것.
                #   proto_dev 계열에서는 evidence_w가 균등 상수(1/k)라
                #   entropy=log(k), n_eff=k, top1_weight=1/k 로 **전 샘플
                #   동일**하다(실측). 상수 컬럼이므로 AUC는 정확히 0.5,
                #   로지스틱 회귀는 특이행렬이 된다. 컬럼 자체는 모델 상태의
                #   실제 값이므로 하위호환을 위해 그대로 둔다.
                _rb_entropy_chunks.append(
                    -(_ew.clamp_min(1e-12) * _ew.clamp_min(1e-12).log()).sum(-1)
                )
                _rb_n_eff_chunks.append(1.0 / _ew.square().sum(-1).clamp_min(1e-12))
                _rb_top1_weight_chunks.append(_ew.max(dim=-1).values)
                # [Step 2 진단] test query의 routing 공간 임베딩.
                # ds=1489에서 "centroid capacity는 충분한데 test만 소수 centroid로
                # 몰린다"(train Gini 0.20 / test 0.96)가 관측됐다. 원인이
                # ① encoder representation shift ② centroid 배치 ③ P 과잉
                # 중 무엇인지 가르려면 query 임베딩 자체가 필요하다.
                # ⚠ **query_emb**를 쓴다(query_retr 아님).
                #   centroid routing은 `self.prototype_layer(query_emb)`이고
                #   memory.keys도 `embedder(raw)` = query_emb다.
                #   query_retr는 retr_proj 적용 후의 **검색 전용 공간**이라
                #   retr_proj_mode != none이면 라우팅과 다른 공간이 된다.
                #   (실측: query_retr로 계산했을 때 test entropy 3.911이 나왔는데
                #    같은 샘플의 routing_confidence는 0.556 — top1=0.556이면
                #    entropy 상한이 2.53이라 산술적으로 불가능. 공간 불일치의 증거.)
                if _out.get("query_emb") is not None:
                    _rb_query_retr_chunks.append(_out["query_emb"].detach().float().cpu())
                # ── [Step 2 진단] head 3-branch 표현 ────────────────
                # 질문: head가 agg_emb를 안 쓰는 이유가
                #   (A) agg가 context/query와 정보가 중복이라 쓸 게 없어서인가
                #   (B) 정보는 있는데 concat+공유 MLP가 못 쓰는가
                # 기존 기록은 B 쪽을 가리킨다 —
                #   · tabera.py:1128 probe 실측(1043/31): concat(q+c+a)가
                #     최고 단일 branch보다도 낮고, branch별 L2-normalize로 회복
                #   · 인계 문서: I(Y;agg|query) 근사 d_plain=+0.0071 (5/5 양수, p=0.016)
                # ⚠ 그러나 그 기록들은 전부 --refresh_on_best 없이(stale sample_groups)
                #   나온 것이라 agg_emb 자체가 다른 그룹에서 검색한 결과였다.
                #   fresh 조건에서 재측정이 필요하다.
                # ⚠ 그리고 기존 조건부는 query 기준이다 — **context 조건부**
                #   중복(I(Y;agg|context))은 아직 측정된 적이 없다.
                #
                # head 입력 직전(정규화/ablation 적용 전) raw 값을 저장한다 —
                # "이 표현 자체에 정보가 있는가"를 보는 것이므로 원본이 맞다.
                for _nm, _key in (("query", "query_emb"),
                                  ("context", "context_emb"),
                                  ("agg", "agg_emb")):
                    _v = _out.get(_key)
                    if _v is not None:
                        _rb_branch_chunks.setdefault(_nm, []).append(
                            _v.detach().float().cpu())
                # [Step 2 진단] T 사후 스윕용 원본 — agg(T) = softmax(sim/T) @ values
                _ed = _out.get("evidence_diag") or {}
                for _nm, _key in (("sim", "similarities_raw"),
                                  ("val", "values_raw"),
                                  ("ew",  "evidence_w_raw"),
                                  ("T",   "evidence_T")):
                    _v = _ed.get(_key)
                    if _v is not None:
                        _rb_evraw_chunks.setdefault(_nm, []).append(
                            _v.detach().float().cpu())
                # [추가, 사용자 요청] retrieval label purity — "무엇을 가져왔는가"를
                # 직접 잼. topk_idx는 memory bank(학습셋) 인덱스라 model.memory.labels
                # 로 바로 라벨을 찾을 수 있음(새 forward 불필요, 이미 계산된 topk_idx/
                # evidence_w 재사용). regression은 label purity 개념이 없어 스킵.
                if tasktype != "regression":
                    _batch_y = y_test[_start:_start + _pred_batch_size].cpu()
                    _batch_y_int = torch.round(_batch_y).long()
                    _neighbor_labels = model.memory.labels[_out["topk_idx"]].cpu().long()  # (B, k)
                    # [2026-07 추가] 이웃 라벨 자체를 저장한다.
                    # [왜] topk_idx는 **memory bank 위치**이지 X_train 행 번호가
                    # 아니다 — memory.update()가 학습 중 셔플된 배치 순서로
                    # 채우고(tabera.py:442~), 원래 행 번호는 sample_ids에 따로
                    # 보관된다. 그래서 외부 분석에서 y_train[topk_idx]로 라벨을
                    # 복원하려 하면 조용히 엉뚱한 값이 나온다(실측: purity
                    # 재계산이 저장값과 최대 0.875~1.00 차이). 인덱스를 매핑하게
                    # 하는 대신 라벨을 직접 내보내 그 실수 자체를 불가능하게 한다.
                    _rb_neighbor_label_chunks.append(_neighbor_labels)
                    # [2026-07 추가] 이웃의 원본 train 행 번호.
                    # [왜] topk_idx는 MemoryBank 슬롯 위치이고, 그 순서는 학습 중
                    # 셔플된 배치 순서로 정해진다(tabera.py:442~). 따라서 **seed가
                    # 다른 두 모델의 topk_idx를 직접 비교하면 무의미**하다 — 같은
                    # 번호가 서로 다른 샘플을 가리킨다. memory.sample_ids가 슬롯 →
                    # 원본 행 번호 매핑을 들고 있으므로, 그걸 통과시킨 값을 저장하면
                    # seed 간 "같은 이웃을 골랐는가"를 비교할 수 있다.
                    # (설명 재현성 측정에 필수 — 없으면 아예 계산이 불가능하다.)
                    _rb_neighbor_sid_chunks.append(
                        model.memory.sample_ids[_out["topk_idx"]].cpu().long())
                    # [추가] 이웃 **라벨 분포**의 entropy — H(Y_N(x)).
                    #
                    # ⚠ 위쪽에서 저장하는 `entropy` 컬럼과 완전히 다른 값이다.
                    #   그건 evidence_w(attention weight) 분포의 entropy이고,
                    #   proto_dev 계열에서는 evidence_w가 균등 상수(1/k)라
                    #   **모든 샘플에서 정확히 log(k)** 로 고정된다(n_eff=k,
                    #   top1_weight=1/k도 같이 상수, weighted_purity는
                    #   unweighted purity와 완전히 동일해짐 — 실측 확인).
                    #   즉 그 컬럼으로는 상수 예측자밖에 만들 수 없고,
                    #   AUC는 정확히 0.5, 회귀는 특이행렬이 된다.
                    #
                    # ⚠ 이름이 그냥 `entropy`라 "retrieval uncertainty"로
                    #   읽히기 쉬운 것이 문제의 핵심이었다. 기존 컬럼은
                    #   하위호환을 위해 그대로 두되(모델 상태의 실제 값이긴
                    #   하다), 출처를 이름에 박은 이 컬럼을 따로 내보낸다.
                    #   retrieval uncertainty를 보려면 **이쪽**을 쓸 것.
                    _n_cls = int(model.memory.labels[:int(model.memory.filled.item())]
                                 .max().item()) + 1
                    _cnt = torch.zeros(_neighbor_labels.shape[0], _n_cls)
                    _cnt.scatter_add_(1, _neighbor_labels,
                                       torch.ones_like(_neighbor_labels, dtype=torch.float))
                    _p_lab = _cnt / _cnt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    _H_lab = -(_p_lab * (_p_lab + 1e-12).log()).sum(-1)      # (B,)
                    _rb_nbr_label_entropy_chunks.append(_H_lab)
                    # 유효 라벨 종류 수 = exp(H). "이웃 k개가 실질적으로 몇
                    # 종류의 답을 담고 있는가" — entropy와 같은 정보를
                    # 개수 단위로 읽는 값(evidence n_eff와 혼동 금지).
                    _rb_nbr_label_neff_chunks.append(torch.exp(_H_lab))
                    _same_label = (_neighbor_labels == _batch_y_int.unsqueeze(-1)).float()  # (B, k)
                    _rb_purity_chunks.append(_same_label.mean(dim=-1))            # unweighted: 단순 top-k 중 동일 라벨 비율
                    _rb_weighted_purity_chunks.append((_ew * _same_label).sum(dim=-1))  # evidence_w-weighted: 실제 aggregation에 반영되는 비중까지 고려
                if _out.get("similarity_top1_per_sample") is not None:
                    _rb_sim_top1_chunks.append(_out["similarity_top1_per_sample"].cpu())
                    _rb_sim_bottomk_chunks.append(_out["similarity_bottomk_per_sample"].cpu())
                    _rb_sim_margin_chunks.append(_out["similarity_margin_per_sample"].cpu())
                    _rb_sim_std_chunks.append(_out["similarity_std_per_sample"].cpu())
                # [추가] value/label/offset 각각의 이웃 간 평균 pairwise cosine —
                # "attention을 sharpen해도 소용없는가"(value collapse)를 판정.
                # k<2면 evidence.py가 None을 반환하므로 그때는 안 쌓임.
                if _out.get("value_pairwise_cos_per_sample") is not None:
                    _rb_val_pwcos_chunks.append(_out["value_pairwise_cos_per_sample"].cpu())
                    _rb_lbl_pwcos_chunks.append(_out["label_pairwise_cos_per_sample"].cpu())
                    if _out.get("offset_pairwise_cos_per_sample") is not None:
                        _rb_off_pwcos_chunks.append(_out["offset_pairwise_cos_per_sample"].cpu())
                    _rb_val_disp_chunks.append(_out["value_rel_dispersion_per_sample"].cpu())
                    _rb_lbl_disp_chunks.append(_out["label_rel_dispersion_per_sample"].cpu())
                    if _out.get("offset_rel_dispersion_per_sample") is not None:
                        _rb_off_disp_chunks.append(_out["offset_rel_dispersion_per_sample"].cpu())
                # [추가] agg_emb 원본 저장 — temperature/metric 개입 전후로
                # cos(agg_before, agg_after)를 직접 비교하기 위함. "attention이
                # 변했는데 agg는 그대로"인지를 중간 지표가 아니라 최종 산출물에서
                # 확인할 수 있어야 원인 사슬(Δattention→Δagg→Δpred)이 닫힌다.
                if _out.get("agg_emb") is not None:
                    _rb_agg_emb_chunks.append(_out["agg_emb"].detach().cpu())
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
        # [메타 진단] 이 실행의 memory가 inference-consistent였는가.
        # refresh를 안 했으면 이 키가 없다 = train-side 지표가
        # 학습 시점 dropout 스냅샷 기준이라는 뜻.
        _ms = globals().get("_MEMORY_STALENESS")
        if _ms:
            for _k, _v in _ms.items():
                _rb_savez_kwargs[f"staleness_{_k}"] = np.array([_v])
        _rb_savez_kwargs["memory_refreshed"] = np.array([1 if args.refresh_on_best else 0])

        # ── [Step 2 진단] head 3-branch 표현 저장 ────────────────
        # branch_query / branch_context / branch_agg  각 (N, D)
        # 사후 분석용: CKA, linear probe, conditional probe
        #   acc(context) vs acc(agg) vs acc(context+agg)
        #   → context+agg ≈ context 이면 중복(A)
        #   → context+agg >> context 이면 정보는 있는데 fusion이 못 씀(B)
        # ⚠ 차원이 커질 수 있어(N×D×3) float16으로 저장한다 —
        #   CKA/probe는 이 정밀도로 충분하다.
        try:
            for _nm, _ch in _rb_branch_chunks.items():
                if _ch:
                    _rb_savez_kwargs[f"branch_{_nm}"] = (
                        torch.cat(_ch, dim=0).numpy().astype(np.float16))
            # T 사후 스윕용 — values는 (N, k, D)라 크므로 float16
            for _nm, _ch in _rb_evraw_chunks.items():
                if _ch:
                    _rb_savez_kwargs[f"ev_{_nm}"] = (
                        torch.cat(_ch, dim=0).numpy().astype(np.float16))
            # ── [Step 2 진단] 표현 다양성 — agg가 centroid 상수로 붕괴했는가 ──
            # 관측: agg의 effective rank가 1.0~8.8 (query는 23~66),
            #       조건수 1e8대 → 수치적으로 특이. within/total 분산비 0.01~0.19.
            # [H] value = label_emb(label_only)라 같은 그룹 안에서 거의 동일한
            #     벡터를 평균내므로 agg가 centroid identity를 그대로 반영한다.
            #     `--value_mode default/offset_only`로 offset(query−neighbor)
            #     성분을 넣으면 query별로 달라져 rank가 회복될 수 있다.
            # ⚠ 원인이 value인지 candidate set인지 aggregation인지는 아직
            #   분리되지 않았다. value_mode 비교가 그 분리 실험이다.
            #
            # effective rank(엔트로피 기반)와 participation ratio를 함께 잰다 —
            #   PR = (Σλ)² / Σλ²   "주성분 몇 개가 실제 분산을 설명하는가"
            # value(집계 전)와 agg(집계 후)를 둘 다 재야
            # "value가 원래 다양한데 평균이 죽인 것"과
            # "value 자체가 이미 같은 것"이 갈린다.
            def _rank_stats(M):
                M = np.asarray(M, dtype=np.float64)
                M = M - M.mean(0, keepdims=True)
                if M.shape[0] < 2:
                    return {}
                sv = np.linalg.svd(M, compute_uv=False)
                sv = sv[sv > 0]
                if sv.size == 0:
                    return {}
                lam = sv ** 2                      # 공분산 고유값
                p = lam / lam.sum()
                q = sv / sv.sum()                  # 특이값 기반(비교용)
                # ⚠ 두 정의를 함께 저장한다. `--branch_information`의
                #   effective_rank는 **특이값** 기반(eff_rank_sv)이고,
                #   표준적인 공분산 정의는 **고유값** 기반(eff_rank)이다.
                #   제곱하면 분포가 뾰족해져 값이 크게 달라진다
                #   (실측 ds=14 query: 23.7 vs 7.8). 섞어서 비교하면 안 된다.
                return {
                    "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
                    "eff_rank_sv": float(np.exp(-(q * np.log(q)).sum())),
                    "participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
                    "top1_var_ratio": float(p[0]),
                }
            try:
                _rk = {}
                for _nm in ("query", "context", "agg"):
                    _k = f"branch_{_nm}"
                    if _k in _rb_savez_kwargs:
                        for _m, _v in _rank_stats(_rb_savez_kwargs[_k]).items():
                            _rk[f"{_nm}_{_m}"] = _v
                if "ev_val" in _rb_savez_kwargs:
                    # 집계 전 value — (N*k, D)로 펼쳐서 잰다
                    _vv = _rb_savez_kwargs["ev_val"]
                    for _m, _v in _rank_stats(_vv.reshape(-1, _vv.shape[-1])).items():
                        _rk[f"value_{_m}"] = _v
                if _rk:
                    _rb_savez_kwargs["rank_stats_keys"] = np.array(
                        sorted(_rk.keys()), dtype=object)
                    _rb_savez_kwargs["rank_stats_vals"] = np.array(
                        [_rk[k] for k in sorted(_rk.keys())], dtype=np.float64)
                    print("  [표현 다양성] " + "  ".join(
                        f"{_n}: eff_rank={_rk.get(_n+'_eff_rank', float('nan')):.2f}"
                        f"/PR={_rk.get(_n+'_participation_ratio', float('nan')):.2f}"
                        for _n in ("query", "context", "agg", "value")
                        if _n + "_eff_rank" in _rk))
            except Exception as _re:
                _rb_savez_kwargs["rank_stats_error"] = np.array(
                    [f"{type(_re).__name__}: {_re}"], dtype=object)

            # 자기검증: 저장된 sim/val로 재계산한 agg가 실제 branch_agg와 같은가.
            # 안 맞으면 어딘가 빠진 변환이 있다는 뜻이라 T 스윕 결과 전체가 무효다.
            if ("ev_sim" in _rb_savez_kwargs and "ev_val" in _rb_savez_kwargs
                    and "branch_agg" in _rb_savez_kwargs):
                _s = _rb_savez_kwargs["ev_sim"].astype(np.float32)
                _v = _rb_savez_kwargs["ev_val"].astype(np.float32)
                _T0 = float(np.mean(_rb_savez_kwargs.get("ev_T", np.array([1.0]))))
                _w = np.exp((_s - _s.max(1, keepdims=True)) / max(_T0, 1e-8))
                _w /= _w.sum(1, keepdims=True)
                _re = np.einsum("bk,bkd->bd", _w, _v)
                _ref = _rb_savez_kwargs["branch_agg"].astype(np.float32)
                _rel = float(np.linalg.norm(_re - _ref) / max(np.linalg.norm(_ref), 1e-12))
                _rb_savez_kwargs["ev_recon_rel_error"] = np.array([_rel])
                print(f"  [ev raw] agg 재계산 상대오차 = {_rel:.2e}"
                      + ("" if _rel < 1e-2 else
                         "   ⚠️ 0.01 초과 — 빠진 변환이 있을 수 있음. T 스윕 결과 신뢰 불가"))
        except Exception as _be:
            _rb_savez_kwargs["branch_export_error"] = np.array(
                [f"{type(_be).__name__}: {_be}"], dtype=object)

        # ── [Step 2 진단] memory staleness — 진단 유효성의 메타 검증 ──
        # `memory.keys`는 **학습 중 특정 시점에, dropout mask가 걸린 채로**
        # 계산된 1회성 스냅샷이다(`--refresh_on_best` 기본값 False).
        # 반면 test query_emb는 최종 frozen weight로 eval 모드에서 계산된다.
        #
        #   memory space  q = f_θ_old(x, ε)     noisy manifold
        #   test   space  q = f_θ_final(x)      deterministic manifold
        #
        # 즉 시점 차이만이 아니라 **함수 자체가 다르다.** 이 상태로 train/test를
        # 비교하면 encoder drift + dropout noise + distribution shift가 섞인다.
        # `train_assignment`/`sample_groups`도 이 keys에서 나오므로
        # k-coverage·occupancy 상관·group size 전부 영향을 받는다.
        #
        # 여기서는 **덮어쓰지 않고** 현재 weight로 다시 인코딩한 값과 비교만 한다.
        # 실제 교정은 `--refresh_on_best`가 담당한다.
        try:
            import torch.nn.functional as _F
            _nmem0 = int(model.memory.filled.item())
            if _nmem0 > 0 and getattr(model, "_feature_store", None) is not None:
                _was = model.training
                model.eval()
                with torch.no_grad():
                    _dev = model.memory.keys.device
                    _fresh = []
                    for _st in range(0, _nmem0, 1024):
                        _en = min(_st + 1024, _nmem0)
                        _fresh.append(model.embedder(model._feature_store._store[_st:_en].to(_dev)))
                    _fresh = torch.cat(_fresh, dim=0).float()
                    _old = model.memory.keys[:_nmem0].detach().float()
                    # ① representation drift
                    _cos = _F.cosine_similarity(_F.normalize(_old, dim=-1),
                                                _F.normalize(_fresh, dim=-1), dim=-1)
                    # ② routing drift — cos만으로는 부족하다. 합성 대조에서
                    #    cos=0.874인데 배정 일치가 51.8%까지 떨어졌다
                    #    (표현은 비슷해 보여도 배정은 절반이 뒤집힘).
                    #    centroid는 고정한 채 두 임베딩의 argmax를 비교한다.
                    _cN0 = _F.normalize(model.prototype_layer.centroid_emb.detach().float(), dim=-1)
                    _a_old = (_F.normalize(_old, dim=-1) @ _cN0.T).argmax(-1)
                    _a_new = (_F.normalize(_fresh, dim=-1) @ _cN0.T).argmax(-1)
                    _agree = (_a_old == _a_new).float().mean()
                if _was:
                    model.train()
                _c = _cos.cpu().numpy()
                # ③ group consistency — ARI. agreement는 centroid **인덱스**까지
                #    같아야 하지만, ARI는 **그룹 구조**만 본다.
                #    합성 대조: 인덱스만 순열하면 agreement 15.4% / ARI 1.0000
                #    (구조는 동일한데 이름표만 바뀐 것 — 오염이 아니다).
                #    10%가 실제로 다른 그룹으로 가면 agreement 89.5% / ARI 0.790.
                #    즉 셋을 함께 봐야 "표현 drift / 배정 drift / 구조 drift"가 갈린다.
                _ari = float("nan")
                try:
                    from sklearn.metrics import adjusted_rand_score as _ARI
                    _ari = float(_ARI(_a_old.cpu().numpy(), _a_new.cpu().numpy()))
                except Exception:
                    pass
                _rb_savez_kwargs.update(
                    memory_refresh_cos=_c,
                    memory_refresh_assign_agreement=np.array([float(_agree)]),
                    memory_refresh_group_ari=np.array([_ari]),
                )
                print(f"  [memory staleness] cos(q_memory, q_refresh) "
                      f"mean={_c.mean():.4f} std={_c.std():.4f} "
                      f"p5={np.percentile(_c,5):.4f} p50={np.median(_c):.4f} p95={np.percentile(_c,95):.4f}"
                      f" | assign agreement={float(_agree)*100:.1f}% | group ARI={_ari:.4f}")
                if _c.mean() < 0.98 or _ari < 0.95:
                    print(f"  ⚠️  [memory staleness] memory.keys가 최종 encoder와 어긋남 — "
                          f"train 쪽 routing 지표(k-coverage, occupancy 상관, group size, "
                          f"dead centroid 판정)를 그대로 해석하지 말 것. "
                          f"`--refresh_on_best`로 재실행 권장.")
        except Exception as _se:
            _rb_savez_kwargs["memory_staleness_error"] = np.array(
                [f"{type(_se).__name__}: {_se}"], dtype=object)

        # ── [Step 2 진단] train/test routing geometry ────────────────
        # 질문: CentroidLayer의 실패가 capacity collapse가 아니라
        #       representation shift 때문인가?
        # ds=1489 관측 — centroid k-coverage 94.2%, train 샘플 k-coverage 98.8%
        #   (capacity 정상)인데 test 샘플 k-coverage는 17.0%.
        #   train은 65개 centroid에 고르게 퍼지고(Gini 0.20) test는 7개에 몰린다(0.96).
        #
        # memory.keys가 train query의 routing 공간 임베딩을 그대로 갖고 있어
        # **재인코딩 없이** train/test를 동일 공간에서 비교할 수 있다.
        # 라우팅이 코사인이므로 모든 거리는 정규화 후 계산한다.
        try:
            if _rb_query_retr_chunks:
                import torch.nn.functional as _F
                _cE = model.prototype_layer.centroid_emb.detach().float()          # (P, D)
                _cN = _F.normalize(_cE, dim=-1)
                _nmem = int(model.memory.filled.item())
                # memory.keys = embedder(raw) = query_emb — 라우팅과 동일 공간
                _trQ = _F.normalize(model.memory.keys[:_nmem].detach().float(), dim=-1)   # (N_tr, D)
                _teQ = _F.normalize(torch.cat(_rb_query_retr_chunks, dim=0).to(_cE.device), dim=-1)

                def _geo(Q):
                    sim = Q @ _cN.T                                   # (n, P) 코사인
                    top2 = sim.topk(2, dim=-1)
                    d1 = 1.0 - top2.values[:, 0]                      # 배정 centroid까지 코사인 거리
                    margin = top2.values[:, 0] - top2.values[:, 1]    # 1등−2등 (가설 C용)
                    p = torch.softmax(sim * float(model.prototype_layer.routing_scale), dim=-1)
                    ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
                    return (d1.cpu().numpy(), margin.cpu().numpy(), ent.cpu().numpy(),
                            top2.indices[:, 0].cpu().numpy())

                # ── [Level 1] local ordering — neighborhood preservation ──
                # [정의] Local ordering is the preservation of meaningful
                #   neighborhood structure within each retrieval partition.
                #   Raw-feature neighborhood preservation is **one measurable
                #   proxy**, not the definition itself — TabERA의 임베딩은
                #   예측을 위해 학습된 표현이므로 raw 유클리드 거리를 반드시
                #   보존해야 할 이유는 없다(범주형 임베딩, 비선형 표현 등).
                #
                # [왜 필요한가] 지금까지 retrieval 평가는 전부 label 축이었다
                #   (gain over random / rank_info / purity). Stage 1이
                #   classification을 거의 완성하므로, label 축으로 재면
                #   추가 정보가 0이 나오는 것이 자연스럽다.
                #   Level 1은 **label을 쓰지 않고** 순서 자체를 평가한다.
                #
                # 측정: 같은 centroid 그룹 안에서
                #   raw feature 거리 순위  vs  임베딩 거리 순위
                #   → Spearman 상관, Recall@k (raw 기준 top-k를 임베딩이 몇 개 찾나)
                #
                # ⚠ Stage 1(proto_only)은 label loss만 쓴다. 그런데도 raw
                #   geometry가 유지되면 흥미로운 결과이고, 붕괴하면
                #   "local ordering이 실제로 사라졌다"고 말할 수 있다.
                try:
                    _fs = getattr(model, "_feature_store", None)
                    if _fs is not None and _nmem > 0:
                        _raw = _fs._store[:_nmem].detach().float().to(_cE.device)
                        _asg = _trQ @ _cN.T
                        _asg = _asg.argmax(-1).cpu().numpy()
                        _sp, _rc, _ng = [], [], 0
                        _kk = min(int(getattr(model, "k", 16)), 16)
                        for _c in np.unique(_asg):
                            _idx = np.where(_asg == _c)[0]
                            if len(_idx) < _kk + 2:
                                continue
                            if len(_idx) > 400:      # 비용 상한
                                _idx = _idx[:400]
                            _ii = torch.as_tensor(_idx, device=_cE.device)
                            _R = _raw[_ii]
                            _E = _trQ[_ii]
                            _dR = torch.cdist(_R, _R)
                            _dE = 1.0 - (_E @ _E.T)          # 임베딩은 코사인 기준
                            _n = len(_idx)
                            _eye = torch.eye(_n, device=_cE.device, dtype=torch.bool)
                            _dR = _dR.masked_fill(_eye, float("inf"))
                            _dE = _dE.masked_fill(_eye, float("inf"))
                            # Spearman: 각 행의 거리 순위 상관
                            _rR = _dR.argsort(-1).argsort(-1).float()
                            _rE = _dE.argsort(-1).argsort(-1).float()
                            _rRc = _rR - _rR.mean(-1, keepdim=True)
                            _rEc = _rE - _rE.mean(-1, keepdim=True)
                            _num = (_rRc * _rEc).sum(-1)
                            _den = (_rRc.norm(dim=-1) * _rEc.norm(dim=-1)).clamp_min(1e-12)
                            _sp.append(float((_num / _den).mean()))
                            # Recall@k: raw 기준 top-k 중 임베딩 top-k에 몇 개
                            _tR = _dR.topk(_kk, dim=-1, largest=False).indices
                            _tE = _dE.topk(_kk, dim=-1, largest=False).indices
                            _hit = (_tR.unsqueeze(-1) == _tE.unsqueeze(1)).any(-1).float().mean()
                            _rc.append(float(_hit))
                            _ng += 1
                        if _sp:
                            _rb_savez_kwargs.update(
                                lo_spearman_mean=np.array([float(np.mean(_sp))]),
                                lo_recall_at_k_mean=np.array([float(np.mean(_rc))]),
                                lo_n_groups=np.array([_ng]),
                                lo_spearman_per_group=np.array(_sp, dtype=np.float64),
                                lo_recall_per_group=np.array(_rc, dtype=np.float64),
                            )
                            print(f"  [local ordering] raw-vs-embedding  "
                                  f"Spearman={np.mean(_sp):.4f}  "
                                  f"Recall@{_kk}={np.mean(_rc):.4f}  "
                                  f"(groups={_ng})")
                            # ⚠ 절대값 해석 기준 (합성 대조, n=120, D=16, k=8):
                            #     항등 사상 + 유클리드 측정   Spearman 1.000 / Recall 1.000
                            #     항등 사상 + **코사인** 측정  Spearman 0.767 / Recall 0.639
                            #     무관(랜덤)                  Spearman 0.016 / Recall 0.072
                            #   TabERA의 라우팅·검색이 코사인이므로 여기서도 코사인을
                            #   쓰지만, 코사인은 norm 정보를 버리므로 **상한이 1.0이 아니다.**
                            #   0.77 근처면 사실상 완전 보존, 0.1 이하면 붕괴로 읽는다.
                            #   조건 간 **상대 비교**(concat vs proto_only)가 주 용도다.
                except Exception as _le:
                    _rb_savez_kwargs["local_ordering_error"] = np.array(
                        [f"{type(_le).__name__}: {_le}"], dtype=object)

                _d_tr, _m_tr, _e_tr, _a_tr = _geo(_trQ)
                _d_te, _m_te, _e_te, _a_te = _geo(_teQ)
                _rb_savez_kwargs.update(
                    # ① assigned centroid distance — test가 centroid manifold 밖으로 갔는가
                    train_centroid_dist=_d_tr,   test_centroid_dist=_d_te,
                    # ② second-best margin — confidence가 "1등이 가까워서"인지
                    #    "나머지가 다 멀어서"인지 구분
                    train_routing_margin=_m_tr,  test_routing_margin=_m_te,
                    # ③ routing entropy (softmax(sim·routing_scale) 기준)
                    train_routing_entropy=_e_tr, test_routing_entropy=_e_te,
                    # ④ 임베딩 분포 shift — 평균 이동과 분산 비
                    #    (D차원 전체를 저장하지 않고 요약 통계만)
                    train_query_mean=_trQ.mean(0).cpu().numpy(),
                    test_query_mean=_teQ.mean(0).cpu().numpy(),
                    train_query_std=_trQ.std(0).cpu().numpy(),
                    test_query_std=_teQ.std(0).cpu().numpy(),
                    # ⑤ centroid 배치 자체 (P, D) — 사후 분석용
                    centroid_emb=_cN.cpu().numpy(),
                )
                print(f"  [routing geometry] train dist={_d_tr.mean():.4f} test dist={_d_te.mean():.4f}"
                      f" | margin train={_m_tr.mean():.4f} test={_m_te.mean():.4f}"
                      f" | mean shift={float((_teQ.mean(0)-_trQ.mean(0)).norm()):.4f}")
                # ── 자기검증: 여기서 계산한 entropy가 모델이 내놓은
                # routing_confidence와 산술적으로 양립하는가.
                # top1 확률 p가 주어지면 entropy 상한은
                #   H_max = -(p·log p + (1-p)·log((1-p)/(P-1)))
                # 이다. 관측 entropy가 이 상한을 넘으면 **다른 공간에서
                # 계산했다는 뜻**이다(과거 query_retr를 잘못 쓴 사례:
                # top1=0.556인데 entropy 3.911 — 상한 2.53을 초과).
                _p1 = float(np.mean(_rb_savez_kwargs["routing_confidence"]))
                _P = int(_cN.shape[0])
                if 0.0 < _p1 < 1.0 and _P > 1:
                    _hmax = -(_p1 * np.log(_p1) + (1 - _p1) * np.log((1 - _p1) / (_P - 1)))
                    if float(_e_te.mean()) > _hmax + 1e-6:
                        print(f"  ⚠️  [routing geometry] 자기검증 실패 — test entropy "
                              f"{_e_te.mean():.3f} > 상한 {_hmax:.3f} (routing_confidence={_p1:.3f}). "
                              f"라우팅과 다른 공간에서 계산됐을 가능성.")
                        _rb_savez_kwargs["routing_geometry_inconsistent"] = np.array([1])
        except Exception as _qe:
            _rb_savez_kwargs["routing_geometry_error"] = np.array(
                [f"{type(_qe).__name__}: {_qe}"], dtype=object)

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
            _rb_savez_kwargs["neighbor_labels"] = torch.cat(_rb_neighbor_label_chunks, dim=0).numpy()
            _rb_savez_kwargs["neighbor_sample_ids"] = torch.cat(_rb_neighbor_sid_chunks, dim=0).numpy()
            # [2026-07] 각 test 샘플이 배정된 centroid of the group train 크기.
            # 재현성 Jaccard의 해석에 필수 — 그룹이 작으면 서로 다른 두 모델이
            # 우연히 같은 이웃을 고를 확률이 높아진다. 무작위 null을
            # E[J] = (k²/G) / (2k − k²/G) 로 계산하려면 G가 있어야 한다.
            # [2026-07] train 샘플의 그룹 배정 (N_train,) — 값은 centroid id.
            # [왜] seed 간 설명 재현성을 잴 때, ①(그룹 배정)과 ②(그룹 안 이웃
            # 선택)의 불안정성이 섞인다. seed마다 test 샘플이 다른 그룹에 가면
            # 후보 풀 자체가 달라져 ② Jaccard가 "같은 풀" 가정의 NULL보다도
            # 낮게 나온다(실측: 1489 0.71배 / 46 0.46배 / 1493 0.52배).
            # 두 seed의 후보 풀 겹침을 직접 계산해야 ②를 분리할 수 있다.
            # centroid id 자체는 seed 간 임의 번호라 비교 불가 — 풀을 **집합**
            # 으로 놓고 겹침을 봐야 한다.
            try:
                _sg0 = model.prototype_layer.sample_groups
                if _sg0:
                    _ta = np.full(int(model.memory.filled.item()), -1, dtype=np.int64)
                    _sid = model.memory.sample_ids.cpu().numpy()
                    for _p, _mem in enumerate(_sg0):
                        for _m in _mem:
                            if 0 <= _m < len(_sid) and _sid[_m] >= 0:
                                _ta[int(_sid[_m])] = _p
                    _rb_savez_kwargs["train_assignment"] = _ta
                    # ⚠ train 라벨도 함께 저장한다. 없으면 centroid purity/Voronoi 상한을
                    #   neighbor_labels로 역추정해야 하는데, 그건 **검색된 이웃만** 담고
                    #   있어 train의 절반 정도만 복원된다(실측 ds=54: 676개 중 360개, 53%).
                    #   그 표본으로 다수결을 추정하면 상한이 실제 accuracy보다 낮게 나오는
                    #   불가능한 결과가 나온다.
                    try:
                        _rb_savez_kwargs["y_train"] = (
                            y_train.detach().cpu().numpy() if torch.is_tensor(y_train)
                            else np.asarray(y_train))
                    except Exception:
                        pass
            except Exception as _te:
                _rb_savez_kwargs["train_assignment_error"] = np.array(
                    [f"{type(_te).__name__}: {_te}"], dtype=object)
            try:
                _sg = model.prototype_layer.sample_groups
                _gsz = np.array([len(_sg[int(c)]) if _sg and int(c) < len(_sg) else -1
                                 for c in _rb_savez_kwargs["centroid_id"]], dtype=np.int64)
                _rb_savez_kwargs["group_size"] = _gsz
            except Exception as _ge:
                _rb_savez_kwargs["group_size_error"] = np.array(
                    [f"{type(_ge).__name__}: {_ge}"], dtype=object)
            # [2026-07] 정합성 지표: 검색된 이웃이 실제로 자기 그룹 안에 있는가.
            # [왜] group-constrained 검색이 정말 걸렸는지는 값만 봐서는 알 수 없다.
            # 실제로 --from_saved_state가 memory._cached_groups를 복원하지 않아
            # 전역 검색으로 동작하던 버그가 있었고(2026-07 수정), purity/margin/
            # n_eff 계열 진단이 전부 잘못된 전제 위에 있었다. 이 비율이 1.0에서
            # 크게 벗어나면 그룹 제약이 안 걸린 것이므로 즉시 드러난다.
            # (그룹 크기 < k 인 샘플은 설계상 cross-group/전역 폴백이므로
            #  1.0 미만이 정상일 수 있다 — group_size와 함께 읽을 것.)
            try:
                _nsi = _rb_savez_kwargs["neighbor_sample_ids"]
                _ta_chk = _rb_savez_kwargs.get("train_assignment")
                _cid_chk = _rb_savez_kwargs["centroid_id"]
                if _ta_chk is not None:
                    _ratio = np.array([
                        float(np.mean(_ta_chk[_nsi[i][_nsi[i] >= 0]] == _cid_chk[i]))
                        if (_nsi[i] >= 0).any() else np.nan
                        for i in range(len(_cid_chk))])
                    _rb_savez_kwargs["neighbor_in_group_ratio"] = _ratio
                    _m = float(np.nanmean(_ratio))
                    print(f"  [정합성] 검색 이웃이 자기 그룹 안에 있는 비율 = {_m:.3f}")
                    if _m < 0.9:
                        print(f"    ⚠️  1.0에서 크게 벗어남 — group-constrained 검색이 "
                              f"제대로 걸리지 않았을 수 있습니다.\n"
                              f"       (a) memory._cached_groups 복원 여부, "
                              f"(b) 그룹 크기 < k 로 인한 폴백을 확인하세요.")
            except Exception as _ie:
                _rb_savez_kwargs["in_group_ratio_error"] = np.array(
                    [f"{type(_ie).__name__}: {_ie}"], dtype=object)
            _rb_savez_kwargs["retrieval_label_purity"] = torch.cat(_rb_purity_chunks, dim=0).numpy()
            # ⚠ proto_dev 계열에서는 evidence_w가 균등 상수라 이 값이
            #   retrieval_label_purity와 **완전히 동일**해진다(실측 max diff 0).
            #   두 컬럼이 같으면 aggregator가 비활성이라는 신호다.
            _rb_savez_kwargs["retrieval_weighted_label_purity"] = torch.cat(_rb_weighted_purity_chunks, dim=0).numpy()
            # [추가] H(Y_N(x)) — 검색된 이웃 **라벨 분포**의 entropy.
            # ⚠ `entropy` 컬럼(evidence_w 기반)과 다른 값이다. retrieval
            #   uncertainty를 보려면 이쪽을 쓸 것 — 위 계산 지점 주석 참고.
            if _rb_nbr_label_entropy_chunks:
                _rb_savez_kwargs["neighbor_label_entropy"] = torch.cat(
                    _rb_nbr_label_entropy_chunks, dim=0).numpy()
                _rb_savez_kwargs["neighbor_label_n_eff"] = torch.cat(
                    _rb_nbr_label_neff_chunks, dim=0).numpy()
        if _rb_sim_top1_chunks:
            _rb_savez_kwargs["similarity_top1"] = torch.cat(_rb_sim_top1_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_bottomk"] = torch.cat(_rb_sim_bottomk_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_margin"] = torch.cat(_rb_sim_margin_chunks, dim=0).numpy()
            _rb_savez_kwargs["similarity_std"] = torch.cat(_rb_sim_std_chunks, dim=0).numpy()
        if _rb_val_pwcos_chunks:
            # value_pairwise_cos: 1에 가까우면 k개 이웃의 value가 사실상 같은 벡터
            #   → evidence_w를 아무리 sharpen해도 Σw·v가 안 바뀜(value collapse).
            # label/offset을 따로 봐야 (a)그룹이 라벨 순수해서인지 (b)T()가 이웃을
            # 구분 못해서인지 (c)둘 다인지 분리됨.
            _rb_savez_kwargs["value_pairwise_cos"] = torch.cat(_rb_val_pwcos_chunks, dim=0).numpy()
            _rb_savez_kwargs["label_pairwise_cos"] = torch.cat(_rb_lbl_pwcos_chunks, dim=0).numpy()
            if _rb_off_pwcos_chunks:
                _rb_savez_kwargs["offset_pairwise_cos"] = torch.cat(_rb_off_pwcos_chunks, dim=0).numpy()
            # rel_dispersion: pairwise cosine이 **방향만** 보는 것을 보완 —
            # mean_i‖v_i-v̄‖/‖v̄‖로 크기 차이까지 포함한 스케일 무관 퍼짐.
            # cos이 1에 가까워도 dispersion이 크면 이웃별 차이가 남아 있는 것.
            _rb_savez_kwargs["value_rel_dispersion"] = torch.cat(_rb_val_disp_chunks, dim=0).numpy()
            _rb_savez_kwargs["label_rel_dispersion"] = torch.cat(_rb_lbl_disp_chunks, dim=0).numpy()
            if _rb_off_disp_chunks:
                _rb_savez_kwargs["offset_rel_dispersion"] = torch.cat(_rb_off_disp_chunks, dim=0).numpy()
        if _rb_agg_emb_chunks:
            _rb_savez_kwargs["agg_emb"] = torch.cat(_rb_agg_emb_chunks, dim=0).numpy()
            # [2026-07, 추가] agg_emb의 between-group / within-group 분산 분해.
            # [무엇을 재는가] agg_emb가 "샘플마다 다른 표현"인지 "그룹마다 고정된
            # 조회값"인지를 직접 구분한다. evidence_w가 사실상 균등하고(n_eff/k≈1)
            # 검색이 centroid 그룹 내로 제한되면, agg는 of the group 평균에 수렴해
            # context_emb(centroid 그대로 읽기)와 같은 층위의 정보가 된다.
            #   ratio → 1 : 그룹 조회에 가까움 (같은 그룹이면 거의 같은 값)
            #   ratio → 0 : 샘플별로 실제로 다른 값
            # n_eff/purity/agg_from_query R²와는 다른 축이라 같이 보면 진단이
            # 선명해진다. 그룹이 1개뿐이면 between이 정의되지 않아 NaN.
            _rb_A = _rb_savez_kwargs["agg_emb"]
            _rb_g = _rb_centroid_ids_all
            _rb_mu = _rb_A.mean(axis=0, keepdims=True)
            _rb_tot = float(((_rb_A - _rb_mu) ** 2).sum())
            _rb_btw = 0.0
            for _c in np.unique(_rb_g):
                _m = (_rb_g == _c)
                _rb_btw += float(_m.sum()) * float(((_rb_A[_m].mean(axis=0) - _rb_mu[0]) ** 2).sum())
            _rb_savez_kwargs["agg_between_var"] = np.float64(_rb_btw)
            _rb_savez_kwargs["agg_within_var"]  = np.float64(_rb_tot - _rb_btw)
            _rb_savez_kwargs["agg_between_ratio"] = np.float64(
                _rb_btw / _rb_tot if (_rb_tot > 0 and len(np.unique(_rb_g)) > 1) else np.nan)
            print(f"  [agg variance] between/total = {_rb_savez_kwargs['agg_between_ratio']:.3f}  "
                  f"(1에 가까우면 agg가 그룹 조회에 가까움 = context_emb와 같은 역할, "
                  f"groups={len(np.unique(_rb_g))})")
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
        print(f"  [export_centroid_retrieval_behavior] sample_id/centroid_id/routing_confidence/topk_idx/neighbor_labels/neighbor_sample_ids/group_size/entropy(=evidence_w)/n_eff/top1_weight/neighbor_label_entropy/neighbor_label_n_eff"
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

    # ── [v3] test 시점 Δ (c1만 vs c1+c2) ────────────────────────────
    # forward 진단은 train 배치 기준이라 **일반화 기여를 못 본다.**
    # train에서 Δacc < 0인데 test 성능이 오르면 "c2가 정보를 더한 것"이
    # 아니라 "노이즈로 과적합을 줄인 것"일 수 있다 — 둘은 다른 결론이다.
    _rvq_test = {}
    if _rvq_logits_c1_chunks:
        _lg1_test = torch.cat(_rvq_logits_c1_chunks, dim=0).numpy()
        with torch.no_grad():
            # ⚠ y_test가 CUDA면 device가 어긋난다 — 같은 디바이스로 맞춘다.
            _lg1_t = torch.from_numpy(_lg1_test)
            if torch.is_tensor(y_test):
                _lg1_t = _lg1_t.to(y_test.device)
            _p1, _pr1 = get_preds_and_probs(_lg1_t, tasktype)
        # ⚠ 회귀는 y_std로 역스케일 후 계산해야 test_metrics와 단위가 맞는다.
        if tasktype == "regression":
            _m1 = calculate_metric(y_test * y_std, _p1 * y_std, None, tasktype, "test")
        else:
            _m1 = calculate_metric(y_test, _p1, _pr1, tasktype, "test")
        for _k, _v in _m1.items():
            _rvq_test[f"c1only_{_k}"] = _v
            if _k in test_metrics:
                _rvq_test[f"delta_{_k}"] = test_metrics[_k] - _v
        # ⚠ changed_rate와 함께 봐야 한다:
        #   changed 낮고 Δ≈0  → head가 c2를 거의 안 씀
        #   changed 높고 Δ≈0  → c2를 쓰지만 방향이 틀림
        # ⚠ preds_test는 CUDA 텐서일 수 있다 — numpy 변환 전에 cpu()를 거친다.
        def _to_np(_t):
            return (_t.detach().cpu().numpy() if torch.is_tensor(_t)
                    else np.asarray(_t))
        _rvq_test["changed_rate"] = float(
            (_to_np(preds_test).ravel() != _to_np(_p1).ravel()).mean())
        _rvq_test["unique_pred_c1"] = float(
            len(np.unique(np.round(_lg1_test, 5), axis=0)))
        _rvq_test["unique_pred_c1c2"] = float(
            len(np.unique(np.round(logits, 5), axis=0)))
        print(f"  [v3 test Δ] acc {_m1.get('acc_test', float('nan')):.4f} → "
              f"{test_metrics.get('acc_test', float('nan')):.4f}  "
              f"(Δ{_rvq_test.get('delta_acc_test', float('nan')):+.4f}, "
              f"changed {_rvq_test['changed_rate']:.3f})")

    meta = {
        "openml_id":   openml_id,
        "tasktype":    tasktype,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "rvq_test_delta": _rvq_test,
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
        # ── [축 2] prototype behavior ────────────────────────────────
        # 논문 주장 "prototype은 class prototype이 아니라 density-driven
        # anchor이며 granularity가 task 복잡도에 맞춰진다"를 지지하는 지표.
        # ⚠ 지금까지는 체크포인트에서 사후 계산했는데, 그러면 결과 표를
        #   만들 때마다 전 체크포인트를 다시 열어야 하고 계산 코드가
        #   바뀌면 예전 결과와 어긋난다. 학습 시점에 같이 남긴다.
        "prototype_alignment": diag.prototype_class_alignment(model),
        "context_diversity":   diag.context_space_diversity(model),
        # ⚠ --log_beta 로 기록한 dev_beta_raw 궤적. 켜지 않았으면 빈 리스트다.
        #   콘솔 출력만으로는 나중에 재분석할 수 없어 meta에도 남긴다.
        "beta_history": getattr(wrapper, "beta_history", []),
        "beta_lr_mult": getattr(wrapper, "beta_lr_mult", 1.0),
        # ── [config freeze] 이 run 이 v3 기본 설정에서 벗어났는가 ──────
        # ⚠ freeze_check.py 는 **코드의 기본값**만 검사한다. 실행할 때
        #   플래그로 덮어쓰면 못 잡는다. 결과 파일 자체에 "무엇이 기본과
        #   달랐는지"를 남겨야 나중에 표를 만들 때 조건을 특정할 수 있다.
        #   이번 세션에서 P=35 study가 P=100 실행에 덮여 사라진 적이 있다.
        "freeze_deviations": {
            k: v for k, v in {
                "n_prototypes":  args.n_prototypes,
                "beta_lr_mult":  args.beta_lr_mult,
                # A/O 실험 조건 — meta.pkl 만 보고 어느 조건이었는지 알 수 있어야 한다
                "gradient_codebook":   args.gradient_codebook,
                "commitment":          args.commitment,
                "disable_dead_reinit": args.disable_dead_reinit,
                "epochs":        args.epochs,
                "patience":      args.patience,
                "nbr_lambda":    args.nbr_lambda,
                "num_bins":      args.num_bins,
                "cat_combine":   args.cat_combine,
                "num_embedding": args.num_embedding,
                "evidence_metric": args.evidence_metric,
                "fusion_mode":   args.fusion_mode,
            }.items()
            if v != {"n_prototypes": None, "beta_lr_mult": 1.0,
                     "gradient_codebook": False, "commitment": False,
                     "disable_dead_reinit": False,
                     "epochs": HPO_TRAINING_SCHEDULE["epochs"],
                     "patience": HPO_TRAINING_SCHEDULE["patience"],
                     # ⚠ 기본값 전환(2026-08)에 맞춰 0.005 → 0.0.
                     #   전환 후 `--nbr_lambda 0` 은 기본이므로 이탈이 아니다.
                     "nbr_lambda": 0.0, "num_bins": 8,
                     "cat_combine": "onehot", "num_embedding": "ple",
                     "evidence_metric": "cosine", "fusion_mode": "proto_dev"}[k]
        },
        # --time_epoch 계측도 같은 이유로 저장한다.
        "epoch_timing": getattr(wrapper, "_timing", {}),
        "evidence_stats_history": wrapper.evidence_stats_history,
        # ── optimizer update budget ────────────────────────────────
        # ⚠ epoch 수만으로는 데이터셋 간 학습량을 비교할 수 없다.
        #   batch=256 고정이라 steps/epoch 이 N 에 비례해 2~16 으로 갈린다
        #   (ds=54 는 epoch 당 2회, ds=1489 는 16회 — 8배 차이).
        #   early stopping 이 부분적으로 흡수하지만 통제되지는 않으므로
        #   실제 update 횟수를 남긴다. batch_size 를 다시 검토할 때 필요하다.
        "steps_per_epoch": (
            (len(X_train) // best_params["batch_size"])
            if best_params.get("batch_size") else None),
        "deterministic": args.deterministic,
        "deterministic_warn_only": args.deterministic_warn_only if args.deterministic else None,
        "use_offset_correction": True,
        "retr_proj_mode": args.retr_proj_mode,
        "detach_retr_grad": args.detach_retr_grad,
        "global_retrieve": args.global_retrieve,
        "use_context_emb": not args.no_context_emb,
        "use_query_emb_in_head": not args.no_query_emb,
        "use_ema_codebook": not args.gradient_codebook,
        "ema_decay": (args.ema_decay_override if args.ema_decay_override is not None else 0.99) if not args.gradient_codebook else None,
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
        if args.verbose:
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
                # pred_code: supporting/contrasting 분리 기준. 라벨 문자열만
                # 있으면 이웃 라벨(정수 코드)과 비교할 수 없다.
                pred_infos.append({"pred_label": label, "pred_confidence": conf,
                                    "pred_code": idx})

        # 설명 재료는 모델이 아니라 관찰자 계층에서 만든다.
        # forward는 예측 상태(logits/topk_idx/context_emb/query_retr/
        # neighbor_mask)만 내보내고, 아래 네 함수가 그걸 관찰해서 재구성한다.
        # 동일성은 verify_equivalence.py로 확인함(최대 오차 3e-08).
        _nbrs = diag.retrieved_neighbors(model, out)
        _le   = diag.local_label_evidence(model, out)
        _pdv  = diag.prototype_deviation(model, out)
        _gst  = diag.group_relative_feature_stats(model, out, X_show)

        cat_names = {dataset.col_names[i] for i in dataset.X_cat}
        X_show_cpu = X_show.detach().cpu().numpy()
        for b, exp in enumerate(explanations):
            query_dict = {name: float(X_show_cpu[b, i])
                          for i, name in enumerate(dataset.col_names)}
            exp["neighbors"]           = (_nbrs[b] if _nbrs else [])
            exp["local_evidence"]      = (_le[b]   if _le   else None)
            exp["prototype_deviation"] = (_pdv[b]  if _pdv  else None)
            exp["group_stats"]         = (_gst[b]  if _gst  else None)
            for nb in exp["neighbors"]:
                if nb.get("features"):
                    # 전체 gap을 붙인다 — 정렬/절단은 표시 계층에서.
                    nb["gaps"] = diag.feature_gaps(
                        query_dict, nb["features"], cat_names)
        if not explanations:
            print("  (no explanations — memory bank has not been filled yet)")
            print("  → try increasing epochs or n_trials.")
        else:
            for i in range(n_show):
                print_explanation(explanations, i, dataset.col_names,
                                   cat_category_names=dataset.cat_category_names,
                                   quantile_transformer=dataset.quantile_transformer,
                                   num_cols=list(dataset.X_num),
                                   pred_info=pred_infos[i],
                                   target_class_names=getattr(
                                       dataset, "target_class_names", None),
                                   tasktype=tasktype,
                                   verbose=getattr(args, "explain_verbose", False))

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
    parser.add_argument("--n_prototypes", type=int, default=None,
                        help=("optimize.py에서 --n_prototypes 로 만든 study를 찾을 때 "
                              "같은 값을 지정하세요. study 파일명에 ..P{n}이 붙어 "
                              "있습니다. --from_saved_state 를 쓰면 불필요합니다."))
    parser.add_argument("--beta_lr_mult", type=float, default=1.0,
                        help=("dev_beta_raw 전용 학습률 배수 (기본 1.0 = 기존 동작). "
                              "β가 학습 내내 단조 상승하다 early stopping에서 잘리는 "
                              "것이 실측돼, 균형점이 존재하는지 확인하기 위한 실험용 "
                              "플래그입니다. (0,1) 경계는 유지됩니다."))
    parser.add_argument("--log_beta", action="store_true",
                        help=("proto_dev 계열의 dev_beta_raw 값과 그래디언트를 "
                              "epoch별로 기록해 학습 종료 후 출력합니다. "
                              "β가 초기값 근처에 머무는 것이 '균형'인지 '정체'인지 "
                              "가릅니다. 배치마다 동기화가 생기므로 기본은 꺼짐."))
    parser.add_argument("--time_epoch", action="store_true",
                        help=("epoch 구간별(regroup_update / cache_sample_groups / "
                              "feature_store 전송 / label 계산) 누적 소요 시간을 "
                              "학습 종료 후 표로 출력합니다. 켜면 정확한 측정을 위해 "
                              "CUDA sync가 들어가므로 기본은 꺼져 있습니다."))
    parser.add_argument("--verbose", action="store_true",
                        help=(
                            "Also print the lines that describe how a run got "
                            "where it did rather than what it produced: which "
                            "study was loaded, memory refresh, and centroid "
                            "geometry. Training progress is controlled "
                            "separately by --regroup_log_every."))
    parser.add_argument("--explain_verbose", action="store_true",
                        help=("설명 출력에 해석 규칙 주석을 함께 표시합니다. "
                              "기본값(꺼짐)에서는 샘플마다 반복되는 고정 문구 "
                              "(② 경고 5줄, 그룹 대비 주석 2줄, ③ 분해 주석 1줄)를 "
                              "생략합니다 — 규칙은 한 번만 읽으면 되는 내용이고, "
                              "샘플 14건이면 그것만 100줄이 넘습니다."))
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
    parser.add_argument("--global_retrieve", action="store_true",
                        help=(
                            "[진단/실험] retrieve()에서 centroid 그룹 제약을 끄고 "
                            "전체 memory bank에서 전역 KNN 검색. context_emb(설명①)는 "
                            "그대로 유지되고 evidence_w/agg_emb(설명②)만 영향받음. "
                            "optimize.py에 이미 있던 플래그인데 reproduce.py에는 없어서 "
                            "S1 실험이 실행 불가였음(2026-07 추가). "
                            "⚠ 반드시 `optimize.py --global_retrieve`로 따로 HPO한 study가 "
                            "있어야 함 — 이 플래그는 study 파일명 태그(..global_retrieve)와 "
                            "출력 파일명 태그에 모두 반영되므로, 없으면 FileNotFoundError로 "
                            "즉시 드러난다(조용히 baseline study를 읽는 사고 방지)."))
    # ── [2026-07, S-1A] retrieval 전용 표현 분기 ──────────────────
    parser.add_argument("--branch_info_shuffles", type=int, default=5,
                        help="information gain의 null 대조 shuffle 횟수 "
                             "(0=null 계산 안 함/기존 동작, 3=빠른 디버깅, "
                             "5=논문 기본, 10=분산 확인)")
    parser.add_argument("--branch_info_residual_null", action="store_true",
                        help="query 잔차 셔플 null을 추가로 계산(R-3 보조 진단). "
                             "기본 결과는 plain shuffle null을 사용.")
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
                                 "rank_correlation", "centroid_geometry",
                                 "centroid_representativeness", "dataset_profile"],
                        help=(
                            "ablation 모드 (학습된 모델에 inference 단계에서 적용):\n"
                            "  none                        : full model 기준 (기본값)\n"
                            "  random_neighbor             : 같은 centroid 대신 무작위 이웃 —\n"
                            "                                'prototype grouping이 의미 있는\n"
                            "                                 검색 제한인가'\n"
                            "  neighbor_noise              : 이웃 임베딩을 노이즈로 대체 —\n"
                            "                                'neighbor identity가 중요한가'\n"
                            "  rank_correlation            : 검색 순위와 각종 기준의 상관 —\n"
                            "                                Level 1(local ordering)과 직결\n"
                            "  centroid_geometry           : prototype space가 어떻게 형성되는가\n"
                            "  centroid_representativeness : prototype이 실제 data manifold를\n"
                            "                                대표하는가 (설명 가능성)\n"
                            "  dataset_profile             : 모델이 아니라 데이터 분석 도구\n"
                            "\n"
                            "[v2에서 제거된 모드] agg_emb_zero/shuffle/constant/centered,\n"
                            "  context_emb_zero/shuffle, query_emb_zero/shuffle,\n"
                            "  dual_space_faithfulness, interaction_check,\n"
                            "  evidence_compensation — 전부 head([query, context, agg])\n"
                            "  구조를 검증하던 실험이다. v2는 head(context) 하나이므로\n"
                            "  실행 자체가 무의미하다. 과거 결과는 git 이력 참조."
                        ))
    parser.add_argument("--use_context_emb", action="store_true",
                        help=(
                            "[2026-07, deprecated — 하위호환용] use_context_emb=True가 다시 "
                            "기본값(v1 복원)이라 이 플래그는 더 이상 아무 효과가 없음(줘도 "
                            "안전 — 어차피 기본 동작). context_emb를 head에서 빼려면(v2식) "
                            "--no_context_emb를 쓸 것."
                        ))
    parser.add_argument("--ema_decay_override", type=float, default=None,
                        help=(
                            "[통제 실험용] EMA prototype memory 의 decay(문헌 기본값 0.99 — "
                            "van den Oord et al. 2017 Appendix, VQ-VAE-2/Jukebox/SoundStream "
                            "공통. 이 프로젝트 데이터로 검증된 값 아님, 스윕 대상). "
                            "--gradient_codebook 과 함께 쓰면 무효과."
                        ))
    parser.add_argument("--value_mode", type=str, default="default",
                        choices=["default", "label_only"],
                        help=(
                            "[ablation] AttentionAggregator value 구성. 이 인자는 "
                            "이제 use_offset_correction 하나만 결정한다 — "
                            "'label_only' 면 T(query-neighbour) 항을 빼고 "
                            "value=label_emb 만 쓴다. "
                            "⚠ 나머지 변형(offset_only/balanced/interaction 등)은 "
                            "제거했다: aggregator 를 만들지 않는 fusion_mode 에서는 "
                            "value 구성 자체가 존재하지 않고, 최종 모델(proto_dev)이 "
                            "여기에 해당한다."))
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
    parser.add_argument("--fusion_mode", type=str, default="proto_dev",
                        choices=["concat", "residual", "gated_sum", "anchor_gate", "context_gated_beta",
                                 "proto_residual", "proto_query_residual", "proto_only",
                                 "proto_only_linear", "query_only_linear", "proto_dev", "proto_dev_vec", "proto_dev_agg",
                                 "proto_residual_query", "proto_dev_retr"],
                        help=(
                            "[2026-07, 되돌림] 'residual'을 잠시 기본값으로 뒀었으나, 이후 "
                            "6개 데이터셋에 걸친 폭넓은 비교(ablation/trajectory/retrieval-free "
                            "baseline)에서도 concat 대비 일관된 우위를 못 찾아 기본값을 원래대로 "
                            "('concat', main 브랜치와 동일) 되돌림 — 'v2가 확정된 게 아니었다'는 "
                            "뜻이지 틀렸다는 뜻은 아님, residual은 계속 --fusion_mode residual로 "
                            "명시적으로 선택 가능(비교/ablation 목적). "
                            "head가 [query,context,agg]를 합치는 방식. "
                            "'concat'(기본값): [query‖context‖agg] → 공유 MLP. "
                            "'residual': z = LN(q) + α·LN(c) + β·LN(a) (α,β 학습 가능한 "
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
                            "--evidence_metric_override(아래)와 differs: 이건 '그 metric으로 "
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
    parser.add_argument("--learn_evidence_temperature", action="store_true",
                        help=(
                            "[2026-07, 신규] evidence softmax의 온도 T를 고정값(1.0) 대신 "
                            "학습 파라미터로 둔다(softplus로 양수 보장, 초기값은 기존 T와 동일). "
                            "배경: T=1.0은 HPO 탐색 대상도 아니어서 한 번도 조정된 적이 없었고, "
                            "관측된 similarity margin(0.03~0.68)에서는 산술적으로 거의 균등분포가 "
                            "된다(n_eff/k≈1, 10/10). 또 최적 sharpness가 k에 따라 달라지는데"
                            "(k=4에선 sharp가 +1.4%%p 우세, k=48에선 -14.1%%p) k는 HPO가 정하고 "
                            "T는 고정이라 둘이 맞을 이유가 없었다. 학습된 T 값 자체가 진단 "
                            "지표가 된다 — 1.0 근처에 머물면 '구분할 것이 없어서', 내려가면 "
                            "'선별을 시작'으로 읽는다."
                        ))
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
    parser.add_argument("--inference_evidence_temperature", type=float, default=None,
                        help=(
                            "[2026-07, 신규 — 통제 실험용] --from_saved_state로 로드한 "
                            "모델의 **추론 시점 evidence softmax 온도만** 이 값으로 바꿈"
                            "(가중치 재학습 없음). --evidence_temperature_override와의 "
                            "차이가 핵심: 그쪽은 재학습 경로라 encoder까지 다시 학습되므로, "
                            "agg가 변해도 'attention이 sharpen돼서'인지 '모델이 다르게 "
                            "학습돼서'인지 분리가 안 된다. 이 플래그는 동일 가중치에서 T만 "
                            "바꾸므로 --export_centroid_retrieval_behavior의 agg_emb를 "
                            "baseline(T 그대로)과 비교하면 cos(agg_before, agg_after)가 "
                            "순수하게 aggregation 효과만 반영한다. "
                            "권장 사용: 같은 체크포인트에 T=기본값/0.3/0.1/0.03을 각각 적용해 "
                            "(n_eff/k, cos(agg_before,agg_after), agg_from_query R²)를 비교 — "
                            "n_eff/k는 내려가는데 agg cosine이 1에 가까우면 'attention은 "
                            "변했는데 aggregation 결과는 그대로'이므로 병목이 attention이 "
                            "아니라는 뜻. --from_saved_state 없이 쓰면 무시됨(재학습 경로에서는 "
                            "--evidence_temperature_override를 쓸 것)."
                        ))
    parser.add_argument("--k_override", type=int, default=None,
                        help=(
                            "[통제 실험용] best_params의 k(검색 이웃 수)만 이 값으로 "
                            "덮어쓰고 나머지는 그대로 재학습. --evidence_temperature_override"
                            "와 짝으로 쓰는 것이 주 용도 — sharp attention이 성능을 "
                            "떨어뜨릴 때 그 원인이 (a) sharpening 자체인지 (b) k가 큰 채로 "
                            "소수 이웃만 쓰게 되어 추정 분산이 커진 것인지 분리한다"
                            "(k=48에서 n_eff≈1.4면 48개 평균 대비 분산이 30배 이상). "
                            "가중치 shape에 영향을 주므로 --from_saved_state와 같이 쓰면 "
                            "로드가 깨진다 — 재학습 경로에서만 쓸 것."
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
    parser.add_argument("--residual_vq", action="store_true",
                        help=(
                            "[v3] 2단계 residual 양자화. context = c1 + c2 이고 "
                            "c2는 잔차 r = q - sg(c1)을 euclidean으로 양자화한 코드다. "
                            "⚠ v2는 f(x) = W·c_k 라 최대 P개의 서로 다른 예측만 "
                            "가능하다(ds=14: test 200개 → 고유 예측 31개). Voronoi "
                            "accuracy 상한 0.852의 97.3%%에 이미 도달했고 auroc 손실은 "
                            "동점이 114%% 설명한다 — 병목은 학습이 아니라 표현력이다. "
                            "⚠ continuous residual(E 조건)로 풀면 안 된다: lambda를 "
                            "작게 둬도 optimizer가 ||W_q||를 24배 키워 우회하고 "
                            "prototype이 무력화된다(query 기여 98.6%%). 두 항 모두 "
                            "discrete여야 무제한 정보 경쟁자가 없다. "
                            "판정: rvq_H_c2_given_c1 > 0 (게이트), rvq_delta_acc, "
                            "rvq_unique_pred_c12 증가, auroc 회복."
                        ))
    parser.add_argument("--residual_vq_size", type=int, default=None,
                        help="stage2 코드북 크기 P2. 미지정이면 P1과 동일(√N).")
    parser.add_argument("--commitment", action="store_true",
                        help=("[ablation] commitment_loss 를 켠다(기본 꺼짐, §10). "
                              "⚠ optimize.py 와 반드시 같아야 한다."))
    parser.add_argument("--gradient_codebook", action="store_true",
                        help=("centroid 를 gradient 로 학습합니다(v3 이전 기본값). "
                              "지금은 EMA 가 기본입니다."))
    parser.add_argument("--nbr_lambda", type=float, default=0.0,
                        help=(
                            "[v3] L_nbr 가중치 — raw feature 이웃 구조를 encoder에 "
                            "보존시키는 contrastive loss. positive는 raw kNN(prototype "
                            "무관), negative는 raw 거리가 먼 샘플만 쓴다. "
                            "⚠ centroid/assignment를 전혀 쓰지 않는다 — encoder만 "
                            "규제해야 'representation을 바꾼 효과'가 분리된다. "
                            "⚠ snn(제거됨)과 다르다: snn은 positive가 같은 라벨이라 "
                            "CE와 같은 방향이었고, L_nbr은 raw 이웃이라 직교한다. "
                            "sweep 권장: 0 / 0.001 / 0.005 / 0.01 / 0.05 / 0.1 "
                            "(InfoNCE는 log(batch)≈3~5 스케일이라 CE보다 크다). "
                            "판정: kNN recall·Level 1 상승 + accuracy 유지."
                        ))
    parser.add_argument("--disable_dead_reinit", action="store_true",
                        help=(
                            "[통제 실험용, Phase 1] dead_reinit을 완전히 끈다. "
                            "patience_override를 크게 주는 것과 달리 재초기화 이벤트가 "
                            "아예 발생하지 않는다. "
                            "⚠ 질문이 바뀌었다 — 'collapse를 막는가'가 아니라 "
                            "'CE가 못 벗어나는 local minimum을 escape시키는가'다. "
                            "proto_only_linear에서는 CE 자체가 collapse 압력을 만든다"
                            "(같은 centroid의 모든 샘플이 같은 예측을 받으므로). "
                            "실측: ds=1489에서 concat 활성 6/65 → proto_only 52/65."
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
    parser.add_argument("--refresh_on_best", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "[v2 기본값 True — 끄려면 --no-refresh_on_best] "
                            "⚠ 이 플래그를 끄면 retrieval 관련 진단이 **전부 무효**가 된다. "
                            "학습 중 저장된 memory.keys는 dropout이 걸린 옛 시점 임베딩이라 "
                            "최종 가중치로 인코딩한 test query와 다른 함수다. 실측 그룹 구조 "
                            "일치도(ARI): ds=14 0.518 / ds=46 0.067 / ds=1489 0.006 → "
                            "refresh 후 1.000. 이 설정을 빠뜨려 diversity/routing 관련 "
                            "결론이 대거 무효화된 전례가 있어 기본값을 True로 뒀다. "
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
    # The centroid initialisation line lives in libs.prototypes, which does

    # ── [Step A, v2 정리] 폐기된 옵션의 기본값 주입 ────────────────
    # 아래 항목은 v2에서 폐기 확정되어 **CLI에서 제거**했다. 다만 코드
    # 곳곳(96개소)이 args.<이름>을 참조하므로, 그 참조를 전부 지우는 대신
    # 폐기 시점의 기본값을 주입한다. 실행 경로는 항상 이 값을 타므로
    # 해당 기능은 사실상 죽은 코드가 되고, CLI 표면적만 줄어든다.
    #
    # ⚠ 이후 실제 코드 제거(Step B)에서 이 블록도 함께 사라져야 한다.
    #   지금 지우지 않는 이유는 fusion_alpha/beta가 supervised.py 학습
    #   루프까지 80회 참조되어 있어 한 번에 건드리면 위험하기 때문이다.
    _V2_DEPRECATED_DEFAULTS = {
        # retrieval projection — 7지표 전부 null (TABERA_V2_DESIGN.md §2-8)
        "retr_proj_mode": "none",
        "detach_retr_grad": False,
        # fusion variants — v1 중간 실험. 결과표에 없음
        "fusion_alpha_override": None,
        "fusion_beta_override": None,
        "fusion_gate_temperature": 1.0,
        # head 실험 — 결과표에 없음
        "blockwise_layernorm": False,
        "head_branch_l2norm": False,
        "confidence_scaling": False,
        "confidence_scaling_detach": False,
        "context_projection": False,
        "head_attn_alpha_override": None,
        "head_neighbor_source": "real",
        "interaction_n_heads": 2,
        # trajectory 로깅 — 미사용
        "log_fusion_trajectory": False,
        "log_centroid_label_mi_trajectory": False,
        "log_shuffle_ablation_trajectory": False,
        "log_representation_drift_trajectory": False,
        # v1 잔여 — head 입력 구성 실험
        "no_query_emb": False,
        "no_context_emb": False,
        "detach_context_grad": False,
        "query_detach_warmup_epochs": 0,
        "query_detach_warmup_steps": 0,
        "freeze_encoder_retrain_head": False,
        "freeze_head_epochs": 50,   # ⚠ freeze_encoder_retrain_head=False면 미사용
        "evidence_metric_override": None,
    }
    for _k, _v in _V2_DEPRECATED_DEFAULTS.items():
        if not hasattr(args, _k):
            setattr(args, _k, _v)

    # [2026-07, 되돌림] use_context_emb=True가 다시 기본값(v1 복원) — 기존 여러
    # 곳에 흩어진 "not args.no_context_emb" 사용처를 하나하나 안 고치고, 여기서
    # args.no_context_emb 자체를 보정해서 그 아래 로직은 전부 그대로 두는 방식
    # (각 호출부를 개별 수정하는 것보다 훨씬 안전). --no_context_emb를 명시적으로
    # 주지 않으면(기본, 대부분의 실행) context_emb 포함(v1). --no_context_emb를
    # 주면 v2식(context_emb 제외)으로 명시적 전환. --use_context_emb는 이제
    # 기본 동작과 같아서 아무 효과 없는 하위호환 플래그.
    if args.no_context_emb:
        args.no_context_emb = True
    else:
        args.no_context_emb = False

    if args.query_detach_warmup_epochs > 0 and args.query_detach_warmup_steps > 0:
        parser.error(
            "--query_detach_warmup_epochs와 --query_detach_warmup_steps는 "
            "동시에 0이 아닐 수 없습니다 — 하나만 지정하세요."
        )

    if args.no_query_emb and args.no_context_emb:
        # [2026-07, 되돌림] use_context_emb=True가 다시 기본값(v1)이라, 이제
        # "극단 케이스"(head 입력이 agg_emb 하나만 남음)는 --no_query_emb와
        # --no_context_emb를 둘 다 명시적으로 켰을 때만 발생 — 그 경우에만 경고.
        print(f"  ℹ️  --no_query_emb와 --no_context_emb를 둘 다 켰습니다 — "
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