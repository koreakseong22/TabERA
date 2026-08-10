"""
libs/search_space.py
====================
TabHERA용 Optuna 하이퍼파라미터 탐색 공간.
MultiTab의 search_space.py 형식을 따릅니다.

get_search_space      : trial → params dict
suggest_initial_trial : 첫 번째 trial의 기본값 (빠른 warmup)
params_to_model_kwargs: params → TabHERA 생성자 인자
"""

from __future__ import annotations
import math
import optuna


# ─────────────────────────────────────────────────────────────
# HPO ↔ reproduce 학습 스케줄 단일 소스
# ─────────────────────────────────────────────────────────────

HPO_TRAINING_SCHEDULE = {"epochs": 100, "patience": 20}
"""[2026-08 변경] 200/30 → 100/20 — MultiTab benchmark(Lee et al.)와
학습 예산을 맞추기 위함. 같은 조건이어야 baseline 대비 성능 차이가
스케줄 차이로 오염되지 않는다.

⚠ 이는 **원래 값으로 되돌리는 것**이다 — 아래 [배경]에 있듯 예전에
optimize.py가 100/20을 하드코딩하고 있었고, 단일 소스로 묶으면서
reproduce.py 쪽 기본값(200/30)을 채택했었다.

⚠ ds=14 실측: 200/30 조건에서도 실제로는 72 epoch에 종료됐다 — 작은
데이터셋은 이미 early stopping이 걸려 있어 영향이 작다. 큰 데이터셋에서는
학습이 잘릴 수 있으므로, 이전 결과와 비교할 때 스케줄이 다르다는 점을
반드시 명시해야 한다.

optimize.py의 각 HPO trial과 reproduce.py의 최종 재현 학습이 반드시
같은 epochs/patience를 쓰도록 강제하는 단일 소스.

[배경] 예전엔 optimize.py에 epochs=100/patience=20이 하드코딩돼 있고,
reproduce.py는 --epochs(기본 200)/--patience(기본 30)를 별도 CLI로
노출하고 있어서, "HPO가 찾은 최적 config를 재현한다"는 이름의 스크립트가
실제로는 HPO 때와 다른 학습 스케줄로 돌아가는 불일치가 있었음(adult(1590)
데이터셋에서 실측: reproduce.py가 더 오래/관대하게 학습했는데도 val
acc가 HPO best trial보다 오히려 낮게 나왔고, centroid 쏠림도 max_cluster_
size 기준 optimize.py 대비 reproduce.py에서 훨씬 심하게 진행됨 —
regroup/dead-centroid reinit이 매 epoch 실행되는 구조라, 학습이 길어질수록
쏠림이 누적되는 방향으로 작동하는 것으로 추정).

원본 MultiTab benchmark(Lee et al.)의 reproduce.py는 애초에 이런 불일치가
구조적으로 발생할 수 없음 — HPO trial과 최종 재현이 model wrapper의
같은 fit() 경로를 그대로 재사용하기 때문. 이 상수는 TabERA 쪽에서도 같은
원칙(학습 스케줄은 하나의 소스에서만 나온다)을 강제하기 위함 —
optimize.py/reproduce.py 둘 다 이 값을 직접 import해서 쓰고, 각자
따로 epochs/patience 숫자를 하드코딩하거나 CLI 기본값으로 재정의하지
않는다. 이 값 자체를 바꾸고 싶으면 여기 한 곳만 바꾸면 양쪽에 다 반영됨.
"""


# ─────────────────────────────────────────────────────────────
# AdaCos fixed-scale 공식 (Zhang et al. 2019, CVPR)
# ─────────────────────────────────────────────────────────────

# ── [v2] k 탐색 제외 대상 ────────────────────────────────────────
# retrieval이 예측 경로 밖인 fusion_mode에서는 k가 HPO objective에
# 영향을 주지 않는다. 아래 모드에서는 k를 고정한다.
_K_UNTUNED_MODES = ("proto_only", "proto_only_linear",
                    "query_only_linear", "proto_dev", "proto_dev_agg", "proto_residual_query",
                    # proto_dev_retr does use k in the prediction path, but it
                    # is held fixed so that it can be compared against
                    # proto_dev on equal footing: tuning k only in the
                    # retrieval arm would give it one extra degree of freedom
                    # and confound "retrieval carries information" with
                    # "k was tuned". The effect of k is a separate question.
                    "proto_dev_retr")
DEFAULT_K_NO_TUNE = 8
# k 탐색 후보. optimize.py의 n_prototypes 상한(N/k) 계산이 같은 목록을
# 봐야 하므로 상수로 둔다 — 두 곳에 하드코딩하면 한쪽만 바뀔 수 있다.
SEARCHED_K_CHOICES = [4, 8, 16, 32, 48, 64]


def adacos_fixed_scale(n_prototypes: int) -> float:
    """centroid routing softmax의 scale factor를 탐색 대신 공식으로 계산.

    [배경] 기존엔 routing_scale을 Optuna가 [1.0, 20.0] 구간에서 데이터셋
    마다 탐색했음(get_search_space 옛 버전 참고). 그런데 이미 그 탐색
    범위를 정할 때 AdaCos 논문(Zhang et al. 2019)의 fixed-scale 공식
    s = √2·log(C-1)을 실제 n_prototypes(P=√N_train) 범위에 대입해본 적이
    있고, 그 결과가 이 프로젝트가 다루는 전체 데이터셋 규모(P=7~322)에서
    s=2.5~8.2로 좁게 수렴한다는 게 이미 확인돼 있었음(search_space.py
    routing_scale 탐색 범위 상한을 20으로 좁힌 근거이기도 함).

    즉 이 파라미터는 "데이터셋마다 안 다른 파라미터"가 아니라 "데이터셋
    마다 다르지만 n_prototypes로부터 원리적으로 계산 가능한 값"이라는
    뜻 — k/n_prototypes처럼 "안 중요해서 고정"이 아니라, "중요하지만
    탐색이 아니라 계산으로 풀 수 있어서 탐색 공간에서 제외"하는 종류의
    컴팩트화. Optuna가 매번 헛수고로 이 축을 훑는 대신, 그 예산을
    lr/dropout처럼 진짜 데이터셋마다 다른 파라미터에 더 배분하기 위함.

    [주의] AdaCos 원 논문은 얼굴 인식 태스크의 클래스 수(C)를 기준으로
    한 공식이라, "centroid 수(P)를 C 대신 넣는다"는 이 프로젝트의
    치환 자체는 검증된 이론이 아니라 유비(analogy)에 기반한 실용적
    근사임 — 성능이 유지되는지 A/B로 반드시 확인해야 함.
    """
    # P<=2면 log(P-1)<=0이 되어 scale이 0 이하로 나올 수 있음 — cosine
    # softmax의 최소한의 뾰족함은 보장하도록 1.0 하한을 둠(기존 탐색
    # 범위 하한과 동일).
    return max(1.0, math.sqrt(2) * math.log(max(n_prototypes - 1, 1)))


# ─────────────────────────────────────────────────────────────
# 초기 trial 기본값 (MultiTab suggest_initial_trial 패턴)
# ─────────────────────────────────────────────────────────────

def study_pkl_tag(
    no_offset_correction: bool = False,
    global_retrieve: bool = False,
    detach_context_grad: bool = False,
    context_projection: bool = False,
    cat_combine: str = "onehot",
    num_embedding: str = "ple",
    evidence_metric: str = "euclidean",  # [추가] AttentionAggregator의 evidence_w
        # 유사도 공간. euclidean(기본값, 기존과 동일)이면 태그 없음 —
        # 기존에 이미 저장된 euclidean study들의 파일명이 안 바뀌게(하위 호환).
    fusion_mode: str = "concat",  # [2026-07, 추가] optimize.py가 이제 fusion_mode를
        # 실제로 바꿀 수 있게 되면서(예전엔 항상 암묵적으로 concat 고정) 처음으로
        # 필요해짐 — concat으로 한 번, residual+no_context_emb로 한 번 돌리면
        # 태그 없이는 둘 다 같은 파일명이라 나중 게 앞 걸 덮어씀. concat(TabERA
        # 생성자 기본값)이면 태그 없음 — 기존에 이미 저장된 study들의 파일명이
        # 안 바뀌게(하위 호환).
    use_context_emb: bool = True,  # [2026-07, 추가] fusion_mode='residual'일 때만
        # 의미 있음(query+β·agg만 쓸지, context도 head에 넣을지). True(기본값)면
        # 태그 없음 — 하위 호환.
    # [2026-07, S-1A] retrieval 전용 표현 분기. study를 분리해야 baseline
    # 하이퍼파라미터로 조용히 학습되는 사고를 막는다(--global_retrieve 때 겪음).
    retr_proj_mode: str = "none",
    disable_retrieval_branch: bool = False,  # [2026-07, 추가] "진짜" retrieval-free
    n_prototypes: "int | None" = None,       # [2026-08, 추가] 아래 설명 참고
    # [2026-08, 추가] 학습 결과를 바꾸는데 태그에 없던 것들. 기본값이면
    # 아무것도 안 붙으므로 기존 study 파일명이 그대로 유지된다.
    # [조건 A/O 실험] centroid 갱신 방식과 dead reinit 은 구조 변경이므로
    # study 를 분리한다 — gradient 조건에서 튜닝한 HP 를 EMA 에 쓰면
    # "centroid 갱신 방식" 이 아니라 "갱신 방식 + HP 부적합" 비교가 된다.
    gradient_codebook: bool = False,
    # commitment 는 기본이 꺼짐(§10). CLI 기본값과 어긋나면 이 함수를 직접
    # 부르는 경로에서 태그가 달라져 study 가 갈린다.
    no_commitment: bool = True,
    disable_dead_reinit: bool = False,
    # ⚠ 기본이 0 이다(L_nbr 제거). 0 이 아닐 때만 태그에 붙는다.
    nbr_lambda: float = 0.0,
    # ⚠ nbr_k 는 이제 libs/tabera.py 의 모듈 상수(NBR_K)다. 호출부가 넘기지
    #   않으므로 항상 기본값이고 태그에 아무것도 안 붙는다. 인자를 남겨둔
    #   것은 하위호환 때문이다 — 상수를 바꾸면 그건 코드 버전 변경이고
    #   freeze_check.py 가 잡는다.
    nbr_k: int = 10,
    num_bins: int = 8,
    cat_embed_dim: int = 16,
    detach_retr_grad: bool = False,
        # baseline(Model 2, TabERA.__init__ 참고). False(기본값)면 태그 없음 —
        # 하위 호환. [교훈] fusion_mode를 "기본값=태그 없음"으로 짰다가 v2 freeze
        # 이후 기본값이 바뀌면서 옛날 study 파일과 충돌한 사고가 있었음(위 참고) —
        # 그 반복을 피하려고 이 플래그는 처음부터 "값이 뭐든 명시적으로 태그"
        # 하는 대신, 기본값(False)일 때만 태그 생략하는 표준 패턴을 그대로 따름
        # (이 자체가 위험한 게 아니라, "기본값 자체가 나중에 바뀔 수 있다"는 걸
        # 놓치는 게 위험했던 것 — disable_retrieval_branch=False는 앞으로도
        # 계속 "기존 TabERA와 동일"을 뜻하므로 여기서는 안전).
) -> str:
    """optimize.py가 저장하는 study .pkl 파일명의 태그 부분을 만든다.

    [배경] optimize.py는 이 태그로 study 파일을 저장하는데, reproduce.py는
    그 파일을 다시 읽어올 때 자기만의 별도 태그 로직(_save_tag, 자기
    출력 파일용이라 구성 항목이 다름 — 예: cat_combine="onehot"이 기본값
    인데도 태그를 붙이고, num_embedding="plr_lite"에도 태그를 붙이는 등)을
    썼음. 두 로직이 따로 존재하면서 한쪽만 바뀌면(이번엔 --num_embedding
    기본값을 plr_lite→ple로 바꾼 것) 조용히 어긋나는 사고가 실제로 남 —
    optimize.py가 "data=41143..num_ple..model=tabera.pkl"로 저장했는데,
    reproduce.py는 태그 없이 "data=41143..model=tabera.pkl"을 찾다가
    FileNotFoundError.

    두 파일이 이 함수 하나를 공유하게 해서, 앞으로 태그 구성 항목이
    늘어나도(예: 새 ablation 플래그 추가) 한 곳만 고치면 항상 일치하도록
    한다.

    [주의 — fusion_mode를 태그에 넣기로 한 결정 변경 이력] V2 시절에는
    fusion_mode를 "재학습 시점 구조 선택"으로 취급해 일부러 태그에서
    뺐음(concat/residual/gated_sum/anchor_gate/context_gated_beta 여러
    설계를 같은 HPO study 위에서 비교하기 위함 — reproduce.py 쪽 설계
    의도). 그런데 optimize.py가 fusion_mode를 실제로 못 바꿨던 그 시절엔
    이게 안전했지만(HPO는 항상 암묵적으로 concat이었으므로), 이제
    optimize.py도 --fusion_mode를 받게 되면서(V2 채택 구조로 HPO 자체를
    다시 돌리기 위함) 파일명 충돌 위험이 실재하게 됨 — 그래서 여기서는
    태그에 포함시킴. optimize.py/reproduce.py 둘 다 각자의 args.fusion_mode/
    args.no_context_emb를 그대로 이 함수에 전달해야 함 — "study는 fusion_mode
    A로 탐색됐는데 reproduce.py가 fusion_mode B로 재현 시도" 같은 실수 조합은
    이제 서로 다른 파일을 가리키므로 애초에 섞일 수 없음(같은 파일명을
    가리키려면 두 스크립트에 같은 --fusion_mode/--no_context_emb를 명시적으로
    맞춰줘야 함 — 이게 오히려 안전한 방향).
    """
    # [2026-08 추가] n_prototypes
    # ⚠ P는 일반 하이퍼파라미터가 아니라 **구조 변수**다. logits ≈ W·c 이고
    #   c는 P개 값만 취하므로 P가 곧 prototype 항의 클래스 표현 용량이고,
    #   동시에 |G_p| = N/P 가 retrieval 지역성을 정한다.
    # ⚠ 그런데 예전 태그에는 P가 없어서, **P만 다른 실험이 같은 study
    #   파일을 공유했다.** ds=1493에서 P=35 study가 P=100 실행에 덮여
    #   사라진 사례가 실제로 있었다(trial 번호가 0부터 새로 시작).
    #   재현성 문제이므로 태그에 넣는다.
    # ⚠ 하위호환: n_prototypes=None(미지정)이면 예전과 같은 이름이 나온다.
    #   기존 study 재개가 깨지지 않는다. 호출부는 P를 알게 된 시점부터
    #   넘기고, 파일이 없으면 legacy 이름으로 폴백한다.
    # ⚠ nbr_lambda / nbr_k / num_bins / cat_embed_dim / detach_retr_grad 는
    #   학습 결과를 바꾸는데 예전 태그에 없었다. 특히 nbr_lambda 는 L_nbr
    #   가중치라 v2(0)와 v3(0.005)가 **같은 study 파일을 공유**한다 —
    #   한쪽이 다른 쪽을 덮어쓰면 재현이 불가능해진다. n_prototypes 에서
    #   실제로 그 사고가 있었다(ds=1493, P=35 study가 P=100 실행에 사라짐).
    # ⚠ 기본값과 같으면 아무것도 붙이지 않는다 — 기존 study 파일명이 그대로
    #   유지되어 재개가 안 깨진다. 비대칭이지만 하위호환이 우선이다.
    # ⚠ [2026-08 기본값 전환] centroid 갱신이 gradient → EMA 로, L_nbr 이
    #   0.005 → 0 으로 바뀌었다. 그런데 두 변화가 모두 "기본값" 이라 태그가
    #   안 붙고, **옛 v3 study 와 같은 파일명이 된다.**
    #   구조가 다른 두 모델이 같은 study 를 쓰면 안 되므로 모델 버전을 넣는다.
    #   (n_prototypes 때 같은 사고가 실제로 있었다 — ds=1493 의 P=35 study 가
    #    P=100 실행에 사라졌다.)
    # ⚠ 모델 버전 태그. 기본값이 바뀔 때마다 올린다 — 안 그러면 구조가 다른
    #   모델이 같은 study 를 재개해서 오염된다.
    #     v3ema   EMA prototype memory + L_nbr 제거 (2026-08)
    #     v3ema2  + commitment 제거 (§10)
    return "..v3ema2" \
        + ("..no_offset" if no_offset_correction else "") \
         + (f"..retrproj_{retr_proj_mode}" if retr_proj_mode != "none" else "") \
        + ("..global_retrieve" if global_retrieve else "") \
        + ("..detach_ctx" if detach_context_grad else "") \
        + ("..ctx_proj" if context_projection else "") \
        + ("..cat_concat" if cat_combine == "concat" else "") \
        + ("..cat_sum" if cat_combine == "sum" else "") \
        + ("..num_ple" if num_embedding == "ple" else "") \
        + ("..num_linear" if num_embedding == "linear" else "") \
        + (f"..evM_{evidence_metric}" if evidence_metric != "euclidean" else "") \
        + f"..fusion_{fusion_mode}" \
        + ("..noctx" if not use_context_emb else "") \
        + ("..no_retrieval" if disable_retrieval_branch else "") \
        + (f"..P{int(n_prototypes)}" if n_prototypes is not None else "") \
        + ("..grad_cb" if gradient_codebook else "") \
        + ("" if no_commitment else "..commit") \
        + ("..nodr" if disable_dead_reinit else "") \
        + (f"..nbr{nbr_lambda:g}" if abs(nbr_lambda) > 1e-12 else "") \
        + (f"..nbrk{int(nbr_k)}" if int(nbr_k) != 10 else "") \
        + (f"..bins{int(num_bins)}" if int(num_bins) != 8 else "") \
        + (f"..catdim{int(cat_embed_dim)}" if int(cat_embed_dim) != 16 else "") \
        + ("..detach_retr" if detach_retr_grad else "")


def suggest_initial_trial() -> dict:
    """
    첫 번째 trial에 enqueue할 기본 하이퍼파라미터.
    논문에서 제시한 설정 기반.
    """
    return {
        "embed_dim":        128,
        "k":                16,
        "embedder_layers":  2,
        "dropout":          0.1,
        "loss_diversity":   0.10,   # 새 하한(5e-2) 안에서 안정적인 초기값
        "loss_commitment":  0.05,
        "loss_codebook":    0.05,   # commitment와 같은 스케일로 시작 (신규)
        "lr":               3e-4,
        "weight_decay":     1e-5,
        "batch_size":       256,
        # "anneal_factor":    0.97,
        # "n_heads":          4,  
    }


# ─────────────────────────────────────────────────────────────
# 탐색 공간 (MultiTab get_search_space 패턴)
# ─────────────────────────────────────────────────────────────

def get_search_space(
    trial: optuna.Trial,
    num_features: int = 0,   # MultiTab 호환 인자 (현재 미사용)
    data_id: int = 0,        # MultiTab 호환 인자 (현재 미사용)
    metric: str = "l2",      # TabR Retriever 거리 지표
    num_embedding: str = "ple",  # [수정] 공식 채택값(2026-07)과 일관되게 변경.
                                   # optimize.py는 항상 args.num_embedding을 명시적으로
                                   # 넘기므로 실제 파이프라인 동작엔 영향 없음 — 이 함수를
                                   # 직접 호출하는 경우(테스트/노트북 등)를 위한 방어적 기본값.
    # Default matches the CLI (commitment is off, §10). A mismatch here would
    # mean any caller that omits the argument searches a weight the forward
    # pass never reads.
    no_commitment: bool = True,      # [ablation] commitment_loss 를 끈다.
                                   # EMA 에서 commitment 는 **encoder 를 prototype
                                   # memory 에 정렬시키는 유일한 손실**이다
                                   # (codebook 은 EMA 가 대체). 그 필요성을 재려면
                                   # 탐색 공간에서도 빼야 한다 — 안 그러면
                                   # HPO 가 commitment 를 튜닝한 뒤 재현에서만
                                   # 끄는 꼴이 된다.
    use_ema_codebook: bool = True,   # [2026-08] loss_codebook 탐색 여부를 결정한다.
                                   # EMA 가 기본이 되면서 codebook_loss 가 호출조차
                                   # 되지 않으므로(실측 배치당 0회) 죽은 차원을
                                   # 탐색하지 않는다. optimize.py 가
                                   # not args.gradient_codebook 을 넘긴다.
    fusion_mode: str = "proto_dev",  # [v2] k 탐색 여부를 결정한다.
                                   # retrieval이 예측 경로 밖인 모드에서는 k를 고정
                                   # (_K_UNTUNED_MODES 참고). optimize.py가
                                   # args.fusion_mode를 명시적으로 넘겨야 한다.
) -> dict:
    """
    Optuna Trial로부터 TabHERA 하이퍼파라미터를 샘플링합니다.

    Parameters
    ----------
    trial        : optuna.Trial
    num_features : 입력 특성 수 (조건부 탐색에 활용 가능)
    data_id      : 데이터셋 ID (조건부 탐색에 활용 가능)
    metric       : TabR Retriever 거리 지표
    num_embedding: "linear"/"ple"/"plr_lite" — plr_lite일 때만 PLR 하이퍼
                   파라미터(sigma, n_frequencies, out_dim)를 탐색 공간에 포함

    Returns
    -------
    dict: 모델 생성 및 학습에 필요한 전체 파라미터
    """
    space = {
        # ── 모델 구조 ───────────────────────────────────
        "embed_dim":       trial.suggest_categorical("embed_dim",   [64, 128, 256]),
        # n_prototypes: optimize.py에서 sqrt(N)으로 자동 설정 (탐색 대상 아님)
        # [2026-07, 재개방] retrieval geometry 분석(centroid_retrieval_behavior,
        # margin/N_eff/topk_idx overlap)이 끝나고 "centroid마다 retrieval
        # behavior가 실제로 다르다"는 결론이 확인된 뒤 k를 다시 탐색
        # 대상으로 되돌림. 예전에 k=16으로 고정했던 근거(RandomForest
        # 중요도 분석 k importance=0.027)는 embedder_layers 때와 같은
        # restriction-of-range 문제(애초에 탐색 공간에 없었던 study들
        # 위에서 나온 결과라 재확인 자체가 불가능한 순환논리)일 수 있어
        # 재검증 필요 — 특히 이제 fusion_mode=residual+no_context_emb를
        # HPO 시점부터 직접 탐색하므로(아래 참고), k의 중요도도 이
        # architecture 기준으로 다시 봐야 함.
        # ⚠ [v2] retrieval이 예측 경로 밖인 모드에서는 k가 **예측에 영향을 주지
        #   않는다** — logits가 context_emb(또는 query_emb)만으로 계산되므로
        #   HPO objective가 k에 반응하지 않는다. 탐색하면 무작위 선택이 되어
        #   탐색 예산만 낭비한다.
        #   k는 Stage 2(설명용 이웃 개수)의 configuration이며, 필요하면
        #   Level 1 지표(lo_spearman_mean 등)로 별도 선택한다.
        #   기본값 8 — LLM 입력 길이를 고려한 값(4는 부족, 16은 과다).
        "k": (DEFAULT_K_NO_TUNE if fusion_mode in _K_UNTUNED_MODES
              else trial.suggest_categorical("k", SEARCHED_K_CHOICES)),
        # [원복 — 2026-07] embedder_layers를 [2,4] → [1,4]로 되돌림.
        # 좁혔던 근거({1:2, 2:1, 3:9, 4:10} 분포, 아래 원래 주석)가 이후
        # 재검토 결과 두 가지 문제가 있었음:
        # (1) 그 뒤로 수집된 study들은 이미 [2,4]로 좁혀진 상태에서 돌았기
        #     때문에, "1을 아무도 안 고른다"는 재확인 자체가 불가능한
        #     순환논리였음(애초에 선택지에 없었으므로).
        #     RandomForest 중요도 분석에서도 embedder_layers가 최하위로
        #     나왔는데, 이것도 같은 이유(범위 제한→분산 감소→중요도
        #     과소평가, restriction of range) 로 신뢰할 수 없음.
        # (2) 1을 선호했던 2개 데이터셋(51 heart-h, 41143 jasmine) 중
        #     jasmine은 이후 PLE 인코더 채택 검증 과정에서 PLR 대비 test
        #     성능이 낮게 나왔는데, 이 데이터셋이 정확히 layers=1 선호
        #     이력이 있는 곳이라 — 그 결과가 인코더 문제가 아니라 이미
        #     막혀있던 layers=1 선택지 때문일 가능성을 배제 못 함.
        # 편향 없이 재확인하기 전까지는 원래 범위로 복원.
        "embedder_layers": trial.suggest_int("embedder_layers", 1, 4),
        # [원래 주석, 참고용] embedder_layers 후보 [1,2,3,4] → [2,3,4]로 축소.
        # 근거: 같은 22개 데이터셋 best trial의 embedder_layers 분포가
        # {1:2, 2:1, 3:9, 4:10}으로 3~4에 압도적으로 쏠림(19/22). 1을 완전히
        # 빼지 않고 2~4로 좁힌 건, 1을 선호한 2개 데이터셋(51 heart-h,
        # 41143 jasmine)의 여지를 조금 남겨두기 위함. embed_dim은 분포가
        # {64:6, 128:5, 256:11}로 3분할이 팽팽하고 n_features/n_train과도
        # 상관관계 없어(rho=0.15~0.20, p>0.37) 안전하게 좁힐 근거가 부족해
        # 그대로 유지. batch_size도 이후 electricity/nomao(대형 데이터셋)를
        # 다룰 예정이라 미리 좁히지 않고 그대로 둠.
        "dropout":         trial.suggest_float("dropout", 0.0, 0.5, step=0.05),

        # ── 보조 손실 가중치 ────────────────────────────
        # [수정] loss_diversity 하한: 1e-2 → 5e-2
        # 근거: australian(id=40981) seed=1에서 loss_diversity=0.036이 선택됐을 때
        #       off-diagonal mean=0.172로 centroid 분리 불충분 확인.
        #       top1-top2 logit gap=0.020으로 confidence flat 심화.
        #       하한을 5e-2로 올려 최소한의 centroid 분리도 보장.
        # ⚠ [v2] diversity loss는 **비활성화**되었다.
        #   fresh 조건(--refresh_on_best) 재검증에서 retrieval 지표
        #   (rank_info / fallback / gain)와 예측 성능 모두 유의차가 없었다.
        #   코드는 CentroidLayer 최소화 실험(Phase 1)을 위해 남겨두되
        #   탐색 공간에서는 제외한다.
        #   Disabled and excluded from optimization search.
        #   Retained only for backward compatibility.
        "loss_diversity":  0.0,

        # [수정] loss_commitment 하한: 1e-4 → 1e-2
        # 근거: colic(id=25) seed=1에서 loss_diversity=0.466, loss_commitment=0.0019으로
        #       diversity:commitment 비율이 244:1이 되어 centroid가 데이터에서 멀어지는
        #       현상 확인. 두 loss가 같은 order of magnitude에서 탐색되도록 하한 통일.
        "loss_commitment": (0.0 if no_commitment else
                            trial.suggest_float("loss_commitment", 1e-2, 1e-1, log=True)),

        # [추가] loss_codebook — codebook_loss(commitment_loss의 반대 방향,
        # centroid를 배정된 쿼리 쪽으로 당김) 가중치. VQ-VAE 원 논문은 이
        # 항이 없으면 centroid가 "자기 그룹의 대표"가 되도록 당겨주는 직접
        # 신호가 아예 없다는 게 실측으로 확인됨(credit-g centroid_geometry
        # 진단에서 cohesion이 낮은 centroid들이 다수 발견됨). commitment_loss와
        # 같은 스케일(1e-2~1e-1)로 시작 — 둘 다 같은 ‖query-centroid‖²를
        # 미는 손실이라 크기가 크게 벌어지면 한쪽만 지배적이 될 위험이 있음
        # (loss_diversity/loss_commitment 하한을 통일했던 것과 같은 이유).
        # ⚠ [2026-08] EMA 에서는 이 값이 **아무 데도 쓰이지 않는다** —
        #   tabera.py 가 use_ema_codebook 분기에서 codebook_loss 를 호출하지
        #   않는다(실측: 배치당 0회). 탐색하면 예산 한 차원을 죽은 값에
        #   쓰고, checkpoint 를 보면 codebook=0.07 이라 "쓰이는 줄" 오해한다.
        #   loss_diversity 가 이미 같은 이유로 제외돼 있다(위 참고).
        #   gradient codebook 실험(--gradient_codebook)에서는 그대로 탐색한다.
        "loss_codebook":   (0.0 if use_ema_codebook else
                            trial.suggest_float("loss_codebook", 1e-2, 1e-1, log=True)),

        # ── 학습 파라미터 ───────────────────────────────
        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        # [수정 — 컴팩트화, 2026-07] batch_size를 탐색 대상에서 제외, 고정값
        # 256 사용. 근거: (1) RandomForest 중요도 분석(18개 데이터셋)에서
        # batch_size importance=0.037로 9개 중 7~8위, embed_dim/embedder_layers
        # 다음으로 낮음. (2) 데이터셋 크기별 최적값을 직접 스윕(profb/credit-g/
        # vehicle/jasmine, {64,128,256,512})해본 결과 신뢰할 만한 크기-batch_size
        # 관계를 못 찾음 — vehicle은 bs256→0.847, bs512→0.671로 인접 값 사이에서
        # 조차 낙폭이 컸는데, 이건 STE+dead-centroid reinit이 학습 초반의 작은
        # 차이를 증폭시키는 이 아키텍처 고유의 노이즈(regroup_warmup_epochs
        # 도입 배경과 동일)에 묻힌 것으로 보임 — 즉 batch_size의 순수한 효과가
        # 노이즈보다 작아서 Optuna가 탐색해봤자 노이즈에 맞추는 것에 가까움.
        # 256은 TabR 계보 문헌에서 소규모 데이터셋(N_train ~수백~수천)에 흔히
        # 쓰이는 값이자 suggest_initial_trial()의 기존 기본값과도 일치.
        # [주의 — 잠정값] 이건 "batch_size가 중요하지 않다"는 확정이 아니라
        # "지금은 노이즈 때문에 신호를 못 본다"는 상태임. regroup_warmup_epochs로
        # 노이즈를 줄인 뒤, 또는 electricity/nomao 같은 대형 데이터셋을 다룰
        # 때 이 결정을 다시 검토해야 함.
        #
        # ── [2026-08 갱신] EMA prototype memory 전환 후의 근거 ──────────
        # 위 근거들은 전부 gradient codebook 시절 것이다. EMA 로 바꾸면서
        # batch_size 가 **새로운 경로**를 하나 더 갖게 됐다.
        #
        #     B  →  gradient noise                    (원래 있던 경로)
        #     B  →  batch 당 prototype 별 표본 수  →  EMA 갱신의 통계적 안정성
        #
        # `c_p ← m_p / N_p` 에서 N_p 는 그 배치에서 prototype p 에 배정된
        # 샘플 수다. 즉 **N_train 이 아니라 B/P 를 봐야 한다.**
        #
        #   ds     N     P    B/P    빈 prototype 기대((1−1/P)^B)
        #   54    676   26    9.8    0.0%
        #   31    800   28    9.1    0.0%
        #   14   1600   40    6.4    0.2%
        #   1043 3649   60    4.3    1.4%
        #   1489 4322   65    3.9    1.9%
        #
        # P = √N 규칙 덕분에 B=256 에서 B/P 가 3.9~9.8 로 좁게 모인다.
        # ⚠ 그래서 "작은 데이터셋이니 B 를 32 로" 같은 N 기반 축소 규칙은
        #   **오히려 해롭다** — ds=54 에 B=32 면 B/P=1.2, 빈 prototype 기대
        #   29% 가 되어 매 배치 prototype 셋 중 하나가 갱신되지 않는다.
        #
        # ⚠ 256 이 최적이라는 뜻이 아니라 **고정된 benchmark protocol** 이다.
        #   다시 건드린다면 두 축을 같이 봐야 한다:
        #     ① B/P              EMA 갱신 안정성
        #     ② (N/B) × epochs   optimizer update budget
        #   현재 ①은 양호하고 ②는 데이터셋 간 편차가 크다(steps/epoch 2~16).
        #   ②는 early stopping 이 부분적으로 흡수하지만 통제되지는 않는다.
        "batch_size":      256,
        # "anneal_factor":   trial.suggest_float("anneal_factor", 0.90, 0.99),
        # "n_heads":         trial.suggest_categorical("n_heads", [1, 2, 4, 8]),

        # ── TabR Retriever 거리 지표 (확장용) ──────────
        "metric":          metric,

        # [수정 — 컴팩트화] routing_scale을 더 이상 여기서 탐색하지 않음.
        # 기존엔 trial.suggest_float("routing_scale", 1.0, 20.0, log=True)
        # 였는데, 이 축 하나가 AdaCos fixed-scale 공식(adacos_fixed_scale,
        # 이 파일 상단)으로 n_prototypes만 알면 계산 가능하다는 게 이미
        # 이 프로젝트 자체 분석(탐색 범위를 20으로 좁힌 근거)으로 확인돼
        # 있었음. 실제 계산은 params_to_model_kwargs()에서 n_prototypes를
        # 이용해 수행 — 여기서 굳이 trial 정보 없이도 계산 가능하므로
        # get_search_space() 시그니처(n_prototypes 미보유)를 안 건드림.
        # [하위 호환] 구버전 study의 best_params에는 trial.suggest_float로
        # 기록된 routing_scale이 그대로 남아있으므로, 그 study를 다시
        # 불러오면(reproduce.py) 이 새 공식이 아니라 기존 탐색값을 그대로
        # 사용함(params_to_model_kwargs의 .get() fallback 참고) — 재현성
        # 보존.
    }

    if num_embedding == "plr_lite":
        # [추가] PLR(lite) 하이퍼파라미터를 고정값이 아니라 trial마다 탐색.
        # 근거: Gorishniy et al. 2022(Periodic embedding 원 논문)가 sigma(주파수
        # 스케일)와 k(주파수 개수)를 feature 전체 공통 하이퍼파라미터로 두고
        # 데이터셋마다 튜닝한다고 명시함(σ: LogUniform 계열 권장, k: UniformInt
        # 계열). 이전엔 optimize.py가 --plr_freq_scale/--plr_n_frequencies를
        # 전체 실행에 고정값(0.01, 16)으로 넣어서 100 trial이 전부 같은 sigma를
        # 썼음 — mfeat-fourier/vehicle(둘 다 categorical 없이 PLR이 numeric
        # 인코딩을 전담하는 데이터셋)에서 100 trial 중 7~8개가 완전 붕괴
        # (val_acc가 정확히 무작위 확률 수준)하는 현상 관찰 후 이 고정값이
        # 해당 데이터셋 분포에 안 맞았을 가능성으로 판단해 탐색 대상으로 전환.
        space["plr_freq_scale"] = trial.suggest_float("plr_freq_scale", 0.01, 100.0, log=True)
        space["plr_n_frequencies"] = trial.suggest_int("plr_n_frequencies", 8, 96)
        space["plr_out_dim"] = trial.suggest_categorical("plr_out_dim", [4, 8, 16, 32])

    return space


# ─────────────────────────────────────────────────────────────
# params → TabHERA 생성자 인자 변환
# ─────────────────────────────────────────────────────────────

def params_to_model_kwargs(params: dict, n_features: int, n_output: int) -> dict:
    kwargs = {
        "n_features":      n_features,
        "embed_dim":       params["embed_dim"],
        "n_prototypes":    params["n_prototypes"],
        # [버그 수정] k는 get_search_space()에서 trial.suggest_*가 아니라
        # 고정값(16)으로 dict에 직접 넣는데, Optuna의 study.best_params는
        # trial.suggest_*()로 "제안된" 파라미터만 기록하고 이렇게 수동으로
        # 넣은 키는 기록하지 않는다. 즉 optimize.py 실행 중(그 trial의
        # fresh params 딕셔너리)에는 k=16이 있지만, reproduce.py가 나중에
        # study.best_params를 다시 읽어올 때는 k가 아예 없어서
        # KeyError가 남 — .get()으로 같은 고정값(16)을 fallback으로 사용.
        # ⚠ [v2] retrieval이 예측 경로 밖인 fusion_mode에서는 get_search_space가
        #   k를 suggest하지 않으므로 best_params에 키가 없다. 그 경우
        #   DEFAULT_K_NO_TUNE(8)을 쓴다 — 학습 시점의 값과 일치해야 한다.
        #   (구버전 study 호환을 위해 키가 있으면 그 값을 그대로 쓴다.)
        "k":               params.get("k", DEFAULT_K_NO_TUNE),
        "embedder_layers": params["embedder_layers"],
        "dropout":         params["dropout"],
        "n_output":        n_output,
        "loss_weights": {
            # ⚠ [v2] diversity는 탐색 공간에서 제외됐으므로 신규 study의
            #   best_params에는 이 키가 없다. 0.0으로 fallback한다
            #   (탐색에서 제외 = 비활성화와 같은 의미).
            # ⚠ [TODO — dead_reinit ablation 후 정리] use_ema_codebook=True 가
            #   기본이 되면서 loss_codebook / loss_diversity 가 **아무 데도
            #   쓰이지 않는다**(tabera.py 가 EMA 분기에서 두 손실을 호출조차
            #   안 함 — 실측: 배치당 호출 0회). 그런데 HPO 는 loss_codebook 을
            #   계속 탐색하고 값이 checkpoint 에 저장된다.
            #     · 결과에 영향 없음 (버그 아님)
            #     · 탐색 예산 한 차원 낭비
            #     · checkpoint 를 보면 codebook=0.07 이라 "쓰이는 줄" 오해
            #   ⚠ 지금 고치면 진행 중인 ablation 과 탐색 공간이 달라지므로,
            #     dead_reinit ON/OFF 결과가 나온 뒤에 정리한다.
            "diversity":   params.get("loss_diversity", 0.0),
            # ⚠ [버그 수정] 직접 인덱싱이었다. `--no_commitment` 로 탐색을
            #   건너뛰면 best_params 에 이 키가 아예 없어 KeyError 가 났다.
            #   codebook 과 같은 방식(.get)으로 통일한다. 키가 없다는 것은
            #   "그 손실을 안 쓴다" 는 뜻이므로 0.0 이 맞는 기본값이다.
            "commitment":  params.get("loss_commitment", 0.0),
            # .get() — codebook_loss 추가 이전에 저장된 구버전 study의
            # best_params에는 이 키가 없음 (그런 경우 codebook_loss 자체가
            # 없던 이전 동작과 동일하게 0.0 사용 — tabera.py의 aux_loss
            # 조합에서도 동일한 fallback을 씀).
            "codebook":    params.get("loss_codebook", 0.0),
        },
        # [수정] routing_scale이 없는 경우(신규 study — get_search_space가
        # 더 이상 trial.suggest_float로 이 키를 채우지 않음)엔 AdaCos
        # fixed-scale 공식으로 계산. 반대로 구버전 study의 best_params에
        # 이 키가 이미 있으면(예전엔 tuning 대상이었으므로) 그 값을 그대로
        # 씀 — 재현성 보존(기존 체크포인트/study의 결과가 이 변경으로
        # 조용히 달라지지 않게 함).
        "routing_scale":   params.get("routing_scale", adacos_fixed_scale(params["n_prototypes"])),
    }
    # PLR(lite) 하이퍼파라미터 — get_search_space가 num_embedding="plr_lite"일
    # 때만 params에 넣어주므로, 있을 때만 그대로 전달 (없으면 TabERA 기본값
    # 또는 CLI --plr_* 고정값 사용).
    for key in ("plr_freq_scale", "plr_n_frequencies", "plr_out_dim"):
        if key in params:
            kwargs[key] = params[key]
    return kwargs