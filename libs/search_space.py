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
import optuna


# ─────────────────────────────────────────────────────────────
# 초기 trial 기본값 (MultiTab suggest_initial_trial 패턴)
# ─────────────────────────────────────────────────────────────

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
    num_embedding: str = "linear",  # "plr_lite"면 PLR 관련 하이퍼파라미터도 탐색
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
        # [수정] k 고정(16) — RandomForest 기반 하이퍼파라미터 중요도 분석
        # (22개 데이터셋, 100 trial씩) 결과 k importance=0.027로 최하위권,
        # 그리고 이전에 이미 인과적으로도 검증됨(global_retrieve ablation —
        # group-constrained retrieval의 k 값 자체가 성능을 좌우하지 않음).
        # 12차원 탐색 공간에서 안 중요한 차원을 고정해 trial 예산을 lr/
        # plr_freq_scale 같은 실제 중요한 차원에 더 배분.
        "k":               16,
        # [수정] embedder_layers 후보 [1,2,3,4] → [2,3,4]로 축소.
        # 근거: 같은 22개 데이터셋 best trial의 embedder_layers 분포가
        # {1:2, 2:1, 3:9, 4:10}으로 3~4에 압도적으로 쏠림(19/22). 1을 완전히
        # 빼지 않고 2~4로 좁힌 건, 1을 선호한 2개 데이터셋(51 heart-h,
        # 41143 jasmine)의 여지를 조금 남겨두기 위함. embed_dim은 분포가
        # {64:6, 128:5, 256:11}로 3분할이 팽팽하고 n_features/n_train과도
        # 상관관계 없어(rho=0.15~0.20, p>0.37) 안전하게 좁힐 근거가 부족해
        # 그대로 유지. batch_size도 이후 electricity/nomao(대형 데이터셋)를
        # 다룰 예정이라 미리 좁히지 않고 그대로 둠.
        "embedder_layers": trial.suggest_int("embedder_layers", 2, 4),
        "dropout":         trial.suggest_float("dropout", 0.0, 0.5, step=0.05),

        # ── 보조 손실 가중치 ────────────────────────────
        # [수정] loss_diversity 하한: 1e-2 → 5e-2
        # 근거: australian(id=40981) seed=1에서 loss_diversity=0.036이 선택됐을 때
        #       off-diagonal mean=0.172로 centroid 분리 불충분 확인.
        #       top1-top2 logit gap=0.020으로 confidence flat 심화.
        #       하한을 5e-2로 올려 최소한의 centroid 분리도 보장.
        "loss_diversity":  trial.suggest_float("loss_diversity",  5e-2, 5e-1, log=True),

        # [수정] loss_commitment 하한: 1e-4 → 1e-2
        # 근거: colic(id=25) seed=1에서 loss_diversity=0.466, loss_commitment=0.0019으로
        #       diversity:commitment 비율이 244:1이 되어 centroid가 데이터에서 멀어지는
        #       현상 확인. 두 loss가 같은 order of magnitude에서 탐색되도록 하한 통일.
        "loss_commitment": trial.suggest_float("loss_commitment", 1e-2, 1e-1, log=True),

        # ── 학습 파라미터 ───────────────────────────────
        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":      trial.suggest_categorical("batch_size", [128, 256, 512]),
        # "anneal_factor":   trial.suggest_float("anneal_factor", 0.90, 0.99),
        # "n_heads":         trial.suggest_categorical("n_heads", [1, 2, 4, 8]),

        # ── TabR Retriever 거리 지표 (확장용) ──────────
        "metric":          metric,

        # [추가] centroid routing softmax의 scale factor. ArcFace/CosFace/
        # AdaCos/von Mises-Fisher Loss 등 코사인 유사도 기반 softmax 문헌에서
        # 공통 지적: cos 유사도가 [-1,1]이라는 좁은 범위라 스케일링 없이
        # softmax에 넣으면 분포가 평평해지고 STE backward gradient가 약해짐
        # (실측: P=10, scale=1.0일 때 1등 확률 평균 0.142 — 거의 무작위
        # 수준인 0.1에 가까움).
        #
        # [범위 재검증] 처음엔 [1,50]으로 뒀는데, AdaCos 논문(Zhang et al.
        # 2019)의 fixed-scale 공식 s≈√2·log(C-1)을 실제 n_prototypes(P=
        # sqrt(N_train)) 범위에 대입해보니: 현재 다루는 소형 데이터셋(P=7~65)
        # 은 s=2.5~5.9, 향후 다룰 대형 데이터셋(electricity/nomao 등, P=166~
        # 322)까지 포함해도 s=7.2~8.2를 넘지 않음 — 즉 이론적으로 타당한
        # 구간은 전부 ~1~8 안에 있는데 상한을 50으로 두면 로그스케일 확률
        # 질량의 절반 가까이가 그 이론 구간 밖에 낭비됨. 상한을 20으로
        # 줄여 여유(이론 최대값의 ~2.5배)는 남기되 탐색 효율을 높임.
        "routing_scale":   trial.suggest_float("routing_scale", 1.0, 20.0, log=True),
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
        "k":               params["k"],
        "embedder_layers": params["embedder_layers"],
        "dropout":         params["dropout"],
        "n_output":        n_output,
        "loss_weights": {
            "diversity":   params["loss_diversity"],
            "commitment":  params["loss_commitment"],
        },
        # .get() — 이 파라미터 추가 이전에 저장된 구버전 study의 best_params
        # 에는 이 키가 없을 수 있음 (그런 경우 기존과 동일한 1.0 사용).
        "routing_scale":   params.get("routing_scale", 1.0),
    }
    # PLR(lite) 하이퍼파라미터 — get_search_space가 num_embedding="plr_lite"일
    # 때만 params에 넣어주므로, 있을 때만 그대로 전달 (없으면 TabERA 기본값
    # 또는 CLI --plr_* 고정값 사용).
    for key in ("plr_freq_scale", "plr_n_frequencies", "plr_out_dim"):
        if key in params:
            kwargs[key] = params[key]
    return kwargs