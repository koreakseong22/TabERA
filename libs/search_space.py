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
        "n_prototypes":     8,
        "k":                16,
        "embedder_layers":  2,
        "dropout":          0.1,
        "loss_diversity":   0.01,
        "loss_commitment":  0.01,
        "loss_entropy":     0.01,
        "lr":               3e-4,
        "weight_decay":     1e-5,
        "batch_size":       256,
        "anneal_factor":    0.97,
        "n_heads":          4,
    }


# ─────────────────────────────────────────────────────────────
# 탐색 공간 (MultiTab get_search_space 패턴)
# ─────────────────────────────────────────────────────────────

def get_search_space(
    trial: optuna.Trial,
    num_features: int = 0,   # MultiTab 호환 인자 (현재 미사용)
    data_id: int = 0,        # MultiTab 호환 인자 (현재 미사용)
    metric: str = "l2",      # TabR Retriever 거리 지표
) -> dict:
    """
    Optuna Trial로부터 TabHERA 하이퍼파라미터를 샘플링합니다.

    Parameters
    ----------
    trial        : optuna.Trial
    num_features : 입력 특성 수 (조건부 탐색에 활용 가능)
    data_id      : 데이터셋 ID (조건부 탐색에 활용 가능)
    metric       : TabR Retriever 거리 지표

    Returns
    -------
    dict: 모델 생성 및 학습에 필요한 전체 파라미터
    """
    return {
        # ── 모델 구조 ───────────────────────────────────
        "embed_dim":       trial.suggest_categorical("embed_dim",   [64, 128, 256]),
        "n_prototypes":    trial.suggest_int("n_prototypes", 4, 16, step=4),
        # k 확장: 소수 클래스 이웃 포함 확률 향상 → recall↑ → f1 gap 완화
        "k":               trial.suggest_categorical("k",           [8, 16, 32, 64]),
        "embedder_layers": trial.suggest_int("embedder_layers", 1, 4),
        "dropout":         trial.suggest_float("dropout", 0.0, 0.5, step=0.05),

        # ── 보조 손실 가중치 ────────────────────────────
        # [수정] loss_diversity 하한: 1e-4 → 1e-2
        # 근거: credit-approval(id=29) seed=1에서 loss_diversity=0.000357이
        #       선택됐을 때 centroid collapse 발생이 실험적으로 확인됨.
        #       diversity=0.1로 override 시 collapse 해소 확인.
        #       하한을 1e-2로 올려 HPO가 collapse 유발 범위를 탐색하지 않도록 제한.
        "loss_diversity":  trial.suggest_float("loss_diversity",  1e-2, 5e-1, log=True),
        "loss_commitment": trial.suggest_float("loss_commitment", 1e-4, 1e-1, log=True),
        # STE collapse 방지 — 배정 분포 entropy 최대화 (VQ-VAE-2, Razavi et al., 2019)
        "loss_entropy":    trial.suggest_float("loss_entropy",    1e-3, 1e-1, log=True),

        # ── 학습 파라미터 ───────────────────────────────
        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size":      trial.suggest_categorical("batch_size", [128, 256, 512]),
        "anneal_factor":   trial.suggest_float("anneal_factor", 0.90, 0.99),
        "n_heads":         trial.suggest_categorical("n_heads", [1, 2, 4, 8]),

        # ── TabR Retriever 거리 지표 (확장용) ──────────
        "metric":          metric,
    }


# ─────────────────────────────────────────────────────────────
# params → TabHERA 생성자 인자 변환
# ─────────────────────────────────────────────────────────────

def params_to_model_kwargs(params: dict, n_features: int, n_output: int) -> dict:
    return {
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
            "entropy":     params["loss_entropy"],
        },
    }
