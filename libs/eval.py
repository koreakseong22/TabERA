"""
libs/eval.py
============
MultiTab 스타일 평가 지표.

calculate_metric : val/test split 별 dict 반환
compute_metric   : 내부 학습 루프용 단순 버전
is_study_todo    : 최적화 재개 여부 판단
"""

from __future__ import annotations

import numpy as np
import optuna
import torch
import torch.nn as nn
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────
# MultiTab / ModernNCA 호환 지표 계산
# ─────────────────────────────────────────────────────────────

def calculate_metric(
    y_true: torch.Tensor,
    preds:  torch.Tensor,
    probs:  Optional[torch.Tensor],
    tasktype: str,
    split: str,   # 'val' or 'test'
) -> Dict[str, float]:
    """
    MultiTab optimize.py 경로가 기록하는 지표에 bacc 와 비교용 _mt_ 키를
    더해 반환합니다.

    classification:
        acc_{split}, auroc_{split}, f1_{split}, logloss_{split}
                                                     (MultiTab과 같은 4개)
        bacc_{split}                                 (TabERA 진단용 추가)
        f1_mt_{split}                                (MultiTab 규약 F1)
        logloss_mt_{split}, auroc_mt_{split}         (multiclass 전용)
    regression:
        rmse_{split}

    ⚠ 이 함수는 한때 "ModernNCA와 동일한 지표 세트를 반환한다"고 적혀
      있었는데 사실이 아니다. 확인 결과:

      (1) MultiTab 의 optimize.py -> eval.py 경로가 내는 것은
          acc / auroc / f1 / logloss **4개뿐**이고 bacc 는 없다. 저장소
          전체에서 balanced_accuracy_score 는 libs/utils_modernnca.py 에만
          있는데, 그 파일은 ModernNCA 내부 유틸이라 optimize.py 나
          reproduce.py 어디서도 import 되지 않는다. 실제로 배포된 baseline
          로그 CSV 에 bacc 컬럼이 없다.

      (2) 그 ModernNCA 유틸의 지표 세트와도 같지 않다. 유틸은
          (Accuracy, Avg_Recall, Avg_Precision, F1, LogLoss, AUC) 6개를
          내는데, 여기에는 Avg_Precision 이 없고 이진 F1 의 average 도
          다르다(유틸은 'binary', 여기는 'macro' -- 그래서 f1_mt 가 따로
          필요하다).

      즉 bacc 는 **비교 대상이 없는 TabERA 전용 진단 지표**다. 논문 표에
      baseline 과 나란히 올릴 수 없다. 다만 내부 분석에는 유용하다 --
      (acc, bacc, f1) 삼중항으로 이진 데이터셋의 클래스 사전확률을 역산해
      f1 의 average 규약이 MultiTab 과 다르다는 것을 특정한 것이 이 값이다.

    ⚠ 반환 순서에서 acc_{split} 이 첫 키라는 계약은 compute_metric() 의
      checkpoint selection 이 의존한다. 앞쪽에 키를 끼워 넣지 말 것.
    """
    y_np = (y_true.detach().cpu().numpy()
            if isinstance(y_true, torch.Tensor) else np.array(y_true))
    p_np = (preds.detach().cpu().numpy()
            if isinstance(preds,  torch.Tensor) else np.array(preds))
    if probs is None:
        pr_np = None
    elif isinstance(probs, torch.Tensor):
        pr_np = probs.detach().cpu().numpy()
    else:
        pr_np = np.array(probs)

    # NaN/Inf 체크: 모델 수치 불안정 시 발생 → None으로 처리
    if pr_np is not None:
        if not np.isfinite(pr_np).all():
            import warnings
            nan_pct = (~np.isfinite(pr_np)).mean() * 100
            warnings.warn(f"[{split}] probs에 NaN/Inf {nan_pct:.1f}% → auroc/logloss=nan")
            pr_np = None
        elif pr_np.ndim == 2:
            # [버그 수정] 이 재정규화는 (N, C) 형태(각 행이 클래스별 확률
            # 분포, 합이 1이어야 함)에만 적용돼야 함. 원래 코드가 ndim
            # 체크 없이 무조건 axis=-1로 sum했는데, binclass에서 probs를
            # 1차원 배열(P(class=1) 스칼라, shape=(N,))로 넘기면 axis=-1이
            # 전체 배열 하나를 가리켜서 "행 합"이 아니라 "전체 N개 확률의
            # 합" 하나로 나눠버림 — 모든 확률이 대략 1/N 배로 쪼그라들어
            # logloss가 수십 배 폭증하는 원인이 됐음(reproduce.py의 binary
            # ablation 분기에서 실측: 정상 logloss 0.83이 이 버그 때문에
            # 3.02로 나옴). 1차원 배열은 이미 각 원소가 그 자체로 유효한
            # 확률(sigmoid 출력)이라 재정규화 자체가 필요 없음 — 아래
            # else 분기에서 [0,1] 범위로 clip만 하고 넘어감(부동소수점
            # 오차로 아주 살짝 벗어나는 경우 대비).
            row_sum = pr_np.sum(axis=-1, keepdims=True)
            pr_np = pr_np / np.where(row_sum > 0, row_sum, 1.0)
        else:
            pr_np = np.clip(pr_np, 0.0, 1.0)

    metrics: Dict[str, float] = {}

    # ── Regression ───────────────────────────────────────────
    if tasktype == "regression":
        metrics[f"rmse_{split}"] = float(np.sqrt(np.mean((y_np - p_np) ** 2)))
        return metrics

    # ── Classification 공통 ──────────────────────────────────
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        roc_auc_score, f1_score, log_loss,
    )

    # Accuracy
    metrics[f"acc_{split}"]  = float(accuracy_score(y_np, p_np))

    # Balanced Accuracy
    # ⚠ MultiTab 에는 없는 지표다(위 docstring 참조). baseline 과 비교하지
    #   말고 TabERA 내부 분석에만 쓸 것.
    try:
        metrics[f"bacc_{split}"] = float(balanced_accuracy_score(y_np, p_np))
    except Exception:
        metrics[f"bacc_{split}"] = metrics[f"acc_{split}"]

    # AUROC
    try:
        if pr_np is not None:
            if tasktype == "binclass":
                prob_pos = pr_np[:, 1] if pr_np.ndim == 2 else pr_np
                metrics[f"auroc_{split}"] = float(roc_auc_score(y_np, prob_pos))
            else:
                # y에 실제 등장한 클래스만 선택 + 재정규화
                # (일부 클래스가 split에 없을 때 probs 열 수와 불일치 방지)
                present = sorted(np.unique(y_np).tolist())
                pr_sub  = pr_np[:, present]
                pr_sub  = pr_sub / pr_sub.sum(axis=1, keepdims=True).clip(1e-8)
                if len(present) == 2:
                    # [수정] subsetting 결과 클래스가 2개만 남으면
                    # roc_auc_score 는 multi_class 인자를 무시하고 **binary
                    # 경로로 라우팅**한다(type_of_target(y_true)=='binary').
                    # binary 경로는 1차원 점수를 기대하므로 (N,2) 배열을 주면
                    #     ValueError: y should be a 1d array, got (N, 2)
                    # 가 나고, 아래 except 가 이를 삼켜 auroc 가 통째로 nan 이
                    # 된다. 모델 문제가 아니라 지표 계산 경로 문제다.
                    #
                    # 실측(lymph, id=10): 클래스 분포 [2, 81, 61, 4] 에 fold
                    # 크기 14 이므로 fold 의 55%에서 클래스가 2개 이하만
                    # 등장한다 -- 비층화 KFold 를 쓰는 한 구조적으로 발생한다.
                    # 여기서 나가떨어지면 그 데이터셋의 AUROC 가 사실상 전부
                    # 사라지므로, 남은 두 클래스에 대한 이진 AUROC 로 계산한다.
                    #
                    # ⚠ 이 값은 "K개 클래스 중 2개만 등장한 split 에서의 이진
                    #   AUROC" 다. 다른 fold 의 macro-OVR 값과 같은 축에 있지
                    #   않으므로, 이런 fold 가 많은 데이터셋(lymph)의 AUROC 는
                    #   평균 내어 해석하지 말 것.
                    metrics[f"auroc_{split}"] = float(
                        roc_auc_score(y_np, pr_sub[:, 1])
                    )
                else:
                    metrics[f"auroc_{split}"] = float(
                        roc_auc_score(
                            y_np, pr_sub,
                            multi_class="ovr", average="macro",
                            labels=present,
                        )
                    )
        else:
            metrics[f"auroc_{split}"] = float("nan")
    except Exception as e:
        metrics[f"auroc_{split}"] = float("nan")

    # F1 (macro)
    try:
        metrics[f"f1_{split}"] = float(
            f1_score(y_np, p_np, average="macro", zero_division=0)
        )
    except Exception:
        metrics[f"f1_{split}"] = float("nan")

    # Log-loss
    try:
        if pr_np is not None:
            n_prob_classes = pr_np.shape[1] if pr_np.ndim == 2 else 2
            labels         = list(range(n_prob_classes))
            metrics[f"logloss_{split}"] = float(
                log_loss(y_np, pr_np, labels=labels)
            )
        else:
            metrics[f"logloss_{split}"] = float("nan")
    except Exception as e:
        metrics[f"logloss_{split}"] = float("nan")

    # ─────────────────────────────────────────────────────────
    # MultiTab 호환 지표 (비교 전용)
    # ─────────────────────────────────────────────────────────
    #
    # 위의 f1_/logloss_/auroc_ 는 **올바른 값**이다. 논문 본문에서 TabERA
    # 자신의 수치를 말할 때는 그쪽을 쓴다. 아래 _mt_ 접미사 키는 MultiTab이
    # 실제로 계산하는 방식을 그대로 재현한 값으로, 오직 baseline과 같은 축
    # 위에 올려놓기 위한 것이다.
    #
    # 왜 필요한가 (MultiTab libs/eval.py 확인 결과):
    #
    #  (1) F1 — MultiTab은 binclass에서 average='binary'(양성 클래스만),
    #      multiclass에서 average='weighted'를 쓴다. 여기는 둘 다 'macro'다.
    #      불균형 이진에서 차이가 크다: profb(양성 33%)에서 macro 0.601 vs
    #      binary 0.262, credit-g(양성=다수 70%)에서는 반대로 macro 0.649 vs
    #      binary 0.846. 정의만으로 최대 0.34가 갈리므로 그대로 비교할 수 없다.
    #      (로그의 acc/bacc/f1 삼중항에서 클래스 사전확률을 역산해 확인:
    #       이진 19개 전부 macro로 재현되며 잔차 1e-4 수준.)
    #
    #  (2) LogLoss/AUROC — MultiTab optimize.py:88은 model.predict_proba()의
    #      결과(=이미 확률)를 calculate_metric(..., prob=False)로 넘기고,
    #      MultiTab eval.py가 거기에 expit()/softmax()를 **한 번 더** 적용한다.
    #      multiclass에서는 이 변환이 14개 baseline 전부에 예외 없이 걸린다.
    #      이중 softmax의 이론 하한 ln(1+(K-1)/e)와 로그의 실측 최솟값을
    #      대조하면 r = 0.9986 (balance scale K=3: 0.551 vs 0.552,
    #      cnae-9 K=9: 1.372 vs 1.373, 100-plants K=100: 3.622 vs 3.722).
    #      즉 모델별로 다른 게 아니라 하나의 결정적 함수이므로, 우리 확률에
    #      같은 변환을 걸면 multiclass log loss와 AUROC가 비교 가능해진다.
    #
    #      ⚠ binclass는 재현하지 않는다. 이진에서는 변환 결과가 모델 계열마다
    #        다르다(test logloss 하한 실측: 트리 0.313 / TabM 0.419 /
    #        SAINT 0.404 / ModernNCA 0.685 / MLP·ResNet·EmbedMLP·MLP-PLR·
    #        T2G 0.000). 공통 함수가 아니므로 재현해도 일부 모델과만 맞는다.
    #        이진 log loss는 변환이 걸리지 않은 5개 모델과만 비교하고,
    #        논문에는 각주로 그 사실을 밝힌다.
    #
    #      ⚠ MultiTab reproduce.py:159는 predict_proba(..., logit=True)로
    #        불러서 정상이다. 즉 이 왜곡은 **튜닝 로그에만** 존재하며,
    #        MultiTab 논문 수치 자체는 멀쩡하다. baseline을 reproduce.py로
    #        다시 돌릴 수 있게 되면 _mt_ 키는 더 이상 필요 없다.
    #
    # ⚠ _mt_ 키를 model selection이나 objective에 쓰지 말 것. 의도적으로
    #   왜곡된 값이며 오직 비교표를 채우기 위한 것이다.
    # ⚠ 이 블록은 반드시 dict의 맨 뒤에 있어야 한다 — compute_metric()의
    #   "첫 키 = acc_val" 계약을 건드리지 않기 위해서다.
    try:
        mt_average = "binary" if tasktype == "binclass" else "weighted"
        metrics[f"f1_mt_{split}"] = float(
            f1_score(y_np, p_np, average=mt_average, zero_division=0)
        )
    except Exception:
        metrics[f"f1_mt_{split}"] = float("nan")

    if tasktype == "multiclass" and pr_np is not None and pr_np.ndim == 2:
        from scipy.special import softmax as _softmax
        pr_mt = _softmax(pr_np, axis=1)
        try:
            metrics[f"logloss_mt_{split}"] = float(
                log_loss(y_np, pr_mt, labels=list(range(pr_np.shape[1])))
            )
        except Exception:
            metrics[f"logloss_mt_{split}"] = float("nan")
        try:
            # MultiTab calculate_multi_auroc()와 순서를 맞춘다: softmax를 먼저
            # 걸고, split에 없는 클래스가 있으면 그때 열을 골라 재정규화한다.
            #
            # ⚠ 순서가 중요하다. softmax는 행 전체를 보고 정규화하므로
            #   "subset 후 softmax"와 "softmax 후 subset"은 다른 값이 된다.
            #   MultiTab optimize.py는 eval.py 진입 시점에 이미 softmax를
            #   적용하므로 subsetting은 그 뒤에 온다.
            #
            # ⚠ 참고: MultiTab **원본** calculate_multi_auroc()에는 이
            #   subsetting이 없어서 클래스가 빠진 split에서는 ValueError ->
            #   None을 반환한다. 지금 배포된 baseline optim_logs가 그 상태이며
            #   lymph에서 test AUROC의 19.9%가 NaN인 이유다. 즉 이 키를 기존
            #   로그와 맞댈 때는 baseline 쪽 결측을 그대로 두고 짝지어야 한다
            #   (수정된 MultiTab eval.py로 baseline을 다시 돌리면 양쪽이
            #    동일한 규약이 된다).
            # ⚠ 여기는 auroc_{split} 과 달리 2-클래스 폴백을 **넣지 않는다**.
            #   MultiTab 쪽도 같은 ValueError 를 맞고 None 을 반환하므로,
            #   폴백을 넣으면 baseline 이 결측인 자리에 우리만 값이 생겨
            #   "같은 규약" 이라는 이 키의 존재 이유가 무너진다. 결측은 결측
            #   대로 두고 pairwise 비교에서 함께 빠지는 것이 맞다.
            #   (TabERA 자신의 AUROC 는 위 auroc_{split} 을 쓴다.)
            present = sorted(np.unique(y_np).tolist())
            pr_mt_sub = pr_mt
            if len(present) < pr_mt.shape[1]:
                pr_mt_sub = pr_mt[:, present]
                _rs = pr_mt_sub.sum(axis=1, keepdims=True)
                pr_mt_sub = pr_mt_sub / np.where(_rs > 0, _rs, 1.0)
            metrics[f"auroc_mt_{split}"] = float(
                roc_auc_score(y_np, pr_mt_sub, multi_class="ovr",
                              average="macro", labels=present)
            )
        except Exception:
            metrics[f"auroc_mt_{split}"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────
# 내부 학습 루프용 단순 버전
# ─────────────────────────────────────────────────────────────

def compute_metric(
    logits: torch.Tensor,
    y: torch.Tensor,
    tasktype: str,
    full: bool = True,
) -> Dict[str, float]:
    """학습 루프에서 epoch마다 호출된다.

    [2026-08] 예전에는 accuracy(회귀는 rmse) 하나만 계산했다. 그 값이
    checkpoint selection에 쓰이는 값이라 그것만 있으면 충분해 보였지만,
    "AUROC로 골랐다면 어느 epoch이 뽑혔고 그때 prototype 파티션은 어떤
    상태였나"를 묻는 순간 지표당 재학습이 한 번씩 필요해진다. ds=46에서
    accuracy로 뽑힌 epoch 13과 더 늦게 뽑힌 epoch 148은 test logloss가
    0.42 대 0.14로 갈리고 dead_ratio도 56%와 22%로 갈린다 — 어떤 기준으로
    고르느냐가 질적으로 다른 모델을 선택한다. 그래서 전체 지표를 매 epoch
    기록해 두고 비교는 사후에 한 번의 실행으로 끝낸다.

    ⚠ 반환 dict의 **첫 번째 키는 그대로 acc_val / rmse_val**이다. 호출부가
      list(val_m.values())[0]으로 selection 값을 집으므로 순서가 바뀌면
      선택 기준 자체가 조용히 바뀐다. calculate_metric도 같은 순서로
      만들지만, 이 계약이 깨지지 않도록 아래에서 다시 첫 키로 세운다.
      (MultiTab 호환 지표 _mt_ 는 calculate_metric의 맨 뒤에 붙으므로 이
       계약에 영향이 없다.)

    full=False면 예전 동작(지표 하나)으로 돌아간다 — sklearn 호출이 epoch마다
    부담되는 상황을 위한 탈출구.
    """
    with torch.no_grad():
        if tasktype == "regression":
            preds = logits.squeeze(-1)
            rmse  = torch.sqrt(nn.MSELoss()(preds, y.float())).item()
            return {"rmse_val": rmse}
        if not full:
            if tasktype == "binclass":
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                return {"acc_val": (preds == y.float()).float().mean().item()}
            preds = logits.argmax(dim=-1)
            return {"acc_val": (preds == y).float().mean().item()}

        # 빠른 경로로 계산한 accuracy를 첫 키로 먼저 넣는다. 이렇게 하면
        # calculate_metric 쪽 키 순서가 바뀌더라도 selection 값은 안 바뀐다.
        if tasktype == "binclass":
            _p = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            out: Dict[str, float] = {"acc_val": (_p == y.float()).float().mean().item()}
        else:
            _p = logits.argmax(dim=-1)
            out = {"acc_val": (_p == y).float().mean().item()}
        try:
            preds, probs = get_preds_and_probs(logits, tasktype)
            for k, v in calculate_metric(y, preds, probs, tasktype, "val").items():
                if k != "acc_val":
                    out[k] = v
        except Exception:
            # 진단용 부가 지표 때문에 학습이 멈추면 안 된다.
            pass
        return out


def get_preds_and_probs(logits: torch.Tensor, tasktype: str):
    with torch.no_grad():
        if tasktype == "regression":
            return logits.squeeze(-1), None
        elif tasktype == "binclass":
            probs_pos = torch.sigmoid(logits.squeeze(-1))
            probs = torch.stack([1 - probs_pos, probs_pos], dim=-1)
            preds = (probs_pos > 0.5).long()
            return preds, probs
        else:
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            return preds, probs


# ─────────────────────────────────────────────────────────────
# Optuna study 재개 판단
# ─────────────────────────────────────────────────────────────

def is_study_todo(study: optuna.Study, tasktype: str) -> bool:
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) == 0:
        return True
    if tasktype != "regression" and study.best_value >= 1.0:
        return False
    return True


def check_if_fname_exists_in_error(fname: str) -> bool:
    try:
        with open("error.log", "r") as f:
            if fname in f.read():
                print(f"[SKIP] {fname} found in error.log")
                return False
    except FileNotFoundError:
        pass
    return True


# ─────────────────────────────────────────────────────────────
# Ordinal Cross-Entropy Loss
# ─────────────────────────────────────────────────────────────

class OrdinalCrossEntropyLoss(nn.Module):
    """
    순서형 분류를 위한 Ordinal Cross-Entropy Loss.

    가설 맥락에서의 역할
    ─────────────────
    - Centroid가 데이터의 실제 순서 구조를 반영하도록 유도
    - OT Evidence의 인접 이웃 집중 현상과 정합적으로 작동
    - 인접 등급 오류(5→6)를 비인접 오류(5→9)보다 작게 처벌
      → Confidence 과잉 억제 → logloss 안정화

    수식
    ────
    weight_c = 1 / (|c - y| + 1)  for each class c
    soft_target[y]   = (1 - ε) + ε * weight_y / Σ weight_c
    soft_target[c≠y] =             ε * weight_c / Σ weight_c
    loss = -Σ soft_target * log(softmax(logits))

    ε=0 이면 일반 CE, ε=1 이면 완전 순서형 smoothing.
    기본값 ε=0.1: CE의 안정적 gradient를 유지하면서
    인접 클래스에 소량의 확률 질량을 나눠줌.
    """

    def __init__(self, n_classes: int, epsilon: float = 0.1) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.epsilon   = epsilon

        # 클래스 간 거리 기반 가중치 행렬 (C, C) — 미리 계산
        # weight[y, c] = 1 / (|y - c| + 1)
        idx = torch.arange(n_classes).float()
        dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))  # (C, C)
        self.register_buffer('weight_matrix', 1.0 / (dist + 1.0))

    def forward(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        B, C = logits.shape
        device = logits.device

        # 각 샘플의 정답 클래스에 대한 거리 가중치 (B, C)
        w = self.weight_matrix[targets.long()]  # (B, C)
        w_sum = w.sum(dim=-1, keepdim=True)     # (B, 1)
        w_norm = w / w_sum                       # (B, C) 정규화

        # Soft target:
        # 정답 위치: (1 - ε) + ε * w_norm[y]
        # 나머지:           ε * w_norm[c]
        one_hot = torch.zeros(B, C, device=device)
        one_hot.scatter_(1, targets.long().unsqueeze(1), 1.0)
        soft_target = (1.0 - self.epsilon) * one_hot + self.epsilon * w_norm

        # log-softmax와 내적
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, C)
        loss = -(soft_target * log_prob).sum(dim=-1).mean()
        return loss


# ─────────────────────────────────────────────────────────────
# 손실 함수 / 비교 유틸
# ─────────────────────────────────────────────────────────────

# tasktype별 n_classes 캐시 (OrdinalCrossEntropyLoss 생성용)
_criterion_cache: dict = {}

def get_criterion(tasktype: str) -> nn.Module:
    """
    tabular deep learning 표준 손실 함수.
    TabR, ModernNCA와 동일한 설정.

    multiclass → CrossEntropyLoss
    binclass   → BCEWithLogitsLoss
    regression → MSELoss
    """
    if tasktype == "regression":
        return nn.MSELoss()
    elif tasktype == "binclass":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss()


def is_better(new_val: float, old_val: Optional[float], tasktype: str) -> bool:
    if old_val is None:
        return True
    return new_val < old_val if tasktype == "regression" else new_val > old_val