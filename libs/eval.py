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
    """MultiTab final-reproduction 규약으로 평가한다.

    Classification에서는 비교용 metric set을 하나만 반환한다.

        acc_{split}      : accuracy
        auroc_{split}    : binary ROC-AUC / multiclass macro OVR ROC-AUC
        f1_{split}       : binary F1(pos_label=1) / multiclass weighted F1
        logloss_{split}  : sklearn log loss

    과거 ``*_mt_*`` 별도 키는 사용하지 않는다. 위 네 키 자체가 MultiTab
    benchmark 정의다.

    ``probs``는 TabERA의 sigmoid/softmax가 이미 한 번 적용된 최종 확률이다.
    MultiTab reproduce.py는 raw logits를 eval.py에 넘기고 그곳에서 sigmoid/
    softmax를 한 번 적용하므로, 여기서 다시 변환하지 않는 것이 final
    reproduction metric과 동등하다.

    Multiclass AUROC는 split에 없는 class를 임의로 제거하지 않는다. MultiTab
    data.py처럼 전체 class 차원의 one-hot target을 재구성하여 macro OVR을
    계산한다. 어떤 class가 fold에 전혀 없어서 AUROC가 정의되지 않으면 NaN을
    그대로 남긴다(MultiTab 원본의 ``None``과 같은 결측 의미; TabERA 저장
    경로가 ``float(v)``를 사용하므로 NaN으로 표현).

    주의: balanced accuracy는 MultiTab benchmark metric이 아니다. 필요하면
    ``compute_metric()``이 학습/진단용 ``bacc_val``만 별도로 계산한다.
    """
    y_np = (y_true.detach().cpu().numpy()
            if isinstance(y_true, torch.Tensor) else np.asarray(y_true))
    p_np = (preds.detach().cpu().numpy()
            if isinstance(preds, torch.Tensor) else np.asarray(preds))
    pr_np = (None if probs is None else
             (probs.detach().cpu().numpy()
              if isinstance(probs, torch.Tensor) else np.asarray(probs)))

    y_np = np.asarray(y_np)
    p_np = np.nan_to_num(np.asarray(p_np))

    metrics: Dict[str, float] = {}

    # ── Regression ───────────────────────────────────────────
    if tasktype == "regression":
        metrics[f"rmse_{split}"] = float(np.sqrt(np.mean((y_np - p_np) ** 2)))
        return metrics

    from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

    # TabERA classification labels are integer class indices after data loading.
    y_cls = np.asarray(y_np).reshape(-1).astype(np.int64, copy=False)
    pred_cls = np.asarray(p_np).reshape(-1).astype(np.int64, copy=False)

    # Keep acc first. Some legacy training code assumes the first metric is accuracy.
    metrics[f"acc_{split}"] = float(accuracy_score(y_cls, pred_cls))

    valid_probs = (
        pr_np is not None
        and np.asarray(pr_np).size > 0
        and np.isfinite(np.asarray(pr_np)).all()
    )
    if valid_probs:
        pr_np = np.asarray(pr_np)

    # ── Binary classification ────────────────────────────────
    if tasktype == "binclass":
        # MultiTab: f1_score(..., average='binary'), positive label is 1.
        try:
            f1 = float(f1_score(
                y_cls, pred_cls,
                average="binary", pos_label=1, zero_division=0,
            ))
        except Exception:
            f1 = float("nan")

        if valid_probs:
            prob_pos = pr_np[:, 1] if pr_np.ndim == 2 else pr_np.reshape(-1)
            try:
                auroc = float(roc_auc_score(y_cls, prob_pos))
            except ValueError:
                auroc = float("nan")
            try:
                # Exactly the MultiTab binary label set.
                ll = float(log_loss(y_cls, pr_np, labels=[0, 1]))
            except ValueError:
                ll = float("nan")
        else:
            auroc = float("nan")
            ll = float("nan")

        # Match MultiTab key order: acc, auroc, f1, logloss.
        metrics[f"auroc_{split}"] = auroc
        metrics[f"f1_{split}"] = f1
        metrics[f"logloss_{split}"] = ll
        return metrics

    # ── Multiclass classification ────────────────────────────
    try:
        f1 = float(f1_score(
            y_cls, pred_cls, average="weighted", zero_division=0,
        ))
    except Exception:
        f1 = float("nan")

    if (not valid_probs) or pr_np.ndim != 2:
        auroc = float("nan")
        ll = float("nan")
    else:
        n_classes = int(pr_np.shape[1])
        if np.any(y_cls < 0) or np.any(y_cls >= n_classes):
            auroc = float("nan")
            ll = float("nan")
        else:
            # MultiTab one_hot(y) is performed before fold indexing, so a class
            # absent from this split still occupies an all-zero target column.
            y_onehot = np.eye(n_classes)[y_cls]

            # MultiTab original returns an undefined AUROC when a fold omits
            # any class. Detect it explicitly so the result is stable across
            # scikit-learn versions (older versions raise ValueError; newer
            # versions may emit a warning and return NaN).
            if np.unique(y_cls).size < n_classes:
                auroc = float("nan")
            else:
                try:
                    auroc = float(roc_auc_score(
                        y_onehot,
                        pr_np,
                        average="macro",
                        multi_class="ovr",
                    ))
                except ValueError:
                    auroc = float("nan")

            try:
                ll = float(log_loss(y_onehot, pr_np))
            except ValueError:
                ll = float("nan")

    metrics[f"auroc_{split}"] = auroc
    metrics[f"f1_{split}"] = f1
    metrics[f"logloss_{split}"] = ll
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
      (calculate_metric의 acc/auroc/f1/logloss 자체가 MultiTab benchmark
       정의이며 별도 _mt_ 키는 없다.)

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

            # MultiTab benchmark에는 없는 TabERA 내부 진단값. calculate_metric()
            # 의 최종 Performance에는 넣지 않고 epoch diagnostics/선택 실험에서만
            # 유지한다. --early_stop_metric bacc의 기존 기능도 이 값에 의존한다.
            from sklearn.metrics import balanced_accuracy_score
            _y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
            _p_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.asarray(preds)
            out["bacc_val"] = float(balanced_accuracy_score(_y_np, _p_np))
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