"""
libs/eval.py
============
MultiTab 스타일 평가 지표.

calculate_metric : val/test split 별 dict 반환 (ModernNCA 동일 지표 포함)
compute_metric   : 내부 학습 루프용 단순 버전
is_study_todo    : 최적화 재개 여부 판단

변경사항
────────
get_criterion(): label_smoothing 파라미터 추가
    - multiclass: CrossEntropyLoss(label_smoothing=ε)
    - ε=0이면 기존과 동일 → 하위 호환 보장
    - 근거: Müller et al. (NeurIPS 2019) "When Does Label Smoothing Help?"
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
    ModernNCA와 동일한 지표 세트를 반환합니다.

    classification:
        acc_{split}, bacc_{split},
        auroc_{split}, f1_{split}, logloss_{split}
    regression:
        rmse_{split}
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
        else:
            # 행 합이 1이 되도록 재정규화 (float32 오차 보정)
            row_sum = pr_np.sum(axis=-1, keepdims=True)
            pr_np = pr_np / np.where(row_sum > 0, row_sum, 1.0)

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
                present = sorted(np.unique(y_np).tolist())
                pr_sub  = pr_np[:, present]
                pr_sub  = pr_sub / pr_sub.sum(axis=1, keepdims=True).clip(1e-8)
                metrics[f"auroc_{split}"] = float(
                    roc_auc_score(
                        y_np, pr_sub,
                        multi_class="ovr", average="macro",
                        labels=present,
                    )
                )
        else:
            metrics[f"auroc_{split}"] = float("nan")
    except Exception:
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
    except Exception:
        metrics[f"logloss_{split}"] = float("nan")

    return metrics


# ─────────────────────────────────────────────────────────────
# 내부 학습 루프용 단순 버전
# ─────────────────────────────────────────────────────────────

def compute_metric(
    logits: torch.Tensor,
    y: torch.Tensor,
    tasktype: str,
) -> Dict[str, float]:
    with torch.no_grad():
        if tasktype == "regression":
            preds = logits.squeeze(-1)
            rmse  = torch.sqrt(nn.MSELoss()(preds, y.float())).item()
            return {"rmse_val": rmse}
        elif tasktype == "binclass":
            preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
            acc   = (preds == y.float()).float().mean().item()
            return {"acc_val": acc}
        else:
            preds = logits.argmax(dim=-1)
            acc   = (preds == y).float().mean().item()
            return {"acc_val": acc}


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
# (리팩토링 시 제거 예정 — 현재 미사용)
# ─────────────────────────────────────────────────────────────

class OrdinalCrossEntropyLoss(nn.Module):
    """
    순서형 분류를 위한 Ordinal Cross-Entropy Loss.
    현재 미사용 (리팩토링 시 제거 예정).
    """

    def __init__(self, n_classes: int, epsilon: float = 0.1) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.epsilon   = epsilon

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

        w = self.weight_matrix[targets.long()]  # (B, C)
        w_sum = w.sum(dim=-1, keepdim=True)     # (B, 1)
        w_norm = w / w_sum                       # (B, C)

        one_hot = torch.zeros(B, C, device=device)
        one_hot.scatter_(1, targets.long().unsqueeze(1), 1.0)
        soft_target = (1.0 - self.epsilon) * one_hot + self.epsilon * w_norm

        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, C)
        loss = -(soft_target * log_prob).sum(dim=-1).mean()
        return loss


# ─────────────────────────────────────────────────────────────
# 손실 함수 / 비교 유틸
# ─────────────────────────────────────────────────────────────

_criterion_cache: dict = {}

def get_criterion(tasktype: str, label_smoothing: float = 0.1) -> nn.Module:
    """
    tabular deep learning 표준 손실 함수.

    변경사항: multiclass에 label_smoothing 추가
    ─────────────────────────────────────────────────────────
    근거: Müller et al. (NeurIPS 2019) "When Does Label Smoothing Help?"

    hard routing 구조에서 CrossEntropyLoss는 정답 클래스 확률을 1.0으로
    만드는 것을 목표로 학습합니다. 이것이 inference에서 구조적
    overconfidence를 만드는 원인입니다.

    label_smoothing=ε는 목표를 완화합니다:
        p(y_true)  → 1 - ε + ε/C
        p(y_other) →         ε/C

    ε=0이면 기존 CrossEntropyLoss와 완전히 동일 → 하위 호환 보장.
    순서형 가정 없이 어떤 분류 데이터에도 적용 가능 → 범용성 유지.

    multiclass → CrossEntropyLoss(label_smoothing=ε)
    binclass   → BCEWithLogitsLoss  (변경 없음)
    regression → MSELoss            (변경 없음)
    """
    if tasktype == "regression":
        return nn.MSELoss()
    elif tasktype == "binclass":
        return nn.BCEWithLogitsLoss()
    else:
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def is_better(new_val: float, old_val: Optional[float], tasktype: str) -> bool:
    if old_val is None:
        return True
    return new_val < old_val if tasktype == "regression" else new_val > old_val
