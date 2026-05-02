"""
libs/supervised.py
==================
TabERA 학습 래퍼.

MultiTab의 supmodel 패턴을 참조하되,
TabERA 전용으로 재작성한 버전입니다.

  - TqdmLoggingHandler  : tqdm과 logging 충돌 방지 (MultiTab 원본 동일)
  - EarlyStopping       : val_loss 기준 조기 종료 (MultiTab 원본 동일)
  - TabERAWrapper       : TabERA용 fit/predict/predict_proba
                          (MultiTab supmodel 인터페이스와 동일)

변경사항
────────
fit() 내부 criterion 생성 시 label_smoothing 전달:
    get_criterion(tasktype, label_smoothing=params.get("label_smoothing", 0.1))
    params.get("label_smoothing", 0.1) → 기존 study 하위 호환 보장

버그 수정
────────
ema_update 호출 시그니처를 prototypes.py 실제 구현에 맞게 수정:
    기존(잘못됨): ema_update(X_train, y_train, self.model.embedder)
    수정:        ema_update(emb_ema, X_train)
    → AttributeError: 'bool' object has no attribute 'sum' 해결
"""

import math
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List

from libs.eval import compute_metric, get_criterion, get_preds_and_probs, is_better
from libs.tabera import TabERA


# ─────────────────────────────────────────────────────────────
# TqdmLoggingHandler  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

class TqdmLoggingHandler(logging.StreamHandler):
    """tqdm 진행 바가 logger 출력에 의해 깨지지 않도록 방지."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────────────────────
# EarlyStopping  (MultiTab 원본과 동일한 인터페이스)
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 20):
        self.patience         = patience
        self.best_value       = None
        self.patience_counter = 0
        self.should_stop      = False

    def step(self, val_metric: float, higher_is_better: bool) -> bool:
        """Returns True if training should stop."""
        if self.best_value is None:
            self.best_value = val_metric
            return False

        improved = (
            (val_metric > self.best_value) if higher_is_better
            else (val_metric < self.best_value)
        )
        if improved:
            self.best_value       = val_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────
# TabERAWrapper  (MultiTab supmodel 인터페이스와 동일)
# ─────────────────────────────────────────────────────────────

class TabERAWrapper:
    """
    MultiTab의 supmodel과 동일한 fit / predict / predict_proba 인터페이스.

    Parameters
    ----------
    model    : TabERA 인스턴스
    params   : search_space에서 샘플링된 파라미터 dict
    tasktype : 'binclass' | 'multiclass' | 'regression'
    device   : 'cpu' or 'cuda:N'
    epochs   : 최대 에폭 수
    patience : 조기 종료 patience
    """

    def __init__(
        self,
        model: TabERA,
        params: dict,
        tasktype: str,
        device: str = "cpu",
        epochs: int = 100,
        patience: int = 20,
    ) -> None:
        self.model    = model.to(device)
        self.params   = params
        self.tasktype = tasktype
        self.device   = device
        self.epochs   = epochs
        self.patience = patience
        self._best_state = None
        self._data_id    = "?"
        self.ema_history: List[Dict[str, float]] = []
        self.final_ema_stats: Optional[Dict[str, float]] = None

    # ── fit ─────────────────────────────────────────────────

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> None:
        # [변경] label_smoothing을 params에서 꺼내 criterion에 전달
        # params.get("label_smoothing", 0.1): 기존 study(파라미터 없는 버전)
        # 하위 호환 보장 — KeyError 없이 기본값 0.1로 동작
        criterion = get_criterion(
            self.tasktype,
            label_smoothing=self.params.get("label_smoothing", 0.1),
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        es = EarlyStopping(patience=self.patience)

        # ── CentroidLayer 초기화 ──────────────────────────────
        if (hasattr(self.model, 'prototype_layer') and
                hasattr(self.model.prototype_layer, 'initialize_from_data')):
            with torch.no_grad():
                n_init   = min(len(X_train), 5000)
                init_emb = self.model.embedder(X_train[:n_init])
                init_x   = X_train[:n_init]
                init_y   = y_train[:n_init]
                self.model.prototype_layer.initialize_from_data(
                    init_emb, init_x, y_labels=init_y
                )

        higher_is_better     = (self.tasktype != "regression")
        best_state           = None
        best_val             = None
        self.ema_history     = []
        self.final_ema_stats = None

        pbar = tqdm(
            range(1, self.epochs + 1),
            desc=f"EPOCH: 1",
            ncols=88,
            leave=True,
        )

        for epoch in pbar:
            # ── 학습 ────────────────────────────────────────
            self.model.train()
            perm    = torch.randperm(len(y_train), device=X_train.device)
            tr_loss = 0.0
            n_batch = 0

            for start in range(0, len(y_train), self.params["batch_size"]):
                idx = perm[start:start + self.params["batch_size"]]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                out = self.model(xb, labels=yb)
                lg  = out["logits"]

                if self.tasktype in ("regression", "binclass"):
                    task_loss = criterion(lg.squeeze(-1), yb.float())
                else:
                    task_loss = criterion(lg, yb.long())

                loss = task_loss + out["aux_loss"]
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tr_loss += loss.item()
                n_batch += 1

            scheduler.step()
            self.model.anneal(self.params.get("anneal_factor", 0.97))

            # ── EMA centroid 업데이트 ─────────────────────────
            # ema_update(X_emb, X_raw, assignments=None) 시그니처
            # assignments=None → 내부에서 현재 centroid 기준 재배정
            if (hasattr(self.model, 'prototype_layer') and
                    hasattr(self.model.prototype_layer, 'ema_update')):
                with torch.no_grad():
                    # 전체 훈련 데이터 임베딩 계산 (청크 처리)
                    n_chunk = 1024
                    all_emb = []
                    for s in range(0, len(X_train), n_chunk):
                        all_emb.append(self.model.embedder(X_train[s:s + n_chunk]))
                    emb_ema = torch.cat(all_emb, dim=0)

                    ema_stats = self.model.prototype_layer.ema_update(
                        emb_ema,   # X_emb: (N, D) 임베딩
                        X_train,   # X_raw: (N, F) 원본 feature
                    )
                    self.final_ema_stats = dict(ema_stats)
                    self.ema_history.append({
                        "epoch":             float(epoch),
                        "active_ratio":      float(ema_stats.get("active_ratio", 0.0)),
                        "active_centroids":  float(ema_stats.get("active_centroids", 0.0)),
                        "pruned_this_epoch": float(ema_stats.get("pruned_this_epoch", 0.0)),
                        "min_cluster_size":  float(ema_stats.get("min_cluster_size", 0.0)),
                        "max_cluster_size":  float(ema_stats.get("max_cluster_size", 0.0)),
                    })

                    # sample_groups GPU 캐시 갱신
                    if hasattr(self.model, 'memory') and hasattr(
                            self.model.memory, 'cache_sample_groups'):
                        self.model.memory.cache_sample_groups(
                            self.model.prototype_layer.sample_groups,
                            device=torch.device(self.device),
                        )

                    if epoch % 10 == 0 or epoch == 1:
                        active_pct = ema_stats.get("active_ratio", 0)
                        alive      = ema_stats.get("active_centroids", 0)
                        min_count  = ema_stats.get("min_cluster_size", 0)
                        max_count  = ema_stats.get("max_cluster_size", 0)
                        tqdm.write(
                            f"  [EMA] active={active_pct:.0%}  "
                            f"alive={alive}  "
                            f"min={min_count}  max={max_count}"
                        )

            avg_loss = tr_loss / max(n_batch, 1)

            # ── Validation ───────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_logits = self._forward_batched(X_val)
                val_m      = compute_metric(val_logits, y_val, self.tasktype)

            val_v = list(val_m.values())[0]

            if is_better(val_v, best_val, self.tasktype):
                best_val   = val_v
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            pbar.set_description(f"EPOCH: {epoch}")
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                id=self._data_id,
                refresh=False,
            )

            if es.step(val_v, higher_is_better):
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

        pbar.close()

        if best_state:
            self.model.load_state_dict(best_state)
        self._best_state = best_state

        if self.ema_history:
            self.final_ema_stats = self.ema_history[-1]

    # ── predict ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """MultiTab: preds = model.predict(X)"""
        self.model.eval()
        logits = self._forward_batched(X)
        preds, _ = get_preds_and_probs(logits, self.tasktype)
        return preds

    # ── predict_proba ────────────────────────────────────────

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """MultiTab: probs = model.predict_proba(X)"""
        self.model.eval()
        logits = self._forward_batched(X)
        _, probs = get_preds_and_probs(logits, self.tasktype)
        return probs

    # ── 배치 추론 ────────────────────────────────────────────

    def _forward_batched(self, X: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        parts = []
        for start in range(0, len(X), batch_size):
            parts.append(self.model(X[start:start + batch_size])["logits"])
        return torch.cat(parts, dim=0)
