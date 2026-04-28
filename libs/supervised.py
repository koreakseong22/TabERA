"""
libs/supervised.py
==================
TabERA 학습 래퍼.

MultiTab의 supmodel 패턴을 참조하되,
TabERA 전용으로 재작성한 버전입니다.

  - TqdmLoggingHandler  : tqdm과 logging 충돌 방지 (MultiTab 원본 동일)
  - EarlyStopping       : val_loss 기준 조기 종료 (MultiTab 원본 동일)
  - TabERAWrapper      : TabERA용 fit/predict/predict_proba
                          (MultiTab supmodel 인터페이스와 동일)
"""

import math
import torch
import torch.nn as nn
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict

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
        """
        Returns True if training should stop.
        """
        if self.best_value is None:
            self.best_value = val_metric
            return False

        improved = (val_metric > self.best_value) if higher_is_better else (val_metric < self.best_value)
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
        self._data_id    = "?"      # tqdm 표시용 (optimize.py에서 주입)

    # ── fit ─────────────────────────────────────────────────

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> None:
        criterion  = get_criterion(self.tasktype)  # weight_matrix GPU 이동
        optimizer  = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        es         = EarlyStopping(patience=self.patience)

        # ── CentroidLayer 초기화 (이중 공간 설정) ─────────────
        if hasattr(self.model, 'prototype_layer') and hasattr(self.model.prototype_layer, 'initialize_from_data'):
            with torch.no_grad():
                n_init   = min(len(X_train), 5000)
                init_emb = self.model.embedder(X_train[:n_init])
                init_x   = X_train[:n_init]
                init_y   = y_train[:n_init]  # 소수 클래스 보장용
                self.model.prototype_layer.initialize_from_data(
                    init_emb, init_x, y_labels=init_y
                )
        higher_is_better = (self.tasktype != "regression")

        best_state = None
        best_val   = None

        # MultiTab 스타일 에폭 tqdm
        pbar = tqdm(
            range(1, self.epochs + 1),
            desc=f"EPOCH: 1",
            ncols=88,
            leave=True,
        )

        for epoch in pbar:
            # ── 학습 ──────────────────────────────────────
            self.model.train()
            perm    = torch.randperm(len(y_train), device=self.device)
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

            # ── (3) EMA centroid 업데이트 ───────────────────────
            if hasattr(self.model, 'prototype_layer') and hasattr(self.model.prototype_layer, 'ema_update'):
                with torch.no_grad():
                    # EMA: 전체 훈련 데이터 사용
                    # 샘플링하면 일부 centroid가 배정 못 받아 dead 발생
                    n_chunk = 1024  # GPU 메모리 고려 청크 처리
                    all_emb = []
                    for s in range(0, len(X_train), n_chunk):
                        all_emb.append(self.model.embedder(X_train[s:s+n_chunk]))
                    emb_ema  = torch.cat(all_emb, dim=0)
                    x_ema    = X_train
                    ema_stats = self.model.prototype_layer.ema_update(emb_ema, x_ema)

                    # 에폭당 1회: sample_groups를 GPU 텐서로 캐시 (76,800번 변환 제거)
                    self.model.memory.cache_sample_groups(
                        self.model.prototype_layer.sample_groups,
                        device=torch.device(self.device),
                    )
                    if epoch % 10 == 0:
                        # 검색 범위 축소율 계산 (가설 ②)
                        n_total = len(X_train)
                        n_proto = self.model.prototype_layer.P
                        avg_cand = n_total / n_proto if n_proto > 0 else n_total
                        reduction = (1 - avg_cand / n_total) * 100
                        pbar.write(
                            f"  [EMA] active={ema_stats['active_ratio']*100:.0f}%  "
                            f"alive={ema_stats.get('active_centroids', 0)}  "
                            f"pruned={ema_stats.get('pruned_this_epoch', 0)}  "
                            f"min={ema_stats['min_cluster_size']}  "
                            f"max={ema_stats['max_cluster_size']}  "
                        )

            avg_loss = tr_loss / max(n_batch, 1)

            # ── 검증 ──────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                val_logits = self._forward_batched(X_val)
                val_m  = compute_metric(val_logits, y_val, self.tasktype)
            val_v = list(val_m.values())[0]

            # best 모델 저장
            if is_better(val_v, best_val, self.tasktype):
                best_val   = val_v
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # tqdm postfix: dict 형태로 전달 → 터미널 너비 초과 시 자동 축약
            pbar.set_description(f"EPOCH: {epoch}")
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                id=self._data_id,
                refresh=False,
            )

            # 조기 종료
            if es.step(val_v, higher_is_better):
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

        pbar.close()

        if best_state:
            self.model.load_state_dict(best_state)
        self._best_state = best_state

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
