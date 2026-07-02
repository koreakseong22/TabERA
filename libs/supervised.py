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
        criterion  = get_criterion(self.tasktype)  # weight_matrix GPU 이동

        # [최적화] AdamW.step()이 프로파일에서 retrieve() 전체보다 더 큰
        # 단일 비용으로 확인됨 (파라미터 텐서별로 개별 커널을 발사하기 때문).
        # fused=True는 전체 업데이트를 커널 1개로 묶어 처리 (CUDA + PyTorch 2.0+).
        # 일부 dtype/파라미터 구성에서 미지원일 수 있어 실패 시 foreach로 폴백.
        try:
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params["lr"],
                weight_decay=self.params["weight_decay"],
                fused=(self.device.startswith("cuda")),
            )
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.params["lr"],
                weight_decay=self.params["weight_decay"],
                foreach=True,
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
        self.ema_history = []
        self.final_ema_stats = None
        _low_active_streak = 0   # 연속으로 active_ratio<10%인 EMA 체크 횟수 (runaway collapse 감지용)

        # [버그 수정] 이전엔 emb_cache로 X_train 전체(N_train개)를 캐시해서
        # ema_update에 넘겼음 → sample_groups가 "X_train 행 번호"로 만들어짐.
        # 그런데 MemoryBank는 별도의 링버퍼(크기 min(2*N_train, 10000))라서
        # 두 인덱스 공간이 어긋남 (N_train > memory_size인 경우 특히 심각 —
        # retrieve()에서 clamp로 인덱스가 뭉개짐). 아래에서 ema_update에
        # MemoryBank/FeatureStore의 실제 내용을 직접 넘기도록 수정 →
        # sample_groups가 처음부터 MemoryBank 인덱스 공간을 가리키게 됨.
        # 이에 따라 emb_cache/hook(구 PATCH ⑥)은 더 이상 필요 없어 제거.

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
            tr_loss_gpu = torch.zeros((), device=self.device)  # [최적화] GPU 텐서로 누적
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

                tr_loss_gpu += loss.detach()          # .item() 없이 GPU에 누적 (동기화 제거)
                n_batch += 1

            scheduler.step()
            self.model.anneal(self.params.get("anneal_factor", 0.97))

            # ── (3) EMA centroid 업데이트 ───────────────────────
            if hasattr(self.model, 'prototype_layer') and hasattr(self.model.prototype_layer, 'ema_update'):
                with torch.no_grad():
                    # [버그 수정] X_train 전체 대신 MemoryBank에 실제로 들어있는
                    # 내용(최대 memory_size개)만 클러스터링 → sample_groups가
                    # MemoryBank 인덱스 공간과 항상 일치하도록 보장.
                    n_mem = self.model.memory.filled.item()
                    if n_mem < 1:
                        # 메모리가 아직 하나도 안 채워진 극초반 → 스킵
                        ema_stats = {"active_ratio": 0.0, "min_cluster_size": 0, "max_cluster_size": 0}
                    else:
                        emb_ema = self.model.memory.keys[:n_mem]           # (n_mem, D) — MemoryBank 임베딩
                        fs = self.model.feature_store
                        x_ema = (
                            fs._store[:n_mem].to(self.device)              # (n_mem, F) — 원본 feature
                            if fs is not None else None
                        )
                        ema_stats = self.model.prototype_layer.ema_update(emb_ema, x_ema)

                    self.final_ema_stats = dict(ema_stats)
                    self.ema_history.append({
                        "epoch": float(epoch),
                        "active_ratio": float(ema_stats.get("active_ratio", 0.0)),
                        "active_centroids": float(ema_stats.get("active_centroids", 0.0)),
                        "pruned_this_epoch": float(ema_stats.get("pruned_this_epoch", 0.0)),
                        "min_cluster_size": float(ema_stats.get("min_cluster_size", 0.0)),
                        "max_cluster_size": float(ema_stats.get("max_cluster_size", 0.0)),
                    })

                    self.model.memory.cache_sample_groups(
                        self.model.prototype_layer.sample_groups,
                        device=torch.device(self.device),
                        centroid_emb=self.model.prototype_layer.centroid_emb,
                    )

                    # ── Runaway collapse 감지 ────────────────────────
                    # 단발성 dip(일시적으로 낮았다가 회복되는 경우)은 봐주고,
                    # "연속으로 계속 나쁜 상태가 유지"될 때만 중단한다.
                    # 매 에폭 체크(로그 출력 주기 10epoch과는 별개) — 추세가
                    # 뚜렷해지는 즉시 반응하기 위함. 그냥 낮은 값 한 번이
                    # 아니라 2회 연속(=최소 2 에폭 연속, 보통 10epoch 단위
                    # 로그 사이에도 매 에폭 계산되므로 실제로는 곧바로 반응)
                    # 유지될 때만 중단하여, 회복 가능한 dip을 살려둔다.
                    active_ratio_now = ema_stats.get("active_ratio", 1.0)
                    if active_ratio_now < 0.05:
                        _low_active_streak += 1
                    else:
                        _low_active_streak = 0   # 회복하면 카운터 리셋

                    if _low_active_streak >= 5:
                        tqdm.write(
                            f"  [STOP] Runaway centroid collapse at epoch {epoch} "
                            f"(active={active_ratio_now:.0%}, {_low_active_streak}epoch 연속 "
                            f"5% 미만). Early exit."
                        )
                        break

                    if epoch % 10 == 0:
                        pbar.write(
                            f"  [EMA] active={ema_stats['active_ratio']*100:.0f}%  "
                            f"alive={ema_stats.get('active_centroids', 0)}  "
                            f"min={ema_stats['min_cluster_size']}  "
                            f"max={ema_stats['max_cluster_size']}"
                        )

            avg_loss = (tr_loss_gpu / max(n_batch, 1)).item()  # [최적화] 에폭당 딱 1회만 동기화

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