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

                    # ── 안전장치 1: retrieve()의 다음 배치 메모리 요구량 추정 ──
                    # active_ratio 스트릭과 무관하게 독립적으로 체크한다.
                    # 이유: active_ratio가 9%→8%→11%처럼 매 epoch 미세하게
                    # 오르내리면 streak 카운터가 계속 리셋되어 아래 스트릭
                    # 조건을 영영 못 채울 수 있음 (실제로 이렇게 빠져나간 사례
                    # 확인됨).
                    #
                    # [정정] cache_sample_groups()의 (P, max_g) 캐시는 int64
                    # 인덱스만 담아 실제로는 작음(수십MB 수준) — OOM의 원인이
                    # 아니었음. 진짜 위험은 retrieve() 내부에서 이 max_g와
                    # embed_dim(D)을 곱한 배치별 텐서(keys_u 등)이며, D는
                    # trial마다 64~256으로 다름. "N_train 대비 비율" 같은
                    # 고정 임계값은 D를 반영 못 해 또 다른 임의의 숫자가 될
                    # 뿐이라, 대신 실제 남은 GPU 메모리와 직접 비교한다.
                    max_cluster_now = ema_stats.get("max_cluster_size", 0)
                    if (torch.cuda.is_available()
                            and str(self.device).startswith("cuda")
                            and max_cluster_now > 0):
                        try:
                            D = self.model.embed_dim
                            free_bytes, _ = torch.cuda.mem_get_info(self.device)
                            # retrieve()에서 이 큰 그룹이 배치에 걸리면 필요한
                            # 텐서(keys_u, sim_u, Q_pad 등 여러 개)를 대략적으로
                            # 추정 — U_pad 최소 8(라운딩 단위), 부수 텐서 포함
                            # 안전 마진으로 4배를 곱함 (정확한 수치가 아니라
                            # "이 정도 자릿수면 위험하다"는 대략적 판단용).
                            projected_bytes = 8 * max_cluster_now * D * 4 * 4
                            if projected_bytes > free_bytes * 0.7:
                                # [버그 수정] mem_get_info()는 "CUDA 드라이버에
                                # 반납된" 메모리만 보고함 — PyTorch가 이전 trial
                                # 에서 쓰고 내부 캐시에 쌓아둔(재사용 가능한) 메모리는
                                # "사용 중"으로 잘못 잡힘. 그 결과 한 번이라도 크게
                                # 메모리를 쓴 뒤로는 이후 모든(사실은 안전한) trial도
                                # "여유 0"으로 오판해 즉시 종료되는 버그가 있었음
                                # (실측: trial 2 이후 trial 3~14가 전부 epoch 1에서
                                # 즉시 종료됨, 실제로는 필요 메모리가 0.02~0.2GB뿐).
                                # → 위험해 보일 때만 empty_cache()로 캐시를 드라이버에
                                # 반납시켜 재확인 (매 epoch 호출 X, 오탐일 때만 1회).
                                torch.cuda.empty_cache()
                                free_bytes, _ = torch.cuda.mem_get_info(self.device)

                            if projected_bytes > free_bytes * 0.7:
                                tqdm.write(
                                    f"  [STOP] Runaway centroid collapse at epoch {epoch} "
                                    f"(max_cluster_size={int(max_cluster_now)}, D={D} → "
                                    f"다음 배치 예상 메모리 {projected_bytes/1e9:.2f}GB "
                                    f"vs 남은 GPU 메모리 {free_bytes/1e9:.2f}GB, "
                                    f"empty_cache() 이후에도 부족). Early exit (OOM 방지)."
                                )
                                break
                        except Exception:
                            pass  # 메모리 조회 실패 시 이 안전장치만 건너뜀 (학습은 계속)

                    # ── 안전장치 2: active_ratio 지속 저하 감지 ──────────
                    # 단발성 dip(일시적으로 낮았다가 회복되는 경우)은 봐주고,
                    # "연속으로 계속 나쁜 상태가 유지"될 때만 중단한다.
                    # 매 에폭 체크(로그 출력 주기 10epoch과는 별개) — 추세가
                    # 뚜렷해지는 즉시 반응하기 위함. 그냥 낮은 값 한 번이
                    # 아니라 2회 연속(=최소 2 에폭 연속, 보통 10epoch 단위
                    # 로그 사이에도 매 에폭 계산되므로 실제로는 곧바로 반응)
                    # 유지될 때만 중단하여, 회복 가능한 dip을 살려둔다.
                    # (이것도 cache_sample_groups() 이전에 체크 — 어차피
                    # 중단할 거면 불필요한 텐서 할당을 피함)
                    active_ratio_now = ema_stats.get("active_ratio", 1.0)
                    if active_ratio_now < 0.1:
                        _low_active_streak += 1
                    else:
                        _low_active_streak = 0   # 회복하면 카운터 리셋

                    if _low_active_streak >= 5:
                        tqdm.write(
                            f"  [STOP] Runaway centroid collapse at epoch {epoch} "
                            f"(active={active_ratio_now:.0%}, {_low_active_streak}epoch 연속 "
                            f"10% 미만). Early exit."
                        )
                        break

                    self.model.memory.cache_sample_groups(
                        self.model.prototype_layer.sample_groups,
                        device=torch.device(self.device),
                        centroid_emb=self.model.prototype_layer.centroid_emb,
                    )

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