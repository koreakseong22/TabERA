"""
libs/supervised.py
==================
Training wrapper for TabERA.

Follows MultiTab's supmodel pattern, rewritten for this model.

  - TqdmLoggingHandler : keeps logging output from breaking the tqdm bar
                         (unchanged from upstream MultiTab)
  - EarlyStopping      : stops on the validation metric
                         (unchanged from upstream MultiTab)
  - TabERAWrapper      : fit / predict / predict_proba, the same interface
                         as MultiTab's supmodel
"""

import math
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, List

from libs.eval import compute_metric, get_criterion, get_preds_and_probs, is_better
from libs.tabera import TabERA
from libs.prototypes import label_all_groups, label_groups_by_target


# ─────────────────────────────────────────────────────────────
# TqdmLoggingHandler (unchanged from upstream MultiTab)
# ─────────────────────────────────────────────────────────────

class TqdmLoggingHandler(logging.StreamHandler):
    """Keep logger output from breaking the tqdm progress bar."""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────────────────────
# EarlyStopping (same interface as upstream MultiTab)
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
# TabERAWrapper (same interface as MultiTab's supmodel)
# ─────────────────────────────────────────────────────────────

class TabERAWrapper:
    """
    The same fit / predict / predict_proba interface as MultiTab's supmodel.

    Parameters
    ----------
    model    : a TabERA instance
    params   : parameter dict sampled from the search space
    tasktype : 'binclass' | 'multiclass' | 'regression'
    device   : 'cpu' or 'cuda:N'
    epochs   : maximum number of epochs
    patience : early-stopping patience
    """

    def __init__(
        self,
        model: TabERA,
        params: dict,
        tasktype: str,
        device: str = "cpu",
        epochs: int = 100,
        patience: int = 20,
        # ── Checkpoint-selection timing ────────────────────────────────
        # Measured problem this addresses (2026-08): on dataset 46 all five
        # low-utilisation runs reached active=50 at some epoch, yet their
        # best_epoch landed at 8-13 with active=14-24. The selected snapshot
        # came from the unstable phase the EMA codebook passes through before
        # it settles, and patience=20 then ended the run about 20 epochs
        # later, removing the chance to recover. The same interval that
        # produces a bad snapshot also cuts the run short -- two failures out
        # of one cause. credit-g (31) shows the same shape: best at epoch 9
        # with active=2, while that run later reached 28.
        #
        # defer_early_stopping gates BOTH best_state selection and the
        # patience counter on the same epoch, so patience measures "no
        # improvement since selection opened" instead of "no improvement
        # since epoch 1". Off by default: turning it on changes which
        # checkpoint every existing study returns, so old results stay
        # reproducible unless the flag is passed.
        #
        # ⚠ Not a fix for every low-utilisation run. Dataset 934 has runs
        #   whose assign_change_rate falls to 0.01-0.04 and whose active
        #   count decays after the best epoch with no later recovery. There
        #   the best epoch really is the best available and deferring
        #   selection picks something worse. Check that a run's trajectory
        #   max exceeds its active@best before expecting this to help.
        defer_early_stopping: bool = False,
        # Hard floor on when selection may open, independent of
        # regroup_warmup_epochs. 934 produced best_epoch=1 runs, which a
        # warmup gate alone does not exclude when warmup is short.
        min_epochs: int = 0,
        cat_cols: Optional[List[int]] = None,
        num_cols: Optional[List[int]] = None,
        col_names: Optional[List[str]] = None,
        cat_category_names: Optional[Dict[str, List[str]]] = None,
        target_class_names: Optional[List[str]] = None,
        quantile_transformer=None,
        # 0 disables the per-epoch regroup / refresh lines. reproduce.py
        # passes 0 unless --verbose is given: they describe how a run got
        # where it did, which matters while developing and not while using.
        regroup_log_every: int = 10,
        time_epoch: bool = False,
        log_beta: bool = False,
        beta_lr_mult: float = 1.0,
        refresh_on_best: bool = False,
        # Whether to call model.refresh_memory_keys() right after restoring
        # best_state. Off by default, so existing studies and best_params are
        # unaffected. Turning it on makes memory.keys a deterministic function
        # of the raw features; if the dropout noise had been contributing to
        # retrieval robustness, validation and test scores can move. Compare
        # A/B on a dataset before making it the default. Not in the HPO search
        # space.
    ) -> None:
        self.model    = model.to(device)
        self.params   = params
        self.tasktype = tasktype
        self.device   = device
        self.epochs   = epochs
        self.patience = patience
        self.defer_early_stopping = bool(defer_early_stopping)
        self.min_epochs = max(0, int(min_epochs))
        # Epoch at which best_state selection actually opened, so a run can be
        # audited afterwards without re-deriving it from the flags.
        self.selection_open_epoch = None
        # max(1, ...) would turn 0 into 1 and print every epoch, which is
        # the opposite of what 0 asks for. Clamp only negatives.
        self.regroup_log_every = max(0, int(regroup_log_every))
        # Per-phase epoch timing. Off by default: enabling it inserts CUDA syncs.
        self.time_epoch = bool(time_epoch)
        # Record the dev_beta_raw trajectory. Off by default: it syncs per batch.
        self.log_beta = bool(log_beta)
        # Learning-rate multiplier applied to dev_beta_raw only (1.0 = same lr)
        self.beta_lr_mult = float(beta_lr_mult)
        self._best_state = None
        self._data_id    = "?"      # shown in the tqdm bar; set by optimize.py
        self.regroup_history: List[Dict[str, float]] = []
        self.best_epoch: Optional[int] = None
        self.final_regroup_stats: Optional[Dict[str, float]] = None
        # Filled after fit(): z_top1, z_margin and related values, which the
        # objective in optimize.py can read alongside val_v. Stays None if the
        # model has no prototype_layer or the computation fails.
        self.centroid_geometry_diag: Optional[Dict[str, float]] = None
        # ── Group text labelling ────────────────────────────────
        # All three must be provided for labelling to run. If any is missing
        # (optimize.py, for instance, does not pass them) the step is skipped
        # and group_labels stays None. This is optional extra output, so
        # skipping it breaks nothing.
        self.cat_cols  = cat_cols
        self.num_cols  = num_cols
        self.col_names = col_names
        # {col_name: [original category strings, ...]}, as returned by
        # load_data(). Without it, label_all_groups() falls back to
        # "Category N".
        self.cat_category_names = cat_category_names
        # [original target label strings, ...], as returned by load_data(),
        # so groups show "good"/"bad" rather than "Class 0"/"Class 1".
        # Without it, label_groups_by_target() falls back to "Class N".
        self.target_class_names = target_class_names
        # The fitted QuantileTransformer returned by prep_data(). When
        # present, label_all_groups() maps numeric values back to their
        # original units instead of showing the [0,1] quantile.
        self.quantile_transformer = quantile_transformer
        self.refresh_on_best = refresh_on_best

    # ── fit ─────────────────────────────────────────────────

    def _resync_groups_after_refresh(self) -> Optional[Dict[str, float]]:
        """
        Called right after refresh_memory_keys().

        sample_groups was computed against the noisy memory.keys as they stood
        during training. refresh_memory_keys() replaces memory.keys entirely
        with the clean embeddings but leaves sample_groups alone, so the two
        stores end up describing different moments -- the same class of
        problem as the reinit-then-regroup issue documented inside
        regroup_update() in libs/prototypes.py.

        regroup_update() ignores X_raw (the parameter is kept for signature
        compatibility) and, even when a dead-prototype reinit happens,
        recomputes assignments against the final centroid_emb and overwrites
        sample_groups. Calling it once more on the clean embeddings therefore
        returns sample_groups, centroid_emb and memory.keys to a consistent
        state.

        ⚠ regroup_update() increments current_epoch and may reinitialise a
          prototype depending on its dead_streak. That is the same safeguard
          as during training and is fine in principle, but it does mean the
          final centroid_emb can differ slightly from the saved best_state --
          for the few prototypes that were reinitialised.
        """
        if not (hasattr(self.model, 'prototype_layer')
                and hasattr(self.model.prototype_layer, 'regroup_update')):
            return None

        with torch.no_grad():
            n_mem = self.model.memory.filled.item()
            if n_mem < 1:
                return None
            emb_regroup = self.model.memory.keys[:n_mem]   # the values just refreshed
            regroup_stats = self.model.prototype_layer.regroup_update(emb_regroup)

            # Refresh the GPU group cache that retrieve() reads as well.
            # Otherwise the freshly updated sample_groups and the cache built
            # by an earlier cache_sample_groups() describe different groups.
            self.model.memory.cache_sample_groups(
                self.model.prototype_layer.sample_groups,
                device=torch.device(self.device),
                centroid_emb=self.model.prototype_layer.centroid_emb,
            )

            # The text labels for explanation layer (1) are cached against
            # the old groups too, so they are recomputed from the new
            # sample_groups.
            fs = self.model.feature_store
            if (fs is not None and self.cat_cols is not None
                    and self.num_cols is not None and self.col_names is not None):
                x_regroup = fs._store[:n_mem].to(self.device)
                _xn = x_regroup.detach().cpu().numpy()
                # Guard: if the feature_store width disagrees with the
                # cat_cols / num_cols indices, this dies with an IndexError
                # (observed: index 5 out of bounds for size 5). group_labels
                # is explanation text, so there is no reason to abort the run;
                # out-of-range indices are skipped with a warning.
                _w = _xn.shape[1]
                _cat = [i for i in (self.cat_cols or []) if 0 <= i < _w]
                _num = [i for i in (self.num_cols or []) if 0 <= i < _w]
                _dropped = (len(self.cat_cols or []) - len(_cat)) + \
                           (len(self.num_cols or []) - len(_num))
                if _dropped:
                    tqdm.write(
                        f"  !  [resync] feature_store width is {_w}; skipping "
                        f"{_dropped} feature indices beyond it. Only "
                        f"group_labels (explanation text) becomes incomplete; "
                        f"numeric results are unaffected. Check whether the "
                        f"checkpoint and --openml_id refer to different "
                        f"datasets.")
                if _cat or _num:
                    self.model.prototype_layer.group_labels = label_all_groups(
                        _xn,
                        self.model.prototype_layer.sample_groups,
                        _cat, _num, self.col_names,
                        cat_category_names=self.cat_category_names,
                        quantile_transformer=self.quantile_transformer,
                    )
            y_regroup = self.model.memory.labels[:n_mem]
            self.model.prototype_layer.target_labels = label_groups_by_target(
                y_regroup.detach().cpu().numpy(),
                self.model.prototype_layer.sample_groups,
                self.tasktype,
                class_names=self.target_class_names,
            )
        return regroup_stats

    # ─────────────────────────────────────────────────────────
    # Per-phase epoch timing
    # ─────────────────────────────────────────────────────────
    # Why: training felt like "batches are fast but it pauses at the epoch
    # boundary". The likely cause is regroup_update and cache_sample_groups
    # scanning the whole dataset each epoch -- but optimising on a guess fixes
    # the wrong thing. Measure the split first, then decide.
    #
    # ⚠ CUDA is asynchronous, so timing without a sync pushes the cost into
    #   the next phase. The sync itself costs something, so it only runs under
    #   --time_epoch (off by default). Normal training speed is unaffected.
    def _t(self):
        if not getattr(self, "_timing_on", False):
            return None
        if self._timing_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    def _tick(self, key, t0):
        if t0 is None:
            return
        if self._timing_cuda:
            torch.cuda.synchronize()
        self._timing[key] = self._timing.get(key, 0.0) + (time.perf_counter() - t0)

    def _timing_report(self):
        if not getattr(self, "_timing_on", False) or not self._timing:
            return
        total = self._timing.get("epoch_total", 0.0)
        if total <= 0:
            return
        print(f"\n  [timing] cumulative epoch phases  (total {total:.1f}s, "
              f"{self._timing.get('n_epoch', 0):.0f} epoch)")
        # epoch_total is the denominator, so it is excluded from the list
        items = [(k, v) for k, v in self._timing.items()
                 if k not in ("epoch_total", "n_epoch")]
        for k, v in sorted(items, key=lambda x: -x[1]):
            print(f"    {k:<28}{v:>8.1f}s   {v / total:>6.1%}")
        _acc = sum(v for _, v in items)
        print(f"    {'(rest = batch training)':<28}{total - _acc:>8.1f}s   "
              f"{(total - _acc) / total:>6.1%}")

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        skip_centroid_init: bool = False,
        # When True, initialize_from_data() is skipped. Needed whenever
        # already-trained centroids must survive a re-fit -- otherwise
        # entering fit() overwrites them, so freezing the encoder would still
        # reinitialise the prototypes every time.
    ) -> None:
        criterion  = get_criterion(self.tasktype)

        # centroid_emb is not excluded from weight decay. It follows the
        # ArcFace / CosFace convention instead: a parameter used through a
        # normalisation is not exempted from decay but reprojected to norm 1
        # after every step. With that, weight decay can stay on as usual and
        # the centroid directions do not wander. Routing only ever uses
        # F.normalize(centroid_emb), so keeping the raw parameter on the unit
        # sphere is the consistent choice. The reprojection happens right
        # after optimizer.step() in the training loop below.
        centroid_param = self.model.prototype_layer.centroid_emb

        # Profiling showed AdamW.step() as a larger single cost than the whole
        # of retrieve(), because it launches one kernel per parameter tensor.
        # fused=True folds the update into a single kernel (CUDA, PyTorch 2.0+)
        # and falls back to foreach where a dtype or parameter layout does not
        # support it.
        #
        # ⚠ beta and gamma are excluded from weight decay. They are used as
        #   sigma(raw), so decaying raw toward 0 pulls the coefficient toward
        #   0.5 -- the opposite of "use only as much as needed". This is the
        #   same reason BatchNorm parameters and biases are excluded.
        _no_decay, _decay = [], []
        for _n, _p in self.model.named_parameters():
            if not _p.requires_grad:
                continue
            (_no_decay if _n.endswith(("beta_raw", "gamma_raw")) else _decay).append(_p)
        # ── beta_lr_mult: a learning-rate multiplier for dev_beta_raw ────
        # Why: beta = sigma(dev_beta_raw) was measured rising in **one
        #   direction only** for the whole run, then cut off by early stopping.
        #     ds=31    32 epochs, beta 0.100 -> 0.111, 0% sign flips,
        #              |signed| / |grad| = 1.000
        #              actual d(raw) 0.112 vs theoretical max
        #              (lr x batches x epochs) 0.110 -> ratio 1.02
        #     ds=1489  98 epochs, beta 0.109 -> 0.806, 0% sign flips
        #   AdamW normalises the gradient magnitude, so with a constant sign
        #   each step is approximately lr. A ratio of 1.02 means it climbed at
        #   the maximum possible rate the whole way, i.e. the final beta is a
        #   product of the optimisation budget rather than a converged value.
        #   The difference between datasets is explained by budget, not data:
        #     ds=31    lr 0.00086 x  4 batches x 32 ep =  110 step-lr -> 0.111
        #     ds=1489  lr 0.00579 x 17 batches x 98 ep = 9648 step-lr -> 0.806
        #
        # ⚠ Reparameterisation (softplus) does not fix it. sigma'(beta=0.1) is
        #   0.09, which is 36% of the maximum 0.25 -- not a dead region -- and
        #   switching to softplus only shortens the distance from 3.58 to 2.46
        #   (69%). The problem is the step budget relative to the distance, not
        #   saturation, so raising lr is the direct intervention.
        #
        # ⚠ The (0, 1) bound stays. centroid_emb is normalised so ||c|| = 1,
        #   which makes beta exactly ||beta * r|| / ||c||, and beta < 1 is the
        #   substantive constraint that the correction cannot exceed the
        #   prototype in magnitude. Removing the bound reopens the door that
        #   `W_c.c + lambda.W_q.q` went through, where ||W_q|| grew 24-fold to
        #   evade the constraint.
        #
        # ⚠ The default 1.0 leaves behaviour unchanged. This is an
        #   experimental flag.
        _beta_lr = self.params["lr"] * float(self.beta_lr_mult)
        _pg = [{"params": _decay, "weight_decay": self.params["weight_decay"]},
               {"params": _no_decay, "weight_decay": 0.0, "lr": _beta_lr}]
        if abs(float(self.beta_lr_mult) - 1.0) > 1e-9:
            print(f"  [beta_lr_mult] lr for dev_beta_raw/gamma_raw = "
                  f"{_beta_lr:.6f}  (base {self.params['lr']:.6f} x {self.beta_lr_mult})")
        try:
            optimizer = torch.optim.AdamW(
                _pg,
                lr=self.params["lr"],
                fused=(self.device.startswith("cuda")),
            )
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(
                _pg,
                lr=self.params["lr"],
                foreach=True,
            )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        es         = EarlyStopping(patience=self.patience)

        # ── Seed the prototype memory ─────────────────────────
        #
        # [2026-08] 이전에는 `min(len(X_train), 5000)` 으로 잘라
        # `X_train[:n_init]` 를 넘겼다. 두 가지가 문제였다.
        #
        # (1) 슬라이스는 무작위 표본이 아니다. libs/data.py 의 split 이
        #     `np.setdiff1d(tr_idx, val_idx)` 를 쓰는데 이 함수는 **정렬된**
        #     배열을 반환하므로, KFold(shuffle=True) 를 거쳤더라도 최종
        #     tr_idx 는 오름차순이고 X_train 은 원본 행 순서를 그대로
        #     유지한다. 따라서 X_train[:5000] 은 "원본 데이터의 앞부분"이다.
        #     electricity(id=151, N_train=36,250)에서 실측하면 원본 인덱스
        #     2~6331, 즉 전체의 앞쪽 14% 구간만 뽑힌다. ELEC2 는 시간순
        #     정렬된 시계열(concept drift 벤치마크의 표준 사례)이라, 190개
        #     프로토타입이 가장 이른 구간 하나에서만 초기화되고 있었다.
        #     첫 regroup 이 active=25%, alive=47/190, reinit=24 로 시작하던
        #     것과 부합한다.
        #
        # (2) 캡이 아끼는 비용이 없다. initialize_from_data() 는
        #     `torch.randperm(N)[:P]` 로 P개를 뽑아 복사할 뿐이다
        #     (k-means++ 는 이미 제거됨 -- libs/prototypes.py:554 의 ablation
        #     참조). 남는 비용은 임베더 forward 한 번인데, regroup_update()
        #     는 **매 epoch** 전체 학습 임베딩(memory_size = N_train)에 대해
        #     돌고 있다. 초기화 1회가 이미 92회 도는 작업보다 쌀 수 없다.
        #
        # 그래서 전체 X_train 을 넘긴다. 표본 추출은 initialize_from_data()
        # 안의 randperm 이 이미 담당하므로 여기서 섞을 필요가 없다.
        #
        # ⚠ 캡이 걸리지 않던 데이터셋(N_train <= 5000, 27개 중 22개)에서는
        #   넘기는 텐서가 이전과 완전히 동일하고 randperm 의 N 도 같으므로
        #   난수 소비량까지 같다 -- 결과가 비트 단위로 보존된다. 재실행이
        #   필요한 것은 캡이 걸리던 5개뿐이다(electricity, jungle_chess,
        #   nomao, elevators, artificial characters).
        #
        # ⚠ 임베더 forward 는 청크로 나눈다. nomao 는 118 feature 에 PLE 가
        #   붙어 한 번에 27,572 행을 태우면 peak memory 가 불필요하게 커진다.
        #   초기화는 no_grad 이므로 청크 경계가 결과를 바꾸지 않는다.
        if (not skip_centroid_init and hasattr(self.model, 'prototype_layer')
                and hasattr(self.model.prototype_layer, 'initialize_from_data')):
            with torch.no_grad():
                _CHUNK = 4096
                if len(X_train) <= _CHUNK:
                    init_emb = self.model.embedder(X_train)
                else:
                    init_emb = torch.cat(
                        [self.model.embedder(X_train[s:s + _CHUNK])
                         for s in range(0, len(X_train), _CHUNK)],
                        dim=0)
                # X_raw / y_labels 는 initialize_from_data 가 쓰지 않는다
                # (시그니처 호환용). 그래도 계약대로 넘겨 둔다.
                self.model.prototype_layer.initialize_from_data(
                    init_emb, X_train, y_labels=y_train
                )


        higher_is_better = (self.tasktype != "regression")

        best_state = None
        best_val   = None
        best_sample_groups = None
        best_feature_store = None
        best_group_labels  = None
        best_target_labels = None
        self.regroup_history = []
        self.final_regroup_stats = None
        # Routing-stability diagnostics across the whole run. Logged only, not
        # fed into any penalty: check the correlation first, decide later.
        # reinit_total counts every reinitialisation (fewer is more stable).
        # active_ratio_history collects the active_ratio from each
        # regroup_update so its std can be taken -- the instability of the
        # *process*, which centroid_geometry_diag cannot see because it only
        # looks at the final snapshot.
        _reinit_total = 0
        _active_ratio_history: List[float] = []
        # label_all_groups and label_groups_by_target now run inside the
        # is_better block, so their inputs have to survive to the end of the
        # epoch. Declared None up front because they may never be filled --
        # in the very first epochs where n_mem < 1, or for a model without a
        # prototype_layer -- which would otherwise raise NameError.
        x_regroup = None
        y_regroup = None

        # ⚠ regroup_update is handed the MemoryBank and FeatureStore contents
        # directly, so sample_groups indexes the MemoryBank space from the
        # start. Previously all of X_train was cached and passed instead,
        # which built sample_groups over X_train row numbers while MemoryBank
        # is a separate ring buffer -- two index spaces that disagree. The
        # mismatch was worst when N_train exceeded memory_size, where the
        # clamp in retrieve() silently folded the indices.

        # Epoch progress bar, MultiTab style
        pbar = tqdm(
            range(1, self.epochs + 1),
            desc=f"EPOCH: 1",
            ncols=88,
            leave=True,
        )

        # Timing state. Without --time_epoch every call below is a no-op.
        self.beta_history = []
        self._beta_grad_sum = 0.0
        self._beta_grad_signed = 0.0
        self._beta_grad_n = 0

        self._timing = {}
        self._timing_on = bool(getattr(self, "time_epoch", False))
        self._timing_cuda = self._timing_on and torch.cuda.is_available()

        for epoch in pbar:
            _t_epoch = self._t()
            # ── Train ─────────────────────────────────────
            self.model.train()
            perm    = torch.randperm(len(y_train), device=self.device)
            tr_loss_gpu = torch.zeros((), device=self.device)  # accumulate on GPU
            n_batch = 0

            for start in range(0, len(y_train), self.params["batch_size"]):
                idx = perm[start:start + self.params["batch_size"]]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()


                # idx is the X_train row number. Storing it as sample_ids
                # alongside the batch in both MemoryBank and FeatureStore lets
                # their slot correspondence be verified afterwards by exact
                # equality rather than by statistics.
                out = self.model(xb, labels=yb, sample_ids=idx)
                lg  = out["logits"]




                if self.tasktype in ("regression", "binclass"):
                    task_loss = criterion(lg.squeeze(-1), yb.float())
                else:
                    task_loss = criterion(lg, yb.long())

                loss = task_loss + out["aux_loss"]
                loss.backward()




                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # ── Measure dev_beta_raw and its gradient ─────────────
                # Why: on 7 datasets the final beta stays near its initial
                #   sigma(-2.197) = 0.100 (a factor of 1.02-1.23). Two
                #   readings fit that, and the final value alone cannot tell
                #   them apart:
                #     (a) equilibrium -- it did not rise because the
                #         correction was not needed
                #     (b) stalling -- no gradient flowed and it stayed stuck
                #         at the initial value
                #   If (b), then the initial value we chose decides the result,
                #   which is not defensible in the paper.
                # ⚠ Must be read **before** clip_grad_norm_ and
                #   optimizer.step(); afterwards the gradient is clipped or
                #   cleared.
                # ⚠ Off by default (--log_beta). Enabling it adds an .item()
                #   per batch and therefore a CUDA sync, so normal training
                #   is unaffected.
                if getattr(self, "log_beta", False):
                    _bp = getattr(self.model, "dev_beta_raw", None)
                    if _bp is not None and _bp.grad is not None:
                        self._beta_grad_sum += float(_bp.grad.abs().sum())
                        self._beta_grad_signed += float(_bp.grad.sum())
                        self._beta_grad_n += 1
                optimizer.step()
                # CosFace convention: reproject centroid_emb to unit norm
                # after every step. Weight decay then only affects direction,
                # since the magnitude is reset to 1 each time, so routing
                # cannot destabilise through a norm collapse.
                with torch.no_grad():
                    centroid_param.data = F.normalize(centroid_param.data, dim=-1)

                tr_loss_gpu += loss.detach()          # stays on GPU: no sync
                n_batch += 1

            scheduler.step()
            self.model.anneal(self.params.get("anneal_factor", 0.97))

            # ── (3) Refresh sample_groups (regroup_update) ─────────────
            # Skipped when skip_centroid_init is True: dead-prototype
            # recovery overwrites centroid_emb.data directly, without a
            # gradient, which would break a fully frozen condition.
            if (not skip_centroid_init and hasattr(self.model, 'prototype_layer')
                    and hasattr(self.model.prototype_layer, 'regroup_update')):
                with torch.no_grad():
                    # Cluster over what MemoryBank actually holds (at most
                    # memory_size entries) rather than all of X_train, so
                    # sample_groups always agrees with the MemoryBank index
                    # space.
                    n_mem = self.model.memory.filled.item()
                    if n_mem < 1:
                        # Nothing in memory yet: skip
                        regroup_stats = {"active_ratio": 0.0, "min_cluster_size": 0, "max_cluster_size": 0}
                        x_regroup = None
                        y_regroup = None
                    else:
                        emb_regroup = self.model.memory.keys[:n_mem]           # (n_mem, D)
                        fs = self.model.feature_store
                        _t0 = self._t()
                        x_regroup = (
                            fs._store[:n_mem].to(self.device)              # (n_mem, F) raw features
                            if fs is not None else None
                        )
                        self._tick("feature_store -> GPU copy", _t0)
                        _t0 = self._t()
                        # sample_ids makes the per-epoch partition comparable
                        # across epochs: MemoryBank slot i holds a different
                        # training row each epoch, so a slot-keyed comparison
                        # measures nothing. See the aligned-churn block in
                        # regroup_update.
                        _sids = getattr(self.model.memory, "sample_ids", None)
                        regroup_stats = self.model.prototype_layer.regroup_update(
                            emb_regroup, x_regroup,
                            sample_ids=(_sids[:n_mem] if _sids is not None else None))
                        self._tick("regroup_update", _t0)
                        # label_all_groups and label_groups_by_target are
                        # read-only text caching for explanations and do not
                        # affect weights or the early-stopping decision. They
                        # used to run every epoch, but only the values from an
                        # epoch where validation improved are ever used -- the
                        # best_* snapshot logic below keeps those and the rest
                        # are overwritten next epoch. So the text computation
                        # is deferred and only its inputs (x_regroup,
                        # y_regroup) are kept for the is_better block. On
                        # datasets with many columns and prototypes the text
                        # itself was measured at seconds to tens of seconds
                        # per epoch (nomao: 118 features, P=166).
                        y_regroup = self.model.memory.labels[:n_mem]

                    # Accumulate for the stability diagnostics. Epochs skipped
                    # by warmup record active_ratio = 0.0 as-is: that is the
                    # actual state before activation, so it is fact rather
                    # than a distortion of the std.
                    _reinit_total += regroup_stats.get("reinit_count", 0)
                    _active_ratio_history.append(regroup_stats.get("active_ratio", 0.0))

                    self.final_regroup_stats = dict(regroup_stats)
                    # ⚠ Copying a fixed list of keys silently loses any new
                    #   diagnostic regroup_update starts returning. That is
                    #   how reinit_count and the drift metrics disappeared,
                    #   leaving no way to tell whether dead-prototype recovery
                    #   had fired at all. Every scalar key is copied
                    #   automatically instead.
                    _rec = {"epoch": float(epoch)}
                    for _k, _v in regroup_stats.items():
                        if isinstance(_v, bool):
                            _rec[_k] = float(_v)
                        elif isinstance(_v, (int, float)):
                            _rec[_k] = float(_v)
                    self.regroup_history.append(_rec)

                    # ── Refresh retrieve()'s hybrid threshold from the actual
                    # free GPU memory each epoch, rather than from an
                    # unjustified constant. The query runs once per epoch here
                    # so that retrieve() itself -- called every batch -- never
                    # touches the GPU memory info.
                    if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                        try:
                            _free_b, _ = torch.cuda.mem_get_info(self.device)
                            self.model.memory.update_outlier_threshold(
                                n_prototypes=self.model.prototype_layer.P,
                                free_bytes=_free_b,
                            )
                        except Exception:
                            pass

                    # ── Guard 1: estimate retrieve()'s memory demand for the
                    # next batch. Checked independently of the active_ratio
                    # streak, because an active_ratio that drifts slightly
                    # every epoch (9% -> 8% -> 11%) keeps resetting the streak
                    # counter and the streak condition below may never be
                    # met -- a case that really did slip through.
                    #
                    # The (P, max_g) cache in cache_sample_groups() holds only
                    # int64 indices and is small (tens of MB); it was not the
                    # OOM cause. The real risk is the per-batch tensors inside
                    # retrieve() that multiply max_g by embed_dim, and D
                    # varies from 64 to 256 across trials. A fixed threshold
                    # like "a fraction of N_train" cannot reflect D and would
                    # just be another arbitrary number, so this compares
                    # against the actual free GPU memory instead.
                    max_cluster_now = regroup_stats.get("max_cluster_size", 0)
                    if (torch.cuda.is_available()
                            and str(self.device).startswith("cuda")
                            and max_cluster_now > 0):
                        try:
                            D = self.model.embed_dim
                            free_bytes, _ = torch.cuda.mem_get_info(self.device)
                            # Rough estimate of the tensors retrieve() needs
                            # (keys_u, sim_u, Q_pad and others) when a batch
                            # hits this large group. U_pad rounds up to at
                            # least 8, and a factor of 4 covers the auxiliary
                            # tensors as a safety margin. This is an
                            # order-of-magnitude judgement, not an exact
                            # figure.
                            projected_bytes = 8 * max_cluster_now * D * 4 * 4
                            if projected_bytes > free_bytes * 0.7:
                                # mem_get_info() reports only memory returned
                                # to the CUDA driver, so whatever PyTorch is
                                # holding in its own cache from a previous
                                # trial -- and could reuse -- counts as "in
                                # use". After any one memory-heavy trial, every
                                # later trial then read "zero free" and exited
                                # immediately, even though it was safe
                                # (measured: after trial 2, trials 3-14 all
                                # stopped in epoch 1 while actually needing
                                # 0.02-0.2 GB).
                                # So empty_cache() is called only when the
                                # situation already looks dangerous, to return
                                # the cache to the driver and re-check -- once,
                                # on a suspected false positive, not every
                                # epoch.
                                torch.cuda.empty_cache()
                                free_bytes, _ = torch.cuda.mem_get_info(self.device)

                            if projected_bytes > free_bytes * 0.7:
                                tqdm.write(
                                    f"  [STOP] Runaway centroid collapse at epoch {epoch} "
                                    f"(max_cluster_size={int(max_cluster_now)}, D={D} → "
                                    f"next batch needs about {projected_bytes/1e9:.2f}GB "
                                    f"vs {free_bytes/1e9:.2f}GB free on GPU, "
                                    f"still short after empty_cache()). Early exit to avoid OOM."
                                )
                                break
                        except Exception:
                            pass  # if the query fails, skip this guard only

                    # A second guard -- stop early after 5 consecutive epochs
                    # of falling active_ratio -- was removed when
                    # dead-prototype recovery arrived. A low active_ratio is
                    # no longer left alone but actively repaired, which
                    # undercuts the reason for detecting an "unrecoverable"
                    # state early. Its threshold (5 consecutive epochs) also
                    # matched dead_reinit_patience exactly, so it was measured
                    # cutting training off just as recovery was about to
                    # intervene. Guard 1 stays: preventing an OOM crash is
                    # separate from collapse.

                    _t0 = self._t()
                    self.model.memory.cache_sample_groups(
                        self.model.prototype_layer.sample_groups,
                        device=torch.device(self.device),
                        centroid_emb=self.model.prototype_layer.centroid_emb,
                    )
                    self._tick("cache_sample_groups", _t0)

                    if self.regroup_log_every and epoch % self.regroup_log_every == 0:
                        _reinit = regroup_stats.get('reinit_count', 0)
                        pbar.write(
                            f"  [Regroup] active={regroup_stats['active_ratio']*100:.0f}%  "
                            f"alive={regroup_stats.get('active_centroids', 0)}  "
                            f"min={regroup_stats['min_cluster_size']}  "
                            f"max={regroup_stats['max_cluster_size']}"
                            + (f"  reinit={_reinit}" if _reinit > 0 else "")
                        )

            avg_loss = (tr_loss_gpu / max(n_batch, 1)).item()  # one sync per epoch




            # ── Validation ────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                # Processing X_val in a fixed order means that when similar
                # samples cluster in one stretch, those batches route to the
                # same few centroids every epoch -- the worst case of small U
                # with a large local_max_g, repeated (measured: val_forward
                # blowing up from 1s to 76s). Training already averages this
                # out through randperm each epoch; validation being fixed was
                # part of the cause. Shuffling here averages it the same way,
                # and the aggregate metrics do not depend on order.
                _val_perm  = torch.randperm(len(X_val), device=self.device)
                val_logits = self._forward_batched(X_val[_val_perm])
                val_m  = compute_metric(val_logits, y_val[_val_perm], self.tasktype)
            val_v = list(val_m.values())[0]




            # During warmup, sample_groups has never been published
            # ([[], [], ...]). If validation happens to look good at that
            # point -- and it can, because retrieve() then behaves like an
            # unconstrained global search -- that group-less snapshot becomes
            # best_state. Later epochs fill sample_groups properly but are no
            # longer the best, and if they cannot beat that score within
            # patience the run stops with a final model frozen on empty groups
            # (measured on vehicle rwe5: early stop at epoch 33 with no
            # verifiable groups). regroup_update() has already run this epoch,
            # so whether current_epoch has passed warmup tells us whether
            # sample_groups was actually published, and such epochs are
            # excluded from the best_state candidates.
            _past_regroup_warmup = (
                self.model.prototype_layer.current_epoch.item()
                >= self.model.prototype_layer.regroup_warmup_epochs
            )

            # ⚠ Attach the validation score to this epoch's regroup record.
            #   Without it, active_ratio and val live on separate axes and the
            #   question "what was prototype utilisation at the epoch that got
            #   selected" cannot be answered at all. That question matters
            #   here: active_ratio was measured oscillating on a 5-epoch cycle
            #   (0.73 -> 0.10 -> 0.70 ...), matching dead_reinit_patience, so
            #   which phase early stopping lands in may decide the outcome.
            if self.regroup_history and self.regroup_history[-1].get("epoch") == float(epoch):
                self.regroup_history[-1]["val_score"] = float(val_v)
                # ⚠ Only the *selection* metric used to be recorded here, so
                #   the question "which epoch would AUROC (or logloss) have
                #   chosen, and what did the prototype partition look like
                #   there" could not be answered without retraining once per
                #   metric. That question is now load-bearing: on ds=46 the
                #   run selected at epoch 13 by accuracy and the run selected
                #   at epoch 65 differ by test logloss 0.42 vs 0.19 and
                #   dead_ratio 56% vs 12%, while test accuracy moves the other
                #   way (0.962 -> 0.956). Different criteria select
                #   qualitatively different prototype states, so all of them
                #   are stored and the comparison is done afterwards on one
                #   run instead of one run per criterion.
                for _mk, _mv in val_m.items():
                    self.regroup_history[-1][f"val_{_mk}"] = float(_mv)

            # Whether this epoch may contribute a checkpoint at all. The same
            # predicate gates the patience counter below: when selection is
            # not open, "no improvement" is not yet a meaningful statement.
            _selection_open = _past_regroup_warmup and (epoch >= self.min_epochs)
            if _selection_open and self.selection_open_epoch is None:
                self.selection_open_epoch = int(epoch)

            # Save the best model
            if is_better(val_v, best_val, self.tasktype) and _selection_open:
                best_val   = val_v
                # Which epoch the returned model actually comes from.
                self.best_epoch = int(epoch)
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                # sample_groups and group_labels are plain Python attributes
                # of CentroidLayer, and feature_store._store is not an
                # nn.Module, so none of them appear in state_dict().
                # load_state_dict() rolls centroid_emb and memory.keys back to
                # the best validation epoch while these three would stay at
                # the last training epoch -- snapshots of two different
                # moments mixed together. They are therefore snapshotted and
                # restored alongside best_state.
                #
                # label_all_groups and label_groups_by_target run here rather
                # than every epoch. Only the values from a new best survive
                # anyway: without an improvement they remain the final version,
                # and with one they are recomputed and overwritten. The end
                # result is identical to computing them every epoch, minus the
                # work on epochs that did not improve. x_regroup and y_regroup
                # come from this epoch's regroup_update() block, so the
                # MemoryBank state matches exactly -- recomputing after
                # training would read a MemoryBank that has moved on.
                if (
                    self.cat_cols is not None
                    and self.num_cols is not None
                    and self.col_names is not None
                    and x_regroup is not None
                ):
                    _t0 = self._t()
                    self.model.prototype_layer.group_labels = label_all_groups(
                        x_regroup.detach().cpu().numpy(),
                        self.model.prototype_layer.sample_groups,
                        self.cat_cols,
                        self.num_cols,
                        self.col_names,
                        cat_category_names=self.cat_category_names,
                        quantile_transformer=self.quantile_transformer,
                    )
                if y_regroup is not None:
                    self.model.prototype_layer.target_labels = label_groups_by_target(
                        y_regroup.detach().cpu().numpy(),
                        self.model.prototype_layer.sample_groups,
                        self.tasktype,
                        class_names=self.target_class_names,
                    )
                    self._tick("label_all_groups / target (on new best)", _t0)
                best_sample_groups = copy.deepcopy(self.model.prototype_layer.sample_groups)
                best_group_labels  = copy.deepcopy(self.model.prototype_layer.group_labels)
                best_target_labels = copy.deepcopy(self.model.prototype_layer.target_labels)
                if self.model.feature_store is not None:
                    best_feature_store = (
                        self.model.feature_store._store.clone(),
                        self.model.feature_store._ptr,
                        self.model.feature_store._filled,
                        self.model.feature_store._sample_ids.clone(),
                    )
                else:
                    best_feature_store = None

            # tqdm postfix takes a dict and abbreviates it when the terminal
            # is too narrow
            pbar.set_description(f"EPOCH: {epoch}")
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                id=self._data_id,
                refresh=False,
            )

            # Early stopping
            if getattr(self, "log_beta", False):
                _bp = getattr(self.model, "dev_beta_raw", None)
                if _bp is not None:
                    _n = max(self._beta_grad_n, 1)
                    self.beta_history.append({
                        "epoch": int(epoch),
                        "raw": float(_bp.detach().mean()),
                        "beta": float(torch.sigmoid(_bp.detach()).mean()),
                        # Batch means. |grad| is magnitude, signed is
                        # direction. A signed near 0 with a large |grad| means
                        # the direction keeps flipping, i.e. the term is
                        # needed but in equilibrium.
                        "grad_abs_mean":    self._beta_grad_sum / _n,
                        "grad_signed_mean": self._beta_grad_signed / _n,
                        "n_batch": int(self._beta_grad_n),
                    })
                    self._beta_grad_sum = 0.0
                    self._beta_grad_signed = 0.0
                    self._beta_grad_n = 0

            self._tick("epoch_total", _t_epoch)
            if self._timing_on:
                self._timing["n_epoch"] = self._timing.get("n_epoch", 0) + 1

            # With defer_early_stopping the counter starts when selection
            # opens. Seeding es.best_value during warmup is the specific
            # failure being avoided: a warmup score later epochs cannot beat
            # spends patience on epochs that were never selectable.
            if (not self.defer_early_stopping) or _selection_open:
                if es.step(val_v, higher_is_better):
                    tqdm.write(f"Early stopping at epoch {epoch}")
                    break

        pbar.close()

        if best_state is None:
            # If regroup_warmup_epochs is too long (or patience too short),
            # the run can stop before warmup ever ends and best_state is never
            # filled. The model then keeps the weights from the final training
            # step, but sample_groups may still be empty, so explanations and
            # group-constrained retrieval may not work. Warn about it.
            tqdm.write(
                f"  !  best_state was never updated. "
                f"regroup_warmup_epochs({self.model.prototype_layer.regroup_warmup_epochs}) "
                f"/ min_epochs({self.min_epochs}) "
                f"may have outlasted the early-stopping point. Retry with a "
                f"shorter warmup or a larger patience."
            )

        if best_state:
            self.model.load_state_dict(best_state)
            # sample_groups and feature_store are not in the state_dict, so
            # they are restored to the same best epoch explicitly, keeping
            # them in step with centroid_emb and memory.keys.
            # memory.keys / labels / ptr / filled / sample_ids are all
            # nn.Module buffers and were already restored by the
            # load_state_dict() above; feature_store is restored next so the
            # two always describe the same moment.
            if best_sample_groups is not None:
                self.model.prototype_layer.sample_groups = best_sample_groups
                # ⚠ INVARIANT: sample_groups and memory._cached_groups must
                #   describe the **same assignment snapshot**. retrieve() reads
                #   the cache, not sample_groups, so restoring one without the
                #   other leaves retrieval searching the groups of whichever
                #   epoch happened to be last.
                #
                #   That is exactly what happened. The line above rolled
                #   sample_groups back to the best epoch while the cache stayed
                #   at the final one, and the two agreed on 2 of 15 groups.
                #   Only 434 of 640 retrieved neighbours came from the assigned
                #   group; after this call, 640 of 640 do.
                #
                #   ⚠ The bug was invisible to accuracy: retrieval is outside
                #     the prediction path, so every metric was unaffected while
                #     the explanations cited the wrong group. It survived
                #     because --refresh_on_best (on by default) rebuilds both
                #     further below, so only --no-refresh_on_best runs were
                #     wrong. Re-caching here costs one pass and makes the
                #     invariant hold regardless of that flag.
                self.model.memory.cache_sample_groups(
                    best_sample_groups,
                    device=torch.device(self.device),
                    centroid_emb=self.model.prototype_layer.centroid_emb,
                )
            if best_group_labels is not None:
                self.model.prototype_layer.group_labels = best_group_labels
            if best_target_labels is not None:
                self.model.prototype_layer.target_labels = best_target_labels
            if best_feature_store is not None:
                store, ptr, filled, sample_ids = best_feature_store
                self.model.feature_store._store       = store
                self.model.feature_store._ptr         = ptr
                self.model.feature_store._filled      = filled
                self.model.feature_store._sample_ids  = sample_ids
            # Refresh only after both memory (restored above) and
            # feature_store (just restored) sit at the same best epoch. Doing
            # it in the wrong order trips the assert inside
            # refresh_memory_keys() that compares their filled counts.
            if self.refresh_on_best:
                refresh_stats = self.model.refresh_memory_keys()
                if refresh_stats is not None:
                    tqdm.write(f"  [refresh_on_best] recomputed {refresh_stats['n_refreshed']} "
                               f"memory.keys slots with the frozen weights")
                    regroup_stats = self._resync_groups_after_refresh()
                    if regroup_stats is not None:
                        tqdm.write(f"  [refresh_on_best] resynced sample_groups on the clean "
                                   f"embeddings (active={regroup_stats.get('active_ratio', 0)*100:.0f}%, "
                                   f"reinit={regroup_stats.get('reinit_count', 0)})")
        self._timing_report()
        if getattr(self, "log_beta", False) and self.beta_history:
            h = self.beta_history
            print(f"\n  [beta] dev_beta_raw trajectory  ({len(h)} epochs)")
            print(f"    {'epoch':<8}{'beta':<11}{'raw':<11}{'mean |grad|':<14}{'mean signed'}")
            _idx = sorted(set(list(range(min(5, len(h))))
                              + list(range(0, len(h), max(1, len(h) // 8)))
                              + [len(h) - 1]))
            for i in _idx:
                r = h[i]
                print(f"    {r['epoch']:<8}{r['beta']:<11.5f}{r['raw']:<11.4f}"
                      f"{r['grad_abs_mean']:<14.3e}{r['grad_signed_mean']:+.3e}")
            _ga = [r["grad_abs_mean"] for r in h]
            _gs = [r["grad_signed_mean"] for r in h]
            print(f"    beta {h[0]['beta']:.5f} → {h[-1]['beta']:.5f}"
                  f"   median |grad| {float(np.median(_ga)):.3e}"
                  f"   median signed {float(np.median(_gs)):+.3e}")
            # How often the sign of `signed` flips: the equilibrium test
            _sg = np.sign(_gs)
            _flip = float(np.mean(_sg[1:] != _sg[:-1])) if len(_sg) > 1 else float("nan")
            print(f"    signed sign-flip rate {_flip:.1%}  "
                  f"(high means equilibrium, near 0 means a one-way push)")
        self._best_state = best_state

        # ── Centroid margin z-score, on the model restored to the best epoch
        # Computed automatically for every HPO trial and stored in
        # self.centroid_geometry_diag. Whether the objective in optimize.py
        # uses it alongside val_v is decided there; the search space is not
        # touched from here.
        self.centroid_geometry_diag = self._compute_centroid_margin_zscore(X_val)

        # Routing stability across the whole run; not fed into any penalty.
        # centroid_geometry_diag above sees only the best-epoch snapshot, so
        # it misses a trial whose snapshot happened to look good while the run
        # never settled (measured: credit-g trial #47). Whether these two
        # correlate with bad outcomes -- unstable reproducibility, low test
        # score -- should be checked across many trials before either enters
        # the penalty. That is why they are not used yet; the z_margin penalty
        # This avoids repeating the pattern where a threshold was set without
        # verification and then revised twice.
        if self.centroid_geometry_diag is not None:
            n_epochs_seen = max(1, len(_active_ratio_history))
            self.centroid_geometry_diag["reinit_per_epoch"] = _reinit_total / n_epochs_seen
            self.centroid_geometry_diag["active_ratio_std"] = (
                float(np.std(_active_ratio_history)) if len(_active_ratio_history) > 1 else 0.0
            )
            # Mean pairwise cosine distance between centroids at the end of
            # training (with best_state restored). Same definition as the log
            # printed right after initialisation (1 - cosine_sim, averaged off
            # the diagonal), so start and end are directly comparable. This
            # matters under EMA, where no term pushes centroids apart: if this
            # value is clearly smaller at the end than at the start, the
            # centroids have bunched together.
            with torch.no_grad():
                c_norm   = F.normalize(self.model.prototype_layer.centroid_emb, dim=-1)
                sim_mat  = c_norm @ c_norm.T
                mask     = ~torch.eye(c_norm.shape[0], dtype=torch.bool, device=c_norm.device)
                self.centroid_geometry_diag["avg_inter_dist_final"] = (
                    (1.0 - sim_mat[mask]).mean().item()
                )

    def _compute_centroid_margin_zscore(
        self, X_val: torch.Tensor, n_null_trials: int = 50,
    ) -> Optional[Dict[str, float]]:
        """
        Diagnose how far the top1-top2 query-centroid cosine margin departs
        from a null baseline built from entirely random (untrained) centroid
        and query vectors, reported as a z-score.

        routing_scale does not affect the forward pass -- the straight-through
        hard assignment is invariant to a positive scale -- but it does affect
        how peaked the backward gradient is. With a low routing_scale the
        gradient blends across several centroid directions, and queries never
        cluster sharply around a centroid; in the worst case the margin ends
        up narrower than random (credit-g at routing_scale 1.49:
        z_margin = -3.40, significantly worse than chance). Datasets that
        landed on a large routing_scale (socmob 19.8, SpeedDating 13.77) reach
        z_margin +18 to +22.

        Returns
        ───────
        None when there is no prototype_layer or P < 2.
        dict: {z_top1, z_margin, top1_median, margin_mean,
               null_top1_mean, null_margin_mean}
        """
        if not (hasattr(self.model, "prototype_layer")
                and self.model.prototype_layer is not None):
            return None

        P = self.model.prototype_layer.P
        if P < 2:
            return None
        D = self.model.prototype_layer.centroid_emb.shape[1]
        n_val = X_val.shape[0]

        self.model.eval()
        with torch.no_grad():
            c_norm = F.normalize(self.model.prototype_layer.centroid_emb, dim=-1)
            top1_sims_list, margins_list = [], []
            _batch = 256
            for start in range(0, n_val, _batch):
                q_norm = F.normalize(
                    self.model.embedder(X_val[start:start + _batch]), dim=-1
                )
                sim  = q_norm @ c_norm.T
                top2 = sim.topk(min(2, P), dim=-1).values
                top1_sims_list.append(top2[:, 0].cpu())
                if top2.shape[1] > 1:
                    margins_list.append((top2[:, 0] - top2[:, 1]).cpu())

        if not margins_list:
            return None
        top1_sims = torch.cat(top1_sims_list).numpy()
        margins   = torch.cat(margins_list).numpy()

        # Null baseline: entirely random vectors at the same D, P and n_val.
        # Computed on CPU -- pure tensor work unrelated to the model, so there
        # is no GPU sync to pay for.
        null_top1_medians = np.empty(n_null_trials)
        null_margin_means = np.empty(n_null_trials)
        for t in range(n_null_trials):
            g = torch.Generator().manual_seed(t)
            q_null = F.normalize(torch.randn(n_val, D, generator=g), dim=-1)
            c_null = F.normalize(torch.randn(P, D, generator=g), dim=-1)
            sim_null  = q_null @ c_null.T
            top2_null = sim_null.topk(min(2, P), dim=-1).values
            null_top1_medians[t] = top2_null[:, 0].median().item()
            null_margin_means[t] = (
                (top2_null[:, 0] - top2_null[:, 1]).mean().item()
                if top2_null.shape[1] > 1 else float("nan")
            )

        null_top1_mean,   null_top1_std   = float(null_top1_medians.mean()), float(null_top1_medians.std())
        null_margin_mean, null_margin_std = float(np.nanmean(null_margin_means)), float(np.nanstd(null_margin_means))

        z_top1   = (float(np.median(top1_sims)) - null_top1_mean) / (null_top1_std + 1e-8)
        z_margin = (float(margins.mean()) - null_margin_mean) / (null_margin_std + 1e-8)

        # The percentile is computed directly, separately from the z-score. A
        # z-score assumes normality, and choosing where to put the "how
        # significant is significant enough" threshold (z = 2.0, say) is
        # another arbitrary decision. With 50 null samples already in hand,
        # the rank of the measured margin among them can simply be counted --
        # no normality assumption and no threshold. The HPO penalty in
        # optimize.py uses this percentile as a continuous value, which
        # removes the threshold entirely.
        margin_percentile = float((null_margin_means < margins.mean()).mean())

        return {
            "z_top1":            z_top1,
            "z_margin":          z_margin,
            "margin_percentile": margin_percentile,
            "top1_median":       float(np.median(top1_sims)),
            "margin_mean":       float(margins.mean()),
            "null_top1_mean":    null_top1_mean,
            "null_margin_mean":  null_margin_mean,
        }

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

    # ── Batched inference ───────────────────────────────────

    def _forward_batched(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        # The old default of 1024 exceeded the training batch size (128-512 as
        # chosen by HPO), which made validation far slower than training once a
        # group had grown large (measured: val_forward going from 1s to 76s
        # within a few epochs while training stayed at 5-6s). A larger batch is
        # more likely to contain a query pointing at a huge group, and the
        # U / local_max_g rounding inside retrieve() then makes the per-batch
        # tensors larger than during training. Validation now uses the same
        # batch size as training.
        if batch_size is None:
            batch_size = self.params.get("batch_size", 512)
        parts = []
        for start in range(0, len(X), batch_size):
            parts.append(self.model(X[start:start + batch_size])["logits"])
        return torch.cat(parts, dim=0)