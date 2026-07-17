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
import copy
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
        cat_cols: Optional[List[int]] = None,
        num_cols: Optional[List[int]] = None,
        col_names: Optional[List[str]] = None,
        cat_category_names: Optional[Dict[str, List[str]]] = None,
        target_class_names: Optional[List[str]] = None,
        quantile_transformer=None,
        regroup_log_every: int = 10,   # [Regroup] 로그 출력 주기(epoch). 진단 목적으로 좁힐 수 있게.
        refresh_on_best: bool = False,
        # [추가] best_state 복원 직후 model.refresh_memory_keys()를 호출할지.
        # 기본 False — 기존 Optuna study/best_params는 이 플래그와 무관하게
        # 전혀 오염되지 않음(변경 전 재현 그대로 유지). True로 켜면
        # memory.keys가 raw feature의 결정론적 함수가 되지만, dropout 노이즈가
        # retrieval 강건성에 기여하고 있었다면 val/test 성능이 달라질 수
        # 있음 — 그래서 재고 데이터셋에서 A/B 비교부터 하고 기본값 전환을
        # 결정하는 걸 권장 (bare 플래그로 시작, HPO search space엔 아직
        # 안 넣음).
        log_branch_gradients: bool = False,
        # [진단용, 추가] query/context/agg_emb가 head concat 직전에 받는
        # gradient norm·활성값 norm·head 첫 Linear의 block별 weight norm을
        # epoch마다 기록(self.branch_gradient_history). 목적: "head가
        # 세 브랜치 중 특정 하나(주로 query_emb)에만 의존하게 학습되는지"
        # 진단 — context_emb/agg_emb ablation 시 성능이 거의 안 변한다는
        # 기존 관찰(사후 정적 진단)을, 학습 "과정 중" gradient 흐름으로
        # 보완하기 위함. TabERA.log_branch_gradients로 그대로 전달되며,
        # retain_grad() 기반이라 학습 결과에는 영향 없음(메모리만 소폭 증가).
        log_branch_gradients_first_n_epochs: int = 3,
        # [진단용, 추가] 위 진단 중 배치 단위 세부 기록(branch_gradient_
        # batch_history)은 학습 전체에 남기면 메모리가 배치 수만큼 계속
        # 쌓이므로, 학습 초반 n epoch만 배치 단위로 남기고 그 이후는
        # epoch 평균(branch_gradient_history)만 남긴다 — OGM 계열 문헌이
        # 강조하는 게 "초기 학습 dynamics"라, 초반만 촘촘히 보면 충분하다는
        # 판단. 검증 안 된 기본값(3)이라 필요시 조정.
        log_evidence_stats: bool = False,
        # [진단용, 추가] evidence_w(②의 AttentionAggregator 가중치)의 entropy·
        # dominant weight를 epoch마다 기록(self.evidence_stats_history).
        # out["evidence_w"]가 이미 매 forward에서 나오므로 backward/retain_grad
        # 없이 순수 forward 통계만 뽑음 — log_branch_gradients보다 훨씬 저렴.
        # 목적: --explain에서 학습 끝난 뒤 딱 한 시점(3개 샘플)만 봐서는
        # "언제부터 evidence가 소수 이웃으로 붕괴됐는지" 알 수 없었던 것을
        # 학습 전체 epoch에 걸쳐 정량적으로 보기 위함.
        # [진단용, 추가] fusion_mode="residual"의 α/β와 branch norm(‖LN(q)‖/
        # ‖LN(c)‖/‖LN(a)‖) 궤적을 epoch마다 기록(self.fusion_trajectory_history).
        # 지금까지는 meta.pkl에 최종값만 있어서 "처음부터 거의 안 움직였다"와
        # "여러 번 오르내리다 지금 값에 안착했다"를 구분 못 했음 — evidence_w
        # 진단과 같은 이유로 이것도 순수 forward 통계라 backward/retain_grad
        # 없이 저렴하게 뽑음. fusion_mode!="residual"인 모델에 켜도 에러는
        # 안 나되(out 딕셔너리에 항상 존재하는 키들이라) alpha/beta 값이 항상
        # None으로 기록됨 — 그런 경우 그냥 안 켜는 게 맞음.
        log_fusion_trajectory: bool = False,
    ) -> None:
        self.model    = model.to(device)
        self.params   = params
        self.tasktype = tasktype
        self.device   = device
        self.epochs   = epochs
        self.patience = patience
        self.regroup_log_every = max(1, regroup_log_every)
        self._best_state = None
        self._data_id    = "?"      # tqdm 표시용 (optimize.py에서 주입)
        self.regroup_history: List[Dict[str, float]] = []
        self.final_regroup_stats: Optional[Dict[str, float]] = None
        # fit() 완료 후 채워짐 — z_top1/z_margin 등 (compute_metric처럼
        # optimize.py의 objective가 val_v 외에 추가로 참조할 수 있는 값).
        # prototype_layer가 없는 모델이거나 계산 실패 시 None으로 남음.
        self.centroid_geometry_diag: Optional[Dict[str, float]] = None
        # ── 그룹 텍스트 라벨링 (libs/group_labels.py) ────────────
        # 셋 다 주어져야 라벨링을 수행함. 하나라도 없으면(예: 기존
        # optimize.py처럼 이 인자들을 안 넘기는 호출부) 조용히 건너뛰고
        # group_labels는 계속 None으로 남는다 — centroid_x/medoid를
        # 없앤 것과 달리 이건 선택적 부가 기능이라 하위 호환을 깨지 않음.
        self.cat_cols  = cat_cols
        self.num_cols  = num_cols
        self.col_names = col_names
        # {col_name: [원본 카테고리 문자열, ...]} — libs/data.py의
        # load_data()가 반환하는 것. 없어도(None) label_all_groups()가
        # "Category N" fallback으로 계속 동작하니 하위 호환 안 깨짐.
        self.cat_category_names = cat_category_names
        # [원본 target 라벨 문자열, ...] — libs/data.py의 load_data()가
        # 반환하는 것. "Class 0"/"Class 1" 대신 실제 라벨명("good"/"bad" 등)
        # 표시에 씀. 없어도(None) label_groups_by_target()가 "Class N"
        # fallback으로 계속 동작.
        self.target_class_names = target_class_names
        # fit된 QuantileTransformer(numeric feature용) — libs/data.py의
        # prep_data()가 반환하는 것. 있으면 label_all_groups()가 numeric
        # 값을 [0,1] uniform 대신 실제 단위로 역변환해 보여준다.
        self.quantile_transformer = quantile_transformer
        self.refresh_on_best = refresh_on_best
        # [진단용] 모델 forward()가 참조하는 플래그 자체는 TabERA에 있음 —
        # 여기서 wrapper 생성 시점에 그대로 배선. state_dict에 안 들어가는
        # 순수 런타임 속성이라 체크포인트 저장/로드와 무관.
        self.log_branch_gradients = log_branch_gradients
        self.model.log_branch_gradients = log_branch_gradients
        self.log_branch_gradients_first_n_epochs = log_branch_gradients_first_n_epochs
        # epoch별 요약(항상, log_branch_gradients=True인 동안 전체 학습에 걸쳐):
        #   {"epoch", "query_grad_norm", "query_act_norm", "query_weight_norm", ...}
        self.branch_gradient_history: List[Dict[str, float]] = []
        # 배치 단위 세부 기록(처음 log_branch_gradients_first_n_epochs epoch만):
        #   {"epoch", "batch", "branch", "grad_norm", "act_norm"}
        self.branch_gradient_batch_history: List[Dict[str, float]] = []
        self.log_evidence_stats = log_evidence_stats
        # epoch별 evidence_w 요약: {"epoch", "entropy", "dominant_weight"}
        self.evidence_stats_history: List[Dict[str, float]] = []
        self.log_fusion_trajectory = log_fusion_trajectory
        # epoch별 요약: {"epoch", "alpha", "beta", "query_norm_mean",
        #                "context_norm_mean", "agg_norm_mean"}
        self.fusion_trajectory_history: List[Dict[str, float]] = []

    # ── fit ─────────────────────────────────────────────────

    def _resync_groups_after_refresh(self) -> Optional[Dict[str, float]]:
        """refresh_memory_keys() 직후 호출. sample_groups(centroid별 그룹
        캐시)는 학습 중 그 시점의 memory.keys(noisy)를 기준으로 계산돼
        있는데, refresh_memory_keys()는 memory.keys를 완전히 다른 값
        (clean embedding)으로 갈아치우면서 sample_groups는 그대로 둔다 —
        그러면 두 저장소가 서로 다른 시점의 스냅샷이 되어버림
        (dual_space_faithfulness의 "사전 검증 1" 재배정 일치율이 무너지는
        원인 — libs/prototypes.py의 regroup_update() 안에 있는 dead-code
        reinit 버그 수정 주석과 정확히 같은 종류의 문제).

        regroup_update()는 X_raw를 안 쓰고(하위 호환용 시그니처만 유지),
        dead-centroid reinit이 일어나도 그 직후 최종 centroid_emb 기준
        으로 assignment를 다시 계산해 sample_groups를 덮어쓰므로, 이걸
        clean 임베딩으로 한 번 더 불러주면 (sample_groups, centroid_emb,
        memory.keys) 셋이 항상 서로 일치하는 상태로 돌아온다.

        [참고] regroup_update()가 내부적으로 current_epoch을 1 증가시키고,
        dead_streak 상태에 따라 일부 centroid를 재초기화할 수 있다 —
        학습 중과 동일한 안전장치가 그대로 적용되는 것이라 원칙적으로는
        문제없지만, 그 결과 최종 centroid_emb가 저장된 best_state의 값과
        (재초기화된 소수의 centroid에 한해) 미세하게 달라질 수 있다는
        점은 알아둘 필요가 있음.
        """
        if not (hasattr(self.model, 'prototype_layer')
                and hasattr(self.model.prototype_layer, 'regroup_update')):
            return None

        with torch.no_grad():
            n_mem = self.model.memory.filled.item()
            if n_mem < 1:
                return None
            emb_regroup = self.model.memory.keys[:n_mem]   # 방금 refresh된 clean 값
            regroup_stats = self.model.prototype_layer.regroup_update(emb_regroup)

            # retrieve()가 참조하는 GPU 그룹 캐시도 같이 갱신 — 안 하면
            # sample_groups(방금 갱신됨)와 cache_sample_groups()가 예전에
            # 만들어둔 캐시(옛 그룹 기준)가 또 어긋남.
            self.model.memory.cache_sample_groups(
                self.model.prototype_layer.sample_groups,
                device=torch.device(self.device),
                centroid_emb=self.model.prototype_layer.centroid_emb,
            )

            # ①의 텍스트 라벨(group_labels/target_labels)도 옛 그룹 기준
            # 캐시라 stale함 — 새 sample_groups로 다시 계산.
            fs = self.model.feature_store
            if (fs is not None and self.cat_cols is not None
                    and self.num_cols is not None and self.col_names is not None):
                x_regroup = fs._store[:n_mem].to(self.device)
                self.model.prototype_layer.group_labels = label_all_groups(
                    x_regroup.detach().cpu().numpy(),
                    self.model.prototype_layer.sample_groups,
                    self.cat_cols, self.num_cols, self.col_names,
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

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        skip_centroid_init: bool = False,
        # [추가] True면 initialize_from_data()(KMeans++ 재초기화)를 건너뜀.
        # freeze-encoder-retrain-head 실험처럼 이미 학습된 centroid를 그대로
        # 유지한 채 head만 재학습하고 싶을 때 필수 — 안 그러면 fit() 진입
        # 시점에 무조건 KMeans++로 centroid가 덮어써져서, 인코더를 얼려도
        # 프로토타입 자체는 매번 새로 초기화되는 모순이 생김.
    ) -> None:
        criterion  = get_criterion(self.tasktype)  # weight_matrix GPU 이동

        # [정정] centroid_emb를 weight_decay에서 빼는 대신, ArcFace/CosFace
        # 표준을 그대로 따른다 — "정규화해서 쓰는 파라미터는 weight_decay를
        # 빼는 게 아니라, 매 스텝 후 norm=1로 강제 재투영"하는 방식(CosFace:
        # "norm(W)를 반드시 불변으로 고정"). 이러면 weight_decay를 평범하게
        # 켜둬도(ArcFace/CosFace 논문들의 실제 학습 세팅과 동일) centroid_emb
        # 방향이 흔들리지 않는다 — weight_decay 제외는 지도학습(ArcFace)이
        # 하지 않는 방식이었고, 저희 라우팅도 어차피 F.normalize(centroid_emb)
        # 만 쓰므로 원본 파라미터 자체를 늘 단위벡터로 유지하는 쪽이 일관적.
        # 재투영은 fit() 아래 학습 루프의 optimizer.step() 직후에서 수행.
        centroid_param = self.model.prototype_layer.centroid_emb

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
        if (not skip_centroid_init and hasattr(self.model, 'prototype_layer')
                and hasattr(self.model.prototype_layer, 'initialize_from_data')):
            with torch.no_grad():
                n_init   = min(len(X_train), 5000)
                init_emb = self.model.embedder(X_train[:n_init])
                init_x   = X_train[:n_init]
                init_y   = y_train[:n_init]  # 소수 클래스 보장용
                self.model.prototype_layer.initialize_from_data(
                    init_emb, init_x, y_labels=init_y
                )

        # [진단용, 추가] 순수 초기화 상태(학습 0 step) evidence entropy 측정.
        # 목적: epoch 1(이미 몇 스텝 학습된 상태)의 낮은 entropy가 "학습이
        # collapse를 유도했다"인지 "애초에 정규화 안 된 거리 공간에서
        # softmax가 시작부터 뾰족했다"인지 구분(리뷰 지적 — Case A/B).
        # MemoryBank가 아직 하나도 안 채워진 시점이라 정식 retrieve()는
        # 못 씀 — 방금 만든 init_emb(학습 전 임베더로 인코딩된 X_train
        # 샘플) 안에서 자기 자신을 제외한 top-k 최근접만 뽑아 같은
        # 유사도 공식(-‖q-k‖², evidence_temperature 포함)을 그대로 적용한
        # 근사치. 정식 MemoryBank.retrieve()와 후보 풀이 다르다는 한계는
        # 있지만(랜덤 배치 내 검색 vs 그룹 제약 검색), "이 embedding
        # 공간에서 raw 유클리드 거리 softmax가 얼마나 뾰족한가" 자체는
        # 충분히 잰다.
        if self.log_evidence_stats and hasattr(self.model, "ot_selector"):
            with torch.no_grad():
                _sample_n = min(512, init_emb.shape[0])
                _q = init_emb[:_sample_n]
                k_probe = min(self.model.k, _sample_n - 1)
                if k_probe > 0:
                    d2 = torch.cdist(_q, _q, p=2) ** 2         # (n, n)
                    d2.fill_diagonal_(float("inf"))            # 자기 자신 제외
                    topk_d2, _ = d2.topk(k_probe, largest=False, dim=-1)  # 최근접 k개
                    _temp = getattr(self.model.ot_selector, "evidence_temperature", 1.0)
                    ew_init = F.softmax((-topk_d2) / _temp, dim=-1)
                    ent_init = float((-(ew_init * (ew_init + 1e-8).log()).sum(dim=-1)).mean().item())
                    dom_init = float(ew_init.max(dim=-1).values.mean().item())
                    _qn = float(_q.norm(dim=-1).mean().item())
                    self.evidence_stats_history.append({
                        "epoch": 0.0,
                        "entropy": ent_init,
                        "n_eff": float(np.exp(ent_init)),
                        "dominant_weight": dom_init,
                        "query_norm": _qn,
                        "key_norm": _qn,  # 같은 배치 내 검색이라 query/key가 같은 분포
                        "distance_mean": float(topk_d2.mean().item()),
                        "distance_std": float(topk_d2.std().item()),
                    })
                    tqdm.write(
                        f"  [EvidenceStats init] entropy={ent_init:.4f}  "
                        f"n_eff={float(np.exp(ent_init)):.3f}  dominant_weight={dom_init:.4f}  "
                        f"(학습 0 step, 순수 초기화 상태 — epoch 1 이후 값과 비교할 기준선)"
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
        # [추가, penalty 미반영 순수 로깅용] 학습 전체에 걸친 라우팅 안정성
        # 진단 — "상관관계부터 확인하고 penalty는 나중에" 원칙에 따라
        # 일단 기록만 한다. reinit_total: 전체 reinit 발생 횟수(적을수록
        # 안정). active_ratio_history: 매 regroup_update의 active_ratio를
        # 전부 모아뒀다가 std를 계산(요동 정도 — 최종 스냅샷만 보는
        # centroid_geometry_diag가 못 보는 "과정 전체의 불안정성"을 보완).
        _reinit_total = 0
        _active_ratio_history: List[float] = []
        # [최적화] label_all_groups/label_groups_by_target를 is_better 블록
        # 안으로 옮기면서 그 재료(x_regroup/y_regroup)를 epoch 끝까지 들고가야 함 —
        # n_mem<1인 극초반 epoch이나 prototype_layer가 없는 모델에서는
        # 아예 안 채워질 수 있어 None으로 미리 선언 (참조 시 NameError 방지).
        x_regroup = None
        y_regroup = None

        # [버그 수정] 이전엔 emb_cache로 X_train 전체(N_train개)를 캐시해서
        # regroup_update에 넘겼음 → sample_groups가 "X_train 행 번호"로 만들어짐.
        # 그런데 MemoryBank는 별도의 링버퍼(크기 min(2*N_train, 10000))라서
        # 두 인덱스 공간이 어긋남 (N_train > memory_size인 경우 특히 심각 —
        # retrieve()에서 clamp로 인덱스가 뭉개짐). 아래에서 regroup_update에
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
            # [진단용] log_branch_gradients — 이번 epoch 누적용. 브랜치
            # 이름은 use_query_emb_in_head/use_context_emb에 따라 forward()가
            # self.model._branch_grad_tensors에 실제로 채운 키만 등장하므로
            # 여기서는 dict를 비워두고 첫 배치에서 만난 키를 그대로 채택한다.
            _branch_grad_sum: Dict[str, float] = {}
            _branch_act_sum:  Dict[str, float] = {}
            _branch_batches:  Dict[str, int] = {}  # [수정] 브랜치별 등장 배치 수를
            # 따로 센다 — agg 브랜치는 memory.filled < k인 극초반 배치에서
            # requires_grad=False라 tabera.py forward()가 아예 등록을 건너뛴다
            # (위 tabera.py 주석 참고). 공유 카운터 하나로 나누면 그런 배치들
            # 때문에 epoch 평균이 실제보다 작게 나오는 오류가 생김.
            # [진단용] log_evidence_stats — evidence_w entropy/dominant weight
            # 누적용. gradient 필요 없는 순수 forward 통계라 backward 전에도
            # 계산 가능.
            _evidence_entropy_sum  = 0.0
            _evidence_dominant_sum = 0.0
            _evidence_batches      = 0
            # [추가] Step 0 진단 — "distance 값 자체가 큰가"(temperature 문제)
            # vs "query/key norm이 커지는가"(normalization 문제) 구분용.
            # evidence_diag가 None인 배치(memory warmup fallback)는 자동 제외.
            _evidence_qnorm_sum = 0.0
            _evidence_knorm_sum = 0.0
            _evidence_dist_mean_sum = 0.0
            _evidence_dist_std_sum  = 0.0
            _evidence_diag_batches  = 0
            # [진단용, 추가] self-retrieval rate — log_evidence_stats 플래그에
            # 얹음(별도 CLI 플래그 안 만듦 — evidence 진단과 같은 성격,
            # 오버헤드도 이미 계산된 값을 배치마다 누적하는 것뿐이라 저렴함).
            _self_retrieval_top1_sum = 0.0
            _self_retrieval_topk_sum = 0.0
            _self_retrieval_batches  = 0
            # [진단용, 추가] log_fusion_trajectory — norm/cos/grad는 배치마다
            # 다르므로 epoch 평균을 위해 누적. alpha/beta 값 자체는 epoch
            # 마지막 배치 시점 파라미터 값을 그대로 기록(파라미터는 "지금
            # 어디 있는가"가 중요 — 평균 내면 "0.95→0.90으로 계속 하락"이
            # "0.925로 안정"인 것처럼 흐려짐). grad는 반대로 "매 배치 존재하는
            # 신호"라 평균이 자연스러움(clip_grad_norm_ 이전 raw grad).
            _fusion_qnorm_sum = 0.0
            _fusion_cnorm_sum = 0.0
            _fusion_anorm_sum = 0.0
            _fusion_combined_norm_sum = 0.0
            _fusion_cos_qc_sum = 0.0
            _fusion_cos_qa_sum = 0.0
            _fusion_cos_ca_sum = 0.0
            _fusion_norm_batches = 0
            _fusion_cnorm_batches = 0     # context 관련 값들(cnorm, cos_qc, cos_ca)은
            _fusion_combined_batches = 0  # use_context_emb/fusion_mode에 따라 None일
            _fusion_cos_qc_batches = 0    # 수 있어 항목별로 분모를 따로 셈
            _fusion_cos_qa_batches = 0
            _fusion_cos_ca_batches = 0
            _fusion_alpha_grad_sum = 0.0
            _fusion_beta_grad_sum  = 0.0
            _fusion_alpha_grad_batches = 0
            _fusion_beta_grad_batches  = 0

            for start in range(0, len(y_train), self.params["batch_size"]):
                idx = perm[start:start + self.params["batch_size"]]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()

                # [추가] idx가 그대로 X_train 행 번호 — MemoryBank/FeatureStore가
                # 이 배치를 저장할 때 같은 값을 sample_ids로 같이 넣어두면,
                # 두 저장소의 슬롯 대응을 사후에 통계가 아니라 정확한 등식으로
                # 검증할 수 있다 (reproduce.py --ablation dual_space_faithfulness).
                out = self.model(xb, labels=yb, sample_ids=idx)
                lg  = out["logits"]

                # [진단용, 추가] log_fusion_trajectory — ||LN(q)||/||LN(c)||/
                # ||LN(a)||/||q+αc+βa||/cos(q,c)/cos(q,a)/cos(c,a) 배치 평균 누적.
                # gradient 불필요(out의 head_* 값들은 이미 tabera.py에서 .detach()
                # 처리됨) — 순수 forward 통계라 매 배치 켜둬도 저렴함(evidence_stats와
                # 같은 이유). 항목마다 None일 수 있는 조건이 달라(예:
                # use_context_emb=False면 context 관련 전부 None, concat 모드면
                # combined_norm만 None) 분모(batches)를 항목별로 따로 센다.
                if self.log_fusion_trajectory:
                    if out.get("head_query_norm_mean") is not None:
                        _fusion_qnorm_sum += out["head_query_norm_mean"]
                    if out.get("head_context_norm_mean") is not None:
                        _fusion_cnorm_sum += out["head_context_norm_mean"]
                        _fusion_cnorm_batches += 1
                    if out.get("head_agg_norm_mean") is not None:
                        _fusion_anorm_sum += out["head_agg_norm_mean"]
                    if out.get("head_combined_norm_mean") is not None:
                        _fusion_combined_norm_sum += out["head_combined_norm_mean"]
                        _fusion_combined_batches += 1
                    if out.get("head_cos_qc_mean") is not None:
                        _fusion_cos_qc_sum += out["head_cos_qc_mean"]
                        _fusion_cos_qc_batches += 1
                    if out.get("head_cos_qa_mean") is not None:
                        _fusion_cos_qa_sum += out["head_cos_qa_mean"]
                        _fusion_cos_qa_batches += 1
                    if out.get("head_cos_ca_mean") is not None:
                        _fusion_cos_ca_sum += out["head_cos_ca_mean"]
                        _fusion_cos_ca_batches += 1
                    _fusion_norm_batches += 1

                # [진단용] log_evidence_stats — evidence_w(②)가 소수 이웃으로
                # 붕괴하는지(dominant→1, entropy→0) 매 배치 기록. AttentionAggregator
                # 의 similarity가 -‖q-k‖²(temperature 없음)라, 임베더 norm이 커질수록
                # softmax가 포화될 수 있다는 가설을 검증하기 위함 — --explain이 학습
                # 끝난 뒤 소수 샘플만 보여주는 것과 달리, 학습 전체 epoch에 걸친 추세를 봄.
                if self.log_evidence_stats:
                    # [버그 수정] AttentionAggregator.forward()가 softmax 이후에
                    # dropout을 또 거는데(evidence_w = self.dropout(evidence_w)),
                    # 학습 모드에서는 살아남은 값이 1/(1-p)로 스케일업돼서(jasmine
                    # dropout=0.5면 최대 2배) evidence_w가 더 이상 확률분포가
                    # 아님(합이 1이 아니고 개별 값이 1을 넘을 수 있음) — 그 상태로
                    # entropy를 재면 음수가 나오는 등 해석 불가능한 값이 됨(실측
                    # 확인됨: entropy=-0.68, dominant_weight=1.02 같은 값 발생).
                    # 합이 1이 되도록 재정규화해서 스케일 왜곡만 제거(dropout으로
                    # 완전히 죽은 이웃의 정보 자체는 복구 안 됨 — 근사치임을 감안).
                    ew = out["evidence_w"].detach()
                    ew = ew / ew.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    _entropy   = float((-(ew * (ew + 1e-8).log()).sum(dim=-1)).mean().item())
                    _dominant  = float(ew.max(dim=-1).values.mean().item())
                    _evidence_entropy_sum  += _entropy
                    _evidence_dominant_sum += _dominant
                    _evidence_batches += 1

                    _diag = out.get("evidence_diag")
                    if _diag is not None:
                        _evidence_qnorm_sum     += _diag["query_norm"]
                        _evidence_knorm_sum     += _diag["key_norm"]
                        _evidence_dist_mean_sum += _diag["distance_mean"]
                        _evidence_dist_std_sum  += _diag["distance_std"]
                        _evidence_diag_batches  += 1

                    # [진단용, 추가] self-retrieval — 값이 None인 배치(memory
                    # warmup fallback, 또는 sample_ids 미전달)는 자연스럽게 제외
                    if out.get("self_retrieval_top1_rate") is not None:
                        _self_retrieval_top1_sum += out["self_retrieval_top1_rate"]
                        _self_retrieval_topk_sum += out["self_retrieval_topk_rate"]
                        _self_retrieval_batches  += 1

                if self.tasktype in ("regression", "binclass"):
                    task_loss = criterion(lg.squeeze(-1), yb.float())
                else:
                    task_loss = criterion(lg, yb.long())

                loss = task_loss + out["aux_loss"]
                loss.backward()

                # [진단용, 추가] log_fusion_trajectory — fusion_alpha/beta.grad를
                # clip_grad_norm_/optimizer.step() 전에 읽음(그 이후엔 다음
                # zero_grad()가 지울 값이라 이 시점이 유일한 기회). override로
                # buffer가 됐으면(requires_grad=False) .grad가 항상 None이라
                # 자동으로 걸러짐 — 그 경우 "grad=0"이 아니라 "애초에 안 잼"이
                # 맞는 의미라 None으로 남기고 0으로 채우지 않는다.
                if self.log_fusion_trajectory:
                    _fa = getattr(self.model, "fusion_alpha", None)
                    _fb = getattr(self.model, "fusion_beta", None)
                    if _fa is not None and _fa.grad is not None:
                        _fusion_alpha_grad_sum += abs(float(_fa.grad.item()))
                        _fusion_alpha_grad_batches += 1
                    if _fb is not None and _fb.grad is not None:
                        _fusion_beta_grad_sum += abs(float(_fb.grad.item()))
                        _fusion_beta_grad_batches += 1

                # [진단용] log_branch_gradients — clip_grad_norm_은 model.
                # parameters()만 대상이라 아래 활성값(파라미터 아님)의 .grad에는
                # 영향을 안 주므로, backward() 직후 아무데서나 읽어도 무방.
                if self.log_branch_gradients:
                    _fine_grained = epoch <= self.log_branch_gradients_first_n_epochs
                    for name, t in self.model._branch_grad_tensors.items():
                        if t.grad is None:
                            continue
                        g_norm = t.grad.norm(dim=-1).mean().item()
                        a_norm = t.detach().norm(dim=-1).mean().item()
                        _branch_grad_sum[name] = _branch_grad_sum.get(name, 0.0) + g_norm
                        _branch_act_sum[name]  = _branch_act_sum.get(name, 0.0) + a_norm
                        _branch_batches[name]  = _branch_batches.get(name, 0) + 1
                        if _fine_grained:
                            self.branch_gradient_batch_history.append({
                                "epoch": float(epoch), "batch": float(n_batch),
                                "branch": name, "grad_norm": g_norm, "act_norm": a_norm,
                            })

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                # [CosFace 표준] centroid_emb를 매 step 후 단위벡터로 강제
                # 재투영 — weight_decay가 켜져 있어도 방향만 남고 크기는
                # 항상 1로 리셋되므로, 크기 붕괴로 인한 라우팅 불안정이
                # 원천적으로 발생하지 않음.
                with torch.no_grad():
                    centroid_param.data = F.normalize(centroid_param.data, dim=-1)

                tr_loss_gpu += loss.detach()          # .item() 없이 GPU에 누적 (동기화 제거)
                n_batch += 1

            scheduler.step()
            self.model.anneal(self.params.get("anneal_factor", 0.97))

            # ── (3) sample_groups 재계산(regroup_update) ───────────────
            # skip_centroid_init=True(freeze-encoder-retrain-head 실험)면 이것도
            # 건너뜀 — dead-centroid 재초기화가 gradient 없이 centroid_emb.data를
            # 직접 덮어써서, "인코더 완전 고정" 조건을 깨뜨리기 때문.
            if (not skip_centroid_init and hasattr(self.model, 'prototype_layer')
                    and hasattr(self.model.prototype_layer, 'regroup_update')):
                with torch.no_grad():
                    # [버그 수정] X_train 전체 대신 MemoryBank에 실제로 들어있는
                    # 내용(최대 memory_size개)만 클러스터링 → sample_groups가
                    # MemoryBank 인덱스 공간과 항상 일치하도록 보장.
                    n_mem = self.model.memory.filled.item()
                    if n_mem < 1:
                        # 메모리가 아직 하나도 안 채워진 극초반 → 스킵
                        regroup_stats = {"active_ratio": 0.0, "min_cluster_size": 0, "max_cluster_size": 0}
                        x_regroup = None
                        y_regroup = None
                    else:
                        emb_regroup = self.model.memory.keys[:n_mem]           # (n_mem, D) — MemoryBank 임베딩
                        fs = self.model.feature_store
                        x_regroup = (
                            fs._store[:n_mem].to(self.device)              # (n_mem, F) — 원본 feature
                            if fs is not None else None
                        )
                        regroup_stats = self.model.prototype_layer.regroup_update(emb_regroup, x_regroup)
                        # [최적화] label_all_groups/label_groups_by_target는
                        # 순수 읽기 전용(설명용 텍스트 캐싱)이라 학습(가중치/
                        # early stopping 판단)에 전혀 영향을 안 준다. 예전엔
                        # 매 epoch 계산했는데, 그중 실제로 쓰이는 건 "val이
                        # 갱신된 epoch"의 값뿐이다(바로 아래 best_* 스냅샷
                        # 로직이 그 값만 남기고 나머지는 다음 epoch에 덮어써
                        # 버림). n_mem이 채워진 epoch이면 이번 epoch의 x_regroup/
                        # y_regroup를 나중(is_better 블록)에 재사용할 수 있도록
                        # 여기서는 사람이 읽는 텍스트 계산 자체를 생략하고
                        # 재료만 남겨둔다 (컬럼 수·centroid 수가 많은 데이터셋
                        # 에서 이 텍스트 계산 자체가 epoch당 수 초~수십 초까지
                        # 걸리는 게 실측 확인됨 — nomao: feature 118개, P=166).
                        y_regroup = self.model.memory.labels[:n_mem]

                    # [추가] 안정성 진단용 누적 — warmup으로 스킵된 epoch도
                    # active_ratio=0.0을 그대로 기록(실제로 아직 활성화 전인
                    # 상태이므로 std 계산에서 왜곡 요인이 아니라 사실 그대로).
                    _reinit_total += regroup_stats.get("reinit_count", 0)
                    _active_ratio_history.append(regroup_stats.get("active_ratio", 0.0))

                    self.final_regroup_stats = dict(regroup_stats)
                    self.regroup_history.append({
                        "epoch": float(epoch),
                        "active_ratio": float(regroup_stats.get("active_ratio", 0.0)),
                        "active_centroids": float(regroup_stats.get("active_centroids", 0.0)),
                        "pruned_this_epoch": float(regroup_stats.get("pruned_this_epoch", 0.0)),
                        "min_cluster_size": float(regroup_stats.get("min_cluster_size", 0.0)),
                        "max_cluster_size": float(regroup_stats.get("max_cluster_size", 0.0)),
                    })

                    # ── retrieve()의 하이브리드 임계값을 실제 GPU 여유 메모리
                    # 기준으로 매 epoch 갱신 (근거 없는 고정 상수 대신).
                    # retrieve() 자체(배치마다 호출됨)에서는 GPU를 조회하지
                    # 않도록, 조회는 여기서 epoch당 1회만 수행.
                    if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                        try:
                            _free_b, _ = torch.cuda.mem_get_info(self.device)
                            self.model.memory.update_outlier_threshold(
                                n_prototypes=self.model.prototype_layer.P,
                                free_bytes=_free_b,
                            )
                        except Exception:
                            pass

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
                    max_cluster_now = regroup_stats.get("max_cluster_size", 0)
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

                    # [제거됨] 안전장치 2(active_ratio 5epoch 연속 저하 시
                    # 조기종료)는 dead-code 재초기화(regroup_update의
                    # dead_reinit_patience) 도입으로 제거함 — 이제 낮은
                    # active_ratio는 방치되는 게 아니라 적극적으로 복구를
                    # 시도하는 대상이라, "회복 불가능한 상태"를 조기에
                    # 판별해 시간을 아끼려던 이 장치의 존재 이유 자체가
                    # 약해짐. 게다가 이 장치의 임계값(5epoch 연속)이
                    # dead_reinit_patience(기본 5)와 정확히 같아서, 재초기화가
                    # 막 개입하려는 타이밍에 먼저 학습을 끊어버리는 경쟁
                    # 상황이 실측으로 확인됨. 안전장치 1(OOM 방지)은 그대로
                    # 유지 — 그건 collapse 여부와 무관하게 실제 메모리
                    # 크래시를 막는 별개의 안전장치.

                    self.model.memory.cache_sample_groups(
                        self.model.prototype_layer.sample_groups,
                        device=torch.device(self.device),
                        centroid_emb=self.model.prototype_layer.centroid_emb,
                    )

                    if epoch % self.regroup_log_every == 0:
                        _reinit = regroup_stats.get('reinit_count', 0)
                        pbar.write(
                            f"  [Regroup] active={regroup_stats['active_ratio']*100:.0f}%  "
                            f"alive={regroup_stats.get('active_centroids', 0)}  "
                            f"min={regroup_stats['min_cluster_size']}  "
                            f"max={regroup_stats['max_cluster_size']}"
                            + (f"  reinit={_reinit}" if _reinit > 0 else "")
                        )

            avg_loss = (tr_loss_gpu / max(n_batch, 1)).item()  # [최적화] 에폭당 딱 1회만 동기화

            # [진단용] log_branch_gradients — 이번 epoch의 브랜치별 gradient/
            # activation norm 평균 + head 첫 Linear의 block별 weight norm
            # (이 epoch 끝 시점 스냅샷)을 함께 기록. 셋을 같이 봐야 "학습
            # 신호가 적었고, 그 결과 head도 구조적으로 그 브랜치에 작은
            # weight만 배정했다"는 걸 gradient 하나만으로는 못 하는 방식으로
            # 뒷받침할 수 있음(gradient 작다≠head가 그 브랜치를 안 쓴다 —
            # 위 tabera.py forward()의 주의사항 참고, 최종 판단은 반드시
            # --ablation *_shuffle/zero 결과와 같이 볼 것).
            if self.log_branch_gradients and _branch_batches:
                epoch_record: Dict[str, float] = {"epoch": float(epoch)}
                for name in _branch_grad_sum:
                    _n = _branch_batches.get(name, 0)
                    if _n == 0:
                        continue
                    epoch_record[f"{name}_grad_norm"] = _branch_grad_sum[name] / _n
                    epoch_record[f"{name}_act_norm"]  = _branch_act_sum[name] / _n
                    epoch_record[f"{name}_n_batches"] = float(_n)  # 진단용 — 이 브랜치가
                    # 이번 epoch 몇 배치에서 실제로 gradient를 받았는지(agg는
                    # 학습 극초반 memory warmup 중엔 전체 배치보다 적을 수 있음).
                with torch.no_grad():
                    W = self.model._head_first_linear.weight  # (out, in)
                    for name, (s, e) in self.model._head_block_slices.items():
                        epoch_record[f"{name}_weight_norm"] = float(W[:, s:e].norm().item())
                self.branch_gradient_history.append(epoch_record)
                if epoch % self.regroup_log_every == 0:
                    _names = list(_branch_grad_sum.keys())
                    pbar.write(
                        "  [BranchGrad] " + "  ".join(
                            f"{n}: grad={epoch_record[f'{n}_grad_norm']:.4f} "
                            f"act={epoch_record[f'{n}_act_norm']:.4f} "
                            f"W={epoch_record[f'{n}_weight_norm']:.4f}"
                            for n in _names
                        )
                    )

            # [진단용] log_evidence_stats — evidence_w의 epoch 평균 entropy/
            # dominant weight 기록. entropy가 0에 가깝고 dominant가 1에 가까울수록
            # ②가 사실상 1개 이웃만 보는 hard 1-NN으로 붕괴했다는 뜻.
            if self.log_evidence_stats and _evidence_batches > 0:
                _ent_avg = _evidence_entropy_sum / _evidence_batches
                ev_record = {
                    "epoch": float(epoch),
                    "entropy": _ent_avg,
                    "n_eff": float(np.exp(_ent_avg)),  # [추가] "실질적으로 몇 명의
                    # 이웃을 보는가" — entropy보다 직관적(entropy=0.05 → n_eff≈1.05명).
                    "dominant_weight": _evidence_dominant_sum / _evidence_batches,
                }
                if _evidence_diag_batches > 0:
                    ev_record["query_norm"]    = _evidence_qnorm_sum / _evidence_diag_batches
                    ev_record["key_norm"]      = _evidence_knorm_sum / _evidence_diag_batches
                    ev_record["distance_mean"] = _evidence_dist_mean_sum / _evidence_diag_batches
                    ev_record["distance_std"]  = _evidence_dist_std_sum / _evidence_diag_batches
                # [진단용, 추가] self-retrieval rate — 배치 없으면(warmup fallback
                # 뿐이었던 epoch 등) 키 자체를 안 넣음(0.0으로 채우면 "확인했는데
                # 0%"와 "애초에 못 쟀음"이 구분 안 되므로).
                if _self_retrieval_batches > 0:
                    ev_record["self_retrieval_top1_rate"] = _self_retrieval_top1_sum / _self_retrieval_batches
                    ev_record["self_retrieval_topk_rate"] = _self_retrieval_topk_sum / _self_retrieval_batches
                self.evidence_stats_history.append(ev_record)
                if epoch % self.regroup_log_every == 0:
                    _extra = ""
                    if "query_norm" in ev_record:
                        _extra = (f"  qnorm={ev_record['query_norm']:.2f} "
                                  f"knorm={ev_record['key_norm']:.2f} "
                                  f"dist={ev_record['distance_mean']:.2f}±{ev_record['distance_std']:.2f}")
                    if "self_retrieval_top1_rate" in ev_record:
                        _extra += (f"  self_top1={ev_record['self_retrieval_top1_rate']*100:.2f}% "
                                  f"self_topk={ev_record['self_retrieval_topk_rate']*100:.2f}%")
                    pbar.write(
                        f"  [EvidenceStats] entropy={ev_record['entropy']:.4f}  "
                        f"n_eff={ev_record['n_eff']:.3f}  "
                        f"dominant_weight={ev_record['dominant_weight']:.4f}{_extra}"
                    )

            # [진단용, 추가] log_fusion_trajectory — epoch 마지막 alpha/beta
            # 값(그 시점 nn.Parameter/buffer 그대로, 파라미터는 "지금 어디
            # 있는가"가 중요해 평균 대신 마지막 값을 씀) + 이 epoch 동안의
            # grad 절대값 평균(grad는 매 배치 존재하는 신호라 평균이 자연스러움)
            # + branch norm/cos/combined-norm 평균(순수 forward 통계).
            # fusion_mode!="residual"이면 model.fusion_alpha/beta 속성 자체가
            # 없거나 None이므로 getattr로 방어.
            if self.log_fusion_trajectory and _fusion_norm_batches > 0:
                _alpha_val = getattr(self.model, "fusion_alpha", None)
                _beta_val  = getattr(self.model, "fusion_beta", None)
                fusion_record = {
                    "epoch": float(epoch),
                    "alpha": float(_alpha_val.detach().item()) if _alpha_val is not None else None,
                    "beta":  float(_beta_val.detach().item())  if _beta_val  is not None else None,
                    "mean_alpha_grad": (
                        _fusion_alpha_grad_sum / _fusion_alpha_grad_batches
                        if _fusion_alpha_grad_batches > 0 else None
                    ),
                    "mean_beta_grad": (
                        _fusion_beta_grad_sum / _fusion_beta_grad_batches
                        if _fusion_beta_grad_batches > 0 else None
                    ),
                    "query_norm_mean": _fusion_qnorm_sum / _fusion_norm_batches,
                    "context_norm_mean": (
                        _fusion_cnorm_sum / _fusion_cnorm_batches if _fusion_cnorm_batches > 0 else None
                    ),
                    "agg_norm_mean": _fusion_anorm_sum / _fusion_norm_batches,
                    "combined_norm_mean": (
                        _fusion_combined_norm_sum / _fusion_combined_batches
                        if _fusion_combined_batches > 0 else None
                    ),
                    "cos_qc_mean": (
                        _fusion_cos_qc_sum / _fusion_cos_qc_batches if _fusion_cos_qc_batches > 0 else None
                    ),
                    "cos_qa_mean": (
                        _fusion_cos_qa_sum / _fusion_cos_qa_batches if _fusion_cos_qa_batches > 0 else None
                    ),
                    "cos_ca_mean": (
                        _fusion_cos_ca_sum / _fusion_cos_ca_batches if _fusion_cos_ca_batches > 0 else None
                    ),
                }
                self.fusion_trajectory_history.append(fusion_record)
                if epoch % self.regroup_log_every == 0:
                    _a, _b = fusion_record["alpha"], fusion_record["beta"]
                    _ag, _bg = fusion_record["mean_alpha_grad"], fusion_record["mean_beta_grad"]
                    _qn, _cn, _an = (fusion_record["query_norm_mean"], fusion_record["context_norm_mean"],
                                     fusion_record["agg_norm_mean"])
                    _zn = fusion_record["combined_norm_mean"]
                    _a_str  = f"{_a:.4f}" if _a is not None else "N/A"
                    _b_str  = f"{_b:.4f}" if _b is not None else "N/A"
                    _ag_str = f"{_ag:.5f}" if _ag is not None else "N/A"
                    _bg_str = f"{_bg:.5f}" if _bg is not None else "N/A"
                    _line1 = (f"  [FusionTraj] alpha={_a_str}(grad={_ag_str}) beta={_b_str}(grad={_bg_str})  "
                              f"||q||={_qn:.3f}  ||a||={_an:.3f}")
                    if _cn is not None:
                        _line1 += f"  ||c||={_cn:.3f}"
                    if _zn is not None:
                        _line1 += f"  ||q+αc+βa||={_zn:.3f}"
                    pbar.write(_line1)
                    _cqc, _cqa, _cca = (fusion_record["cos_qc_mean"], fusion_record["cos_qa_mean"],
                                        fusion_record["cos_ca_mean"])
                    if _cqc is not None or _cqa is not None or _cca is not None:
                        _line2 = "    cos: "
                        if _cqc is not None: _line2 += f"qc={_cqc:+.3f}  "
                        if _cqa is not None: _line2 += f"qa={_cqa:+.3f}  "
                        if _cca is not None: _line2 += f"ca={_cca:+.3f}"
                        pbar.write(_line2)

            # ── 검증 ──────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                # [버그 수정] X_val을 항상 고정된 순서로 처리하면, 특정 구간에
                # 비슷한 샘플이 몰려있을 경우 그 구간 배치들이 매 epoch 계속
                # 같은(적은 수의) centroid로만 라우팅되어 U는 작고
                # local_max_g만 큰 최악의 조합이 반복되는 현상이 실측 확인됨
                # (val_forward가 1초→76초까지 폭증). 학습은 randperm으로 매
                # epoch 섞여서 이런 편중이 평균화되는데 검증만 고정 순서였던
                # 것이 원인 중 하나 — 매 epoch 셔플해서 동일하게 평균화되게 함
                # (집계 지표 계산에는 순서가 무관하므로 결과에 영향 없음).
                _val_perm  = torch.randperm(len(X_val), device=self.device)
                val_logits = self._forward_batched(X_val[_val_perm])
                val_m  = compute_metric(val_logits, y_val[_val_perm], self.tasktype)
            val_v = list(val_m.values())[0]

            # [버그 수정] regroup_warmup_epochs 도입 후 발견된 문제 — warmup
            # 중엔 sample_groups가 아직 한 번도 발행 안 된 상태([[],[],...])라,
            # 이 시점 val이 우연히 좋게 나오면(retrieve()가 그룹 제약 없이
            # 사실상 global 검색처럼 동작해서 오히려 잘 나올 수 있음) 그
            # "그룹 없는" 스냅샷이 best_state로 뽑혀버린다. 그러면 이후
            # epoch에서 sample_groups가 정상적으로 채워져도 이미 best가
            # 아니라 반영이 안 되고, patience 안에 그 val을 못 넘기면 그대로
            # 조기 종료되어 최종 모델이 영원히 빈 sample_groups로 굳는다
            # (실측: vehicle rwe5에서 epoch 33 조기 종료, "검증 가능한
            # 그룹이 없어 일치율 계산 불가"). regroup_update()가 이번 epoch
            # 안에서 이미 호출됐으므로(위), current_epoch이 warmup을
            # 지났는지로 "이번 epoch에 sample_groups가 실제로 발행됐는가"를
            # 판단해 best_state 후보에서 제외한다.
            _past_regroup_warmup = (
                self.model.prototype_layer.current_epoch.item()
                >= self.model.prototype_layer.regroup_warmup_epochs
            )

            # best 모델 저장
            if is_better(val_v, best_val, self.tasktype) and _past_regroup_warmup:
                best_val   = val_v
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                # [버그 수정] sample_groups/group_labels(CentroidLayer의 일반
                # Python 속성)와 feature_store._store(nn.Module이 아님)는 위
                # state_dict()에 포함되지 않는다. load_state_dict()로
                # centroid_emb/memory.keys 등은 "best 검증 epoch" 시점으로
                # 되돌아가는데, 이 셋만 "마지막 학습 epoch" 시점에 남아있게
                # 되어 서로 다른 시점의 스냅샷이 섞이는 문제가 있었음
                # (reproduce.py의 dual_space_faithfulness 사전 검증에서
                # sample_groups 재배정 일치율이 무작위 수준으로 나온 원인).
                # → best_state와 함께 별도로 스냅샷/복원한다.
                #
                # [최적화] label_all_groups/label_groups_by_target를 여기로
                # 옮김 — 최종적으로 남는 건 어차피 "이 시점(새 베스트)의" 값
                # 뿐이므로(다음 epoch에 개선이 없으면 이 값이 계속 최종본으로
                # 남고, 개선되면 그때 다시 계산돼 덮어씀), 매 epoch 계산하던
                # 이전 방식과 최종 결과가 100% 동일하면서 개선 없는 epoch의
                # 계산만 생략된다. x_regroup/y_regroup는 바로 이번 epoch의 regroup_update()
                # 블록에서 만들어진 것이라 MemoryBank 시점이 정확히 일치함
                # (학습 다 끝난 뒤 재계산하면 MemoryBank가 그 사이 더 갱신돼
                # 시점이 어긋날 수 있어 — 그 방식은 채택 안 함).
                if (
                    self.cat_cols is not None
                    and self.num_cols is not None
                    and self.col_names is not None
                    and x_regroup is not None
                ):
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

        if best_state is None:
            # [추가] regroup_warmup_epochs가 너무 길어서(또는 patience가
            # 너무 짧아서) warmup을 한 번도 못 지나고 조기 종료된 경우 —
            # best_state가 끝까지 채워지지 않는다. 이 경우 모델은 학습
            # 마지막 시점 가중치를 그대로 쓰게 되는데, sample_groups가
            # 여전히 비어있을 수 있어 explanation/group-constrained
            # retrieval이 정상 동작 안 할 수 있음을 알림.
            tqdm.write(
                f"  ⚠️  best_state가 한 번도 갱신되지 않았습니다 — "
                f"regroup_warmup_epochs({self.model.prototype_layer.regroup_warmup_epochs})가 "
                f"조기 종료 시점보다 길었을 가능성이 있습니다. warmup을 줄이거나 "
                f"patience를 늘려서 재시도하세요."
            )

        if best_state:
            self.model.load_state_dict(best_state)
            # [버그 수정] state_dict에 없는 sample_groups/feature_store도
            # 같은 best epoch 시점으로 함께 복원 — centroid_emb/memory.keys와
            # 시점이 어긋나지 않도록 함.
            # [참고] memory.keys/labels/ptr/filled/sample_ids는 모두
            # nn.Module buffer라 위 load_state_dict() 한 줄로 이미 best
            # epoch 시점으로 복원됨 — feature_store(아래)와 항상 같은
            # 시점이 되도록, 그 다음 순서로 feature_store를 복원한다.
            if best_sample_groups is not None:
                self.model.prototype_layer.sample_groups = best_sample_groups
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
            # [추가] memory(위에서 이미 복원됨)와 feature_store(방금 복원됨)가
            # 모두 같은 best epoch 시점이 된 뒤에만 refresh를 실행한다 —
            # 순서가 바뀌면 refresh_memory_keys() 내부 assert(filled 일치)가
            # 즉시 잡아낸다.
            if self.refresh_on_best:
                refresh_stats = self.model.refresh_memory_keys()
                if refresh_stats is not None:
                    tqdm.write(f"  [refresh_on_best] memory.keys {refresh_stats['n_refreshed']}개 "
                               f"슬롯을 frozen weight로 재계산 완료")
                    regroup_stats = self._resync_groups_after_refresh()
                    if regroup_stats is not None:
                        tqdm.write(f"  [refresh_on_best] clean 임베딩 기준으로 sample_groups 재동기화 "
                                   f"완료 (active={regroup_stats.get('active_ratio', 0)*100:.0f}%, "
                                   f"reinit={regroup_stats.get('reinit_count', 0)})")
        self._best_state = best_state

        # ── centroid margin z-score 진단 (best epoch 복원된 모델 기준) ──
        # reproduce.py --ablation centroid_geometry와 동일 로직. 매 HPO
        # trial마다 자동 계산해 self.centroid_geometry_diag에 저장 —
        # optimize.py의 objective가 val_v(정확도/logloss)와 함께 이 값을
        # (예: routing_scale이 너무 낮아 z_margin이 무작위보다도 낮게 나온
        # trial에 페널티를 주는 방식으로) 반영할지는 optimize.py 쪽에서
        # 결정한다 — search space(routing_scale 탐색 범위) 자체는 안 건드림.
        self.centroid_geometry_diag = self._compute_centroid_margin_zscore(X_val)

        # [추가, penalty 미반영] 학습 과정 전체의 라우팅 안정성 지표.
        # centroid_geometry_diag(위)는 best epoch 하나의 스냅샷만 보므로,
        # "그 스냅샷 하나는 좋았지만 학습 내내 계속 흔들리다 우연히 멈춘"
        # trial(credit-g trial #47 실측 사례)을 못 잡아낸다. 이 두 값이
        # 실제로 나쁜 결과(불안정한 재현성, 낮은 test 성능 등)와 상관관계가
        # 있는지 먼저 여러 trial에 걸쳐 확인한 뒤, 유의미하면 그때 penalty에
        # 반영할지 결정한다(바로 반영하지 않는 이유) — z_margin penalty
        # 임계값을 검증 없이 정했다가 두 번 다시 만졌던 전례를 반복하지
        # 않기 위함.
        if self.centroid_geometry_diag is not None:
            n_epochs_seen = max(1, len(_active_ratio_history))
            self.centroid_geometry_diag["reinit_per_epoch"] = _reinit_total / n_epochs_seen
            self.centroid_geometry_diag["active_ratio_std"] = (
                float(np.std(_active_ratio_history)) if len(_active_ratio_history) > 1 else 0.0
            )
            # [추가] 학습 종료 시점(best_state 복원된 상태)의 centroid 간
            # 평균 코사인 거리 — initialize_from_data()가 KMeans++ 직후
            # 찍는 로그와 정확히 같은 정의(1 - cosine_sim, 대각선 제외
            # 평균)라 "학습 시작 vs 끝"을 그대로 비교할 수 있음. --ema_codebook
            # 도입 계기: EMA를 쓰면 diversity_loss(centroid 서로 밀어내기)가
            # 자동으로 꺼지므로, centroid들이 서로 가까워지며 뭉치는(붕괴)
            # 부작용이 생기는지 이 값으로 직접 확인 가능 — 학습 시작보다
            # 끝에서 이 값이 뚜렷이 작아졌다면 그 부작용이 실제로 발생한 것.
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
        top1-top2 query-centroid cosine similarity margin이 완전 무작위
        (학습 전혀 안 된) centroid/query 벡터로 만든 null 베이스라인과
        비교해 얼마나 유의하게 다른지 z-score로 진단.

        [배경] routing_scale은 forward(예측)에는 영향이 없지만(STE라서
        hard_assignment는 양수 스케일에 불변), 학습 중 STE backward
        gradient의 뾰족함에는 직접 영향을 준다 — routing_scale이 낮으면
        gradient가 여러 centroid 방향으로 뭉근하게 blend되어, 학습이
        끝나도 query가 centroid 주변에 뚜렷하게 뭉치지 못하고 심하면
        margin이 무작위보다도 더 좁아지는 현상이 실측 확인됨(credit-g,
        routing_scale=1.49: z_margin=-3.40 — 무작위보다 유의하게 나쁨).
        반대로 routing_scale이 큰 데이터셋(socmob 19.8, SpeedDating
        13.77)은 z_margin이 +18~+22로 무작위보다 압도적으로 큼.

        Returns
        ───────
        None이면 prototype_layer가 없거나(ablation 등) P<2라 계산 불가.
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

        # null 베이스라인 — 완전 무작위 벡터, 동일 D/P/n_val 조건, CPU에서
        # (모델과 무관한 순수 텐서 연산이라 GPU 동기화 비용 없이 저렴함)
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

        # [percentile — z-score와 별개로 직접 계산] z-score는 정규분포
        # 근사가 깔려있고, "얼마나 유의하면 봐줄지" 임계값(예: z=2.0)을
        # 어디에 둘지가 결국 다시 임의적인 선택이 됨. 이미 50개 null 샘플을
        # 실제로 갖고 있으니, 정규분포 가정도 임계값도 없이 "실측 margin이
        # 이 50개 중 몇 %보다 나은가"를 직접 셀 수 있음 — HPO penalty(아래
        # optimize.py)는 이 percentile을 그대로 연속값으로 써서, "몇 z
        # 이상이면 괜찮다"는 문턱값 자체를 없앤다.
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

    # ── 배치 추론 ────────────────────────────────────────────

    def _forward_batched(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        # [버그 수정] 기본값 1024가 학습 배치 크기(HPO가 고른 128~512)보다
        # 커서, centroid collapse로 그룹이 비대해진 상황에서 검증이 학습보다
        # 훨씬 느려지는 현상이 실측됨(val_forward가 몇 epoch 만에 1초→76초로
        # 폭증, 학습 자체는 5~6초로 안정적이었음). 배치가 클수록 그 배치 안에
        # 거대 그룹을 가리키는 쿼리가 포함될 확률이 높아지고, retrieve()의
        # U/local_max_g 라운딩까지 겹쳐 배치별 텐서가 학습 때보다 커짐.
        # 검증도 학습과 같은 배치 크기를 쓰도록 통일.
        if batch_size is None:
            batch_size = self.params.get("batch_size", 512)
        parts = []
        for start in range(0, len(X), batch_size):
            parts.append(self.model(X[start:start + batch_size])["logits"])
        return torch.cat(parts, dim=0)