## Main file for reproducing the best TabERA configuration.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import sys, os, argparse

# ── CUDA_VISIBLE_DEVICES: torch import 전 설정 ──────────────
_parser_pre = argparse.ArgumentParser(add_help=False)
_parser_pre.add_argument("--gpu_id", type=int, default=0)
_pre, _ = _parser_pre.parse_known_args()
if _pre.gpu_id >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre.gpu_id)

import joblib, json, pickle, datetime
import numpy as np
import torch
from pathlib import Path

from libs.data         import TabularDataset
from libs.search_space import params_to_model_kwargs
from libs.supervised   import TabERAWrapper
from libs.tabera         import TabERA
from libs.eval         import calculate_metric
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ─────────────────────────────────────────────────────────────
# 설명 출력 (①② architectural + ③ IG post-hoc)
# ─────────────────────────────────────────────────────────────

def print_explanation(explanations: list, sample_idx: int, col_names: list) -> None:
    e = explanations[sample_idx]

    print(f"\n{'━'*52}")
    print(f"  TabERA Explanation — Sample #{sample_idx}")
    print(f"{'━'*52}")

    # ① 프로토타입 그룹 (centroid_features — 역정규화 없이 원본값 표시)
    proto = e["prototype"]
    print(f"\n  ① 프로토타입 그룹")
    print(f"     → \"{proto['assigned_group']}\"  (confidence={proto['group_confidence']:.1%})")
    if proto["runners_up"]:
        ru = ", ".join(f"\"{l}\"({s:.1%})" for l, s in proto["runners_up"])
        print(f"     Runner-up: {ru}")

    # centroid 원본 feature 값 출력 (Medoid 기반 대표 사례)
    cf = proto.get("centroid_features", {})
    if cf:
        feat_str = ",  ".join(
            f"{k}={v:.3f}" for k, v in sorted(cf.items(), key=lambda x: -abs(x[1]))[:6]
        )
        print(f"     대표 사례: {feat_str}")
        print(f"     (그룹 내 centroid 최근접 실제 훈련 샘플)")

    # ② 이웃 증거 (Attention weight)
    ev = e["evidence"]
    print(f"\n  ② 이웃 증거 (Attention)")
    print(f"     dominant={ev['dominant_weight']:.1%},  entropy={ev['entropy']:.3f}")
    for rank, (idx, w) in enumerate(ev["top_neighbours"]):
        print(f"     #{rank+1} Neighbour {idx}: {w:.1%}")

    # 이웃의 원본 feature 값 (FeatureStore에서 조회된 경우)
    nf = e.get("neighbour_features")
    if nf:
        for rank, (idx, w) in enumerate(ev["top_neighbours"][:3]):
            if rank < len(nf):
                feat_str = ", ".join(f"{k}={v:.3f}" for k, v in list(nf[idx].items())[:4])
                print(f"        → {feat_str}")

    print(f"{'━'*52}")


# ─────────────────────────────────────────────────────────────
# Integrated Gradients (Sundararajan et al. 2017, ICML)
# ─────────────────────────────────────────────────────────────
#
# [1-step Gradient×Input과의 차이]
# 1-step: grad(x) * (x - baseline)
#   → α=1 지점(원본 입력)의 gradient만 사용
#   → sigmoid/softmax가 saturate된 영역에서 gradient ≈ 0
#   → Sensitivity, Completeness axiom을 만족하지 못함
#     (Shrikumar et al. 2016, "Gradient * Input"과 동일한 방법)
#
# Multi-step IG: baseline → input 경로를 n_steps개 지점에서 샘플링,
#   각 지점의 gradient를 평균 → (x - baseline)과 곱함
#   → fundamental theorem of calculus에 의해 Completeness axiom 만족
#   → saturation 구간도 경로 적분으로 우회 가능
def compute_integrated_gradients(
    model, X, X_baseline, target_fn, n_steps: int = 20,
    check_convergence: bool = False,
):
    """
    Parameters
    ──────────
    model      : TabERA 모델 (eval 모드)
    X          : (N, F) 입력 배치
    X_baseline : (F,) 또는 (N, F) baseline
    target_fn  : model(x) 출력(dict)을 받아 (N,) 형태의 per-sample 스칼라를
                 리턴하는 함수. 배치를 sum()해서 단일 스칼라로 합치면 안 됨
                 (개별 샘플의 completeness 검증이 불가능해짐).
                 예: lambda out: out["logits"].gather(1, target_class.unsqueeze(1)).squeeze(1)
                     (multiclass, 예측 클래스의 logit)
                 예: lambda out: out["logits"].squeeze(-1)
                     (binclass/regression, 단일 출력)
    n_steps    : 적분 근사에 사용할 step 수 (기본 20)
    check_convergence : True면 Completeness axiom 오차를 샘플별로 측정해 출력
                 (IG_i(x).sum() ≈ f(x) - f(baseline) 이어야 함 — Riemann sum
                 근사 오차이므로 n_steps가 부족하면 이 값이 커짐)

    Returns
    ──────────
    attribution : (N, F) — |IG| (절댓값, 순위 비교용)
    """
    if X_baseline.dim() == 1:
        X_baseline = X_baseline.unsqueeze(0).expand_as(X)

    alphas = torch.linspace(0.0, 1.0, n_steps, device=X.device)
    grads_accum = torch.zeros_like(X)

    for alpha in alphas:
        x_interp = (X_baseline + alpha * (X - X_baseline)).clone().detach().requires_grad_(True)
        out = model(x_interp)
        target = target_fn(out)                              # (N,) per-sample
        # 배치의 각 샘플 출력은 서로 다른 입력에서 나오므로 (배치 내 cross term 없음),
        # target.sum()의 gradient = 각 샘플 target의 gradient를 합친 것과 동일.
        # 이렇게 하면 1회 backward로 (N, F) gradient를 모두 얻으면서도
        # per-sample completeness 검증에 필요한 분리된 의미를 유지함.
        grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
        grads_accum = grads_accum + grad

    avg_grad = grads_accum / n_steps
    ig_signed = avg_grad * (X - X_baseline)             # (N, F) 부호 보존 (completeness 검증용)

    if check_convergence:
        # Completeness axiom (per-sample): Σ_i IG_i(x) ≈ f(x) - f(baseline)
        # 오차가 클수록 n_steps가 적분 근사에 부족하다는 신호.
        # 배치 합산이 아니라 샘플별로 직접 비교해야 상쇄/증폭으로 인한
        # 진단 오류를 피할 수 있음.
        with torch.no_grad():
            f_x        = target_fn(model(X))                 # (N,)
            f_baseline = target_fn(model(X_baseline))         # (N,)
        ig_sum_per_sample  = ig_signed.sum(dim=-1)             # (N,)
        actual_diff_per_sample = f_x - f_baseline              # (N,)

        abs_error = (ig_sum_per_sample - actual_diff_per_sample).abs()
        rel_error = abs_error / (actual_diff_per_sample.abs() + 1e-8)

        print(f"    [IG convergence check] n_steps={n_steps}  "
              f"mean|Σ IG_i - (f(x)-f(baseline))| = {abs_error.mean().item():.4f}  "
              f"(median relative: {rel_error.median().item():.2%}, "
              f"mean relative: {rel_error.mean().item():.2%})")

    return ig_signed.abs().detach()


def make_logit_target_fn(tasktype: str, target_class=None):
    """
    compute_integrated_gradients에 넘길 per-sample target_fn 생성.

    IG는 "최종 예측"에 대한 feature 기여도를 측정해야 하므로,
    agg_emb(중간 retrieval 표현)가 아니라 logits을 target으로 삼는다.

    Parameters
    ──────────
    tasktype     : "regression" | "multiclass" | 그 외(binclass)
    target_class : multiclass일 때 각 샘플의 추적 대상 클래스 인덱스, shape (N,)
                   numpy array 또는 torch tensor. None이면 호출 시점에
                   model 출력에서 argmax로 자동 결정(단, baseline 평가 시
                   클래스가 바뀌면 completeness가 깨지므로 고정해서 넘기는
                   것을 권장).
    """
    if tasktype == "regression":
        return lambda out: out["logits"].squeeze(-1)            # (N,)

    elif tasktype == "multiclass":
        def _fn(out):
            logits = out["logits"]                              # (N, C)
            if target_class is None:
                tc = logits.argmax(dim=-1)
            else:
                tc = torch.as_tensor(target_class, device=logits.device, dtype=torch.long)
            return logits.gather(1, tc.unsqueeze(1)).squeeze(1)  # (N,)
        return _fn

    else:  # binclass — 단일 logit, 그 자체가 곧 class=1 방향의 점수
        return lambda out: out["logits"].squeeze(-1)             # (N,)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TabERA Reproduce Best Config")
    parser.add_argument("--gpu_id",    type=int, default=0)
    parser.add_argument("--openml_id", type=int, required=True)
    parser.add_argument("--savepath",  type=str, default=".",
                        help="optim_logs가 있는 상위 경로")
    parser.add_argument("--seed",      type=int, default=1,
                        help="optimize.py와 동일한 seed 사용")
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    parser.add_argument("--epochs",    type=int, default=200)
    parser.add_argument("--patience",  type=int, default=30)
    parser.add_argument("--n_explain", type=int, default=3,
                        help="설명 출력할 테스트 샘플 수")
    parser.add_argument("--explain",   action="store_true",
                        help="학습 후 feature 기여도 설명 출력")
    parser.add_argument("--ablation",  type=str, default="none",
                        choices=["none", "random_neighbor",
                                 "rank_correlation", "dual_space_faithfulness",
                                 "deletion_auc", "insertion_auc",
                                 "value_diagnosis"],
                        help=(
                            "ablation 모드 선택 (학습된 모델에 inference 단계에서 적용):\n"
                            "  none                  : full model 기준 (기본값)\n"
                            "  random_neighbor       : neighbor 임베딩 랜덤 교체\n"
                            "  rank_correlation      : IG feature 순위 vs 실제 prediction\n"
                            "                         영향력 순위 Spearman 상관계수\n"
                            "                         (TabERA vs SHAP vs Random 3자 비교)\n"
                            "  dual_space_faithfulness : centroid_x 대표성 + 그룹 분리도 검증\n"
                            "  deletion_auc          : attribution 순위로 feature 누적 마스킹 →\n"
                            "                         ŷ 곡선의 AUC (낮을수록 좋음)\n"
                            "  insertion_auc         : baseline에서 시작 → 중요 feature부터 복원 →\n"
                            "                         ŷ 곡선의 AUC (높을수록 좋음)\n"
                            "                         Deletion과 짝 (Petsiuk et al. 2018, RISE)\n"
                            "  value_diagnosis        : AttentionAggregator의 value 구성 진단 —\n"
                            "                         value=label_emb+T(query-neighbour)에서\n"
                            "                         T() 항이 label_emb 대비 얼마나 큰지 측정\n"
                            "                         (재학습 없음, 저비용 사전 진단)"
                        ))
    parser.add_argument("--no_offset_correction", action="store_true",
                        help=(
                            "[ablation] optimize.py --no_offset_correction으로 학습한 "
                            "study를 불러와 재현. T(query-neighbour) 오프셋 보정 없이 "
                            "value=label_emb만 사용한 모델. optimize.py와 반드시 일치시켜야 "
                            "같은 study 파일을 정확히 찾음."
                        ))
    parser.add_argument("--global_retrieve", action="store_true",
                        help=(
                            "[진단용] 기존(그룹-제약) study의 best_params를 그대로 불러오되, "
                            "retrieve()만 그룹 제약 없이 전역 검색으로 바꿔서 1회 재학습. "
                            "별도 study를 요구하지 않음 — --no_offset_correction과 달리 "
                            "불러오는 study 파일은 바뀌지 않고, 모델 구성에만 반영됨."
                        ))
    parser.add_argument("--no_context_emb", action="store_true",
                        help=(
                            "[진단용] context_emb(설명① 신호)를 head 입력에서 제외하고 "
                            "1회 재학습. STE 라우팅/centroid 학습 자체는 그대로 유지됨 "
                            "(diversity_loss/commitment_loss는 계속 작동) — 'context_emb가 "
                            "head에 보이는 것 자체가 예측에 얼마나 기여하는지'만 격리해서 "
                            "측정. --global_retrieve와 마찬가지로 별도 study 불필요."
                        ))
    parser.add_argument("--detach_context_grad", action="store_true",
                        help=(
                            "[진단용] context_emb는 head 입력으로 그대로 전달하되, "
                            "그쪽에서 오는 gradient만 centroid_emb로 안 흐르게 끊음 "
                            "(commitment_loss는 원래도 detach라 영향 없음, diversity_loss "
                            "gradient는 그대로 흐름). 'task_loss와 diversity_loss가 "
                            "centroid_emb를 두고 서로 다른 방향으로 당기며 충돌하고 있는지' "
                            "검증용. --no_context_emb와 동시 사용 시 의미 없음(그땐 애초에 "
                            "context_emb가 head에 안 들어감)."
                        ))
    parser.add_argument("--context_projection", action="store_true",
                        help=(
                            "[구조 조정] context_emb를 head로 보내기 전 학습 가능한 "
                            "Linear를 하나 거치게 함. detach_context_grad와 달리 "
                            "gradient가 여전히 centroid_emb까지 도달함. optimize.py "
                            "--context_projection으로 학습한 study가 있으면 그 "
                            "best_params를 쓰는 게 이상적이지만, 없으면 기존 study "
                            "best_params 위에 이 구조만 얹어 1회 재학습(별도 study 불필요)."
                        ))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── 데이터 로드 ────────────────────────────────────────
    with open(args.json, "r") as f:
        data_info = json.load(f)

    openml_id    = str(args.openml_id)
    dataset_info = data_info[openml_id]
    tasktype     = dataset_info["tasktype"]
    print(f"[TabERA Reproduce] {dataset_info['fullname']} (id={openml_id}, task={tasktype})")

    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std      = dataset.y_std
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")

    # ── best params 로드 ───────────────────────────────────
    # optimize.py 저장 경로와 동일하게 맞춤
    if not args.savepath.endswith("optim_logs"):
        log_dir = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
    else:
        log_dir = args.savepath

    _ablation_tag = "..no_offset" if args.no_offset_correction else ""
    fname = os.path.join(log_dir, f"data={openml_id}{_ablation_tag}..model=tabera.pkl")
    if not os.path.exists(fname):
        _flag_hint = " --no_offset_correction" if args.no_offset_correction else ""
        _hint_cmd = f"optimize.py --openml_id {openml_id} --seed {args.seed}{_flag_hint}"
        raise FileNotFoundError(
            f"최적화 로그 없음: {fname}\n"
            f"먼저 {_hint_cmd} 를 실행하세요."
        )
    # 출력 파일명 태그: global_retrieve는 별도 study를 요구하지 않으므로
    # 로딩(_ablation_tag)엔 영향 없이, 저장 파일명에만 추가로 반영.
    _save_tag = _ablation_tag + ("..global_retrieve" if args.global_retrieve else "") \
                              + ("..no_context" if args.no_context_emb else "") \
                              + ("..detach_ctx" if args.detach_context_grad else "") \
                              + ("..ctx_proj" if args.context_projection else "")

    study       = joblib.load(fname)
    best_params = study.best_params
    print(f"  Best trial #{study.best_trial.number}  val={study.best_value:.4f}")

    # optimize.py가 실제 사용한 n_prototypes 그대로 복원
    best_params["n_prototypes"] = study.best_trial.user_attrs["n_prototypes_actual"]
    print(f"  n_prototypes (from optimize.py): {best_params['n_prototypes']}")
    print(f"  Params: {best_params}")

    # ── 모델 구성 ──────────────────────────────────────────
    model_kwargs = params_to_model_kwargs(best_params, dataset.n_features, output_dim)
    model = TabERA(
        **model_kwargs,
        column_names=dataset.col_names,
        # [수정] optimize.py와 동일하게 캡 제거 (memory_size가 다르면
        # HPO 때 찾은 best_params가 이 재현 실행에서 재현되지 않음)
        memory_size=len(y_train),
        # [ablation] optimize.py에서 학습할 때 쓴 것과 반드시 일치해야 함
        use_offset_correction=not args.no_offset_correction,
        # [진단용] best_params는 그룹-제약 study에서 그대로 가져오되,
        # retrieve()만 전역 검색으로 바꿈 (context_emb/설명①은 안 바뀜)
        global_retrieve=args.global_retrieve,
        # [진단용] context_emb를 head 입력에서 제외 (STE/centroid 학습은 그대로)
        use_context_emb=not args.no_context_emb,
        # [진단용] context_emb는 head에 그대로 전달하되 gradient만 끊음
        detach_context_grad=args.detach_context_grad,
        # [구조 조정] context_emb를 head 직전 Linear 프로젝션에 통과시킴
        use_context_projection=args.context_projection,
    )

    # ── 학습 ──────────────────────────────────────────────
    wrapper = TabERAWrapper(
        model, best_params, tasktype,
        device=str(device), epochs=args.epochs, patience=args.patience,
    )
    wrapper._data_id = args.openml_id
    wrapper.fit(X_train, y_train, X_val, y_val)

    # ── 평가 ──────────────────────────────────────────────
    preds_val  = wrapper.predict(X_val)
    preds_test = wrapper.predict(X_test)
    probs_val  = wrapper.predict_proba(X_val)  if tasktype != "regression" else None
    probs_test = wrapper.predict_proba(X_test) if tasktype != "regression" else None

    if tasktype == "regression":
        val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, None, tasktype, "val")
        test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, None, tasktype, "test")
    else:
        val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
        test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

    print(f"\n  {env_info}  {openml_id}  {dataset_info['name']}  tabera  {log_dir}")
    print(f"  val  : {val_metrics}")
    print(f"  test : {test_metrics}")

    # ── 예측 확신도(overconfidence) 진단 ──────────────────────
    # deletion/insertion AUC가 multiclass에서 Random과 잘 구별되지 않는 원인 후보:
    # 모델이 거의 항상 한 클래스에 매우 쏠린(overconfident) 예측을 한다면,
    # 개별 feature 하나를 지워도 그 확신이 잘 안 흔들려 deletion 효과가 둔감해질 수 있음
    # (attribution 방법의 문제가 아니라 prediction surface 자체가 saturate된 경우).
    if tasktype != "regression" and probs_test is not None:
        probs_test_cpu = probs_test.detach().cpu() if torch.is_tensor(probs_test) else probs_test
        probs_np = np.asarray(probs_test_cpu)
        if tasktype == "multiclass":
            max_probs = probs_np.max(axis=-1)
        else:  # binclass — predict_proba가 (N,) 또는 (N,2) 형태일 수 있음
            if probs_np.ndim == 2:
                max_probs = probs_np.max(axis=-1)
            else:
                max_probs = np.where(probs_np >= 0.5, probs_np, 1.0 - probs_np)

        print(f"\n  [예측 확신도 진단]")
        print(f"    평균 max_prob : {max_probs.mean():.4f}")
        print(f"    표준편차      : {max_probs.std():.4f}")
        print(f"    median        : {np.median(max_probs):.4f}")
        print(f"    >0.9 비율     : {(max_probs > 0.9).mean()*100:.1f}%")
        print(f"    >0.99 비율    : {(max_probs > 0.99).mean()*100:.1f}%")
        if tasktype == "multiclass":
            n_classes = probs_np.shape[-1]
            print(f"    (참고: uniform이면 max_prob ≈ {1.0/n_classes:.3f}, n_classes={n_classes})")

    # ── Ablation 평가 ──────────────────────────────────────────
    # 학습된 모델 가중치는 고정한 채, inference 단계에서만 ablation 적용.
    # 따라서 별도 재학습 없이 동일 가중치로 3가지 ablation을 빠르게 비교 가능.
    if args.ablation != "none":
        print(f"\n{'='*60}")
        print(f"  Ablation Mode: {args.ablation}")
        print(f"{'='*60}")

        model.eval()

        # ── rank_correlation: IG feature 순위 vs 실제 prediction 영향력 순위 ──
        #
        # [측정 방식]
        # 1. TabERA Integrated Gradients (50-step) → feature별 중요도 순위
        # 2. SHAP KernelExplainer → feature별 중요도 순위
        # 3. Random attribution → baseline
        # 4. 각 feature를 평균값으로 교체 → prediction 변화량(delta) 순위
        # 5. 세 attribution의 순위와 delta 순위의 Spearman 상관계수 비교
        #
        # [왜 이게 semantic faithfulness 근거가 되는가]
        # "중요하다고 판단한 feature를 실제로 바꿨을 때
        #  prediction이 더 많이 바뀐다"는 걸 순위 상관으로 보여줌.
        # TabERA ≥ SHAP >> Random 이면:
        # "TabERA explanation이 SHAP만큼 의미있으면서, prediction path 안에 있다"
        if args.ablation == "rank_correlation":
            import shap
            from scipy.stats import spearmanr

            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            # 샘플 수 제한 (SHAP KernelExplainer가 느림)
            n_rc       = min(100, X_test.shape[0])
            X_rc       = X_test[:n_rc]
            X_rc_np    = X_rc.detach().cpu().numpy()
            X_train_np = X_train.detach().cpu().numpy()

            print(f"\n  Rank Correlation Faithfulness (n={n_rc})")
            print(f"  {'─'*60}")

            # ── Step 1. delta 순위 계산 ─────────────────────────
            # feature 하나씩 훈련셋 평균으로 교체 → logit 변화량 측정
            # delta가 클수록 그 feature가 prediction에 실제로 중요한 것
            print(f"  [1/4] Delta 순위 계산 중 (feature {n_features}개)...")
            with torch.no_grad():
                logits_orig = model(X_rc)["logits"]           # (N, C)
                train_mean  = X_train.mean(dim=0)             # (F,)

                delta_per_feat = []
                for f in range(n_features):
                    X_masked       = X_rc.clone()
                    X_masked[:, f] = train_mean[f]
                    logits_masked  = model(X_masked)["logits"]
                    delta_f        = (logits_orig - logits_masked).abs().mean().item()
                    delta_per_feat.append(delta_f)

            delta_arr  = np.array(delta_per_feat)
            delta_rank = np.argsort(np.argsort(-delta_arr))   # 0-based, 낮을수록 중요

            # ── Step 2. TabERA IG 순위 (Integrated Gradients, multi-step) ──
            #
            # Sundararajan et al. 2017 (ICML)의 정의를 따라 baseline → input
            # 경로를 50-step으로 적분 근사. target은 retrieval 중간 표현인
            # agg_emb가 아니라 최종 logits — "이 feature를 바꾸면 최종 예측이
            # 얼마나 변하는가"를 delta(perturbation 기반)와 같은 좌표계에서 측정.
            # -> 모델 구조/학습 변경 없음, eval 모드에서 50회 backward만 사용.
            print(f"  [2/4] TabERA IG 순위 계산 중 (Integrated Gradients, 50-step)...")

            X_baseline = X_train.mean(dim=0)               # (F,) -- delta 계산과 동일 baseline

            with torch.no_grad():
                _logits_for_class = model(X_rc)["logits"]
                _target_class = (
                    _logits_for_class.argmax(dim=-1).cpu().numpy()
                    if tasktype == "multiclass" else None
                )
            ig_target_fn = make_logit_target_fn(tasktype, target_class=_target_class)

            tabera_imp = compute_integrated_gradients(
                model, X_rc, X_baseline,
                target_fn=ig_target_fn,
                n_steps=50,
                check_convergence=True,
            ).cpu().numpy()  # (N, F)

            if True:
                tabera_mean = tabera_imp.mean(axis=0)          # (F,)
                tabera_rank = np.argsort(np.argsort(-tabera_mean))

                # ── Step 3. SHAP 순위 ───────────────────────────
                print(f"  [3/4] SHAP KernelExplainer 실행 중 (background=50, nsamples=100)...")

                def model_predict(x_np):
                    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits_np = model(x_t)["logits"].cpu().numpy()
                    if tasktype == "multiclass":
                        exp_l = np.exp(logits_np - logits_np.max(-1, keepdims=True))
                        return exp_l / exp_l.sum(-1, keepdims=True)
                    elif tasktype == "binary":
                        return 1 / (1 + np.exp(-logits_np))
                    else:
                        return logits_np

                bg_idx      = np.random.choice(len(X_train_np), size=50, replace=False)
                bg_data     = X_train_np[bg_idx]
                explainer   = shap.KernelExplainer(model_predict, bg_data)
                shap_values = explainer.shap_values(X_rc_np, nsamples=100)

                # shap_values 형태 처리
                # multiclass: list[C개의 (N,F)] / binary/regression: (N,F)
                # → 목표: shap_arr.shape == (N, F)
                if isinstance(shap_values, list):
                    # list[C x (N,F)] → 절댓값 평균 (N,F)
                    arrays = [np.abs(np.array(sv, dtype=float)) for sv in shap_values]
                    # 각 array가 (N,F)인지 확인
                    valid = [a for a in arrays if a.ndim == 2 and a.shape[1] == n_features]
                    if valid:
                        shap_arr = np.mean(valid, axis=0)          # (N, F)
                    else:
                        # fallback: 첫 번째 배열 사용
                        shap_arr = arrays[0]
                else:
                    shap_values = np.array(shap_values, dtype=float)
                    if shap_values.ndim == 3:
                        # 어느 축이 F인지 판별
                        for ax in [0, 1, 2]:
                            remaining = [i for i in range(3) if i != ax]
                            if shap_values.shape[ax] == n_features:
                                # ax가 F축 → 나머지 두 축 중 N축을 찾아 평균
                                shap_arr = np.abs(shap_values).mean(
                                    axis=remaining[0]
                                )                                  # (N or C, F)
                                if shap_arr.shape[0] != n_features:
                                    shap_arr = shap_arr            # (N, F)
                                else:
                                    shap_arr = shap_arr.T          # transpose to (N, F)
                                break
                        else:
                            shap_arr = np.abs(shap_values).mean(axis=0)
                    else:
                        shap_arr = np.abs(shap_values)             # (N, F)

                # shape 확인 후 (F,)로 보정
                shap_mean_raw = np.array(shap_arr.mean(axis=0), dtype=float)
                # shap_arr이 (N,F)이어야 하는데 다른 shape인 경우 flatten
                if shap_mean_raw.shape[0] != n_features:
                    # (F,C) 형태로 나온 경우 → 클래스 축 평균
                    shap_mean_raw = shap_arr.mean(axis=0)
                    if shap_mean_raw.ndim > 1:
                        shap_mean_raw = shap_mean_raw.mean(axis=-1)
                    shap_mean_raw = shap_mean_raw[:n_features]
                shap_mean = np.array(shap_mean_raw, dtype=float).flatten()[:n_features]
                assert shap_mean.shape[0] == n_features, f"shap_mean shape {shap_mean.shape} != {n_features}"
                shap_rank = np.argsort(np.argsort(-shap_mean)).astype(int)

                # ── Step 4. Random attribution 순위 ─────────────
                print(f"  [4/4] Random attribution baseline 계산 중...")
                np.random.seed(args.seed)
                rand_mean = np.random.rand(n_features)
                rand_rank = np.argsort(np.argsort(-rand_mean)).astype(int)

                # 모든 rank 배열 타입 통일 (인덱싱 오류 방지)
                tabera_rank = np.array(tabera_rank, dtype=int)
                delta_rank  = np.array(delta_rank,  dtype=int)
                shap_rank   = np.array(shap_rank,   dtype=int)
                rand_rank   = np.array(rand_rank,   dtype=int)

                # ── Step 5. Spearman 상관계수 ────────────────────
                corr_tabera, p_tabera = spearmanr(tabera_rank, delta_rank)
                corr_shap,   p_shap   = spearmanr(shap_rank,   delta_rank)
                corr_rand,   p_rand   = spearmanr(rand_rank,   delta_rank)

                print(f"\n  {'─'*60}")
                print(f"  {'Method':<20} {'Spearman ρ':>12}  {'p-value':>12}")
                print(f"  {'─'*60}")
                print(f"  {'TabERA (ours)':<20} {corr_tabera:>12.4f}  {p_tabera:>12.4f}")
                print(f"  {'SHAP':<20} {corr_shap:>12.4f}  {p_shap:>12.4f}")
                print(f"  {'Random':<20} {corr_rand:>12.4f}  {p_rand:>12.4f}")
                print(f"  {'─'*60}")

                print(f"\n  [Delta 상위 5개 feature — 방법별 순위 비교]")
                top5_delta = np.argsort(delta_arr)[::-1][:5]
                print(f"  {'Feature':<25} {'Delta순위':>8}  {'TabERA':>8}  {'SHAP':>8}")
                print(f"  {'─'*55}")
                for fi in top5_delta:
                    fn = col_names[fi] if fi < len(col_names) else f"f{fi}"
                    print(
                        f"  {fn:<25} "
                        f"  #{int(delta_rank[fi])+1:>4}    "
                        f"  #{int(tabera_rank[fi])+1:>4}    "
                        f"  #{int(shap_rank[fi])+1:>4}"
                    )

                print(f"\n  [해석]")
                if corr_tabera >= corr_shap:
                    print(f"  ✅ TabERA(ρ={corr_tabera:.3f}) ≥ SHAP(ρ={corr_shap:.3f})")
                    print(f"     prediction 영향력 순위와의 일치도가 SHAP 이상")
                    print(f"     + explanation이 prediction path 안에 있는 구조적 차별성 보유")
                else:
                    diff = corr_shap - corr_tabera
                    print(f"  TabERA(ρ={corr_tabera:.3f})  SHAP(ρ={corr_shap:.3f})  차이={diff:.3f}")
                    print(f"  → semantic 순위 일치도는 SHAP이 높지만,")
                    print(f"    TabERA는 explanation이 prediction graph 안에 있다는")
                    print(f"    구조적 차별성을 추가로 보유 (SHAP은 불가능)")
                print(f"  Random baseline: ρ={corr_rand:.3f}")

                # 결과 저장
                rc_save = {
                    "corr_tabera":  corr_tabera,
                    "corr_shap":    corr_shap,
                    "corr_random":  corr_rand,
                    "p_tabera":     p_tabera,
                    "p_shap":       p_shap,
                    "p_random":     p_rand,
                    "delta_arr":    delta_arr.tolist(),
                    "tabera_mean":  tabera_mean.tolist(),
                    "shap_mean":    shap_mean.tolist(),
                    "col_names":    col_names,
                    "openml_id":    openml_id,
                    "seed":         args.seed,
                }
                rc_path = (
                    Path(log_dir)
                    / f"data={openml_id}..seed{args.seed}_rank_correlation.pkl"
                )
                with open(rc_path, "wb") as f:
                    pickle.dump(rc_save, f)
                print(f"\n  저장: {rc_path}")
        # ── dual_space_faithfulness: dual-space centroid 설계 검증 ──────
        elif args.ablation == "dual_space_faithfulness":
            from scipy.stats import spearmanr

            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features
            n_val      = min(512, X_test.shape[0])
            X_val_sub  = X_test[:n_val]

            print(f"\n  Dual-Space Faithfulness Analysis")
            print(f"  {'─'*58}")

            with torch.no_grad():
                out_ds        = model(X_val_sub)
                hard_assign   = out_ds["hard_group"].cpu()
                evidence_w_ds = out_ds.get("evidence_w")
                topk_idx_ds   = out_ds.get("topk_idx")

            cx            = model.prototype_layer.centroid_x
            sample_groups = model.prototype_layer.sample_groups
            X_train_cpu   = X_train.detach().cpu()
            X_val_cpu     = X_val_sub.detach().cpu()

            # ── 검증 1: centroid_x representation quality ─────────────
            # centroid_x[p]가 그룹 p 샘플들의 feature 분포를 얼마나 잘 대표하는가.
            # medoid 기반이므로 항상 실제 훈련 샘플이고,
            # gradient로 최적화된 centroid_emb와 가장 가까운 샘플.
            # random centroid 대비 얼마나 더 가까운지(compression ratio)로 측정.
            print(f"\n  [검증 1] centroid_x Representation Quality")

            centroid_dists, random_dists = [], []
            torch.manual_seed(args.seed)
            cx_cpu = cx.detach().cpu()
            random_cx = (
                torch.rand_like(cx_cpu)
                * (X_train_cpu.max(0).values - X_train_cpu.min(0).values)
                + X_train_cpu.min(0).values
            )

            for p in range(model.prototype_layer.P):
                grp = sample_groups[p] if sample_groups else []
                if len(grp) < 2:
                    continue
                grp_samples = X_train_cpu[grp]
                cx_p        = cx_cpu[p]
                rand_p      = random_cx[p]
                centroid_dists.append((grp_samples - cx_p).abs().mean().item())
                random_dists.append((grp_samples - rand_p).abs().mean().item())

            if centroid_dists:
                mean_cx   = float(np.mean(centroid_dists))
                mean_rand = float(np.mean(random_dists))
                compression = mean_rand / (mean_cx + 1e-8)
                print(f"  centroid_x  평균 L1 거리: {mean_cx:.4f}")
                print(f"  random      평균 L1 거리: {mean_rand:.4f}")
                print(f"  compression ratio       : {compression:.2f}x")
                if compression > 1.5:
                    print(f"  ✅ centroid_x가 random 대비 {compression:.1f}x 더 그룹을 잘 대표함")
                    print(f"     (medoid가 gradient-optimized centroid_emb를 정확히 반영)")
                else:
                    print(f"  ⚠️  centroid_x 대표성이 낮음 (ratio={compression:.2f}x)")

            # ── 검증 2: between-group feature separation ──────────────
            # centroid_x들 간 feature 분산 (between) vs 그룹 내 분산 (within).
            # separation이 높은 feature = centroid가 실제로 그 feature로 그룹을 구분.
            # 이게 높아야 "이 그룹은 high-alcohol, low-pH 그룹" 설명이 의미있음.
            print(f"\n  [검증 2] Between-Group Feature Separation")

            if cx is not None and sample_groups:
                cx_np       = cx_cpu.numpy()
                between_var = cx_np.var(axis=0)

                within_vars = []
                for p in range(model.prototype_layer.P):
                    grp = sample_groups[p] if sample_groups else []
                    if len(grp) < 2:
                        continue
                    within_vars.append(X_train_cpu[grp].numpy().var(axis=0))

                if within_vars:
                    within_var  = np.mean(within_vars, axis=0)
                    separation  = between_var / (within_var + 1e-8)
                    top_sep_idx = np.argsort(separation)[::-1][:5]

                    print(f"  {'Feature':<25} {'Separation':>12}  {'Between':>10}  {'Within':>10}")
                    print(f"  {'─'*62}")
                    for fi in top_sep_idx:
                        fname = col_names[fi] if fi < len(col_names) else f"f{fi}"
                        print(f"  {fname:<25} {separation[fi]:>12.3f}  {between_var[fi]:>10.4f}  {within_var[fi]:>10.4f}")

                    best_f = col_names[separation.argmax()] if separation.argmax() < len(col_names) else f"f{separation.argmax()}"
                    print(f"\n  mean separation : {separation.mean():.3f}")
                    print(f"  max separation  : {separation.max():.3f}  ({best_f})")
                    print(f"  → 높은 separation = centroid_x 설명이 실제 그룹 경계를 반영")

            # 저장
            dsf_save = {
                "centroid_dists":  centroid_dists,
                "random_dists":    random_dists,
                "openml_id":       openml_id,
                "seed":            args.seed,
            }
            dsf_path = (
                Path(log_dir)
                / f"data={openml_id}{_save_tag}..seed{args.seed}_dual_space_faithfulness.pkl"
            )
            with open(dsf_path, "wb") as f:
                pickle.dump(dsf_save, f)
            print(f"\n  저장: {dsf_path}")

        # ── value_diagnosis: AttentionAggregator value 구성 진단 ──────────
        #
        # [측정 방식]
        # value = label_emb + T(query - neighbour) 에서 두 항의 L2 norm을
        # 직접 비교. 재학습 없이 학습된 가중치로 측정하는 저비용 사전
        # 진단으로, Gated Fusion 제거 때 썼던 방식(gate 값이 항상 ≈0.5로
        # 고정돼 학습이 안 됐음을 진단)과 같은 부류의 검증이다.
        #
        # [주의] 이건 "T()가 필요한가"에 대한 확정적 증거가 아니라 정황
        # 증거임. 확실한 ablation은 T() 없는 아키텍처를 처음부터 재학습해서
        # 비교하는 것 — 여기서 ratio가 작게 나오면 그 재학습이 해볼 만한
        # 가치가 있다는 신호로 쓰면 된다.
        elif args.ablation == "value_diagnosis":
            model.eval()
            n_val     = min(512, X_test.shape[0])
            X_val_sub = X_test[:n_val]

            print(f"\n  Value Component Diagnosis (label_emb vs T(query-neighbour))")
            print(f"  {'─'*58}")

            with torch.no_grad():
                # query_emb는 model(X_val_sub) 내부에서도 계산되지만, nk를
                # 재구성하려면 topk_idx가 필요해서 forward를 그대로 한 번
                # 호출하고 embedder만 별도로 다시 불러 query_emb를 얻는다.
                # (eval 모드라 dropout 등 확률적 요소 없음 → 완전히 동일한 값)
                query_emb_vd = model.embedder(X_val_sub)              # (n_val, D)
                out_vd       = model(X_val_sub)
                topk_idx_vd  = out_vd.get("topk_idx")                 # (n_val, k)
                n_mem        = model.memory.filled.item()

            if not model.use_offset_correction:
                print("  (진단 불가 — 이 모델은 --no_offset_correction으로 학습되어 "
                      "T()가 없습니다. value_diagnosis는 T()가 있는 모델 전용입니다.)")
            elif topk_idx_vd is None or n_mem < 1:
                print("  (진단 불가 — memory bank가 아직 채워지지 않았습니다)")
            else:
                with torch.no_grad():
                    safe_idx  = topk_idx_vd.clamp(0, n_mem - 1)
                    nk_vd     = model.memory.keys[safe_idx]            # (n_val, k, D)
                    labels_vd = model.memory.labels[safe_idx]          # (n_val, k)

                    stats = model.ot_selector.diagnose_value_components(
                        query_emb_vd, nk_vd, labels_vd
                    )

                print(f"  label_emb  norm : {stats['label_emb_norm_mean']:.4f} "
                      f"± {stats['label_emb_norm_std']:.4f}")
                print(f"  T(offset)  norm : {stats['offset_norm_mean']:.4f} "
                      f"± {stats['offset_norm_std']:.4f}")
                print(f"  ratio (offset/label) : {stats['ratio_mean']:.4f} "
                      f"± {stats['ratio_std']:.4f}")

                if stats["ratio_mean"] < 0.1:
                    print(f"\n  ⚠️  T(query-neighbour)가 label_emb 대비 매우 작습니다 "
                          f"(ratio={stats['ratio_mean']:.2%}) — Gated Fusion 때와 유사한 패턴.")
                    print(f"     T()가 유의미한 보정을 학습하지 못했을 가능성이 있습니다 — "
                          f"T() 없는 아키텍처로 재학습 비교를 권장합니다.")
                elif stats["ratio_mean"] > 0.5:
                    print(f"\n  ✅ T(query-neighbour)가 label_emb와 비슷하거나 더 큽니다 "
                          f"(ratio={stats['ratio_mean']:.2%}) — 유의미한 보정을 학습했을 가능성이 높습니다.")
                else:
                    print(f"\n  ℹ️  중간 수준입니다 (ratio={stats['ratio_mean']:.2%}) — "
                          f"결정적이지 않으니 재학습 비교로 확인을 권장합니다.")

                # 저장
                vd_save = {**stats, "openml_id": openml_id, "seed": args.seed}
                vd_path = (
                    Path(log_dir)
                    / f"data={openml_id}{_save_tag}..seed{args.seed}_value_diagnosis.pkl"
                )
                with open(vd_path, "wb") as f:
                    pickle.dump(vd_save, f)
                print(f"\n  저장: {vd_path}")

        # ── deletion_auc: attribution 순위로 feature 누적 마스킹 → ŷ AUC ──
        #
        # [측정 방식]
        # 1. 각 샘플에 대해 IG / SHAP / Random attribution 순위 계산
        # 2. 가장 중요한 feature부터 1개씩 누적해서 X̄(평균)로 마스킹
        # 3. 매 step마다 ŷ 측정 → 곡선 형성
        # 4. 곡선 아래 면적 (AUC) 계산 — 낮을수록 좋은 attribution
        #
        # [rank_correlation과의 차이]
        # rank_correlation: 순위 일치도만 측정
        # deletion_auc    : 누적 효과의 크기 측정
        # (1위 feature가 압도적인 경우와 1~5위가 골고루 영향 주는 경우 구분 가능)
        elif args.ablation == "deletion_auc":
            from scipy.stats import spearmanr

            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            n_test = min(100, X_test.shape[0])
            X_da   = X_test[:n_test].clone()
            X_baseline = X_train.mean(dim=0)              # (F,) 마스킹 시 사용

            print(f"\n  Deletion AUC Faithfulness (n={n_test})")
            print(f"  {'─'*60}")

            # ── Step 1. 원본 prediction (logits) ──────────────
            with torch.no_grad():
                logits_orig = model(X_da)["logits"]
                if tasktype == "regression":
                    pred_orig = logits_orig.squeeze(-1).cpu().numpy()
                    target_class = None
                elif tasktype == "multiclass":
                    pred_orig = torch.softmax(logits_orig, dim=-1).cpu().numpy()
                    # 다중 클래스: argmax 클래스의 확률 추적
                    target_class = pred_orig.argmax(axis=-1)
                    pred_orig = pred_orig[np.arange(n_test), target_class]
                else:  # binclass
                    probs = torch.sigmoid(logits_orig.squeeze(-1)).cpu().numpy()
                    # 모델이 예측한 클래스(0 또는 1) 방향으로 확률 통일
                    # (그렇지 않으면 class=0으로 예측된 샘플에서 deletion 효과의
                    #  방향이 반대로 해석됨 — class=1 확률만 추적하면
                    #  "정답 클래스 확신도"가 아니라 임의 방향의 수치가 됨)
                    target_class = (probs >= 0.5).astype(int)            # (N,) predicted class
                    pred_orig = np.where(target_class == 1, probs, 1.0 - probs)

            # ── Step 2. TabERA IG attribution (Integrated Gradients, multi-step) ──
            print(f"  [1/3] TabERA IG attribution 계산 중 (50-step)...")
            ig_target_fn = make_logit_target_fn(
                tasktype,
                target_class=target_class if tasktype == "multiclass" else None,
            )
            tabera_imp = compute_integrated_gradients(
                model, X_da, X_baseline,
                target_fn=ig_target_fn,
                n_steps=50,
                check_convergence=True,
            ).cpu().numpy()  # (N, F)

            # ── Step 3. SHAP attribution 순위 ──────────────────
            print(f"  [2/3] SHAP attribution 계산 중...")
            try:
                import shap
                from tqdm import tqdm

                def model_predict_fn(x_np):
                    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        lg = model(x_t)["logits"]
                        if tasktype == "regression":
                            return lg.squeeze(-1).cpu().numpy()
                        elif tasktype == "multiclass":
                            return torch.softmax(lg, dim=-1).cpu().numpy()
                        else:
                            return torch.sigmoid(lg.squeeze(-1)).cpu().numpy()

                bg_n   = min(50, len(X_train))
                bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                bg     = X_train[bg_idx].cpu().numpy()
                explainer = shap.KernelExplainer(model_predict_fn, bg, silent=True)

                shap_imp = np.zeros((n_test, n_features))
                for i in tqdm(range(n_test), ncols=120, leave=False):
                    sv = explainer.shap_values(X_da[i:i+1].cpu().numpy(), nsamples=100, silent=True)
                    if isinstance(sv, list):  # multiclass
                        sv = sv[target_class[i]] if target_class is not None else sv[0]
                    shap_imp[i] = np.abs(np.array(sv).flatten()[:n_features])
                shap_available = True
            except Exception as e:
                print(f"  [SHAP 실패: {e}] SHAP 없이 진행")
                shap_imp = None
                shap_available = False

            # ── Step 4. Random baseline ──────────────────────
            print(f"  [3/3] Random baseline 계산 중...")
            random_imp = np.abs(np.random.randn(n_test, n_features))

            # ── Step 5. Deletion AUC 계산 ────────────────────
            def compute_deletion_auc(attribution):
                """
                attribution: (N, F) — 각 샘플의 feature 중요도
                반환: 평균 deletion AUC (낮을수록 좋음)
                """
                aucs = []
                for n in range(n_test):
                    # 순위 (높은 importance 먼저)
                    order = np.argsort(-attribution[n])  # (F,)

                    # 곡선: 마스킹 0개 → F개
                    masked = X_da[n].clone()
                    preds  = [pred_orig[n]]
                    for f_idx in order:
                        masked[f_idx] = X_baseline[f_idx]
                        with torch.no_grad():
                            lg = model(masked.unsqueeze(0))["logits"]
                            if tasktype == "regression":
                                p = lg.squeeze(-1).item()
                            elif tasktype == "multiclass":
                                p = torch.softmax(lg, dim=-1)[0, target_class[n]].item()
                            else:  # binclass — predicted class 방향으로 통일
                                prob1 = torch.sigmoid(lg.squeeze(-1))[0].item()
                                p = prob1 if target_class[n] == 1 else 1.0 - prob1
                            preds.append(p)
                    # 정규화된 AUC (trapezoidal rule)
                    # numpy 2.0+: np.trapz → np.trapezoid
                    try:
                        auc = np.trapezoid(preds) / n_features
                    except AttributeError:
                        auc = np.trapz(preds) / n_features
                    aucs.append(auc)
                return np.array(aucs)

            print(f"\n  Deletion curve 계산 중...")
            tabera_aucs = compute_deletion_auc(tabera_imp)
            shap_aucs   = compute_deletion_auc(shap_imp)   if shap_available else None
            random_aucs = compute_deletion_auc(random_imp)

            print(f"\n  {'─'*60}")
            print(f"  {'Method':<25} {'Deletion AUC':>15} {'std':>10}")
            print(f"  {'─'*60}")
            print(f"  {'TabERA (ours)':<25} {tabera_aucs.mean():>15.4f} {tabera_aucs.std():>10.4f}")
            if shap_available:
                print(f"  {'SHAP':<25} {shap_aucs.mean():>15.4f} {shap_aucs.std():>10.4f}")
            print(f"  {'Random':<25} {random_aucs.mean():>15.4f} {random_aucs.std():>10.4f}")
            print(f"  {'─'*60}")

            print(f"\n  [해석]")
            print(f"  → Deletion AUC가 낮을수록 attribution이 prediction-relevant한")
            print(f"    feature를 정확히 가리킴 (그것을 먼저 지웠을 때 ŷ가 더 빠르게 변함)")
            if shap_available:
                if tabera_aucs.mean() < shap_aucs.mean():
                    print(f"  ✅ TabERA AUC ({tabera_aucs.mean():.4f}) < SHAP AUC ({shap_aucs.mean():.4f})")
                else:
                    print(f"  △ TabERA AUC ({tabera_aucs.mean():.4f}) ≥ SHAP AUC ({shap_aucs.mean():.4f})")
            print(f"  Random baseline: {random_aucs.mean():.4f}")

            # 저장
            da_save = {
                "tabera_aucs": tabera_aucs.tolist(),
                "shap_aucs":   shap_aucs.tolist() if shap_available else None,
                "random_aucs": random_aucs.tolist(),
                "n_samples":   n_test,
                "openml_id":   openml_id,
                "seed":        args.seed,
            }
            da_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_deletion_auc.pkl"
            )
            with open(da_path, "wb") as f:
                pickle.dump(da_save, f)
            print(f"\n  저장: {da_path}")

        # ── insertion_auc: baseline에서 시작 → 중요 feature부터 복원 ──
        #
        # [측정 방식] — Petsiuk et al. 2018 (RISE)
        # 1. X_baseline (평균) 상태에서 시작 → ŷ_baseline
        # 2. 각 샘플에 대해 attribution 순위 (가장 중요한 것부터) 계산
        # 3. 가장 중요한 feature부터 1개씩 원본 값으로 복원
        # 4. 매 step마다 ŷ 측정 → 곡선 형성
        # 5. 곡선 아래 면적 (AUC) 계산 — 높을수록 좋은 attribution
        #
        # [Deletion과의 짝]
        # Deletion: 원본 → 중요 feature 제거 → ŷ 빠르게 감소 (낮은 AUC가 좋음)
        # Insertion: baseline → 중요 feature 추가 → ŷ 빠르게 회복 (높은 AUC가 좋음)
        # 두 metric은 RISE 논문에서 짝으로 제안된 표준 평가 조합.
        elif args.ablation == "insertion_auc":
            from scipy.stats import spearmanr

            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            n_test = min(100, X_test.shape[0])
            X_ia   = X_test[:n_test].clone()
            X_baseline = X_train.mean(dim=0)              # (F,) 복원 시작점

            print(f"\n  Insertion AUC Faithfulness (n={n_test})")
            print(f"  {'─'*60}")

            # ── Step 1. 원본 prediction (logits) — 최종 target ──
            with torch.no_grad():
                logits_orig = model(X_ia)["logits"]
                if tasktype == "regression":
                    pred_orig = logits_orig.squeeze(-1).cpu().numpy()
                    target_class = None
                elif tasktype == "multiclass":
                    probs = torch.softmax(logits_orig, dim=-1).cpu().numpy()
                    target_class = probs.argmax(axis=-1)
                    pred_orig = probs[np.arange(n_test), target_class]
                else:  # binclass — predicted class 방향으로 확률 통일
                    probs = torch.sigmoid(logits_orig.squeeze(-1)).cpu().numpy()
                    target_class = (probs >= 0.5).astype(int)            # (N,)
                    pred_orig = np.where(target_class == 1, probs, 1.0 - probs)

            # ── Step 2. TabERA IG attribution (Integrated Gradients, multi-step) ──
            print(f"  [1/3] TabERA IG attribution 계산 중 (50-step)...")
            ig_target_fn = make_logit_target_fn(
                tasktype,
                target_class=target_class if tasktype == "multiclass" else None,
            )
            tabera_imp = compute_integrated_gradients(
                model, X_ia, X_baseline,
                target_fn=ig_target_fn,
                n_steps=50,
                check_convergence=True,
            ).cpu().numpy()

            # ── Step 3. SHAP attribution ──────────────────────
            print(f"  [2/3] SHAP attribution 계산 중...")
            try:
                import shap
                from tqdm import tqdm

                def model_predict_fn(x_np):
                    x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        lg = model(x_t)["logits"]
                        if tasktype == "regression":
                            return lg.squeeze(-1).cpu().numpy()
                        elif tasktype == "multiclass":
                            return torch.softmax(lg, dim=-1).cpu().numpy()
                        else:
                            return torch.sigmoid(lg.squeeze(-1)).cpu().numpy()

                bg_n   = min(50, len(X_train))
                bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                bg     = X_train[bg_idx].cpu().numpy()
                explainer = shap.KernelExplainer(model_predict_fn, bg, silent=True)

                shap_imp = np.zeros((n_test, n_features))
                for i in tqdm(range(n_test), ncols=120, leave=False):
                    sv = explainer.shap_values(X_ia[i:i+1].cpu().numpy(), nsamples=100, silent=True)
                    if isinstance(sv, list):
                        sv = sv[target_class[i]] if target_class is not None else sv[0]
                    shap_imp[i] = np.abs(np.array(sv).flatten()[:n_features])
                shap_available = True
            except Exception as e:
                print(f"  [SHAP 실패: {e}] SHAP 없이 진행")
                shap_imp = None
                shap_available = False

            # ── Step 4. Random baseline ──────────────────────
            print(f"  [3/3] Random baseline 계산 중...")
            random_imp = np.abs(np.random.randn(n_test, n_features))

            # ── Step 5. Insertion AUC 계산 ────────────────────
            def compute_insertion_auc(attribution):
                """
                attribution: (N, F) — 각 샘플의 feature 중요도
                반환: 평균 insertion AUC (높을수록 좋음)
                """
                aucs = []
                for n in range(n_test):
                    # 순위 (높은 importance 먼저)
                    order = np.argsort(-attribution[n])  # (F,)

                    # baseline 상태에서 시작 → 중요 feature부터 1개씩 복원
                    inserted = X_baseline.clone()
                    preds    = []

                    # baseline prediction
                    with torch.no_grad():
                        lg = model(inserted.unsqueeze(0))["logits"]
                        if tasktype == "regression":
                            p = lg.squeeze(-1).item()
                        elif tasktype == "multiclass":
                            p = torch.softmax(lg, dim=-1)[0, target_class[n]].item()
                        else:  # binclass — predicted class 방향으로 통일
                            prob1 = torch.sigmoid(lg.squeeze(-1))[0].item()
                            p = prob1 if target_class[n] == 1 else 1.0 - prob1
                        preds.append(p)

                    # 중요 feature부터 복원
                    for f_idx in order:
                        inserted[f_idx] = X_ia[n, f_idx]
                        with torch.no_grad():
                            lg = model(inserted.unsqueeze(0))["logits"]
                            if tasktype == "regression":
                                p = lg.squeeze(-1).item()
                            elif tasktype == "multiclass":
                                p = torch.softmax(lg, dim=-1)[0, target_class[n]].item()
                            else:  # binclass — predicted class 방향으로 통일
                                prob1 = torch.sigmoid(lg.squeeze(-1))[0].item()
                                p = prob1 if target_class[n] == 1 else 1.0 - prob1
                            preds.append(p)

                    # 정규화된 AUC (trapezoidal rule)
                    try:
                        auc = np.trapezoid(preds) / n_features
                    except AttributeError:
                        auc = np.trapz(preds) / n_features
                    aucs.append(auc)
                return np.array(aucs)

            print(f"\n  Insertion curve 계산 중...")
            tabera_aucs = compute_insertion_auc(tabera_imp)
            shap_aucs   = compute_insertion_auc(shap_imp)   if shap_available else None
            random_aucs = compute_insertion_auc(random_imp)

            print(f"\n  {'─'*60}")
            print(f"  {'Method':<25} {'Insertion AUC':>15} {'std':>10}")
            print(f"  {'─'*60}")
            print(f"  {'TabERA (ours)':<25} {tabera_aucs.mean():>15.4f} {tabera_aucs.std():>10.4f}")
            if shap_available:
                print(f"  {'SHAP':<25} {shap_aucs.mean():>15.4f} {shap_aucs.std():>10.4f}")
            print(f"  {'Random':<25} {random_aucs.mean():>15.4f} {random_aucs.std():>10.4f}")
            print(f"  {'─'*60}")

            print(f"\n  [해석]")
            print(f"  → Insertion AUC가 높을수록 attribution이 prediction-relevant한")
            print(f"    feature를 정확히 가리킴 (그것을 먼저 복원했을 때 ŷ가 더 빠르게 회복)")
            if shap_available:
                if tabera_aucs.mean() > shap_aucs.mean():
                    print(f"  ✅ TabERA AUC ({tabera_aucs.mean():.4f}) > SHAP AUC ({shap_aucs.mean():.4f})")
                else:
                    print(f"  △ TabERA AUC ({tabera_aucs.mean():.4f}) ≤ SHAP AUC ({shap_aucs.mean():.4f})")
            print(f"  Random baseline: {random_aucs.mean():.4f}")

            # 저장
            ia_save = {
                "tabera_aucs": tabera_aucs.tolist(),
                "shap_aucs":   shap_aucs.tolist() if shap_available else None,
                "random_aucs": random_aucs.tolist(),
                "n_samples":   n_test,
                "openml_id":   openml_id,
                "seed":        args.seed,
            }
            ia_path = (
                Path(log_dir)
                / f"data={openml_id}..seed{args.seed}_insertion_auc.pkl"
            )
            with open(ia_path, "wb") as f:
                pickle.dump(ia_save, f)
            print(f"\n  저장: {ia_path}")

        # ── random_neighbor: 성능 비교 ───────────────────────
        # full model 대비 성능 하락을 측정.
        # random_neighbor: neighbor 랜덤화 시 성능 하락 → neighbor evidence가 의미 있게 사용
        else:
            with torch.no_grad():
                abl_logits_list, abl_labels_list = [], []
                batch_size = 256
                n_test     = X_test.shape[0]

                for start in range(0, n_test, batch_size):
                    X_batch = X_test[start:start + batch_size]
                    out_batch = model(X_batch, ablation_mode=args.ablation)
                    abl_logits_list.append(out_batch["logits"].cpu())

                abl_logits = torch.cat(abl_logits_list, dim=0)

            if tasktype == "regression":
                abl_preds   = abl_logits.squeeze(-1).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy() * y_std,
                    abl_preds * y_std,
                    None, tasktype, "test"
                )
                abl_probs = None
            elif tasktype == "multiclass":
                abl_preds   = abl_logits.argmax(-1).numpy()
                abl_probs   = torch.softmax(abl_logits, dim=-1).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy(), abl_preds, abl_probs, tasktype, "test"
                )
            else:  # binary
                abl_preds   = (abl_logits.squeeze(-1) > 0).long().numpy()
                abl_probs   = torch.sigmoid(abl_logits.squeeze(-1)).numpy()
                abl_metrics = calculate_metric(
                    y_test.cpu().numpy(), abl_preds, abl_probs, tasktype, "test"
                )

            print(f"\n  {'Metric':<20} {'Full Model':>12}  {'Ablation':>12}  {'Δ':>10}")
            print(f"  {'-'*58}")
            for k_name, v_full in test_metrics.items():
                v_abl = abl_metrics.get(k_name, float("nan"))
                delta = v_abl - v_full
                arrow = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "─")
                print(f"  {k_name:<20} {v_full:>12.4f}  {v_abl:>12.4f}  {delta:>+9.4f} {arrow}")

            print(f"\n  해석:")
            if args.ablation == "random_neighbor":
                print(f"  → 성능 하락이 클수록 neighbor evidence가 의미 있게 사용됨")
                print(f"    (retrieval이 단순 lookup이 아님을 의미)")

            # ablation 결과 저장
            abl_save = {
                "ablation_mode":  args.ablation,
                "full_metrics":   test_metrics,
                "abl_metrics":    abl_metrics,
                "openml_id":      openml_id,
                "seed":           args.seed,
            }
            abl_path = Path(log_dir) / f"data={openml_id}{_save_tag}..seed{args.seed}_ablation_{args.ablation}.pkl"
            with open(abl_path, "wb") as f:
                pickle.dump(abl_save, f)
            print(f"\n  저장: {abl_path}")



    # ── 결과 저장 ──────────────────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_meta.pkl"

    model.eval()
    with torch.no_grad():
        logits = model(X_test)["logits"].cpu().numpy()
    np.save(str(pred_path), logits)

    meta = {
        "openml_id":   openml_id,
        "tasktype":    tasktype,
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics":test_metrics,
        "seed":        args.seed,
        "use_offset_correction": not args.no_offset_correction,
        "global_retrieve": args.global_retrieve,
        "use_context_emb": not args.no_context_emb,
        "detach_context_grad": args.detach_context_grad,
        "use_context_projection": args.context_projection,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  저장: {pred_path}")

    # ── model state 저장 (visualize_embeddings.py --from_state 용) ──
    # model_kwargs에 use_offset_correction을 명시적으로 넣어둠 — best_params
    # (Optuna 탐색 대상)에는 없는 값이라, 이걸 안 넣으면 --from_state로
    # 복원할 때 기본값(True)으로 되돌아가 버려 재현이 어긋남.
    state_path = save_dir / f"data={openml_id}{_save_tag}..seed{args.seed}_model_state.pt"
    torch.save({
        "state_dict":   model.state_dict(),
        "model_kwargs": {**model_kwargs, "use_offset_correction": not args.no_offset_correction, "global_retrieve": args.global_retrieve, "use_context_emb": not args.no_context_emb, "detach_context_grad": args.detach_context_grad, "use_context_projection": args.context_projection},
        "col_names":    dataset.col_names,
        "n_train":      len(X_train),
        "tasktype":     tasktype,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "seed":         args.seed,
    }, str(state_path))
    print(f"  저장: {state_path}")

    # ── Feature 기여도 설명 출력 ─────────────────────────
    if args.explain:
        print(f"\n{'='*52}")
        print(f"  TabERA 설명 출력 (--explain)")
        print(f"{'='*52}")

        model.eval()
        n_show = min(args.n_explain, len(y_test))
        X_show = X_test[:n_show]

        with torch.no_grad():
            out = model(X_show, return_explanations=True)

        explanations = out.get("explanations", [])

        # FeatureStore에서 이웃 feature 값 조회하여 설명에 추가
        topk_idx = out.get("topk_idx")
        if model.feature_store is not None and topk_idx is not None:
            # topk_idx: (B, k) → B개 샘플별 k개 이웃 인덱스
            neighbour_feats = model.feature_store.retrieve(topk_idx)  # list[list[dict]]
            for b, exp in enumerate(explanations):
                if b < len(neighbour_feats):
                    # 상위 5개 feature만 선택
                    exp["neighbour_features"] = [
                        model.feature_store.top_features(nd, n=5)
                        for nd in neighbour_feats[b]
                    ]
        if not explanations:
            print("  (설명 없음 — memory bank가 채워지지 않았습니다)")
            print("  → epochs를 늘리거나 n_trials를 더 실행하세요.")
        else:
            for i in range(n_show):
                print_explanation(explanations, i, dataset.col_names)


if __name__ == "__main__":
    main()