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
import torch.nn.functional as F
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
    use_soft_forward: bool = False,
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
    use_soft_forward : True면 model(x) 대신 model.forward_soft_for_ig(x, target_fn)를
                 사용. hard_assignment(argmax)로 인한 context_emb 불연속을
                 제거한 대체 함수 F_soft(x)에 대해 IG를 계산한다.
                 실제 예측(forward)은 전혀 바뀌지 않음 — 설명 계산 전용.
                 completeness axiom이 근사가 아니라 엄밀하게 성립해야 하므로
                 convergence error가 크게 개선될 것으로 기대됨.

    Returns
    ──────────
    attribution : (N, F) — |IG| (절댓값, 순위 비교용)
    """
    if X_baseline.dim() == 1:
        X_baseline = X_baseline.unsqueeze(0).expand_as(X)

    def _eval_target(x):
        if use_soft_forward:
            return model.forward_soft_for_ig(x, target_fn)
        return target_fn(model(x))

    alphas = torch.linspace(0.0, 1.0, n_steps, device=X.device)
    grads_accum = torch.zeros_like(X)

    for alpha in alphas:
        x_interp = (X_baseline + alpha * (X - X_baseline)).clone().detach().requires_grad_(True)
        target = _eval_target(x_interp)                      # (N,) per-sample
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
            f_x        = _eval_target(X)                      # (N,)
            f_baseline = _eval_target(X_baseline)              # (N,)
        ig_sum_per_sample  = ig_signed.sum(dim=-1)             # (N,)
        actual_diff_per_sample = f_x - f_baseline              # (N,)

        abs_error = (ig_sum_per_sample - actual_diff_per_sample).abs()
        rel_error = abs_error / (actual_diff_per_sample.abs() + 1e-8)

        print(f"    [IG convergence check] n_steps={n_steps}  "
              f"{'(soft-forward) ' if use_soft_forward else ''}"
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
                        choices=["none", "random_neighbor", "neighbor_noise",
                                 "rank_correlation", "dual_space_faithfulness",
                                 "deletion_auc", "insertion_auc",
                                 "value_diagnosis",
                                 "soft_ig_check",
                                 "nsteps_sweep", "illusion_check",
                                 "dataset_profile"],
                        help=(
                            "ablation 모드 선택 (학습된 모델에 inference 단계에서 적용):\n"
                            "  none                  : full model 기준 (기본값)\n"
                            "  random_neighbor       : nk/labels를 같은 permutation으로\n"
                            "                         통째로 셔플 — 배치 내 다른 쿼리의 진짜\n"
                            "                         (real) 이웃 세트로 통째로 교체.\n"
                            "                         retrieval이 '맞는 이웃'을 찾았는지만\n"
                            "                         순수하게 검증 (이웃 정보 자체는 real).\n"
                            "  neighbor_noise        : nk/labels 전부 실제 데이터와 무관한\n"
                            "                         노이즈/재추출 라벨로 교체. '이웃 정보가\n"
                            "                         조금이라도 존재하는가' 자체를 검증.\n"
                            "                         (random_neighbor와 함께 봐야 함 —\n"
                            "                         이 둘의 성능 하락 차이가 '틀린 이웃'과\n"
                            "                         '이웃 없음'의 영향을 분리해서 보여줌)\n"
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
                            "                         (재학습 없음, 저비용 사전 진단)\n"
                            "  soft_ig_check          : hard-forward IG vs soft-forward IG의\n"
                            "                         convergence error 비교 (재학습 없음).\n"
                            "                         soft-forward는 context_emb를 argmax 대신\n"
                            "                         soft-weighted sum으로 계산한 대체 함수\n"
                            "                         F_soft(x)에 대해 IG를 계산 — completeness\n"
                            "                         axiom이 엄밀히 성립. |F_soft(x)-ŷ(x)| 갭도\n"
                            "                         함께 보고 (routing이 confident할수록 작아짐)\n"
                            "  nsteps_sweep           : n_steps를 여러 단계로 스윕(예: 50,100,\n"
                            "                         200,500)하며 completeness error 변화를\n"
                            "                         관찰. mean/medoid baseline 두 곡선을 함께\n"
                            "                         그려, medoid 개선이 '불연속 회피'인지\n"
                            "                         '단순 경로 단축(discretization)'인지 분리.\n"
                            "                         오차가 1/n_steps로 매끄럽게 줄면 discretization,\n"
                            "                         어느 지점에서 plateau되면 불연속이 원인.\n"
                            "  illusion_check         : mean baseline에서 TabERA의 deletion/\n"
                            "                         insertion 우위가 completeness 붕괴로 인한\n"
                            "                         착시인지 검증. 샘플별 completeness error와\n"
                            "                         샘플별 deletion AUC 기여도의 상관관계를\n"
                            "                         분포 확인→Spearman→jackknife→confound 통제까지\n"
                            "                         한 파이프라인으로 실행 (재학습 불필요).\n"
                            "  dataset_profile        : 새 데이터셋에서 IG/deletion 신뢰도를\n"
                            "                         빠르게 분류하기 위한 통합 진단. 예측\n"
                            "                         확신도, fallback 비율, mean/medoid\n"
                            "                         completeness error, deletion_auc std\n"
                            "                         (TabERA/SHAP/Random 모두)를 한 번에 출력\n"
                            "                         하고 A(척도 자체 문제)/B(mean baseline\n"
                            "                         문제)/C(정상) 중 하나로 자동 분류."
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
    parser.add_argument("--mean_baseline", action="store_true",
                        help=(
                            "[ablation 전용] IG baseline으로 centroid medoid 대신 "
                            "X_train.mean()을 사용. 기본값은 medoid이며, 4개 데이터셋 "
                            "전부에서 completeness axiom 만족도가 일관되게 크게 개선됨을 "
                            "확인함 (median relative error 기준: vehicle 146%→17.5%, "
                            "ada 19.4%→1.5%, qsar 53.4%→1.75%, wine 319%→4.1%). "
                            "이 플래그는 'baseline 선택이 completeness에 미치는 영향'을 "
                            "보여주는 ablation 전용이며 메인 결과에는 사용하지 않는다. "
                            "soft_ig_check, deletion_auc, insertion_auc에서 공통 사용."
                        ))
    parser.add_argument("--ig_nsteps", type=int, default=50,
                        help=(
                            "IG 적분에 사용할 n_steps. 기본 50. soft_ig_check, "
                            "deletion_auc, insertion_auc에서 공통으로 사용. "
                            "nsteps_sweep 확인 결과 mean baseline은 500-step에서도 "
                            "잔여 오차가 남지만 medoid baseline(기본값)은 빠르게 "
                            "수렴하므로, 기본값(50)으로도 충분히 낮은 completeness "
                            "error를 얻을 수 있음."
                        ))
    parser.add_argument("--shap_repeats", type=int, default=1,
                        help=(
                            "rank_correlation에서 SHAP KernelExplainer 자체의 몬테카를로 "
                            "노이즈(같은 샘플에 대해서도 background/nsamples 표본추출에 "
                            "따라 값이 흔들리는 정도)를 진단하기 위해 SHAP 계산을 몇 번 "
                            "반복할지. 기본값 1 = 반복 안 함(기존 동작과 동일, 추가 비용 "
                            "없음). 2 이상이면 매번 다른 random background로 SHAP을 "
                            "다시 계산해 corr_shap의 반복 간 표준편차를 보고함 — feature 수가 "
                            "많은 데이터셋에서는 그만큼 배로 느려지므로 필요할 때만 켤 것."
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
            # [수정] X_test[:n_rc] (앞에서부터 고정 슬라이스) → 랜덤 샘플로 교체.
            # 데이터셋이 이미 셔플돼 있다면 차이가 없지만, 그게 보장돼
            # 있지 않으므로 순서에 의존하지 않게 함. args.seed로 재현 가능.
            _rc_perm   = np.random.RandomState(args.seed).permutation(X_test.shape[0])[:n_rc]
            X_rc       = X_test[_rc_perm]
            X_rc_np    = X_rc.detach().cpu().numpy()
            X_train_np = X_train.detach().cpu().numpy()

            print(f"\n  Rank Correlation Faithfulness (n={n_rc})")
            print(f"  {'─'*60}")

            # ── [정합성 수정] 세 방법(delta/SHAP/IG)이 전부 같은 대상을 재도록
            # _target_class를 가장 먼저 한 번만 계산해서 공유한다.
            # [이전 문제] delta와 SHAP은 전체 클래스 평균을, IG는 샘플별
            # 예측 클래스 하나만 쟀음 — 세 방법이 서로 다른 걸 측정하고
            # 있었음. deletion_auc 블록은 이미 SHAP을 target_class로만
            # 골라 썼는데(sv[target_class[i]]), 여기서는 그 패턴이 빠져
            # 있었음 — 의도된 설계 차이가 아니라 구현 누락으로 판단하여
            # deletion_auc와 동일한 방식으로 통일함.
            with torch.no_grad():
                logits_orig = model(X_rc)["logits"]           # (N, C) or (N, 1)
                _target_class = (
                    logits_orig.argmax(dim=-1).cpu().numpy()
                    if tasktype == "multiclass" else None
                )

            def _pick_target(logits: torch.Tensor) -> torch.Tensor:
                """(N, C) 또는 (N, 1) logits에서 샘플별 대상 스칼라를 뽑는다.
                multiclass: 그 샘플의 예측 클래스(_target_class) 하나.
                binary/regression: 애초에 출력이 1개라 그대로."""
                if tasktype == "multiclass":
                    idx = torch.as_tensor(_target_class, device=logits.device, dtype=torch.long)
                    return logits[torch.arange(logits.shape[0], device=logits.device), idx]
                return logits.squeeze(-1)

            # ── Step 1. delta 순위 계산 ─────────────────────────
            # feature 하나씩 훈련셋 평균으로 교체 → "예측 클래스" logit
            # 변화량만 측정 (전체 클래스 평균 아님 — IG와 같은 대상)
            # [샘플별 값을 따로 보관] 나중에 bootstrap으로 TabERA vs SHAP
            # 차이의 안정성을 검정하려면, feature별 평균(delta_arr)만이
            # 아니라 (N, F) 전체가 필요함.
            print(f"  [1/4] Delta 순위 계산 중 (feature {n_features}개)...")
            with torch.no_grad():
                train_mean   = X_train.mean(dim=0)             # (F,)
                orig_target  = _pick_target(logits_orig)       # (N,)

                delta_samples = np.zeros((n_rc, n_features))   # (N, F)
                for f in range(n_features):
                    X_masked       = X_rc.clone()
                    X_masked[:, f] = train_mean[f]
                    logits_masked  = model(X_masked)["logits"]
                    masked_target  = _pick_target(logits_masked)
                    delta_samples[:, f] = (orig_target - masked_target).abs().cpu().numpy()

            delta_arr  = delta_samples.mean(axis=0)            # (F,) 점추정치 (기존과 동일)
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

                def _run_shap_once(bg_rng: np.random.RandomState):
                    """SHAP 1회 실행 → (shap_arr, shap_mean, shap_rank).
                    bg_rng만 다르게 넣으면 매번 다른 background로 재계산되므로,
                    이 함수를 여러 번 호출해서 SHAP 자체의 몬테카를로 노이즈를
                    측정할 수 있다 (--shap_repeats)."""
                    bg_idx      = bg_rng.choice(len(X_train_np), size=50, replace=False)
                    bg_data     = X_train_np[bg_idx]
                    explainer   = shap.KernelExplainer(model_predict, bg_data)
                    shap_values = explainer.shap_values(X_rc_np, nsamples=100)

                    # shap_values 형태 처리
                    # multiclass: list[C개의 (N,F)] / binary/regression: (N,F)
                    # → 목표: shap_arr.shape == (N, F), 그리고 multiclass면
                    #   샘플별로 "그 샘플의 예측 클래스" 하나만 선택 (delta/IG와
                    #   같은 대상 — deletion_auc의 sv[target_class[i]] 패턴과 동일)
                    if isinstance(shap_values, list):
                        arrays = [np.abs(np.array(sv, dtype=float)) for sv in shap_values]
                        valid = [a for a in arrays if a.ndim == 2 and a.shape[1] == n_features]
                        if valid and _target_class is not None:
                            n_valid = len(valid)
                            # 클래스 c의 배열이 valid에서 빠져 있을 수 있으므로 방어적으로 clamp
                            shap_arr_ = np.stack([
                                valid[min(int(_target_class[i]), n_valid - 1)][i]
                                for i in range(n_rc)
                            ])                                          # (N, F)
                        elif valid:
                            # binary/regression은 애초에 클래스가 하나뿐이라
                            # target_class 선택 자체가 필요 없음 — 기존 방식 유지
                            shap_arr_ = np.mean(valid, axis=0)           # (N, F)
                        else:
                            # fallback: 첫 번째 배열 사용
                            shap_arr_ = arrays[0]
                    else:
                        shap_values = np.array(shap_values, dtype=float)
                        if shap_values.ndim == 3:
                            # [버그 수정] 이전 버전은 "어느 축이 F인지"만 찾고
                            # 나머지 두 축 중 아무거나 평균내버렸는데, 그 "아무거나"가
                            # 샘플(N) 축인 경우 샘플 자체가 사라져버림 (실측:
                            # shap이 (N,F,C)를 직접 반환하는 최신 버전에서, remaining
                            # 축 중 첫 번째가 N이라 N을 평균해버리고 (F,C)만 남아 →
                            # 이후 transpose 처리에서 (C,F)가 되어 샘플이 통째로
                            # 사라짐 → bootstrap에서 IndexError로 드러남).
                            # 이제 n_rc(샘플 수)와 n_features를 모두 알고 있으니,
                            # 두 축을 각각 명시적으로 찾아 샘플 축을 절대 없애지 않음.
                            shape3 = shap_values.shape
                            sample_axis, feat_axis = None, None
                            for ax, sz in enumerate(shape3):
                                if sz == n_rc and sample_axis is None:
                                    sample_axis = ax
                            for ax, sz in enumerate(shape3):
                                if ax != sample_axis and sz == n_features and feat_axis is None:
                                    feat_axis = ax

                            if sample_axis is not None and feat_axis is not None:
                                class_axis = [a for a in range(3) if a not in (sample_axis, feat_axis)][0]
                                # (N, F, C) 순서로 재배열
                                shap_moved = np.moveaxis(shap_values, [sample_axis, feat_axis, class_axis], [0, 1, 2])
                                if _target_class is not None:
                                    # delta/IG와 같은 대상(샘플별 예측 클래스)만 선택
                                    shap_arr_ = np.abs(np.stack([
                                        shap_moved[i, :, int(_target_class[i])] for i in range(n_rc)
                                    ]))                                       # (N, F)
                                else:
                                    shap_arr_ = np.abs(shap_moved).mean(axis=2)  # (N, F)
                            else:
                                # 두 축을 못 찾으면(n_rc == n_features 등 드문 충돌)
                                # 안전하게 클래스 평균으로 폴백 — 최소한 크래시는 방지
                                shap_arr_ = np.abs(shap_values).mean(axis=-1)
                                if shap_arr_.shape[0] != n_rc:
                                    shap_arr_ = shap_arr_.T
                        else:
                            shap_arr_ = np.abs(shap_values)             # (N, F)

                    assert shap_arr_.shape[0] == n_rc, (
                        f"shap_arr의 첫 축이 샘플 수(n_rc={n_rc})와 안 맞습니다: "
                        f"shap_arr.shape={shap_arr_.shape}. shap_values의 반환 형태가 "
                        f"예상과 다를 수 있습니다 (shap 버전 확인 필요)."
                    )
                    shap_mean_raw_ = np.array(shap_arr_.mean(axis=0), dtype=float)
                    if shap_mean_raw_.shape[0] != n_features:
                        shap_mean_raw_ = shap_arr_.mean(axis=0)
                        if shap_mean_raw_.ndim > 1:
                            shap_mean_raw_ = shap_mean_raw_.mean(axis=-1)
                        shap_mean_raw_ = shap_mean_raw_[:n_features]
                    shap_mean_ = np.array(shap_mean_raw_, dtype=float).flatten()[:n_features]
                    assert shap_mean_.shape[0] == n_features, f"shap_mean shape {shap_mean_.shape} != {n_features}"
                    shap_rank_ = np.argsort(np.argsort(-shap_mean_)).astype(int)
                    return shap_arr_, shap_mean_, shap_rank_

                # 기본(주) 실행 — 이후 bootstrap 등 전체 파이프라인은 이 결과 사용
                shap_arr, shap_mean, shap_rank = _run_shap_once(np.random.RandomState(args.seed))

                # ── [진단, opt-in] SHAP 자체의 몬테카를로 노이즈 ──
                # bootstrap(샘플 재표본)과는 다른 종류의 불확실성: 같은 샘플
                # 집합이라도 background/nsamples 표본추출 난수에 따라 SHAP
                # 값 자체가 흔들리는 정도. 기본값(--shap_repeats=1)이면 실행
                # 안 하고 기존과 동일한 비용. 2 이상이면 매번 다른 background로
                # SHAP을 다시 계산해 corr_shap의 반복 간 분산을 직접 잼.
                shap_mc_std = None
                if args.shap_repeats > 1:
                    print(f"  [SHAP MC 노이즈 진단] {args.shap_repeats}회 반복 재계산 중"
                          f"(매번 다른 background)...")
                    shap_mc_corrs = [spearmanr(shap_rank, delta_rank)[0]]
                    for _r in range(1, args.shap_repeats):
                        _, _, shap_rank_r = _run_shap_once(np.random.RandomState(args.seed * 1000 + _r))
                        corr_r, _ = spearmanr(shap_rank_r, delta_rank)
                        shap_mc_corrs.append(corr_r)
                    shap_mc_corrs = np.array(shap_mc_corrs)
                    shap_mc_std = float(shap_mc_corrs.std())
                    print(f"    corr_shap (반복 {args.shap_repeats}회): "
                          f"{shap_mc_corrs.mean():.4f} ± {shap_mc_std:.4f}  "
                          f"(min={shap_mc_corrs.min():.4f}, max={shap_mc_corrs.max():.4f})")
                    if shap_mc_std > 0.02:
                        print(f"    ⚠️  SHAP 자체 노이즈(±{shap_mc_std:.4f})가 꽤 큽니다 — "
                              f"이전 bootstrap CI 폭의 일부는 샘플 선택이 아니라 이 노이즈")
                        print(f"       때문일 수 있습니다. nsamples/background를 늘리는 걸 "
                              f"고려하세요.")

                # ── Step 4. Random attribution baseline (1000회 반복) ──
                # [수정 이전] 랜덤 순위를 딱 한 번만 뽑아서 하나의 ρ만 봤음 —
                # feature 수가 적을 때(예: vehicle 18개) 순수 랜덤끼리도
                # 우연히 상관관계가 크게 나올 수 있어(귀무분포 표준오차가
                # 1/sqrt(F-1) 수준), 단일 추출값을 "baseline"으로 쓰는 건
                # 통계적으로 불안정함. R=1000회 반복해서 귀무분포 자체를
                # 만들고, TabERA/SHAP의 관측 ρ가 이 분포에서 얼마나 극단적인
                # 값인지(경험적 p-value)를 직접 계산함.
                print(f"  [4/4] Random attribution baseline 계산 중 (1000회 반복)...")
                rng_rc = np.random.RandomState(args.seed)
                n_rand_draws = 1000
                rand_corrs = np.empty(n_rand_draws)
                for r in range(n_rand_draws):
                    rand_mean_r = rng_rc.rand(n_features)
                    rand_rank_r = np.argsort(np.argsort(-rand_mean_r))
                    rand_corrs[r], _ = spearmanr(rand_rank_r, delta_rank)

                corr_rand      = float(rand_corrs.mean())
                corr_rand_std  = float(rand_corrs.std())

                # 모든 rank 배열 타입 통일 (인덱싱 오류 방지)
                tabera_rank = np.array(tabera_rank, dtype=int)
                delta_rank  = np.array(delta_rank,  dtype=int)
                shap_rank   = np.array(shap_rank,   dtype=int)

                # ── Step 5. Spearman 상관계수 (점추정치) ────────────
                corr_tabera, p_tabera = spearmanr(tabera_rank, delta_rank)
                corr_shap,   p_shap   = spearmanr(shap_rank,   delta_rank)

                # 귀무분포(random) 대비 TabERA/SHAP의 경험적 p-value:
                # "랜덤 순위끼리의 ρ가 관측된 ρ 이상으로 나올 확률"
                p_tabera_vs_null = float((rand_corrs >= corr_tabera).mean())
                p_shap_vs_null   = float((rand_corrs >= corr_shap).mean())

                print(f"\n  {'─'*60}")
                print(f"  {'Method':<20} {'Spearman ρ':>12}  {'p-value':>12}")
                print(f"  {'─'*60}")
                print(f"  {'TabERA (ours)':<20} {corr_tabera:>12.4f}  {p_tabera:>12.4f}")
                print(f"  {'SHAP':<20} {corr_shap:>12.4f}  {p_shap:>12.4f}")
                print(f"  {'Random (1000회)':<20} {corr_rand:>12.4f}  {'±' + f'{corr_rand_std:.4f}':>12}")
                print(f"  {'─'*60}")
                print(f"  랜덤 귀무분포 대비 경험적 p-value:")
                print(f"    P(random ρ ≥ TabERA ρ) = {p_tabera_vs_null:.4f}")
                print(f"    P(random ρ ≥ SHAP ρ)   = {p_shap_vs_null:.4f}")

                # ── Step 6. Bootstrap: TabERA vs SHAP 차이의 안정성 검정 ──
                # [목적] 지금까지는 "TabERA ρ ≥ SHAP ρ"를 점추정치 하나로만
                # 판단했음 — 그 차이가 n_rc개 샘플의 특정 조합에서 우연히
                # 나온 건지, 샘플을 다르게 뽑아도 안정적으로 유지되는지
                # 알 수 없었음. 이미 계산해둔 샘플별 값(delta_samples,
                # tabera_imp, shap_arr)을 재표본추출(bootstrap)해서,
                # "TabERA ρ - SHAP ρ"의 분포와 그게 0보다 큰 비율을 직접 봄.
                print(f"\n  [Bootstrap] TabERA vs SHAP 차이 안정성 검정 (200회 재표본추출)...")
                n_boot = 200
                rng_boot = np.random.RandomState(args.seed + 1)
                boot_diffs = np.empty(n_boot)
                for b in range(n_boot):
                    idx_b = rng_boot.randint(0, n_rc, size=n_rc)  # 복원추출
                    delta_b  = delta_samples[idx_b].mean(axis=0)
                    tabera_b = tabera_imp[idx_b].mean(axis=0)
                    shap_b   = shap_arr[idx_b].mean(axis=0)

                    delta_rank_b  = np.argsort(np.argsort(-delta_b))
                    tabera_rank_b = np.argsort(np.argsort(-tabera_b))
                    shap_rank_b   = np.argsort(np.argsort(-shap_b))

                    corr_tabera_b, _ = spearmanr(tabera_rank_b, delta_rank_b)
                    corr_shap_b,   _ = spearmanr(shap_rank_b,   delta_rank_b)
                    boot_diffs[b] = corr_tabera_b - corr_shap_b

                boot_diff_mean = float(boot_diffs.mean())
                boot_ci_low, boot_ci_high = np.percentile(boot_diffs, [2.5, 97.5])
                boot_win_rate = float((boot_diffs > 0).mean())

                print(f"    TabERA ρ - SHAP ρ = {boot_diff_mean:+.4f}  "
                      f"(95% CI: [{boot_ci_low:+.4f}, {boot_ci_high:+.4f}])")
                print(f"    재표본 중 TabERA > SHAP 비율: {boot_win_rate:.0%}")
                if boot_ci_low > 0:
                    print(f"    → 95% CI가 0을 포함하지 않음: TabERA가 SHAP보다 "
                          f"안정적으로 낫다고 볼 근거 있음")
                elif boot_ci_high < 0:
                    print(f"    → 95% CI가 0을 포함하지 않음: SHAP이 TabERA보다 "
                          f"안정적으로 낫다고 볼 근거 있음")
                else:
                    print(f"    → 95% CI가 0을 포함함: 이 데이터셋에서는 TabERA와 SHAP의")
                    print(f"      차이가 샘플 구성에 따라 뒤집힐 수 있는 수준 — 점추정치")
                    print(f"      하나만으로 우열을 단정하면 안 됨")

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
                if boot_ci_low > 0:
                    print(f"  ✅ TabERA(ρ={corr_tabera:.3f}) > SHAP(ρ={corr_shap:.3f}) — "
                          f"bootstrap 95% CI가 0을 안 포함, 안정적으로 우위")
                elif boot_ci_high < 0:
                    print(f"  SHAP(ρ={corr_shap:.3f}) > TabERA(ρ={corr_tabera:.3f}) — "
                          f"bootstrap 95% CI가 0을 안 포함, 안정적으로 우위")
                else:
                    print(f"  TabERA(ρ={corr_tabera:.3f}) vs SHAP(ρ={corr_shap:.3f}) — "
                          f"bootstrap 95% CI가 0을 포함 (동률, 통계적으로 유의한 차이 아님)")
                print(f"     + explanation이 prediction path 안에 있다는 구조적 차별성은")
                print(f"       (SHAP은 원래 불가능한 특성) ρ 우열과 무관하게 항상 성립")

                if p_tabera_vs_null < 0.05:
                    print(f"  TabERA는 랜덤(ρ={corr_rand:.3f}±{corr_rand_std:.3f})보다 유의하게 "
                          f"나음 (p={p_tabera_vs_null:.4f})")
                else:
                    print(f"  ⚠️  TabERA가 랜덤(ρ={corr_rand:.3f}±{corr_rand_std:.3f})보다 유의하게 "
                          f"낫다고 말하기 어려움 (p={p_tabera_vs_null:.4f})")

                # 결과 저장
                rc_save = {
                    "corr_tabera":       corr_tabera,
                    "corr_shap":         corr_shap,
                    "corr_random_mean":  corr_rand,
                    "corr_random_std":   corr_rand_std,
                    "p_tabera":          p_tabera,
                    "p_shap":            p_shap,
                    "p_tabera_vs_null":  p_tabera_vs_null,
                    "p_shap_vs_null":    p_shap_vs_null,
                    "boot_diff_mean":    boot_diff_mean,
                    "boot_diff_ci":      [float(boot_ci_low), float(boot_ci_high)],
                    "boot_win_rate":     boot_win_rate,
                    "shap_mc_std":       shap_mc_std,
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
            cx_cpu        = cx.detach().cpu()
            sample_groups = model.prototype_layer.sample_groups
            X_val_cpu     = X_val_sub.detach().cpu()

            # ── [수정] sample_groups의 실제 인덱스 공간 ────────────────
            # libs/supervised.py의 학습 루프를 추적한 결과: ema_update()는
            # X_train 전체가 아니라 model.memory(MemoryBank)의 실제 내용
            # (model.memory.keys[:n_mem], model.feature_store._store[:n_mem])
            # 을 받아 sample_groups를 만든다 (supervised.py 162~169행,
            # 212~228행 주석 — "MemoryBank 인덱스 공간과 항상 일치하도록
            # 보장"하기 위해 의도적으로 그렇게 바꾼 것). MemoryBank는 크기
            # min(2*N_train, memory_size)의 FIFO 링버퍼이고, 매 에폭 학습
            # 순서가 랜덤 셔플(perm)되기 때문에, 링버퍼 슬롯 j번째 내용물이
            # X_train의 j번째 행과 같다는 보장이 전혀 없다.
            # → sample_groups[p]의 인덱스는 "X_train 행 번호"가 아니라
            #   "MemoryBank/FeatureStore 슬롯 번호"이므로, 검증도 반드시
            #   같은 슬롯 번호로 embedder를 다시 태우는 게 아니라
            #   model.memory.keys / model.feature_store._store에서 직접
            #   가져와야 한다.
            n_mem = model.memory.filled.item()
            ref_emb = model.memory.keys[:n_mem].detach().cpu()          # (n_mem, D)
            ref_raw = (
                model.feature_store._store[:n_mem].detach().cpu()
                if model.feature_store is not None else None
            )                                                            # (n_mem, F)

            # ── [수정] 명목형(categorical) feature의 거리 계산 ─────────
            # libs/data.py를 확인한 결과, categorical 컬럼은 QuantileTransformer를
            # 거치지 않고 LabelEncoder 정수 코드(0, 1, 2, ...) 그대로 X에 남는다
            # (numeric 컬럼만 [0,1] uniform으로 정규화됨). 아래 feat_centrality가
            # 이 정수 코드에 그대로 L1 거리를 적용하면, 순서가 없는 명목형
            # 카테고리에 "카테고리 0과 3이 0과 1보다 멀다"는 우연한(그리고
            # 의미 없는) 인코딩 순서를 실제 거리로 잘못 해석하게 된다
            # (splice처럼 cat 비율이 높은 데이터셋에서 correspondence ρ가
            # 계속 약하게 나온 원인 중 하나로 확인됨).
            # → Gower distance 방식으로 교체: numeric은 그대로 L1(이미
            #   [0,1] 정규화돼 있어 추가 range 정규화 불필요), categorical은
            #   "같으면 0, 다르면 1"의 불일치 카운트로 대체한다. (참고:
            #   원-핫으로 펼친 뒤 L1을 재는 것과 동치이나, 원-핫은 불일치당
            #   거리가 2가 되어 categorical에 암묵적으로 2배 가중치를 주는
            #   함정이 있어 이 방식이 더 안전함.)
            cat_cols = list(dataset.X_cat)
            num_cols = list(dataset.X_num)

            # ── 검증 1: embedding-space ↔ feature-space 대응(correspondence) ──
            # [재설계 2차] 1차 수정(group_mean과 비교)은 여전히 불공정했음:
            # group_mean은 L1 거리를 최소화하도록 "정의"된 값이라 실제
            # 샘플인 medoid가 이길 수 없는 게 수학적으로 당연하고, 게다가
            # centroid_x는애초에 "임베딩 공간"에서 centroid_emb와 가장
            # 가까운 실제 샘플로 뽑히는데(코사인 유사도 argmax), 그걸
            # "feature 공간" L1 거리로 재고 있었음 — 애초에 다른 공간의
            # 기준을 비교하고 있었음.
            #
            # [진짜 질문] "dual-space"라는 설계가 성립하려면, 임베딩 공간에서
            # 중심적인 샘플이 feature 공간에서도 중심적이어야 함. 이제 이걸
            # 직접 측정:
            #   ① correspondence  : 그룹 내에서 "centroid_emb와의 코사인 유사도
            #      (medoid를 뽑을 때 쓰는 바로 그 기준)"와 "feature 공간에서
            #      같은 그룹 동료들과의 평균 근접도"의 순위가 얼마나 일치하는가
            #      (그룹별 Spearman ρ, 그룹 크기로 가중 평균)
            #   ② 실제 뽑힌 centroid_x가, 그 그룹의 "진짜 동료들"(인위적
            #      baseline 없음) 기준으로 feature 공간 중심성 몇 백분위에
            #      해당하는가 — 이게 유일하게 완전히 공정한 기준선임
            #      (medoid도 실제 샘플이어야 하니, 비교 대상도 반드시
            #      실제 샘플이어야 공평함)
            print(f"\n  [검증 1] Embedding-Space ↔ Feature-Space Correspondence")

            valid_p_all = [p for p in range(model.prototype_layer.P)
                           if sample_groups and len(sample_groups[p]) >= 2]
            valid_p = valid_p_all
            if ref_raw is None:
                print(f"    ⚠️  model.feature_store가 없어 원본 feature 공간 비교를 할 수 없습니다 —")
                print(f"       검증 1/2를 건너뜁니다 (인덱스 정합성 확인만 아래에서 진행).")
                valid_p = []

            centroid_emb_cpu = model.prototype_layer.centroid_emb.detach().cpu()  # (P, D)

            # ── [사전 검증, 필수] sample_groups 인덱스 정합성 재확인 ──────
            # [배경 — 확인된 사실] libs/supervised.py를 추적한 결과, ema_update()는
            # 매 에폭 X_train 전체가 아니라 MemoryBank/FeatureStore의 실제
            # 내용(emb_ema = model.memory.keys[:n_mem], x_ema = feature_store
            # 의 슬롯)을 받아 sample_groups를 만든다(162~228행 주석 참고).
            # 즉 sample_groups[p]의 인덱스는 애초부터 "MemoryBank 슬롯 번호"
            # 이지 "X_train 행 번호"가 아니다. 이번 수정 전에는 여기서 X_train
            # 을 처음부터 다시 embedder에 태워 X_train 행 순서로 비교했으므로,
            # 두 인덱스 공간(슬롯 번호 vs 행 번호)이 애초에 다른 것을 같다고
            # 가정하고 검증한 셈 — 이게 7.8%(≈무작위 2.5%) 결과의 원인이다.
            #
            # [이번 수정] X_train을 재-embed하지 않고, ema_update가 실제로
            # 받았던 것과 같은 소스(model.memory.keys / model.feature_store)
            # 에서 ref_emb/ref_raw를 가져와 같은 슬롯 번호로 비교한다.
            #
            # [해석 기준 — 재설정] reproduce.py는 학습이 끝난 뒤 모델을 그대로
            # eval()에서 평가만 하고, 이 시점 이후 추가 gradient step은 없다.
            # 즉 마지막 ema_update() 호출과 이 검증 사이에 centroid_emb가
            # 전혀 움직이지 않으므로, 인덱스가 맞다면 일치율은 100%에 매우
            # 가까워야 한다(자연스러운 EMA 지연이 성립하려면 그 사이에 추가
            # 학습 스텝이 있어야 하는데, 여기엔 없음). 따라서 "70~95%면
            # 지연, 3배 이상이면 안전" 같은 관대한 기준은 이 시점에는 근거가
            # 없다 — 검증 시점 기준으로는 사실상 100% 아니면 버그다.
            with torch.no_grad():
                q_check = F.normalize(ref_emb, dim=-1)
                c_check = F.normalize(centroid_emb_cpu, dim=-1)
                assign_check = (q_check @ c_check.T).argmax(dim=-1).numpy()  # (n_mem,)

            match_count, total_count = 0, 0
            for p in valid_p_all:
                grp = sample_groups[p]
                total_count += len(grp)
                match_count += int((assign_check[grp] == p).sum())
            chance_rate = 1.0 / model.prototype_layer.P

            print(f"  [사전 검증] sample_groups 인덱스 정합성 확인 (MemoryBank 슬롯 기준)")
            if total_count == 0:
                print(f"    ⚠️  검증 가능한 그룹(크기≥2)이 없어 일치율을 계산할 수 없습니다.")
                index_ok = False
            else:
                match_rate = match_count / total_count
                print(f"    재배정 일치율: {match_rate:.1%}  (무작위 기대치: {chance_rate:.1%})")
                # 100%는 부동소수점 argmax의 동률(tie) 등으로 아주 드물게 못 미칠
                # 수 있어 약간의 여유(0.99)만 둔다. 그 밑은 "지연"으로 설명되지
                # 않으므로(위 근거 참고) 전부 인덱스/소스 불일치로 취급한다.
                index_ok = match_rate >= 0.99
                if not index_ok:
                    print(f"    ❌ 검증 시점에는 추가 학습이 없어 EMA 지연으로 설명될 수 없습니다 —")
                    print(f"       sample_groups가 가리키는 소스(MemoryBank/FeatureStore)와 지금")
                    print(f"       비교에 쓴 소스가 여전히 어긋나 있을 가능성이 높습니다.")
                    print(f"       아래 검증 1/2 및 하이브리드 medoid 결과는 재확인 전까지 신뢰할 수 없습니다.")
                else:
                    print(f"    ✅ 인덱스 정합성 확인됨 (MemoryBank 슬롯 기준) — 아래 결과를 신뢰할 수 있습니다.")

            if not index_ok:
                valid_p = []

            correspondence_rhos, group_weights = [], []
            centroid_x_percentiles = []

            for p in valid_p:
                grp      = sample_groups[p]
                n_p      = len(grp)
                grp_feat = ref_raw[grp].numpy()              # (n_p, F)
                grp_emb  = ref_emb[grp]                       # (n_p, D)

                # 임베딩 공간 중심성: centroid_x를 뽑을 때 쓰는 것과 정확히
                # 같은 기준 (centroid_emb와의 코사인 유사도, 높을수록 중심적)
                c_emb   = centroid_emb_cpu[p]
                emb_sim = F.cosine_similarity(
                    grp_emb, c_emb.unsqueeze(0).expand(n_p, -1), dim=-1
                ).numpy()                                      # (n_p,)

                # feature 공간 중심성: 그룹 내 다른 실제 멤버들과의 평균
                # Gower-style 거리 (leave-one-out) — 높을수록(음수 거리가
                # 클수록) 중심적. numeric은 L1(이미 [0,1] 정규화됨),
                # categorical은 불일치 카운트(같으면 0, 다르면 1)로 계산해
                # LabelEncoder 정수 코드의 우연한 순서가 거리에 반영되지
                # 않도록 함.
                diffs = np.zeros((n_p, n_p))
                if num_cols:
                    num_part = grp_feat[:, num_cols]
                    diffs += np.abs(num_part[:, None, :] - num_part[None, :, :]).sum(axis=-1)
                if cat_cols:
                    cat_part = grp_feat[:, cat_cols]
                    diffs += (cat_part[:, None, :] != cat_part[None, :, :]).sum(axis=-1)
                np.fill_diagonal(diffs, np.nan)
                feat_centrality = -np.nanmean(diffs, axis=1)    # (n_p,)

                if n_p >= 4:
                    rho, _ = spearmanr(emb_sim, feat_centrality)
                    if not np.isnan(rho):
                        correspondence_rhos.append(rho)
                        group_weights.append(n_p)

                # 실제 medoid 위치(임베딩 유사도 argmax — 선택 기준과 동일)가
                # feature 중심성 기준으로 그 그룹 안에서 몇 백분위인지
                medoid_local_idx = int(np.argmax(emb_sim))
                percentile = float((feat_centrality <= feat_centrality[medoid_local_idx]).mean()) * 100
                centroid_x_percentiles.append(percentile)

            if correspondence_rhos:
                correspondence_rhos = np.array(correspondence_rhos)
                group_weights       = np.array(group_weights, dtype=float)
                weighted_rho        = float(np.average(correspondence_rhos, weights=group_weights))
                median_rho          = float(np.median(correspondence_rhos))
                frac_positive       = float((correspondence_rhos > 0).mean())

                print(f"  ① 그룹 내 대응(correspondence): 임베딩 유사도 순위 vs "
                      f"feature 중심성 순위")
                print(f"    그룹 크기 가중 평균 ρ : {weighted_rho:+.3f}")
                print(f"    중앙값 ρ              : {median_rho:+.3f}")
                print(f"    ρ>0인 그룹 비율        : {frac_positive:.0%}  ({len(correspondence_rhos)}개 그룹 중)")
                if weighted_rho > 0.3:
                    print(f"    → 두 공간이 어느 정도 일관되게 대응함")
                elif weighted_rho > 0.1:
                    print(f"    ⚠️  대응이 약함 — 임베딩 공간 중심성이 feature 공간 "
                          f"중심성을 부분적으로만 반영")
                else:
                    print(f"    ⚠️  대응이 거의 없음(ρ≈0) — 임베딩 공간에서 중심적이어도 "
                          f"feature 공간과는 무관할 수 있음. centroid_x 설명이 그 그룹을")
                    print(f"       feature 관점에서 대표한다는 주장이 이 데이터셋에서는")
                    print(f"       약하다는 뜻")
            else:
                weighted_rho = None
                print(f"  ① 대응 분석 불가 (그룹 크기 4 미만이 대부분)")

            if centroid_x_percentiles:
                cxp = np.array(centroid_x_percentiles)
                from scipy.stats import wilcoxon
                # 귀무가설: medoid가 그룹 내에서 무작위 멤버와 다를 바 없다
                # (기대 백분위 50) — 이 귀무값과의 짝지은 비교
                try:
                    stat_w, p_w = wilcoxon(cxp - 50.0)
                except ValueError:
                    p_w = float("nan")

                print(f"\n  ② 실제 centroid_x의 feature-중심성 백분위 (같은 그룹 진짜 동료 기준)")
                print(f"    평균 백분위   : {cxp.mean():.1f}  (50=무작위 멤버와 동급, 100=그룹 내 최고)")
                print(f"    중앙값 백분위 : {np.median(cxp):.1f}")
                print(f"    50 대비 p-value (Wilcoxon): "
                      f"{p_w:.4f}" if not np.isnan(p_w) else "    계산 불가")
                if not np.isnan(p_w) and p_w < 0.05 and cxp.mean() > 50:
                    print(f"    ✅ centroid_x가 무작위 멤버보다 유의하게 feature-중심적임")
                elif not np.isnan(p_w) and p_w < 0.05 and cxp.mean() < 50:
                    print(f"    ⚠️  centroid_x가 무작위 멤버보다 오히려 feature-주변부에 있음")
                else:
                    print(f"    ℹ️  centroid_x가 무작위 멤버와 유의한 차이 없음 (백분위 50 근처)")
            else:
                cxp, p_w = None, None

            # ── 검증 2: between-group feature separation ──────────────
            # centroid_x들 간 feature 분산 (between) vs 그룹 내 분산 (within).
            # separation이 높은 feature = centroid가 실제로 그 feature로 그룹을 구분.
            # 이게 높아야 "이 그룹은 high-alcohol, low-pH 그룹" 설명이 의미있음.
            #
            # [재설계] 기존엔 (1) within_var를 그룹별 분산의 비가중 평균으로
            # 계산해서 표본 1~2개짜리 노이즈 많은 그룹이 큰 그룹과 동일하게
            # 취급됐고, (2) between_var는 모든 P개 centroid를, within_var는
            # size>=2인 그룹만 써서 두 항이 서로 다른 그룹 집합 기준이었으며,
            # (3) 유의성 검정이 전혀 없어 "separation이 몇이면 의미 있는지"
            # 판단 기준이 없었음. → 표준 one-way ANOVA F-검정으로 교체.
            #
            # [추가 수정] F-test는 값이 연속형(등간/비율 척도)이라는 가정을
            # 깔고 있는데, categorical 컬럼은 libs/data.py에서 LabelEncoder로
            # 매긴 정수 코드일 뿐 순서가 없는 명목형이다. 이 코드에 그대로
            # 분산 기반 F-test를 적용하면, "카테고리 0과 3이 0과 1보다
            # 멀다"는 우연한 인코딩 순서를 실제 분산으로 잘못 해석하게 된다
            # (검증1에서 L1 거리에 있었던 것과 동일한 문제). numeric 컬럼만
            # F-test를 쓰고, categorical 컬럼은 "그룹 소속 × 카테고리 값"
            # 분할표에 대한 카이제곱 독립성 검정으로 교체한다 — 그룹마다
            # 카테고리 분포가 실제로 다른지(=그 카테고리로 그룹이 구분되는지)
            # 순서 가정 없이 직접 검정한다.
            print(f"\n  [검증 2] Between-Group Feature Separation")
            print(f"  (numeric: One-way ANOVA F-test / categorical: Chi-square 독립성 검정)")

            if cx is not None and valid_p:
                from scipy.stats import f as f_dist, chi2_contingency

                group_sizes = np.array([len(sample_groups[p]) for p in valid_p])
                P_valid     = len(valid_p)

                stat_arr  = np.full(n_features, np.nan)
                p_arr     = np.full(n_features, np.nan)
                test_type = np.array(["-"] * n_features, dtype=object)

                # ── numeric 컬럼: 그룹 크기로 가중한 pooled MSW/MSB F-test ──
                if num_cols:
                    cx_valid_num = cx_cpu.numpy()[valid_p][:, num_cols]      # (P_valid, F_num)
                    ss_within = np.zeros(len(num_cols))
                    total_n   = 0
                    for p in valid_p:
                        grp_data = ref_raw[sample_groups[p]].numpy()[:, num_cols]
                        grp_mean = grp_data.mean(axis=0)
                        ss_within += ((grp_data - grp_mean) ** 2).sum(axis=0)
                        total_n   += grp_data.shape[0]
                    df_within = max(total_n - P_valid, 1)
                    msw       = ss_within / df_within

                    grand_mean = np.average(cx_valid_num, axis=0, weights=group_sizes)
                    ssb        = np.sum(group_sizes[:, None] * (cx_valid_num - grand_mean) ** 2, axis=0)
                    df_between = max(P_valid - 1, 1)
                    msb        = ssb / df_between

                    F_stat_num = msb / (msw + 1e-8)
                    p_num      = f_dist.sf(F_stat_num, df_between, df_within)

                    for j, fi in enumerate(num_cols):
                        stat_arr[fi]  = F_stat_num[j]
                        p_arr[fi]     = p_num[j]
                        test_type[fi] = "F"

                # ── categorical 컬럼: 그룹×카테고리 분할표 카이제곱 검정 ──
                if cat_cols:
                    for fi in cat_cols:
                        cats_per_group = [
                            np.rint(ref_raw[sample_groups[p]].numpy()[:, fi]).astype(int)
                            for p in valid_p
                        ]
                        all_cats = np.unique(np.concatenate(cats_per_group))
                        table = np.zeros((P_valid, len(all_cats)), dtype=int)
                        for gi, vals in enumerate(cats_per_group):
                            for c in vals:
                                table[gi, np.searchsorted(all_cats, c)] += 1
                        # 카이제곱 검정이 성립하려면 카테고리 2종 이상 +
                        # 모든 행/열 합이 0보다 커야 함
                        if table.shape[1] >= 2 and (table.sum(axis=0) > 0).all() and (table.sum(axis=1) > 0).all():
                            try:
                                chi2, p, dof, _ = chi2_contingency(table)
                                stat_arr[fi]  = chi2
                                p_arr[fi]     = p
                                test_type[fi] = "χ²"
                            except ValueError:
                                pass   # 검정 불가(예: 기대빈도 문제) → NaN 유지

                valid_mask       = ~np.isnan(p_arr)
                bonferroni_alpha = 0.05 / n_features   # 다중비교 보정 (전체 feature 수 기준)
                n_significant    = int((p_arr[valid_mask] < bonferroni_alpha).sum())

                # 랭킹/표시용: 검정 종류가 달라도 비교 가능하도록 -log10(p)로 통일
                neglogp = np.full(n_features, -1.0)
                neglogp[valid_mask] = -np.log10(np.clip(p_arr[valid_mask], 1e-300, 1.0))
                top_sep_idx = np.argsort(neglogp)[::-1][:5]

                print(f"  (유효 그룹 {P_valid}개 / numeric {len(num_cols)}개 F-test / "
                      f"categorical {len(cat_cols)}개 χ²-test, 검정 가능 {int(valid_mask.sum())}/{n_features})")
                print(f"  {'Feature':<20} {'Test':>6} {'Stat':>10}  {'p-value':>12}")
                print(f"  {'─'*52}")
                for fi in top_sep_idx:
                    fname = col_names[fi] if fi < len(col_names) else f"f{fi}"
                    if np.isnan(p_arr[fi]):
                        print(f"  {fname:<20} {'-':>6} {'(검정 불가)':>21}")
                        continue
                    sig_mark = "*" if p_arr[fi] < bonferroni_alpha else " "
                    print(f"  {fname:<20} {test_type[fi]:>6} {stat_arr[fi]:>10.3f}  {p_arr[fi]:>10.4f}{sig_mark}")

                print(f"\n  Bonferroni 보정(α={bonferroni_alpha:.2e}) 후 유의한 feature 수: "
                      f"{n_significant}/{n_features}")
                if n_significant > 0:
                    print(f"  → centroid가 최소 {n_significant}개 feature에서 통계적으로 "
                          f"유의하게 그룹을 구분함")
                else:
                    print(f"  ⚠️  다중비교 보정 후 유의한 feature가 하나도 없음 — "
                          f"'이 그룹은 X, Y 특성이 다르다'는 설명의 통계적 근거가 약함")

                F_stat, p_values = stat_arr, p_arr   # 저장용 변수명 유지(하위 호환)
            else:
                F_stat, p_values, test_type = None, None, None

            # 저장
            dsf_save = {
                "correspondence_rhos":      correspondence_rhos.tolist() if isinstance(correspondence_rhos, np.ndarray) else correspondence_rhos,
                "correspondence_weighted_rho": weighted_rho,
                "centroid_x_percentiles":   centroid_x_percentiles,
                "percentile_wilcoxon_p":    float(p_w) if (p_w is not None and not np.isnan(p_w)) else None,
                "anova_F_stat":     F_stat.tolist() if F_stat is not None else None,
                "anova_p_values":   p_values.tolist() if p_values is not None else None,
                "anova_test_type":  test_type.tolist() if test_type is not None else None,
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

        # ── nv_utility_probe: 제거됨 ──────────────────────────
        # [이력] nv(이웃의 과거 context_emb)가 AttentionAggregator에서 실제로
        # 쓰이는지를 재학습 없이 진단하던 ablation이었다. mfeat-zernike/vehicle/
        # credit-approval 3개 데이터셋 실측(RepeatedKFold + noise 대조군) 결과,
        # nv가 nk/label로 이미 설명되는 잔차 이상의 추가 정보를 noise 대조군과
        # 통계적으로 구분되게 보여주지 못해 (설계 가치 없음 판단), nv 자체를
        # MemoryBank/AttentionAggregator에서 완전히 제거했다. 지난 전체
        # 결과물은 optim_logs의 *_nv_utility_probe.pkl 파일에 남아 있다.

        # ── soft_ig_check: hard-forward IG vs soft-forward IG 비교 ──
        #
        # [배경]
        # 실제 예측(forward)은 hard_assignment=argmax로 계산되는 불연속
        # 함수이며, 이 불연속성이 IG의 completeness axiom을 깨뜨리는
        # 근본 원인이다. 이 모드는 forward_soft_for_ig()를 이용해
        # context_emb를 soft-weighted sum으로 계산한 대체 함수 F_soft(x)에
        # 대해 IG를 계산하고, 기존 hard-forward IG와 convergence를 비교한다.
        #
        # 실제 예측(ŷ)이나 학습된 파라미터는 전혀 바뀌지 않는다 —
        # 순수하게 "IG를 계산하는 방식"만 두 가지로 비교하는 진단 모드.
        elif args.ablation == "soft_ig_check":
            from scipy.stats import spearmanr

            model.eval()
            n_test = min(100, X_test.shape[0])
            X_sc   = X_test[:n_test].clone()

            # ── Baseline 선택: mean(전체 평균) vs medoid(그룹별 대표 샘플) ──
            #
            # [배경] X_train.mean()은 어느 centroid에도 확실히 속하지 않는
            # 위치라, retrieve 시 baseline 시점 fallback이 100%까지 치솟고
            # (vehicle 실측), neighbor_change가 fallback 여부와 무관하게도
            # 포화(median≈1.0)되는 현상이 binclass(ada, qsar)에서도 확인됨.
            #
            # medoid baseline: 각 샘플이 속한 centroid의 실제 대표 훈련
            # 샘플(model.prototype_layer.centroid_x)을 baseline으로 사용.
            # 정의상 그 centroid 그룹 안에 확실히 속하므로, baseline 시점의
            # fallback을 줄이고 neighbor_change 포화를 완화할 가능성이 있음.
            # 이 경우 baseline이 샘플마다 다름 (개인화된 baseline).
            use_medoid_baseline = (not args.mean_baseline)
            if use_medoid_baseline:
                with torch.no_grad():
                    q_for_medoid = F.normalize(model.embedder(X_sc), dim=-1)
                    c_for_medoid = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    ha_for_medoid = (q_for_medoid @ c_for_medoid.T).argmax(dim=-1)  # (N,)
                    X_baseline = model.prototype_layer.centroid_x[ha_for_medoid].to(X_sc.device)  # (N, F)
                print(f"\n  [Baseline] medoid (샘플별 소속 그룹의 대표 훈련 샘플)")
            else:
                X_baseline = X_train.mean(dim=0)
                print(f"\n  [Baseline] mean (전체 훈련 데이터 평균)")

            print(f"\n  Soft-forward IG Convergence Check (n={n_test})")
            print(f"  {'─'*60}")

            with torch.no_grad():
                logits_orig = model(X_sc)["logits"]
                if tasktype == "regression":
                    target_class = None
                elif tasktype == "multiclass":
                    target_class = torch.softmax(logits_orig, dim=-1).argmax(dim=-1).cpu().numpy()
                else:
                    target_class = None

            ig_target_fn = make_logit_target_fn(
                tasktype,
                target_class=target_class if tasktype == "multiclass" else None,
            )

            # ── 1. 기존 hard-forward IG (실제 예측 함수에 대한 IG) ──
            print(f"  [1/3] Hard-forward IG 계산 중 (n_steps={args.ig_nsteps}, 실제 예측 함수)...")
            _ = compute_integrated_gradients(
                model, X_sc, X_baseline,
                target_fn=ig_target_fn,
                n_steps=args.ig_nsteps,
                check_convergence=True,
                use_soft_forward=False,
            )

            # ── 2. soft-forward IG (F_soft(x)에 대한 IG) ──
            print(f"  [2/3] Soft-forward IG 계산 중 (n_steps={args.ig_nsteps}, F_soft, 연속 대체 함수)...")
            _ = compute_integrated_gradients(
                model, X_sc, X_baseline,
                target_fn=ig_target_fn,
                n_steps=args.ig_nsteps,
                check_convergence=True,
                use_soft_forward=True,
            )

            # ── 3. |F_soft(x) - ŷ(x)| 갭 측정 ──
            # routing이 confident할수록(top-1 softmax 확률이 1에 가까울수록)
            # 이 갭이 작아질 것으로 예상됨 — soft와 hard 사이 근사 정도.
            print(f"  [3/3] |F_soft(x) - ŷ(x)| 근사 갭 측정 중...")
            with torch.no_grad():
                y_hard = ig_target_fn(model(X_sc))                        # (N,)
                y_soft = model.forward_soft_for_ig(X_sc, ig_target_fn)    # (N,)

                gap = (y_soft - y_hard).abs()

                # routing confidence: top-1 softmax 확률
                query_emb = model.embedder(X_sc)
                q = F.normalize(query_emb, dim=-1)
                c = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                routing_conf = F.softmax(q @ c.T, dim=-1).max(dim=-1).values  # (N,)

            print(f"\n  {'─'*60}")
            print(f"  |F_soft - ŷ| gap:  mean={gap.mean().item():.4f}  "
                  f"median={gap.median().item():.4f}  max={gap.max().item():.4f}")
            print(f"  routing confidence: mean={routing_conf.mean().item():.4f}  "
                  f"median={routing_conf.median().item():.4f}")

            # 상관관계: routing이 confident할수록 gap이 작아지는지
            corr = torch.corrcoef(torch.stack([gap, -routing_conf]))[0, 1].item()
            print(f"  corr(gap, -confidence) = {corr:.3f}  "
                  f"(양수이면 confident할수록 gap이 작다는 가설 지지)")
            print(f"  {'─'*60}")

            print(f"\n  [해석]")
            print(f"  → soft-forward의 convergence error가 hard-forward보다 낮으면")
            print(f"    F_soft(x)가 진짜로 연속 함수이기 때문 (completeness 엄밀 성립).")
            print(f"  → gap이 작을수록 F_soft가 실제 예측(ŷ)의 좋은 근사임을 의미.")

            # ── 4. 그룹 전환 횟수 vs completeness error 상관관계 ──
            #
            # [가설] baseline→input 경로에서 hard_assignment(centroid 소속)가
            # 자주 바뀌는 샘플일수록 completeness error가 클 것이다.
            # (경로 위 불연속 점프 횟수가 많을수록 IG 근사가 어긋난다는 가설)
            #
            # 재학습 불필요 — 이미 학습된 모델에 50-step 경로를 재실행하며
            # 매 step의 hard_assignment만 추가로 기록하면 됨.
            #
            # n=85~100으로는 상관계수가 불안정할 수 있으므로:
            #   - Spearman 상관 사용 (mean/median 격차가 크게 나는 분포이므로
            #     정규성 가정이 깨질 가능성이 높음 — Pearson보다 안전)
            #   - bootstrap 95% CI 함께 보고 (|r|이 작아도 "관계 없음" 단정 금지)
            #   - 전환 여부(binary)가 아니라 전환 횟수(0,1,2,...)로 세분화
            print(f"\n  [4/4] 그룹 전환 횟수 vs completeness error 상관관계 "
                  f"(n_steps={args.ig_nsteps})...")

            n_steps_scan = args.ig_nsteps
            alphas_scan  = torch.linspace(0.0, 1.0, n_steps_scan, device=X_sc.device)
            X_base_exp   = (
                X_baseline.unsqueeze(0).expand_as(X_sc) if X_baseline.dim() == 1
                else X_baseline  # 이미 (N, F) — medoid baseline (샘플별로 다름)
            )

            with torch.no_grad():
                # 매 alpha step에서 hard_assignment 계산 → (n_steps_scan, N)
                assignments_over_path = torch.zeros(
                    n_steps_scan, X_sc.shape[0], dtype=torch.long, device=X_sc.device
                )
                for si, alpha in enumerate(alphas_scan):
                    x_interp = X_base_exp + alpha * (X_sc - X_base_exp)
                    q_i = F.normalize(model.embedder(x_interp), dim=-1)
                    c_i = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    assignments_over_path[si] = (q_i @ c_i.T).argmax(dim=-1)

                # 전환 횟수: 연속된 step 간 hard_assignment가 바뀐 횟수
                transitions = (assignments_over_path[1:] != assignments_over_path[:-1]).sum(dim=0)  # (N,)

            # per-sample completeness error 재계산 (hard-forward, signed 버전)
            # compute_integrated_gradients는 |IG| abs 값만 반환하므로,
            # completeness 검증에는 부호를 보존한 IG 합이 필요해 여기서 직접 계산
                f_x_h        = ig_target_fn(model(X_sc))
                f_baseline_h = ig_target_fn(model(X_base_exp))
            # signed IG 합이 필요하므로 (compute_integrated_gradients는 abs 반환)
            # 여기서 별도로 부호를 보존한 버전을 계산
            grads_accum = torch.zeros_like(X_sc)
            for alpha in alphas_scan:
                x_interp = (X_base_exp + alpha * (X_sc - X_base_exp)).clone().detach().requires_grad_(True)
                target = ig_target_fn(model(x_interp))
                grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
                grads_accum = grads_accum + grad
            avg_grad = grads_accum / n_steps_scan
            ig_signed = avg_grad * (X_sc - X_base_exp)
            ig_sum_signed = ig_signed.sum(dim=-1)  # (N,)

            actual_diff = f_x_h - f_baseline_h
            abs_error_per_sample = (ig_sum_signed - actual_diff).abs().cpu().numpy()
            transitions_np = transitions.cpu().numpy()

            # 전환 횟수별 completeness error 분포
            unique_trans = sorted(set(transitions_np.tolist()))
            print(f"\n  전환 횟수 분포 (50-step 경로 기준):")
            for t in unique_trans:
                mask_t = (transitions_np == t)
                n_t = mask_t.sum()
                if n_t == 0:
                    continue
                print(f"    전환={t:2d}회  n={n_t:3d}  "
                      f"completeness error mean={abs_error_per_sample[mask_t].mean():.4f}  "
                      f"median={np.median(abs_error_per_sample[mask_t]):.4f}")

            # Spearman 상관 + bootstrap 95% CI
            spearman_r, spearman_p = spearmanr(transitions_np, abs_error_per_sample)

            n_boot = 2000
            rng = np.random.default_rng(0)
            boot_rs = []
            n_samp = len(transitions_np)
            for _ in range(n_boot):
                idx = rng.integers(0, n_samp, n_samp)
                r_b, _ = spearmanr(transitions_np[idx], abs_error_per_sample[idx])
                if not np.isnan(r_b):
                    boot_rs.append(r_b)
            ci_low, ci_high = np.percentile(boot_rs, [2.5, 97.5])

            print(f"\n  Spearman ρ(전환 횟수, completeness error) = {spearman_r:.3f}  "
                  f"(p={spearman_p:.4f})")
            print(f"  Bootstrap 95% CI: [{ci_low:.3f}, {ci_high:.3f}]  (n_boot={n_boot})")
            print(f"  {'─'*60}")

            # 전환 횟수 자체가 한 값에 쏠려 있으면(포화) 상관계수를
            # "관계 있음/없음"의 근거로 쓸 수 없음 — 5단계(neighbor_change)와
            # 동일한 원리. 최빈값이 전체의 90% 이상이면 포화로 판정.
            unique_t, counts_t = np.unique(transitions_np, return_counts=True)
            max_count_ratio = counts_t.max() / n_samp
            is_transition_saturated = max_count_ratio >= 0.90

            print(f"\n  [해석]")
            if is_transition_saturated:
                dominant_val = unique_t[counts_t.argmax()]
                print(f"  🛑 전환 횟수 분포가 포화됨 (값={dominant_val}회에 "
                      f"{max_count_ratio*100:.0f}% 샘플 쏠림)")
                print(f"    → 이 지표는 변별력을 잃었으므로, 위 상관계수(ρ={spearman_r:.3f})는")
                print(f"      소수 이상치 샘플의 레버리지로 결정된 값일 수 있음.")
                print(f"      \"관계 있다/없다\" 어느 쪽 결론도 이 지표만으로는 내리지 말 것.")
                print(f"      \"이 지표·이 baseline 조합에서는 검증 불가능했다\"가 정확한 결론.")
            elif ci_low > 0:
                print(f"  ✅ CI가 0을 포함하지 않고 양수 → 전환 횟수가 많을수록")
                print(f"    completeness error가 커진다는 가설이 실증적으로 지지됨")
                print(f"    → 불연속(STE routing 경계 통과)이 IG 오차의 원인이라는")
                print(f"      가설이 뒷받침됨. Piecewise 접근이나 baseline 재설계")
                print(f"      (medoid baseline 등)을 고려할 근거가 생김.")
            elif ci_high < 0:
                print(f"  △ CI가 0을 포함하지 않고 음수 → 예상과 반대 방향")
                print(f"    (전환이 많을수록 오히려 오차가 작음) — 추가 조사 필요")
            else:
                print(f"  ⚠️  CI가 0을 포함 → 전환 횟수와 completeness error 사이")
                print(f"    통계적으로 유의한 관계를 확인하지 못함 (n={n_samp}, 표본 한계일 수 있음)")
                print(f"    → routing_probs의 거의-uniform 문제(온도 미조정)나")
                print(f"      agg_emb 쪽(top-K 이웃 변화)이 더 큰 원인일 가능성이 높아짐")

            # ── 5. agg_emb 경로: top-K 이웃 집합 변화 vs completeness error ──
            #
            # [배경] 그룹 전환(④) 실험에서 상관관계가 확인되지 않음(vehicle
            # ρ=-0.114, wine ρ=0.000, 둘 다 CI가 0 포함). 다음 유력 후보는
            # agg_emb 경로 — MemoryBank.retrieve가 반환하는 top-K 이웃 집합이
            # baseline→input 경로에서 얼마나 바뀌는지가 completeness error를
            # 설명할 수 있는지 확인한다.
            #
            # [측정 방식] 단순 Jaccard(집합 겹침 비율)가 아니라 evidence_w
            # 가중치를 반영한 차이를 쓴다 — Jaccard는 "바뀐 인덱스 개수"만
            # 보고 "그 이웃이 실제로 얼마나 중요했는지(evidence_w)"를 무시하기
            # 때문. 예를 들어 evidence_w가 낮은 이웃 10개가 바뀌는 것과
            # evidence_w가 가장 높은 이웃 1개가 바뀌는 것은 agg_emb에 미치는
            # 영향이 다른데, Jaccard는 이 둘을 구분하지 못한다.
            #
            # [relative error 함정 점검] 4단계에서 mean/median 사이 큰 괴리가
            # 있었으므로(예: 1회 전환 mean=22 vs median=15), 여기서도 raw
            # abs_error와 정규화된 값 모두 로깅해 결과 해석 시 폭주하는 소수
            # 샘플에 상관계수가 끌려가는지 직접 확인할 수 있게 한다.
            if model.memory.filled.item() >= model.k:
                print(f"\n  [5/5] agg_emb top-K 이웃 변화 vs completeness error...")

                # ── 5-0. Fallback 비율 측정 (선행 진단) ──────────────
                #
                # [배경] neighbor_change가 포화(mean/median≈1.0)되면 원인이
                # completeness와 무관하게 "K가 평균 그룹 크기보다 훨씬 커서
                # cross-group fallback이 상시 발동하기 때문"일 수 있다.
                # retrieve() 내부의 fallback_mask = (grp_sizes < k) 조건을
                # 그대로 재현해서, input/baseline 각각에서 fallback을 타는
                # 샘플 비율을 직접 측정한다. 재학습 불필요, 기존 캐시
                # (_cached_group_sizes)만 조회하면 되는 저비용 진단.
                cached_sizes = getattr(model.memory, "_cached_group_sizes", None)
                fallback_rate_x = None
                avg_group_size = None
                if cached_sizes is not None:
                    with torch.no_grad():
                        q_x0 = F.normalize(model.embedder(X_sc), dim=-1)
                        ha_x0 = (q_x0 @ F.normalize(model.prototype_layer.centroid_emb, dim=-1).T).argmax(dim=-1)
                        grp_sizes_x = cached_sizes[ha_x0]
                        fallback_rate_x = (grp_sizes_x < model.k).float().mean().item()

                        q_b0 = F.normalize(model.embedder(X_base_exp), dim=-1)
                        ha_b0 = (q_b0 @ F.normalize(model.prototype_layer.centroid_emb, dim=-1).T).argmax(dim=-1)
                        grp_sizes_b = cached_sizes[ha_b0]
                        fallback_rate_b = (grp_sizes_b < model.k).float().mean().item()

                        avg_group_size = cached_sizes[cached_sizes > 0].float().mean().item()

                    print(f"\n  [Fallback 진단] k={model.k}, 평균 alive 그룹 크기={avg_group_size:.1f}")
                    print(f"    input 시점    fallback 비율: {fallback_rate_x*100:.1f}%  "
                          f"({int(fallback_rate_x*X_sc.shape[0])}/{X_sc.shape[0]} 샘플)")
                    print(f"    baseline 시점 fallback 비율: {fallback_rate_b*100:.1f}%")
                    if model.k > avg_group_size:
                        print(f"    ⚠️  k({model.k}) > 평균 그룹 크기({avg_group_size:.1f})")
                        print(f"       → cross-group fallback이 구조적으로 상시 발동할 가능성 높음")
                        print(f"       → 아래 neighbor_change가 포화(≈1.0)되어 있다면, 이건")
                        print(f"          completeness와 무관하게 이 fallback 구조 때문일 수 있음")
                    print()
                else:
                    print(f"\n  [Fallback 진단] _cached_group_sizes 없음 — skip")

                with torch.no_grad():
                    # baseline과 input 각각에서의 top-K 이웃 + evidence_w
                    q_x = F.normalize(model.embedder(X_sc), dim=-1)
                    ha_x = (q_x @ F.normalize(model.prototype_layer.centroid_emb, dim=-1).T).argmax(dim=-1)
                    nk_x, nv_x, nl_x, topk_x = model.memory.retrieve(
                        model.embedder(X_sc), model.k,
                        hard_assignment=(None if model.global_retrieve else ha_x),
                    )
                    _, evidence_w_x = model.ot_selector(model.embedder(X_sc), nk_x, nv_x, nl_x)

                    q_b = F.normalize(model.embedder(X_base_exp), dim=-1)
                    ha_b = (q_b @ F.normalize(model.prototype_layer.centroid_emb, dim=-1).T).argmax(dim=-1)
                    nk_b, nv_b, nl_b, topk_b = model.memory.retrieve(
                        model.embedder(X_base_exp), model.k,
                        hard_assignment=(None if model.global_retrieve else ha_b),
                    )
                    _, evidence_w_b = model.ot_selector(model.embedder(X_base_exp), nk_b, nv_b, nl_b)

                    # 가중치 기반 이웃 집합 차이:
                    # 각 샘플에서 "input 시점에 중요했던 이웃(evidence_w 높음)이
                    # baseline 시점에도 top-K에 남아있었는가"를 evidence_w로
                    # 가중해 측정. 완전히 같은 집합+같은 가중치면 0.
                    neighbor_change = torch.zeros(X_sc.shape[0], device=X_sc.device)
                    for i in range(X_sc.shape[0]):
                        set_x = set(topk_x[i].tolist())
                        set_b = set(topk_b[i].tolist())
                        common = set_x & set_b
                        # input 시점 evidence_w 중, baseline에도 남아있는 이웃에
                        # 걸린 가중치의 합 → 1에 가까울수록 "변화 없음"
                        if len(set_x) > 0:
                            kept_weight = sum(
                                evidence_w_x[i, j].item()
                                for j, idx in enumerate(topk_x[i].tolist())
                                if idx in common
                            )
                            neighbor_change[i] = 1.0 - kept_weight  # 0=변화없음, 1=완전히 다름

                neighbor_change_np = neighbor_change.cpu().numpy()

                # 원본 로그(raw abs_error)와 함께 보고 — 폭주하는 소수 샘플이
                # 상관계수를 왜곡하는지 확인 가능하도록
                print(f"  neighbor_change 분포: mean={neighbor_change_np.mean():.4f}  "
                      f"median={np.median(neighbor_change_np):.4f}  "
                      f"max={neighbor_change_np.max():.4f}")
                print(f"  completeness error(raw) 분포: mean={abs_error_per_sample.mean():.4f}  "
                      f"median={np.median(abs_error_per_sample):.4f}  "
                      f"max={abs_error_per_sample.max():.4f}")

                spearman_r2, spearman_p2 = spearmanr(neighbor_change_np, abs_error_per_sample)
                boot_rs2 = []
                for _ in range(n_boot):
                    idx = rng.integers(0, n_samp, n_samp)
                    r_b2, _ = spearmanr(neighbor_change_np[idx], abs_error_per_sample[idx])
                    if not np.isnan(r_b2):
                        boot_rs2.append(r_b2)
                ci_low2, ci_high2 = np.percentile(boot_rs2, [2.5, 97.5])

                print(f"\n  Spearman ρ(neighbor_change, completeness error) = {spearman_r2:.3f}  "
                      f"(p={spearman_p2:.4f})")
                print(f"  Bootstrap 95% CI: [{ci_low2:.3f}, {ci_high2:.3f}]  (n_boot={n_boot})")
                print(f"  {'─'*60}")

                # ── 정식 Jackknife 검증 (85-fold, 단일 포인트 확인이 아님) ──
                #
                # [배경] 단일 최댓값 하나만 빼는 건 진짜 leave-one-out이
                # 아니라 "단일 포인트 민감도 확인"에 불과함 — 2, 3번째로
                # 큰 값들이 함께 상관관계를 떠받치고 있을 가능성을 배제
                # 못 함. neighbor_change가 median=0인 zero-inflated 분포이므로
                # 상위 소수 샘플이 상관계수 대부분을 만들어낼 위험이 있음.
                #
                # 방법: n_samp번 반복해서 매번 샘플 하나씩 제외한 상관계수를
                # 계산 (jackknife). 이 분포의 최솟값(가장 크게 흔들렸을 때)과
                # p-value가 유의성 문턱(0.05)을 넘는 fold의 비율을 함께 본다.
                # p-value가 fold 중 상당수에서 문턱을 넘나들면, 이 관계는
                # 소수 포인트에 의존하는 취약한 신호로 판정한다.
                jack_rs, jack_ps = [], []
                for j in range(n_samp):
                    jmask = np.ones(n_samp, dtype=bool)
                    jmask[j] = False
                    r_j, p_j = spearmanr(neighbor_change_np[jmask], abs_error_per_sample[jmask])
                    if not np.isnan(r_j):
                        jack_rs.append(r_j)
                        jack_ps.append(p_j)
                jack_rs = np.array(jack_rs)
                jack_ps = np.array(jack_ps)

                frac_not_significant = (jack_ps >= 0.05).mean()
                r_range = jack_rs.max() - jack_rs.min()

                print(f"\n  [Jackknife 검증] {n_samp}-fold (매번 샘플 1개씩 제외)")
                print(f"    ρ 범위: [{jack_rs.min():.3f}, {jack_rs.max():.3f}]  "
                      f"(원본 ρ={spearman_r2:.3f}, 범위폭={r_range:.3f})")
                print(f"    p≥0.05인 fold 비율: {frac_not_significant*100:.1f}%  "
                      f"({int(frac_not_significant*n_samp)}/{n_samp})")

                # 판정: fold의 상당수(20% 이상)에서 유의성이 사라지거나,
                # 부호가 바뀌는 fold가 하나라도 있으면 신뢰 불가로 판정.
                # 단일 포인트 확인보다 훨씬 엄격한 기준.
                sign_flip_in_jack = (np.sign(jack_rs) != np.sign(spearman_r2)).any()
                is_leverage_driven = (frac_not_significant > 0.20) or sign_flip_in_jack

                if is_leverage_driven:
                    print(f"    🛑 Jackknife 결과 다수 fold에서 유의성이 불안정함")
                    print(f"       (p≥0.05 비율 {frac_not_significant*100:.1f}%, "
                          f"부호 반전 fold 존재: {sign_flip_in_jack})")
                    print(f"       → 이 상관관계는 소수 포인트에 의존하는 취약한 신호로 보이며,")
                    print(f"         독립적인 증거로 사용할 수 없음. ④번과 동일하게")
                    print(f"         \"검증 불가능\"으로 재분류해야 함.")
                else:
                    print(f"    ✅ Jackknife 전체 fold에서 유의성과 부호가 안정적으로 유지됨")

                # ── Partial correlation: confound(경계 근접성) 통제 ──
                #
                # [배경] neighbor_change와 completeness error가 실제로는
                # "경계 근접성"이라는 제3의 변수(routing이 애매한 샘플일수록
                # 이웃도 잘 바뀌고, 동시에 그 지점에서 F가 원래 더 가파를
                # 수 있음)에서 함께 나오는 confound일 가능성이 있음.
                # routing_confidence(top-1 softmax 확률)와 raw feature-space
                # 거리(baseline-input) 두 후보를 통제 변수로 써서 이 대안
                # 설명을 배제할 수 있는지 확인한다.
                def _partial_spearman(x, y, z):
                    """x, y의 Spearman 상관에서 z의 선형 영향을 제거한 partial correlation.
                    순위로 변환 후 z에 대해 residual을 구하고, 그 residual끼리 상관 계산."""
                    from scipy.stats import rankdata
                    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
                    def _resid(a, b):
                        slope, intercept = np.polyfit(b, a, 1)
                        return a - (slope * b + intercept)
                    rx_resid = _resid(rx, rz)
                    ry_resid = _resid(ry, rz)
                    r_partial = np.corrcoef(rx_resid, ry_resid)[0, 1]
                    return r_partial

                with torch.no_grad():
                    q_conf = F.normalize(model.embedder(X_sc), dim=-1)
                    c_conf = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    routing_confidence_np = F.softmax(q_conf @ c_conf.T, dim=-1).max(dim=-1).values.cpu().numpy()

                # raw feature-space 거리 (baseline-input) — 두 번째 confound 후보.
                # 경로가 길수록 discretization도, neighbor_change도 커질 수 있음.
                raw_dist_np = (X_sc - X_base_exp).norm(dim=-1).cpu().numpy()

                confound_vars = {
                    "routing_confidence": routing_confidence_np,
                    "raw_feature_distance": raw_dist_np,
                }

                is_leverage_driven_confound = False
                for cname, cvar in confound_vars.items():
                    # ── trivial 여부 먼저 확인 ──────────────────────────
                    # partial correlation이 "안 변했다"는 결과가 의미 있으려면,
                    # 통제 변수가 애초에 두 원본 변수와 어느 정도 상관이
                    # 있어야 한다. 만약 통제 변수가 둘 다와 거의 무관하면
                    # (|corr|<0.1), "통제해봤자 변화 없음"은 confound를
                    # 배제한 게 아니라 "통제할 게 없었다"는 뜻일 뿐이다.
                    r_c_nc, _  = spearmanr(cvar, neighbor_change_np)
                    r_c_err, _ = spearmanr(cvar, abs_error_per_sample)
                    is_trivial = (abs(r_c_nc) < 0.10) and (abs(r_c_err) < 0.10)

                    print(f"\n  [Partial correlation] {cname} 통제:")
                    print(f"    corr({cname}, neighbor_change)      = {r_c_nc:.3f}")
                    print(f"    corr({cname}, completeness error)   = {r_c_err:.3f}")

                    if is_trivial:
                        print(f"    ⚠️  {cname}이 두 변수 모두와 거의 무관함(|corr|<0.10)")
                        print(f"       → 아래 partial correlation은 \"confound 배제\"가 아니라")
                        print(f"         \"통제할 게 애초에 없었다\"는 trivial한 결과일 수 있음")

                    r_partial = _partial_spearman(neighbor_change_np, abs_error_per_sample, cvar)
                    print(f"    ρ_partial(neighbor_change, completeness error | {cname}) = {r_partial:.3f}")
                    print(f"    (원본 ρ={spearman_r2:.3f}과 비교)")

                    if not is_trivial:
                        confound_reduces = abs(r_partial) < abs(spearman_r2) * 0.5
                        if confound_reduces:
                            print(f"    🛑 통제 후 상관관계가 절반 이하로 줄어듦 (non-trivial confound)")
                            print(f"       → {cname}이 관계의 상당 부분을 설명하는 공통 원인일 수 있음")
                            is_leverage_driven_confound = True
                        else:
                            print(f"    ✅ non-trivial한 통제 변수인데도 상관관계 유지됨")

                is_leverage_driven = is_leverage_driven or is_leverage_driven_confound

                print(f"\n  [주의] 위 partial correlation은 관찰적 상관관계에 대한 확인일 뿐,")
                print(f"    인과관계의 직접적 증거는 아님. 인과관계를 확인하려면 top-K 이웃을")
                print(f"    input 시점 기준으로 고정한 채 IG를 재계산해, agg_emb의 이산적")
                print(f"    전환을 인위적으로 제거했을 때 completeness error가 실제로")
                print(f"    줄어드는지 별도로 검증해야 함 (gold-standard 인과 검증, 미실시).")

                # neighbor_change 자체가 포화되어 있거나(is_saturated),
                # leave-one-out에서 이상치 하나에 결과가 좌우되면
                # (is_leverage_driven) 상관관계를 "관계 있음"의 근거로
                # 쓸 수 없음 — 두 조건 모두 confirmation bias 방지를 위한
                # 필수 게이트. ④번에서 이미 한 번 놓쳤던 실수(같은 데이터에
                # 다른 기준 적용)를 반복하지 않기 위해 이 둘을 함께 체크한다.
                nc_median = float(np.median(neighbor_change_np))
                nc_std    = float(neighbor_change_np.std())
                is_saturated = (nc_median > 0.95) or (nc_std < 0.05)
                is_unreliable = is_saturated or is_leverage_driven

                print(f"\n  [해석]")
                if is_unreliable:
                    if is_saturated:
                        print(f"  🛑 neighbor_change 자체가 포화됨 (median={nc_median:.3f}, "
                              f"std={nc_std:.3f})")
                    if is_leverage_driven:
                        print(f"  🛑 Jackknife 또는 partial correlation 검증 실패 — 소수 포인트")
                        print(f"    의존 또는 confound 가능성이 확인됨")
                    print(f"    → 이 지표는 신뢰할 수 없으므로, 위 상관계수(ρ={spearman_r2:.3f})는")
                    print(f"      \"agg_emb가 원인이다/아니다\" 어느 쪽의 근거로도 쓸 수 없음.")
                    print(f"      \"이 지표·이 데이터셋 조합에서는 검증 불가능했다\"가 정확한 결론.")
                    if fallback_rate_x is not None and fallback_rate_x > 0.5:
                        print(f"    → 위 Fallback 진단(input 시점 {fallback_rate_x*100:.0f}%)과 함께 보면,")
                        print(f"      k={model.k}가 평균 그룹 크기({avg_group_size:.1f})보다 커서")
                        print(f"      cross-group fallback이 상시 발동하는 구조적 문제로 보임.")
                        print(f"      → 다음 확인: k가 평균 그룹 크기보다 작은 데이터셋에서")
                        print(f"        같은 진단을 반복해 neighbor_change 분산이 회복되는지 확인.")
                elif ci_low2 > 0:
                    print(f"  ✅ CI가 0을 포함하지 않고 양수 → top-K 이웃 집합이 많이")
                    print(f"    바뀔수록 completeness error가 커짐. agg_emb 경로의")
                    print(f"    불연속(retrieve의 hard_assignment 의존)이 IG 오차의")
                    print(f"    주요 원인이라는 가설이 뒷받침됨.")
                    print(f"    → 다음 개선 방향: retrieve top-K 내에서 evidence_w를")
                    print(f"      soft하게 유지하는 방식 (전체 memory attention 아님,")
                    print(f"      이미 뽑힌 K개 내에서만 재가중치 부여) 검토 가능.")
                elif ci_high2 < 0:
                    print(f"  △ CI가 0을 포함하지 않고 음수 → 예상과 반대 방향")
                    print(f"    (이웃이 많이 바뀔수록 오히려 오차가 작음) — 추가 조사 필요")
                else:
                    print(f"  ⚠️  CI가 0을 포함 → neighbor_change와 completeness error")
                    print(f"    사이 통계적으로 유의한 관계를 확인하지 못함")
                    print(f"    → context_emb(④)와 agg_emb(⑤) 둘 다 단독 원인이 아닐")
                    print(f"      가능성. relative error 지표 자체의 문제(f(x)-f(baseline)가")
                    print(f"      0에 가까운 샘플에서 분모가 작아 폭주)나, 두 경로의")
                    print(f"      상호작용(교호작용) 효과를 봐야 할 수 있음.")
            else:
                print(f"\n  [5/5] Memory 미충족 — agg_emb 진단 skip")

            # ── 6. Gold-standard 인과 검증: 이웃을 input 기준으로 고정 ──
            #
            # [배경] 5단계에서 확인된 것은 관찰적 상관관계(neighbor_change와
            # completeness error)일 뿐, agg_emb 경로의 불연속이 "진짜 원인"
            # 이라는 인과적 증거는 아니다. 이 단계는 retrieve()가 경로 중간에
            # hard_assignment 변화에 따라 이웃을 다시 뽑는 것을 막고, input(X)
            # 시점에 뽑힌 이웃을 경로 전체에서 고정한 채 IG를 재계산한다.
            # 즉 agg_emb 경로의 이산적 전환만 인위적으로 제거한 상태.
            #
            # 이 상태에서 completeness error가 실제로 줄어들면, neighbor_change
            # ↔ completeness error 관계가 상관관계를 넘어 인과관계라는
            # gold-standard 증거가 되며, 동시에 "top-K 내 soft weighting"
            # 개선책이 실제로 작동할지에 대한 저비용 사전 검증(pilot)이 된다.
            if model.memory.filled.item() >= model.k:
                print(f"\n  [6/6] Gold-standard 인과 검증: 이웃을 input 시점으로 고정...")

                fixed_nk, fixed_labels = model.get_fixed_neighbors_for_ig(X_sc)

                def _fixed_neighbors_target_fn(x_batch, target_fn_inner):
                    return model.forward_fixed_neighbors_for_ig(
                        x_batch, target_fn_inner, fixed_nk, fixed_labels
                    )

                alphas_fixed = torch.linspace(0.0, 1.0, n_steps_scan, device=X_sc.device)
                grads_accum_fixed = torch.zeros_like(X_sc)
                for alpha in alphas_fixed:
                    x_interp = (X_base_exp + alpha * (X_sc - X_base_exp)).clone().detach().requires_grad_(True)
                    target = _fixed_neighbors_target_fn(x_interp, ig_target_fn)
                    grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
                    grads_accum_fixed = grads_accum_fixed + grad
                avg_grad_fixed = grads_accum_fixed / n_steps_scan
                ig_signed_fixed = avg_grad_fixed * (X_sc - X_base_exp)
                ig_sum_fixed = ig_signed_fixed.sum(dim=-1)

                with torch.no_grad():
                    f_x_fixed = _fixed_neighbors_target_fn(X_sc, ig_target_fn)
                    f_b_fixed = _fixed_neighbors_target_fn(X_base_exp, ig_target_fn)
                actual_diff_fixed = f_x_fixed - f_b_fixed
                abs_error_fixed = (ig_sum_fixed - actual_diff_fixed).abs().cpu().numpy()

                print(f"    이웃 고정 전(원본, agg_emb 경로 불연속 포함):")
                print(f"      completeness error mean={abs_error_per_sample.mean():.4f}  "
                      f"median={np.median(abs_error_per_sample):.4f}")
                print(f"    이웃 고정 후(agg_emb 경로 이산적 전환 제거):")
                print(f"      completeness error mean={abs_error_fixed.mean():.4f}  "
                      f"median={np.median(abs_error_fixed):.4f}")

                # median 기준 감소율로 판정 (mean은 이상치에 취약하므로 참고용)
                med_before = np.median(abs_error_per_sample)
                med_after  = np.median(abs_error_fixed)
                reduction  = (med_before - med_after) / (med_before + 1e-8)

                print(f"\n  [해석]")
                if reduction > 0.3:
                    print(f"  ✅ 이웃을 고정하니 completeness error가 {reduction*100:.1f}% 감소함")
                    print(f"    → agg_emb 경로의 이산적 전환이 IG 오차의 인과적 원인이라는")
                    print(f"      gold-standard 증거. 5단계의 상관관계가 진짜 인과관계였음을")
                    print(f"      뒷받침함. \"retrieve top-K 내 soft weighting\" 개선 방향이")
                    print(f"      실제로 completeness를 개선할 가능성이 높다는 사전 검증 완료.")
                elif reduction < -0.1:
                    print(f"  △ 이웃을 고정했더니 오히려 오차가 커짐 — 예상과 반대 방향")
                    print(f"    (다른 경로의 오차가 이 조작으로 드러났을 가능성) — 추가 조사 필요")
                else:
                    print(f"  ⚠️  이웃 고정으로 인한 유의미한 변화 없음 ({reduction*100:.1f}%)")
                    print(f"    → agg_emb 경로의 불연속이 completeness error의 주된 원인이")
                    print(f"      아닐 수 있음. 5단계의 상관관계는 confound 또는 다른 경로")
                    print(f"      (context_emb, 혹은 두 경로의 상호작용)에서 왔을 가능성")
            else:
                print(f"\n  [6/6] Memory 미충족 — gold-standard 인과 검증 skip")

            print(f"\n  [참고] 지금까지의 검증(④,⑤,jackknife,partial correlation,⑥)은")
            print(f"    이 대화에서 진행한 다수의 탐색적(exploratory) 검정 중 하나임.")
            print(f"    다중비교로 인해 개별 결과가 우연히 유의하게 나올 가능성이 개별")
            print(f"    검정보다 높으므로, 다른 seed나 데이터셋에서의 재현은 사전에 정한")
            print(f"    확증적(confirmatory) 검증으로 명확히 구분해 보고할 것.")

        # ── nsteps_sweep: n_steps 스윕으로 discretization vs 불연속 분리 ──
        #
        # [배경] medoid baseline이 completeness error를 크게 개선했지만
        # (vehicle: median relative 146%→17.5%), 이게 "불연속을 안 만나서"인지
        # "단순히 경로가 짧아져서(discretization 오차 감소)"인지 지금까지의
        # 실험만으로는 구분이 안 됨.
        #
        # [검증 원리] n_steps를 늘리며 completeness error 변화를 관찰:
        #   - discretization이 주원인 → 오차가 1/n_steps에 비례해 매끄럽게 감소
        #   - 불연속(경로 중 hard_assignment 점프)이 주원인 → 어느 지점부터
        #     오차 감소가 멈추고 평평해짐(plateau) — Riemann sum이 jump를
        #     근본적으로 못 잡기 때문에 step을 늘려도 해결 안 됨
        #
        # mean과 medoid 두 baseline을 함께 스윕해서 비교선으로 사용:
        #   mean이 500 step에서도 plateau → 불연속이 원인, medoid가 그걸 회피
        #   mean이 500 step에서 medoid 수준까지 떨어짐 → 경로 길이가 주원인
        elif args.ablation == "nsteps_sweep":
            model.eval()
            n_test = min(100, X_test.shape[0])
            X_sw   = X_test[:n_test].clone()

            n_steps_list = [50, 100, 200, 500, 1000, 2000]

            with torch.no_grad():
                logits_orig = model(X_sw)["logits"]
                if tasktype == "multiclass":
                    target_class = torch.softmax(logits_orig, dim=-1).argmax(dim=-1).cpu().numpy()
                else:
                    target_class = None
            ig_target_fn = make_logit_target_fn(
                tasktype, target_class=target_class if tasktype == "multiclass" else None,
            )

            # medoid baseline (샘플별)
            with torch.no_grad():
                q_med = F.normalize(model.embedder(X_sw), dim=-1)
                c_med = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                ha_med = (q_med @ c_med.T).argmax(dim=-1)
                X_baseline_medoid = model.prototype_layer.centroid_x[ha_med].to(X_sw.device)

            X_baseline_mean = X_train.mean(dim=0)

            print(f"\n  n_steps Sweep: discretization vs 불연속 분리 (n={n_test})")
            print(f"  {'─'*70}")
            print(f"  {'n_steps':>8} │ {'mean median%':>14} {'mean mean%':>12} │ "
                  f"{'medoid median%':>16} {'medoid mean%':>14}")
            print(f"  {'─'*70}")

            results = {"mean": [], "medoid": []}

            def _completeness_rel_error(X_baseline_this, n_steps):
                if X_baseline_this.dim() == 1:
                    X_base_this = X_baseline_this.unsqueeze(0).expand_as(X_sw)
                else:
                    X_base_this = X_baseline_this

                alphas = torch.linspace(0.0, 1.0, n_steps, device=X_sw.device)
                grads_accum = torch.zeros_like(X_sw)
                for alpha in alphas:
                    x_interp = (X_base_this + alpha * (X_sw - X_base_this)).clone().detach().requires_grad_(True)
                    target = ig_target_fn(model(x_interp))
                    grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
                    grads_accum = grads_accum + grad
                avg_grad = grads_accum / n_steps
                ig_signed = avg_grad * (X_sw - X_base_this)
                ig_sum = ig_signed.sum(dim=-1)

                with torch.no_grad():
                    f_x = ig_target_fn(model(X_sw))
                    f_b = ig_target_fn(model(X_base_this))
                actual_diff = f_x - f_b
                abs_err = (ig_sum - actual_diff).abs()
                rel_err = abs_err / (actual_diff.abs() + 1e-8)
                return rel_err.median().item(), rel_err.mean().item()

            for n_steps in n_steps_list:
                med_mean, mean_mean = _completeness_rel_error(X_baseline_mean, n_steps)
                med_medoid, mean_medoid = _completeness_rel_error(X_baseline_medoid, n_steps)
                results["mean"].append((n_steps, med_mean, mean_mean))
                results["medoid"].append((n_steps, med_medoid, mean_medoid))
                print(f"  {n_steps:>8} │ {med_mean*100:>13.2f}% {mean_mean*100:>11.2f}% │ "
                      f"{med_medoid*100:>15.2f}% {mean_medoid*100:>13.2f}%")

            print(f"  {'─'*70}")

            # ── 패턴 판정: 절대 수준(asymptote) 기반 ──────────────
            # [이전 버그] 상대 감소율(last_decrease < 0.1)만으로 plateau를
            # 판정하면, 33%→0.19%처럼 절대 수준이 100배 이상 다른 두 곡선을
            # 구분하지 못한다 — 둘 다 "여전히 감소 중"으로 보이기 때문.
            # 실제로는 mean이 200→500 구간에서도 25.8% 감소하고 있었지만,
            # 그 절대 수준(33%)이 medoid(0.19%)보다 압도적으로 높다는 게
            # 핵심 신호였음. plateau 여부가 아니라 "충분히 많은 step에서도
            # 남아있는 절대 오차 수준"을 직접 비교해야 한다.
            #
            # [주의] 4개 점(50,100,200,500)만으로는 mean baseline이 정말
            # 어떤 값 근처에서 평평해지는지, 아니면 계속 감소해서 결국
            # medoid 수준까지 갈지 확정할 수 없다. 아래 판정은 "현재까지
            # 관찰된 데이터에 근거한 잠정적 신호"이며, n_steps=1000~2000
            # 추가 확인 전까지는 확정으로 서술하지 않는다.
            mean_first, mean_last = results["mean"][0][1], results["mean"][-1][1]
            medoid_first, medoid_last = results["medoid"][0][1], results["medoid"][-1][1]

            # 마지막 두 구간(200→500) 감소율은 참고용으로만 출력 (판정에는 미사용)
            mean_last_decrease   = (results["mean"][-2][1]   - mean_last)   / (results["mean"][-2][1]   + 1e-8)
            medoid_last_decrease = (results["medoid"][-2][1] - medoid_last) / (results["medoid"][-2][1] + 1e-8)

            print(f"\n  [절대 수준 비교] (n_steps={n_steps_list[-1]} 기준, 참고용 감소율 포함)")
            print(f"    mean baseline  : {mean_last*100:.2f}%  "
                  f"(직전 구간 대비 {mean_last_decrease*100:.1f}% 추가 감소 중)")
            print(f"    medoid baseline: {medoid_last*100:.2f}%  "
                  f"(직전 구간 대비 {medoid_last_decrease*100:.1f}% 추가 감소 중)")

            # 절대 수준 기준 판정 (경험적 임계값 — 10%는 "여전히 큰 잔여 오차",
            # 5% 미만은 "거의 수렴"으로 간주. 두 임계값 모두 근거는 관례적이며
            # 엄밀한 통계 기준이 아니므로, 결과 해석 시 이 점을 감안할 것)
            HIGH_THRESH = 0.10
            LOW_THRESH  = 0.05

            print(f"\n  [해석] (잠정적 — n_steps={n_steps_list[-1]}까지의 관찰에 근거)")
            if mean_last > HIGH_THRESH and medoid_last < LOW_THRESH:
                print(f"  ✅ mean baseline은 n_steps={n_steps_list[-1]}에서도 여전히 큰 잔여 오차")
                print(f"    ({mean_last*100:.1f}%)를 보이는 반면, medoid baseline은 거의 수렴")
                print(f"    ({medoid_last*100:.1f}%)함. 이는 mean의 잔여 오차 대부분이")
                print(f"    discretization이 아니라 불연속(hard_assignment 점프) 때문이라는")
                print(f"    가설과 정합적임 — 만약 discretization만의 문제였다면 medoid와")
                print(f"    비슷한 step 수에서 비슷한 수준으로 수렴했어야 함.")
                print(f"\n    ⚠️  다만 mean baseline이 직전 구간 대비 여전히")
                print(f"    {mean_last_decrease*100:.1f}% 감소 중이므로, 진짜 asymptote(수렴값)를")
                print(f"    확정하려면 n_steps=1000~2000까지 추가 확인이 필요함.")
                print(f"    현재로서는 \"불연속 가설을 지지하는 잠정적 증거\"로 서술하는 것이")
                print(f"    \"확정적으로 입증됨\"이라고 서술하는 것보다 정확함.")
            elif mean_last < LOW_THRESH:
                print(f"  △ mean baseline도 n_steps={n_steps_list[-1]}에서 낮은 수준")
                print(f"    ({mean_last*100:.1f}%)까지 도달함 → 개선의 상당 부분이")
                print(f"    경로 길이(discretization) 때문일 가능성이 있음")
                print(f"    → medoid의 이점은 '더 적은 step으로 같은 정확도'라는")
                print(f"      계산 효율성 측면으로 재해석해야 할 수 있음")
            else:
                print(f"  ⚠️  두 baseline 모두 아직 뚜렷한 수렴 신호를 보이지 않음")
                print(f"    (mean: {mean_first*100:.1f}%→{mean_last*100:.1f}%, "
                      f"medoid: {medoid_first*100:.1f}%→{medoid_last*100:.1f}%)")
                print(f"    → n_steps를 1000 이상으로 늘리거나 여러 seed로 반복 검증 필요")

            print(f"\n  [교차 검증 제안]")
            print(f"    같은 데이터로 그룹 전환(④) 상관관계를 n_steps={n_steps_list[-1]}로")
            print(f"    재계산해보면(--ablation soft_ig_check --ig_nsteps {n_steps_list[-1]}) "
                  f"discretization 노이즈가")
            print(f"    줄어든 상태에서 \"전환 횟수가 많을수록 잔여 오차가 크다\"는 관계가")
            print(f"    더 선명하게 드러나는지 확인 가능 (불연속 가설의 독립적 교차 검증).")

        # ── illusion_check: mean baseline 우위가 completeness 붕괴로 인한
        #    착시인지 검증 ──
        #
        # [배경] mean baseline에서 TabERA(IG)가 SHAP/Random보다 deletion/
        # insertion AUC에서 우월했는데, medoid baseline(completeness가 훨씬
        # 좋음)으로 바꾸니 오히려 역전됨. 이게 "mean baseline에서는
        # completeness가 깨져서 attribution이 과장된 착시 신호를 만든
        #것"인지, 아니면 다른 이유인지 확인이 필요함.
        #
        # [방법] mean baseline 결과 안에서:
        #   샘플별 completeness error (얼마나 IG가 부정확한가)
        #   샘플별 deletion AUC 기여도 (그 샘플에서 attribution이 얼마나
        #     "효과적으로" 보였는가 — 값이 낮을수록 attribution이 예측을
        #     잘 설명하는 것처럼 보임)
        #   이 둘의 상관관계를 본다. completeness error가 클수록 deletion
        #   AUC가 낮게(좋게) 나온다면, "부정확한 IG일수록 우연히 좋아
        #   보이는" 착시 패턴이 실증적으로 확인되는 것.
        #
        # [설계 원칙 — 이 대화에서 반복된 실수를 방지하기 위해 처음부터 포함]
        #   1. 두 변수의 분포(히스토그램 요약)를 상관계수 계산 전에 먼저 출력
        #      → 포화 여부를 상관계수 해석 전에 미리 확인
        #   2. Spearman + 85-fold jackknife를 한 번에 실행
        #   3. confound 후보(raw_feature_distance, 예측 확신도 max_prob)를
        #      한 번에 통제 — "나중에 발견해서 재검증"이 아니라 처음부터 포함
        elif args.ablation == "illusion_check":
            from scipy.stats import spearmanr, rankdata

            model.eval()
            n_test = min(100, X_test.shape[0])
            X_ic   = X_test[:n_test].clone()
            X_baseline_mean = X_train.mean(dim=0)

            print(f"\n  Illusion Check: mean baseline 우위가 착시인지 검증 (n={n_test})")
            print(f"  {'─'*70}")

            with torch.no_grad():
                logits_orig = model(X_ic)["logits"]
                if tasktype == "regression":
                    target_class_ic = None
                    max_prob_np = None
                elif tasktype == "multiclass":
                    probs_ic = torch.softmax(logits_orig, dim=-1)
                    target_class_ic = probs_ic.argmax(dim=-1).cpu().numpy()
                    max_prob_np = probs_ic.max(dim=-1).values.cpu().numpy()
                else:
                    probs_ic = torch.sigmoid(logits_orig.squeeze(-1))
                    target_class_ic = None
                    max_prob_np = torch.where(probs_ic >= 0.5, probs_ic, 1 - probs_ic).cpu().numpy()

            ig_target_fn_ic = make_logit_target_fn(
                tasktype, target_class=target_class_ic if tasktype == "multiclass" else None,
            )

            # ── Step 1. 샘플별 completeness error (mean baseline, signed IG) ──
            print(f"  [1/4] 샘플별 completeness error 계산 중 (mean baseline)...")
            n_steps_ic = args.ig_nsteps
            alphas_ic = torch.linspace(0.0, 1.0, n_steps_ic, device=X_ic.device)
            X_base_ic = X_baseline_mean.unsqueeze(0).expand_as(X_ic)

            grads_accum_ic = torch.zeros_like(X_ic)
            for alpha in alphas_ic:
                x_interp = (X_base_ic + alpha * (X_ic - X_base_ic)).clone().detach().requires_grad_(True)
                target = ig_target_fn_ic(model(x_interp))
                grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
                grads_accum_ic = grads_accum_ic + grad
            avg_grad_ic = grads_accum_ic / n_steps_ic
            ig_signed_ic = avg_grad_ic * (X_ic - X_base_ic)
            tabera_imp_ic = ig_signed_ic.abs().detach().cpu().numpy()

            with torch.no_grad():
                f_x_ic = ig_target_fn_ic(model(X_ic))
                f_b_ic = ig_target_fn_ic(model(X_base_ic))
            completeness_error = (ig_signed_ic.sum(dim=-1) - (f_x_ic - f_b_ic)).abs().cpu().numpy()

            # ── Step 2. 샘플별 deletion AUC 기여도 (mean baseline, TabERA IG 순위) ──
            print(f"  [2/4] 샘플별 deletion AUC 기여도 계산 중...")
            n_features_ic = model.n_features

            def _pred_at(x_batch, sample_idx):
                with torch.no_grad():
                    lg = model(x_batch.unsqueeze(0))["logits"]
                    if tasktype == "regression":
                        return lg.squeeze(-1).item()
                    elif tasktype == "multiclass":
                        return torch.softmax(lg, dim=-1)[0, target_class_ic[sample_idx]].item()
                    else:
                        p1 = torch.sigmoid(lg.squeeze(-1))[0].item()
                        return p1  # binclass는 max_prob 방향 이미 처리했으므로 raw prob1 사용

            deletion_auc_per_sample = np.zeros(n_test)
            for n_i in range(n_test):
                order = np.argsort(-tabera_imp_ic[n_i])
                masked = X_ic[n_i].clone()
                preds = [_pred_at(masked, n_i)]
                for f_idx in order:
                    masked[f_idx] = X_baseline_mean[f_idx]
                    preds.append(_pred_at(masked, n_i))
                try:
                    auc = np.trapezoid(preds) / n_features_ic
                except AttributeError:
                    auc = np.trapz(preds) / n_features_ic
                deletion_auc_per_sample[n_i] = auc

            # ── Step 3. 분포 확인 (상관계수 계산 전에 먼저) ──────────
            print(f"\n  [3/4] 분포 확인 (상관계수 계산 전 포화 여부 사전 점검)")
            print(f"  {'─'*70}")
            for name, arr in [("completeness_error", completeness_error),
                               ("deletion_auc_per_sample", deletion_auc_per_sample)]:
                med, mn, sd = np.median(arr), arr.mean(), arr.std()
                p10, p90 = np.percentile(arr, [10, 90])
                print(f"    {name:<28} median={med:.4f}  mean={mn:.4f}  std={sd:.4f}  "
                      f"[p10={p10:.4f}, p90={p90:.4f}]")
            ce_saturated  = (np.median(completeness_error) > np.percentile(completeness_error, 95) * 0.9) or (completeness_error.std() < 1e-6)
            dau_saturated = (deletion_auc_per_sample.std() < 1e-6)
            if ce_saturated or dau_saturated:
                print(f"    🛑 변수 자체가 포화된 것으로 보임 — 아래 상관계수는 참고용으로만 볼 것")

            # ── Step 4. Spearman + Jackknife + Confound (한 파이프라인) ──
            print(f"\n  [4/4] Spearman + Jackknife + Confound 통제")
            print(f"  {'─'*70}")

            spearman_ic, p_ic = spearmanr(completeness_error, deletion_auc_per_sample)
            print(f"    Spearman ρ(completeness_error, deletion_auc) = {spearman_ic:.3f}  (p={p_ic:.4f})")
            print(f"    (가설: ρ<0 이면 completeness error가 클수록 deletion AUC가 낮아짐(좋아 보임)")
            print(f"     → mean baseline 우위가 착시였다는 가설을 지지)")

            # Jackknife (85-fold)
            jack_rs_ic, jack_ps_ic = [], []
            for j in range(n_test):
                jmask = np.ones(n_test, dtype=bool)
                jmask[j] = False
                r_j, p_j = spearmanr(completeness_error[jmask], deletion_auc_per_sample[jmask])
                if not np.isnan(r_j):
                    jack_rs_ic.append(r_j)
                    jack_ps_ic.append(p_j)
            jack_rs_ic = np.array(jack_rs_ic)
            jack_ps_ic = np.array(jack_ps_ic)
            frac_ns_ic = (jack_ps_ic >= 0.05).mean()
            sign_flip_ic = (np.sign(jack_rs_ic) != np.sign(spearman_ic)).any()

            print(f"\n    [Jackknife {n_test}-fold]")
            print(f"      ρ 범위: [{jack_rs_ic.min():.3f}, {jack_rs_ic.max():.3f}]")
            print(f"      p≥0.05인 fold 비율: {frac_ns_ic*100:.1f}%")
            is_jack_unstable = (frac_ns_ic > 0.20) or sign_flip_ic

            # Confound: raw_feature_distance, max_prob
            raw_dist_ic = (X_ic - X_base_ic).norm(dim=-1).cpu().numpy()
            confound_candidates_ic = {"raw_feature_distance": raw_dist_ic}
            if max_prob_np is not None:
                confound_candidates_ic["max_prob (예측 확신도)"] = max_prob_np

            def _partial_spearman_ic(x, y, z):
                rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
                def _resid(a, b):
                    slope, intercept = np.polyfit(b, a, 1)
                    return a - (slope * b + intercept)
                return np.corrcoef(_resid(rx, rz), _resid(ry, rz))[0, 1]

            is_confound_found = False
            for cname, cvar in confound_candidates_ic.items():
                r_c_x, _ = spearmanr(cvar, completeness_error)
                r_c_y, _ = spearmanr(cvar, deletion_auc_per_sample)
                is_trivial_ic = (abs(r_c_x) < 0.10) and (abs(r_c_y) < 0.10)
                r_partial_ic = _partial_spearman_ic(completeness_error, deletion_auc_per_sample, cvar)
                print(f"\n    [Confound: {cname}]")
                print(f"      corr(.,completeness_error)={r_c_x:.3f}  corr(.,deletion_auc)={r_c_y:.3f}"
                      f"{'  (trivial)' if is_trivial_ic else ''}")
                print(f"      ρ_partial = {r_partial_ic:.3f}  (원본 ρ={spearman_ic:.3f})")
                if not is_trivial_ic and abs(r_partial_ic) < abs(spearman_ic) * 0.5:
                    print(f"      🛑 non-trivial confound로 상관관계 절반 이하 붕괴")
                    is_confound_found = True

            is_unreliable_ic = ce_saturated or dau_saturated or is_jack_unstable or is_confound_found

            print(f"\n  {'─'*70}")
            print(f"  [최종 해석]")
            if is_unreliable_ic:
                print(f"  🛑 이 상관관계는 신뢰할 수 없음 (포화/jackknife 불안정/confound 중 하나 이상 해당)")
                print(f"    → \"mean baseline 우위가 착시\"라는 가설을 이 검증만으로는")
                print(f"      지지도 기각도 할 수 없음. 검증 불가능으로 재분류.")
            elif spearman_ic < -0.1 and p_ic < 0.05:
                print(f"  ✅ completeness error가 클수록 deletion AUC가 낮아지는(좋아 보이는)")
                print(f"    경향이 통계적으로 확인됨. mean baseline에서의 TabERA 우위가")
                print(f"    completeness 붕괴로 인한 착시일 가능성을 지지하는 증거.")
            elif spearman_ic > 0.1 and p_ic < 0.05:
                print(f"  △ 반대 방향 — completeness error가 클수록 deletion AUC도 나빠짐")
                print(f"    (착시 가설과 반대). mean baseline 우위가 착시가 아닐 수 있음.")
            else:
                print(f"  ⚠️  completeness error와 deletion AUC 기여도 사이 유의미한 관계 없음")
                print(f"    → 착시 가설을 지지하는 직접 증거는 확인 못함. 다른 설명")
                print(f"      (mean/medoid가 서로 다른 척도에서 측정하되, 우연히 IG의")
                print(f"      순위 자체가 mean에서 더 SHAP/Random과 달랐을 가능성 등)을")
                print(f"      고려해야 함.")

        # ── dataset_profile: 새 데이터셋 IG/deletion 신뢰도 빠른 분류 ──
        #
        # [배경] vehicle에서 겪은 시행착오(포화 미확인 → 잘못된 결론 →
        # 재검증 반복)를 다른 데이터셋에서 되풀이하지 않기 위한 통합 진단.
        # 예측 확신도, fallback 비율, mean/medoid completeness error,
        # deletion_auc의 샘플별 분산(TabERA 기준, SHAP/Random은 시간 관계상
        # --deletion_auc를 별도로 돌려 비교 권장)을 한 번에 출력하고
        # A/B/C 중 하나로 자동 분류한다.
        #
        # A: 척도(deletion_auc) 자체가 이 데이터셋에서 둔감 (overconfidence 등)
        #    → 이 데이터셋의 deletion/insertion AUC는 메인 결과에서 제외하거나
        #      caveat과 함께 보고. rank_correlation으로 대체 가능한지 확인.
        # B: mean baseline에서만 문제, medoid에서는 정상 → medoid를 메인으로 채택
        # C: 정상 (mean, medoid 모두 deletion_auc 분산 충분) → 그대로 사용
        elif args.ablation == "dataset_profile":
            model.eval()
            n_test = min(100, X_test.shape[0])
            X_dp   = X_test[:n_test].clone()
            n_features_dp = model.n_features

            print(f"\n  Dataset Profile — IG/Deletion 신뢰도 빠른 진단 (n={n_test})")
            print(f"  {'='*70}")

            # ── 1. 예측 확신도 ──────────────────────────────────
            with torch.no_grad():
                logits_dp = model(X_dp)["logits"]
                if tasktype == "regression":
                    max_prob_dp = None
                    target_class_dp = None
                elif tasktype == "multiclass":
                    probs_dp = torch.softmax(logits_dp, dim=-1)
                    target_class_dp = probs_dp.argmax(dim=-1).cpu().numpy()
                    max_prob_dp = probs_dp.max(dim=-1).values.cpu().numpy()
                else:
                    probs_dp = torch.sigmoid(logits_dp.squeeze(-1))
                    target_class_dp = None
                    max_prob_dp = torch.where(probs_dp >= 0.5, probs_dp, 1 - probs_dp).cpu().numpy()

            print(f"\n  [1. 예측 확신도]")
            if max_prob_dp is not None:
                print(f"    mean={max_prob_dp.mean():.4f}  median={np.median(max_prob_dp):.4f}  "
                      f"std={max_prob_dp.std():.4f}")
                overconfident = np.median(max_prob_dp) > 0.9
                if overconfident:
                    print(f"    ⚠️  median > 0.9 — overconfident, deletion/insertion 신호 둔감 위험")

            # ── 2. Fallback 비율 (input 시점) ──────────────────
            cached_sizes_dp = getattr(model.memory, "_cached_group_sizes", None)
            print(f"\n  [2. Fallback 비율]")
            if cached_sizes_dp is not None:
                with torch.no_grad():
                    q_dp = F.normalize(model.embedder(X_dp), dim=-1)
                    c_dp = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    ha_dp = (q_dp @ c_dp.T).argmax(dim=-1)
                    grp_sizes_dp = cached_sizes_dp[ha_dp]
                    fallback_rate_dp = (grp_sizes_dp < model.k).float().mean().item()
                    avg_group_size_dp = cached_sizes_dp[cached_sizes_dp > 0].float().mean().item()
                print(f"    k={model.k}, 평균 alive 그룹 크기={avg_group_size_dp:.1f}, "
                      f"fallback 비율={fallback_rate_dp*100:.1f}%")
            else:
                print(f"    _cached_group_sizes 없음 — skip")

            # ── 3. Mean / Medoid completeness error (50-step, 저비용) ──
            print(f"\n  [3. Completeness error: mean vs medoid baseline]")
            ig_target_fn_dp = make_logit_target_fn(
                tasktype, target_class=target_class_dp if tasktype == "multiclass" else None,
            )

            def _completeness_and_ig(X_baseline_this, n_steps=50):
                if X_baseline_this.dim() == 1:
                    X_base_this = X_baseline_this.unsqueeze(0).expand_as(X_dp)
                else:
                    X_base_this = X_baseline_this
                alphas = torch.linspace(0.0, 1.0, n_steps, device=X_dp.device)
                grads_accum = torch.zeros_like(X_dp)
                for alpha in alphas:
                    x_interp = (X_base_this + alpha * (X_dp - X_base_this)).clone().detach().requires_grad_(True)
                    target = ig_target_fn_dp(model(x_interp))
                    grad = torch.autograd.grad(target.sum(), x_interp, retain_graph=False)[0]
                    grads_accum = grads_accum + grad
                avg_grad = grads_accum / n_steps
                ig_signed = avg_grad * (X_dp - X_base_this)
                with torch.no_grad():
                    f_x = ig_target_fn_dp(model(X_dp))
                    f_b = ig_target_fn_dp(model(X_base_this))
                abs_err = (ig_signed.sum(dim=-1) - (f_x - f_b)).abs()
                rel_err = abs_err / ((f_x - f_b).abs() + 1e-8)
                return rel_err.cpu().numpy(), ig_signed.abs().detach().cpu().numpy()

            X_baseline_mean_dp = X_train.mean(dim=0)
            with torch.no_grad():
                q_med_dp = F.normalize(model.embedder(X_dp), dim=-1)
                c_med_dp = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                ha_med_dp = (q_med_dp @ c_med_dp.T).argmax(dim=-1)
                X_baseline_medoid_dp = model.prototype_layer.centroid_x[ha_med_dp].to(X_dp.device)

            rel_err_mean, ig_mean = _completeness_and_ig(X_baseline_mean_dp)
            rel_err_medoid, ig_medoid = _completeness_and_ig(X_baseline_medoid_dp)

            print(f"    mean   : median={np.median(rel_err_mean)*100:.2f}%  mean={rel_err_mean.mean()*100:.2f}%")
            print(f"    medoid : median={np.median(rel_err_medoid)*100:.2f}%  mean={rel_err_medoid.mean()*100:.2f}%")

            # ── 4. Deletion AUC 샘플별 분산 (TabERA IG, mean/medoid 각각) ──
            print(f"\n  [4. Deletion AUC 샘플별 분산 (TabERA IG 기준)]")

            def _pred_at_dp(x_batch, sample_idx):
                with torch.no_grad():
                    lg = model(x_batch.unsqueeze(0))["logits"]
                    if tasktype == "regression":
                        return lg.squeeze(-1).item()
                    elif tasktype == "multiclass":
                        return torch.softmax(lg, dim=-1)[0, target_class_dp[sample_idx]].item()
                    else:
                        return torch.sigmoid(lg.squeeze(-1))[0].item()

            def _deletion_auc_per_sample(ig_attr, X_baseline_this):
                baseline_is_per_sample = (X_baseline_this.dim() == 2)
                aucs = np.zeros(n_test)
                for n_i in range(n_test):
                    baseline_n = X_baseline_this[n_i] if baseline_is_per_sample else X_baseline_this
                    order = np.argsort(-ig_attr[n_i])
                    masked = X_dp[n_i].clone()
                    preds = [_pred_at_dp(masked, n_i)]
                    for f_idx in order:
                        masked[f_idx] = baseline_n[f_idx]
                        preds.append(_pred_at_dp(masked, n_i))
                    try:
                        auc = np.trapezoid(preds) / n_features_dp
                    except AttributeError:
                        auc = np.trapz(preds) / n_features_dp
                    aucs[n_i] = auc
                return aucs

            dau_mean   = _deletion_auc_per_sample(ig_mean, X_baseline_mean_dp)
            dau_medoid = _deletion_auc_per_sample(ig_medoid, X_baseline_medoid_dp)

            print(f"    mean   baseline: median={np.median(dau_mean):.4f}  std={dau_mean.std():.4f}")
            print(f"    medoid baseline: median={np.median(dau_medoid):.4f}  std={dau_medoid.std():.4f}")

            mean_dau_saturated   = dau_mean.std() < 0.01
            medoid_dau_saturated = dau_medoid.std() < 0.01

            # ── 5. 자동 분류: A / B / C ─────────────────────────
            print(f"\n  {'='*70}")
            print(f"  [자동 분류]")
            if mean_dau_saturated and medoid_dau_saturated:
                print(f"  → 분류 A: deletion_auc 척도 자체가 이 데이터셋에서 둔감함")
                print(f"    (mean std={dau_mean.std():.4f}, medoid std={dau_medoid.std():.4f} 모두 낮음)")
                print(f"    권장: deletion/insertion AUC를 메인 결과에서 제외하거나 caveat 명시.")
                print(f"    rank_correlation ablation으로 이 데이터셋의 IG faithfulness를")
                print(f"    대체 평가할 수 있는지 별도 확인 권장 (--ablation rank_correlation).")
                if overconfident:
                    print(f"    (예측 확신도가 높음 — overconfidence가 원인일 가능성)")
            elif mean_dau_saturated and not medoid_dau_saturated:
                print(f"  → 분류 B: mean baseline에서만 문제, medoid에서는 정상")
                print(f"    (mean std={dau_mean.std():.4f} 낮음, medoid std={dau_medoid.std():.4f} 정상)")
                print(f"    권장: medoid baseline을 이 데이터셋의 메인 결과로 채택.")
            else:
                print(f"  → 분류 C: 정상 (mean std={dau_mean.std():.4f}, "
                      f"medoid std={dau_medoid.std():.4f} 모두 충분한 분산)")
                print(f"    권장: mean/medoid 둘 다 사용 가능. medoid의 completeness")
                print(f"    개선 효과(위 3번)를 참고해 최종 baseline 선택.")

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
            model.eval()
            col_names  = dataset.col_names or [f"f{i}" for i in range(model.n_features)]
            n_features = model.n_features

            n_test = min(100, X_test.shape[0])
            X_da   = X_test[:n_test].clone()

            # ── Baseline: medoid(기본) vs mean(--mean_baseline, ablation) ──
            # medoid baseline은 4개 데이터셋 전부에서 IG의 completeness를
            # 일관되게 크게 개선함을 확인함 (median relative error 기준:
            # vehicle 146%→17.5%, ada 19.4%→1.5%, qsar 53.4%→1.75%,
            # wine 319%→4.1%). deletion/insertion AUC의 feature masking에도
            # 동일한 baseline을 일관되게 적용해, IG/SHAP/Random 세 방법이
            # 공정하게 같은 기준으로 비교되도록 함.
            if (not args.mean_baseline):
                with torch.no_grad():
                    q_da = F.normalize(model.embedder(X_da), dim=-1)
                    c_da = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    ha_da = (q_da @ c_da.T).argmax(dim=-1)
                    X_baseline = model.prototype_layer.centroid_x[ha_da].to(X_da.device)  # (N, F)
                print(f"\n  [Baseline] medoid (샘플별 소속 그룹의 대표 훈련 샘플)")
            else:
                X_baseline = X_train.mean(dim=0)              # (F,) 마스킹 시 사용
                print(f"\n  [Baseline] mean (전체 훈련 데이터 평균)")

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
            print(f"  [1/3] TabERA IG attribution 계산 중 (n_steps={args.ig_nsteps})...")
            ig_target_fn = make_logit_target_fn(
                tasktype,
                target_class=target_class if tasktype == "multiclass" else None,
            )
            tabera_imp = compute_integrated_gradients(
                model, X_da, X_baseline,
                target_fn=ig_target_fn,
                n_steps=args.ig_nsteps,
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

                # ── SHAP background: IG baseline 선택과 정합성 맞춤 ──
                # IG가 medoid를 baseline으로 쓰면, SHAP의 background도
                # 같은 기준(centroid medoid들)으로 맞춰야 공정한 비교가 됨.
                # 그렇지 않으면 "IG는 medoid 대비, SHAP은 전체 평균 근처
                # random sample 대비"로 서로 다른 기준점에서 평가되는
                # 문제가 생김 (IG의 attribution 자체가 (x-baseline)을
                # 포함하는 구조이므로 이 불일치가 결과를 왜곡할 수 있음).
                if (not args.mean_baseline):
                    with torch.no_grad():
                        all_medoids = model.prototype_layer.centroid_x  # (P, F)
                        valid_medoid_mask = all_medoids.abs().sum(dim=-1) > 0
                        bg = all_medoids[valid_medoid_mask].cpu().numpy()
                    if len(bg) < 2:
                        bg_n   = min(50, len(X_train))
                        bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                        bg     = X_train[bg_idx].cpu().numpy()
                    print(f"  [SHAP background] medoid 기준 ({len(bg)}개 centroid 대표 샘플)")
                else:
                    bg_n   = min(50, len(X_train))
                    bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                    bg     = X_train[bg_idx].cpu().numpy()
                    print(f"  [SHAP background] mean 기준 (random {bg_n}개 훈련 샘플)")
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
                # X_baseline이 (F,) 1D(mean)면 전체 샘플 공통,
                # (N,F) 2D(medoid)면 샘플별로 다른 baseline 사용
                baseline_is_per_sample = (X_baseline.dim() == 2)

                aucs = []
                for n in range(n_test):
                    baseline_n = X_baseline[n] if baseline_is_per_sample else X_baseline

                    # 순위 (높은 importance 먼저)
                    order = np.argsort(-attribution[n])  # (F,)

                    # 곡선: 마스킹 0개 → F개
                    masked = X_da[n].clone()
                    preds  = [pred_orig[n]]
                    for f_idx in order:
                        masked[f_idx] = baseline_n[f_idx]
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

                # ── Paired Wilcoxon signed-rank test ──────────────
                # mean/std만으로는 std가 0.2~0.35로 큰 상황에서 TabERA-SHAP
                # 차이(대개 0.01~0.05)가 통계적으로 유의한지 판단할 수 없음.
                # 같은 샘플에 대한 TabERA·SHAP 값을 짝지어 비교하는
                # paired test로 이 차이가 노이즈인지 확인.
                from scipy.stats import wilcoxon
                try:
                    diff = tabera_aucs - shap_aucs
                    if np.allclose(diff, 0):
                        print(f"  [Wilcoxon] TabERA와 SHAP 값이 완전히 동일 — 검정 불가")
                    else:
                        stat, p_wilcoxon = wilcoxon(tabera_aucs, shap_aucs)
                        sig = "유의함 (p<0.05)" if p_wilcoxon < 0.05 else "유의하지 않음 (노이즈일 가능성)"
                        print(f"  [Wilcoxon signed-rank] TabERA vs SHAP: p={p_wilcoxon:.4f}  → {sig}")
                except Exception as e:
                    print(f"  [Wilcoxon 실패: {e}]")
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

            # ── Baseline: medoid(기본) vs mean(--mean_baseline, ablation) ──
            # deletion_auc와 동일한 근거로 medoid를 기본값으로 사용.
            if (not args.mean_baseline):
                with torch.no_grad():
                    q_ia = F.normalize(model.embedder(X_ia), dim=-1)
                    c_ia = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
                    ha_ia = (q_ia @ c_ia.T).argmax(dim=-1)
                    X_baseline = model.prototype_layer.centroid_x[ha_ia].to(X_ia.device)  # (N, F)
                print(f"\n  [Baseline] medoid (샘플별 소속 그룹의 대표 훈련 샘플)")
            else:
                X_baseline = X_train.mean(dim=0)              # (F,) 복원 시작점
                print(f"\n  [Baseline] mean (전체 훈련 데이터 평균)")

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
            print(f"  [1/3] TabERA IG attribution 계산 중 (n_steps={args.ig_nsteps})...")
            ig_target_fn = make_logit_target_fn(
                tasktype,
                target_class=target_class if tasktype == "multiclass" else None,
            )
            tabera_imp = compute_integrated_gradients(
                model, X_ia, X_baseline,
                target_fn=ig_target_fn,
                n_steps=args.ig_nsteps,
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

                # ── SHAP background: IG baseline 선택과 정합성 맞춤 ──
                # IG가 medoid를 baseline으로 쓰면, SHAP의 background도
                # 같은 기준(centroid medoid들)으로 맞춰야 공정한 비교가 됨.
                # 그렇지 않으면 "IG는 medoid 대비, SHAP은 전체 평균 근처
                # random sample 대비"로 서로 다른 기준점에서 평가되는
                # 문제가 생김 (IG의 attribution 자체가 (x-baseline)을
                # 포함하는 구조이므로 이 불일치가 결과를 왜곡할 수 있음).
                if (not args.mean_baseline):
                    with torch.no_grad():
                        all_medoids = model.prototype_layer.centroid_x  # (P, F)
                        valid_medoid_mask = all_medoids.abs().sum(dim=-1) > 0
                        bg = all_medoids[valid_medoid_mask].cpu().numpy()
                    if len(bg) < 2:
                        bg_n   = min(50, len(X_train))
                        bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                        bg     = X_train[bg_idx].cpu().numpy()
                    print(f"  [SHAP background] medoid 기준 ({len(bg)}개 centroid 대표 샘플)")
                else:
                    bg_n   = min(50, len(X_train))
                    bg_idx = np.random.choice(len(X_train), size=bg_n, replace=False)
                    bg     = X_train[bg_idx].cpu().numpy()
                    print(f"  [SHAP background] mean 기준 (random {bg_n}개 훈련 샘플)")
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
                baseline_is_per_sample = (X_baseline.dim() == 2)

                aucs = []
                for n in range(n_test):
                    baseline_n = X_baseline[n] if baseline_is_per_sample else X_baseline

                    # 순위 (높은 importance 먼저)
                    order = np.argsort(-attribution[n])  # (F,)

                    # baseline 상태에서 시작 → 중요 feature부터 1개씩 복원
                    inserted = baseline_n.clone()
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

                from scipy.stats import wilcoxon
                try:
                    diff = tabera_aucs - shap_aucs
                    if np.allclose(diff, 0):
                        print(f"  [Wilcoxon] TabERA와 SHAP 값이 완전히 동일 — 검정 불가")
                    else:
                        stat, p_wilcoxon = wilcoxon(tabera_aucs, shap_aucs)
                        sig = "유의함 (p<0.05)" if p_wilcoxon < 0.05 else "유의하지 않음 (노이즈일 가능성)"
                        print(f"  [Wilcoxon signed-rank] TabERA vs SHAP: p={p_wilcoxon:.4f}  → {sig}")
                except Exception as e:
                    print(f"  [Wilcoxon 실패: {e}]")
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

        # ── random_neighbor / neighbor_noise: 성능 비교 ─────────────
        # full model 대비 성능 하락을 측정.
        # random_neighbor: 틀린(그러나 real) 이웃로 교체 시 하락 → retrieval 정확도의 가치
        # neighbor_noise : 이웃 정보 자체를 노이즈로 교체 시 하락 → neighbor evidence 자체의 가치
        # 두 값을 같이 보면: neighbor_noise 하락폭이 random_neighbor보다 커야
        # "틀린 이웃이라도 real data가 낫다"는 게 일관되게 확인됨
        else:
            with torch.no_grad():
                abl_logits_list, abl_labels_list = [], []
                full_evw_list, abl_evw_list = [], []
                batch_size = 256
                n_test     = X_test.shape[0]

                for start in range(0, n_test, batch_size):
                    X_batch = X_test[start:start + batch_size]
                    out_batch      = model(X_batch, ablation_mode=args.ablation)
                    out_batch_full = model(X_batch, ablation_mode="none")
                    abl_logits_list.append(out_batch["logits"].cpu())
                    if out_batch.get("evidence_w") is not None:
                        abl_evw_list.append(out_batch["evidence_w"].cpu())
                    if out_batch_full.get("evidence_w") is not None:
                        full_evw_list.append(out_batch_full["evidence_w"].cpu())

                abl_logits = torch.cat(abl_logits_list, dim=0)
                abl_evw    = torch.cat(abl_evw_list, dim=0) if abl_evw_list else None
                full_evw   = torch.cat(full_evw_list, dim=0) if full_evw_list else None

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

            # ── evidence_w 엔트로피 비교 ──────────────────────────
            # 목적: neighbor_noise/random_neighbor에서 성능 하락이 왜
            # 다른지, evidence_w(attention weight)가 얼마나 uniform하게
            # 퍼졌는지로 설명되는지 확인. nk가 진짜 임베딩이면 유사도가
            # 뾰족(집중)할 수 있고, nk가 순수 노이즈면 고차원에서 거의
            # 직교라 유사도가 다 비슷해져 evidence_w가 uniform에 가까워질
            # 것이라는 가설을 직접 검증.
            evw_stats = {}
            if full_evw is not None and abl_evw is not None:
                k_dim = full_evw.shape[-1]

                def _norm_entropy(w):
                    ent = -(w * (w + 1e-8).log()).sum(dim=-1)   # (N,)
                    return (ent / torch.log(torch.tensor(float(k_dim)))).numpy()

                full_ent = _norm_entropy(full_evw)
                abl_ent  = _norm_entropy(abl_evw)
                full_max = full_evw.max(dim=-1).values.numpy()
                abl_max  = abl_evw.max(dim=-1).values.numpy()

                print(f"\n  evidence_w 엔트로피 (0=한 이웃에 완전 집중, 1=완전 uniform, k={k_dim})")
                print(f"  {'-'*58}")
                print(f"  {'':<20} {'Full Model':>12}  {'Ablation':>12}")
                print(f"  {'정규화 엔트로피 평균':<18} {full_ent.mean():>12.4f}  {abl_ent.mean():>12.4f}")
                print(f"  {'최대 가중치 평균':<18} {full_max.mean():>12.4f}  {abl_max.mean():>12.4f}")

                evw_stats = {
                    "full_entropy_mean": float(full_ent.mean()),
                    "abl_entropy_mean":  float(abl_ent.mean()),
                    "full_max_w_mean":   float(full_max.mean()),
                    "abl_max_w_mean":    float(abl_max.mean()),
                }

            print(f"\n  해석:")
            if args.ablation == "random_neighbor":
                print(f"  → 성능 하락 = '검색이 틀린 이웃을 찾았을 때'의 대가")
                print(f"    (이웃 정보 자체는 여전히 real data — retrieval 정확도의 가치)")
            elif args.ablation == "neighbor_noise":
                print(f"  → 성능 하락 = '이웃 정보가 조금이라도 있는가'의 대가")
                print(f"    (real이든 아니든 neighbor evidence 자체의 존재 가치)")
                print(f"  참고: random_neighbor보다 여기서 하락폭이 훨씬 커야 정상")
                print(f"    (같은 배치 크기지만 '틀린 진짜 이웃' < '이웃 자체 없음'이 더")
                print(f"    나쁜 상황이어야 두 ablation이 일관된 이야기를 함)")
                if evw_stats and evw_stats["abl_entropy_mean"] > evw_stats["full_entropy_mean"] + 0.1:
                    print(f"  → evidence_w가 실제로 uniform 쪽으로 이동함 "
                          f"(엔트로피 {evw_stats['full_entropy_mean']:.3f} → "
                          f"{evw_stats['abl_entropy_mean']:.3f}). nk가 노이즈가 되면서")
                    print(f"    attention이 '누구를 볼지 못 정하는' 상태가 됐다는 뜻 —")
                    print(f"    성능이 덜 떨어진 건 uniform 평균이 이 데이터셋에서")
                    print(f"    우연히 나쁘지 않은 예측이기 때문일 수 있음 (value_diagnosis로")
                    print(f"    T() 항의 크기도 같이 봐야 완전한 그림이 됨).")

            # ablation 결과 저장
            abl_save = {
                "ablation_mode":  args.ablation,
                "full_metrics":   test_metrics,
                "abl_metrics":    abl_metrics,
                "evidence_w_stats": evw_stats,
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

    # ── vectorized_fallback 정확성 검증 (학습된 모델 그대로, 재학습 없음) ──
    # 목적: retrieve()의 cross-group fallback 경로를 벡터화(bmm+gather+
    # masked topk)로 바꾼 게, 지금 막 학습이 끝난 이 모델의 실제 가중치와
    # 실제 테스트 데이터에서도 출력을 하나도 안 바꾸는지 확인.
    # (optimize.py 전체 재학습 비교와는 다른 검증임 — 여기는 재학습 없이
    # 같은 가중치로 vectorized_fallback만 스위칭해서 순수 forward만 비교)
    if hasattr(model.memory, "_vectorized_fallback"):
        print(f"\n{'='*52}")
        print(f"  vectorized_fallback 정확성 검증")
        print(f"{'='*52}")

        model.eval()
        with torch.no_grad():
            model.memory._vectorized_fallback = False
            out_false = model(X_test, return_explanations=True)

            model.memory._vectorized_fallback = True
            out_true = model(X_test, return_explanations=True)

        same_logits = torch.equal(out_false["logits"], out_true["logits"])
        same_topk   = torch.equal(out_false["topk_idx"], out_true["topk_idx"])
        same_evw    = torch.equal(out_false["evidence_w"], out_true["evidence_w"])

        print(f"  logits 완전 동일:        {same_logits}")
        print(f"  topk_idx(이웃) 완전 동일: {same_topk}")
        print(f"  evidence_w(설명) 완전 동일: {same_evw}")

        if not same_logits:
            diff = (out_false["logits"] - out_true["logits"]).abs()
            print(f"    logits 최대 절대오차: {diff.max().item():.2e}")
        if not same_topk:
            n_diff = (out_false["topk_idx"] != out_true["topk_idx"]).any(dim=-1).sum().item()
            print(f"    topk_idx가 다른 샘플 수: {n_diff} / {X_test.shape[0]}")
        if not same_evw:
            diff_evw = (out_false["evidence_w"] - out_true["evidence_w"]).abs()
            print(f"    evidence_w 최대 절대오차: {diff_evw.max().item():.2e}")

        # 학습 종료 후 검증용으로만 forward를 두 번 더 돌렸으므로, 이후
        # (--explain 등) 로직이 기존 설정(False)으로 계속 진행되도록 복원.
        model.memory._vectorized_fallback = False
    else:
        print("  (참고: 이 libs/tabera.py는 vectorized_fallback을 지원하지 않는 "
              "버전입니다 — 검증을 건너뜁니다.)")

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