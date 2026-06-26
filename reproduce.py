## Main file for reproducing the best TabERA configuration.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import sys, os, argparse
from xml.parsers.expat import model

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
                                 "rank_correlation", "dual_space_faithfulness"],
                        help=(
                            "ablation 모드 선택 (학습된 모델에 inference 단계에서 적용):\n"
                            "  none                  : full model 기준 (기본값)\n"
                            "  random_neighbor       : neighbor 임베딩 랜덤 교체\n"
                            "  rank_correlation      : IG feature 순위 vs 실제 prediction\n"
                            "                         영향력 순위 Spearman 상관계수\n"
                            "                         (TabERA vs SHAP vs Random 3자 비교)\n"
                            "  dual_space_faithfulness : centroid_x 대표성 + 그룹 분리도 검증"
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

    fname = os.path.join(log_dir, f"data={openml_id}..model=tabera.pkl")
    if not os.path.exists(fname):
        raise FileNotFoundError(
            f"최적화 로그 없음: {fname}\n"
            f"먼저 optimize.py --openml_id {openml_id} --seed {args.seed} 를 실행하세요."
        )

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
        memory_size=min(int(len(y_train) * 2), 10_000),
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
        # 1. TabERA Input × Gradient → feature별 중요도 순위
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

            # ── Step 2. TabERA IG 순위 (Input × Gradient) ──
            #
            # agg_emb(=retrieval 경로의 aggregation)을
            # raw input X에 대해 직접 미분(Input × Gradient, IG의 1-step 근사).
            # embedder의 비선형성을 grad가 그대로 통과하므로,
            # "이 feature를 바꾸면 retrieval 결과가 얼마나 변하는가"를
            # delta(perturbation 기반)와 같은 좌표계(원본 feature space)에서 측정.
            # -> 모델 구조/학습 변경 없음, eval 모드에서 1차 backward만 사용.
            print(f"  [2/4] TabERA IG 순위 계산 중 (Input x Gradient)...")

            X_rc_grad = X_rc.clone().detach().requires_grad_(True)
            out_rc    = model(X_rc_grad)

            target = out_rc["agg_emb"].sum()
            grad_X = torch.autograd.grad(target, X_rc_grad, retain_graph=False)[0]  # (N, F)

            X_baseline = X_train.mean(dim=0)               # (F,) -- delta 계산과 동일 baseline
            tabera_imp = (grad_X * (X_rc_grad - X_baseline)).abs().detach().cpu().numpy()  # (N, F)

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
                / f"data={openml_id}..seed{args.seed}_dual_space_faithfulness.pkl"
            )
            with open(dsf_path, "wb") as f:
                pickle.dump(dsf_save, f)
            print(f"\n  저장: {dsf_path}")

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
            abl_path = Path(log_dir) / f"data={openml_id}..seed{args.seed}_ablation_{args.ablation}.pkl"
            with open(abl_path, "wb") as f:
                pickle.dump(abl_save, f)
            print(f"\n  저장: {abl_path}")



    # ── 결과 저장 ──────────────────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}..seed{args.seed}_meta.pkl"

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
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print(f"\n  저장: {pred_path}")

    # ── model state 저장 (visualize_embeddings.py --from_state 용) ──
    state_path = save_dir / f"data={openml_id}..seed{args.seed}_model_state.pt"
    torch.save({
        "state_dict":   model.state_dict(),
        "model_kwargs": model_kwargs,
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