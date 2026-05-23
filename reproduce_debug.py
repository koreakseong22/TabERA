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
# 설명 출력 (§3.3 feature_match 포함)
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

    # centroid 원본 feature 값 출력 (Dual-Space의 핵심)
    cf = proto.get("centroid_features", {})
    if cf:
        feat_str = ",  ".join(
            f"{k}={v:.3f}" for k, v in sorted(cf.items(), key=lambda x: -abs(x[1]))[:6]
        )
        print(f"     특성: {feat_str}")

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

    # ③ Feature 기여도 (Cross-Attention)
    fm = e.get("feature_match")
    if fm and "top_features" in fm:
        print(f"\n  ③ Feature 기여도 (Cross-Attention)")
        for feat in fm["top_features"]:
            bar = "█" * int(feat["importance"] * 30)
            print(f"     {feat['feature']:25s} {feat['importance']*100:5.1f}%  {bar}")

    # Gated Fusion 진단
    gate_mean = e.get("gate_mean")
    if gate_mean is not None:
        label = "feature path 우세" if gate_mean > 0.5 else "neighbor path 우세"
        print(f"\n  gate_mean: {gate_mean:.2f}  ({label})")

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

    # ── Routing Logit 진단 ───────────────────────────────
    # confidence가 1/P에 수렴하는 원인 파악용
    # (1) logit 분포, (2) entropy loss 영향, (3) centroid 분리도 확인
    print(f"\n{'='*52}")
    print(f"  [진단] Routing Logit 분포 분석")
    print(f"{'='*52}")

    model.eval()
    with torch.no_grad():
        # 전체 테스트셋 임베딩
        query_emb_all = model.embedder(X_test)                          # (N_test, D)
        q_norm = F.normalize(query_emb_all, dim=-1)                     # (N_test, D)
        c_norm = F.normalize(
            model.prototype_layer.centroid_emb.detach(), dim=-1)        # (P, D)
        logits_all = q_norm @ c_norm.T                                  # (N_test, P)
        soft_all   = torch.softmax(logits_all, dim=-1)                  # (N_test, P)

        # ── (1) Logit 값 분포 ──────────────────────────
        print(f"\n  [1] Logit 값 분포 (cosine similarity)")
        print(f"      전체 min  : {logits_all.min().item():.4f}")
        print(f"      전체 max  : {logits_all.max().item():.4f}")
        print(f"      전체 mean : {logits_all.mean().item():.4f}")
        print(f"      전체 std  : {logits_all.std().item():.4f}")

        # 샘플별 top1 - top2 gap (얼마나 1등이 2등을 앞서는지)
        top2_vals, _ = logits_all.topk(2, dim=-1)                       # (N_test, 2)
        gap = top2_vals[:, 0] - top2_vals[:, 1]                         # (N_test,)
        print(f"\n  [2] Top1 - Top2 logit gap (샘플별)")
        print(f"      mean : {gap.mean().item():.4f}")
        print(f"      min  : {gap.min().item():.4f}")
        print(f"      max  : {gap.max().item():.4f}")
        print(f"      → gap이 작을수록 confidence가 flat해짐")

        # ── (2) Softmax 분포 균등도 ──────────────────
        # 샘플별 max confidence
        max_conf, _ = soft_all.max(dim=-1)                              # (N_test,)
        uniform_val = 1.0 / model.prototype_layer.P
        print(f"\n  [3] Softmax confidence 분포")
        print(f"      1/P (균등 기준) : {uniform_val:.4f}")
        print(f"      max_conf mean   : {max_conf.mean().item():.4f}")
        print(f"      max_conf min    : {max_conf.min().item():.4f}")
        print(f"      max_conf max    : {max_conf.max().item():.4f}")
        print(f"      → mean ≈ 1/P 이면 flat, 크게 차이나면 정상")

        # ── (3) Centroid 간 분리도 ───────────────────
        sim_matrix = c_norm @ c_norm.T                                  # (P, P)
        mask = ~torch.eye(
            model.prototype_layer.P, dtype=torch.bool,
            device=sim_matrix.device)
        off_diag = sim_matrix[mask]
        print(f"\n  [4] Centroid 간 cosine similarity")
        print(f"      off-diagonal mean : {off_diag.mean().item():.4f}")
        print(f"      off-diagonal max  : {off_diag.max().item():.4f}")
        print(f"      off-diagonal min  : {off_diag.min().item():.4f}")
        print(f"      avg_inter_dist    : {(1 - off_diag).mean().item():.4f}")
        print(f"      → off-diagonal이 높을수록 centroid가 뭉쳐있음")

        # ── (4) 샘플 3개 상세 logit ──────────────────
        print(f"\n  [5] 테스트 샘플 #0~2 상위 5개 logit")
        for i in range(min(3, len(X_test))):
            top5_vals, top5_idx = logits_all[i].topk(5)
            pairs = ", ".join(
                f"C{idx.item()}={val.item():.4f}"
                for val, idx in zip(top5_vals, top5_idx)
            )
            print(f"      Sample #{i}: {pairs}")

        # ── (5) Temperature별 confidence 미리보기 ────
        print(f"\n  [6] Temperature별 max_confidence 미리보기 (샘플 #0)")
        for tau in [1.0, 0.5, 0.2, 0.1, 0.07, 0.05]:
            s = torch.softmax(logits_all[0] / tau, dim=-1)
            print(f"      tau={tau:.2f} → max={s.max().item():.4f}  "
                  f"top3: {s.topk(3).values.tolist()}")

    print(f"\n{'='*52}")

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