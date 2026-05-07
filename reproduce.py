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
from libs.diagnostics  import run_phase1_diagnostics
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

    # ② 프로토타입 그룹 (가설 ①: centroid_features — 역정규화 없이 원본값 표시)
    proto = e["prototype"]
    print(f"\n  ① 프로토타입 그룹")
    print(f"     → \"{proto['assigned_group']}\"  (confidence={proto['group_confidence']:.1%})")
    if proto["runners_up"]:
        ru = ", ".join(f"\"{l}\"({s:.1%})" for l, s in proto["runners_up"])
        print(f"     Runner-up: {ru}")

    # centroid 원본 feature 값 출력 (이중 공간의 핵심)
    cf = proto.get("centroid_features", {})
    if cf:
        feat_str = ",  ".join(
            f"{k}={v:.3f}" for k, v in sorted(cf.items(), key=lambda x: -abs(x[1]))[:6]
        )
        print(f"     특성: {feat_str}")

    # ③ OT 증거 선택
    ev = e["evidence"]
    print(f"\n  ② OT 증거 선택")
    print(f"     {ev['summary']}")
    for t in ev["top_evidence"]:
        print(f"     #{t['rank']} Neighbour {t['neighbour_idx']}: {t['weight_pct']}")

    # ④ §3.3 Feature Cross-Attention (핵심 신규)
    fm = e.get("feature_match")
    if fm and "per_neighbour" in fm:
        print(f"\n  ③ Feature 기여도 (§3.3 Cross-Attention)")

        # 전체 결정의 feature 기여도
        print(f"     [전체 결정 기여 feature]")
        for fname, score in fm["overall_features"]:
            bar = "█" * int(score * 30)
            print(f"       {fname:25s} {score*100:5.1f}%  {bar}")

        # 상위 이웃 2개의 feature 기여도
        print(f"\n     [이웃별 유사 이유]")
        for nb in fm["per_neighbour"][:3]:
            print(f"       Neighbour #{nb['neighbour_idx']} "
                  f"(weight={nb['evidence_weight']*100:.1f}%)")
            for fname, score in nb["top_features"]:
                bar = "█" * int(score * 20)
                print(f"         {fname:23s} {score*100:5.1f}%  {bar}")

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
    parser.add_argument("--loss_cohesion", type=float, default=1e-3,
                        help="Phase 2 soft feature cohesion loss 가중치 (0.0=비활성화)")
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

    fname = os.path.join(log_dir, f"data={openml_id}..seed{args.seed}..model=tabera.pkl")
    # 구파일명 패턴 fallback (data_10__seed1__model_tabera.pkl)
    fname_alt = os.path.join(log_dir, f"data_{openml_id}__seed{args.seed}__model_tabera.pkl")
    if not os.path.exists(fname) and os.path.exists(fname_alt):
        fname = fname_alt
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

    # ── Phase 2: loss_cohesion 주입 ──────────────────────────
    # HPO params에 없는 경우(Phase 1 시절 결과) --loss_cohesion 인자로 주입
    best_params["loss_cohesion"] = args.loss_cohesion
    cohesion_status = f"{args.loss_cohesion:.0e}" if args.loss_cohesion > 0 else "OFF"
    print(f"  n_prototypes (from optimize.py): {best_params['n_prototypes']}")
    print(f"  loss_cohesion: {cohesion_status}")
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

    # ── Phase 1 진단 ───────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Phase 1 Diagnostics")
    print(f"{'='*52}")

    model._wrapper = wrapper  # ema_history 접근용

    diag = run_phase1_diagnostics(
        model          = model,
        X_test         = X_test,
        y_test         = y_test,
        X_train        = X_train,
        tasktype       = tasktype,
        top_m          = 3,
        n_faithfulness = min(200, len(y_test)),
        n_entropy      = min(500, len(y_test)),
        device         = str(device),
    )

    # 진단 결과를 meta에 함께 저장
    meta["phase1_diagnostics"] = diag
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"  진단 결과 저장 완료: {meta_path}")

    # ── §3.3 Feature 기여도 설명 출력 ─────────────────────
    if args.explain:
        print(f"\n{'='*52}")
        print(f"  Feature 기여도 설명 (--explain)")
        print(f"{'='*52}")

        # Centroid cosine similarity matrix 출력
        print(f"\n  [Centroid Similarity Matrix]")
        sim_mat = model.prototype_layer.cosine_similarity_matrix().numpy()
        cl = model.prototype_layer.centroid_labels
        P  = model.prototype_layer.P
        if cl is not None and not cl.isnan().any():
            # 레이블 순서대로 centroid 정렬
            order = cl.cpu().argsort().numpy()
            sim_sorted = sim_mat[order][:, order]
            cl_sorted  = cl.cpu().numpy()[order]
            print(f"  centroid를 평균 레이블 순으로 정렬 ({P}개):")
            print(f"  {'lbl':>5}" + "".join(f"{cl_sorted[j]:5.1f}" for j in range(min(P,10))))
            for i in range(min(P,10)):
                row = "".join(f"{sim_sorted[i,j]:5.2f}" for j in range(min(P,10)))
                print(f"  {cl_sorted[i]:5.1f} {row}")
            # 순서 정합성 지표: 인접 centroid 유사도 평균
            adj_sim = np.mean([sim_sorted[i,i+1] for i in range(min(P,10)-1)])
            far_sim = np.mean([sim_sorted[i,min(i+3,min(P,10)-1)]
                               for i in range(min(P,10)-3)])
            print(f"\n  인접 centroid 유사도 평균:  {adj_sim:.3f}")
            print(f"  원거리 centroid 유사도 평균: {far_sim:.3f}")
            if adj_sim > far_sim:
                print(f"  → 순서 정합: 인접>원거리 (Rank-Consistency 반영됨)")
            else:
                print(f"  → 순서 미반영: Rank-Consistency Loss 효과 확인 필요")
        else:
            print(f"  (centroid_labels 미초기화 — epochs를 늘려주세요)")

        model.eval()
        n_show = min(args.n_explain, len(y_test))
        X_show = X_test[:n_show]

        with torch.no_grad():
            out = model(X_show, return_explanations=True)

        explanations = out.get("explanations", [])
        topk_idx     = out.get("topk_idx", None)

        # FeatureStore에서 이웃 feature 값 조회하여 설명에 추가
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
    