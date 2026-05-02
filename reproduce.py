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

import joblib, json, pickle
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize_scalar

from libs.data         import TabularDataset
from libs.search_space import params_to_model_kwargs
from libs.supervised   import TabERAWrapper
from libs.tabera       import TabERA
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

    proto = e["prototype"]
    print(f"\n  ① 프로토타입 그룹")
    print(f"     → \"{proto['assigned_group']}\"  (confidence={proto['group_confidence']:.1%})")
    if proto["runners_up"]:
        ru = ", ".join(f"\"{l}\"({s:.1%})" for l, s in proto["runners_up"])
        print(f"     Runner-up: {ru}")

    cf = proto.get("centroid_features", {})
    if cf:
        feat_str = ",  ".join(
            f"{k}={v:.3f}" for k, v in sorted(cf.items(), key=lambda x: -abs(x[1]))[:6]
        )
        print(f"     특성: {feat_str}")

    ev = e["evidence"]
    print(f"\n  ② OT 증거 선택")
    print(f"     {ev['summary']}")
    for t in ev["top_evidence"]:
        print(f"     #{t['rank']} Neighbour {t['neighbour_idx']}: {t['weight_pct']}")

    fm = e.get("feature_match")
    if fm and "per_neighbour" in fm:
        print(f"\n  ③ Feature 기여도 (§3.3 Cross-Attention)")

        print(f"     [전체 결정 기여 feature]")
        for fname, score in fm["overall_features"]:
            bar = "█" * int(score * 30)
            print(f"       {fname:25s} {score*100:5.1f}%  {bar}")

        print(f"\n     [이웃별 유사 이유]")
        for nb in fm["per_neighbour"][:3]:
            print(f"       Neighbour #{nb['neighbour_idx']} "
                  f"(weight={nb['evidence_weight']*100:.1f}%)")
            for fname, score in nb["top_features"]:
                bar = "█" * int(score * 20)
                print(f"         {fname:23s} {score*100:5.1f}%  {bar}")

    print(f"{'━'*52}")


# ─────────────────────────────────────────────────────────────
# Per-sample 진단
# ─────────────────────────────────────────────────────────────

def save_per_sample_diagnostics(
    model: TabERA,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    tasktype: str,
    save_dir: Path,
    openml_id: str,
    seed: int,
) -> pd.DataFrame:
    """
    per-sample 진단 정보를 추출하여 CSV로 저장하고 요약을 출력합니다.

    저장 항목
    ─────────
    y_true            : 정답 레이블
    y_pred            : 예측 레이블
    max_prob          : 예측 클래스의 확률 (모델의 확신도)
    correct           : 정답 여부 (True/False)
    sample_loss       : 샘플별 cross-entropy loss (-log p(y_true))
    assigned_centroid : 배정된 centroid 인덱스 (hard_group)
    """
    model.eval()

    batch_size    = 512
    all_logits    = []
    all_centroids = []

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):
            xb  = X_test[start:start + batch_size]
            out = model(xb)
            all_logits.append(out["logits"].cpu())
            all_centroids.append(out["hard_group"].cpu())

    logits_all    = torch.cat(all_logits,    dim=0)
    centroids_all = torch.cat(all_centroids, dim=0)

    if tasktype == "binclass":
        probs_pos  = torch.sigmoid(logits_all.squeeze(-1))
        probs_all  = torch.stack([1 - probs_pos, probs_pos], dim=-1)
    else:
        probs_all  = F.softmax(logits_all, dim=-1)

    y_true_np   = y_test.cpu().numpy().astype(int)
    y_pred_np   = probs_all.argmax(dim=-1).numpy().astype(int)
    max_prob_np = probs_all.max(dim=-1).values.numpy()
    correct_np  = (y_pred_np == y_true_np)
    centroid_np = centroids_all.numpy().astype(int)

    true_probs  = probs_all[torch.arange(len(y_true_np)), y_true_np].numpy()
    sample_loss = -np.log(true_probs + 1e-8)

    df = pd.DataFrame({
        "y_true":            y_true_np,
        "y_pred":            y_pred_np,
        "max_prob":          max_prob_np.round(4),
        "correct":           correct_np,
        "sample_loss":       sample_loss.round(4),
        "assigned_centroid": centroid_np,
    })

    diag_path = save_dir / f"data={openml_id}..seed{seed}_per_sample.csv"
    df.to_csv(diag_path, index=False)
    print(f"\n  저장: {diag_path}")

    n_total   = len(df)
    n_correct = correct_np.sum()
    n_wrong   = n_total - n_correct

    print(f"\n  {'─'*52}")
    print(f"  [Per-sample 진단 요약]  seed={seed}  N={n_total}")
    print(f"  {'─'*52}")
    print(f"  전체 accuracy          : {n_correct/n_total*100:.1f}%  ({n_correct}/{n_total})")
    print(f"  전체 평균 max_prob     : {max_prob_np.mean():.4f}")

    if n_correct > 0:
        print(f"  correct 평균 max_prob  : {df[df.correct]['max_prob'].mean():.4f}")
    if n_wrong > 0:
        mean_conf_wrong = df[~df.correct]["max_prob"].mean()
        print(f"  wrong   평균 max_prob  : {mean_conf_wrong:.4f}")

        wrong_high_90 = ((~df.correct) & (df.max_prob > 0.9)).sum()
        wrong_high_70 = ((~df.correct) & (df.max_prob > 0.7)).sum()
        print(f"\n  wrong인데 max_prob > 0.9 : {wrong_high_90}개  "
              f"({wrong_high_90/n_wrong*100:.1f}% of wrong)")
        print(f"  wrong인데 max_prob > 0.7 : {wrong_high_70}개  "
              f"({wrong_high_70/n_wrong*100:.1f}% of wrong)")

        overconf_ratio = wrong_high_90 / n_wrong if n_wrong > 0 else 0
        if overconf_ratio > 0.3:
            verdict = "⚠ overconfidence 현상 강함"
        elif overconf_ratio > 0.1:
            verdict = "△ overconfidence 현상 일부 존재"
        else:
            verdict = "○ overconfidence 현상 미미"
        print(f"\n  판정: {verdict}")

    print(f"\n  [Centroid별 오류 분포]")
    centroid_stats = df.groupby("assigned_centroid").agg(
        n=("correct", "count"),
        n_wrong=("correct", lambda x: (~x).sum()),
        mean_max_prob=("max_prob", "mean"),
    ).reset_index()
    centroid_stats["error_rate"] = (
        centroid_stats["n_wrong"] / centroid_stats["n"] * 100
    ).round(1)
    centroid_stats["mean_max_prob"] = centroid_stats["mean_max_prob"].round(4)
    centroid_stats = centroid_stats.sort_values("error_rate", ascending=False)

    print(f"  {'centroid':>9} {'n':>5} {'n_wrong':>7} "
          f"{'error%':>7} {'mean_conf':>10}")
    for _, row in centroid_stats.iterrows():
        flag = " ← collapse?" if row["error_rate"] > 60 else ""
        print(f"  C{int(row['assigned_centroid']):>8} {int(row['n']):>5} "
              f"{int(row['n_wrong']):>7} {row['error_rate']:>6.1f}% "
              f"{row['mean_max_prob']:>10.4f}{flag}")

    print(f"  {'─'*52}")
    return df


# ─────────────────────────────────────────────────────────────
# Temperature Scaling
# ─────────────────────────────────────────────────────────────

def find_temperature(logits_val: np.ndarray, y_val_np: np.ndarray) -> float:
    """
    Validation set NLL을 최소화하는 Temperature T를 찾습니다.

    근거: Guo et al. (ICML 2017) "On Calibration of Modern Neural Networks"
    ────────────────────────────────────────────────────────────────────────
    T > 1로 나누면 softmax 입력이 작아져 확률 분포가 완화됩니다.
        calibrated_probs = softmax(logits / T)

    T는 단 하나의 스칼라 파라미터이며 모델 구조와 예측 순서,
    centroid 배정, retrieval, 설명 구조에 영향을 주지 않습니다.
    """
    def nll(T: float) -> float:
        scaled     = torch.tensor(logits_val, dtype=torch.float32) / T
        probs      = F.softmax(scaled, dim=-1).numpy()
        true_probs = probs[np.arange(len(y_val_np)), y_val_np.astype(int)]
        return float(-np.log(true_probs + 1e-8).mean())

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(result.x)


def apply_temperature_scaling(
    model: TabERA,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    y_val: torch.Tensor,
    y_test: torch.Tensor,
    tasktype: str,
    output_dim: int,
    save_dir: Path,
    openml_id: str,
    seed: int,
) -> dict:
    """
    Temperature Scaling을 적용하고 보정 전후 지표를 비교 출력합니다.
    regression에는 적용하지 않습니다.
    """
    if tasktype == "regression":
        return {}

    from sklearn.metrics import log_loss

    model.eval()
    batch_size = 512

    def get_logits(X):
        parts = []
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                parts.append(model(X[start:start + batch_size])["logits"].cpu())
        return torch.cat(parts, dim=0).numpy()

    logits_val  = get_logits(X_val)
    logits_test = get_logits(X_test)
    y_val_np    = y_val.cpu().numpy()
    y_test_np   = y_test.cpu().numpy().astype(int)

    # ── T 최적화 ─────────────────────────────────────────
    T = find_temperature(logits_val, y_val_np)
    print(f"\n  {'─'*52}")
    print(f"  [Temperature Scaling]  T = {T:.4f}")
    print(f"  {'─'*52}")

    # ── 보정 전후 logloss 비교 ───────────────────────────
    labels = list(range(output_dim))

    probs_before = F.softmax(torch.tensor(logits_test, dtype=torch.float32), dim=-1).numpy()
    probs_after  = F.softmax(
        torch.tensor(logits_test, dtype=torch.float32) / T, dim=-1
    ).numpy()

    logloss_before = log_loss(y_test_np, probs_before, labels=labels)
    logloss_after  = log_loss(y_test_np, probs_after,  labels=labels)

    preds_after  = probs_after.argmax(axis=1)
    max_prob_before = probs_before.max(axis=1).mean()
    max_prob_after  = probs_after.max(axis=1).mean()

    wrong_before = y_test_np != probs_before.argmax(axis=1)
    wrong_after  = y_test_np != preds_after

    print(f"  logloss before TS  : {logloss_before:.4f}")
    print(f"  logloss after  TS  : {logloss_after:.4f}  "
          f"({'↓' if logloss_after < logloss_before else '↑'}"
          f"{abs(logloss_before - logloss_after):.4f})")

    print(f"\n  평균 max_prob before : {max_prob_before:.4f}")
    print(f"  평균 max_prob after  : {max_prob_after:.4f}")

    if wrong_before.sum() > 0:
        wmp_before = probs_before.max(axis=1)[wrong_before].mean()
        wmp_after  = probs_after.max(axis=1)[wrong_after].mean() if wrong_after.sum() > 0 else float("nan")
        print(f"\n  wrong 평균 max_prob before : {wmp_before:.4f}")
        print(f"  wrong 평균 max_prob after  : {wmp_after:.4f}")

    # 보정 후 전체 지표
    cal_preds  = torch.tensor(preds_after)
    cal_probs  = torch.tensor(probs_after)
    cal_test_metrics = calculate_metric(y_test, cal_preds, cal_probs, tasktype, "test")
    print(f"\n  test (calibrated)  : {cal_test_metrics}")

    # ── 저장 ─────────────────────────────────────────────
    ts_path = save_dir / f"data={openml_id}..seed{seed}_temperature.pkl"
    with open(ts_path, "wb") as f:
        pickle.dump({
            "T":                 T,
            "logloss_before":    logloss_before,
            "logloss_after":     logloss_after,
            "cal_probs_test":    probs_after,
            "cal_preds_test":    preds_after,
            "cal_test_metrics":  cal_test_metrics,
        }, f)
    print(f"\n  저장: {ts_path}")
    print(f"  {'─'*52}")

    return {
        "T":               T,
        "cal_probs_test":  probs_after,
        "cal_preds_test":  preds_after,
        "cal_test_metrics": cal_test_metrics,
    }


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
    parser.add_argument("--diagnose",  action="store_true",
                        help="per-sample 진단 저장 (overconfidence 검증용)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Temperature Scaling으로 확률 보정")
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

    # ── 결과 저장 (기존과 동일) ────────────────────────────
    save_dir  = Path(log_dir)
    pred_path = save_dir / f"data={openml_id}..seed{args.seed}_preds.npy"
    meta_path = save_dir / f"data={openml_id}..seed{args.seed}_meta.pkl"

    model.eval()
    with torch.no_grad():
        logits = model(X_test)["logits"].cpu().numpy()
    np.save(str(pred_path), logits)

    meta = {
        "openml_id":    openml_id,
        "tasktype":     tasktype,
        "best_params":  best_params,
        "val_metrics":  val_metrics,
        "test_metrics": test_metrics,
        "seed":         args.seed,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"\n  저장: {pred_path}")

    # ── model state 저장 (기존과 동일) ────────────────────
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

    # ── Per-sample 진단 (--diagnose) ──────────────────────
    if args.diagnose and tasktype != "regression":
        save_per_sample_diagnostics(
            model     = model,
            X_test    = X_test,
            y_test    = y_test,
            tasktype  = tasktype,
            save_dir  = save_dir,
            openml_id = openml_id,
            seed      = args.seed,
        )

    # ── Temperature Scaling (--calibrate) ─────────────────
    if args.calibrate and tasktype != "regression":
        apply_temperature_scaling(
            model      = model,
            X_val      = X_val,
            X_test     = X_test,
            y_val      = y_val,
            y_test     = y_test,
            tasktype   = tasktype,
            output_dim = output_dim,
            save_dir   = save_dir,
            openml_id  = openml_id,
            seed       = args.seed,
        )

    # ── §3.3 Feature 기여도 설명 출력 (기존과 동일) ───────
    if args.explain:
        print(f"\n{'='*52}")
        print(f"  Feature 기여도 설명 (--explain)")
        print(f"{'='*52}")

        print(f"\n  [Centroid Similarity Matrix]")
        sim_mat = model.prototype_layer.cosine_similarity_matrix().numpy()
        cl = model.prototype_layer.centroid_labels
        P  = model.prototype_layer.P
        if cl is not None and not cl.isnan().any():
            order      = cl.cpu().argsort().numpy()
            sim_sorted = sim_mat[order][:, order]
            cl_sorted  = cl.cpu().numpy()[order]
            print(f"  centroid를 평균 레이블 순으로 정렬 ({P}개):")
            print(f"  {'lbl':>5}" + "".join(f"{cl_sorted[j]:5.1f}" for j in range(min(P,10))))
            for i in range(min(P,10)):
                row = "".join(f"{sim_sorted[i,j]:5.2f}" for j in range(min(P,10)))
                print(f"  {cl_sorted[i]:5.1f} {row}")
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

        if model.feature_store is not None and out.get("topk_idx") is not None:
            topk_idx        = out["topk_idx"]
            neighbour_feats = model.feature_store.retrieve(topk_idx)
            for b, exp in enumerate(explanations):
                if b < len(neighbour_feats):
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
