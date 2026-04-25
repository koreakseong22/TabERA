"""
ensemble.py
===========
여러 seed의 재현 결과를 앙상블합니다.
MultiTab의 ensemble.py 형식을 따릅니다.

사용법
------
python ensemble.py \
    --openml_id 45068 \
    --seeds 0 1 2 3 4 \
    --savepath ./optim_logs

결과: optim_logs/{openml_id}_ensemble_result.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from libs.data import load_dataset, load_registry
from libs.eval import compute_metric
import torch


def main():
    parser = argparse.ArgumentParser(description="HypoTabR Ensemble")
    parser.add_argument("--openml_id", type=int, required=True)
    parser.add_argument("--seeds",     type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--savepath",  type=str, default="./optim_logs")
    parser.add_argument("--json",      type=str, default="dataset_id.json")
    args = parser.parse_args()

    save_dir  = Path(args.savepath)
    openml_id = str(args.openml_id)

    registry     = load_registry(args.json)
    dataset_info = registry[openml_id]
    tasktype     = dataset_info["tasktype"]

    print(f"[HypoTabR] Ensemble: {dataset_info['fullname']} (id={openml_id})")

    # ── 예측 로드 ──────────────────────────────────────
    all_preds = []
    for seed in args.seeds:
        pred_path = save_dir / f"{openml_id}_seed{seed}_preds.npy"
        if not pred_path.exists():
            print(f"  [SKIP] seed={seed}: {pred_path} 없음")
            continue
        all_preds.append(np.load(str(pred_path)))
        print(f"  Loaded seed={seed}: {pred_path}")

    if not all_preds:
        raise RuntimeError("앙상블할 예측 파일이 없습니다. reproduce.py를 먼저 실행하세요.")

    # ── 앙상블 ────────────────────────────────────────
    stacked = np.stack(all_preds, axis=0)   # (n_seeds, N, C) or (n_seeds, N, 1)
    ensemble_pred = stacked.mean(axis=0)    # (N, C) or (N, 1)

    # ── 테스트 레이블 로드 ────────────────────────────
    data    = load_dataset(args.openml_id, dataset_info)
    y_test  = data["y_test"]
    n_classes = data["n_classes"]

    # 메트릭 계산
    logits_t = torch.tensor(ensemble_pred, dtype=torch.float32)
    y_t      = torch.tensor(y_test)
    metrics  = compute_metric(logits_t, y_t, tasktype, n_classes)

    print(f"\n  앙상블 결과 ({len(all_preds)} seeds)")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # ── 결과 저장 ─────────────────────────────────────
    result = {
        "openml_id":  openml_id,
        "tasktype":   tasktype,
        "n_seeds":    len(all_preds),
        "metrics":    metrics,
        "ensemble_pred": ensemble_pred,
        "y_test":     y_test,
    }
    out_path = save_dir / f"{openml_id}_ensemble_result.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result, f)
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    main()
