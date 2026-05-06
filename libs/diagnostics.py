"""
libs/diagnostics.py
===================
Phase 1 진단 도구 모음.

포함 기능:
    1. compute_faithfulness_score  — feature explanation faithfulness test
    2. compute_soft_entropy_stats  — soft_probs 기반 entropy (Phase 2 준비)
    3. run_phase1_diagnostics      — 전체 Phase 1 진단 통합 실행

사용법 (reproduce.py 끝에 추가):
    from libs.diagnostics import run_phase1_diagnostics
    diag = run_phase1_diagnostics(model, X_test, y_test, X_train, tasktype)
    print(diag)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1. Faithfulness Test
# ─────────────────────────────────────────────────────────────

def compute_faithfulness_score(
    model,
    X: torch.Tensor,           # (N, F) 평가할 샘플
    X_train_mean: torch.Tensor, # (F,) 학습 데이터 feature 평균
    tasktype: str,
    top_m: int = 3,
    n_samples: int = 200,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Feature explanation faithfulness test.

    핵심 아이디어:
        모델이 중요하다고 한 feature를 제거(→ train mean 대체)했을 때
        예측이 많이 바뀐다면, 그 설명은 예측과 연결되어 있다.

    세 가지 masking 조건 비교:
        top_mask:    모델이 중요하다고 한 상위 top_m feature 제거
        random_mask: 무작위 top_m feature 제거
        low_mask:    모델이 중요하지 않다고 한 하위 top_m feature 제거

    기대 결과 (faithfulness가 있는 경우):
        top_mask 예측 변화 > random_mask > low_mask

    Args:
        model:          TabERA 모델 (eval 모드)
        X:              평가 샘플 (N, F)
        X_train_mean:   feature별 train 평균 (F,)
        tasktype:       "binclass" / "multiclass" / "regression"
        top_m:          제거할 feature 수
        n_samples:      평가에 사용할 샘플 수 (대형 데이터셋 효율)
        device:         "cpu" or "cuda"

    Returns:
        top_mask_delta:    top feature 제거 시 예측 변화량
        random_mask_delta: random feature 제거 시 예측 변화량
        low_mask_delta:    low feature 제거 시 예측 변화량
        faithfulness_gap:  top_mask_delta - random_mask_delta
                           > 0 이면 explanation이 예측에 연결됨
        is_faithful:       faithfulness_gap > 0 여부
    """
    model.eval()

    # 샘플 수 제한
    N = min(n_samples, X.shape[0])
    X_eval = X[:N].to(device)
    X_mean = X_train_mean.to(device)

    top_deltas    = []
    random_deltas = []
    low_deltas    = []

    with torch.no_grad():
        for i in range(N):
            xi = X_eval[i:i+1]  # (1, F)

            # ── 원본 예측 ──────────────────────────────────
            out_orig = model(xi, return_explanations=True)
            logit_orig = out_orig["logits"]

            # ── feature importance 추출 ────────────────────
            explanations = out_orig.get("explanations", [])
            if not explanations:
                continue

            feat_match = explanations[0].get("feature_match")
            if feat_match is None:
                continue

            top_features = feat_match.get("top_features", [])
            if len(top_features) < top_m:
                continue

            F_dim = xi.shape[1]
            n_feats = len(top_features)

            # feature 이름 → 인덱스 매핑은 col_names 순서 기반
            # top_features는 importance 내림차순으로 정렬되어 있음
            top_indices = list(range(min(top_m, n_feats)))
            low_indices = list(range(max(0, n_feats - top_m), n_feats))

            # feature_imp가 (k, F) 텐서로 있을 경우 직접 사용
            feature_imp = out_orig.get("feature_imp")
            if feature_imp is not None and feature_imp.ndim >= 2:
                # (B, k, F) or (B, F)
                imp = feature_imp[0]
                if imp.ndim == 2:
                    imp = imp.mean(0)  # (F,)
                imp_np = imp.cpu().numpy()
                sorted_idx = np.argsort(imp_np)[::-1]
                top_feat_idx  = sorted_idx[:top_m].tolist()
                low_feat_idx  = sorted_idx[-top_m:].tolist()
                rand_feat_idx = np.random.choice(F_dim, top_m, replace=False).tolist()
            else:
                # feature_imp 없을 때: top_features 리스트의 순서로 근사
                # col_names 기반 인덱스 복원
                col_names = getattr(model, 'column_names', None) or \
                            [f"f{j}" for j in range(F_dim)]
                name_to_idx = {n: j for j, n in enumerate(col_names)}

                top_feat_idx = [
                    name_to_idx.get(f["feature"], j)
                    for j, f in enumerate(top_features[:top_m])
                ]
                low_feat_idx = [
                    name_to_idx.get(f["feature"], j)
                    for j, f in enumerate(top_features[-top_m:])
                ]
                rand_feat_idx = np.random.choice(F_dim, top_m, replace=False).tolist()

            # ── 세 가지 masking ────────────────────────────
            def masked_pred(feat_idx):
                xi_masked = xi.clone()
                xi_masked[0, feat_idx] = X_mean[feat_idx]
                return model(xi_masked)["logits"]

            logit_top  = masked_pred(top_feat_idx)
            logit_rand = masked_pred(rand_feat_idx)
            logit_low  = masked_pred(low_feat_idx)

            # ── 예측 변화량 계산 ───────────────────────────
            if tasktype == "regression":
                delta_top  = (logit_orig - logit_top).abs().item()
                delta_rand = (logit_orig - logit_rand).abs().item()
                delta_low  = (logit_orig - logit_low).abs().item()
            else:
                # 정답 클래스 확률 변화
                if tasktype == "binclass":
                    prob_orig  = torch.sigmoid(logit_orig).item()
                    prob_top   = torch.sigmoid(logit_top).item()
                    prob_rand  = torch.sigmoid(logit_rand).item()
                    prob_low   = torch.sigmoid(logit_low).item()
                else:
                    probs_orig = F.softmax(logit_orig, dim=-1)
                    pred_class = probs_orig.argmax().item()
                    prob_orig  = probs_orig[0, pred_class].item()
                    prob_top   = F.softmax(logit_top,  dim=-1)[0, pred_class].item()
                    prob_rand  = F.softmax(logit_rand, dim=-1)[0, pred_class].item()
                    prob_low   = F.softmax(logit_low,  dim=-1)[0, pred_class].item()

                delta_top  = abs(prob_orig - prob_top)
                delta_rand = abs(prob_orig - prob_rand)
                delta_low  = abs(prob_orig - prob_low)

            top_deltas.append(delta_top)
            random_deltas.append(delta_rand)
            low_deltas.append(delta_low)

    if not top_deltas:
        return {
            "top_mask_delta":    0.0,
            "random_mask_delta": 0.0,
            "low_mask_delta":    0.0,
            "faithfulness_gap":  0.0,
            "is_faithful":       False,
            "n_evaluated":       0,
        }

    top_mean    = float(np.mean(top_deltas))
    random_mean = float(np.mean(random_deltas))
    low_mean    = float(np.mean(low_deltas))
    gap         = top_mean - random_mean

    return {
        "top_mask_delta":    round(top_mean, 5),
        "random_mask_delta": round(random_mean, 5),
        "low_mask_delta":    round(low_mean, 5),
        "faithfulness_gap":  round(gap, 5),
        "is_faithful":       gap > 0,
        "n_evaluated":       len(top_deltas),
    }


# ─────────────────────────────────────────────────────────────
# 2. Soft Entropy Stats (Phase 2 준비용 — 현재는 로깅만)
# ─────────────────────────────────────────────────────────────

def compute_soft_entropy_stats(
    model,
    X: torch.Tensor,
    n_samples: int = 500,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    soft_probs 기반 entropy 통계.

    현재 forward()는 soft_probs를 반환하지 않으므로,
    여기서는 내부적으로 직접 계산한다.

    Phase 2에서 soft_probs를 반환하도록 수정하면
    이 함수는 더 간단해진다.

    반환:
        sample_entropy_norm:      샘플별 soft routing entropy 평균 [0,1]
                                  낮을수록 routing이 명확함 (좋음)
        usage_entropy_soft_norm:  batch-level soft usage entropy [0,1]
                                  높을수록 centroid가 골고루 쓰임 (좋음)
    """
    model.eval()
    P = model.prototype_layer.P

    all_soft = []
    N = min(n_samples, X.shape[0])
    X_eval = X[:N].to(device)

    with torch.no_grad():
        # batch 처리
        batch_size = 128
        for start in range(0, N, batch_size):
            xb = X_eval[start:start+batch_size]
            query_emb = model.embedder(xb)

            # soft 직접 계산 (forward를 우회하지 않고 prototype layer 내부 재현)
            q = F.normalize(query_emb, dim=-1)
            c = F.normalize(model.prototype_layer.centroid_emb, dim=-1)
            logits = q @ c.T           # (B, P)
            soft   = F.softmax(logits, dim=-1)  # (B, P)
            all_soft.append(soft.cpu())

    soft_all = torch.cat(all_soft, dim=0)  # (N, P)

    # sample-level entropy
    sample_ent = -(soft_all * torch.log(soft_all + 1e-8)).sum(dim=1).mean().item()
    max_ent    = math.log(P)
    sample_ent_norm = sample_ent / max_ent if max_ent > 0 else 0.0

    # batch-level soft usage entropy
    avg_soft = soft_all.mean(dim=0)  # (P,)
    usage_ent_soft = -(avg_soft * torch.log(avg_soft + 1e-8)).sum().item()
    usage_ent_soft_norm = usage_ent_soft / max_ent if max_ent > 0 else 0.0

    return {
        "sample_entropy_norm":       round(sample_ent_norm, 4),
        "usage_entropy_soft_norm":   round(usage_ent_soft_norm, 4),
        # 해석 가이드
        # sample_entropy_norm 이상적 범위: < 0.5
        # usage_entropy_soft_norm 이상적 범위: > 0.7
    }


# ─────────────────────────────────────────────────────────────
# 3. 통합 Phase 1 진단
# ─────────────────────────────────────────────────────────────

def run_phase1_diagnostics(
    model,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    X_train: torch.Tensor,
    tasktype: str,
    top_m: int = 3,
    n_faithfulness: int = 200,
    n_entropy: int = 500,
    device: str = "cpu",
) -> Dict[str, object]:
    """
    Phase 1 전체 진단 통합 실행.

    Args:
        model:           학습 완료된 TabERA 모델
        X_test:          테스트 feature (N, F)
        y_test:          테스트 레이블
        X_train:         학습 feature (train mean 계산용)
        tasktype:        task 유형
        top_m:           faithfulness test에서 제거할 feature 수
        n_faithfulness:  faithfulness test 샘플 수
        n_entropy:       entropy 계산 샘플 수
        device:          "cpu" or "cuda"

    Returns:
        전체 진단 결과 dict
    """
    print("\n" + "═" * 56)
    print("  Phase 1 진단 시작")
    print("═" * 56)

    model.eval()
    model = model.to(device)

    X_train_mean = X_train.float().mean(dim=0).to(device)

    # ── 1. Entropy stats ─────────────────────────────────────
    print("\n[1/3] Soft entropy 계산 중...")
    entropy_stats = compute_soft_entropy_stats(
        model, X_test, n_samples=n_entropy, device=device
    )
    print(f"      sample_entropy_norm     : {entropy_stats['sample_entropy_norm']:.4f}")
    print(f"        → {'✅ 명확한 routing' if entropy_stats['sample_entropy_norm'] < 0.5 else '⚠️  routing이 흐림'}")
    print(f"      usage_entropy_soft_norm : {entropy_stats['usage_entropy_soft_norm']:.4f}")
    print(f"        → {'✅ 균등 사용' if entropy_stats['usage_entropy_soft_norm'] > 0.7 else '⚠️  일부 centroid 편중'}")

    # ── 2. Faithfulness test ──────────────────────────────────
    print(f"\n[2/3] Faithfulness test (top_m={top_m}, n={n_faithfulness})...")
    faith_stats = compute_faithfulness_score(
        model, X_test, X_train_mean,
        tasktype=tasktype,
        top_m=top_m,
        n_samples=n_faithfulness,
        device=device,
    )
    print(f"      n_evaluated     : {faith_stats['n_evaluated']}")
    print(f"      top_mask_delta  : {faith_stats['top_mask_delta']:.5f}")
    print(f"      random_delta    : {faith_stats['random_mask_delta']:.5f}")
    print(f"      low_delta       : {faith_stats['low_mask_delta']:.5f}")
    print(f"      faithfulness_gap: {faith_stats['faithfulness_gap']:+.5f}")
    is_f = faith_stats['is_faithful']
    print(f"      → {'✅ Faithful (top > random)' if is_f else '❌ Not faithful (top ≤ random)'}")

    # ── 3. Bucket stats 요약 (ema_history 마지막 값 사용) ─────
    print("\n[3/3] Bucket 상태 요약...")
    ema_hist = getattr(
        getattr(model, '_wrapper', None), 'ema_history', None
    )
    bucket_summary = {}
    if ema_hist and len(ema_hist) > 0:
        last = ema_hist[-1]
        bucket_summary = {
            "mean_cluster_size":       last.get("mean_cluster_size", "N/A"),
            "max_mean_ratio":          last.get("max_mean_ratio", "N/A"),
            "empty_ratio":             last.get("empty_ratio", "N/A"),
            "usage_entropy_hard_norm": last.get("usage_entropy_hard_norm", "N/A"),
            "bucket_label_purity":     last.get("bucket_label_purity", "N/A"),
        }
        for k, v in bucket_summary.items():
            print(f"      {k:30s}: {v}")
    else:
        print("      (ema_history 없음 — supervised.py 패치 필요)")

    # ── 결과 통합 ─────────────────────────────────────────────
    result = {
        **entropy_stats,
        **faith_stats,
        **bucket_summary,
    }

    print("\n" + "═" * 56)
    print("  Phase 1 진단 완료")
    print("  이 수치들이 Phase 2 soft cohesion 추가 후 기준값이 됩니다.")
    print("  기대 변화:")
    print("    sample_entropy_norm    ↓  (routing이 더 명확해짐)")
    print("    bucket_label_purity    ↑  (bucket이 더 순수해짐)")
    print("    faithfulness_gap       ↑  (설명이 예측에 더 연결됨)")
    print("═" * 56 + "\n")

    return result


# ─────────────────────────────────────────────────────────────
# 빠른 테스트
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("libs/diagnostics.py 로드 성공")
    print("함수 목록:")
    print("  compute_faithfulness_score(model, X, X_train_mean, tasktype)")
    print("  compute_soft_entropy_stats(model, X)")
    print("  run_phase1_diagnostics(model, X_test, y_test, X_train, tasktype)")
