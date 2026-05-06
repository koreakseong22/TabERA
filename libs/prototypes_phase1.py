"""
libs/prototypes_phase1.py
=========================
Phase 1 헬퍼 함수 모음.

사용법:
    from libs.prototypes_phase1 import compute_bucket_stats, compute_entropy_stats

    # ema_update() return 문 직전에:
    bucket_stats  = compute_bucket_stats(sizes, self.P)
    entropy_stats = compute_entropy_stats(sizes, self.P)

    return {
        # 기존 키 유지 ...
        **bucket_stats,
        **entropy_stats,
    }
"""

from __future__ import annotations
import math
from typing import Dict, List


def compute_bucket_stats(sizes: List[int], P: int) -> Dict[str, float]:
    """
    Bucket size 분포 통계.

    시간복잡도 주장 근거:
        TabERA의 retrieval cost = O(PD + |B_k|D)
        max_mean_ratio가 낮을수록 "balanced buckets" 가정이 성립하고
        효율성 주장이 강해진다.

    Args:
        sizes: 각 centroid에 배정된 샘플 수 리스트 (len == P)
        P:     centroid 수

    Returns:
        mean_cluster_size : 활성 bucket 평균 크기
        std_cluster_size  : 활성 bucket 크기 표준편차
        max_mean_ratio    : max_bucket / mean_bucket  (핵심 지표)
        empty_ratio       : 빈 bucket 비율
    """
    active_sizes = [s for s in sizes if s > 0]
    n_active = len(active_sizes)

    if n_active == 0:
        return {
            "mean_cluster_size": 0.0,
            "std_cluster_size":  0.0,
            "max_mean_ratio":    0.0,
            "empty_ratio":       1.0,
        }

    mean_s = sum(active_sizes) / n_active
    var_s  = sum((s - mean_s) ** 2 for s in active_sizes) / n_active
    std_s  = math.sqrt(var_s)
    max_s  = max(active_sizes)

    return {
        "mean_cluster_size": round(mean_s, 2),
        "std_cluster_size":  round(std_s, 2),
        "max_mean_ratio":    round(max_s / mean_s, 3) if mean_s > 0 else 0.0,
        "empty_ratio":       round(1.0 - n_active / P, 3),
    }


def compute_entropy_stats(sizes: List[int], P: int) -> Dict[str, float]:
    """
    Hard assignment 기준 usage entropy.

    현재 entropy_loss(routing_probs)의 forward 값은 hard_one_hot이므로,
    실질적으로 이 분포의 entropy를 최대화하는 것과 같다.
    이 지표는 entropy_loss가 실제로 높이려는 값을 직접 측정한다.

    Args:
        sizes: 각 centroid에 배정된 샘플 수
        P:     centroid 수

    Returns:
        usage_entropy_hard_norm: 정규화 entropy [0, 1]
            1.0 = 완전 균등 사용 (이상적)
            0.0 = 하나의 centroid로 collapse
    """
    total = sum(sizes)
    if total == 0 or P <= 1:
        return {"usage_entropy_hard_norm": 0.0}

    entropy = 0.0
    for s in sizes:
        if s > 0:
            p = s / total
            entropy -= p * math.log(p + 1e-8)

    max_entropy = math.log(P)
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {"usage_entropy_hard_norm": round(norm_entropy, 4)}


def compute_label_purity(
    sizes: List[int],
    sample_groups: List[List[int]],
    y_train_np,   # np.ndarray (N,)
    P: int,
) -> Dict[str, float]:
    """
    Bucket label purity.

    각 centroid bucket 내 샘플들의 label 분포를 보고,
    majority label 비율(purity)을 계산한다.

    class imbalance 주의:
        weighted purity는 bucket 크기로 가중 평균하므로,
        majority class가 큰 데이터에서 과대평가될 수 있다.
        따라서 purity 해석 시 class 분포와 함께 봐야 한다.

    Args:
        sizes:         각 bucket의 샘플 수
        sample_groups: 각 bucket의 샘플 인덱스 리스트
        y_train_np:    학습 레이블 (numpy array)
        P:             centroid 수

    Returns:
        bucket_label_purity:          weighted average purity [0, 1]
        bucket_label_purity_unweighted: simple average purity [0, 1]
    """
    import numpy as np

    purity_list = []
    weight_list = []

    for p in range(P):
        grp = sample_groups[p]
        if len(grp) == 0:
            continue
        labels_in_bucket = y_train_np[grp]
        # 가장 많은 label의 비율
        unique, counts = np.unique(labels_in_bucket, return_counts=True)
        majority_count = int(counts.max())
        purity_p = majority_count / len(grp)
        purity_list.append(purity_p)
        weight_list.append(len(grp))

    if not purity_list:
        return {
            "bucket_label_purity":            0.0,
            "bucket_label_purity_unweighted": 0.0,
        }

    total_weight = sum(weight_list)
    weighted_purity = sum(
        p * w for p, w in zip(purity_list, weight_list)
    ) / total_weight
    simple_purity = sum(purity_list) / len(purity_list)

    return {
        "bucket_label_purity":            round(weighted_purity, 4),
        "bucket_label_purity_unweighted": round(simple_purity, 4),
    }