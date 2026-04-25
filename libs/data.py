"""
libs/data.py
============
MultiTab 스타일의 데이터 로더.
dataset_id.json 기반으로 OpenML 데이터셋을 로드하고
train(80%) / val(10%) / test(10%) 고정 분할을 제공합니다.

CA(999999)는 sklearn에서 로드하며, 나머지는 openml 라이브러리를 사용합니다.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
SPLIT_RATIO  = (0.8, 0.1, 0.1)   # train / val / test


# ─────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────

def _sanitize_col(name: str) -> str:
    """Python 식별자로 사용할 수 있도록 컬럼명 정규화."""
    import re
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(name))
    if s and s[0].isdigit():
        s = "f_" + s
    return s


def _encode_target(y: pd.Series, tasktype: str):
    if tasktype == "regression":
        return y.astype(np.float32).values, None
    le = LabelEncoder()
    y_enc = le.fit_transform(y.astype(str)).astype(np.int64)
    return y_enc, le


def _split_data(arrays, seed=RANDOM_SEED, stratify_by=None):
    """80/10/10 stratified 분할.
    stratify_by가 주어지면 클래스 비율을 보존 (auroc NaN 방지)."""
    N = len(arrays[0])
    idx = np.arange(N)
    idx_tv, idx_te = train_test_split(
        idx, test_size=0.1, random_state=seed,
        stratify=stratify_by
    )
    strat_tv = stratify_by[idx_tv] if stratify_by is not None else None
    idx_tr, idx_va = train_test_split(
        idx_tv, test_size=1/9, random_state=seed,
        stratify=strat_tv
    )
    splits = []
    for arr in arrays:
        splits.append((arr[idx_tr], arr[idx_va], arr[idx_te]))
    return splits


def _preprocess_X(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    수치형 + 범주형을 모두 float32로 변환 (Label-encode cat).
    컬럼명은 sanitize 처리.
    반환: X (N, F) float32,  col_names list[str]
    """
    out_cols = []
    out_parts = []

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        X_num = df[num_cols].astype(np.float32).fillna(df[num_cols].median())
        for c in num_cols:
            out_cols.append(_sanitize_col(c))
        out_parts.append(X_num.values)

    for c in cat_cols:
        le = LabelEncoder()
        enc = le.fit_transform(df[c].astype(str)).astype(np.float32)
        out_cols.append(_sanitize_col(c))
        out_parts.append(enc.reshape(-1, 1))

    X = np.concatenate(out_parts, axis=1) if out_parts else np.zeros((len(df), 0), dtype=np.float32)
    return X, out_cols


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def load_dataset(
    openml_id: int | str,
    dataset_info: dict,
    cache_dir: str | Path = "./data_cache",
    seed: int = RANDOM_SEED,
) -> dict:
    """
    단일 데이터셋 로드.

    Parameters
    ----------
    openml_id  : OpenML dataset ID (999999 → California Housing)
    dataset_info : dict from dataset_id.json (name, fullname, tasktype)
    cache_dir  : parquet 캐시 디렉토리
    seed       : 분할 시드

    Returns
    -------
    dict:
        X_train, X_val, X_test : np.ndarray float32 (N, F)  — 정규화 완료
        y_train, y_val, y_test : np.ndarray
        col_names              : list[str]
        n_features             : int
        n_classes              : int or None
        tasktype               : str
        name / fullname        : str
        label_encoder          : LabelEncoder or None
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasktype = dataset_info["tasktype"]
    name     = dataset_info["name"]
    fullname = dataset_info["fullname"]
    oid      = str(openml_id)

    # ── 로드 ──────────────────────────────────────────
    if oid == "999999":
        from sklearn.datasets import fetch_california_housing
        raw = fetch_california_housing()
        df  = pd.DataFrame(raw.data, columns=raw.feature_names)
        y_s = pd.Series(raw.target, name="MedHouseVal")
    else:
        cache_X = cache_dir / f"{oid}_X.parquet"
        cache_y = cache_dir / f"{oid}_y.npy"

        if cache_X.exists() and cache_y.exists():
            df  = pd.read_parquet(cache_X)
            y_s = pd.Series(np.load(str(cache_y), allow_pickle=True))
        else:
            import openml
            ds = openml.datasets.get_dataset(
                int(oid),
                download_data=True,
                download_qualities=False,
                download_features_meta_data=False,
            )
            df_raw, y_s, _, _ = ds.get_data(dataset_format="dataframe")

            # OpenML이 target을 DataFrame 안에 포함해서 반환하는 경우 처리
            if y_s is None:
                target_col = ds.default_target_attribute
                if target_col and target_col in df_raw.columns:
                    y_s = df_raw[target_col]
                    df_raw = df_raw.drop(columns=[target_col])
                else:
                    # 마지막 컬럼을 target으로 간주
                    y_s = df_raw.iloc[:, -1]
                    df_raw = df_raw.iloc[:, :-1]

            df = df_raw
            df.to_parquet(cache_X)
            np.save(str(cache_y), y_s.values, allow_pickle=True)
            y_s = pd.Series(y_s.values)

    # ── 전처리 ────────────────────────────────────────
    X, col_names = _preprocess_X(df)
    y, label_encoder = _encode_target(y_s, tasktype)
    n_classes = len(np.unique(y)) if tasktype != "regression" else None

    # ── 분할 ──────────────────────────────────────────
    # stratify=y로 클래스 비율 보존 → auroc/logloss NaN 방지
    strat = y if tasktype != 'regression' else None
    (X_tr, X_va, X_te), (y_tr, y_va, y_te) = _split_data([X, y], seed=seed, stratify_by=strat)

    # ── 정규화 (train 기준) ───────────────────────────
    mean = X_tr.mean(axis=0)
    std  = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_va = (X_va - mean) / std
    X_te = (X_te - mean) / std

    # ── regression y 정규화 (MultiTab: y_std 저장) ──────
    y_std = np.array(1.0, dtype=np.float32)
    if tasktype == "regression":
        y_mean_tr = y_tr.mean()
        y_std_tr  = y_tr.std() + 1e-8
        y_tr = (y_tr - y_mean_tr) / y_std_tr
        y_va = (y_va - y_mean_tr) / y_std_tr
        y_te = (y_te - y_mean_tr) / y_std_tr
        y_std = np.float32(y_std_tr)

    return dict(
        X_train=X_tr, X_val=X_va, X_test=X_te,
        y_train=y_tr, y_val=y_va, y_test=y_te,
        col_names=col_names,
        n_features=X_tr.shape[1],
        n_classes=n_classes,
        tasktype=tasktype,
        name=name,
        fullname=fullname,
        openml_id=oid,
        label_encoder=label_encoder,
        norm_mean=mean,
        norm_std=std,
        y_std=y_std,          # regression 역정규화용 (MultiTab 호환)
    )


def load_registry(json_path: str | Path = "dataset_id.json") -> dict:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def get_batch_size(n_samples: int) -> int:
    """
    MultiTab 원본과 동일한 배치 사이즈 자동 결정 함수.
    샘플 수에 따라 적절한 배치 사이즈를 반환합니다.
    """
    if n_samples < 1000:
        return 64
    elif n_samples < 10000:
        return 128
    elif n_samples < 50000:
        return 256
    else:
        return 512


# ─────────────────────────────────────────────────────────────
# TabularDataset  (MultiTab TabularDataset 호환 클래스)
# ─────────────────────────────────────────────────────────────

class TabularDataset:
    """
    MultiTab의 TabularDataset과 동일한 인터페이스.

    optimize.py / reproduce.py 에서:
        dataset = TabularDataset(openml_id, tasktype, device=device, seed=seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
        y_std = dataset.y_std
    형태로 사용합니다.
    """

    def __init__(
        self,
        openml_id: int,
        tasktype: str,
        device: "torch.device | str" = "cpu",
        seed: int = RANDOM_SEED,
        cache_dir: str | Path = "./data_cache",
        json_path: str | Path = "dataset_id.json",
    ) -> None:
        import torch

        registry = load_registry(json_path)
        dataset_info = registry[str(openml_id)]

        data = load_dataset(openml_id, dataset_info, cache_dir=cache_dir, seed=seed)

        def _t(arr, is_label=False):
            if tasktype == "multiclass" and is_label:
                return torch.tensor(arr, dtype=torch.long).to(device)
            return torch.tensor(arr, dtype=torch.float32).to(device)

        self._X_train = _t(data["X_train"])
        self._y_train = _t(data["y_train"], is_label=True)   # (N,) 1D
        self._X_val   = _t(data["X_val"])
        self._y_val   = _t(data["y_val"],   is_label=True)   # (N,) 1D
        self._X_test  = _t(data["X_test"])
        self._y_test  = _t(data["y_test"],  is_label=True)   # (N,) 1D

        self.y_std       = data["y_std"]
        self.col_names   = data["col_names"]
        self.n_features  = data["n_features"]
        self.n_classes   = data["n_classes"]
        self.tasktype    = tasktype
        self.name        = data["name"]
        self.fullname    = data["fullname"]
        self.openml_id   = str(openml_id)

    def _indv_dataset(self):
        """
        MultiTab과 동일한 반환 형식:
            (X_train, y_train), (X_val, y_val), (X_test, y_test)
        """
        return (
            (self._X_train, self._y_train),
            (self._X_val,   self._y_val),
            (self._X_test,  self._y_test),
        )
