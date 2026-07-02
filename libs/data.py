"""
libs/data.py
============
TabERA 데이터 로더 — MultiTab 원본 파이프라인 기준.

[통합 배경]
논문에서 MultiTab 계열 baseline(TabM 등)과 공정 비교하려면 동일한
train/val/test 분할이 필요함. 기존에는 TabERA 자체 80/10/10 stratified
split(QuantileTransformer, mean imputation)을 썼는데, MultiTab은
StratifiedKFold(10-fold) 분할에 NaN 행 제거 방식을 씀 — 두 파이프라인이
달라서 초기 TabM 비교(id=41027)가 무효였던 사건이 있었음. 이후
`libs/data_multitab.py` + `optimize_multitab_split.py`를 별도로 만들어
검증했고, 이제 이 파이프라인을 본 실험 기본값으로 승격.

[MultiTab 원본 대비 유일한 추가 사항]
TabularDataset이 optimize.py/reproduce.py가 기대하는 인터페이스
(.n_features, .n_classes, .col_names)를 추가로 제공하고, multiclass
y를 원-핫에서 정수 클래스 인덱스로 변환함 (TabERA의 CrossEntropyLoss/
eval.py가 정수 라벨을 전제로 하기 때문 — 원-핫 → argmax는 정보 손실
없는 역변환).

split_data()의 seed 의미: "몇 번째 fold를 test로 쓸지"
(StratifiedKFold random_state=42로 고정, seed는 fold 인덱스 선택용).
"""

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import openml, torch
import numpy as np
import pandas as pd
import sklearn.datasets
import scipy.stats
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def get_batch_size(n):
    ### n = train data size
    if n > 50000:
        return 1024
    elif n > 10000:
        return 512
    elif n > 5000:
        return 256
    elif n > 1000:
        return 128
    else:
        return 64


def load_data(openml_id):
    if openml_id == 999999:
        dataset = sklearn.datasets.fetch_california_housing()
        X = pd.DataFrame(dataset['data'])
        y = pd.DataFrame(dataset['target'])
        categorical_indicator = []
        attribute_names = X.columns.tolist()
    elif openml_id == 43611:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: class')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target="class"
        )
    elif openml_id == 43454:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: loan_status')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target="loan_status"
        )
    elif openml_id == 43823:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: Heart_Disease')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target="Heart_Disease"
        )
    else:
        dataset = openml.datasets.get_dataset(openml_id)
        print(f'Dataset is loaded.. Data name: {dataset.name}, Target feature: {dataset.default_target_attribute}')
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

    if openml_id == 537:
        y = y / 10000

    nan_counts = X.isna().values.sum()
    cell_counts = X.shape[0] * X.shape[1]
    n_samples = X.shape[0]
    n_cols = X.shape[1]

    ### Preprocess NaN
    # 1. Remove columns containing more than 50% NaN values
    nan_cols = X.isna().sum(0)
    valid_cols = nan_cols.loc[nan_cols < (0.5 * len(X))].index.tolist()
    total_features = len(valid_cols)
    X = X[valid_cols]
    # 2. Exclude samples containing any NaN values in either inputs or labels
    nan_idx = X.isna().any(axis=1)
    X = X[~nan_idx].reset_index(drop=True)
    y = y[~nan_idx].reset_index(drop=True)

    # 3. convert categorical features into integers (but still they are categorical)
    cat_features = np.array(attribute_names)[categorical_indicator]
    cat_features = [c for c in cat_features if c in valid_cols]
    for v in valid_cols:
        if not v in cat_features:
            try:
                X[v].astype(np.float32)
            except ValueError:
                valid_cols.remove(v)
    X = X[valid_cols]

    cat_cols = [valid_cols.index(x) for x in cat_features]
    num_cols = [valid_cols.index(x) for x in valid_cols if not x in cat_features]
    cat_cardinality = [X[c].cat.categories.size for c in cat_features]
    for col in cat_features:
        colencoder = LabelEncoder()
        X[col] = colencoder.fit_transform(X[col])
    X = X.values
    invalid_num_cols = []
    for col in num_cols:
        if X[:, col].dtype == np.object_:
            try:
                X[:, col] = X[:, col].astype(np.float32)
            except (ValueError, TypeError):
                invalid_num_cols.append(col)
    if invalid_num_cols:
        print(f"  [data.py] categorical_indicator 미반영 문자열 컬럼 제거: {invalid_num_cols}")
        keep_mask = [i for i in range(X.shape[1]) if i not in invalid_num_cols]
        X = X[:, keep_mask]
        num_cols = [keep_mask.index(i) for i in num_cols if i not in invalid_num_cols]
        cat_cols = [keep_mask.index(i) for i in cat_cols if i not in invalid_num_cols]
    X = X.astype(np.float32)

    y = y.values
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    print("full data size", X.shape)
    return X, y, cat_cols, cat_cardinality, num_cols


def one_hot(y):
    num_classes = len(np.unique(y))
    min_class = y.min()
    enc = LabelEncoder()
    y_ = enc.fit_transform(y - min_class)
    return np.eye(num_classes)[y_]


def split_data(X, y, tasktype, num_indices=[], seed=0, device='cuda'):
    if tasktype == "multiclass":
        y = one_hot(y)

    # StratifiedKFold: 클래스 비율 보존 (TabZilla/MultiTab 벤치마크 기준 통일)
    from sklearn.model_selection import StratifiedKFold
    y_for_split = np.argmax(y, axis=1) if y.ndim > 1 else y
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_idx = list(kf.split(X, y_for_split))
    tr_idx, te_idx = fold_idx[seed]
    val_split_idx = (seed + 1) % 10
    _, val_idx = fold_idx[val_split_idx]
    tr_idx = np.setdiff1d(tr_idx, val_idx)

    X_train = torch.from_numpy(X[tr_idx]).type(torch.float32).to(device)
    X_val = torch.from_numpy(X[val_idx]).type(torch.float32).to(device)
    X_test = torch.from_numpy(X[te_idx]).type(torch.float32).to(device)

    y_train = torch.from_numpy(y[tr_idx]).type(torch.float32).to(device)
    y_val = torch.from_numpy(y[val_idx]).type(torch.float32).to(device)
    y_test = torch.from_numpy(y[te_idx]).type(torch.float32).to(device)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std = prep_data(
        X_train, X_val, X_test, y_train, y_val, y_test, num_indices=num_indices, tasktype=tasktype
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std


## following Gorishniy et al., 2021
def prep_data(X_train, X_val, X_test, y_train, y_val, y_test, num_indices=[], tasktype='multiclass'):
    device = X_train.device
    if len(num_indices) > 0:
        quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
        X_train[:, num_indices] = torch.as_tensor(
            quantile_transformer.fit_transform(X_train[:, num_indices].cpu().numpy()),
            device=device, dtype=X_train.dtype,
        )
        X_val[:, num_indices] = torch.as_tensor(
            quantile_transformer.transform(X_val[:, num_indices].cpu().numpy()),
            device=device, dtype=X_val.dtype,
        )
        X_test[:, num_indices] = torch.as_tensor(
            quantile_transformer.transform(X_test[:, num_indices].cpu().numpy()),
            device=device, dtype=X_test.dtype,
        )
    if tasktype == "regression":
        standard_transformer = StandardScaler()
        y_train = torch.as_tensor(
            standard_transformer.fit_transform(y_train.reshape(-1, 1).cpu().numpy()).reshape(-1),
            device=device, dtype=y_train.dtype,
        )
        y_std = standard_transformer.scale_.item()
        y_val = torch.as_tensor(
            standard_transformer.transform(y_val.reshape(-1, 1).cpu().numpy()).reshape(-1),
            device=device, dtype=y_val.dtype,
        )
        y_test = torch.as_tensor(
            standard_transformer.transform(y_test.reshape(-1, 1).cpu().numpy()).reshape(-1),
            device=device, dtype=y_test.dtype,
        )
    else:
        y_std = 1.
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std


class TabularDataset(torch.utils.data.Dataset):
    """
    optimize.py / reproduce.py 에서:
        dataset = TabularDataset(openml_id, tasktype, device=device, seed=seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
        dataset.n_features / dataset.n_classes / dataset.col_names / dataset.y_std
    형태로 사용.
    """

    def __init__(self, openml_id, tasktype, device, seed=1):
        X, y, self.X_cat, self.X_cat_cardinality, self.X_num = load_data(openml_id)
        self.tasktype = tasktype

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test), self.y_std = \
            split_data(X, y, self.tasktype, num_indices=self.X_num, seed=seed, device=device)

        # multiclass: one-hot((N,C)) → 정수 클래스 인덱스
        # (TabERA의 CrossEntropyLoss/eval.py가 정수 라벨을 전제로 함;
        #  원-핫 → argmax는 정보 손실 없는 역변환)
        if self.tasktype == "multiclass":
            self.n_classes = self.y_train.shape[1]
            self.y_train = self.y_train.argmax(dim=-1).long()
            self.y_val   = self.y_val.argmax(dim=-1).long()
            self.y_test  = self.y_test.argmax(dim=-1).long()
        elif self.tasktype == "binclass":
            self.n_classes = 2
            self.y_train = self.y_train.float()
            self.y_val   = self.y_val.float()
            self.y_test  = self.y_test.float()
        else:
            self.n_classes = None

        self.n_features = self.X_train.shape[1]
        # 원본 컬럼명은 보존되지 않음 (openml categorical_indicator 처리
        # 과정에서 유실) — 설명① 텍스트 표시("alcohol=10.24" 등)에만
        # 영향을 주고 학습/정확도에는 전혀 무관.
        self.col_names = [f"f{i}" for i in range(self.n_features)]

        print("input dim: %i, cat: %i, num: %i" % (self.n_features, len(self.X_cat), len(self.X_num)))
        self.batch_size = get_batch_size(len(self.X_train))

    def __len__(self, data):
        if data == "train":
            return len(self.X_train)
        elif data == "val":
            return len(self.X_val)
        else:
            return len(self.X_test)

    def _indv_dataset(self):
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)

    def __getitem__(self, idx, data):
        if data == "train":
            return self.X_train[idx], self.y_train[idx]
        elif data == "val":
            return self.X_val[idx], self.y_val[idx]
        else:
            return self.X_test[idx], self.y_test[idx]