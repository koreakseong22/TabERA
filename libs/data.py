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
    # [수정] colencoder.classes_[i] = 코드 i가 원래 뭐였는지(원본 카테고리
    # 문자열). 이전에는 colencoder가 루프 지역 변수라 fit 직후 버려져서,
    # "Category 0"이 실제로 뭘 뜻하는지 나중에 알 방법이 없었음. 컬럼명
    # (col_name)을 키로 저장 — invalid_num_cols 필터링은 numeric 컬럼만
    # 건드리므로 categorical 컬럼명은 이후에도 안 바뀜.
    cat_category_names = {}
    for col in cat_features:
        colencoder = LabelEncoder()
        X[col] = colencoder.fit_transform(X[col])
        cat_category_names[col] = [str(c) for c in colencoder.classes_]
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
        # [수정] X를 keep_mask로 걸렀으면, valid_cols(컬럼명 리스트)도
        # 같은 위치 기준으로 걸러야 최종 X의 컬럼 순서와 이름이 계속
        # 1:1로 맞는다 — 안 그러면 col_names가 실제와 어긋난 채로 반환됨.
        valid_cols = [valid_cols[i] for i in keep_mask]
    X = X.astype(np.float32)

    y = y.values
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    # [수정] categorical feature와 같은 문제: 원래 target 라벨 문자열
    # (예: "good"/"bad")이 정수 코드(0/1)로 바뀐 뒤 매핑이 버려졌음.
    # classification(binclass/multiclass)에서만 의미 있고, regression은
    # 호출부(label_groups_by_target)에서 애초에 이 값을 안 씀.
    target_class_names = [str(c) for c in labelencoder.classes_]

    print("full data size", X.shape)
    return X, y, cat_cols, cat_cardinality, num_cols, valid_cols, cat_category_names, target_class_names


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

    (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std, quantile_transformer = prep_data(
        X_train, X_val, X_test, y_train, y_val, y_test, num_indices=num_indices, tasktype=tasktype
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std, quantile_transformer


## following Gorishniy et al., 2021
def prep_data(X_train, X_val, X_test, y_train, y_val, y_test, num_indices=[], tasktype='multiclass'):
    device = X_train.device
    quantile_transformer = None
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
    # [수정] 이전에는 quantile_transformer가 fit된 채로 함수 지역변수로
    # 버려졌음 — numeric feature 표시 시 "0.328"이 실제 몇 단위인지
    # 역변환(inverse_transform)할 방법이 없어서 [0,1] uniform 값 그대로
    # 노출되던 원인. 반환해서 TabularDataset이 보관하도록 함.
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), y_std, quantile_transformer


class TabularDataset(torch.utils.data.Dataset):
    """
    optimize.py / reproduce.py 에서:
        dataset = TabularDataset(openml_id, tasktype, device=device, seed=seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
        dataset.n_features / dataset.n_classes / dataset.col_names / dataset.y_std
        dataset.cat_category_names  # {col_name: [원본 카테고리 문자열, ...]}
                                     # 코드 i ↔ cat_category_names[col][i]
        dataset.target_class_names # [원본 target 라벨 문자열, ...] (classification만 의미 있음)
                                     # 코드 i ↔ target_class_names[i]
        dataset.quantile_transformer # fit된 QuantileTransformer(numeric feature용) —
                                     # .inverse_transform()으로 [0,1] 값을 실제 단위로 되돌릴 수 있음.
                                     # num_indices(X_num)가 비어있으면 None.
    형태로 사용.
    """

    def __init__(self, openml_id, tasktype, device, seed=1):
        X, y, self.X_cat, self.X_cat_cardinality, self.X_num, raw_col_names, \
            self.cat_category_names, self.target_class_names = load_data(openml_id)
        self.tasktype = tasktype

        (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test), self.y_std, \
            self.quantile_transformer = \
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
        # [수정] 이전에는 load_data()가 실제 컬럼명(valid_cols)을 계산해
        # 놓고도 반환하지 않아서, 여기서 f0/f1/... 같은 자리표시자로
        # 대체했었음 — 설명①/② 텍스트 표시("checking_status=..." 등)에
        # 실제 이름 대신 f0/f1이 나오던 원인. 이제 load_data()가 반환하는
        # 실제 이름을 그대로 쓴다.
        assert len(raw_col_names) == self.n_features, (
            f"col_names 길이({len(raw_col_names)})가 n_features({self.n_features})와 "
            f"다릅니다 — load_data()의 컬럼 필터링 단계 중 하나가 valid_cols와 "
            f"동기화가 안 됐을 수 있습니다."
        )
        self.col_names = raw_col_names

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