## Main file for optimizing TabERA for a specific dataset.
## Paper info: TabERA — Tabular Hierarchical Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import sys, os, argparse

# ── CUDA_VISIBLE_DEVICES를 torch import 전 최우선 설정 ──────────
# MultiTab 원본과 동일하게 argparse 직후, torch import 전에 설정
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",    type=int, default=0,      help="gpu index")
parser.add_argument("--openml_id", type=int, default=45068,  help="dataset index (See dataset_id.json for detailed information)")
parser.add_argument("--seed",      type=int, default=1,      help="seed for dataset split (cross-validation)")
parser.add_argument("--savepath",  type=str, default=".",    help="path to save the results")
parser.add_argument("--n_trials",  type=int, default=100,    help="Number of optimization trials")
parser.add_argument("--metric",    type=str, default="l2",
                    choices=["l2", "l1", "cosine", "mahalanobis", "wasserstein", "kl"],
                    help="Distance metric for TabERA Retriever")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)   # ← MultiTab 원본과 동일한 위치

import optuna, torch, json, joblib, datetime, math
from libs.data import TabularDataset
from libs.eval import calculate_metric, is_study_todo, check_if_fname_exists_in_error
from libs.search_space import get_search_space, suggest_initial_trial, params_to_model_kwargs
from libs.supervised import TabERAWrapper
from libs.tabera import TabERA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)

# ─────────────────────────────────────────────────────────────
# 데이터셋 정보 로드  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

with open("dataset_id.json", "r") as file:
    data_info = json.load(file)

tasktype = data_info.get(str(args.openml_id))["tasktype"]
print(tasktype)

# ─────────────────────────────────────────────────────────────
# 저장 경로 설정  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

if not args.savepath.endswith("optim_logs"):
    savepath = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
else:
    savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)

fname = os.path.join(savepath, f"data={args.openml_id}..model=tabhera.pkl")

# ─────────────────────────────────────────────────────────────
# 중복 실행 방지  (MultiTab 원본과 동일)
# ─────────────────────────────────────────────────────────────

train = True
if os.path.exists(fname):
    study = joblib.load(fname)
    train = is_study_todo(study, tasktype)
else:
    study = (optuna.create_study(direction="minimize") if tasktype == "regression"
             else optuna.create_study(direction="maximize"))
    initial_trial = suggest_initial_trial()
    study.enqueue_trial(initial_trial)
    train = check_if_fname_exists_in_error(fname)

completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining_trials = max(0, args.n_trials - completed_trials_count)

# ─────────────────────────────────────────────────────────────
# 메인 최적화  (MultiTab 원본 구조 최대한 유지)
# ─────────────────────────────────────────────────────────────

if train:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── 헤더 출력 ──────────────────────────────────────────
    print("=" * 60)
    print("  TabERA  Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Dataset : {data_info[str(args.openml_id)]['fullname']} (id={args.openml_id})")
    print(f"  Task    : {tasktype}  |  Device : {device}")
    print(f"  Trials  : {completed_trials_count} done / {args.n_trials} total  ({remaining_trials} remaining)")
    print(f"  Save    : {fname}")
    print("=" * 60)

    # ── 데이터 로드  (MultiTab: TabularDataset + _indv_dataset) ──
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")
    print("-" * 60)

    # output_dim: n_classes 사용 (y가 1D이므로 shape[1] 불가)
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    # n_prototypes: 가설 기반 sqrt(N) 자동 설정
    n_proto_default = max(4, min(int(math.sqrt(len(y_train))), 64))
    print(f"  Auto n_prototypes: sqrt({len(y_train)}) = {n_proto_default}")

    # ── 자동 가설 생성 (컬럼 평균 기준) ──────────────────

    global best_so_far
    best_so_far = study.best_value if completed_trials_count > 0 else None

    # ── Objective  (MultiTab 원본 구조와 동일) ─────────────
    def objective(trial):
        global best_so_far
        params       = get_search_space(trial, num_features=X_train.size(1),
                                        data_id=args.openml_id, metric=args.metric)
        # sqrt(N) 기반 n_prototypes override (가설 ② 복잡도 개선)
        params["n_prototypes"] = n_proto_default
        trial.set_user_attr("n_prototypes_actual", n_proto_default)
        model_kwargs = params_to_model_kwargs(params, dataset.n_features, output_dim)

        model = TabERA(
            **model_kwargs,
            column_names=dataset.col_names,
            memory_size=min(int(len(y_train) * 2), 10_000),
        )

        # 시간 폭발 조합 사전 차단 (batch_size=128 + embed_dim=256 + k=32 → 평균 117s)
        _bs = params.get('batch_size', 256)
        _ed = params.get('embed_dim', 128)
        _k  = params.get('k', 16)
        if _bs == 128 and _ed == 256 and _k == 32:
            raise optuna.exceptions.TrialPruned()  # 예상 시간 >100s → skip

        wrapper = TabERAWrapper(model, params, tasktype,
                                  device=str(device), epochs=100, patience=20)
        wrapper._data_id = args.openml_id   # 에폭 tqdm에 data_id 표시
        wrapper.fit(X_train, y_train, X_val, y_val)

        preds_val  = wrapper.predict(X_val)
        preds_test = wrapper.predict(X_test)
        if tasktype == "regression":
            probs_val, probs_test = None, None
        else:
            probs_val  = wrapper.predict_proba(X_val)
            probs_test = wrapper.predict_proba(X_test)

        # regression: y_std 역정규화 후 metric 계산 (MultiTab 원본과 동일)
        if tasktype == "regression":
            val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, probs_test, tasktype, "test")
        else:
            val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

        for k, v in val_metrics.items():
            trial.set_user_attr(k, v)
        for k, v in test_metrics.items():
            trial.set_user_attr(k, v)

        # MultiTab 원본과 동일한 출력
        print(device, env_info, args.openml_id,
              data_info.get(str(args.openml_id))["name"], "tabhera", savepath)
        print(val_metrics)
        print(test_metrics)
        now      = datetime.datetime.now()
        duration = now - trial.datetime_start
        print(f"### Optimization time for trial {trial.number}: {duration.total_seconds():.0f} secs")
        trial.set_user_attr("training_time", duration.total_seconds())

        # 최적화 목표: regression → rmse_val 최소화 / classification → acc_val 최대화
        if tasktype == "regression":
            return val_metrics["rmse_val"]
        else:
            return val_metrics["acc_val"]

    # ── 콜백  (MultiTab 원본과 동일) ──────────────────────
    def stop_when_reached_optimal(study, trial):
        if study.best_value >= 1.0:
            study.stop()

    if tasktype == "regression":
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[lambda study, trial: joblib.dump(study, fname)])
    else:
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[stop_when_reached_optimal,
                                  lambda study, trial: joblib.dump(study, fname)])

    # ── 학습 시간 집계  (MultiTab 원본과 동일) ─────────────
    total_training_time = sum([
        trial.user_attrs.get("training_time", 0)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ])
    study.set_user_attr("total_training_time", total_training_time)

    # ── 결과 저장  (MultiTab 원본과 동일) ─────────────────
    print("#############################################")
    print(env_info)
    print(study.best_trial.user_attrs)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(savepath, f"data={args.openml_id}..model=tabhera.csv"), index=False)
    joblib.dump(study, fname)
    print(fname)
    print("#############################################")
    