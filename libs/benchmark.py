"""Shared TabERA construction and the frozen MultiTab reproduction contract."""
from pathlib import Path
import hashlib
import json

import numpy as np
import optuna
import torch

from libs.data import get_batch_size
from libs.search_space import HPO_TRAINING_SCHEDULE, PROTOCOL_TAG, params_to_model_kwargs, study_pkl_tag
from libs.supervised import TabERAWrapper
from libs.tabera import TabERA
from libs.benchmark_config import FINAL_CONFIG

UPSTREAM_COMMIT = "b40c74d9be3e315b7e0ad5dfe475af57bddb7bab"


def implementation_id():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("benchmark.py", "benchmark_config.py", "data.py", "eval.py", "search_space.py", "supervised.py", "tabera.py", "prototypes.py"):
        h.update(name.encode())
        h.update((root / name).read_text(encoding="utf-8").encode())
    return h.hexdigest()


def data_signature(dataset):
    h = hashlib.sha256()
    for pair in dataset._indv_dataset():
        for tensor in pair:
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(str((arr.shape, str(arr.dtype))).encode())
            h.update(arr.tobytes())
    h.update(json.dumps([list(dataset.X_cat), list(dataset.X_num),
                         list(dataset.X_cat_cardinality)], default=int).encode())
    return h.hexdigest()


def contract(config, dataset):
    import importlib.metadata
    return dict(version=1, protocol=PROTOCOL_TAG, implementation=implementation_id(),
                data=data_signature(dataset), config=dict(config), schedule=HPO_TRAINING_SCHEDULE,
                environment={p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn", "optuna")})


def final_study_path(root, seed, data_id):
    keys = ("cat_combine", "num_embedding", "num_bins", "cat_embed_dim",
            "disable_dead_reinit", "correction_geometry", "head_input_scale",
            "beta_param", "tie_rule", "early_stop_metric")
    tag = study_pkl_tag(**{k: FINAL_CONFIG[k] for k in keys})
    return Path(root) / "optim_logs" / f"seed={seed}" / f"data={data_id}{tag}..model=tabera.pkl"


def select_trial(study, tasktype, mode="best", member=0):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError("HPO has no completed trials")
    expected_direction = "MINIMIZE" if tasktype == "regression" else "MAXIMIZE"
    if study.direction.name != expected_direction:
        raise ValueError("HPO objective direction does not match task")
    if len(completed) != 100 and not (tasktype != "regression" and study.best_value >= 1.0):
        raise ValueError(f"HPO incomplete or nonstandard budget: {len(completed)} COMPLETE trials; expected 100")
    if mode == "init":
        trial = study.trials[0]
        if trial.state != optuna.trial.TrialState.COMPLETE:
            raise ValueError("Initial trial did not complete")
        return trial
    if mode == "hyper":
        ranked = sorted(completed, key=lambda t: t.value, reverse=tasktype != "regression")
        if member >= len(ranked):
            raise ValueError(f"Not enough completed trials for hyper member {member}")
        return ranked[member]
    return study.best_trial


def restore_params(trial, n_train, config=FINAL_CONFIG):
    for key in ("correction_geometry", "head_input_scale", "beta_param", "tie_rule", "early_stop_metric"):
        if trial.user_attrs.get(key + "_actual") != config[key]:
            raise ValueError(f"Trial {trial.number}: incompatible or missing {key}_actual")
    params = dict(trial.params)
    for name, expected in (("n_prototypes", int(n_train ** .5)), ("batch_size", get_batch_size(n_train))):
        if trial.user_attrs.get(name + "_actual") != expected:
            raise ValueError(f"Trial {trial.number}: {name} does not match final protocol ({expected})")
        params[name] = expected
    params.update({k: config[k] for k in ("correction_geometry", "head_input_scale", "beta_param")})
    return params


def build_wrapper(dataset, params, config, device, num_bin_edges=None):
    task = dataset.tasktype
    output_dim = dataset.n_classes if task == "multiclass" else 1
    train_x, train_y = dataset._indv_dataset()[0]
    if num_bin_edges is None and config["num_embedding"] == "ple" and len(dataset.X_num):
        q = torch.linspace(0, 1, config["num_bins"] + 1, device=train_x.device)
        num_bin_edges = torch.quantile(train_x[:, dataset.X_num], q, dim=0).T.contiguous()
    kwargs = params_to_model_kwargs(params, dataset.n_features, output_dim)
    model = TabERA(**kwargs, column_names=dataset.col_names, tasktype=task,
                   n_classes=dataset.n_classes, memory_size=len(train_y),
                   exclude_self_retrieval=not config["allow_self_retrieval"],
                   **({"dead_reinit_patience": 10**9} if config["disable_dead_reinit"] else {}),
                   cat_col_idx=list(dataset.X_cat), num_col_idx=list(dataset.X_num),
                   cat_cardinalities=list(dataset.X_cat_cardinality),
                   cat_combine=config["cat_combine"], cat_embed_dim=config["cat_embed_dim"],
                   num_embedding=config["num_embedding"], num_bin_edges=num_bin_edges)
    return TabERAWrapper(model, params, task, device=str(device), **HPO_TRAINING_SCHEDULE,
                         early_stop_metric=config["early_stop_metric"], tie_rule=config["tie_rule"])


def result_path(root, seed, data_id, mode="best", member=0):
    deep = member if mode == "deep" else 0
    hyper = member if mode == "hyper" else 0
    return (Path(root) / "reproduce_logs" / f"seed={seed}" / f"data={data_id}" /
            f"model=tabera..init_hps={mode == 'init'}..deep={deep}..hyper={hyper}.npy")


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            np.save(stream, payload)
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
