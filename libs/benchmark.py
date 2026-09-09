"""Shared TabERA construction and the frozen MultiTab reproduction contract."""
from pathlib import Path
import hashlib
import json

import numpy as np
import optuna
import torch

from libs.data import get_batch_size
from libs.search_space import (HPO_TRAINING_SCHEDULE, PROTOCOL_TAG, RECIPE_TAG,
                               params_to_model_kwargs, study_pkl_tag, resolve_dynamics)
from libs.supervised import TabERAWrapper
from libs.tabera import TabERA
from libs.benchmark_config import FINAL_CONFIG

# ⚠ The commit that actually produced the baseline numbers TabERA is compared
#   against -- not the upstream release tag. Provenance, verified rather than
#   assumed: every payload in the baseline archive
#   (multitab/final_reproduce_23datasets_no_tabr.tar.gz -- 2442 archived
#   result payloads spanning 23 datasets, 5 seeds and 12 models) records
#     implementation_id = f686321c5091515d075f58d4992b1fca0ece532f8e9884e3a88f3f167a6dd92d
#   and re-hashing multitab's tree with its own libs/runtime.py
#   implementation_id() over every commit reproduces that digest at exactly one
#   commit, a0c075e.
#
#   This pin was previously b40c74d9 (the earlier upstream state), which is
#   materially different code: there EarlyStopping increments the patience
#   counter on the first epoch (two separate `if`s, so a run whose validation
#   loss never improves after epoch 1 stops one epoch earlier), and binclass
#   probabilities were softmax over a single logit. Both were fixed in
#   2f39ad3, an ancestor of a0c075e -- so the baselines were generated with
#   the corrected behaviour, and matching TabERA to b40c74d9 would have meant
#   matching code that produced no published number here.
#
#   Consequence for TabERA: libs/supervised.py EarlyStopping.step() -- which
#   resets the counter on the first epoch -- already matches a0c075e. See
#   tests/test_benchmark.py::test_early_stopping_counter_matches_upstream_multitab,
#   which transcribes this commit's on_epoch_end.
UPSTREAM_COMMIT = "a0c075e"
BASELINE_IMPLEMENTATION_ID = "f686321c5091515d075f58d4992b1fca0ece532f8e9884e3a88f3f167a6dd92d"


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
    return dict(version=1, protocol=PROTOCOL_TAG, recipe=RECIPE_TAG, implementation=implementation_id(),
                data=data_signature(dataset), config=dict(config), schedule=HPO_TRAINING_SCHEDULE,
                environment={p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn", "optuna")})


def arm_config(disable_dead_reinit=False, early_stop_metric=None):
    """The frozen final configuration, or one of its sanctioned ablation arms.

    An arm changes exactly one *structural* variable of FINAL_CONFIG and is
    treated like a different geometry: its own HPO study (``..nodr`` /
    ``..esm=NAME`` in the study filename), its own contract (``config``
    differs), and its own result files (``arm_tag``). Nothing about the main
    arm's names changes, so every existing study and result keeps resolving
    under its original path.

    Arms:
      disable_dead_reinit   dead-prototype recovery off
      early_stop_metric     which validation criterion drives patience.
                            val_loss is the main, MultiTab-matched protocol
                            (FINAL_CONFIG) and returns the terminal model;
                            accuracy is the earlier TabERA rule (best-val-acc
                            checkpoint restore), kept as an ablation arm
    """
    cfg = dict(FINAL_CONFIG, disable_dead_reinit=bool(disable_dead_reinit))
    if early_stop_metric is not None:
        cfg["early_stop_metric"] = early_stop_metric
    return cfg


def arm_tag(config):
    """Filename infix that separates an ablation arm's outputs from the main arm's."""
    tag = "..nodr" if config.get("disable_dead_reinit") else ""
    if config.get("early_stop_metric", FINAL_CONFIG["early_stop_metric"]) != FINAL_CONFIG["early_stop_metric"]:
        tag += f"..esm={config['early_stop_metric']}"
    return tag


def is_main_arm(config):
    return dict(config) == dict(FINAL_CONFIG)


def final_study_path(root, seed, data_id, config=FINAL_CONFIG):
    keys = ("cat_combine", "num_embedding", "num_bins", "cat_embed_dim",
            "disable_dead_reinit", "correction_geometry", "head_input_scale",
            "beta_param", "tie_rule", "early_stop_metric")
    tag = study_pkl_tag(**{k: config[k] for k in keys})
    return Path(root) / "optim_logs" / f"seed={seed}" / f"data={data_id}{tag}..model=tabera.pkl"


def select_trial(study, tasktype, mode="best", member=0):
    if study.user_attrs.get("validation_only", False):
        raise ValueError("Validation-only pilot study cannot be used as a final benchmark study")
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


def restore_params(trial, n_train, config=FINAL_CONFIG, hpo_config=None):
    """Recover the trial's hyperparameters and check they were searched under
    ``hpo_config`` (default: ``config``). The two differ only in the fixed-HP
    ablation, where the main arm's study is reused for the nodr model."""
    hpo_config = hpo_config or config
    for key in ("correction_geometry", "head_input_scale", "beta_param", "tie_rule", "early_stop_metric"):
        if trial.user_attrs.get(key + "_actual") != hpo_config[key]:
            raise ValueError(f"Trial {trial.number}: incompatible or missing {key}_actual")
    # Recorded by optimize.py since the nodr arm exists; older studies lack it
    # and fall back to the study filename (..nodr) and the contract's config.
    if "disable_dead_reinit_actual" in trial.user_attrs and \
            bool(trial.user_attrs["disable_dead_reinit_actual"]) != bool(hpo_config["disable_dead_reinit"]):
        raise ValueError(f"Trial {trial.number}: disable_dead_reinit_actual does not match the study's arm")
    params = dict(trial.params)
    for name, expected in (("n_prototypes", int(n_train ** .5)), ("batch_size", get_batch_size(n_train))):
        if trial.user_attrs.get(name + "_actual") != expected:
            raise ValueError(f"Trial {trial.number}: {name} does not match final protocol ({expected})")
        params[name] = expected
    params.update({k: config[k] for k in ("correction_geometry", "head_input_scale", "beta_param")})
    expected_dynamics = resolve_dynamics(params, n_train)
    recorded_recipe = (trial.user_attrs.get("benchmark_contract") or {}).get("recipe")
    if recorded_recipe == RECIPE_TAG:
        for name in ("beta_lr_mult", "ema_timescale"):
            if name not in params:
                raise ValueError(f"Trial {trial.number}: missing recipe parameter {name}")
        for name, expected in expected_dynamics.items():
            actual = trial.user_attrs.get(name)
            matches = (actual == expected if isinstance(expected, str) else
                       isinstance(actual, (int, float)) and
                       np.isclose(actual, expected, rtol=1e-12, atol=0.0))
            if not matches:
                raise ValueError(f"Trial {trial.number}: {name} differs from resolved recipe")
    return params


def build_wrapper(dataset, params, config, device, num_bin_edges=None):
    task = dataset.tasktype
    output_dim = dataset.n_classes if task == "multiclass" else 1
    train_x, train_y = dataset._indv_dataset()[0]
    if num_bin_edges is None and config["num_embedding"] == "ple" and len(dataset.X_num):
        q = torch.linspace(0, 1, config["num_bins"] + 1, device=train_x.device)
        num_bin_edges = torch.quantile(train_x[:, dataset.X_num], q, dim=0).T.contiguous()
    kwargs = params_to_model_kwargs(params, dataset.n_features, output_dim)
    dynamics = resolve_dynamics(params, len(train_y))
    kwargs["ema_decay"] = dynamics["ema_decay_actual"]
    model = TabERA(**kwargs, column_names=dataset.col_names, tasktype=task,
                   n_classes=dataset.n_classes, memory_size=len(train_y),
                   exclude_self_retrieval=not config["allow_self_retrieval"],
                   **({"dead_reinit_patience": 10**9} if config["disable_dead_reinit"] else {}),
                   cat_col_idx=list(dataset.X_cat), num_col_idx=list(dataset.X_num),
                   cat_cardinalities=list(dataset.X_cat_cardinality),
                   cat_combine=config["cat_combine"], cat_embed_dim=config["cat_embed_dim"],
                   num_embedding=config["num_embedding"], num_bin_edges=num_bin_edges)
    wrapper = TabERAWrapper(model, params, task, device=str(device), **HPO_TRAINING_SCHEDULE,
                            beta_lr_mult=dynamics["beta_lr_mult_requested"],
                            early_stop_metric=config["early_stop_metric"], tie_rule=config["tie_rule"])
    wrapper.dynamics_provenance = dict(dynamics, ema_decay_actual=model.prototype_layer.ema_decay,
                                       beta_lr_actual=params["lr"] * wrapper.beta_lr_mult)
    return wrapper


def training_diagnostics(wrapper):
    """Small JSON-safe summary and epoch records; no per-batch gradient reads."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        return value
    hist = wrapper.regroup_history
    last = hist[-1] if hist else {}
    active = [r["active_ratio"] for r in hist if "active_ratio" in r]
    total = sum(r.get("reinit_count", 0) for r in hist)
    summary = dict(best_metric_epoch=wrapper.best_metric_epoch, last_epoch=wrapper.last_epoch,
                   beta_final=float(wrapper.model.effective_beta().detach().mean()),
                   reinit_total=total, reinit_per_epoch=total / len(hist) if hist else None,
                   active_ratio_final=last.get("active_ratio"),
                   active_ratio_mean=float(np.mean(active)) if active else None,
                   active_ratio_std=float(np.std(active)) if active else None,
                   n_eff_entropy=last.get("n_eff_entropy"))
    summary["epoch_history"] = hist
    summary["beta_epoch_history"] = wrapper.beta_epoch_history
    churn = [r["assign_change_rate_aligned"] for r in hist
             if "assign_change_rate_aligned" in r and np.isfinite(r["assign_change_rate_aligned"])]
    summary["routing_churn_mean"] = float(np.mean(churn)) if churn else None
    summary["routing_churn_final"] = last.get("assign_change_rate_aligned")
    return clean(summary)


def script_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract_diff(recorded, expected):
    """Human-readable list of the contract fields that differ.

    "code, data or configuration differs" on its own sends the reader digging
    through a study pickle to find out which of the three it was; the answer
    decides whether HPO really has to be rerun.
    """
    out = []
    for key in sorted(set(recorded) | set(expected)):
        a, b = recorded.get(key), expected.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if a.get(sub) != b.get(sub):
                    out.append(f"{key}.{sub}: study={a.get(sub)!r} now={b.get(sub)!r}")
        else:
            fmt = (lambda v: f"{str(v)[:12]}..." if key in ("implementation", "data") else repr(v))
            out.append(f"{key}: study={fmt(a)} now={fmt(b)}")
    return out


def result_path(root, seed, data_id, mode="best", member=0, config=FINAL_CONFIG, hpo_source="own"):
    deep = member if mode == "deep" else 0
    hyper = member if mode == "hyper" else 0
    # ⚠ The arm -- and, for the fixed-HP ablation, where its hyperparameters
    #   came from -- go into the filename. Otherwise a --disable_dead_reinit
    #   run would land on the main arm's result path: not silently (the
    #   identity check refuses), but it could never coexist in one savepath.
    hpo = "" if hpo_source == "own" else f"..hpo={hpo_source}"
    return (Path(root) / "reproduce_logs" / f"seed={seed}" / f"data={data_id}" /
            f"model=tabera{RECIPE_TAG}{arm_tag(config)}{hpo}..init_hps={mode == 'init'}..deep={deep}..hyper={hyper}.npy")


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
