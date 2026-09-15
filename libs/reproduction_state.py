"""Versioned inference checkpoints, including original retrieval caches.

Kept outside the frozen benchmark implementation hash. These checkpoints are
trusted local artifacts (torch.load uses pickle); never load untrusted files.
No refresh, training or prototype update is performed by restoration.
"""
import copy
import importlib.metadata
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

MEMORY_ATTRIBUTES = ("_cached_groups", "_cached_group_sizes", "_cached_extended",
                     "_cached_extended_sizes", "_group_round_unit",
                     "_vectorized_fallback", "_outlier_threshold", "n_size_buckets")
PROTOTYPE_ATTRIBUTES = ("sample_groups", "group_labels", "target_labels")


def copy_tree(value, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: copy_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(copy_tree(v, device) for v in value)
    return copy.deepcopy(value)


def current_environment():
    return {p: importlib.metadata.version(p) for p in ("torch", "numpy", "scikit-learn", "optuna")}


def get_cuda_runtime_info():
    """Best-effort audit metadata only; never part of provenance pass/fail."""
    info = dict(cuda_build_version=torch.version.cuda, cudnn_version=None,
                driver_version=None, gpu_name=None, compute_capability=None)
    try:
        if torch.backends.cudnn.is_available():
            info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception:
        pass
    if not torch.cuda.is_available():
        return info
    try:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
    except Exception:
        pass
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=5, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).splitlines()
        info["driver_version"] = next((line.strip() for line in lines if line.strip()), None)
    except Exception:
        pass
    return info


def compare_predictions(logits, predictions, ref_logits, ref_predictions, targets, atol=1e-6):
    arrays = [np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x)
              for x in (logits, predictions, ref_logits, ref_predictions, targets)]
    z, pred, rz, rp, y = arrays
    pred, rp, y = pred.reshape(-1), rp.reshape(-1), y.reshape(-1)
    if z.shape != rz.shape or pred.shape != rp.shape or pred.shape != y.shape:
        raise ValueError("Prediction/reference shape mismatch")
    if not len(y) or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Empty or nonfinite prediction/reference data")
    diff = np.abs(z.astype(np.float64) - rz.astype(np.float64))
    match = bool(np.array_equal(pred, rp))
    return dict(max_abs_logit_error=float(diff.max()), mean_abs_logit_error=float(diff.mean()),
                prediction_match_rate=float(np.mean(pred == rp)),
                accuracy_recomputed=float(np.mean(pred == y)),
                accuracy_reference=float(np.mean(rp == y)), atol=atol, rtol=0.0,
                passed=bool(diff.max() <= atol and match))


def snapshot(wrapper, dataset, identity, test_logits, test_preds):
    from libs.benchmark import data_signature, implementation_id
    if data_signature(dataset) != identity["contract"]["data"]:
        raise ValueError("Dataset signature differs from run identity")
    if implementation_id() != identity["contract"]["implementation"]:
        raise ValueError("Implementation differs from run identity")
    model = wrapper.model
    if model.training:
        raise ValueError("Checkpoint must be captured in eval mode")
    attributes = dict(tasktype=dataset.tasktype, n_classes=dataset.n_classes,
                      n_features=dataset.n_features, X_cat=list(dataset.X_cat),
                      X_num=list(dataset.X_num), X_cat_cardinality=list(dataset.X_cat_cardinality),
                      col_names=list(dataset.col_names), y_std=float(dataset.y_std))
    fs = model.feature_store
    return copy_tree(dict(
        schema_version=1, state_kind="original_post_fit_no_analysis_refresh",
        identity=identity, model_config=identity["contract"]["config"], selected_params=wrapper.params,
        state_dict=model.state_dict(), head_gamma=model.effective_gamma(),
        dataset_attributes=attributes, dataset_splits=dataset._indv_dataset(),
        preprocessing={name: getattr(dataset, name, None) for name in
                       ("quantile_transformer", "cat_category_names", "target_class_names")},
        memory_attributes={name: getattr(model.memory, name, None) for name in MEMORY_ATTRIBUTES},
        prototype_attributes={name: getattr(model.prototype_layer, name, None) for name in PROTOTYPE_ATTRIBUTES},
        feature_store=None if fs is None else dict(store=fs._store, ptr=fs._ptr,
                                                  filled=fs._filled, sample_ids=fs._sample_ids),
        test_logits=test_logits, test_preds=test_preds,
        last_epoch=wrapper.last_epoch, environment=current_environment(),
        refresh_on_best=wrapper.refresh_on_best,
    ))


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Checkpoint already exists: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            torch.save(payload, stream)
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        # Link creates the destination only if absent, unlike replace().
        os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def restore_checkpoint(path, device="cpu"):
    from libs.benchmark import build_wrapper, data_signature, implementation_id
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"schema_version", "identity", "model_config", "selected_params", "state_dict",
                "dataset_attributes", "dataset_splits", "memory_attributes", "prototype_attributes",
                "feature_store", "test_logits", "test_preds", "head_gamma", "last_epoch", "environment"}
    if not required <= payload.keys() or payload["schema_version"] != 1:
        raise ValueError("Incomplete or unsupported inference checkpoint")
    if payload["identity"]["contract"]["implementation"] != implementation_id():
        raise ValueError("Checkpoint implementation mismatch")
    if payload["environment"] != current_environment():
        raise ValueError("Checkpoint library environment mismatch")
    if payload["model_config"] != payload["identity"]["contract"]["config"]:
        raise ValueError("Checkpoint config/identity mismatch")
    if payload["selected_params"] != payload["identity"]["params"]:
        raise ValueError("Checkpoint parameters/identity mismatch")
    if set(payload["memory_attributes"]) != set(MEMORY_ATTRIBUTES):
        raise ValueError("Incomplete original retrieval state")
    if set(payload["prototype_attributes"]) != set(PROTOTYPE_ATTRIBUTES):
        raise ValueError("Incomplete prototype attributes")
    pairs = copy_tree(payload["dataset_splits"], device)
    dataset = SimpleNamespace(**payload["dataset_attributes"], _indv_dataset=lambda: pairs,
                              **payload.get("preprocessing", {}))
    if data_signature(dataset) != payload["identity"]["contract"]["data"]:
        raise ValueError("Checkpoint dataset signature mismatch")
    edges = payload["state_dict"].get("embedder.ple_edges")
    # Constructor consumes RNG; restoration must not change a later run's RNG.
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        wrapper = build_wrapper(dataset, payload["selected_params"], payload["model_config"], device,
                                num_bin_edges=None if edges is None else edges.to(device))
    model = wrapper.model
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.effective_gamma() != payload["head_gamma"]:
        raise ValueError("Checkpoint head scaling mismatch")
    for name, value in payload["memory_attributes"].items():
        setattr(model.memory, name, copy_tree(value, device))
    for name, value in payload["prototype_attributes"].items():
        setattr(model.prototype_layer, name, copy_tree(value, device))
    fs = payload["feature_store"]
    if (fs is None) != (model.feature_store is None):
        raise ValueError("Feature-store configuration mismatch")
    if fs is not None:
        model.feature_store._store = fs["store"].to(device)
        model.feature_store._sample_ids = fs["sample_ids"].to(device)
        model.feature_store._ptr = fs["ptr"]
        model.feature_store._filled = fs["filled"]
    wrapper.last_epoch = payload["last_epoch"]
    model.eval()
    return wrapper, dataset, payload
