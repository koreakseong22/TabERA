"""Inventory trusted benchmark result files without extracting archives or training.

Saved logits/predictions are copied losslessly into per-run NPZ references. Fold
and training RNG seed remain separate. Checkpoint availability is unknown until
a matching state has been verified; result files alone are not checkpoints.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def result_bytes(source):
    source = Path(source)
    if source.is_dir():
        for path in sorted(source.rglob("*.npy")):
            yield str(path.relative_to(source)), path.read_bytes()
    else:
        with tarfile.open(source, "r:*") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".npy"):
                    yield member.name, archive.extractfile(member).read()


def build_manifest(source, output, expected_runs=105):
    records = []
    seen = set()
    for name, raw in result_bytes(source):
        result = np.load(io.BytesIO(raw), allow_pickle=True).item()
        identity = result["identity"]
        key = (int(identity["dataset_id"]), int(identity["fold"]))
        if key in seen:
            raise ValueError(f"Duplicate dataset/fold {key}; select one benchmark arm/mode")
        seen.add(key)
        if result.get("probability_representation") != "logits":
            raise ValueError(f"Expected saved logits: {name}")
        logits, preds = np.asarray(result["Probability"]), np.asarray(result["Prediction"])
        if logits.ndim not in (1, 2) or len(logits) != len(preds) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid predictions/logits: {name}")
        if identity["tasktype"] not in ("binclass", "multiclass"):
            raise ValueError(f"Classification-only analysis: {name}")
        records.append((key, name, raw, result, logits, preds))
    if len(records) != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs; found {len(records)}")
    if expected_runs == 105:
        datasets = {d for d, _ in seen}
        if len(datasets) != 21 or seen != {(d, s) for d in datasets for s in range(1, 6)}:
            raise ValueError("Expected a complete 21 datasets x folds 1..5 grid")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for (ds, fold), name, raw, result, logits, preds in sorted(records):
        identity = result["identity"]
        reference = Path(f"openml_{ds}/fold_{fold}/benchmark_reference.npz")
        (output / reference).parent.mkdir(parents=True)
        np.savez_compressed(output / reference, logits=logits, predictions=preds)
        cfg = identity["contract"]["config"]
        rows.append(dict(
            dataset_id=ds, fold=fold, train_seed=identity["train_seed"],
            tasktype=identity["tasktype"], geometry=cfg["correction_geometry"],
            head_input_scale=cfg["head_input_scale"], model_config=cfg,
            selected_params=identity["params"], identity=identity,
            implementation_hash=identity["contract"]["implementation"],
            data_signature=identity["contract"]["data"],
            environment=result["environment"], performance=result["Performance"],
            source_member=name, source_result_sha256=hashlib.sha256(raw).hexdigest(),
            reference_path=reference.as_posix(), reference_sha256=sha256(output / reference),
            n_test=len(preds), checkpoint_available=False,
            checkpoint_status="not_in_source_results; external checkpoints not verified",
        ))
    manifest = dict(schema_version=1, source=str(Path(source).resolve()),
                    source_sha256=sha256(source) if Path(source).is_file() else None,
                    aggregation="fold mean, then equal-weight dataset mean",
                    memory_protocol="frozen final encoder and routing; original state retained",
                    runs=rows)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Trusted result directory or tar archive")
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--expected-runs", type=int, default=105)
    args = parser.parse_args()
    manifest = build_manifest(args.source, args.output, args.expected_runs)
    print(f"{args.output}/manifest.json: {len(manifest['runs'])} runs")
