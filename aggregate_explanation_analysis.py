"""Strict fold-then-dataset aggregation for explanation analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_explanation_manifest import sha256
from libs.explanation_metrics import METRIC_PROTOCOL_VERSION


TABLE2 = ["global_majority_acc", "region_majority_acc", "regional_baseline_acc",
          "final_acc", "corrected_rate", "degraded_rate"]
TABLE3 = ["same_region_share", "fallback_rate", "global_knn_jaccard",
          "global_knn_overlap_coverage", "label_agreement_gain",
          "label_gain_eligible_coverage"]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", default="analysis_results")
    p.add_argument("--manifest", default="analysis_results/manifest.json")
    return p


def run(args):
    root, manifest_path = Path(args.analysis_root).resolve(), Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_runs = {(row["dataset_id"], row["fold"]): row for row in manifest["runs"]}
    if len(manifest_runs) != len(manifest["runs"]):
        raise ValueError("Manifest contains duplicate dataset/fold entries")
    expected = set(manifest_runs)
    datasets = {ds for ds, _ in expected}
    if (len(expected) != 105 or len(datasets) != 21 or
            expected != {(ds, fold) for ds in datasets for fold in range(1, 6)}):
        raise ValueError("Aggregation requires the complete 21 x 5 manifest")
    fold_rows, problems, metric_protocols, metric_code_hashes = [], [], set(), set()
    for ds, fold in sorted(expected):
        run_dir = root / f"openml_{ds}" / f"fold_{fold}"
        try:
            audit = json.loads((run_dir / "audit_metrics.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            problems.append(dict(dataset_id=ds, fold=fold, status="missing", detail=str(exc)))
            continue
        if (audit.get("status") != "explanation_metrics_passed" or
                audit.get("eligible_for_aggregation") is not True):
            problems.append(dict(dataset_id=ds, fold=fold,
                                 status=audit.get("status", "incomplete")))
            continue
        integrity = audit.get("integrity") or {}
        metric_protocols.add(integrity.get("metric_protocol_version"))
        metric_code_hashes.add(integrity.get("metrics_code_sha256"))
        if (summary.get("dataset_id"), summary.get("fold")) != (ds, fold):
            problems.append(dict(dataset_id=ds, fold=fold, status="identity_mismatch"))
            continue
        expected_seed = int(manifest_runs[(ds, fold)]["train_seed"])
        if int(summary.get("train_seed", -1)) != expected_seed:
            problems.append(dict(dataset_id=ds, fold=fold, status="train_seed_mismatch",
                                 expected=expected_seed, observed=summary.get("train_seed")))
            continue
        fold_rows.append(summary)
    expected_metric_hash = sha256(Path(__file__).resolve().parent /
                                  "libs" / "explanation_metrics.py")
    if metric_protocols != {METRIC_PROTOCOL_VERSION}:
        problems.append(dict(status="metric_protocol_mismatch",
                             expected=METRIC_PROTOCOL_VERSION,
                             values=sorted(str(value) for value in metric_protocols)))
    if metric_code_hashes != {expected_metric_hash}:
        problems.append(dict(status="metric_code_mismatch", expected=expected_metric_hash,
                             values=sorted(str(value) for value in metric_code_hashes)))
    metric_names = sorted({key for row in fold_rows for key, value in row.items()
                           if isinstance(value, (int, float)) and key not in
                           {"dataset_id", "fold", "train_seed", "n_test", "n_train", "k"}})
    dataset_rows = []
    for ds in sorted(datasets):
        rows = [row for row in fold_rows if row["dataset_id"] == ds]
        item = dict(dataset_id=ds, n_folds=len(rows))
        for metric in metric_names:
            values = [row.get(metric) for row in rows]
            finite = [float(value) for value in values
                      if value is not None and np.isfinite(value)]
            item[metric] = float(np.mean(finite)) if len(finite) == 5 else None
            item[f"{metric}_defined_folds"] = len(finite)
        dataset_rows.append(item)
    overall = {}
    for metric in metric_names:
        values = [row.get(metric) for row in dataset_rows]
        finite = [float(value) for value in values
                  if value is not None and np.isfinite(value)]
        overall[metric] = float(np.mean(finite)) if len(finite) == 21 else None
        overall[f"{metric}_defined_datasets"] = len(finite)
    undefined_metrics = [metric for metric in metric_names
                         if overall[f"{metric}_defined_datasets"] != 21]
    complete = len(fold_rows) == len(expected) and not problems and not undefined_metrics
    out = root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / name for name in ("appendix_full.csv", "dataset_summary.csv",
                                        "table2.csv", "table3.csv", "aggregate_audit.json")]
    # Aggregate files are derived reports and are intentionally regenerated
    # after an incomplete batch is resumed. Per-run source artifacts remain
    # immutable and checksum-audited.
    pd.DataFrame(fold_rows).to_csv(targets[0], index=False)
    pd.DataFrame(dataset_rows).to_csv(targets[1], index=False)
    pd.DataFrame([{"metric": metric, "dataset_equal_mean": overall[metric],
                   "defined_datasets": overall[f"{metric}_defined_datasets"]}
                  for metric in TABLE2]).to_csv(targets[2], index=False)
    pd.DataFrame([{"metric": metric, "dataset_equal_mean": overall[metric],
                   "defined_datasets": overall[f"{metric}_defined_datasets"]}
                  for metric in TABLE3]).to_csv(targets[3], index=False)
    audit = dict(status="complete" if complete else "incomplete", expected_runs=len(expected),
                 valid_runs=len(fold_rows), problems=problems,
                 undefined_metrics=undefined_metrics, overall=overall)
    targets[4].write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "expected_runs", "valid_runs")}, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
