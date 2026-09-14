"""Audit the completed fixed-HP Unit Tangent versus Tangent experiment."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from analyze_split_head_auc import pair_auc, write_json


def audit(root: Path, split_root: Path):
    runs = [json.loads(p.read_text(encoding="utf-8"))
            for p in root.glob("data=*_seed=*_geometry=*.json")]
    lookup = {(r["identity"]["data"], r["identity"]["seed"],
               r["identity"]["geometry"]): r for r in runs}
    expected = {(d, s, g) for d in (51, 1067, 31) for s in (1, 2, 3)
                for g in ("unit_tangent", "tangent")}
    assert set(lookup) == expected, (len(lookup), expected - set(lookup))

    os.environ.setdefault("OPENML_CACHE_DIR", str(Path("data_cache/openml").resolve()))
    from libs.data import TabularDataset

    rows = []
    max_shared_reproduction_error = 0.0
    max_decomposition_error = 0.0
    for data, seed in sorted({(d, s) for d, s, _ in expected}):
        unit = lookup[data, seed, "unit_tangent"]
        tangent = lookup[data, seed, "tangent"]
        assert unit["initial"]["base_state_sha256"] == tangent["initial"]["base_state_sha256"]
        assert unit["identity"]["source_identity"] == tangent["identity"]["source_identity"]
        assert unit["initial"]["gamma"] == tangent["initial"]["gamma"]
        assert unit["initial"]["beta"] == tangent["initial"]["beta"]
        for record in (unit, tangent):
            assert record["terminal_epoch"] == record["training"]["last_epoch"]
            assert record["best_val_loss_epoch"] == record["training"]["best_metric_epoch"]
            stem = root / (f"data={data}_seed={seed}_geometry="
                           f"{record['identity']['geometry']}_terminal.pt")
            assert stem.is_file()

        dataset = TabularDataset(data, "binclass", device="cpu", seed=seed)
        y = dataset._indv_dataset()[2][1].numpy().reshape(-1)
        unit_regions = np.asarray(unit["performance"]["test"]["region"])
        for record in (unit, tangent):
            score = np.asarray(record["performance"]["test"]["logits"])
            parts = pair_auc(y, score, unit_regions)
            reconstructed = (parts["same_pair_fraction"] * parts["same_auc"] +
                             (1.0 - parts["same_pair_fraction"]) * parts["cross_auc"])
            auc = record["performance"]["test"]["auc"]
            error = abs(reconstructed - auc)
            max_decomposition_error = max(max_decomposition_error, error)
            assert error < 1e-12, (data, seed, record["identity"]["geometry"], error)
            rows.append(dict(data=data, seed=seed,
                             geometry=record["identity"]["geometry"], **parts))

        # The Unit arm must reproduce the independently executed Shared control.
        shared_path = split_root / f"data={data}_seed={seed}_head=shared.json"
        shared = json.loads(shared_path.read_text(encoding="utf-8"))
        unit_logits = np.asarray(unit["performance"]["test"]["logits"])
        shared_logits = np.asarray(shared["performance"]["test"]["logits"])
        error = float(np.max(np.abs(unit_logits - shared_logits)))
        max_shared_reproduction_error = max(max_shared_reproduction_error, error)
        assert error == 0.0, (data, seed, error)

    with (root / "unit_partition_diagnostics.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    means = []
    for data in (31, 51, 1067):
        gains = {"same_auc": [], "cross_auc": [], "same_pair_fraction": []}
        for seed in (1, 2, 3):
            u = next(r for r in rows if (r["data"], r["seed"], r["geometry"]) ==
                     (data, seed, "unit_tangent"))
            t = next(r for r in rows if (r["data"], r["seed"], r["geometry"]) ==
                     (data, seed, "tangent"))
            gains["same_auc"].append(t["same_auc"] - u["same_auc"])
            gains["cross_auc"].append(t["cross_auc"] - u["cross_auc"])
            gains["same_pair_fraction"].append(u["same_pair_fraction"])
        means.append(dict(data=data,
                          same_auc_gain=float(np.mean(gains["same_auc"])),
                          cross_auc_gain=float(np.mean(gains["cross_auc"])),
                          same_pair_fraction=float(np.mean(gains["same_pair_fraction"]))))

    result = dict(runs=len(runs),
                  max_fixed_partition_decomposition_error=max_decomposition_error,
                  max_unit_shared_reproduction_error=max_shared_reproduction_error,
                  unit_partition_dataset_means=means)
    write_json(root / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("diagnostics/e2e_tangent_auc_v2"))
    parser.add_argument("--split-output", type=Path,
                        default=Path("diagnostics/e2e_split_auc_v2"))
    args = parser.parse_args()
    audit(args.output, args.split_output)
