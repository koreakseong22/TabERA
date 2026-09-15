"""Locked query-level metrics for TabERA's region/prediction/evidence structure."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from libs.eval import get_preds_and_probs
from libs.retrieval_audit import instrument_retrieval

METRIC_PROTOCOL_VERSION = "explanation-analysis-v1"


def _mean(values):
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def _hard(logits, tasktype):
    return get_preds_and_probs(logits, tasktype)[0].detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def _full_forward(model, X, batch_size):
    fields = ("logits", "context_emb", "correction", "centroid_id", "topk_idx",
              "neighbor_mask", "query_emb")
    parts = {field: [] for field in fields}
    for start in range(0, len(X), batch_size):
        out = model(X[start:start + batch_size])
        for field in fields:
            value = out.get(field)
            if value is None:
                raise ValueError(f"Model output lacks {field}")
            parts[field].append(value.detach().cpu())
    return {field: torch.cat(values) for field, values in parts.items()}


@torch.no_grad()
def compute_explanation_metrics(wrapper, dataset, stored_trace):
    model = wrapper.model
    model.eval()
    (train_x, train_y), _, (test_x, test_y) = dataset._indv_dataset()
    if dataset.tasktype not in ("binclass", "multiclass"):
        raise ValueError("Explanation metrics are currently defined for classification only")
    batch_size = int(wrapper.params.get("batch_size", 512))
    k = int(model.k)
    n_test, n_train = len(test_y), len(train_y)
    if int(model.memory.filled.item()) != n_train or model.memory.max_size != n_train:
        raise ValueError("Explanation metrics require the complete refreshed training memory")

    baseline, _, trace, instrumentation = instrument_retrieval(model, test_x, batch_size)
    if trace != stored_trace:
        raise ValueError("Stored retrieval trace differs from current instrumentation")
    if not instrumentation["metadata_off_on_equal"] or not instrumentation["model_state_match"]:
        raise ValueError("Retrieval instrumentation integrity failed")
    out = _full_forward(model, test_x, batch_size)
    for field in ("logits", "topk_idx", "neighbor_mask", "centroid_id", "query_emb"):
        if not torch.equal(out[field], baseline[field]):
            raise ValueError(f"Metric forward differs from audited retrieval output: {field}")

    # Exact predictive decomposition from the actual forward correction.
    context = out["context_emb"].to(model.dev_head.weight.device)
    correction = out["correction"].to(model.dev_head.weight.device)
    if getattr(model, "split_head", False):
        region_logits = model.dev_head(context)
        correction_logits = F.linear(correction, model.effective_W2())
    else:
        gamma = model.effective_gamma()
        region_logits = model.dev_head(context if gamma == 1.0 else gamma * context)
        correction_logits = F.linear(
            correction if gamma == 1.0 else gamma * correction, model.dev_head.weight)
    final_logits = out["logits"]
    decomposition_error = final_logits - (
        region_logits.detach().cpu() + correction_logits.detach().cpu())
    decomp_max = float(decomposition_error.abs().max())
    decomp_mean = float(decomposition_error.abs().mean())
    if decomp_max >= 1e-5:
        raise ValueError(f"Prediction decomposition error {decomp_max} exceeds 1e-5")

    y_train = train_y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    y_test = test_y.detach().cpu().numpy().reshape(-1).astype(np.int64)
    n_classes = int(dataset.n_classes)
    global_counts = np.bincount(y_train, minlength=n_classes)
    global_majority = int(global_counts.argmax())  # fixed smallest-index tie rule
    groups = model.prototype_layer.sample_groups
    if groups is None or len(groups) != model.prototype_layer.P:
        raise ValueError("Refreshed checkpoint lacks a complete region partition")
    sample_region = np.full(n_train, -1, dtype=np.int64)
    region_counts, region_majority, region_rows = [], [], []
    entropy_weighted = 0.0
    nonempty_entropies = []
    for region, members in enumerate(groups):
        members = np.asarray(members, dtype=np.int64)
        if len(members):
            sample_region[members] = region
            counts = np.bincount(y_train[members], minlength=n_classes)
            majority = int(counts.argmax())
            probability = counts[counts > 0] / len(members)
            entropy = float(-(probability * np.log(probability)).sum())
            normalized = entropy / math.log(n_classes)
            entropy_weighted += len(members) / n_train * normalized
            nonempty_entropies.append(normalized)
        else:
            counts = np.zeros(n_classes, dtype=np.int64)
            majority, normalized = global_majority, None
        region_counts.append(counts)
        region_majority.append(majority)
        row = dict(region=region, train_size=int(len(members)), majority_label=majority,
                   normalized_label_entropy=normalized)
        row.update({f"class_count_{label}": int(count) for label, count in enumerate(counts)})
        region_rows.append(row)
    all_members = np.concatenate([np.asarray(group, dtype=np.int64) for group in groups])
    if (np.any(sample_region < 0) or len(all_members) != n_train or
            len(np.unique(all_members)) != n_train):
        raise ValueError("Training region membership is not complete and unique")

    query_region = out["centroid_id"].numpy().astype(np.int64)
    region_majority_pred = np.asarray([region_majority[region] for region in query_region])
    empty_query = np.asarray([len(groups[region]) == 0 for region in query_region])
    final_pred = _hard(final_logits, dataset.tasktype)
    regional_pred = _hard(region_logits.detach().cpu(), dataset.tasktype)
    global_pred = np.full(n_test, global_majority, dtype=np.int64)
    baseline_correct = regional_pred == y_test
    final_correct = final_pred == y_test
    changed = regional_pred != final_pred
    corrected = ~baseline_correct & final_correct
    degraded = baseline_correct & ~final_correct
    regional_acc, final_acc = float(baseline_correct.mean()), float(final_correct.mean())
    corrected_rate, degraded_rate = float(corrected.mean()), float(degraded.mean())
    identity_error = abs((final_acc - regional_acc) - (corrected_rate - degraded_rate))
    if identity_error > 1e-12 or np.any(corrected & degraded):
        raise ValueError("Prediction-change accuracy identity failed")
    if changed.mean() + 1e-12 < corrected_rate + degraded_rate:
        raise ValueError("Prediction change rate is smaller than corrected + degraded")
    if dataset.tasktype == "binclass" and abs(float(changed.mean()) -
                                               corrected_rate - degraded_rate) > 1e-12:
        raise ValueError("Binary prediction-change identity failed")

    # Global comparator changes only the hard-assignment argument.
    global_slots = []
    for start in range(0, n_test, batch_size):
        query = out["query_emb"][start:start + batch_size].to(model.memory.keys.device)
        _, _, slots = model.memory.retrieve(query, k, hard_assignment=None, exclude_ids=None)
        global_slots.append(slots.detach().cpu())
    global_slots = torch.cat(global_slots).numpy()
    local_slots = out["topk_idx"].numpy()
    local_mask = out["neighbor_mask"].numpy().astype(bool)
    memory_ids = model.memory.sample_ids[:n_train].detach().cpu().numpy().astype(np.int64)
    memory_labels = model.memory.labels[:n_train].detach().cpu().numpy().astype(np.int64)
    query_norm = F.normalize(out["query_emb"], dim=-1)
    key_norm = F.normalize(model.memory.keys[:n_train].detach().cpu(), dim=-1)
    similarities = (query_norm @ key_norm.T).numpy()

    query_rows, local_neighbor_rows, global_neighbor_rows = [], [], []
    for index in range(n_test):
        local_valid_slots = [int(slot) for slot, valid in zip(local_slots[index], local_mask[index])
                             if valid and 0 <= int(slot) < n_train and memory_ids[int(slot)] >= 0]
        global_valid_slots = [int(slot) for slot in global_slots[index]
                              if 0 <= int(slot) < n_train and memory_ids[int(slot)] >= 0]
        local_ids = [int(memory_ids[slot]) for slot in local_valid_slots]
        global_ids = [int(memory_ids[slot]) for slot in global_valid_slots]
        local_full = len(local_ids) == k and len(set(local_ids)) == k
        global_full = len(global_ids) == k and len(set(global_ids)) == k
        neighbor_regions = [int(sample_region[sample_id]) for sample_id in local_ids]
        same_share = (float(np.mean(np.asarray(neighbor_regions) == query_region[index]))
                      if neighbor_regions else None)
        overlap_eligible = local_full and global_full
        if overlap_eligible:
            left, right = set(local_ids), set(global_ids)
            jaccard = len(left & right) / len(left | right)
        else:
            jaccard = None
        no_fallback = trace[index]["fallback_type"] == "none"
        label_eligible = (no_fallback and local_full and
                          all(region == query_region[index] for region in neighbor_regions) and
                          len(groups[query_region[index]]) > 0)
        local_agreement = (float(np.mean(memory_labels[local_valid_slots] == y_test[index]))
                           if local_valid_slots else None)
        global_agreement = (float(np.mean(memory_labels[global_valid_slots] == y_test[index]))
                            if global_valid_slots else None)
        expected = (float(region_counts[query_region[index]][y_test[index]] /
                          len(groups[query_region[index]])) if label_eligible else None)
        label_gain = local_agreement - expected if label_eligible else None
        # This diagnostic isolates the region restriction. A fallback query
        # has already widened or removed that restriction, so it cannot enter
        # the TabERA-same-region versus global-kNN comparison.
        global_delta_eligible = label_eligible and global_full
        global_delta = (local_agreement - global_agreement if global_delta_eligible else None)
        row = dict(
            query_index=index, true_label=int(y_test[index]), query_region=int(query_region[index]),
            region_train_size=len(groups[query_region[index]]),
            global_majority_label=global_majority,
            region_majority_label=int(region_majority_pred[index]),
            global_majority_correct=bool(global_pred[index] == y_test[index]),
            region_majority_correct=bool(region_majority_pred[index] == y_test[index]),
            regional_pred=int(regional_pred[index]), final_pred=int(final_pred[index]),
            regional_correct=bool(baseline_correct[index]), final_correct=bool(final_correct[index]),
            prediction_changed=bool(changed[index]), corrected=bool(corrected[index]),
            degraded=bool(degraded[index]), n_valid_neighbors=len(local_ids),
            full_k_retrieval=bool(local_full), same_region_share=same_share,
            fallback_invoked=bool(trace[index]["fallback_invoked"]),
            fallback_type=trace[index]["fallback_type"],
            global_knn_jaccard=jaccard, global_knn_overlap_eligible=bool(overlap_eligible),
            label_gain_eligible=bool(label_eligible),
            eligible_label_gain=bool(label_eligible),
            retrieved_label_agreement=local_agreement if label_eligible else None,
            expected_region_label_agreement=expected, label_agreement_gain=label_gain,
            global_label_delta_eligible=bool(global_delta_eligible),
            global_retrieved_label_agreement=(global_agreement if global_delta_eligible else None),
            label_agreement_delta_vs_global=global_delta,
        )
        if dataset.tasktype == "binclass":
            # A binary head emits one positive-class log-odds value. Calling
            # that value class 0 would make the query export misleading.
            row["regional_logit"] = float(region_logits[index, 0])
            row["correction_logit"] = float(correction_logits[index, 0])
            row["final_logit"] = float(final_logits[index, 0])
        else:
            for label in range(final_logits.shape[1]):
                row[f"regional_logit_class_{label}"] = float(region_logits[index, label])
                row[f"correction_logit_class_{label}"] = float(correction_logits[index, label])
                row[f"final_logit_class_{label}"] = float(final_logits[index, label])
        query_rows.append(row)
        for rank, slot in enumerate(local_valid_slots, 1):
            sample_id = int(memory_ids[slot])
            local_neighbor_rows.append(dict(
                query_index=index, rank=rank, memory_slot=slot, sample_id=sample_id,
                label=int(memory_labels[slot]), region=int(sample_region[sample_id]),
                cosine_similarity=float(similarities[index, slot]),
                same_region=bool(sample_region[sample_id] == query_region[index]),
                fallback_type=trace[index]["fallback_type"]))
        for rank, slot in enumerate(global_valid_slots, 1):
            sample_id = int(memory_ids[slot])
            global_neighbor_rows.append(dict(
                query_index=index, rank=rank, memory_slot=slot, sample_id=sample_id,
                label=int(memory_labels[slot]), region=int(sample_region[sample_id]),
                cosine_similarity=float(similarities[index, slot]),
                same_region=bool(sample_region[sample_id] == query_region[index])))

    sizes = np.asarray([len(group) for group in groups])
    nonempty_sizes = sizes[sizes > 0]
    summary = dict(
        global_majority_acc=float(np.mean(global_pred == y_test)),
        region_majority_acc=float(np.mean(region_majority_pred == y_test)),
        empty_region_query_rate=float(empty_query.mean()),
        normalized_region_entropy=float(entropy_weighted),
        normalized_region_entropy_unweighted_nonempty=float(np.mean(nonempty_entropies)),
        num_nonempty_regions=int(len(nonempty_sizes)),
        empty_training_region_fraction=float(np.mean(sizes == 0)),
        region_size_median=float(np.median(nonempty_sizes)),
        region_size_q1=float(np.quantile(nonempty_sizes, .25)),
        region_size_q3=float(np.quantile(nonempty_sizes, .75)),
        regional_baseline_acc=regional_acc, final_acc=final_acc,
        prediction_change_rate=float(changed.mean()), corrected_rate=corrected_rate,
        degraded_rate=degraded_rate, accuracy_delta_identity_error=float(identity_error),
        decomposition_max_abs_error=decomp_max, decomposition_mean_abs_error=decomp_mean,
        same_region_share=_mean(row["same_region_share"] for row in query_rows),
        fallback_rate=float(np.mean([row["fallback_invoked"] for row in query_rows])),
        cross_region_query_rate=float(np.mean([
            row["same_region_share"] is not None and row["same_region_share"] < 1
            for row in query_rows])),
        full_k_coverage=float(np.mean([row["full_k_retrieval"] for row in query_rows])),
        global_knn_jaccard=_mean(row["global_knn_jaccard"] for row in query_rows),
        global_knn_overlap_coverage=float(np.mean([
            row["global_knn_overlap_eligible"] for row in query_rows])),
        label_agreement_gain=_mean(row["label_agreement_gain"] for row in query_rows),
        label_gain_eligible_coverage=float(np.mean([
            row["label_gain_eligible"] for row in query_rows])),
        label_agreement_delta_vs_global=_mean(
            row["label_agreement_delta_vs_global"] for row in query_rows),
        global_label_delta_coverage=float(np.mean([
            row["global_label_delta_eligible"] for row in query_rows])),
        n_test=n_test, n_train=n_train, k=k,
    )
    return dict(summary=summary, query_rows=query_rows, region_rows=region_rows,
                local_neighbor_rows=local_neighbor_rows,
                global_neighbor_rows=global_neighbor_rows,
                instrumentation=instrumentation)
