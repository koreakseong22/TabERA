"""Read-only branch instrumentation for the frozen benchmark retrieval code.

This module stays outside benchmark.implementation_id(). It wraps the existing
MemoryBank.retrieve method, calls that exact bound method once, and records the
conditions that select its none/adjacent/global branches. It never substitutes
a retrieval implementation.
"""
from collections import Counter
import types

import torch


def classify_retrieval_branches(memory, batch_size, k, hard_assignment, exclude_ids=None):
    """Record the branch selected by MemoryBank.retrieve for each query."""
    n = int(memory.filled.item())
    cached = getattr(memory, "_cached_groups", None)
    if hard_assignment is None or cached is None or n < k:
        return [dict(fallback_type="global", fallback_invoked=True,
                     candidate_count_initial=None, candidate_count_after_adjacent=None,
                     candidate_count_global=n) for _ in range(batch_size)]
    assignment = hard_assignment.detach().long().to(memory._cached_group_sizes.device)
    sizes = memory._cached_group_sizes[assignment]
    extended = getattr(memory, "_cached_extended", None)
    extended_sizes = getattr(memory, "_cached_extended_sizes", None)
    minimum = k + (1 if exclude_ids is not None else 0)
    records = []
    for pos in range(batch_size):
        initial = int(sizes[pos].item())
        if initial >= minimum:
            kind, after = "none", initial
        elif extended is not None and extended_sizes is not None:
            after = int(extended_sizes[assignment[pos]].item())
            kind = "adjacent_region" if after >= k else "global"
        else:
            kind, after = "global", None
        records.append(dict(
            fallback_type=kind, fallback_invoked=kind != "none",
            candidate_count_initial=initial, candidate_count_after_adjacent=after,
            candidate_count_global=n,
        ))
    return records


@torch.no_grad()
def instrument_retrieval(model, X, batch_size):
    """Run metadata OFF and ON and require identical model outputs."""
    if model.training:
        raise ValueError("Retrieval instrumentation requires model.eval()")
    fields = ("logits", "topk_idx", "neighbor_mask", "centroid_id", "query_emb")

    def forward_all():
        parts = {field: [] for field in fields}
        for start in range(0, len(X), batch_size):
            out = model(X[start:start + batch_size])
            for field in fields:
                value = out.get(field)
                if value is None:
                    raise ValueError(f"Model output lacks {field}")
                parts[field].append(value.detach().cpu())
        return {field: torch.cat(values) for field, values in parts.items()}

    state_before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    baseline = forward_all()
    records = []
    original = model.memory.retrieve

    def wrapped(_memory, query, k, hard_assignment=None, sample_groups=None, exclude_ids=None):
        batch_records = classify_retrieval_branches(
            _memory, len(query), k, hard_assignment, exclude_ids)
        result = original(query, k, hard_assignment=hard_assignment,
                          sample_groups=sample_groups, exclude_ids=exclude_ids)
        records.extend(batch_records)
        return result

    model.memory.retrieve = types.MethodType(wrapped, model.memory)
    try:
        instrumented = forward_all()
    finally:
        del model.memory.retrieve
    if len(records) == 0 and len(X):
        # TabERA.forward skips retrieve entirely when memory has fewer than k.
        records = [dict(fallback_type="not_retrieved", fallback_invoked=False,
                        candidate_count_initial=None, candidate_count_after_adjacent=None,
                        candidate_count_global=int(model.memory.filled.item()))
                   for _ in range(len(X))]
    if len(records) != len(X):
        raise ValueError(f"Retrieval trace has {len(records)} rows for {len(X)} queries")

    equality = {field: bool(torch.equal(baseline[field], instrumented[field])) for field in fields}
    def exactly_equal(left, right):
        right = right.detach().cpu()
        if torch.is_floating_point(left) or torch.is_complex(left):
            # centroid_labels uses NaN as an unset sentinel. NaN != NaN under
            # torch.equal even when neither tensor changed.
            return bool(torch.allclose(left, right, rtol=0, atol=0, equal_nan=True))
        return bool(torch.equal(left, right))

    state_match = all(exactly_equal(value, model.state_dict()[name])
                      for name, value in state_before.items())
    for index, record in enumerate(records):
        record.update(query_index=index,
                      query_region=int(instrumented["centroid_id"][index].item()))
    counts = Counter(record["fallback_type"] for record in records)
    audit = dict(
        metadata_off_on_equal=bool(all(equality.values())), field_equality=equality,
        model_state_match=bool(state_match), n_queries=len(X),
        branch_counts={name: int(counts.get(name, 0)) for name in
                       ("none", "adjacent_region", "global", "not_retrieved")},
        fallback_rate=float(sum(record["fallback_invoked"] for record in records) / len(records))
                      if records else None,
    )
    return baseline, instrumented, records, audit
