# Explanation analysis: locked protocol and execution gates

## Status

Implemented: immutable benchmark manifest/references, provenance preflight,
manifest-driven reproduction without HPO, original-state checkpoint saving and
strict restoration, separate checkpoint-roundtrip and benchmark-logit audits.

Implemented: final-encoder memory reconstruction and its separate checkpoint,
parameter/centroid/sample-ID checks, prediction invariance and refreshed-state
round-trip audit.

Implemented: read-only retrieval branch instrumentation, metadata-off/on output
identity checks and per-query branch trace export.

The three retrieval-instrumentation pilots passed: OpenML 31/folds 1 and 2
and OpenML 10/fold 1. Implemented next: per-run region structure, prediction
decomposition, global-kNN control, label-gain metrics and auditable query,
region and neighbor exports. Pending: metric pilots, 105-run execution,
aggregation and paper tables.

The manifest inventories benchmark results; `checkpoint_available=false` means
no verified checkpoint was supplied with that inventory. Per-run preflight and
audit files record subsequently created checkpoints without rewriting the
source manifest. Old analysis checkpoints must not be substituted by filename.

Each pilot writes separate `preflight.json`, `audit_train.json`, and
`audit_restore.json` files. The `execution_mode` field identifies the invocation.
Early provenance/contract failures also go to that invocation's file, so a
restore audit cannot overwrite the training or standalone preflight audit.
Re-running the same mode replaces that mode's audit; these files are not a
timestamped history. Check both train and independent restore audits for each
of the three pilots before proceeding to memory refresh implementation.

## Commands

Run from the repository root, using the original benchmark environment.
Paths below use `python`; locally use `venv/Scripts/python.exe`.

```sh
python build_explanation_manifest.py \
  --source lab_results/tabera_tangent_unit_reproduce_21x5.tar.gz \
  --output analysis_results

# Default action performs provenance preflight only; no training.
python reproduce_with_checkpoint.py --dataset-id 31 --fold 1

# Explicit new training run using manifest parameters and train_seed.
python reproduce_with_checkpoint.py --dataset-id 31 --fold 1 --train

# No training, no network dataset loading: audit the saved transformed splits.
python reproduce_with_checkpoint.py --dataset-id 31 --fold 1 --restore-only

# Only after train and restore audits pass every reproduction gate.
python refresh_explanation_checkpoint.py --dataset-id 31 --fold 1

# Only after the refreshed checkpoint passes its complete audit.
python audit_retrieval_instrumentation.py --dataset-id 31 --fold 1

# Only after the retrieval instrumentation gate passes.
python analyze_explanation_structure.py --dataset-id 31 --fold 1

# After all three metric pilots pass, resume the full manifest pipeline.
python run_explanation_analysis_batch.py --gpus 0 1

# This returns an incomplete status unless all 105 metric audits pass.
python aggregate_explanation_analysis.py
```

Manifest generation refuses an existing destination. Checkpoint saving refuses
replacement. The original `.npy` benchmark results are never modified. Copy
the manifest directory with its reference NPZ files when moving to the server.
Only use trusted source archives and checkpoints: source `.npy` dictionaries
and `.pt` checkpoints use pickle. This is an inference checkpoint, not an
optimizer/RNG checkpoint for resuming interrupted training.

## Reproduction gates

1. Verify source/reference hashes, implementation, library versions, data
   signature, config, training schedule and fold/train-seed identity.
2. Run OpenML 31/fold 1 first, then 31/fold 2, then 10/fold 1.
3. For each new fit save state and compare disk-restored logits/predictions to
   both the pre-save model and the original benchmark reference.
4. Each comparison requires finite equal-shaped arrays, maximum absolute logit
   error <= 1e-6 (rtol=0), and 100% hard-prediction agreement. Saved benchmark
   accuracy must also agree with reference predictions and restored targets.
5. Failing runs are retained for audit, never marked eligible. Similar accuracy
   does not establish model identity. Do not loosen tolerances after viewing
   results without documenting a new protocol and investigating the cause.

Matching library versions alone does not guarantee deterministic re-training;
the numerical gate is mandatory. `runtime` records the CUDA build version,
cuDNN version, NVIDIA driver version, GPU name and compute capability. These
are informational metadata, not additional strict provenance criteria: the
original benchmark has no corresponding runtime reference. Unavailable values
are null; an unavailable or timed-out nvidia-smi query does not fail a run.
The strict package comparison remains torch/numpy/scikit-learn/optuna. Failed
preflight does not establish that dataset signatures match: data loading has
not yet happened when library provenance already fails.

## Checkpoint content

Store config, selected parameters, identity, all model buffers/parameters,
original memory caches (including adjacent-region caches and outlier threshold),
sample groups/labels, feature store/sample IDs, PLE edges, head scale,
transformed train/validation/test splits, preprocessing metadata, saved logits
and predictions. The ordered transformed test split and data signature bind the
saved predictions to query order. Later exports should include original dataset
row indices as well as split-local query indices.

Restore with strict state_dict loading and original caches, without calling
regroup, refresh, centroid reinitialization or training. Additional unsupported
constructor/runtime changes must cause an implementation/config audit failure.

## Final-model memory protocol

Preserve the original checkpoint first. Freeze model parameters and centroids;
encode every training sample once in eval mode; route using the actual forward
routing function; rebuild memory and search caches. Check training sample IDs
are unique and complete, and labels/features align with them.

Memory slots are canonicalized into training-row order after the final encoder
pass: slot/sample ID `i` contains training row `i`, its label, raw transformed
features and embedding. This makes every later neighbor export auditable.

Do not use a separate argmax implementation for region assignment: the actual
routing module uses topk, whose tie selection can differ. Do not use regroup_update,
which can update training counters or reinitialize centroids.

Assert model parameters and centroids are unchanged and test logits/predictions
are invariant before/after refresh. Save the refreshed state separately. Use
this single final-model partition for both region and retrieval statistics,
including the qualitative example. Describe refresh as post-training inference
preparation in the paper; never imply the original training cache was identical.
The command requires both `audit_train.json` and `audit_restore.json` to pass
all locked reproduction gates and match the original checkpoint checksum. It
writes `checkpoint_refreshed.pt` and `audit_memory_refresh.json` without
replacing the original checkpoint or reproduction audits.

## Locked metric definitions

- Majority tie: smallest class index. Empty region: global-majority prediction;
  report empty-region query rate.
- Normalized region entropy is the training-sample-weighted mean of each
  nonempty region's entropy divided by log(number of classes). Also save the
  unweighted nonempty-region mean. Region size median/IQR excludes empty
  regions; save the fraction of empty training regions separately.
- Hard prediction: `libs.eval.get_preds_and_probs` (binary sigmoid > 0.5).
- Tangent/unit: independently calculate z_region = head(c), z_corr = W*d from
  actual forward correction. Assert max decomposition error < 1e-5.
- Corrected/degraded/change rates: denominator is all test queries. Assert
  final_acc - baseline_acc = corrected - degraded and change >= corrected +
  degraded (binary equality).
- Fallback: record actual query-specific branches, not inferred outcomes;
  types none/adjacent_region/global/not_retrieved. Distinguish an intentionally
  global comparator from TabERA fallback. Metadata on/off must preserve IDs.
- Global kNN: identical keys/query representation, cosine similarity, sample
  IDs, exclusion and k (currently 8); only region restriction changes.
- Primary Jaccard: both sets contain k valid unique training cases. Save
  coverage. Compare sets of training IDs, not unverified slot indices.
- Label gain: no fallback, k valid unique same-region neighbors, valid region
  distribution. Mean(query-label agreement - region proportion of true label).
  Random sampling is unnecessary. Save eligible coverage; zero eligible means
  undefined gain (JSON null / tabular NaN), never zero.
- Label agreement versus global kNN is saved as a secondary diagnostic. It
  isolates the effect of the region constraint, whereas the primary
  random-within-region expectation measures query specificity inside a region.
  It uses the primary label-gain eligibility plus a valid unique global-kNN set,
  so adjacent/global fallback queries are excluded. These two quantities answer
  different questions and are not substituted for one another.

Retrieval instrumentation wraps and calls the frozen benchmark's bound
`MemoryBank.retrieve` method; it does not replace its search implementation or
modify `libs/tabera.py`. It records the exact control conditions selecting the
branch and verifies metadata OFF/ON equality for logits, neighbor slots, valid
mask, region assignment and query embedding. It also verifies the complete
model state is unchanged. Outputs are `audit_retrieval.json` and the per-query
`retrieval_trace.json`.

## Later outputs and aggregation

Each run: query_metrics, region_stats, neighbors_tabera, neighbors_global,
summary and audit. Retain regional/final logits, neighbor labels/regions/IDs and
region label counts, not only summary metrics. Each record must carry run and
split-local query IDs; preserve source row IDs once available.

Complete all pilot gates before the 105-run batch. Aggregate folds within each
dataset, then weight all 21 datasets equally. Never pool queries. Missing runs
or undefined metrics require explicit counts and an incomplete-result status;
do not silently use nanmean to label a partial aggregate as the full benchmark.

Main Table 2: global-majority, region-majority, regional-baseline and final
accuracy; corrected and degraded rate.

Main Table 3: same-region share, fallback invocation rate, global-kNN Jaccard,
global-kNN overlap coverage, label agreement gain and label-gain eligible
coverage. The secondary global-kNN label delta and its coverage remain appendix
diagnostics. Dataset/fold variation and other diagnostics belong in the appendix.
