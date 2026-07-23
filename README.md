# TabERA

**Tabular Explainable Retrieval Architecture**

A retrieval-augmented model for tabular data. Instead of predicting from a
single learned representation, TabERA routes each input to a local region of
feature space, retrieves similar training examples from within that region,
and lets the retrieved evidence inform the prediction — while exposing the
routing and retrieval steps themselves as a built-in, architecturally
grounded explanation.

---

## Core idea

Most tabular deep learning models produce a prediction from a black-box
embedding with no intermediate structure a person can inspect. TabERA takes
a different approach, organized around three questions a prediction should
be able to answer:

1. **Which local region of feature space does this sample belong to?**
2. **Which stored training examples are most relevant to it?**
3. **Which individual features drove the final prediction?**

The first two are answered *architecturally* — they are direct, reproducible
readouts of computation that actually happens in the forward pass, not a
separate post-hoc explainer bolted on afterward. The third is answered by a
standard post-hoc attribution method (SHAP), since feature-level attribution
doesn't have a natural architectural analogue the way routing and retrieval
do.

---

## Architecture

```
Tabular Input X
      │
      ▼
Feature Embedding  (categorical → one-hot, numerical → PiecewiseLinearEmbeddings)
      │
      ▼
Embedder MLP
      │
      ▼
query_emb
      │
      ▼
Centroid Routing            ──►  hard_assignment ──► routing_probs ──► context_emb
(cosine similarity to a                                                 (auxiliary —
 learned set of M centroids,                                            used for
 STE hard assignment)                                                   analysis and
      │                                                                 ablation, not
      ▼                                                                 for prediction)
Local Retrieval
(top-k neighbors, restricted
 to the assigned centroid's
 own region)
      │
      ▼
cosine(query, neighbors) → softmax → weighted aggregation
      │
      ▼
agg_emb
      │
      ├─────────────┐
      │             │
query_emb ───────────┘
      │
      ▼
Residual Fusion:  z = query_emb + β · agg_emb        (β learned)
      │
      ▼
Prediction Head
      │
      ▼
ŷ
```

**Embed.** Categorical features are one-hot encoded — a raw integer code has
no natural ordering, so no learned embedding is used for it. Numerical
features go through `PiecewiseLinearEmbeddings`: quantile bin boundaries are
computed once from the training data, and each column gets its own trainable
per-bin embedding. A periodic (sin/cos) alternative, `PLR(lite)`, is
available via `--num_embedding plr_lite`.

**Route.** `CentroidLayer` assigns each sample to one of `M` centroids via a
straight-through hard-argmax over cosine similarity. Centroids are trained
with a VQ-VAE-style dual loss — a commitment term pulling queries toward
their centroid, a codebook term pulling centroids toward their queries —
kept at unit norm, and protected from permanent dead centroids by a
periodic reset (a centroid with no assignments for several epochs is
reinitialized near a real embedding). `M` is set automatically as
`√N_train` rather than tuned.

**Retrieve.** `MemoryBank` performs k-NN search restricted to the sample's
assigned centroid — retrieval never crosses centroid boundaries except as a
fallback when a region is smaller than `k`. Similarity is computed in
L2-normalized (cosine) space by default, which keeps the resulting evidence
distribution meaningfully spread across several neighbors rather than
collapsing onto a single nearest match as embedding norms grow during
training. Each retrieved value is `label_emb + T(query − neighbor)` — a
learned label embedding plus a learned offset term; how the two combine is
configurable via `value_mode` (`default`, `balanced`, `offset_normalized`,
`sum_normalized`).

**Fuse & predict.** `query_emb` and `agg_emb` are combined as
`z = query_emb + β · agg_emb`, with `β` a learned scalar, and `z` is passed
to the prediction head. `context_emb` — the continuous representation of the
routing outcome — is computed as part of routing but is not included in this
path by default; it remains available for analysis and ablation (see
*Explanations* below).

**Attribute.** SHAP (`KernelExplainer`) estimates each feature's marginal
contribution to the prediction, independent of the routing/retrieval steps
above. Being a black-box method, it needs neither a gradient nor a
continuous path from a baseline to the input — which matters here, since
TabERA's hard routing and one-hot categorical encoding would break a
gradient-based attribution method.

---

## What routing and retrieval actually do

Centroids are not trained with any direct supervision toward class labels —
the loss only ever asks a centroid to be a good anchor for nearby queries.
Empirically, this still produces a routing scheme that captures a
substantial amount of label structure, and this label structure — not the
sharpness of the retrieval itself — is what tracks with prediction quality:

- **Retrieval geometry is real and heterogeneous.** Different centroids
  retrieve neighbors with very different similarity profiles — some regions
  return a tight, consistent set of examples across queries (high margin
  between the closest and farthest retrieved neighbor, low effective
  neighbor count); others return a much more diffuse, sample-dependent set.
  This isn't routing noise: within a centroid, how consistently the same
  neighbors get retrieved correlates strongly with how peaked that
  centroid's similarity scores are.
- **How sharp a centroid's retrieval is does not predict how accurate
  predictions from it are.** A centroid that always retrieves the same
  tight neighbor set is not, on that basis alone, a better or worse
  "expert" than one with diffuse retrieval — accuracy doesn't track
  retrieval sharpness.
- **What does track accuracy is how much the routing captures about the
  label.** Measuring the mutual information between which centroid a
  sample is routed to and its true label (normalized by label entropy)
  correlates strongly with downstream accuracy across datasets — far more
  reliably than how many centroids are actively used, or how sharply any
  one of them retrieves.

The practical reading: centroids function less like specialized predictors
and more like an unsupervised, label-aware partitioning of feature space —
the routing step organizes *where* to look, and that organization, not the
retrieval mechanics on top of it, is where most of the useful structure
lives.

---

## Explanations

Every prediction can be traced back through the levels of the forward pass
that produced it (`--explain`):

| Level | Type | Content |
|---|---|---|
| Prototype Assignment | Architectural | Which centroid the sample was routed to, and how confidently. |
| Retrieved Neighbors | Architectural readout | Which real training examples the model retrieved as similar. |
| Retrieval Signal Magnitude | Architectural readout | How large the retrieved signal is relative to the query. |
| Feature Attribution | Post-hoc (SHAP) | Which individual features moved the prediction, and by how much. |

**Prototype Assignment** — the assigned centroid and its confidence, the
runner-up centroids (each with their own confidence and label
distribution), and what the centroid *represents*: majority label name plus
count/share (e.g. `"good" 27/47 (57%)`), with a second label shown if it
covers ≥20%. Distinctive features are ranked by how unusual this centroid
is *relative to the other centroids* (robust z-score, median/MAD), not
against the whole dataset — numeric values are shown in real units,
categorical features as real category names with their in-group share.

**Retrieved Neighbors** — the actual `k` retrieved neighbors and their
evidence weights, the same values used in the forward pass, not a
separately computed explanation. Neighbors with near-zero weight are
dropped. Each neighbor's displayed features are the ones closest to the
query (smallest gap first) — this is what explains *why* it's similar, as
opposed to just listing the neighbor's own largest values. The evidence
weight reflects retrieval similarity; treating it as a verified causal
contribution to the final prediction is a stronger claim than the weight
itself supports (see *Diagnostics* for the ablations that test this
directly).

**Retrieval Signal Magnitude** — `‖query_emb‖`, `‖agg_emb‖`, and `β`, for
comparing how large the retrieved evidence is relative to the query itself
before fusion.

**Feature Attribution** — SHAP values from `shap.KernelExplainer`,
estimated by perturbing each feature against a background sample from the
training set, independent of the routing/retrieval steps above.

```
① Prototype Assignment   → Centroid_12 (confidence 10.4%, margin +2.0% over runner-up)
                            Label distribution: "good" 76/134 (57%), "bad" 58/134 (43%)
② Retrieved Neighbors    → #0 (3.7%): credit_history=existing paid, purpose=radio/tv
                            #1 (3.4%): credit_history=existing paid, savings_status=<100
③ Retrieval Signal        → ‖query‖=4.56  ‖agg‖=6.40  β=1.00
④ Feature Attribution    → volatile_acidity 15.1%
```

Routing confidence is reported as a raw softmax probability over all
centroids, so it should be read relative to the uniform baseline
(`1/M`) rather than against an absolute scale — with `M=28` centroids,
`10%` is roughly `2.8×` the uniform rate, not a low value in absolute terms.

---

## Validation

Rather than assume trained structure is automatically meaningful, several
`--ablation` modes check specific properties directly on an already-trained
model, without retraining:

- **`dual_space_faithfulness`** — do Prototype Assignment / Retrieved
  Neighbors actually reflect live model state, rather than a stale or
  decorative readout? Checks index integrity (does the recorded group
  assignment match the model's live routing, exactly, not statistically),
  value reproducibility (with `--refresh_on_best`, does a stored key's
  similarity to a fresh re-encoding of its raw features converge to ≈1.0),
  and group separation (do centroids correspond to statistically distinct
  regions of feature space — ANOVA/chi-square, Bonferroni-corrected).
- **`rank_correlation`** — is SHAP's feature ranking at least consistent
  with an independent, low-fidelity signal (single-feature ablation)? A low
  correlation isn't automatically a sign SHAP is wrong — it can mean SHAP
  is capturing an interaction the ablation signal structurally can't see,
  which `interaction_check` (below) tests for directly.
- **`interaction_check`** — is meaningful feature interaction even present
  in a given dataset (does perturbing two features together move the
  prediction more than perturbing them separately)? Establishes whether a
  SHAP–ablation disagreement is attributable to interaction before assuming
  it is.
- **`centroid_geometry`** — is centroid routing distinguishable from a
  random, untrained one at all? Also used automatically during HPO: a
  continuous penalty (no fixed threshold) nudges the search away from
  hyperparameter regions that converge to random-or-worse routing
  structure, without constraining the search space directly.
- **`centroid_representativeness`** — does each individual centroid
  actually represent the members assigned to it (purity against a
  global-majority baseline, cohesion relative to the run's other
  centroids), independent of its size?

Because `MemoryBank` persists across epochs rather than resetting each
epoch, a training sample can in principle retrieve a stored copy of itself
from an earlier epoch as a "neighbor" — this is excluded by default
(`exclude_self_retrieval`); `--allow_self_retrieval` reproduces the
un-excluded behavior for comparison.

---

## Diagnostics

Beyond `--explain`, several flags export structured data for offline
analysis rather than printing per-sample explanations:

- **`--export_centroid_retrieval_behavior`** — for every test sample: which
  centroid it was routed to, routing confidence, the retrieved neighbor
  indices, the raw similarity geometry (top-1 similarity, margin over the
  farthest retrieved neighbor, standard deviation across retrieved
  similarities), the resulting evidence distribution (entropy, effective
  neighbor count, top-1 weight), and the prediction outcome (correct/error).
  Saved as a single `.npz`, keyed by sample so it can be grouped by centroid
  for post-hoc analysis.
- **`--log_centroid_label_mi_trajectory`** — tracks how much label
  information routing captures (see *What routing and retrieval actually
  do* above) epoch by epoch on the validation set, rather than only at the
  end of training.
- **`--log_fusion_trajectory`** — tracks `β` and branch representation norms
  epoch by epoch.
- **`--branch_information` / `--branch_contribution` / `--gradient_attribution`
  / `--head_sensitivity`** — a family of one-shot, no-retraining diagnostics
  on an already-trained model: how informative each branch's representation
  is, how much it contributes at the point the head consumes it, how much
  gradient it receives, and how sensitive the output is to zeroing, shuffling,
  or rescaling it.
- **`--ablation`** — applies a controlled perturbation at inference time on
  an already-trained model (e.g. `agg_emb_zero`, `agg_emb_shuffle`,
  `context_emb_zero`, `query_emb_shuffle`, `random_neighbor`) and reports
  the resulting change in accuracy, for directly testing whether a given
  branch or retrieval step is load-bearing rather than inferring it
  indirectly.

---

## HPO parameters (searched via Optuna)

| Parameter | Range | Role |
|---|---|---|
| `embed_dim` | {64, 128, 256} | Embedding dimension |
| `k` | {4, 8, 16, 32, 48, 64} | Retrieval budget (neighbors per query) |
| `embedder_layers` | 1–4 | Embedder MLP depth |
| `dropout` | 0.0–0.5 | — |
| `loss_diversity` | 5e-2–5e-1 | Centroid separation penalty |
| `loss_commitment` | 1e-2–1e-1 | Query → centroid pull |
| `loss_codebook` | 1e-2–1e-1 | Centroid → query pull |
| `lr` | 1e-4–1e-2 | — |
| `weight_decay` | 1e-6–1e-2 | — |
| `plr_freq_scale` * | 0.01–100 (log) | PLR(lite) frequency init scale |
| `plr_n_frequencies` * | 8–96 | PLR(lite) frequencies per column |
| `plr_out_dim` * | {4, 8, 16, 32} | PLR(lite) output dim per column |

\* only searched with `--num_embedding plr_lite`.

`n_prototypes` is fixed at `√N_train` rather than searched, so it can be
described as a controlled partition granularity rather than a free
hyperparameter. `batch_size` is fixed at 256. `evidence_metric` defaults to
`cosine`. `fusion_mode` defaults to `residual` with `context_emb` excluded
from the prediction path (`use_context_emb=False`) — this is the
architecture described above; `concat` (all three representations
concatenated into the head) remains available for comparison.

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py         # TabERA model, MemoryBank, TabularEmbedder
│   ├── prototypes.py     # CentroidLayer (routing)
│   ├── evidence.py       # AttentionAggregator (retrieval + fusion)
│   ├── supervised.py     # Training loop wrapper
│   ├── search_space.py   # Optuna HPO space
│   ├── eval.py           # Metrics, prediction utilities
│   └── data.py           # OpenML dataset loading
├── optimize.py            # HPO entry point
├── reproduce.py           # Train / evaluate / explain / diagnose entry point
└── requirements.txt
```

Run `python optimize.py --help` and `python reproduce.py --help` for the
full CLI reference.

---

## References

- Gorishniy et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- Gorishniy et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*.
- Ye et al. (2024). Revisiting Nearest Neighbor for Tabular Data (ModernNCA). *arXiv:2407.03257*.
- Snell, Swersky & Zemel (2017). Prototypical Networks for Few-shot Learning. *NeurIPS*.
- Zhang et al. (2019). AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations. *CVPR*.
- van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS*.
- Wang et al. (2018). CosFace: Large Margin Cosine Loss for Deep Face Recognition. *CVPR*.
- Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR*.
- Dhariwal et al. (2020). Jukebox: A Generative Model for Music. *arXiv:2005.00341*.
- Zeghidour et al. (2021). SoundStream: An End-to-End Neural Audio Codec. *arXiv:2107.03312*.
- Bengio et al. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*.
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.
- McElfresh et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS*.