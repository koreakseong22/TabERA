# TabERA

**Tabular Explainable Retrieval Architecture**

Centroid-conditioned hierarchical retrieval with example-based explanations for tabular data.

---

## Overview

TabERA is a retrieval-augmented tabular model that organizes data through learnable centroids and retrieves relevant neighbors within centroid groups. Beyond prediction, the forward pass itself produces **example-based explanations** — which group a sample belongs to, and which training samples it is compared against — information that post-hoc methods (SHAP, LIME, Integrated Gradients) cannot produce for *any* model, since those methods only operate on feature-level input–output relationships.

```
Query → Embedding → Centroid Routing (macro) → Group-constrained KNN (micro) → Prediction
                          ↓                              ↓                         ↓
                    "Which group?"              "Which neighbors?"          "Which features?"
                    (architectural)              (architectural)             (post-hoc, IG)
```

### Three-level explanation chain

| Level | Module | Type | Explains |
|-------|--------|------|----------|
| ① Group context | CentroidLayer (`centroid_x`) | **Architectural** (intrinsic forward-pass output) | "This sample belongs to the high-alcohol, low-pH group" |
| ② Neighbor evidence | MemoryBank + AttentionAggregator (`evidence_w`) | **Architectural** (intrinsic forward-pass output) | "Neighbor #1 (training sample #142) contributes 42%" |
| ③ Feature attribution | Integrated Gradients (Sundararajan et al., 2017) | Post-hoc, standard | "`volatile_acidity` has the largest attribution toward ŷ" |

①② are *case-based / example-based* explanations: they are read directly off intermediate activations of the forward pass (which centroid a sample is routed to, which training examples are retrieved as neighbors). No post-hoc method — gradient-based or perturbation-based — can produce this kind of information for an arbitrary model, because it requires an architecture that explicitly organizes and retrieves training examples at inference time.

③ is feature-level attribution computed via Integrated Gradients on the trained model. We do not claim architectural novelty for ③ itself; we show empirically (§Faithfulness below) that it is substantially more faithful to the model's actual decision boundary than a learned feature-projection alternative we initially explored, and competitive with SHAP.

### How ① and ② work

**① Group context.** Every input `X` is embedded into `query_emb ∈ ℝ^D` and routed (via STE, see below) to exactly one of `P` learnable centroids — `hard_assignment ∈ {1..P}`. This is not a soft mixture or attention weighting: the model commits to a single discrete group per sample, the same way a classifier commits to a single predicted class. Because the assignment is discrete and is read directly from `routing_probs`/`hard_assignment` (no extra computation), explanation ① is literally *"which of the P groups did the routing layer pick for this sample"* — a fact about the forward pass, not an estimate. The group is made human-readable via `centroid_x[p]`: the real training sample whose embedding is closest to `centroid_emb[p]` (the *medoid*, recomputed every epoch). So ① reads as *"this sample was routed to the same group as training sample #87 (alcohol=10.24, pH=3.31, ...)"* — an actual data point, not a synthetic average.

**② Neighbor evidence.** Once a sample is routed to group `p`, `MemoryBank.retrieve` performs a K-nearest-neighbor search **restricted to the training samples routed to group `p`** (group-constrained KNN — this is what makes ② depend on ①, not just on raw embedding distance over the whole training set). If group `p` has fewer than `K` members, the search expands to the embedding-adjacent centroid group(s) (cross-group fallback) rather than silently falling back to a global search — this keeps the *meaning* of "neighbor within your group" intact even for small groups. The retrieved neighbors' similarities are turned into `evidence_w ∈ ℝ^K` via the TabR-style softmax (`AttentionAggregator`, see formula below), and `evidence_w` is the *same* tensor used to compute `agg_emb` (the aggregation that feeds the prediction head). So ② is *"these are the `evidence_w`-weighted training samples that the retrieval step actually aggregated"* — again read directly from the forward pass, not reconstructed afterward.

Both ① and ② therefore answer a question no feature-attribution method (post-hoc or architectural) can answer for an arbitrary model: *"which other data points is this prediction like?"* — because answering that requires the model to maintain and query a structured index over training examples at inference time, which only retrieval-augmented architectures (TabR, TabERA) do.

---

## Architecture

TabERA processes a batch `X ∈ ℝ^(N×F)` through four stages, each producing both a prediction component and (for stages 1–2) an explanation artifact. Throughout, `D` is the embedding dimension, `P` the number of centroids, `K` the number of retrieved neighbors per sample, and `F` the number of input features.

1. **Embed.** `TabularEmbedder` (a stack of `L` residual MLP blocks) maps `X` to `query_emb ∈ ℝ^(N×D)`. This is the only place the raw features `X` are consumed for the prediction path; everything downstream operates in embedding space (except explanation ③, which differentiates *back* through this embedder).
2. **Route (macro, → explanation ①).** `CentroidLayer` compares `query_emb` against `P` learnable centroid embeddings `C_emb ∈ ℝ^(P×D)` and commits each sample to exactly one group via STE hard-argmax routing. This yields `hard_assignment ∈ {1..P}^N`, a `context_emb ∈ ℝ^(N×D)` (the assigned centroid's embedding, concatenated into the head input later), and — for explanation — `centroid_x ∈ ℝ^(P×F)`, the medoid (real training sample) of each group.
3. **Retrieve & aggregate (micro, → explanation ②).** `MemoryBank.retrieve` performs a group-constrained KNN: for each sample, it searches only among training points with the same `hard_assignment` (with cross-group fallback if the group is too small), returning `K` neighbor embeddings `nk ∈ ℝ^(N×K×D)` and labels. `AttentionAggregator` then computes TabR-style similarity weights `evidence_w ∈ ℝ^(N×K)` and aggregates the neighbors' (label, embedding-difference) values into `agg_emb ∈ ℝ^(N×D)`. In parallel, `FeatureCrossAttention` computes a feature-interaction signal `feature_imp ∈ ℝ^(N×K×F)`, which is summarized and gated together with `agg_emb` into `fused_agg ∈ ℝ^(N×D)` (Gated Fusion — a predictive auxiliary path, see below; *not* the source of explanation ③).
4. **Predict.** The concatenation `[query_emb ‖ context_emb ‖ fused_agg] ∈ ℝ^(N×3D)` is passed through an MLP head to produce `ŷ`.

Explanation ③ is *not* part of this forward pass — it is computed afterward by differentiating `ŷ` with respect to `X` (Integrated Gradients), independent of stages 2–3's internal representations.

### Forward flow

```
X ∈ ℝ^(N×F)
  ↓ TabularEmbedder (ResidualMLP × L)
query_emb ∈ ℝ^D
  ├── CentroidLayer
  │     C_emb ∈ ℝ^(P×D)  — learnable, gradient + STE routing
  │     C_x   ∈ ℝ^(P×F)  — medoid only (real sample, no gradient), original-space (explanation ①)
  │     → context_emb, hard_assignment, routing_probs
  │
  ├── MemoryBank.retrieve (group-constrained KNN)
  │     Cross-group fallback for small groups (adjacent centroid expansion)
  │     → nk (B,K,D), neighbor_labels (B,K)            (explanation ②)
  │
  └── AttentionAggregator
        ├─ TabR L2 similarity: evidence_w = softmax(2·⟨q,k⟩ - ‖q‖² - ‖k‖²)  (explanation ②)
        ├─ Value construction: label_emb + T(query - neighbor)
        ├─ agg_emb = weighted sum of values
        ├─ FeatureCrossAttention → feature_imp (B,K,F)  (auxiliary signal, NOT ③)
        └─ Gated Fusion: fused_agg = gate·feat_emb + (1-gate)·agg_emb
              ↓
[query_emb ‖ context_emb ‖ fused_agg] ∈ ℝ^(3D) → MLP Head → ŷ

(explanation ③, computed separately, not part of the forward pass above)
  ŷ ──IG (Integrated Gradients, ∂ŷ/∂X · (X - X̄))──> per-feature attribution
```

### Key design decisions

**Dual-Space Centroid.** Each centroid maintains two representations: `centroid_emb` in embedding space (learnable, used for routing and retrieval) and `centroid_x` in original feature space (medoid-updated each epoch — the real training sample closest to `centroid_emb[p]` in embedding space, used for human-readable explanations). This separation ensures that explanation ① shows an actual training example ("alcohol=10.24") rather than an opaque embedding coordinate or a synthetic EMA average.

**STE Routing.** Hard assignment via Straight-Through Estimator (Bengio et al., 2013; VQ-VAE, van den Oord et al., 2017): `hard_assignment = argmax_p(-‖query_emb - C_emb[p]‖²)`, a discrete one-hot vector. Forward pass uses this discrete argmax for crisp group boundaries (this is what explanation ① reports); backward pass replaces the gradient of the argmax with the gradient of `softmax(-‖query_emb - C_emb[p]‖²))`, so `C_emb` remains trainable despite the discrete forward.

**Cross-group Fallback.** When a centroid group has fewer than `K` members, `MemoryBank.retrieve` expands the candidate pool to include training samples from the nearest adjacent centroid group(s) in embedding space (ranked by `‖C_emb[p] - C_emb[p']‖`), rather than falling back to a global (group-unconstrained) search. This preserves the semantics of explanation ② — "neighbors from your group (or its closest neighboring group)" — even when a group is small, instead of silently degrading ② into "neighbors from anywhere."

**Gated Fusion (Predictive Auxiliary Path).** `FeatureCrossAttention` produces a feature-interaction signal (`feature_imp` → `feat_emb`) that is mixed into the neighbor aggregation via a learned gate (`fused_agg = gate·feat_emb + (1-gate)·agg_emb`). This branch is validated as a **predictive** component — removing it (`--ablation no_feat_path`) measurably degrades performance (e.g., Δlogloss = +0.91 on `ada_agnostic`, id=1043). It is *not* the source of explanation ③; `feature_imp` is an internal signal that improves ŷ, but its per-feature breakdown does not correlate with each feature's actual effect on ŷ (Spearman ρ ≈ 0.08–0.10 against perturbation-based ground truth, statistically indistinguishable from random). Explanation ③ is computed independently via Integrated Gradients (see Faithfulness below).

### Faithfulness of explanation ③

We measure faithfulness as the Spearman rank correlation between a feature-attribution method's ranking and the ranking by *actual* effect on ŷ (each feature individually perturbed to its training-set mean; ranked by |Δlogits|).

**Single-dataset breakdown** (seed=1, n=100 test samples where applicable):

| Dataset | F | TabERA (IG) ρ | SHAP (KernelExplainer) ρ | Random ρ |
|---|---|---|---|---|
| `qsar-biodeg` (id=1494) | 41 | 0.713 (p<0.001) | **0.902** (p<0.001) | 0.051 |
| `ada_agnostic` (id=1043) | 48 | **0.936** (p<0.001) | 0.841 (p<0.001) | −0.169 |
| `nomao` (id=1486) | 118 | **0.933** (p<0.001) | 0.649 (p<0.001) | −0.038 |
| `guillermo` (id=41159) | 4296 | **0.813** (p<0.001) | −0.066 (p<0.001) | −0.009 |

For comparison, the originally-designed `FeatureCrossAttention` (`feature_imp`) projection scored ρ≈0.08–0.10 (≈Random) on `ada_agnostic` — statistically indistinguishable from the Random baseline — which motivated replacing it with Integrated Gradients (Sundararajan et al., 2017) for ③.

`balance-scale` (id=11, F=4) is excluded: with only 4 features, Spearman ρ over 4 items is not statistically meaningful (p≥0.2 for every method, including SHAP).

**Key finding — IG is robust to F, SHAP (sampling-based) is not.** Across F=41–4296, TabERA's IG-based ③ stays in the ρ≈0.71–0.94 range and is always significant (p<0.001). SHAP's KernelExplainer (`nsamples=100`), by contrast, degrades sharply as F grows: ρ=0.90 (F=41) → 0.84 (F=48) → 0.65 (F=118) → **−0.07 (F=4296, indistinguishable from Random)**. This is a direct consequence of SHAP's sampling-based estimation requiring more samples as F grows, whereas IG requires only a single gradient pass regardless of F.

TabERA (IG) ≥ SHAP holds for the three datasets with F≥48 (3/4 valid datasets); for the smallest-F dataset tested (`qsar-biodeg`, F=41), SHAP remains stronger (0.90 vs 0.71). Verification across additional datasets/seeds is ongoing.

### Cognitive inspiration

The macro-micro structure draws from cognitive science, used as conceptual motivation for explanations ①② rather than direct modeling:

- **Central Tendency** (Posner & Keele, 1968): Centroid as group prototype (①)
- **Schema Theory** (Bartlett, 1932): Coarse routing before fine retrieval (① → ②)
- **Dual-Process** (Kahneman, 2011): Fast group assignment, then careful neighbor comparison (① → ②)

---

## Components

| File | Component | Role |
|------|-----------|------|
| `libs/prototypes.py` | CentroidLayer | Dual-space centroids, STE routing, KMeans++ init, medoid update — explanation ① |
| `libs/tabera.py` | TabERA, MemoryBank, FeatureStore | Model, KNN store (cross-group fallback), raw feature store — explanation ② |
| `libs/evidence.py` | AttentionAggregator | TabR L2 attention (② evidence_w), FeatureCrossAttention + Gated Fusion (predictive auxiliary path, see above) |
| `libs/supervised.py` | TabERAWrapper | Training loop, EMA scheduling |
| `libs/search_space.py` | HPO space | 10 hyperparameters (Optuna) + 1 auto-determined (`n_prototypes`) |
| `libs/data.py` | TabularDataset | OpenML data loader |
| `libs/eval.py` | Metrics | Accuracy, F1, AUROC, Logloss |
| `reproduce.py` (`--ablation rank_correlation`) | Integrated Gradients attribution | Explanation ③ + faithfulness validation (vs SHAP, vs ground-truth perturbation ranking) |

---

## Installation

```bash
# Python 3.12+ recommended
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# PyTorch (CUDA 12.8)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Dependencies
pip install -r requirements.txt
```

---

## Usage

### HPO (Hyperparameter Optimization)

```bash
python optimize.py --gpu_id 0 --openml_id 11 --n_trials 100 --seed 1
```

### Reproduce best configuration

```bash
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --explain  # with explanations
```

### Faithfulness / ablations

```bash
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation gate_analysis
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation no_feat_path
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation rank_correlation
```

### TabZilla benchmark (36 datasets)

```powershell
.\run_tabzilla.ps1
```

---

## Explanation output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TabERA Explanation — Sample #0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

① Group context  (architectural)
   → Centroid_3 (confidence=94.3%)
   alcohol=10.24, pH=3.31, fixed_acidity=7.21
   (nearest real training sample to this centroid — medoid)

② Neighbor evidence  (architectural)
   Neighbour #0: 42.1%  →  alcohol=10.41, pH=3.28
   Neighbour #1: 28.3%  →  volatile_acidity=0.28

③ Feature attribution  (Integrated Gradients, post-hoc)
   volatile_acidity  15.1%
   chlorides         14.9%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> Note: `gate_mean` (Gated Fusion diagnostic) is reported separately via `--ablation gate_analysis`; it reflects the predictive auxiliary path's behavior, not explanation ③.

---

## HPO parameters (10 searched + 1 auto-determined)

| Parameter | Range | Role |
|-----------|-------|------|
| `embed_dim` | {64, 128, 256} | Embedding dimension D |
| `k` | {8, 16, 32, 64} | Number of KNN neighbors |
| `embedder_layers` | 1–4 | ResidualMLP depth |
| `dropout` | 0.0–0.5 | Dropout rate |
| `loss_diversity` | 5e-2 – 5e-1 | Centroid spread |
| `loss_commitment` | 1e-2 – 1e-1 | VQ-VAE commitment |
| `loss_entropy` | 1e-3 – 1e-2 | Routing entropy (collapse prevention) |
| `lr` | 1e-4 – 1e-2 | Learning rate |
| `weight_decay` | 1e-6 – 1e-2 | L2 regularization |
| `batch_size` | {128, 256, 512} | Batch size |

> **`n_prototypes` (number of centroids P) is not searched by Optuna.**
> It is automatically set per dataset as `P = min(sqrt(N_train), n_features)`
> (clamped to a minimum of 4), and overridden onto every trial
> (see `optimize.py`, `n_proto_default`). The actual value used is logged
> in `trial.user_attrs["n_prototypes_actual"]` and restored by `reproduce.py`.
> This can range well beyond a fixed small search range
> (e.g., P≈12 for `lymph` (N=148, F=19) up to P≈185 for `nomao` (N=34,465, F=500)).

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py            # TabERA model (TabERA, MemoryBank, FeatureStore) — explanation ②
│   ├── prototypes.py        # CentroidLayer (STE, KMeans++, Dual-Space, medoid update) — explanation ①
│   ├── evidence.py          # AttentionAggregator (② evidence_w), FeatureCrossAttention + Gated Fusion (predictive auxiliary path)
│   ├── supervised.py        # TabERAWrapper (training loop, EMA)
│   ├── eval.py              # Evaluation metrics
│   ├── search_space.py      # Optuna HPO space (10 params; n_prototypes auto-set)
│   └── data.py              # OpenML data loader
├── optim_logs/              # HPO results per seed
├── figures/                 # Embedding visualizations
├── optimize.py              # Run HPO
├── reproduce.py             # Reproduce best config + explanations (①②③) + faithfulness ablations
├── visualize_embeddings.py  # 3-figure embedding visualization
└── requirements.txt
```

---

## References

- Gorishniy, Y., et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). *ICML 2017*.
- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. *SODA 2007*.
- Posner, M. I. & Keele, S. W. (1968). On the genesis of abstract ideas. *Journal of Experimental Psychology*, 77(3), 353–363.
- Bartlett, F. C. (1932). *Remembering*. Cambridge University Press.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh, D., et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS 2023*.