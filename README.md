# TabERA

**Tabular Explainable Retrieval Architecture**

Centroid-conditioned hierarchical retrieval with example-based explanations for tabular data.

---

## Overview

TabERA organizes training data through learnable centroids and retrieves relevant neighbors within each centroid's group. Beyond prediction, the forward pass itself produces **example-based explanations** — which group a sample belongs to, and which training samples it's compared against — information post-hoc methods (SHAP, LIME, IG) cannot produce for any model, since those only operate on feature-level input-output relationships.

```
Query → Embedding → Centroid Routing (macro) → Group-constrained KNN (micro) → Prediction
                          ↓                              ↓                         ↓
                    "Which group?"              "Which neighbors?"          "Which features?"
                    (architectural)              (architectural)             (post-hoc, IG)
```

### Three-level explanation chain

| Level | Module | Type | Explains |
|---|---|---|---|
| ① Group context | CentroidLayer (`centroid_x`) | Architectural | "This sample belongs to the high-alcohol, low-pH group" |
| ② Neighbor evidence | MemoryBank + AttentionAggregator (`evidence_w`) | Architectural | "Neighbor #1 contributes 42%" |
| ③ Feature attribution | Integrated Gradients | Post-hoc | "`volatile_acidity` has the largest attribution" |

①② are read directly off the forward pass's intermediate activations — no post-hoc method can produce this for an arbitrary model, since it requires an architecture that explicitly organizes and retrieves training examples at inference time. ③ is standard IG; we don't claim architectural novelty for it, only that (with the right baseline — see Faithfulness below) it's a reasonably-behaved complement to ①②.

---

## Architecture

![TabERA architecture](docs/TabERA_Figure1.png)

Given `X ∈ ℝ^(N×F)` (F features, D embedding dim, P centroids, K neighbors):

1. **Embed** — `TabularEmbedder` (residual MLP stack) maps `X → query_emb ∈ ℝ^D`. This is the only place raw features feed the prediction path (explanation ③ later differentiates back through it).
2. **Route (→ ①)** — `CentroidLayer` assigns each sample to exactly one of `P` centroids via STE hard-argmax on cosine similarity, producing `hard_assignment`, `context_emb`, and `centroid_x` (the medoid — nearest real training sample — used to make ① human-readable).
3. **Retrieve & aggregate (→ ②)** — `MemoryBank.retrieve` does a group-constrained KNN restricted to the sample's centroid (with cross-group fallback if the group is smaller than K). `AttentionAggregator` turns neighbor similarities into `evidence_w` (TabR-style softmax) and aggregates into `agg_emb`, which feeds the prediction head directly — so `evidence_w` isn't just diagnostic, it's load-bearing.
4. **Predict** — `[query_emb ‖ context_emb ‖ agg_emb] → MLP head → ŷ`.

Explanation ③ is computed afterward via IG, independent of stages 2–3's internals.

### Key design decisions

- **Dual-Space Centroid**: each centroid keeps both a learnable `centroid_emb` (routing/retrieval) and a `centroid_x` (real training sample, medoid-updated each epoch) so explanation ① shows an actual data point, not a synthetic average.
- **STE Routing**: forward uses discrete argmax (crisp groups, what ① reports); backward substitutes the softmax gradient so `C_emb` stays trainable. Active at both train and eval time, so ③'s gradient still flows through routing.
- **Cross-group Fallback**: if a centroid group has fewer than K members, retrieval expands to the nearest adjacent group(s) rather than falling back to a global search — preserves the *meaning* of ② even for small groups (though see the faithfulness caveat below).
- **Auxiliary Losses**: `diversity_loss` (spreads centroids apart) + `commitment_loss` (pulls queries to their assigned centroid), together with epoch-wise medoid updates, are enough to maintain meaningful groups across datasets.

### Faithfulness of explanation ③

IG attributes `ŷ(x) - ŷ(x̄)` to each feature along a straight-line path from baseline `x̄` to `x`. Its *Completeness* axiom (attributions sum exactly to `ŷ(x)-ŷ(x̄)`) is directly measurable, and turns out to hinge heavily on `x̄`.

With the dataset **mean** as baseline, completeness error is large and inconsistent (median relative error 19–319% across datasets) — the mean typically doesn't fall inside any centroid's region, so the IG path often crosses a routing boundary where `context_emb` jumps discretely, a discontinuity IG's theory doesn't account for.

TabERA already has a fix built in: using the **medoid** (`centroid_x`, same one used for ①) as baseline instead collapses completeness error 8–78× across every dataset tested:

| Dataset | Mean-baseline error (median %) | Medoid-baseline error (median %) | Improvement |
|---|---|---|---|
| `vehicle` | 146.5% | 17.5% | 8.4× |
| `ada_agnostic` | 19.4% | 1.5% | 13× |
| `qsar-biodeg` | 53.4% | 1.8% | 30× |
| `wine_quality` | 319.1% | 4.1% | 78× |

This is a genuine secondary contribution — TabERA's own retrieval structure supplies a principled solution to IG's baseline-selection problem that a plain feedforward network doesn't have.

Better completeness doesn't automatically mean better attributions, so we checked directly: comparing TabERA's IG (medoid baseline) against SHAP (background also set to medoids, for a fair comparison) on deletion/insertion AUC, with paired Wilcoxon tests, across the same four datasets — only 3 of 8 comparisons reached significance, and they didn't agree in direction (SHAP better on `ada`/`wine` deletion, TabERA better on `wine` insertion, no significant difference elsewhere). We read this as an honest null: under a rigorous baseline, ③ is not consistently more or less faithful than SHAP. Explanations ①② remain TabERA's primary contribution; ③ is a reasonably-behaved, "comes for free" complement, not evidence that TabERA beats post-hoc attribution.

*(Caveat on ②: cross-group fallback can trigger far more often than the name "group-constrained" suggests when HPO's chosen K exceeds the average group size — 75% of samples on the smallest dataset tested (`vehicle`, N=676) vs. 7–14% on larger ones. Worth keeping in mind when reading ② on small datasets.)*

### Cognitive inspiration

The macro-micro structure draws loosely on cognitive science (conceptual motivation for ①②, not direct modeling): Central Tendency (Posner & Keele, 1968) for the centroid-as-prototype idea, Schema Theory (Bartlett, 1932) for coarse-then-fine routing, Dual-Process theory (Kahneman, 2011) for fast group assignment followed by careful neighbor comparison.

---

## Components

| File | Component | Role |
|---|---|---|
| `libs/prototypes.py` | CentroidLayer | STE routing, KMeans++ init, medoid update — ① |
| `libs/tabera.py` | TabERA, MemoryBank | Model, group-constrained KNN store — ② |
| `libs/evidence.py` | AttentionAggregator | `evidence_w`, direct retrieval path — ② |
| `libs/supervised.py` | TabERAWrapper | Training loop, EMA regrouping, early stopping |
| `libs/search_space.py` | HPO space | 9 params (Optuna) + auto `n_prototypes` |
| `libs/data.py` | TabularDataset | OpenML loader |
| `libs/eval.py` | Metrics | Accuracy, F1, AUROC, Logloss |
| `optimize.py` | HPO runner | Auto-sets `n_prototypes = sqrt(N_train)` |
| `reproduce.py` | Reproducer | Best config, `--explain` for ①②③, `--ablation` for faithfulness checks |
| `visualize_embeddings.py` | Visualizer | Embedding structure, centroid class pies, KNN closeups |

---

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Usage

```bash
# HPO
python optimize.py --gpu_id 0 --openml_id 11 --n_trials 100 --seed 1

# Reproduce best config (add --explain for ①②③)
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --explain

# Faithfulness / ablations
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation rank_correlation

# TabZilla benchmark (36 datasets)
.\run_tabzilla.ps1
```

## Explanation output

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TabERA Explanation — Sample #0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① Group context  (architectural)
   → Centroid_3 (confidence=94.3%)  alcohol=10.24, pH=3.31

② Neighbor evidence  (architectural)
   Neighbour #0: 42.1%  →  alcohol=10.41, pH=3.28

③ Feature attribution  (Integrated Gradients, post-hoc)
   volatile_acidity  15.1%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## HPO parameters (9 searched)

| Parameter | Range | Role |
|---|---|---|
| `embed_dim` | {64, 128, 256} | Embedding dim D |
| `k` | {8, 16, 32, 64} | KNN neighbors |
| `embedder_layers` | 1–4 | ResidualMLP depth |
| `dropout` | 0.0–0.5 | — |
| `loss_diversity` | 5e-2–5e-1 | Centroid spread penalty |
| `loss_commitment` | 1e-2–1e-1 | Query-centroid commitment |
| `lr` | 1e-4–1e-2 | — |
| `weight_decay` | 1e-6–1e-2 | — |
| `batch_size` | {128, 256, 512} | — |

> `n_prototypes` (P) is **not** searched — auto-set as `P = sqrt(N_train)` (min 4), logged in `trial.user_attrs["n_prototypes_actual"]`. Ranges from P≈12 (`lymph`, N=148) to P≈185 (`nomao`, N=34,465).

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py            # Model, MemoryBank — ②
│   ├── prototypes.py        # CentroidLayer — ①
│   ├── evidence.py          # AttentionAggregator — ②
│   ├── supervised.py        # Training wrapper
│   ├── eval.py               
│   ├── search_space.py      # HPO space
│   └── data.py              # OpenML loader
├── optim_logs/
├── figures/
├── optimize.py
├── reproduce.py             # Explanations ①②③ + faithfulness ablations
├── visualize_embeddings.py
└── requirements.txt
```

---

## References

- Gorishniy et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS*.
- Bengio et al. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.
- Posner & Keele (1968). On the genesis of abstract ideas. *J. Exp. Psych.*, 77(3).
- Bartlett (1932). *Remembering*. Cambridge University Press.
- Kahneman (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS*.