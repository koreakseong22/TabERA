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

We evaluate ③ on two axes: how well its attributions satisfy IG's own Completeness axiom (attributions sum to `ŷ(x) - ŷ(x̄)`), and how well they perform against SHAP on standard deletion/insertion faithfulness.

**Completeness, by baseline choice.** Median relative completeness error, dataset mean vs. centroid medoid as baseline `x̄`:

| Dataset | Mean baseline | Medoid baseline |
|---|---|---|
| `vehicle` | 146.5% | 17.5% |
| `ada_agnostic` | 19.4% | 1.5% |
| `qsar-biodeg` | 53.4% | 1.8% |
| `wine_quality` | 319.1% | 4.1% |

We use the centroid medoid (`centroid_x`, the same one used for ①) as the default baseline throughout, since it consistently gives the lower completeness error.

**Deletion/Insertion AUC vs. SHAP** (medoid baseline for TabERA; SHAP background also set to the medoids for a like-for-like comparison; paired Wilcoxon signed-rank test per dataset):

| Dataset | Deletion AUC ↓ (TabERA / SHAP) | Insertion AUC ↑ (TabERA / SHAP) |
|---|---|---|
| `vehicle` | 0.676 / 0.623 (n.s.) | 0.736 / 0.716 (n.s.) |
| `ada_agnostic` | 0.794 / 0.784 (SHAP, p=0.01) | 0.854 / 0.850 (n.s.) |
| `qsar-biodeg` | 0.724 / 0.732 (n.s.) | 0.885 / 0.848 (n.s., p=0.08) |
| `wine_quality` | 0.466 / 0.512 (SHAP, p<0.01) | 0.569 / 0.516 (TabERA, p<0.001) |

*n.s. = not significant (paired Wilcoxon signed-rank test, p≥0.05) — i.e., the observed difference between TabERA and SHAP cannot be distinguished from sample noise.*

Across the 8 dataset×metric comparisons, 3 reach significance and the direction is mixed (2 favoring SHAP, 1 favoring TabERA). ③ is therefore best read as on par with SHAP rather than superior to it — its practical advantages are that it requires no background distribution or sampling budget, just a single gradient pass, and (with the medoid baseline above) a completeness guarantee we can verify directly. Explanations ①② remain TabERA's primary and architecturally distinctive contribution.

*(Caveat on ②: cross-group fallback can trigger far more often than "group-constrained" suggests when HPO's chosen K exceeds the average group size — 75% of samples on the smallest dataset tested (`vehicle`, N=676) vs. 7–14% on larger ones.)*

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