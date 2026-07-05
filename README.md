# TabERA

**Tabular Explainable Retrieval Architecture**

Centroid-conditioned hierarchical retrieval with example-based explanations for tabular data.

---

## Overview

TabERA organizes training data through learnable centroids and retrieves relevant neighbors within each centroid's group. Beyond prediction, the forward pass itself produces **example-based explanations** — which group a sample belongs to, and which training samples it's compared against — something post-hoc methods (SHAP, LIME, IG) can't produce for any model, since those only operate on feature-level input-output relationships.

```
Query → Embedding → Centroid Routing (macro) → Group-constrained KNN (micro) → Prediction
                          ↓                              ↓                         ↓
                    "Which group?"              "Which neighbors?"          "Which features?"
                    (architectural)              (architectural)             (post-hoc, IG)
```

| Level | Module | Type | Explains |
|---|---|---|---|
| ① Group context | CentroidLayer (`centroid_x`) | Architectural | "This sample belongs to the high-alcohol, low-pH group" |
| ② Neighbor evidence | MemoryBank + AttentionAggregator (`evidence_w`) | Architectural | "Neighbor #1 contributes 42%" |
| ③ Feature attribution | Integrated Gradients | Post-hoc | "`volatile_acidity` has the largest attribution" |

①② are read directly off the forward pass's activations. ③ is standard IG — no architectural novelty claimed, just a well-behaved complement to ①②.

---

## Architecture

![TabERA architecture](docs/TabERA_Figure1.png)

Given `X ∈ ℝ^(N×F)` (D = embedding dim, P = centroids, K = neighbors):

1. **Embed** — `TabularEmbedder` maps `X → query_emb ∈ ℝ^D`.
2. **Route (→ ①)** — `CentroidLayer` assigns each sample to one of `P` centroids via STE hard-argmax, yielding `hard_assignment`, `context_emb`, and `centroid_x` (the medoid — nearest real training sample).
3. **Retrieve & aggregate (→ ②)** — `MemoryBank` runs KNN restricted to the sample's centroid (cross-group fallback if too small); `AttentionAggregator` turns similarities into `evidence_w` and aggregates into `agg_emb`, which feeds the head directly.
4. **Predict** — `[query_emb ‖ context_emb ‖ agg_emb] → MLP head → ŷ`.

③ is computed afterward via IG, independent of stages 2–3.

**Key design decisions:**
- **Dual-Space Centroid** — `centroid_emb` (learnable, routing) and `centroid_x` (real sample, medoid-updated each epoch) are kept separate so ① shows an actual data point.
- **STE Routing** — forward uses discrete argmax; backward substitutes the softmax gradient, active at train and eval time so ③ still receives a gradient through routing.
- **Cross-group Fallback** — expands to the nearest adjacent centroid rather than a global search when a group is small.
- **Auxiliary Losses** — `diversity_loss` + `commitment_loss`, plus epoch-wise medoid updates, maintain meaningful groups across datasets.

---

## Faithfulness of explanation ③

**Completeness** (attributions should sum to `ŷ(x)-ŷ(x̄)`) depends on the baseline `x̄`. Median relative completeness error, mean vs. medoid baseline:

| Dataset | Mean baseline | Medoid baseline |
|---|---|---|
| `vehicle` | 146.5% | 17.5% |
| `ada_agnostic` | 19.4% | 1.5% |
| `qsar-biodeg` | 53.4% | 1.8% |
| `wine_quality` | 319.1% | 4.1% |

We use the centroid medoid as the default baseline, since it consistently gives lower completeness error.

**Deletion/Insertion AUC vs. SHAP** (medoid baseline for both; paired Wilcoxon test):

| Dataset | Deletion ↓ (TabERA / SHAP) | Insertion ↑ (TabERA / SHAP) |
|---|---|---|
| `vehicle` | 0.676 / 0.623 (n.s.) | 0.736 / 0.716 (n.s.) |
| `ada_agnostic` | 0.794 / 0.784 (SHAP, p=0.01) | 0.854 / 0.850 (n.s.) |
| `qsar-biodeg` | 0.724 / 0.732 (n.s.) | 0.885 / 0.848 (n.s., p=0.08) |
| `wine_quality` | 0.466 / 0.512 (SHAP, p<0.01) | 0.569 / 0.516 (TabERA, p<0.001) |

*n.s. = not significant (Wilcoxon, p≥0.05) — the difference can't be distinguished from sample noise.*

Of 8 comparisons, 3 reach significance and the direction is mixed. ③ is best read as on par with SHAP, with the practical advantage of needing no background distribution or sampling budget — just one gradient pass. Explanations ①② remain TabERA's primary, architecturally distinctive contribution.

*(Caveat on ②: cross-group fallback triggers more often than "group-constrained" suggests when K exceeds average group size — 75% of samples on the smallest dataset tested (`vehicle`, N=676) vs. 7–14% on larger ones.)*

**Cognitive inspiration** (conceptual motivation only): Central Tendency (Posner & Keele, 1968), Schema Theory (Bartlett, 1932), Dual-Process theory (Kahneman, 2011).

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
| `reproduce.py` | Reproducer | Best config, `--explain` (①②③), `--ablation` (faithfulness) |
| `visualize_embeddings.py` | Visualizer | Embedding structure, class pies, KNN closeups |

---

## Installation & Usage

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

```bash
# HPO
python optimize.py --gpu_id 0 --openml_id 11 --n_trials 100 --seed 1

# Reproduce best config (+ explanations)
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --explain

# Faithfulness / ablations
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation rank_correlation

# TabZilla benchmark (36 datasets)
.\run_tabzilla.ps1
```

**Sample explanation output:**
```
① Group context      → Centroid_3 (conf. 94.3%): alcohol=10.24, pH=3.31
② Neighbor evidence   → #0 42.1%: alcohol=10.41, pH=3.28
③ Feature attribution → volatile_acidity 15.1%
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

> `n_prototypes` (P) is **not** searched — auto-set as `P = sqrt(N_train)` (min 4). Ranges P≈12 (`lymph`, N=148) to P≈185 (`nomao`, N=34,465).

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
├── optim_logs/ / figures/
├── optimize.py / reproduce.py / visualize_embeddings.py
└── requirements.txt
```

---

## References

- Gorishniy et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS*.
- Bengio et al. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.
- Posner & Keele (1968). On the genesis of abstract ideas. *J. Exp. Psych.* 77(3).
- Bartlett (1932). *Remembering*. Cambridge University Press.
- Kahneman (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS*.