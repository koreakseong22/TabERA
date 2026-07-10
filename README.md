# TabERA

**Tabular Explainable Retrieval Architecture**

A retrieval-augmented tabular model that produces architectural, example-based explanations alongside its predictions.

---

## Background

Post-hoc attribution methods (SHAP, LIME, IG) only ever answer "which features mattered," and only after the fact. Retrieval-augmented models (e.g., TabR) suggest a richer alternative: if a model predicts by comparing a query to stored training examples, the retrieval step itself explains "which group this belongs to" and "which examples this prediction is like" — something no post-hoc method can offer for an arbitrary model.

TabERA is built around this idea: group assignment and neighbor retrieval are load-bearing parts of the forward pass, not add-ons, so the explanations they produce are guaranteed to reflect what the model actually did. Feature-level attribution (Integrated Gradients) is added as a complementary, standard third layer.

---

## Architecture

![TabERA architecture](docs/TabERA_Figure1.png)

```
Query → Embedding → Centroid Routing → Group-constrained KNN → Prediction
                          ↓                     ↓                   ↓
                    "Which group?"      "Which neighbors?"   "Which features?"
                    (architectural)      (architectural)       (post-hoc, IG)
```

1. **Embed** — `TabularEmbedder` maps `X → query_emb`.
2. **Route (→ ①)** — `CentroidLayer` assigns each sample to one of `P` centroids via STE hard-argmax, producing `hard_assignment` and `context_emb`. Each centroid also has a `centroid_x` medoid (nearest real training sample), used as the IG baseline in ③ — not shown directly in ① (see below).
3. **Retrieve & aggregate (→ ②)** — `MemoryBank` runs K-NN restricted to the sample's group (cross-group fallback if the group is smaller than K). `AttentionAggregator` turns similarities into `evidence_w` and aggregates into `agg_emb`, fed straight into the prediction head.
4. **Predict** — `[query_emb ‖ context_emb ‖ agg_emb] → MLP head → ŷ`.
5. **Attribute (→ ③, post-hoc)** — Integrated Gradients differentiates `ŷ` w.r.t. `X`, independent of steps 2–3.

**Design notes:** `diversity_loss`/`commitment_loss` keep centroids well-separated and queries committed; cross-group fallback was originally a per-sample Python loop (up to 348× slower on high-fallback datasets) and is now fully vectorized, verified bit-identical to the original.

---

## Explanations

| Level | Type | Answers |
|---|---|---|
| ① Prototype Group | Architectural | What kind of sample is this, and how confidently? |
| ② Neighbor Evidence | Architectural | Which real training examples drove this prediction? |
| ③ Feature Attribution | Post-hoc (IG) | Which features moved the prediction, and by how much? |

**① Prototype Group** — assigned group + confidence, runner-up groups (each with their own confidence and target distribution), and what the group *represents*: majority class name + count/share (e.g. `"good" 27/47 (57%)`), plus a second class if it's ≥20%. This target-distribution info is ①'s core content — neither ② nor ③ carries it. Distinctive features are ranked by how unusual this group is *relative to other groups* (robust z-score, median/MAD), not vs. the whole dataset — numeric values inverse-transformed to real units, categorical shown as real category names with in-group share. No medoid is shown as a representative sample here: ② already shows real, prediction-relevant samples on stronger grounds, and a medoid would be fragile for small groups.

**② Neighbor Evidence** — the `k` actual retrieved neighbors and their `evidence_w`, the same values used in the forward pass. Neighbors with ~zero weight are dropped (noise, not signal). Each neighbor's shown features are picked by closeness to the query (smallest gap first — normalized distance for numeric, match/mismatch for categorical), not by the neighbor's own largest values — this is what explains *why* it's similar. Up to 4 features per neighbor, fewer if fewer are actually close (at least 1 always shown). Numeric/categorical shown separately; numeric in real units, categorical as name + original code (`checking_status=no checking [0]`).

**③ Feature Attribution** — IG with a non-standard baseline: the medoid of the sample's own group, not the dataset mean or zero. This is specific to TabERA's STE hard routing: if the baseline routes to a different group than the query, the IG path crosses a forward discontinuity and completeness breaks. The medoid is structurally guaranteed to route to the same group. This does **not** mean better attributions than SHAP (deletion/insertion AUC results below are mixed) — only better completeness convergence. Those are separate claims.

**Sample output:**
```
① Prototype Group     → "Centroid_3" (94.3%) — "good" 27/47 (57%), also "fair" 12/47 (26%)
                         Distinctive: alcohol=10.24, volatile_acidity=0.31, region=Piedmont (68%)
② Neighbor Evidence   → #0 42.1%: alcohol=10.41, pH=3.28   #1 31.7%: alcohol=10.19
③ Feature Attribution → volatile_acidity 15.1%
```

**IG vs. SHAP** (medoid background, deletion/insertion AUC, paired Wilcoxon) — only 3/8 comparisons significant, direction mixed:

| Dataset | Deletion ↓ (TabERA/SHAP) | Insertion ↑ (TabERA/SHAP) |
|---|---|---|
| `vehicle` | 0.676/0.623 (n.s.) | 0.736/0.716 (n.s.) |
| `ada_agnostic` | 0.794/0.784 (SHAP, p=0.01) | 0.854/0.850 (n.s.) |
| `qsar-biodeg` | 0.724/0.732 (n.s.) | 0.885/0.848 (n.s.) |
| `wine_quality` | 0.466/0.512 (SHAP, p<0.01) | 0.569/0.516 (TabERA, p<0.001) |

We use IG anyway for cost and structural fit, not superior accuracy: it needs only the gradient TabERA already produces plus one baseline point, and the medoid baseline cuts completeness error from 19–319% to 1.5–17.5% vs. a dataset-mean baseline. ①② remain TabERA's primary contribution regardless.

*Cognitive inspiration (conceptual only): Central Tendency (Posner & Keele, 1968), Schema Theory (Bartlett, 1932), Dual-Process theory (Kahneman, 2011).*

---

## Validation

`--ablation dual_space_faithfulness` checks whether ①② actually mean what they claim:
- **Index integrity** — `sample_groups` matches live model state: 100% consistent, reproducibly, across datasets and seeds.
- **Group separation** — centroids correspond to statistically distinct regions of feature space (ANOVA F-test for numeric, chi-square for categorical, Bonferroni-corrected). Strongly and consistently significant across numeric, categorical, and mixed-type datasets and seeds.

*(Caveat: cross-group fallback can widen "neighbor within your group" more than expected — 75% of samples on the smallest dataset tested vs. 7–14% on larger ones. Affects which neighbors get retrieved, not speed.)*

---

## CLI reference

- **`--explain`** — prints ①②③ as text.
- **`--ablation`** — `random_neighbor` (wrong-but-real neighbors, tests retrieval correctness), `neighbor_noise` (fake neighbors, tests whether neighbor info matters at all — read together with `random_neighbor`), `rank_correlation` (IG vs. SHAP vs. Random attribution-rank agreement with actual prediction impact), `dual_space_faithfulness` (validation above), `deletion_auc`/`insertion_auc` (RISE-style faithfulness for ③), `dataset_profile` (quick diagnostic for a new dataset).
- **`--from_saved_state <path>`** — reload a saved model state and rerun `--explain`/`--ablation` without retraining.

---

## Known limitations

- ①'s distinctive-feature contrast isn't equally sharp on every dataset.
- Group sizes can be naturally imbalanced, reflecting the data itself — not a failure mode by itself.
- `--from_saved_state` on GPU can show sub-decimal inference differences from a fresh run despite identical weights, likely from non-deterministic op scheduling (e.g. cuDNN).

---

## Contribution

- **Architecturally-guaranteed explanations (①②)** — read directly off the forward pass, not estimated afterward.
- **A structural fix for IG's baseline problem**, specific to STE hard-routing: the group medoid avoids the cross-group discontinuity a dataset-mean baseline would hit, improving completeness by 8–78×.
- **A low-cost case for IG over SHAP here**: comparable empirical performance, cheaper compute — a fit argument, not a quality trade-off.
- **A documented limitation**: cross-group fallback rate depends on `K` vs. group size (`P = √N_train`) and can dominate small datasets.

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
| `reproduce.py` | Reproducer | `--explain`, `--ablation`, `--from_saved_state` |
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
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --ablation dual_space_faithfulness

# Re-inspect explanations without retraining
python reproduce.py --gpu_id 0 --openml_id 11 --seed 1 --from_saved_state <path> --explain

# TabZilla benchmark (36 datasets)
.\run_tabzilla.ps1
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
- Petsiuk et al. (2018). RISE: Randomized Input Sampling for Explanation of Black-box Models. *BMVC*.
- Posner & Keele (1968). On the genesis of abstract ideas. *J. Exp. Psych.* 77(3).
- Bartlett (1932). *Remembering*. Cambridge University Press.
- Kahneman (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS*.