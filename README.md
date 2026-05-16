# TabERA

**Tabular Explainable Retrieval Architecture**

Centroid-conditioned hierarchical retrieval with faithful explanations for tabular data.

---

## Overview

TabERA is a retrieval-augmented tabular model that organizes data through learnable centroids and retrieves relevant neighbors within centroid groups. The architecture produces predictions and multi-level explanations simultaneously, where the explanation pathway is architecturally guaranteed to participate in the prediction (faithfulness).

```
Query → Embedding → Centroid Routing (macro) → Group-constrained KNN (micro) → Prediction
                          ↓                              ↓                         ↓
                    "Which group?"              "Which neighbors?"          "Which features?"
```

### Three-level explanation chain

| Level | Module | Explains |
|-------|--------|----------|
| ① Group context | CentroidLayer (centroid_x) | "This sample belongs to the high-alcohol, low-pH group" |
| ② Neighbor evidence | AttentionAggregator (evidence_w) | "Neighbor #1 contributes 42%" |
| ③ Feature attribution | FeatureCrossAttention (feature_imp) | "volatile_acidity drives 15.1% of similarity" |

Levels ② and ③ participate in the prediction via **Gated Fusion**, ensuring that the explanation reflects what the model actually uses for its decision.

---

## Architecture

### Forward flow

```
X ∈ ℝ^(N×F)
  ↓ TabularEmbedder (ResidualMLP × L)
query_emb ∈ ℝ^D
  ├── CentroidLayer
  │     C_emb ∈ ℝ^(P×D)  — learnable, gradient + STE routing
  │     C_x   ∈ ℝ^(P×F)  — EMA only, original-space summary (explanation)
  │     → context_emb, hard_assignment, routing_probs
  │
  ├── MemoryBank.retrieve (group-constrained KNN)
  │     Cross-group fallback for small groups (adjacent centroid expansion)
  │     → nk (B,K,D), neighbor_labels (B,K)
  │
  └── AttentionAggregator
        ├─ TabR L2 similarity → evidence_w (B,K)
        ├─ Value construction: label_emb + T(query - neighbor)
        ├─ agg_emb = weighted sum of values
        ├─ FeatureCrossAttention → feature_imp (B,K,F)
        └─ Gated Fusion: fused_agg = gate·feat_emb + (1-gate)·agg_emb
              ↓
[query_emb ‖ context_emb ‖ fused_agg] ∈ ℝ^(3D) → MLP Head → ŷ
```

### Key design decisions

**Dual-Space Centroid.** Each centroid maintains two representations: `centroid_emb` in embedding space (learnable, used for routing and retrieval) and `centroid_x` in original feature space (EMA-updated, used for human-readable explanations). This separation ensures that explanations show actual data statistics ("alcohol=10.24") rather than opaque embedding coordinates.

**STE Routing.** Hard assignment via Straight-Through Estimator (Bengio et al., 2013; VQ-VAE, van den Oord et al., 2017). Forward pass uses discrete argmax for crisp group boundaries; backward pass passes gradients through softmax.

**Cross-group Fallback.** When a centroid group has fewer than K members, the search expands to the nearest adjacent centroid group rather than falling back to global search. This preserves the centroid structure's meaning even for small groups.

**Gated Fusion (Faithfulness).** The feature cross-attention output is projected to embedding space and mixed with the neighbor aggregation via a learned gate. This ensures that the features highlighted in the explanation actually participate in the prediction, providing architectural faithfulness. Without this, explanations and predictions could diverge.

### Cognitive inspiration

The macro-micro structure draws from cognitive science, used as conceptual motivation rather than direct modeling:

- **Central Tendency** (Posner & Keele, 1968): Centroid as group prototype
- **Schema Theory** (Bartlett, 1932): Coarse routing before fine retrieval
- **Dual-Process** (Kahneman, 2011): Fast group assignment, then careful neighbor comparison

---

## Components

| File | Component | Role |
|------|-----------|------|
| `libs/prototypes.py` | CentroidLayer | Dual-space centroids, STE routing, KMeans++ init, EMA |
| `libs/evidence.py` | AttentionAggregator | TabR L2 attention, Gated Fusion, FeatureCrossAttention |
| `libs/tabera.py` | TabERA, MemoryBank, FeatureStore | Model, KNN store (cross-group fallback), raw feature store |
| `libs/supervised.py` | TabERAWrapper | Training loop, EMA scheduling |
| `libs/search_space.py` | HPO space | 11 hyperparameters (Optuna) |
| `libs/data.py` | TabularDataset | OpenML data loader |
| `libs/eval.py` | Metrics | Accuracy, F1, AUROC, Logloss |

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

① Group context
   → Centroid_3 (confidence=94.3%)
   alcohol=10.24, pH=3.31, fixed_acidity=7.21

② Neighbor evidence
   Neighbour #0: 42.1%  →  alcohol=10.41, pH=3.28
   Neighbour #1: 28.3%  →  volatile_acidity=0.28

③ Feature attribution
   volatile_acidity  15.1%
   chlorides         14.9%

   gate_mean: 0.47  (feature path ↔ neighbor path balance)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## HPO parameters (11)

| Parameter | Range | Role |
|-----------|-------|------|
| `embed_dim` | {64, 128, 256} | Embedding dimension D |
| `k` | {8, 16, 32, 64} | Number of KNN neighbors |
| `n_prototypes` | 4–16 (step 4) | Number of centroids P |
| `embedder_layers` | 1–4 | ResidualMLP depth |
| `dropout` | 0.0–0.5 | Dropout rate |
| `loss_diversity` | 1e-2 – 5e-1 | Centroid spread |
| `loss_commitment` | 1e-2 – 1e-1 | VQ-VAE commitment |
| `loss_entropy` | 1e-3 – 1e-1 | Routing entropy (collapse prevention) |
| `lr` | 1e-4 – 1e-2 | Learning rate |
| `weight_decay` | 1e-6 – 1e-2 | L2 regularization |
| `batch_size` | {128, 256, 512} | Batch size |

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py            # TabERA model (TabERA, MemoryBank, FeatureStore)
│   ├── prototypes.py        # CentroidLayer (STE, KMeans++, EMA, Dual-Space)
│   ├── evidence.py          # AttentionAggregator, FeatureCrossAttention, Gated Fusion
│   ├── supervised.py        # TabERAWrapper (training loop, EMA)
│   ├── eval.py              # Evaluation metrics
│   ├── search_space.py      # Optuna HPO space (11 params)
│   └── data.py              # OpenML data loader
├── optim_logs/              # HPO results per seed
├── figures/                 # Embedding visualizations
├── optimize.py              # Run HPO
├── reproduce.py             # Reproduce best config + explanations
├── visualize_embeddings.py  # 3-figure embedding visualization
└── requirements.txt
```

---

## References

- Gorishniy, Y., et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*.
- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Arthur, D. & Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. *SODA 2007*.
- Posner, M. I. & Keele, S. W. (1968). On the genesis of abstract ideas. *Journal of Experimental Psychology*, 77(3), 353–363.
- Bartlett, F. C. (1932). *Remembering*. Cambridge University Press.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh, D., et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS 2023*.