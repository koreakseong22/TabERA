# TabERA

**Tabular Explainable Retrieval Architecture**

A retrieval-augmented tabular model that produces architectural, example-based explanations alongside its predictions.

---

## Background

Post-hoc attribution methods (SHAP, LIME, IG) only ever answer "which features mattered," and only after the fact. Retrieval-augmented models (e.g., TabR) suggest a richer alternative: if a model predicts by comparing a query to stored training examples, the retrieval step itself explains "which group this belongs to" and "which examples this prediction is like" — something no post-hoc method can offer for an arbitrary model.

TabERA is built around this idea: group assignment and neighbor retrieval are load-bearing parts of the forward pass, not add-ons, so the explanations they produce are guaranteed to reflect what the model actually did. These two layers — ① *which group* and ② *which neighbors* — are TabERA's core contribution. ② in particular is holistic by construction: showing a real neighbor sample already reflects however its features acted together, without decomposing them one at a time, so it implicitly carries feature-interaction information that a per-feature breakdown wouldn't.

A third, feature-level layer (→ ③) is added alongside ①② as a standard, complementary post-hoc device — the kind of "which individual features moved the prediction" summary people are used to from other tabular models. Because ①② already carry the paper's core, architecturally-guaranteed claim, ③'s job is narrower: pick whichever post-hoc attribution method works best, without that choice affecting ①②'s guarantees. TabERA uses SHAP here (see *Why SHAP* below for why, over IG/LIME/plain ablation).

---

## Architecture

![TabERA architecture](docs/TabERA_Figure1.png)

```
Query → Embedding → Centroid Routing → Group-constrained KNN → Prediction
                          ↓                     ↓                   ↓
                    "Which group?"      "Which neighbors?"   "Which features?"
                    (architectural)      (architectural)      (post-hoc, SHAP)
```

1. **Embed** — `TabularEmbedder` maps `X → query_emb`. Categorical features are
   one-hot encoded (no learned parameters, no false ordinal structure — a
   raw integer code has no business implying distance between categories);
   numeric features go through PLR(lite) — a learned periodic (sin/cos)
   embedding followed by a Linear+ReLU shared across all numeric columns,
   following TabR/ModernNCA. Both are vectorized (a single offset-indexed
   gather, no per-column Python loop).
2. **Route (→ ①)** — `CentroidLayer` assigns each sample to one of `P`
   centroids via STE hard-argmax over a *scaled* cosine similarity
   (`routing_scale` widens the otherwise narrow [-1,1] range so the softmax
   isn't flat by construction — the same reason ArcFace/CosFace-style losses
   and cosine-router MoE literature scale logits before softmax), producing
   `hard_assignment` and `context_emb`. Centroids are trained with VQ-VAE's
   full dual loss (`commitment_loss` pulls queries toward their centroid,
   `codebook_loss` pulls centroids toward their queries — both operate on
   L2-normalized vectors, matching what routing itself uses), kept at unit
   norm via a CosFace-style reprojection after every optimizer step, and
   protected from permanent dead centroids by a Jukebox/SoundStream-style
   reset (a centroid with no assignments for several consecutive epochs is
   reinitialized to a real embedding). See `docs/centroid-stability.md` for
   the full mechanism and validation.
3. **Retrieve & aggregate (→ ②)** — `MemoryBank` runs K-NN restricted to the
   sample's group (cross-group fallback if the group is smaller than K).
   `AttentionAggregator` turns similarities into `evidence_w` and aggregates
   into `agg_emb`. Retrieved neighbors' labels are embedded via a learned
   lookup table for classification (an index is a class, not a scalar) or a
   linear layer for regression (the label *is* a scalar) — matching how TabR
   itself distinguishes the two. `agg_emb` feeds straight into the
   prediction head.
4. **Predict** — `[query_emb ‖ context_emb ‖ agg_emb] → MLP head → ŷ`.
5. **Attribute (→ ③, post-hoc)** — SHAP (`KernelExplainer`) estimates each
   feature's marginal contribution to `ŷ` by evaluating the model on many
   perturbed inputs, independent of steps 2–3. Being a black-box method, it
   needs neither a gradient nor a continuous path from a baseline to `X` —
   exactly the two things TabERA's STE hard-routing and one-hot categorical
   encoding would break for a gradient-based method (see *Why SHAP*).

**Design notes:** `diversity_loss` keeps centroids well-separated; `commitment_loss`/`codebook_loss` keep queries and centroids mutually anchored (§ above); cross-group fallback is fully vectorized (a single offset-indexed gather, no per-sample Python loop), which matters most on datasets where fallback rate is high.

---

## Explanations

| Level | Type | Answers |
|---|---|---|
| ① Prototype Group | Architectural | What kind of sample is this, and how confidently? |
| ② Neighbor Evidence | Architectural | Which real training examples drove this prediction? |
| ③ Feature Attribution | Post-hoc (SHAP) | Which features moved the prediction, and by how much? |

**① Prototype Group** — assigned group + confidence, runner-up groups (each with their own confidence and target distribution), and what the group *represents*: majority class name + count/share (e.g. `"good" 27/47 (57%)`), plus a second class if it's ≥20%. This target-distribution info is ①'s core content — neither ② nor ③ carries it. Distinctive features are ranked by how unusual this group is *relative to other groups* (robust z-score, median/MAD), not vs. the whole dataset — numeric values inverse-transformed to real units, categorical shown as real category names with in-group share. No medoid is shown as a representative sample here: ② already shows real, prediction-relevant samples on stronger grounds, and a medoid would be fragile for small groups.

**② Neighbor Evidence** — the `k` actual retrieved neighbors and their `evidence_w`, the same values used in the forward pass. Neighbors with ~zero weight are dropped (noise, not signal). Each neighbor's shown features are picked by closeness to the query (smallest gap first — normalized distance for numeric, match/mismatch for categorical), not by the neighbor's own largest values — this is what explains *why* it's similar. Up to 4 features per neighbor, fewer if fewer are actually close (at least 1 always shown). Numeric/categorical shown separately; numeric in real units, categorical as name + original code (`checking_status=no checking [0]`).

**③ Feature Attribution** — SHAP values from `shap.KernelExplainer`, estimated by
perturbing each feature against a small background sample from the training
set. Unlike a gradient-based method, it treats the model as a black box, so
categorical and numeric features are handled identically — no special-casing
needed for TabERA's one-hot categorical encoding or its STE hard routing.

**Sample output:**
```
① Prototype Group     → "Centroid_3" (94.3%) — "good" 27/47 (57%), also "fair" 12/47 (26%)
                         Distinctive: alcohol=10.24, volatile_acidity=0.31, region=Piedmont (68%)
② Neighbor Evidence   → #0 42.1%: alcohol=10.41, pH=3.28   #1 31.7%: alcohol=10.19
③ Feature Attribution → volatile_acidity 15.1%
```

### Why SHAP

③ is deliberately swappable — it doesn't carry ①②'s architectural guarantee,
so the choice of method is a practical one, not a claim the paper depends on.
A few alternatives were ruled out:

- **Integrated Gradients** — assumes a continuous baseline→input path, which
  fits image/continuous-tabular gradients but not categorical columns. TabERA's
  categorical encoding casts to `int` before one-hot (`x.round().long()`),
  which severs the autograd graph at exactly that point: gradients for
  categorical features come back as exactly `0`, silently — not near-zero,
  not noisy, just zero, regardless of how much that feature actually mattered.
  On all-categorical datasets this doesn't just distort attributions, it
  crashes outright (`RuntimeError: ... appears to not have been used in the
  graph`). This is a known, general limitation of gradient-path methods on
  discrete inputs, not something specific to this codebase.
- **Feature Ablation / Occlusion-1 ("Delta")** — perturbs one feature at a
  time, which is cheap and gradient-free, but is explicitly a low-fidelity
  method in the literature precisely because it can't see higher-order
  feature interactions (only single-feature marginal effects). It's used
  in this repo as a cheap sanity signal (see *Validation* below), not as a
  ground truth — a low-fidelity signal doesn't stop being low-fidelity just
  because it's convenient to compute.
- **LIME** — also gradient-free and handles categorical features fine, but
  its local linear surrogate is known to be less stable than Shapley values
  across repeated runs (sensitive to perturbation sampling and kernel width),
  and it lacks Shapley's axiomatic uniqueness guarantee (below).

SHAP is the only candidate here with an axiomatic backing: Shapley values are
the *unique* allocation satisfying efficiency, symmetry, dummy, and additivity
(Lundberg & Lee, 2017), and — being black-box — they need no gradient at all,
so the categorical-encoding problem above doesn't arise in the first place.
That said, SHAP isn't free of caveats: `KernelExplainer`'s background-sample
perturbation assumes features are independent of whatever's held fixed, which
can evaluate the model on combinations that don't really occur when features
are correlated (a known critique — e.g. Aas et al., 2019).

Its cost also scales with feature count — by design: `KernelExplainer`'s
default `nsamples='auto'` is `2*F + 2048`, growing with `F` because the
number of feature coalitions worth sampling grows with `F` too. TabERA
follows this library formula rather than a fixed sample count, because a
fixed count that ignores `F` underdetermines the KernelSHAP regression on
wide datasets and produces systematically *biased* attributions, not just
noisier ones (on a 144-feature dataset, `nsamples=100` yields a
SHAP-Delta Spearman ρ of 0.53; `nsamples=500` alone moves it to 0.63 —
well past what Monte Carlo noise across repeated runs, ±0.05, could
explain). `--shap_nsamples` remains available to override this manually
if a cheaper, lower-precision run is preferred.

*Cognitive inspiration (conceptual only): Central Tendency (Posner & Keele, 1968), Schema Theory (Bartlett, 1932), Dual-Process theory (Kahneman, 2011).*

---

## Validation

`--ablation dual_space_faithfulness` checks whether ①② actually mean what they claim:
- **Index integrity** — `sample_groups` matches live model state: 100% consistent, reproducibly, across datasets and seeds.
- **Group separation** — centroids correspond to statistically distinct regions of feature space (ANOVA F-test for numeric, chi-square for categorical, Bonferroni-corrected). Strongly and consistently significant across numeric, categorical, and mixed-type datasets and seeds.

*(Caveat: cross-group fallback can widen "neighbor within your group" more than expected — 75% of samples on the smallest dataset tested vs. 7–14% on larger ones. Affects which neighbors get retrieved, not speed.)*

For ③, `--ablation rank_correlation` checks whether SHAP's feature ranking is
at least consistent with a simple, independent signal: Delta (feature
ablation), compared against a random-ranking null via Spearman correlation
and bootstrap CIs. **This is a consistency check, not a correctness proof** —
Delta itself is a low-fidelity, first-order signal (see *Why SHAP*), so a
low SHAP–Delta correlation doesn't necessarily mean SHAP is wrong; it can
mean SHAP is picking up an interaction Delta structurally can't see. To tell
these apart, `--ablation interaction_check` measures whether meaningful
feature interaction is even present in a given dataset (pairwise perturbation:
does perturbing two features together move the prediction more than the sum
of perturbing them separately?) before attributing any SHAP–Delta disagreement
to "SHAP is capturing interaction."

For ①'s underlying centroid layer specifically, two more ablations check
health from different angles rather than assuming trained structure is
automatically meaningful:
- `--ablation centroid_geometry` — is centroid routing distinguishable from
  a random, untrained one at all (50-trial null simulation, see
  `docs/centroid-stability.md`)? Also used automatically inside HPO: a
  percentile-based penalty (no fixed threshold) nudges Optuna away from
  hyperparameter regions that converge to random-or-worse geometry, without
  constraining the search space itself.
- `--ablation centroid_representativeness` — does each individual centroid
  actually represent the members assigned to it (purity vs. a global-
  majority baseline, cohesion vs. the run's other centroids), independent
  of group size?

---

## CLI reference

- **`--explain`** — prints ①②③ as text.
- **`--ablation`** — `random_neighbor` (wrong-but-real neighbors, tests retrieval correctness), `neighbor_noise` (fake neighbors, tests whether neighbor info matters at all — read together with `random_neighbor`), `rank_correlation` (SHAP-vs-Delta rank consistency check, see *Validation*), `interaction_check` (direct test for feature interaction, to interpret `rank_correlation` disagreements correctly), `dual_space_faithfulness` (validation above), `centroid_geometry` (is routing distinguishable from random? see *Validation* and `docs/centroid-stability.md`), `centroid_representativeness` (does each centroid represent its own members, independent of size?), `dataset_profile` (quick diagnostic for a new dataset — prediction confidence, fallback rate).
- **`--shap_background` / `--shap_nsamples`** — SHAP `KernelExplainer` background-sample count and perturbation count for `rank_correlation`. `--shap_nsamples` defaults to the library's own `auto` formula (`2*F + 2048`) rather than a fixed value — see *Why SHAP* for why a fixed cap turned out to bias attributions on wide datasets. `--shap_background` defaults to 50; raising it only helps once `nsamples` is adequate (raising background alone, with `nsamples` still too low for `F`, made agreement worse in testing, not better). `--shap_repeats` reruns SHAP multiple times with different backgrounds to report its own Monte-Carlo noise, separate from sampling noise across explained examples.
- **`--from_saved_state <path>`** — reload a saved model state and rerun `--explain`/`--ablation` without retraining.
- **`--cat_combine {onehot,sum,concat}`** — categorical encoding (default `onehot`). `sum`/`concat` exist for comparison and backward compatibility with earlier checkpoints.
- **`--num_embedding {plr_lite,linear,ple}`** — numeric encoding (default `plr_lite`). `plr_lite` can be unstable on datasets with very few numeric columns (e.g. 1–5) — `linear`/`ple` are safer fallbacks there.
- **`--regroup_log_every <N>`** — how often (in epochs) to print the `[Regroup]` routing-health line during training (default 10). Lower it (e.g. 1–2) to inspect whether `active_ratio`/dead-code-reset activity is actually settling by the end of training, not just at coarse checkpoints.
- **`--loss_codebook_override <float>` / `--dropout_override <float>`** — controlled-ablation overrides: retrain with every other hyperparameter held at `best_params`, only this one value changed. Output filenames get a distinguishing tag (`..lcb<v>`, `..do<v>`) so runs don't overwrite each other. No effect combined with `--from_saved_state` (that path skips retraining entirely).

---

## Known limitations

- ①'s distinctive-feature contrast isn't equally sharp on every dataset.
- Group sizes can be naturally imbalanced, reflecting the data itself — not a failure mode by itself.
- `--from_saved_state` on GPU can show sub-decimal inference differences from a fresh run despite identical weights, likely from non-deterministic op scheduling (e.g. cuDNN).
- `plr_lite` numeric encoding (default) trades some calibration (logloss) for discrimination (accuracy/AUROC) relative to a plain linear projection — a pattern consistent with what the literature it's drawn from also reports. Datasets with very few numeric columns and few binary-classification samples are the most exposed; `--num_embedding linear` is the fallback for those.
- ③'s SHAP estimates inherit `KernelExplainer`'s own limitations: a feature-independence assumption during background perturbation (can be misleading on strongly correlated features) and a cost that scales with feature count, both discussed in *Why SHAP*. These are properties of the method, not of TabERA's use of it, but they bound how much weight ③'s attributions should be given relative to ①②'s architectural guarantees.
- On at least one dataset (credit-g, N_train=800), centroid routing shows persistent instability across training (`active_ratio` oscillating, dead-code reset never tapering off by the end of a run) that isn't explained by `dropout`, `routing_scale`, or accuracy cost — ruled out via controlled ablation — and doesn't reproduce on other datasets tested with the same mechanism (e.g. mfeat-zernike converges cleanly). Root cause open; see `docs/centroid-stability.md` §8.

---

## Contribution

- **Architecturally-guaranteed explanations (①②)** — read directly off the forward pass, not estimated afterward.
- **A clean separation of what's load-bearing from what's swappable**: ①② are architectural and central to the paper's claim; ③ is a standard post-hoc layer that could be any axiomatically-grounded attribution method — SHAP is used here for its categorical/numeric-agnostic behavior and Shapley's uniqueness guarantees, not because the paper's core claim depends on that specific choice.
- **A direct empirical check for feature interaction** (`--ablation interaction_check`), rather than assuming SHAP's theoretical interaction-handling automatically applies to a given dataset.
- **Null-simulation-based diagnostics for the centroid layer itself** (`centroid_geometry`, `centroid_representativeness`) — checking routing structure and per-centroid representativeness against empirical baselines rather than assuming trained structure is automatically meaningful, plus a training-time mechanism (dual commitment/codebook loss, CosFace-style reprojection, dead-code reset) that measurably improves it without a manual floor on any hyperparameter (`docs/centroid-stability.md`).
- **A documented limitation**: cross-group fallback rate depends on `K` vs. group size (`P = √N_train`) and can dominate small datasets.

---

## Components

| File | Component | Role |
|---|---|---|
| `libs/prototypes.py` | CentroidLayer | STE routing (scaled cosine similarity), KMeans++ init, dual commitment/codebook loss, dead-code reset, periodic sample-group regrouping — ① |
| `libs/tabera.py` | TabERA, MemoryBank, TabularEmbedder | Model, feature encoding, group-constrained KNN store — ② |
| `libs/evidence.py` | AttentionAggregator | `evidence_w`, direct retrieval path, task-aware label encoding — ② |
| `libs/supervised.py` | TabERAWrapper | Training loop, sample-group regrouping, early stopping |
| `libs/search_space.py` | HPO space | Optuna search space + auto `n_prototypes` |
| `libs/data.py` | TabularDataset | OpenML loader |
| `libs/eval.py` | Metrics | Accuracy, F1, AUROC, Logloss |
| `optimize.py` | HPO runner | Auto-sets `n_prototypes = sqrt(N_train)`, mirrors `reproduce.py`'s architecture |
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

## HPO parameters (searched via Optuna)

| Parameter | Range | Role |
|---|---|---|
| `embed_dim` | {64, 128, 256} | Embedding dim D |
| `embedder_layers` | {2, 3, 4} | ResidualMLP depth |
| `dropout` | 0.0–0.5 | — |
| `loss_diversity` | 5e-2–5e-1 | Centroid spread penalty |
| `loss_commitment` | 1e-2–1e-1 | Query→centroid pull (VQ commitment loss) |
| `loss_codebook` | 1e-2–1e-1 | Centroid→query pull (VQ codebook loss, same scale as `loss_commitment` — see `docs/centroid-stability.md`) |
| `lr` | 1e-4–1e-2 | — |
| `weight_decay` | 1e-6–1e-2 | — |
| `batch_size` | {128, 256, 512} | — |
| `routing_scale` | 1–20 (log) | Cosine routing logit scale (see Architecture, step 2) |
| `plr_freq_scale` * | 0.01–100 (log) | PLR(lite) periodic-embedding frequency init scale |
| `plr_n_frequencies` * | 8–96 | PLR(lite) frequencies per numeric column |
| `plr_out_dim` * | {4, 8, 16, 32} | PLR(lite) output dim per numeric column |

\* only searched when `--num_embedding plr_lite` (the default).

> `k` (KNN neighbors) is fixed at 16, not searched — measured hyperparameter
> importance (RandomForest on realized HPO trials, 22 datasets) placed it in
> the lowest tier, consistent with an earlier causal ablation
> (`--global_retrieve`) showing group-constrained retrieval's exact `k` value
> doesn't drive performance.
>
> `n_prototypes` (P) is **not** searched — auto-set as `P = sqrt(N_train)`
> (min 4). Ranges P≈12 (`lymph`, N=148) to P≈185 (`nomao`, N=34,465).
>
> The Optuna objective isn't raw accuracy/RMSE — it's scaled by a
> centroid-geometry penalty (0–5%, continuous, no fixed threshold) that
> discourages hyperparameter regions converging to random-or-worse routing
> structure, without constraining any search range directly. See
> `docs/centroid-stability.md` §2.

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py            # Model, MemoryBank, TabularEmbedder — ②
│   ├── prototypes.py        # CentroidLayer — ①
│   ├── evidence.py          # AttentionAggregator — ②
│   ├── supervised.py        # Training wrapper
│   ├── eval.py
│   ├── search_space.py      # HPO space
│   └── data.py              # OpenML loader
├── docs/                    # Design/technical notes (see centroid-stability.md)
├── optim_logs/ / figures/
├── optimize.py / reproduce.py / visualize_embeddings.py
└── requirements.txt
```

---

## References

- Gorishniy et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- Gorishniy et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*.
- Ye et al. (2024). Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later (ModernNCA). *arXiv:2407.03257*.
- Guo & Berkhahn (2016). Entity Embeddings of Categorical Variables. *arXiv:1604.06737*.
- Snell, Swersky & Zemel (2017). Prototypical Networks for Few-shot Learning. *NeurIPS*.
- Oreshkin, Rodríguez López & Lacoste (2018). TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning. *NeurIPS*.
- Zhang et al. (2019). AdaCos: Adaptively Scaling Cosine Logits for Effectively Learning Deep Face Representations. *CVPR*.
- van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS*.
- Wang et al. (2018). CosFace: Large Margin Cosine Loss for Deep Face Recognition. *CVPR*. — norm-fixing/reprojection convention adopted for `centroid_emb`, see `docs/centroid-stability.md` §4.
- Deng et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. *CVPR*.
- Dhariwal et al. (2020). Jukebox: A Generative Model for Music. *arXiv:2005.00341*. — dead-code "random restart" reset convention.
- Zeghidour et al. (2021). SoundStream: An End-to-End Neural Audio Codec. *arXiv:2107.03312*. — dead-code reset convention.
- Lu et al. (2026). Beyond Stationarity: Rethinking Codebook Collapse in Vector Quantization. *arXiv:2602.18896*. — controlled evidence that collapse persists even under ideal initialization, motivating dead-code reset as an ongoing (not just at-init) mechanism.
- Bengio et al. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS*.
- Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). *ICML*. — cited for why IG was ruled out for ③, see *Why SHAP*.
- Aas, Jullum & Løland (2019). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. *arXiv:1903.10464*. — background for the feature-independence caveat noted in *Why SHAP*.
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.
- Posner & Keele (1968). On the genesis of abstract ideas. *J. Exp. Psych.* 77(3).
- Bartlett (1932). *Remembering*. Cambridge University Press.
- Kahneman (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- McElfresh et al. (2023). When Do Neural Nets Outperform Boosted Trees on Tabular Data? *NeurIPS*.