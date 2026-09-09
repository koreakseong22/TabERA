# TabERA

**Tabular Explainable Retrieval Architecture**

A tabular classifier whose latent partition is part of the forward pass. Every
sample is hard-assigned to one of `P` prototypes, and that single assignment
determines both the prediction baseline and the pool that evidence is retrieved
from — so the explanation describes the same structure the prediction used.

```
x ─→ Encoder ─→ q ─→ argmax cos(q, C) ─→ prototype a
                                          │
                          ┌───────────────┴───────────────┐
                  c = C[a]                        G(a) = its members
                          │                               │
                h = c + d                     NN(q, G(a)), k = 8
              z = W·(γh) + b                              │
                          ▼                               ▼
                     prediction                       explanation
```

| | |
|---|---|
| Prediction | `z = W·(γh) + b`, `h = c + d` |
| Correction | `d = β · p⊥ / max(‖p⊥‖, ε)`, where `p = normalize(q − c)` and `p⊥ = p − (p·c)c` — the unit tangent direction at `c` toward `q` (`correction_geometry="unit_tangent"`). `‖d‖ = β` for every non-degenerate sample (`‖p⊥‖ ≥ ε`, `ε = 1e-6`); the distance from `c` does not enter `d`, only the direction does. `β = σ(β_raw)` is one learned scalar; `γ` is the head input scale (`head_input_scale="auto"`) |
| Decomposition | `z = (W_eff·c + b) + W_eff·d`, `W_eff = γW` — exact in logit space, since `W` is shared |
| Retrieval | k-NN inside `G(a)`, self excluded. Not an input to `z` |
| Objective | cross-entropy only; prototypes carry no loss |

The legacy `additive` arm, `d = β·normalize(q − c)`, is still selectable but is
not the final configuration (`libs/benchmark_config.py`).

---

## What an explanation shows

Printed by `analyze.py --explain` (`--from_saved_state <…_model_state.pt>` skips
training; `--explain_verbose` adds the researcher diagnostics). The region and
the prediction split are read off the computation that produced the
prediction; evidence and position run beside it and are not inputs to it.

`--explain_figure <prefix>` writes the explanation as a paper figure —
`<prefix>_sample<i>.png` at 300 dpi and a vector `.pdf` — drawn from the same
observer outputs and the same formatter as the text. Prediction, ① and ③ match
the text exactly; the figure's ② panel still shows the older contrast-only
comparison rather than the per-case cards, so do not cite it as identical
until it is updated.

Analysing a saved checkpoint is deterministic: the post-refresh resync only
re-derives `sample_groups` from the clean embeddings against the centroids as
saved (`CentroidLayer.reassign_groups`). It used to call the training-epoch
`regroup_update`, whose random dead-prototype reinit altered the restored
model before evaluation and made two analyses of one checkpoint disagree.

| | Question | Source |
|---|---|---|
| Prediction | How did the region baseline become the final prediction? | `W_eff·c + b` vs `z` |
| ① Region | Where does this sample belong? | assignment `a` |
| ② Evidence | Which real cases are nearest within that region? | `NN(q, G(a))` |
| ③ Position | Where does this sample sit relative to the members of its region? | region feature statistics, cosine distance to `c` |

Example: `credit-g`, predicting loan default. This is the verbatim output of
`analyze.py --openml_id 31 --seed 1 --explain --from_saved_state …` for the
first test sample.

### Prediction

```
Prediction
   → good — 58.1%
   Region baseline:  good 63.9%
   Final prediction: good 58.1%
   Correction:       weakens "good" relative to "bad"
```

`Region baseline` is `σ(W_eff·c + b)` and `Final prediction` is `σ(z)`, both
read on the finally predicted class. `W_eff·c + b` is identical for every
sample in the region; the correction `W_eff·d` is what separates them. Where
`P < C` it does the classifying instead — on a 100-class dataset with 35
prototypes, setting `β = 0` drops accuracy from 0.725 to 0.256.

⚠ The two probabilities are shown side by side, never subtracted. The
decomposition is exact in logit space, but each probability is a separate
sigmoid/softmax, so their difference is not a contribution in probability
space. The `Correction:` line is the *direction* only — the sign of the
correction logit — and it is printed for binary tasks only, since with more
classes a single term shifts all of them at once and naming one "main
alternative" would be a choice with no basis. `--explain_verbose` prints the
three logits themselves:

```
Logit decomposition (exact in logit space; binary logit, + favours "good"):
  region baseline +0.5723 · correction -0.2458 · final +0.3265
```

### ① Region

```
① Predictive region
   Region 5 — 93 training cases
   Outcomes: good 70/93 (75%) · bad 23/93 (25%)
```

A prototype is a regional anchor, not a learned class representative:
assignment and the EMA update are both class-agnostic. Read it as "this group
holds 93 training samples, 75% of which were good", not "this group means
good". The prediction uses the hard assignment, not a mixture; the routing
spread (`--explain_verbose`) is a diagnostic of assignment ambiguity.

### ② Evidence

```
② Similar past cases — evidence only
   Retrieved: good 7/8 (88%) · bad 1/8 (12%)
   Closest cases:
     #269    good  ·  similarity 1.000
       matches: job = unskilled resident · purpose = furniture/equipment
       differs: checking_status: <0 ↔ 0<=X<200

     #507    good  ·  similarity 0.999
       matches: personal_status = male single · credit_history = existing paid
       differs: checking_status: <0 ↔ 0<=X<200

   Closest contrasting case:
     #577    bad  ·  similarity 0.999 · contrast
       matches: own_telephone = yes · personal_status = male single
       differs: checking_status: <0 ↔ >=200 · credit_history: existing paid ↔ no credits/all paid

   Similarity is measured in the embedding; shown feature values are descriptive.
```

The partition selects the pool, the query orders it. Retrieved labels are
descriptive, not predictive — TabERA does not vote over neighbours, and
nothing in ② enters `z`. The local distribution is always shown against the
group distribution, since `7/8` means nothing without knowing the group is
already `75%`. Without a contrasting case the first line says so
(`· no contrast among 8`).

The display policy is visible in the headings: the two closest cases, plus
the closest case with a different label as an extra slot when it falls outside
that budget — so counter-evidence is never hidden by presentation, and the
evidence block stays a minority of the explanation. Each case carries a small
card so it reads as an example rather than an id. The rows are ranked, never
thresholded, and none is padded to its budget: `differs` is reserved first
(largest Gower gap; two for the contrast case), `matches` are exact equalities
with exact categorical matches ordered by how rare the shared value is in the
region (a match on a value 2% of the region holds says more than a match on
the mode), and `closest values` appears only when there is no exact match at
all — rank language on purpose, since without a threshold nothing certifies
that the smallest gap is "similar". The gap is measured where the model saw the
values, the quantile space held in `FeatureStore`; only the displayed numbers
are mapped back to original units. The card describes values the two cases
hold; it does not explain why their embedding cosine is high.

### ③ Position

```
③ Position relative to the region
   Values that stand out:
     • installment_commitment = 1
       region reference: 4
       lower than 85% of region cases · equal to 15%
     • duration = 6
       region reference: 12
       lower than 91% of region cases · equal to 9%
     • checking_status = <0
       17% in region · most common: no checking 32%
   Distance from region centre:
     Farther than 66% of region training cases
```

Locates the sample among the training members of its own region, in two
spaces.

*Values that stand out* is raw feature space. Numeric features are ranked by
within-region |z| and stated as exact shares of region training cases
(`higher/lower than X% · equal to Y%`) rather than a midrank percentile, so the
sentence stays true when a discrete feature ties. `region reference` is the
region mean taken in quantile space and mapped back to original units — it is
a representative value, not the arithmetic mean of the original column.
Categorical features show the value's frequency in the region and the region
mode; values equal to the mode are not shown in this section.

*Distance from region centre* is representation space: the cosine distance
`1 − cos(q̂, ĉ)` and its rank among the region's training cases — the space the
assignment was made in. It is not a confidence and not a typicality score;
whether the sample is "atypical" is left to the reader, since a region need not
be spherical. The distance does not enter the prediction either: `d` is a unit
tangent direction scaled by the single scalar `β`, so only the *direction* from
`c` toward `q` reaches `z`, never how far `q` is from `c`. It appears only when
`--refresh_on_best` is on (the default): otherwise memory holds training-time
embeddings taken under a dropout mask while the query is deterministic, and the
rank would be against a different representation.

⚠ This is descriptive statistics, not attribution. "The prediction came out
this way because of this feature" is not a sentence these values support, and
nothing in ③ explains the shift shown under Prediction — only the correction
term does.

---

## How it works

| Stage | |
|---|---|
| Encoding | numeric → piecewise-linear embeddings (bin edges from the training split); categorical → one-hot; MLP → `q` |
| Assignment | `argmax cos(q, C)`; forward hard, backward straight-through |
| Prototypes | `P` observed embeddings sampled before epoch 1, then EMA (`decay 0.99`). No gradient. Unassigned ones reinitialised from an observed embedding |
| Prediction | `h = c + β·normalize(q − c)`, `W` shared between the terms |
| Retrieval | k-NN within `G(a)`, self excluded |

Two design choices carry weight. **`W` is shared**: with separate matrices the
optimiser grew one branch to evade the constraint, leaving `β` meaningless.
**`q − c` is normalised**: `‖c‖ = 1` while `‖q‖` is not, so the raw difference
would let query magnitude swamp the prototype.

Because `‖q‖ ≫ ‖c‖`, that term behaves as a **query-direction correction**, not
a literal displacement from the prototype. Learned `β` ranges 0.10–0.73 across
the evaluated datasets.

Nothing reads labels except the cross-entropy loss: initialisation, assignment
and the EMA update are all class-agnostic. Gradient reaches the encoder
(through the straight-through routing), `W` and `β` — not the prototypes, the
memory bank, or the retrieval.

---

## Two paths, one partition

The assignment is the branching point.

| | Prediction branch | Evidence branch |
|---|---|---|
| Takes | the prototype `c` | its members `G(a)` |
| Gives | region baseline + within-region correction | the `k` nearest training rows |
| Feeds `z` | yes | no |

`W·c` alone yields at most `P` distinct outputs, one per region; the correction
lets samples separate inside a region while the region still sets the baseline.

The evidence branch does not feed the prediction — changing `k` leaves the
logits bit-identical. Whether retrieval *could* improve prediction was measured
across several fusion designs; see `TABERA_V3_ARCHITECTURE.md` §14.

---

## Results

Ten OpenML datasets, five seeds each.

| Dataset | Accuracy | AUROC | Log loss |
|---|---:|---:|---:|
| 31 — credit-g | 0.7580 | 0.6975 | 0.5741 |
| 54 — vehicle | 0.7906 | 0.9449 | 0.5718 |
| 934 — socmob | 0.9517 | 0.9597 | 0.3664 |
| 1493 — plants-texture | 0.7200 | 0.9896 | 1.1772 |
| 14 — mfeat-fourier | 0.8170 | 0.9683 | 0.5792 |
| 22 — mfeat-zernike | 0.8210 | 0.9703 | 0.6133 |
| 41143 — jasmine | 0.7860 | 0.8529 | 0.5235 |
| 46 — splice | 0.9580 | 0.9832 | 0.3629 |
| 1043 — ada_agnostic | 0.8136 | 0.8563 | 0.4205 |
| 1489 — phoneme | 0.8932 | 0.9449 | 0.3012 |
| **Mean** | **0.8309** | **0.9168** | **0.5490** |

`ds=1493` has 100 classes and 35 prototypes — the clearest case of the gap
between region-level prediction and class-level resolution.

---

## Running it

```bash
pip install -r requirements.txt

python optimize.py  --openml_id 31 --seed 1 --n_trials 100
python reproduce.py --openml_id 31 --seed 1 --deterministic --train_seeds 1 2 3 4 5
python analyze.py   --openml_id 31 --seed 1 --deterministic --explain
```

`optimize.py` writes the study file `reproduce.py` reads back, so it runs
first. Both scripts default to the final architecture (`unit_tangent`,
`head_input_scale=auto`) through one shared `FINAL_CONFIG`; no flags needed.
Pass `--correction_geometry additive --head_input_scale unit` only to run the
legacy arm.
`--calibration_analysis` and `--linear_probe` add diagnostics.

**Searched** — 100 TPE trials per dataset and seed.

| | |
|---|---|
| `embed_dim` | {64, 128, 256} |
| `embedder_layers` | 1–4 |
| `dropout` | 0.0–0.5, step 0.05 |
| `lr` | 1e-4 – 1e-2, log |
| `weight_decay` | 1e-6 – 1e-2, log |
| *(plr_lite only)* | `plr_freq_scale`, `plr_n_frequencies`, `plr_out_dim` |

**Fixed by rule**, not tuned.

| | | |
|---|---|---|
| `P` | `floor(√N_train)` | capacity tied to dataset size |
| `k` | 8 | explanation budget — outside the prediction path, so it cannot move the objective |
| `batch_size` | 256 | fixed protocol |
| `routing_scale` | `√2·log(P − 1)` | derived from `P` |
| `ema_decay` | 0.99 | |

Ablation flags are listed by `--help`. Variants no longer in this code — the
alternative prediction heads, the neighbourhood regulariser, the aggregator —
are frozen in `legacy/v3ema2_full/`.

```bash
python visualize_tabera.py --openml_id 54 --seed 1
```

Writes five diagnostic panels to `figures/seed=1/` for one checkpoint: the
prediction decomposition per sample, the evidence chain, the prototype
partition, per-prototype profiles, and the pairwise prototype geometry.

---

## Controlled dynamics pilot

The current `betaema1` controlled pilot fixes width 128, layers 2, dropout
0.1, lr 3e-4 and weight decay 1e-5. Only `beta_lr_mult` (continuous log
1–30) and `ema_timescale` (`legacy_099`, `hl_05`, `hl_1`, `hl_3`, `hl_10`)
are searched. These are the same dynamics ranges as the final seven-dimensional
recipe; the pilot does not change that recipe.

```bash
python optimize.py --openml_id 31 --seed 1 --n_trials 25 --validation_only --pilot_space dynamics2d --savepath pilot_dynamics2d
```

The five constants are direct model parameters, recorded as study/trial
`fixed_hyperparameters`, not one-choice Optuna distributions. Trial 0 uses
multiplier 1 and legacy decay 0.99. Names include
`..validation_only..pilot=dynamics2d`, separate from the joint pilot and
final HPO. The mode requires validation-only evaluation and the fixed final
architecture. Omitting `--pilot_space dynamics2d` searches all seven HPs.

A proposed small panel is 31, 54, 1067, 1493 and 151 (electricity), with
seed 1 and 20–30 trials each. Compare validation improvement over trial 0,
beta trajectories, reinit/churn/utilization and class-margin diagnostics.
This is an adequacy check around one common anchor, not a proof of the
globally optimized benefit of the two axes. Do not fix the pilot's winning
beta/EMA values in final HPO or expand bounds based only on boundary frequency.

No test predictions or metrics are computed in validation-only mode. The
loader still creates all splits and the provenance contract hashes them;
this is not a claim that the test split is never loaded. Use only validation
evidence for the one-time adequacy check, then freeze the final ranges.

## Layout

```
libs/
  tabera.py         model, MemoryBank, TabularEmbedder
  prototypes.py     CentroidLayer — routing, EMA update, dead-prototype recovery
  supervised.py     training loop
  search_space.py   Optuna space, study naming
  diagnostics.py    read-only observers over a forward pass
  eval.py           metrics
  data.py           OpenML loading
optimize.py         hyperparameter search
reproduce.py        train / evaluate / explain
visualize_tabera.py per-dataset diagnostic panels
legacy/v3ema2_full/ frozen pre-cleanup code, for reproducing the ablations
tools/              golden regression, structural audit, smoke harnesses
```
