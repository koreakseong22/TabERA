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
          h = c + β·normalize(q − c)          NN(q, G(a)), k = 8
                 z = W·h + b                              │
                          ▼                               ▼
                     prediction                       explanation
```

| | |
|---|---|
| Prediction | `z = W·h + b`, `h = c + β·normalize(q − c)` |
| Decomposition | `z = (W·c + b) + W·(β·r)` — exact, since `W` is shared |
| Retrieval | k-NN inside `G(a)`, self excluded. Not an input to `z` |
| Objective | cross-entropy only; prototypes carry no loss |

---

## What an explanation shows

Printed by `reproduce.py --explain`. The region and the prediction split are
read off the computation that produced the prediction; evidence runs beside it.

| | Question | Source |
|---|---|---|
| Prediction | Where did the prediction come from? | `W·c + b` vs `z` |
| ① Region | Where does this sample belong? | assignment `a` |
| ② Evidence | Which real cases are nearest within that region? | `NN(q, G(a))` |
| ③ Region profile | What is this region like, and where in it is this sample? | group feature statistics |

Example: `credit-g`, predicting loan default.

### Prediction

```
Region baseline prediction   bad 72.1%   (shared by every sample in this region)
Final prediction             bad 73.8%
Sample-specific component    strengthens "bad" relative to "good"
```

`W·c + b` is identical for every sample in the region; the sample-specific term
is what separates them. Where `P < C` it does the classifying instead — on a
100-class dataset with 35 prototypes, setting `β = 0` drops accuracy from 0.725
to 0.256.

⚠ The two probabilities are shown side by side, never subtracted. The
decomposition is exact in logit space, but each probability is a separate
softmax, so their difference is not a contribution in probability space. The
*direction* is exact, being the sign of the logit term — and it is reported for
binary tasks only, since with more classes a single term shifts all of them at
once and naming one "main alternative" would be a choice with no basis.

### ① Region

```
Assigned prototype: "Centroid_6"
Label distribution: "bad" 124/160 (78%), "good" 36/160 (22%)
Routing:  Centroid_6 14.6% (assigned) · Centroid_20 13.6% · others 62.2%
Characteristic: installment_commitment = 4, other_parties = none (95%)
```

The routing spread is a diagnostic of assignment ambiguity — the prediction
uses the hard assignment, not a mixture. A prototype is a regional anchor, not
a learned class representative: assignment and the EMA update are both
class-agnostic. Read it as "this group holds 160 samples, 78% of which
defaulted", not "this group means default".

### ② Evidence

```
neighbourhood (k=8)   bad 7/8 (88%)      H(label) 0.377
whole group (n=160)   bad 124/160 (78%)  H(label) 0.533
→ relative ambiguity 0.71

Outcome-matched      #2  sim 0.984 → bad   [train #400]
Outcome-contrasting  #1  sim 0.985 → good  [train #293]
                         differs: purpose business → furniture, duration 48 → 18
```

The partition selects the pool, the query orders it. Retrieved labels are
descriptive, not predictive — TabERA does not vote over neighbours. The local
distribution is always shown against the group distribution, since `7/8` means
nothing without knowing the group is already `78%`.

### ③ Region profile

Translates the assigned region into the original feature space and locates the
sample within it.

```
against the group (n=160)
  credit_amount = 5,951   (group typical 2,382,  2.5x, top 9%,  |Δmean|/σ 0.71)
  duration      = 48      (group typical 18,     2.7x, top 4%,  |Δmean|/σ 0.09)

position in the representation (n=160)
  distance to region centre 0.043   (closer to the centre than 60% of the region)
```

`|Δmean|/σ` is the standardised difference between the region mean and the
global mean, taken in units of the **global** feature standard deviation. It is
a descriptive statistic, not a threshold: it appears on every line and hides
nothing. Near 0 means the region's distribution barely differs from the
dataset, so "unusual within this region" reads the same as "unusual overall" —
the percentile beside it then carries no region-specific information. Gating on
it would need a cut-off nobody can justify, and a reader could not tell a
hidden feature from an absent one.

The representation position is a cosine distance to the region centre and its
rank within the region — not a confidence, and not a typicality score. Whether
that makes the sample atypical is left to the reader, since a region need not
be spherical. It appears only when `--refresh_on_best` is on: otherwise memory
holds training-time embeddings taken under a dropout mask while the query is
deterministic, and the rank would be against a different representation.

⚠ This is descriptive statistics, not attribution. "The prediction came out
this way because of this feature" is not a sentence these values support.

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
python reproduce.py --openml_id 31 --seed 1 --deterministic --explain
```

`optimize.py` writes the study file `reproduce.py` reads back, so it runs
first. Defaults reproduce the architecture above; no flags needed.
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

`TABERA_V3_ARCHITECTURE.md` records the measurement behind each design
decision. `REFERENCES.md` lists prior work by the component it supports.

---

## References

| | |
|---|---|
| Gorishniy et al. (2022), *NeurIPS* | piecewise-linear embeddings |
| van den Oord et al. (2017), *NeurIPS* | hard assignment with straight-through gradient, EMA codebook |
| Razavi et al. (2019), *NeurIPS* | EMA as the default update |
| Bengio et al. (2013), *arXiv:1308.3432* | straight-through estimator |
| Dhariwal et al. (2020), *arXiv:2005.00341* | restarting unused codes |
| Chen et al. (2019), *NeurIPS* | prototype-based prediction rather than post-hoc explanation |
| Kim et al. (2016), *NeurIPS* | why contrasting cases belong beside supporting ones |