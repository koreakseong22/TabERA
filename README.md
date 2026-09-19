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
| Correction | `d = β · p⊥`, where `p = normalize(q − c)` and `p⊥ = p − (p·c)c` (`correction_geometry="tangent"`). No second normalization: `‖d‖ = β‖p⊥‖`. `β = σ(β_raw)` is one learned scalar; `head_input_scale="unit"` sets `γ = 1`. |
| Decomposition | `z = (W_eff·c + b) + W_eff·d`, `W_eff = γW` — exact in logit space, since `W` is shared |
| Retrieval | k-NN inside `G(a)`, self excluded. Not an input to `z` |
| Objective | cross-entropy only; prototypes carry no loss |

The legacy `additive` arm, `d = β·normalize(q − c)`, is still selectable but is
not the final configuration (`libs/benchmark_config.py`).

---

## What an explanation shows

Printed by `analyze.py --explain` (`--from_saved_state <…_model_state.pt>` skips
training; `--explain_verbose` adds the researcher view). The default view
answers three user questions, one block each. Everything a user does not need
to answer them — sample and region ids, the full feature list, the logit
decomposition, cosine similarities, routing mass — is kept for
`--explain_verbose`, so the default screen stays short enough to read. The
default view says *group*; the code and the verbose view say *region* for the
same thing.

| | Question | Source |
|---|---|---|
| ① Group | Which group was this case assigned to, and what is that group like? | assignment `a`; group profile against the training set |
| ② Position | Where does this case depart from the members of that group? | within-group feature statistics |
| ③ Evidence | Which training cases were retrieved from that group? | `NN(q, G(a))` |

Analysing a saved checkpoint is deterministic: the post-refresh resync only
re-derives `sample_groups` from the clean embeddings against the centroids as
saved (`CentroidLayer.reassign_groups`). It used to call the training-epoch
`regroup_update`, whose random dead-prototype reinit altered the restored
model before evaluation and made two analyses of one checkpoint disagree.

Example: `credit-g`, predicting loan default. This is the verbatim output of
`analyze.py --openml_id 31 --seed 1 --explain --from_saved_state …` for test
sample 16.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TabERA Explanation  # 16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Prediction
     good — 94.8%

  Input summary
     checking_status = 0<=X<200, duration = 6, purpose = retraining, age = 39
     +16 more

  ① Your assigned group
     260 training cases; good 250 (96%), bad 10 (4%)

     Group profile
                       typical in group    this case
     checking_status   no checking (66%)   ≠ 0<=X<200
     age               38 (median)         39
     savings_status    <100 (42%)          ≠ no known savings

  ② How does this case compare within the group?
                     this case    typical in group   position
     purpose         retraining   radio/tv (37%)     seen in 2 of 260
     duration        6            13 (median)        lower than 87%
     credit_amount   932          1,984 (median)     lower than 87%

  ③ Retrieved past cases from this group

              checking_status   duration   purpose    age   outcome
     Case 1   no checking       12         radio/tv   35    good
     Case 2   no checking       21         radio/tv   41    good
     Case 3   no checking       24         radio/tv   53    good

     3 of 8 shown; 8 good, 0 bad
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The same few features recur from top to bottom on purpose. `Input summary`
and the columns of ③ are one shared set of at most four features — the first
two rows of ① and of ②, then the remaining shown rows of each — so no block
introduces a feature the others do not show, and a retrieved case can be
compared with the query and with the group on the same columns. It is called
a summary, not "key input", because nothing here is attribution: the rows are
a display selection. The rules below are display rules
(`libs/explain_format.py`), not model thresholds and not claims of the paper;
`print_explanation` only prints what `select_explanation_features` picked.

### ① Group

`260 training cases; good 250 (96%)` reads as "this group holds 260 training
samples, 96% of which were good", not "this group means good": a prototype is
a regional anchor, not a class representative, since assignment and the EMA
update never read labels. The prediction uses the hard assignment, not a
mixture.

The *group profile* is ranked **without looking at the case**, so the rows
cannot be the ones that happen to agree with it — and two `≠` marks are a
finding, not a fault. Rank and display are separate. The rank is a
distribution-shift score on one 0–1 scale for both feature kinds, so they can
be ordered together: the KS distance `sup_x |F_group(x) − F_all(x)|` for a
numeric feature and the total-variation distance `½ Σ_v |p_group(v) − p_all(v)|`
for a categorical one, both against the whole training store and both
computed where the model saw the values (the KS distance is invariant to the
quantile transform). The displayed value is the group median or the group
mode with its share.

Only after the top three are fixed is the case compared to them, and only
where a comparison is exact: `✓ same` when it holds the group's modal value,
`≠ value` otherwise. A numeric row shows the case's value with no mark,
because nothing equals a median — a tick there would stand for an unstated
band ("48 against a median of 42 — within what?") that the reader cannot
check. The band that does exist, the group's middle 50%, is a column of its
own under `--explain_verbose` rather than a tick.

⚠ These are typical values *observed* in the group's training members, not
the features that routed the case there. Routing is a cosine in the
representation, and nothing in raw feature space attributes it — which is
why the profile is not filtered to what the case shares with the group: that
would invent a cause.

With fewer than 10 training cases (the `n<10` cut of Figure 2) a median or a
mode is luck, so the profile and block ② are replaced by one line each
(`Group profile unavailable — only 8 training cases in this group.`); the
outcome counts and ③ remain.

### ② Position

Where this case departs from the members of its own group, as a table of
`this case, typical in group, position`. A numeric row qualifies when the
case lies below the 15th or above the 85th within-group percentile
(`2·|q − ½| ≥ 0.7` on the midrank percentile `q`); a categorical row when its
value is held by fewer than 10% of the group (an absent value counts). Rows
are ranked by that atypicality, at most three are shown, and a feature may
appear in both ① and ②: the two answer different questions — what the group
is like, and where this case sits in it — and meeting both criteria is itself
information (`duration` above). When nothing qualifies the block says
`This case is typical of its group.` instead of filling three rows: the more
typical the case, the shorter its explanation.

The reference column shows the scale that the position alone does not:
`lower than 87%` reads differently against a median of 1,000 than of 1,984.
Numeric references are the group **median** (a mean is dragged by the long
tail of a column like `credit_amount`), mapped back to original units, and
categorical references are the group mode with its share, since a mode at
37% and one at 90% are different claims.

A share of at most three cases is printed as a count (`seen in 2 of 260`),
not a percentage. In a group of 260 both one case and two round to "1%", and
in a group of 83 the percentage moves in 1.2-point steps: the figure reads
more precise than the data is, and the count is what a reader of a small
share wants anyway. A value no member holds says `not seen in group`.
`--explain_verbose` adds the middle 50% of the group and the cosine distance
to the group centre.

⚠ This is descriptive statistics, not attribution. "The prediction came out
this way because of this feature" is not a sentence these values support, and
nothing in ② explains how the group baseline became the final prediction —
only the correction term does. The default view shows that decomposition in
exactly one situation: when the correction changed the predicted class.
Without it ① (`bad 37 (60%)`) and the prediction (`good`) would read as a
contradiction, so one line states the two decisions — a fact, not an
attribution to the rows above, since the correction is latent:

```
     Group-based prediction: bad → Final prediction: good
     The case-specific adjustment changed the predicted class.
```

`--explain_verbose` always prints the full path:

```
Prediction path
   ...
   Region 25
     ↓ region prediction
   good — 94.7%
     ↓ sample-specific correction
   good — 94.8%
   Group-based prediction: good → Final prediction: good (unchanged)

   Logit decomposition (binary logit; + favours "good"):
     region       +2.8900
     correction   +0.0055
     final        +2.8955
```

`z = (W_eff·c + b) + W_eff·d` is exact in logit space; the two probabilities
are each a separate sigmoid/softmax and are placed side by side, never
subtracted. `W_eff·c + b` is identical for every sample in the group; the
correction `W_eff·d` is what separates them. Where `P < C` it does the
classifying instead — on a 100-class dataset with 35 prototypes, setting
`β = 0` drops accuracy from 0.725 to 0.256.

### ③ Evidence

The partition selects the pool, the query orders it: the retrieved cases are
the `k` nearest training cases inside the assigned group, ranked by cosine
similarity in the learned representation, shown on the shared columns with
their observed outcome. The block says *retrieved*, not *similar*: proximity
in the representation need not look like raw-feature similarity (above, the
query's `retraining` against three `radio/tv` cases), and the title must not
promise what the columns cannot show. Three of the `k` are shown by default
and the counts are given without shares (`8 good, 0 bad`) so the line does
not read as a vote; `--explain_verbose` lists all of them with training ids,
similarities and each case's full input. Retrieved labels are descriptive,
not predictive — TabERA does not vote over neighbours, and nothing in ③
enters `z`. `retrieve()` expands beyond the assigned group only when the group
cannot supply `k` candidates, and the block then says so in its title and a
note rather than calling the result within-group.

### `--explain_verbose`

Adds, in place: the sample and region ids, the full input in column order,
the prediction path and logit decomposition above, the ranking rules under ①
and ②, the middle-50% ranges and centre distance, all retrieved cases with
ids, similarities and full inputs, and the researcher diagnostics (correction
configuration, routing mass and runners-up, label entropies, and the
cross-group distinctive features from `label_all_groups`, which rank by a
different rule from ① and are named as such).

### Putting the explanation in a paper

`--explain_png <prefix>` typesets the view above as
`<prefix>_sample<i>.png` (300 dpi) and a vector `<prefix>_sample<i>.pdf`, on a
white page — for when the figure should be the explanation a user actually
sees, rather than a chart drawn from the same numbers.

It is the *same lines*, not a second layout. `print_explanation` emits its
output through one emitter; with a sink it collects those lines instead of
printing them, each tagged with the role it plays, and `libs/explain_png.py`
draws them in order. Nothing there recomputes, re-wraps or re-words anything,
and a round-trip test asserts that the collected rows rebuild the printed text
character for character. `--explain_verbose` therefore changes the image the
same way it changes the terminal (and makes it very tall: the full researcher
view of credit-g runs about 9 × 52 inches, against 6 × 7 for the default).

Typography follows the roles: titles and section headings, the prediction, and
a decision that changed are DejaVu Sans Bold; prose is DejaVu Sans; secondary
notes are DejaVu Sans in dark gray; feature names, values and every aligned
table are DejaVu Sans Mono. A value *inside* a monospace table is set in
DejaVu Sans **Mono** Bold rather than the proportional bold, since a
proportional face would move every column boundary after it — the monospace
bold has the identical advance and the renderer verifies that before using
it. The font files are pinned rather than resolved by family name, so a
system font cannot silently substitute itself between machines. The ━ runs
are drawn as rules spanning the content width instead of at their literal
60-character length, which is the one place the image departs from the
terminal's geometry.

`--explain_figure <prefix>` is a different thing: a composed publication
figure (bars, dumbbells, panels) from the same observer outputs and the same
formatter. Its panels predate the display rules above (it still shows the
contrast-case layout and z-ranked position rows), so do not cite it as
identical to the text until it is updated.

---

## How it works

| Stage | |
|---|---|
| Encoding | numeric → piecewise-linear embeddings (bin edges from the training split); categorical → one-hot; MLP → `q` |
| Assignment | `argmax cos(q, C)`; forward hard, backward straight-through |
| Prototypes | `P` observed embeddings sampled before epoch 1, then EMA with a tuned epoch-based half-life. No gradient. Unassigned ones reinitialised from an observed embedding |
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
logits bit-identical. It is also not *run* on the prediction path:
`TabERAWrapper.predict` / `predict_proba` / `_forward_batched` call
`forward(..., retrieve=False)`, which never touches the memory bank, so
prediction-only inference routes over the `P = ⌊√N⌋` prototypes only.
Explanation and diagnostics callers use the default (`retrieve=None`) and
still get `topk_idx` / `neighbor_mask`. `tests/test_prediction_retrieval_free.py`
pins both facts. Whether retrieval *could* improve prediction was measured
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
first. Both scripts default to the final architecture (`tangent`,
`head_input_scale=unit`) through one shared `FINAL_CONFIG`; no flags needed.
The default `early_stop_metric=val_loss` uses batch-averaged validation loss,
patience 20, and the terminal model without best-checkpoint restore, matching
the completed 105-run benchmark protocol.
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
| `beta_lr_mult` | 1–30, log |
| `ema_timescale` | epoch half-life {0.5, 1, 3, 10} |
| `num_bins` | integer 2–128 |
| `ple_d_embedding` | integer 8–32, step 4 |
| *(plr_lite only)* | `plr_freq_scale`, `plr_n_frequencies`, `plr_out_dim` |

**Fixed by rule**, not tuned.

| | | |
|---|---|---|
| `P` | `floor(√N_train)` | capacity tied to dataset size |
| `k` | 8 | explanation budget — outside the prediction path, so it cannot move the objective |
| `batch_size` | MultiTab size rule | derived from training-set size |
| `routing_scale` | `√2·log(P − 1)` | derived from `P` |

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

The controlled dynamics pilot fixes PLE at 8 bins / width 12, encoder width 128, layers 2, dropout
0.1, lr 3e-4 and weight decay 1e-5. Only `beta_lr_mult` (continuous log
1–30) and `ema_timescale` (`hl_05`, `hl_1`, `hl_3`, `hl_10`)
are searched. These are the same dynamics ranges as the final nine-dimensional
`betaema1_plehpo_epochhl` recipe. The joint recipe also searches `num_bins` (2–128)
and `ple_d_embedding` (8–32, step 4), with initial values 8 and 12.
See [the reproduction contract](docs/MULTITAB_REPRODUCTION.md#ple-hpo-recipe)
for the full space and separation from the previous fixed-PLE `betaema1` results.

```bash
python optimize.py --openml_id 31 --seed 1 --n_trials 25 --validation_only --pilot_space dynamics2d --savepath pilot_dynamics2d
```

The five constants are direct model parameters, recorded as study/trial
`fixed_hyperparameters`, not one-choice Optuna distributions. Trial 0 uses
multiplier 1 and a nominal EMA half-life of 1 epoch. Names include
`..validation_only..pilot=dynamics2d`, separate from the joint pilot and
final HPO. The mode requires validation-only evaluation and the fixed final
architecture. Omitting `--pilot_space dynamics2d` searches all nine HPs with PLE.

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
