# TabERA

**Tabular Explainable Retrieval Architecture**

A tabular classifier that exposes three model-native views of each prediction:
the latent region the sample belongs to, the training cases retrieved from that
region, and how the sample departs from its region-level prediction baseline.

> *Which region does this sample belong to?*
> *Which real cases are nearest within that region?*
> *How does it differ from the region's baseline?*

---

## The idea

TabERA does not aim to attribute a prediction to individual input features. It
exposes the latent structure the model itself used: the assigned region, nearby
training cases within that region, and the sample-specific correction to the
region baseline.

Many tabular classifiers predict directly from a learned representation,
leaving no such structure to inspect alongside the prediction.

TabERA makes the latent partition part of the forward computation. Every sample
is hard-assigned to one of `P` prototypes, and that single assignment `a`
determines the prediction baseline `c` and the retrieval pool `G(a)`.

```
x ─→ Encoder ─→ q ─→ argmax cos(q, c) ─→ prototype a
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                  c = prototype[a]                G(a) = its members
                          │                               │
          h = c + β·normalize(q − c)          NN(q, G(a)), k = 8
                 z = W·h                                  │
                          ▼                               ▼
                     prediction                       explanation
```

The same latent partition serves two roles: it provides the region-level
baseline for prediction and defines the candidate pool for evidence retrieval.
Retrieval does not run a second, dataset-wide neighbour search after the
prediction is made.

---

## What an explanation looks like

Three views, printed by `reproduce.py --explain`:

```
REGION      Where does this sample belong?
EVIDENCE    Which real training cases are nearest within that region?
DEVIATION   How does it differ from the region's prediction baseline?
```

Region and deviation are read off the computation that produced the
prediction; evidence is the retrieval, which runs beside it. The example is
`credit-g`, predicting loan default.

### ① Region — prototype assignment

```
Assigned prototype: "Centroid_6"
Prototype label distribution: "bad" 124/160 (78%), also "good" 36/160 (22%)
Routing distribution:
  • Centroid_6            14.6%  (assigned)
  • Centroid_20           13.6%  ("bad" 10/11 (91%))
  • Others                62.2%
Characteristic features:
  numeric:     installment_commitment = 4
  categorical: other_parties = none (95%), housing = own (56%)
```

Which prototype the sample was routed to, what the label distribution of that
group is, and which feature values are characteristic of the group relative to
the others.

The routing distribution shows how concentrated the soft assignment is around
the chosen prototype. Here the top two are one percentage point apart, which
means a less decisive assignment than one landing well inside a single region.
It is a diagnostic of assignment ambiguity — the prediction itself uses the hard
assignment, not a mixture over prototypes.

A prototype is a regional anchor, not an explicitly learned class
representative. Assignment and the EMA update are both class-agnostic — the
prototypes follow the geometry of the learned representation rather than being
optimised to stand for particular labels. The reading is "this group holds 160
samples, 78% of which defaulted", not "this group means default".

### ② Evidence — retrieved cases

```
neighbourhood (k=8)   bad 7/8 (88%), good 1/8 (12%)   H(label) 0.377
whole group (n=160)   bad 124/160 (78%)               H(label) 0.533
→ relative ambiguity 0.71

Outcome-matched cases
  #2  similarity 0.984  → bad   [train #400]
       checking_status < 0, credit_history existing paid, employment < 1

Outcome-contrasting cases
  #1  similarity 0.985  → good  [train #293]
       differs: purpose    business → furniture/equipment
       differs: duration   48 → 18
```

The retrieved labels are descriptive, not predictive — TabERA does not vote
over neighbours to produce the prediction.

These cases provide examples of local similarity, not causal evidence for the
predicted label.

What is shown: the `k` nearest training rows inside the assigned group, ranked
by cosine similarity — the partition selects the pool, the query orders it —
as outcome-matched and outcome-contrasting cases. For
each contrasting case, the columns where it differs from the query.

Above them, the label distributions and entropies of the neighbourhood and of
the whole group, and their ratio:

```
relative label entropy = H(labels of the retrieved cases)
                       / H(labels of the group)

  ~ 1   neighbourhood entropy similar to its region
  < 1   lower local label entropy
  > 1   higher local label entropy
```

Here 0.71 — less mixed than the group as a whole. This is a local-structure
diagnostic, not a calibrated confidence.

Because prototypes form without label supervision, one can hold several
outcomes. Neighbour labels are therefore presented as descriptive rather than
predictive: reading a majority as support would present sampling noise from the
group distribution as a reason. The cases are intended for inspection rather
than as predictive votes; the quantity to read is the ratio.

### ③ Deviation — sample-specific correction

```
prototype-only prediction:  bad 70.4%
this sample:               bad 72.3%   (+1.8pp)

against the group (n=160)
  duration = 48           group typical 24,    2.0×,  top 9% within group
  credit_amount = 4,308   group typical 2,382, 1.8×,  top 26% within group
  purpose = business      10% of the group  |  group mode: new car (36%)
```

The deviation is the sample-specific correction to the region-level baseline:
what the model predicts from the prototype alone, what that becomes after the
correction, and — separately — where this sample sits within its group on each
feature.

The first part is model-exact rather than an attribution approximation. The
head is a single `Linear` with `W` shared across both terms, so

```
z = W·c + β·W·normalize(q − c)
```

holds to floating point: the prototype baseline and the correction are the
prediction, decomposed rather than approximated. When the correction changes
the argmax, the output says so.

The second part positions the sample within its group; these are not feature
attributions. It says the sample's `duration` is twice its group's typical
value — it does not say that is why the correction moved the logits. The two
are printed together because both concern the same group, and they should not
be read causally.

The group typical value is the inverse transform of a mean taken in quantile
space, not an arithmetic mean.

---

## How the model produces them

**Encoding.** Numeric columns go through piecewise-linear embeddings: quantile
bin edges computed once from the training split, one trainable embedding per
bin per column. Categorical columns are one-hot — a raw integer code carries no
ordering. An MLP maps the concatenation to a query embedding `q`.

The encoder produces the representation that routing and retrieval operate on;
it does not itself provide feature-level explanations.

**Assignment.** `q` goes to the prototype with the highest cosine similarity.
The forward pass uses the hard one-hot assignment; the straight-through
backward path uses the corresponding soft routing probabilities, so the encoder
trains through the routing even though the choice is discrete.

The prototypes receive no gradient. They start as `P` training embeddings
sampled uniformly without replacement from what the freshly initialised encoder
produces, before the first epoch — every prototype is an observed
representation, not a synthetic point — and are then maintained by an
exponential moving average over the embeddings assigned to them, following the EMA codebook-update pattern from
VQ-VAE-style discrete representation learning. Prototypes that stay unassigned
for several epochs are reinitialised from an observed embedding to keep the
memory populated. Like the initialisation, this step does not read labels, so
initialisation, assignment and the EMA update are all class-agnostic.

The EMA keeps the prototype memory tracking the representation as the encoder
moves, without introducing an additional prototype loss alongside the task
objective.

The prototype count is fixed by the rule `P = floor(√N_train)` rather than
tuned per dataset, keeping partition capacity tied to dataset size.

**Prediction.** `h = c + β·normalize(q − c)`, with `β` a learned scalar and `W`
shared between the two terms.

Two design choices matter here. In the architectural ablations, separate output
matrices let one branch dominate the other; sharing `W` leaves `β` as the
explicit scalar controlling their relative contribution. And `c` is unit-norm
while `q` is not, so adding `q − c` unnormalised would let the query magnitude
swamp the prototype.

Because `‖q‖` is much larger than `‖c‖`, the normalised term behaves as a
query-direction correction rather than a literal geometric displacement from
the prototype.

**Retrieval.** k-NN inside the assigned prototype's members, self excluded.
`k = 8` is fixed — it sets how many cases an explanation shows.

---

## Two paths, one partition

The assignment is the branching point: prediction uses the assigned prototype
as its region-level baseline, while evidence retrieval uses the prototype's
members as its candidate pool.

**The prediction branch** takes the prototype. `c` is the EMA-maintained
prototype for the region — it tracks the evolving centre of the embeddings
assigned there — and so provides a region-level representation shared by its
members; `β·normalize(q − c)` supplies the within-region variation. Across the
evaluated datasets, learned `β` values range from 0.10 to 0.73.

The effect is a soft lookup table. `W·c` alone would give at most `P` distinct
outputs, one per region; the correction lets samples inside a region separate
while the region still sets the baseline. On a 100-class dataset with 35
prototypes, setting `β = 0` on the trained model drops accuracy from 0.725 to
0.256 — the within-region correction matters most when there are more classes
than prototypes.

**The evidence branch** takes the members. `G(a)` is the set of training rows
sharing the prototype, and the retrieval ranks them by `cos(q, ·)`: **the
partition selects the pool, the query orders it.** The top `k` come back with
their raw feature values and labels.

This branch does not feed the prediction — the logits are computed without it,
and the neighbours are shown alongside the decision as the cases the model's
own partition places nearest. Whether they could also improve the prediction
was measured across a range of fusion designs; the result is in
`TABERA_V3_ARCHITECTURE.md` §14.

Because the same `a` drives both branches, the retrieved cases come from the
same latent region that supplies the prediction baseline. A person reading them
is reading that region, not a similarity search run afterwards.

## Training

Training uses a single cross-entropy objective on the logits. The prototypes
are not optimised by a separate prototype loss; they are updated by EMA outside
the gradient path, and unassigned ones are recovered by reinitialisation.

The gradient reaches the encoder (through the straight-through routing), `W`,
and `β`. It does not reach the prototypes, the memory bank, or the retrieval —
the prototypes move by EMA rather than backpropagation.

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

# hyperparameter search
python optimize.py --openml_id 31 --seed 1 --n_trials 100

# train and evaluate over five seeds
python reproduce.py --openml_id 31 --seed 1 --deterministic \
    --train_seeds 1 2 3 4 5 --export_centroid_retrieval_behavior

# print explanations
python reproduce.py --openml_id 31 --seed 1 --deterministic --explain
```

The default configuration reproduces the architecture described above; no
flags are needed.

Five hyperparameters are searched per dataset: `embed_dim` ∈ {64, 128, 256},
`embedder_layers` 1–4, `dropout` 0.05–0.45, and log-uniform `lr` and
`weight_decay`. The rest is fixed by rule: `P = floor(√N_train)`, `batch_size = 256`,
`k = 8`, `ema_decay = 0.99`.

`--fusion_mode` selects among the prediction-head variants used for the
architectural studies; `--help` lists those and the other ablation flags.

### Figures

```bash
python visualize_tabera.py --openml_id 54 --seed 1
```

Writes five diagnostic panels to `figures/seed=1/` for one trained checkpoint.
Two of them map onto the explanation levels above: the prediction decomposition
is `z = W·c + β·W·(q−c)` drawn per sample, and the evidence chain is the query,
its prototype, and the retrieved cases. The other three are the prototype
partition, per-prototype profiles, and the pairwise geometry of the prototypes.

These are per-dataset diagnostics, separate from the architecture figure in the
paper.

---

## Layout

```
libs/
  tabera.py         model, MemoryBank, TabularEmbedder
  prototypes.py     CentroidLayer — routing, EMA update, dead-prototype recovery
  evidence.py       retrieval aggregation (ablation paths only)
  supervised.py     training loop
  search_space.py   Optuna space, study naming
  diagnostics.py    read-only observers over a forward pass
  eval.py           metrics
  data.py           OpenML loading
optimize.py         hyperparameter search
reproduce.py        train / evaluate / explain
visualize_tabera.py per-dataset diagnostic panels
```

`TABERA_V3_ARCHITECTURE.md` records the measurement behind each design
decision.
`REFERENCES.md` lists the prior work, organised by which component it supports.

---

## References

Selected references, with the part of the model each supports:

- Gorishniy et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*. — piecewise-linear embeddings
- van den Oord, Vinyals & Kavukcuoglu (2017). Neural Discrete Representation Learning. *NeurIPS*. — hard assignment with a straight-through gradient, EMA codebook
- Razavi, van den Oord & Vinyals (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. *NeurIPS*. — EMA as the default update
- Bengio, Léonard & Courville (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Dhariwal et al. (2020). Jukebox: A Generative Model for Music. *arXiv:2005.00341*. — restarting unused codes

- Chen, Li, Tao, Barnett, Rudin & Su (2019). This Looks Like That. *NeurIPS*. — prototype-based prediction rather than post-hoc explanation
- Kim, Khanna & Koyejo (2016). Examples are not Enough, Learn to Criticize! *NeurIPS*. — why contrasting cases belong beside supporting ones
- Gorishniy et al. (2024). TabR: Tabular Deep Learning Meets Nearest Neighbors. *ICLR*. — retrieval as a prediction component; here it is conditioned by the model's own prototype partition instead