# TabERA

**Tabular Explainable Retrieval Architecture**

A tabular classifier that answers three questions about every prediction it
makes, using the forward pass itself rather than a post-hoc approximation of
it:

> *Which group does this sample belong to?*
> *Which past cases resemble it?*
> *What is unusual about it, for that group?*

---

## The idea

Most tabular models predict from an embedding with no intermediate structure a
person can point at, so explanations are reconstructed afterwards — by
perturbing inputs, fitting a surrogate, or attributing against a dataset-wide
baseline.

TabERA puts the structure in the model. Every sample is hard-assigned to one of
`P` prototypes, and that assignment does two things: it supplies the context
the prediction is built from, and it defines the neighbourhood the evidence is
retrieved from.

```
x ─→ Encoder ─→ q ─→ argmax cos(q, c) ─→ prototype a
                                          │
                          ┌───────────────┴───────────────┐
                          │                               │
                   c = centroid[a]                 G(a) = its members
                          │                               │
                   h = c + β·norm(q−c)              NN(q, G(a)), k = 8
                   z = W·h                                │
                          ▼                               ▼
                     prediction                       explanation
```

The partition is shared: the region a person reads about is the region the
model conditioned on, not a neighbourhood computed separately afterwards.

---

## What an explanation looks like

Three levels, printed by `reproduce.py --explain`. Each reads a different part
of the same forward pass. The example below is `credit-g`, predicting loan
default.

### ① Which group is this?

```
Assigned prototype: "Centroid_6"
Prototype label distribution: "bad" 124/160 (78%), also "good" 36/160 (22%)
Routing distribution:
  • Centroid_6            14.6%  (assigned)
  • Centroid_20           13.6%  ("bad" 10/11 (91%))
  • Others                62.2%
Distinctive features:
  numeric:     installment_commitment = 4
  categorical: other_parties = none (95%), housing = own (56%)
```

The phrasing is deliberate. A centroid is a *similarity anchor*, not a class
representative — the honest reading is "this group holds 160 samples, 78% of
which defaulted", not "this group represents default". The routing distribution
shows how close the call was: a sample between two prototypes is a different
kind of case from one sitting inside one.

### ② Which past cases resemble it?

```
neighbourhood (k=8)   bad 7/8 (88%), good 1/8 (12%)   H(label) 0.377
whole group (n=160)   bad 124/160 (78%)               H(label) 0.533
→ relative ambiguity 0.71

Supporting cases
  #2  similarity 0.984  → bad   [train #400]
       checking_status < 0, credit_history existing paid, employment < 1

Contrasting cases
  #1  similarity 0.985  → good  [train #293]
       differs: purpose    business → furniture/equipment
       differs: duration   48 → 18
```

Real training rows, ranked by the cosine similarity the retrieval uses. Cases
that *disagree* get their own section with the columns that differ — a nearest
neighbour that came out the other way is the most useful thing this level can
surface, and burying it among the agreeing cases would waste it.

### ③ What is unusual about it, for that group?

```
prototype baseline:  bad 70.4%
this sample:         bad 72.3%   (+1.8pp)

against the group (n=160)
  duration = 48           group typical 24,    2.0×,  top 9% within group
  credit_amount = 4,308   group typical 2,382, 1.8×,  top 26% within group
  purpose = business      10% of the group  |  group mode: new car (36%)
```

This is the axis feature-attribution methods do not have. SHAP asks how far
each feature moved the prediction from a dataset-wide baseline; level ③ asks
what makes this sample unusual *within its own group* — often the question a
domain expert was actually asking.

It is exact rather than approximate. The head is a single `Linear` with `W`
shared across both terms, so

```
z = W·c + β·W·normalize(q − c)
```

holds to floating point. The prototype baseline and the sample's correction are
the prediction, decomposed — not a model of it.

---

## How the model produces them

**Encoding.** Numeric columns go through piecewise-linear embeddings: quantile
bin edges computed once from the training split, one trainable embedding per
bin per column. Categorical columns are one-hot — a raw integer code carries no
ordering. An MLP maps the concatenation to a query embedding `q`.

**Assignment.** `q` goes to the prototype with the highest cosine similarity.
The forward pass uses the hard one-hot; the backward pass uses the softmax, so
the encoder is trained through the routing even though the choice is discrete.

The prototypes receive no gradient. They are maintained by an exponential
moving average over the embeddings assigned to them, and one that goes
unclaimed for several epochs is reinitialised near a real embedding — the
codebook machinery from VQ-VAE, borrowed wholesale. It keeps the partition
tracking the representation as the encoder moves, without adding an objective
the prototypes must trade off against the task.

`P` is fixed at `√N_train`, so partition granularity is a stated property
rather than something the search lands on.

**Prediction.** `h = c + β·normalize(q − c)`, with `β` a learned scalar and `W`
shared between the two terms.

Both details matter. With separate output matrices the optimiser scales one
branch up and the other into irrelevance; with one `W`, growing it amplifies
both and `β` is the only knob. And `‖c‖ = 1` while `‖q‖` runs from 7 to 1197
across datasets — unnormalised, the correction drowns the prototype out.

⚠ Because `‖q‖ ≫ ‖c‖`, `normalize(q − c)` has cosine 0.994–1.000 with `q`
itself. The term is a correction along the query direction, and the code names
it accordingly; calling it a deviation *from the prototype* would be wrong.

**Retrieval.** k-NN inside the assigned prototype's members, self excluded.
`k = 8` is fixed: it is an explanation budget, and since retrieval sits outside
the prediction, the search objective could not respond to it anyway.

---

## Two paths, one partition

The assignment is where the model splits. Both branches read the same `a` and
never meet again.

**The prediction branch** takes the centroid. `c` is an average over everything
assigned to the region, so it carries what the group has in common;
`β·normalize(q − c)` then moves the logits by an amount specific to this
sample. `β` settles between 0.10 and 0.73 depending on the dataset.

The effect is a soft lookup table. `W·c` alone would give at most `P` distinct
outputs, one per region; the correction lets samples inside a region separate
while the region still sets the baseline. On a 100-class dataset with 35
prototypes that difference is the whole task — `β = 0` gives accuracy 0.256,
learning it gives 0.725.

**The evidence branch** takes the members. `G(a)` is the set of training rows
sharing the prototype, and the retrieval ranks them by `cos(q, ·)`: the
partition selects the pool, the query orders it. The top `k` come back with
their raw feature values and labels.

This branch does not feed the prediction — the logits are computed without it,
and the neighbours are shown alongside the decision as the cases the model's
own partition places nearest. Whether they could also improve the prediction
was measured across a range of fusion designs; the result is in
`TABERA_V3_ARCHITECTURE.md` §14.

Because the same `a` drives both, the evidence cannot drift from the decision.
A person reading the retrieved cases is reading the neighbourhood the
prediction was conditioned on, not a similarity search run afterwards.

---

## Training

A single cross-entropy objective on the logits. The prototype layer carries no
loss of its own — centroids move by EMA, outside the gradient, and dead ones
are recovered by reinitialisation.

The gradient reaches the encoder (through the straight-through routing), `W`,
and `β`. It does not reach the centroids, the memory bank, or the retrieval.

---

## Results

Ten OpenML datasets, five seeds each.

| dataset | accuracy | AUROC | log loss |
|---|---|---|---|
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
| **mean** | **0.8309** | **0.9168** | **0.5490** |

`ds=1493` is the capacity-limited case: 100 classes, 35 prototypes. Its log
loss is the price of a partition too coarse for the label space.

---

## Running it

```bash
pip install -r requirements.txt

# hyperparameter search
python optimize.py --openml_id 31 --seed 1 --n_trials 100

# print explanations
python reproduce.py --openml_id 31 --seed 1 --deterministic --explain
```

Defaults are the configuration described above. No flags are needed to
reproduce it.

Five hyperparameters are searched per dataset: `embed_dim` ∈ {64, 128, 256},
`embedder_layers` 1–4, `dropout` 0.05–0.45, and log-uniform `lr` and
`weight_decay`. The rest is fixed by rule — `P = √N_train`, `batch_size = 256`
(3.9–9.8 samples per prototype per batch, which is what the EMA update
consumes), `k = 8`, `ema_decay = 0.99`.

`--fusion_mode` selects the prediction head. `proto_dev` is the default;
`proto_only_linear` predicts from the centroid alone, `query_only_linear`
bypasses it, `proto_dev_vec` makes `β` per-dimension, and `proto_dev_retr` adds
a learned retrieval term. `--gradient_codebook` trains the centroids by
gradient instead of EMA. Run `--help` for the rest.

### Figures

```bash
python visualize_tabera.py --openml_id 54 --seed 1
```

Writes four figures to `figures/seed=1/`: the prototype partition, per-prototype
profiles, the prediction decomposition, and the evidence chain.

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
visualize_tabera.py figures
```

---

## References

The nine that carry the method:

- Gorishniy et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*. — piecewise-linear embeddings
- van den Oord, Vinyals & Kavukcuoglu (2017). Neural Discrete Representation Learning. *NeurIPS*. — hard assignment with a straight-through gradient, EMA codebook
- Razavi, van den Oord & Vinyals (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. *NeurIPS*. — EMA as the default update
- Bengio, Léonard & Courville (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Dhariwal et al. (2020). Jukebox: A Generative Model for Music. *arXiv:2005.00341*. — restarting unused codes
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.
- Chen, Li, Tao, Barnett, Rudin & Su (2019). This Looks Like That. *NeurIPS*. — prototypes as architecture rather than post-hoc explanation
- Kim, Khanna & Koyejo (2016). Examples are not Enough, Learn to Criticize! *NeurIPS*. — why contrasting cases belong beside supporting ones
- Gorishniy et al. (2024). TabR: Tabular Deep Learning Meets Nearest Neighbors. *ICLR*. — the retrieval framing this work departs from