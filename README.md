# TabERA

**Tabular Explainable Retrieval Architecture**

A tabular model whose prediction is grounded in a learned prototype, and
whose explanation is grounded in retrieved training examples. Every sample
is assigned to one of a set of prototypes; the prediction is a function of
that prototype plus a small, bounded correction; and the training examples
retrieved from the same region serve as case-based evidence for what the
model did.

---

## Core idea

Most tabular deep learning models produce a prediction from a black-box
embedding with no intermediate structure a person can inspect. TabERA is
organized so that three questions have direct answers from the forward pass
itself:

1. **Which region of feature space does this sample belong to?** — the
   assigned prototype, with its size, class distribution, and
   representative feature values.
2. **How does this sample differ from that region's typical member?** — the
   normalized deviation from the prototype, and how much it moved the
   prediction.
3. **Which past cases resemble it?** — the retrieved neighbors, with their
   raw feature values, labels, and similarity scores.

The first two are read directly off the computation that produces the
prediction. The third comes from a retrieval branch that runs alongside the
prediction — deliberately not feeding into it.

---

## Architecture

```
              Input x
                 |
           Tabular Embedder        ←── L_nbr (neighbourhood regularization)
                 |
                 q  (query embedding)
                 |
     ┌───────────┴───────────┐
     │                       │
Prototype Routing       Local Retrieval
 cosine argmax           top-k within the
 over M centroids        assigned region
     │                       │
    c  (prototype)      neighbours + raw features
     │                       │
 r = normalize(q − c)        │
     │                       │
 h = c + β · r               │
 logits = W · h              │
     │                       │
     ▼                       ▼
 Prediction              Explanation
```

**Embed.** Categorical features are one-hot encoded — a raw integer code has
no natural ordering, so no learned embedding is used for it. Numerical
features go through `PiecewiseLinearEmbeddings`: quantile bin boundaries are
computed once from the training data, and each column gets its own trainable
per-bin embedding.

**Route.** `CentroidLayer` assigns each sample to one of `M` centroids via a
straight-through hard-argmax over cosine similarity. The forward pass uses
the one-hot assignment; the backward pass uses the softmax, so the centroids
receive gradient. Centroids are initialized with k-means++ over the training
embeddings and protected from permanent death by a periodic reset — a
centroid with no assignments for several epochs is reinitialized near a real
embedding. `M` is set to `√N_train` rather than tuned, so partition
granularity is a controlled property rather than a free hyperparameter.

**Predict.** The prediction is

```
h = c + β · normalize(q − c),     β = sigmoid(β_raw),  initialized at 0.1
logits = W · h
```

The prototype `c` carries the decision; the deviation term restores the
ordering information that hard assignment would otherwise discard. Two
details make this work rather than degenerate into "just use `q`":

- **`W` is shared across both terms.** Giving the prototype and the query
  their own output matrices lets the optimizer scale one branch up and the
  other into irrelevance; with a single `W`, growing it amplifies both, and
  `β` is the only available knob. In practice `β` settles between 0.11 and
  0.37, the deviation contributes about 16% of the representation norm, and
  99.4% of argmax decisions match what the prototype alone would have given.
- **The deviation is normalized.** `‖q − c‖` ranges from roughly 14 to 260
  across datasets while `‖c‖ ≈ 1`; without normalization the prototype is
  simply drowned out.

**Regularize.** `L_nbr` is an InfoNCE term whose positives are each sample's
nearest neighbours in *raw feature space*, computed once before training.
It is deliberately label-free: cross-entropy already pushes same-label
samples together, so label-based positives would reinforce the collapse
rather than counteract it. What `L_nbr` preserves is the local geometry that
cross-entropy has no reason to keep — the structure a partition needs in
order to divide a region meaningfully.

**Retrieve.** `MemoryBank` performs k-NN search restricted to the sample's
assigned centroid. The retrieved neighbours, their raw feature values, and
their labels form the evidence for the explanation. This branch does not
feed the prediction (see below).

---

## Why retrieval is not in the prediction path

TabERA takes the neighbour-retrieval idea from TabR-style tabular models but
uses it for a different purpose. Five ways of routing retrieved information
into the prediction were tried, and all of them degraded ranking
performance:

| Attempt | Outcome |
|---|---|
| Concatenate the aggregate into the head | The query branch grows to dominate the logit |
| Add it as a second normalized residual | AUROC drops on 3/3 datasets even with shared `W` |
| Add retrieved label evidence to the logits | Neighbours within a prototype share labels, so this repeats what `c` already encodes |
| Quantize the residual into a second codebook | The second code learns, but in a direction with 1–3% of the first code's Fisher ratio |
| Modulate `β` or temperature by neighbour entropy | Entropy predicts errors, but the model's own confidence predicts them better |

The common thread: retrieved neighbours within a prototype carry very little
information the prototype does not already carry. Measuring it directly,
within a prototype, neither a linear probe nor an MLP beats the majority
class — and that holds in raw feature space too, so it is not an artifact of
what the encoder learned.

What retrieval *is* good for is showing a person what the model did. That is
where it stays.

---

## Explanations

Explanation is example-based rather than feature-attribution-based. The
question is not "which feature moved the prediction by how much" but "which
group is this, how does it differ from the group, and what happened to
similar cases".

**Level 1 — Prototype**

```
Prototype #17
  routing confidence   0.91
  group size           156  (23% of training data)
  class distribution   class 3: 78%, class 1: 15%, class 0: 7%
  representative       age 45–60, income high, debt low
```

Note the phrasing. A centroid is a *similarity anchor*, not a class
representative — samples near the centre of a prototype are measurably
*more* mixed than samples at its periphery, because the centre is where
several class distributions meet. So the honest statement is "this group
contains samples with these characteristics, 78% of which are class 3",
not "this group represents class 3".

**Level 2 — Similar cases**

```
Case 1   similarity 0.94   age 52  income high    debt low     → class 3
Case 2   similarity 0.91   age 49  income high    debt medium  → class 3
Case 3   similarity 0.89   age 60  income medium  debt high    → class 1
```

Ranked by cosine similarity, which is the actual retrieval criterion.
Attention weights are not used for this — they carry no information in the
current configuration.

**Level 3 — Prototype-relative attribution**

```
prototype anchor      c
deviation             ‖β·r‖ = 0.31   (group mean: 0.24)
dominant directions   feature A ↑, feature B ↓
logit contribution    14%
```

This is the axis SHAP does not have: the comparison baseline is the
prototype, not the dataset. "What is unusual about this sample *for its
group*" is often the question a domain expert is actually asking.

**Level 4 — Natural language**

The structured output above is the input to a language model that turns it
into prose. The language model writes the explanation; it does not make the
prediction.

---

## What the design costs and what it buys

Restricting predictions to `W · c` — the pure prototype form — is
interpretable but expensive: with `M` prototypes there are at most `M`
distinct predictions, so samples in the same region receive identical
scores and ranking metrics suffer. On one dataset with 100 classes and only
65 prototypes, the constraint made the task structurally impossible
(accuracy 0.24).

The deviation term removes that ceiling while keeping the prototype in
charge of the decision. Across nine datasets and five seeds, relative to
the pure prototype form:

- AUROC improves on 9/9 datasets
- accuracy and F1 are unchanged, which is the intended behaviour — the
  argmax is preserved and only the ordering changes
- the 100-class dataset goes from 0.24 to 0.77 accuracy

Against twelve baselines (including XGBoost, FT-Transformer, ResNet,
ModernNCA), TabERA ranks 7th on accuracy, 9th on AUROC, and 6th on F1, and
places 1st on log loss when the 100-class dataset is included. A gap of
roughly 0.02 in accuracy and AUROC remains against the strongest baselines;
the measurements above suggest this is the price of the prototype
constraint rather than a shortcoming of the representation.

---

## Usage

```bash
# hyperparameter search
python optimize.py --openml_id 14 --seed 1 --n_trials 100

# train, evaluate, and export diagnostics
python reproduce.py --openml_id 14 --seed 1 --deterministic \
  --export_centroid_retrieval_behavior --log_evidence_stats \
  --train_seeds 1 2 3 4 5 --run_tag "v3"
```

Defaults are the configuration described above (`--fusion_mode proto_dev`,
`--nbr_lambda 0.005`), so neither flag needs to be passed explicitly.

**Ablation modes**, available through `--fusion_mode` for reproducing the
comparisons in this document:

| Mode | What it does |
|---|---|
| `proto_dev` | default: `h = c + β·normalize(q−c)` |
| `proto_only_linear` | pure prototype: `logits = W·c` |
| `query_only_linear` | query straight into the head |
| `proto_dev_vec` | `β` as a per-dimension vector |
| `proto_dev_agg` | adds the retrieval aggregate as a second residual |
| `proto_residual_query` | separate output matrices for prototype and query |

`--residual_vq` enables the two-stage residual quantizer.

---

## Diagnostics

`--export_centroid_retrieval_behavior` writes a `.npz` per run containing
the training assignment and labels, per-sample centroid ids, query
embeddings, retrieved neighbour ids and labels, routing margins, and
centroid embeddings. `--log_evidence_stats` adds per-epoch traces:
prototype utilization and reset counts, the `β` trajectory and the
deviation's share of the representation norm, how often the deviation flips
an argmax, and how many distinct predictions the model produces.

`probe_group_separability.py` measures how separable classes are *within*
a prototype, in both the learned embedding and raw feature space — the
measurement behind the claim that within-prototype label information is
scarce.

```bash
python probe_group_separability.py --ids 14 31 41143 54 \
  --log_dir ./optim_logs --run_tag v3
```

---

## HPO parameters (searched via Optuna)

| Parameter | Range | Role |
|---|---|---|
| `embed_dim` | {64, 128, 256} | Embedding dimension |
| `embedder_layers` | 1–4 | Embedder MLP depth |
| `dropout` | 0.0–0.5 | — |
| `loss_commitment` | 1e-2–1e-1 | Query → centroid pull |
| `loss_codebook` | 1e-2–1e-1 | Centroid → query pull |
| `lr` | 1e-4–1e-2 | — |
| `weight_decay` | 1e-6–1e-2 | — |
| `plr_freq_scale` * | 0.01–100 (log) | PLR(lite) frequency init scale |
| `plr_n_frequencies` * | 8–96 | PLR(lite) frequencies per column |
| `plr_out_dim` * | {4, 8, 16, 32} | PLR(lite) output dim per column |

\* only with `--num_embedding plr_lite`.

Fixed rather than searched: `n_prototypes` at `√N_train`, `k` at 8 (it is an
explanation budget, not a model hyperparameter — retrieval is outside the
prediction path, so the search objective cannot respond to it),
`batch_size` at 256, `nbr_lambda` at 0.005.

---

## Project structure

```
TabERA/
├── libs/
│   ├── tabera.py         # TabERA model, MemoryBank, TabularEmbedder, L_nbr
│   ├── prototypes.py     # CentroidLayer (routing), ResidualCentroidLayer
│   ├── evidence.py       # Retrieval aggregation (explanation branch)
│   ├── supervised.py     # Training loop wrapper
│   ├── search_space.py   # Optuna HPO space
│   ├── eval.py           # Metrics, prediction utilities
│   └── data.py           # OpenML dataset loading
├── optimize.py                     # HPO entry point
├── reproduce.py                    # Train / evaluate / explain / diagnose
├── probe_group_separability.py     # Within-prototype separability analysis
└── requirements.txt
```

`TABERA_V3.md` documents the architecture in detail, including the
measurements behind each design decision and the alternatives that were
tried and rejected.

Run `python optimize.py --help` and `python reproduce.py --help` for the
full CLI reference.

---

## References

- Gorishniy et al. (2023). TabR: Tabular Deep Learning Meets Nearest Neighbors. *arXiv:2307.14338*.
- Gorishniy et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. *NeurIPS*.
- Ye et al. (2024). Revisiting Nearest Neighbor for Tabular Data (ModernNCA). *arXiv:2407.03257*.
- Snell, Swersky & Zemel (2017). Prototypical Networks for Few-shot Learning. *NeurIPS*.
- van den Oord et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS*.
- Dhariwal et al. (2020). Jukebox: A Generative Model for Music. *arXiv:2005.00341*.
- Zeghidour et al. (2021). SoundStream: An End-to-End Neural Audio Codec. *arXiv:2107.03312*.
- Bengio et al. (2013). Estimating or Propagating Gradients Through Stochastic Neurons. *arXiv:1308.3432*.
- Dwibedi et al. (2021). With a Little Help from My Friends: Nearest-Neighbor Contrastive Learning (NNCLR). *ICCV*.
- Arthur & Vassilvitskii (2007). k-means++. *SODA*.