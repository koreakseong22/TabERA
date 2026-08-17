"""
libs/search_space.py
====================
Optuna hyperparameter search space for TabERA.
Follows the structure of MultiTab's search_space.py.

get_search_space      : trial -> params dict
suggest_initial_trial : defaults enqueued as the first trial (warm start)
params_to_model_kwargs: params -> TabERA constructor arguments
"""

from __future__ import annotations
import math
import optuna


# ─────────────────────────────────────────────────────────────
# Single source of truth for the HPO <-> reproduce training schedule
# ─────────────────────────────────────────────────────────────

HPO_TRAINING_SCHEDULE = {"epochs": 100, "patience": 20}
"""Training budget shared by every HPO trial and by the final reproduce run.

Set to match the MultiTab benchmark (Lee et al.) so that a difference against
a baseline cannot be an artefact of a different schedule.

Why it must be a single constant: optimize.py used to hard-code 100/20 while
reproduce.py exposed --epochs (default 200) / --patience (default 30). A
script named "reproduce the best config found by HPO" was therefore training
on a different schedule than the search. Measured on adult (id=1590):
reproduce.py trained longer and more permissively yet reached a *lower*
validation accuracy than the best HPO trial, and centroid concentration
(max_cluster_size) was markedly worse. Regroup and dead-centroid
reinitialisation run every epoch, so the longer the run the more the
concentration accumulates.

The original MultiTab reproduce.py cannot hit this class of bug at all: the
HPO trial and the final run go through the same wrapper `fit()`. This
constant enforces the same principle here. Both optimize.py and reproduce.py
import it directly and neither hard-codes epochs/patience nor overrides it
through a CLI default. Change this one place and both sides follow.

⚠ On ds=14 the 200/30 setting stopped at epoch 72 anyway, so small datasets
  are largely unaffected by the change. Larger datasets may be cut short —
  when comparing against older results, state the schedule explicitly.
"""


# ─────────────────────────────────────────────────────────────
DEFAULT_K_NO_TUNE = 8


# ─────────────────────────────────────────────────────────────
# Study file naming
# ─────────────────────────────────────────────────────────────

def study_pkl_tag(
    cat_combine: str = "onehot",
    num_embedding: str = "ple",
    n_prototypes: "int | None" = None,
    disable_dead_reinit: bool = False,
    num_bins: int = 8,
    cat_embed_dim: int = 16,
) -> str:
    """Build the tag embedded in the study .pkl filename written by optimize.py.

    optimize.py and reproduce.py must share **this one function**. When each
    file carried its own tag logic, changing one side silently broke the
    other — optimize.py saved `..num_ple..` while reproduce.py looked for an
    untagged name and raised FileNotFoundError.

    ⚠ Only non-default settings go into the tag. Add new flags here too,
      otherwise runs under different conditions overwrite the same file.

    ⚠ Tags for removed components (fusion_mode / L_nbr / aggregator /
      commitment / retr_proj ...) are deliberately absent. Studies produced
      under those conditions came from legacy/v3ema2_full/ and keep the
      filenames they were written with.
    """
    return "..v3ema2" \
        + ("..cat_concat" if cat_combine == "concat" else "") \
        + ("..cat_sum" if cat_combine == "sum" else "") \
        + ("..num_ple" if num_embedding == "ple" else "") \
        + ("..num_linear" if num_embedding == "linear" else "") \
        + (f"..P{int(n_prototypes)}" if n_prototypes is not None else "") \
        + ("..nodr" if disable_dead_reinit else "") \
        + (f"..bins{int(num_bins)}" if int(num_bins) != 8 else "") \
        + (f"..catdim{int(cat_embed_dim)}" if int(cat_embed_dim) != 16 else "")


def suggest_initial_trial() -> dict:
    """Defaults enqueued as the first trial.

    ⚠ Include **only keys that get_search_space() actually searches**. Optuna
      silently ignores anything else, but a reader takes it as "the first
      trial runs with these values". This file really did carry `k: 16`
      while the model always used DEFAULT_K_NO_TUNE (8). `k` and
      `batch_size` are fixed, so they do not belong here.
    """
    return {
        "embed_dim":        128,
        "embedder_layers":  2,
        "dropout":          0.1,
        "lr":               3e-4,
        "weight_decay":     1e-5,
    }


# ─────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────

def get_search_space(
    trial: optuna.Trial,
    num_features: int = 0,   # kept for MultiTab compatibility (unused)
    data_id: int = 0,        # kept for MultiTab compatibility (unused)
    num_embedding: str = "ple",
    # optimize.py always passes num_embedding explicitly, so this default only
    # affects direct callers (tests, notebooks). It matches the CLI default.
) -> dict:
    """Sample TabERA hyperparameters from an Optuna trial.

    Parameters
    ----------
    trial        : optuna.Trial
    num_features : number of input features (available for conditional search)
    data_id      : dataset id (available for conditional search)
    num_embedding: "linear" / "ple" / "plr_lite". The PLR hyperparameters
                   (sigma, n_frequencies, out_dim) enter the space only for
                   "plr_lite".

    Returns
    -------
    dict: every parameter needed to construct and train the model.
    """
    space = {
        # ── Architecture ────────────────────────────────
        "embed_dim":       trial.suggest_categorical("embed_dim",   [64, 128, 256]),

        # n_prototypes is set by optimize.py as sqrt(N_train); not searched.

        # ⚠ k is not searched. Retrieval sits outside the prediction path, so
        #   changing k does not move the HPO objective (validation score) at
        #   all — searching it would spend one dimension of the budget on
        #   noise. k is an explanation budget: 4 is too few to read, 16 is
        #   more than a reader uses.
        "k": DEFAULT_K_NO_TUNE,

        # embedder_layers was narrowed to [2, 4] at one point and then
        # restored to [1, 4]. The evidence for narrowing did not hold up:
        #   (1) It came from best-trial distributions ({1:2, 2:1, 3:9, 4:10}
        #       over 22 datasets), but every study collected afterwards ran
        #       with the narrowed range, so "nobody picks 1" could never be
        #       re-checked — the option was not on the table. The
        #       RandomForest importance analysis ranked embedder_layers last
        #       for the same reason: restricting the range shrinks the
        #       variance and thus understates the importance.
        #   (2) Of the two datasets that preferred 1 (51 heart-h, 41143
        #       jasmine), jasmine later scored worse under PLE than PLR. That
        #       dataset is exactly the one with a layers=1 preference, so the
        #       encoder comparison may have been confounded by the blocked
        #       option rather than by the encoder.
        # Restored until it can be re-checked without that bias.
        "embedder_layers": trial.suggest_int("embedder_layers", 1, 4),
        "dropout":         trial.suggest_float("dropout", 0.0, 0.5, step=0.05),

        # ── Optimisation ────────────────────────────────
        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        # ⚠ The range is deliberately **not** narrowed. Log-uniform over four
        #   orders of magnitude is wide by the usual HPO standards, but wide
        #   and unnecessarily wide are different claims. Measured over the
        #   accumulated studies (N = 1,258 best_params):
        #       min 1.0e-06,  median 9.2e-05,  max 1.0e-02
        #       24% of selected values exceeded 1e-3
        #   The maximum sits essentially at the upper bound, so if anything
        #   the ceiling may be the binding constraint. Capping at 1e-3 -- the
        #   range other tabular benchmarks tend to use -- would discard a
        #   quarter of the values the search actually chose.
        #
        # ⚠ The lower bound is untouched for a different reason: how often
        #   1e-6 is actually selected as best has not been measured, and
        #   raising it to 1e-5 on the assumption that it is rare would repeat
        #   exactly the mistake documented under embedder_layers above --
        #   narrowing a range removes the evidence needed to re-check it.
        #
        #   ⚠ N = 1,258 counts every study on disk, including ablations,
        #     multiple seeds and legacy conditions, with some files duplicated.
        #     It is enough to settle "do not narrow", but a number quoted in
        #     the paper should be recomputed over the final configuration
        #     only, after deduplication.
        "weight_decay":    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        # batch_size is fixed at 256 rather than searched.
        #
        # Original grounds (gradient-codebook era): RandomForest importance
        # ranked it 7th-8th of 9 (0.037), and a direct sweep over
        # {64,128,256,512} on profb / credit-g / vehicle / jasmine found no
        # reliable size-to-batch relationship — vehicle went 0.847 at 256 and
        # 0.671 at 512, a drop between adjacent values that is better
        # explained by this architecture's own noise (STE plus dead-centroid
        # reinitialisation amplify small early-training differences, the same
        # effect that motivated regroup_warmup_epochs) than by batch size.
        #
        # Under EMA prototype memory the parameter gained a second path:
        #
        #     B  ->  gradient noise                          (the old path)
        #     B  ->  samples per prototype per batch  ->  EMA update stability
        #
        # In `c_p <- m_p / N_p`, N_p is how many samples in the batch were
        # assigned to prototype p, so the quantity to watch is **B/P, not
        # N_train**:
        #
        #   ds     N     P    B/P    P(empty prototype) = (1 - 1/P)^B
        #   54    676   26    9.8    0.0%
        #   31    800   28    9.1    0.0%
        #   14   1600   40    6.4    0.2%
        #   1043 3649   60    4.3    1.4%
        #   1489 4322   65    3.9    1.9%
        #
        # Because P = sqrt(N), B = 256 keeps B/P in a narrow 3.9-9.8 band.
        # ⚠ A rule like "small dataset, so drop B to 32" is therefore
        #   actively harmful: on ds=54 that gives B/P = 1.2 and a 29% chance
        #   of an empty prototype, i.e. roughly one prototype in three goes
        #   un-updated every batch.
        #
        # ⚠ This is a fixed benchmark protocol, not a claim that 256 is
        #   optimal. Revisiting it means watching two axes together:
        #     (1) B/P             EMA update stability
        #     (2) (N/B) x epochs  optimizer update budget
        #   (1) is currently healthy; (2) varies a lot across datasets
        #   (2-16 steps/epoch). Early stopping absorbs some of (2) but does
        #   not control it.
        "batch_size":      256,
    }

    if num_embedding == "plr_lite":
        # PLR (lite) hyperparameters are searched per trial rather than fixed.
        # Gorishniy et al. 2022 (the periodic-embedding paper) treats sigma
        # (frequency scale) and k (number of frequencies) as dataset-level
        # hyperparameters to be tuned, recommending a log-uniform prior for
        # sigma and a uniform integer prior for k.
        #
        # Previously optimize.py passed --plr_freq_scale / --plr_n_frequencies
        # as run-wide constants (0.01, 16), so all 100 trials shared one
        # sigma. On mfeat-fourier and vehicle — both without categorical
        # features, so PLR carries the entire numeric encoding — 7-8 trials
        # out of 100 collapsed to exactly chance-level validation accuracy,
        # which points at that constant being wrong for those distributions.
        space["plr_freq_scale"] = trial.suggest_float("plr_freq_scale", 0.01, 100.0, log=True)
        space["plr_n_frequencies"] = trial.suggest_int("plr_n_frequencies", 8, 96)
        space["plr_out_dim"] = trial.suggest_categorical("plr_out_dim", [4, 8, 16, 32])

    return space


# ─────────────────────────────────────────────────────────────
# params -> TabERA constructor arguments
# ─────────────────────────────────────────────────────────────

def params_to_model_kwargs(params: dict, n_features: int, n_output: int) -> dict:
    kwargs = {
        "n_features":      n_features,
        "embed_dim":       params["embed_dim"],
        "n_prototypes":    params["n_prototypes"],

        # ⚠ k is written into the space dict directly, not via
        #   trial.suggest_*. Optuna's study.best_params records only
        #   suggested parameters, so a study reloaded by reproduce.py has no
        #   "k" key at all. Fall back to the same constant the training run
        #   used, or a legacy study's stored value if it has one.
        "k":               params.get("k", DEFAULT_K_NO_TUNE),

        "embedder_layers": params["embedder_layers"],
        "dropout":         params["dropout"],
        "n_output":        n_output,

    }
    # PLR (lite) hyperparameters are present only when num_embedding was
    # "plr_lite"; pass them through when they exist, otherwise the TabERA
    # defaults (or the CLI --plr_* values) apply.
    for key in ("plr_freq_scale", "plr_n_frequencies", "plr_out_dim"):
        if key in params:
            kwargs[key] = params[key]
    return kwargs