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

from libs.data import get_batch_size   # MultiTab과 동일한 batch size 정책


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
# Protocol version tag
# ─────────────────────────────────────────────────────────────

PROTOCOL_TAG = "..mtsplit"
"""Tag marking the benchmark protocol a study was produced under.

Bump this whenever a change makes old studies incomparable to new ones, so
that optimize.py cannot silently resume a study built under the previous
protocol. Without it, optimize.py finds the existing .pkl, computes
`remaining_trials = max(0, 100 - 100) = 0`, runs no trial at all, and
rewrites the CSV from the *old* trials -- no error, no new results.

"..mtsplit" (2026-08) marks the run where three things were aligned to
MultiTab at once:
  * split      : StratifiedKFold -> KFold(10, shuffle=True, random_state=42)
  * batch size : fixed 256 -> get_batch_size(len(X_train))
  * objective  : penalised acc_val -> plain acc_val
Studies without this tag came from a different protocol and must not be
mixed in, resumed, or compared against.
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
    batch_size: "int | None" = None,
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
    return PROTOCOL_TAG \
        + "..v3ema2" \
        + ("..cat_concat" if cat_combine == "concat" else "") \
        + ("..cat_sum" if cat_combine == "sum" else "") \
        + ("..num_ple" if num_embedding == "ple" else "") \
        + ("..num_linear" if num_embedding == "linear" else "") \
        + (f"..P{int(n_prototypes)}" if n_prototypes is not None else "") \
        + ("..nodr" if disable_dead_reinit else "") \
        + (f"..bins{int(num_bins)}" if int(num_bins) != 8 else "") \
        + (f"..catdim{int(cat_embed_dim)}" if int(cat_embed_dim) != 16 else "") \
        + (f"..B{int(batch_size)}" if batch_size is not None else "")


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
    n_train: int = 0,
    batch_size: "int | None" = None,
    # None이면 MultiTab 정책 get_batch_size(n_train)을 따른다(본 실험 기본값).
    # 정수를 주면 그 값으로 고정한다 -- batch size pilot 전용이며, 이 경우
    # study 파일명에 ..B{n} 태그가 붙어 본 실험 study와 섞이지 않는다.
    # Size of the training split. Required: batch_size is derived from it with
    # MultiTab's get_batch_size(). optimize.py and reproduce.py must both pass
    # len(y_train); a caller that forgets raises rather than silently falling
    # back to a different batch size than the benchmark uses.
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
    if n_train <= 0:
        raise ValueError(
            "get_search_space()에 n_train이 전달되지 않았습니다. batch_size가 "
            "MultiTab의 get_batch_size(len(X_train))에서 나오므로 필수입니다 "
            "-- 호출부에서 n_train=len(y_train)을 넘겨 주세요.")

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

        # batch_size follows MultiTab's get_batch_size(len(X_train)) and is
        # not searched.
        #
        #   n_train      B
        #   > 50,000     1024
        #   > 10,000     512
        #   >  5,000     256
        #   >  1,000     128
        #   otherwise    64
        #
        # [2026-08 프로토콜 정정] 이전에는 256 고정이었다. MultiTab의 모든
        # 신경망 baseline은 supervised.py / modernnca.py / saint.py 안에서
        # get_batch_size(len(X_train))를 부르므로, 256 고정은 27개 벤치마크
        # 데이터셋 중 25개에서 baseline과 다른 최적화 조건을 의미했다
        # (N_train<=1000 인 13개에서 MultiTab 64 vs TabERA 256).
        #
        # 이 값이 여기서 자유롭지 않은 이유(원래 주석의 논지는 유효하다):
        # EMA prototype memory에서 `c_p <- m_p / N_p`의 N_p는 배치 안에서
        # 프로토타입 p에 배정된 샘플 수이므로, 봐야 하는 양은 N_train이
        # 아니라 **B/P**이고 빈 프로토타입 확률은 (1 - 1/P)^B 이다.
        # P = floor(sqrt(N_train)) 하에서 두 정책을 실측 비교하면:
        #
        #   ds              N_train    P    B=256  P(empty)   B=auto  P(empty)
        #   lymph               118   10      256      0.0%       64      0.1%
        #   vehicle             676   26      256      0.0%       64      8.1%
        #   socmob              924   30      256      0.0%       64     11.4%
        #   phoneme           4,323   65      256      1.9%      128     13.7%
        #   nomao            27,572  166      256     21.3%      512      4.5%
        #   electricity      36,249  190      256     25.9%      512      6.7%
        #
        # 즉 전환은 한 방향의 손해가 아니다. 소형에서는 빈 프로토타입 확률이
        # 오르지만(최악 phoneme 13.7%, 원래 주석이 경고한 ds=54 B=32의 29%
        # 보다는 낮다) **대형에서는 오히려 크게 개선된다** - 256 고정은
        # electricity/jungle_chess에서 B/P≈1.3, 빈 프로토타입 확률 26%로,
        # 전 구간에 좋은 선택이 아니었다.
        #
        # optimizer update budget 쪽도 같은 방향이다. lymph는 B=256에서
        # epoch당 0.5 step(100 epoch 동안 optimizer가 50번 움직인다)이고
        # B=64에서 1.8 step이 된다.
        #
        # ⚠ 이것은 고정된 벤치마크 프로토콜이지 256/auto 중 무엇이 최적이라는
        #   주장이 아니다. 전환 시 pilot에서 active_ratio_std 와
        #   reinit_per_epoch 를 반드시 함께 볼 것(둘 다 이미 user_attr로
        #   기록된다). 소형에서 EMA가 실제로 무너지면 그때 256을 유지할
        #   근거가 생기며, 그 경우 논문에는 "모든 방법이 동일한 training
        #   protocol에서 평가되었다"고 쓸 수 없고 batch size sensitivity
        #   실험을 함께 실어야 한다.
        "batch_size":      (get_batch_size(n_train) if batch_size is None
                            else int(batch_size)),
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