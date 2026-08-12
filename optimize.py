## Main entry point for tuning TabERA on one dataset.
## Paper: TabERA — Tabular Explainable Retrieval Architecture
## Based on: MultiTab (Kyungeun Lee, kyungeun.lee@lgresearch.ai)

import os, argparse

# ── Set CUDA_VISIBLE_DEVICES before torch is imported ──────────
# Same placement as upstream MultiTab: right after argparse, before torch.
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",    type=int, default=0,      help="gpu index")
parser.add_argument("--openml_id", type=int, default=45068,  help="dataset index (See dataset_id.json for detailed information)")
parser.add_argument("--seed",      type=int, default=1,      help="seed for dataset split (cross-validation)")
parser.add_argument("--savepath",  type=str, default=".",    help="path to save the results")
parser.add_argument("--n_trials",  type=int, default=100,    help="Number of optimization trials")
# ⚠ Any flag that changes the architecture must also appear in
#   study_pkl_tag(), so that the study file is separated. Forgetting that
#   makes reproduce.py load the baseline study and train the new structure
#   with hyperparameters tuned for the old one. This happened with
#   --global_retrieve: no error, silently wrong.
parser.add_argument("--n_prototypes", type=int, default=None,
                    help=("Set the prototype count explicitly (a structural "
                          "variable). Defaults to floor(sqrt(N_train)). When "
                          "set, the study filename gets a ..P{n} tag so runs "
                          "with different P cannot mix -- runs differing only "
                          "in P used to overwrite the same study, which made "
                          "them impossible to reproduce."))
parser.add_argument("--allow_self_retrieval", action="store_true",
                    help=("Do not exclude the query itself from retrieval "
                          "(excluded by default). Must match reproduce.py: "
                          "this flag was once missing here, so HPO retrieved "
                          "with self included while the reproduce run excluded "
                          "it. Prediction is unaffected because retrieval sits "
                          "outside the prediction path."))
parser.add_argument("--disable_dead_reinit", action="store_true",
                    help=("Disable dead-prototype recovery. When comparing "
                          "update rules, turn it off on both sides: "
                          "reinitialisation is a second path that moves "
                          "centroids, so leaving it on compares two things "
                          "at once instead of one."))
parser.add_argument("--cat_combine", type=str, default="onehot", choices=["sum", "concat", "onehot"],
                    help=(
                        "How categorical embeddings are combined. 'onehot' "
                        "(default) follows TabR/ModernNCA: plain one-hot with "
                        "no learned parameters. 'sum' and 'concat' are earlier "
                        "experimental options (same set as reproduce.py)."
                    ))
parser.add_argument("--cat_embed_dim", type=int, default=16,
                    help="Per-column embedding width when cat_combine=concat.")
parser.add_argument("--num_embedding", type=str, default="ple",
                    choices=["linear", "ple", "plr_lite"],
                    help=(
                        "Numeric feature encoding. 'ple' (default) is "
                        "PiecewiseLinearEmbeddings(activation=False), the same "
                        "structure TabM (Gorishniy et al. 2024) uses. It was "
                        "adopted to remove catastrophic failures, not for a "
                        "performance win: across profb/vehicle/credit-g/jasmine, "
                        "PLE had zero collapsed trials (chance-level validation) "
                        "against three for PLR, and dropping the PLR "
                        "hyperparameters shrank the search space. Top-5 test "
                        "performance still favoured PLR on 3 of 4 datasets, and "
                        "centroid margin_percentile was lower under PLE on all "
                        "four (cause unknown). 'plr_lite' is the previous "
                        "default and remains available."
                    ))
parser.add_argument("--num_bins", type=int, default=8,
                    help="Bins per column when num_embedding=ple.")
# --plr_n_frequencies / --plr_freq_scale / --plr_out_dim were removed here.
# For num_embedding=plr_lite these are now searched per trial by
# get_search_space() (the approach Gorishniy et al. 2022 recommend). As fixed
# CLI flags all 100 trials shared one value, and numeric-only datasets
# (mfeat-fourier, vehicle) then produced repeated collapsed trials.
# reproduce.py still needs concrete values for its single final run, so the
# flags remain there; whatever HPO found is stored in best_params and picked
# up automatically when reproduce.py reloads the study.
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)   # same position as upstream

import optuna, torch, json, joblib, datetime, math, gc
from libs.data import TabularDataset
from libs.eval import calculate_metric, is_study_todo, check_if_fname_exists_in_error, get_preds_and_probs
from libs.search_space import (get_search_space, suggest_initial_trial, params_to_model_kwargs, study_pkl_tag, HPO_TRAINING_SCHEDULE, DEFAULT_K_NO_TUNE)
from libs.supervised import TabERAWrapper
from libs.tabera import TabERA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.INFO)

# ─────────────────────────────────────────────────────────────
# Dataset metadata (same as upstream MultiTab)
# ─────────────────────────────────────────────────────────────

with open("dataset_id.json", "r") as file:
    data_info = json.load(file)

tasktype = data_info.get(str(args.openml_id))["tasktype"]
print(tasktype)

# ─────────────────────────────────────────────────────────────
# Output paths (same as upstream MultiTab)
# ─────────────────────────────────────────────────────────────

if not args.savepath.endswith("optim_logs"):
    savepath = os.path.join(args.savepath, "optim_logs", f"seed={args.seed}")
else:
    savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)

_ablation_tag = study_pkl_tag(
    cat_combine=args.cat_combine,
    num_embedding=args.num_embedding,
    # ⚠ The auto-computed P is unknown here (the data is not loaded yet),
    #   so only an explicitly given value goes into the tag.
    n_prototypes=args.n_prototypes,
    disable_dead_reinit=args.disable_dead_reinit,
    num_bins=args.num_bins,
    cat_embed_dim=args.cat_embed_dim,
)
fname = os.path.join(savepath, f"data={args.openml_id}{_ablation_tag}..model=tabera.pkl")

# ─────────────────────────────────────────────────────────────
# Skip work already done (same as upstream MultiTab)
# ─────────────────────────────────────────────────────────────

train = True
if os.path.exists(fname):
    study = joblib.load(fname)
    train = is_study_todo(study, tasktype)
else:
    study = (optuna.create_study(direction="minimize") if tasktype == "regression"
             else optuna.create_study(direction="maximize"))
    initial_trial = suggest_initial_trial()
    study.enqueue_trial(initial_trial)
    train = check_if_fname_exists_in_error(fname)

completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
remaining_trials = max(0, args.n_trials - completed_trials_count)

# ─────────────────────────────────────────────────────────────
# Main optimisation loop (upstream MultiTab structure)
# ─────────────────────────────────────────────────────────────

if train:
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    import platform
    env_info = "{0}:{1}".format(platform.node(), args.gpu_id)
    print(env_info, device)

    # ── Header ─────────────────────────────────────────────
    print("=" * 60)
    print("  TabERA  Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Dataset : {data_info[str(args.openml_id)]['fullname']} (id={args.openml_id})")
    print(f"  Task    : {tasktype}  |  Device : {device}")
    print(f"  Trials  : {completed_trials_count} done / {args.n_trials} total  ({remaining_trials} remaining)")
    print(f"  Encoding: cat_combine={args.cat_combine}, num_embedding={args.num_embedding}")
    print(f"  Save    : {fname}")
    print("=" * 60)

    # ── Data (MultiTab: TabularDataset + _indv_dataset) ────
    dataset = TabularDataset(args.openml_id, tasktype, device=device, seed=args.seed)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset._indv_dataset()
    y_std = dataset.y_std

    print(f"  Train/Val/Test : {len(y_train):,} / {len(y_val):,} / {len(y_test):,}"
          f"  |  Features: {dataset.n_features}")
    print("-" * 60)

    # output_dim comes from n_classes: y is 1-D, so shape[1] is unavailable.
    output_dim = dataset.n_classes if tasktype == "multiclass" else 1

    # ── n_prototypes ────────────────────────────────────────────────
    #
    # The default is **P = sqrt(N_train)** and it is not meant to be changed.
    #
    # Why sqrt(N): the mean group size is |G| = N/P = sqrt(N), so raising P
    # buys prototype expressiveness at the cost of group size. sqrt(N) is the
    # balance point. A centroid plays three roles at once here — prediction
    # anchor (context_emb), retrieval partition, and explanation unit — so
    # this is not the usual "pick k for k-means" question.
    #
    # This was once changed to clip(sqrt(N), C, N/k) and reverted. On ds=1493
    # (N=1279, C=100), raising P from 35 to 100 gives
    #     acc  0.772 -> 0.821   (+4.9pp, paired t p=0.009, 5/5 seeds)
    #     beta 0.691 -> 0.51    (the correction term takes on less of the job)
    # which is a real capacity fix. But at the same time
    #     H(Y_G)     1.32  -> 0.086
    #     alignment  0.367 -> 0.949
    #     |G|/k      4.8   -> 1.5
    # so prototypes start behaving like class-specific memory units and the
    # within-group structure that explanation layer 2 depends on disappears.
    # ⚠ That is a trade-off between two goals, not a bug to fix. A default
    #   must not silently pick one side.
    #
    # So: sqrt(N) by default, because a TabERA prototype is a local region
    # descriptor rather than a class prototype. To look at capacity instead,
    # pass --n_prototypes explicitly and keep it in its own study. When C > P
    # we only warn — you need to know the correction term is carrying the
    # classification in order to read the result.
    #
    # P = floor(sqrt(N_train)). No lower bound: it would only bind below
    # N_train = 16, and a run that small is not a case this rule is for.
    _sqrtN = int(math.sqrt(len(y_train)))
    _C     = dataset.n_classes if tasktype == "multiclass" else None
    # k is not searched (it is an explanation budget); always the same value.
    _k_ref = DEFAULT_K_NO_TUNE

    if args.n_prototypes is not None:
        n_proto_default = int(args.n_prototypes)
        print(f"  n_prototypes: {n_proto_default}  (set via --n_prototypes)")
    else:
        n_proto_default = _sqrtN
        print(f"  Auto n_prototypes: sqrt({len(y_train)}) = {n_proto_default}")

    if _C is not None and n_proto_default < _C:
        print(f"    ! P({n_proto_default}) < C({_C}): logits ~ W*c and c takes "
              f"only P distinct values, so the prototype term alone cannot "
              f"reach every class. The correction term takes over the "
              f"classification (beta rises, argmax changes more often). To "
              f"measure accuracy instead, build a separate study with "
              f"--n_prototypes {_C} -- but group size drops to "
              f"{len(y_train)//_C}, which weakens the explanation layer.")
    _grp = len(y_train) // max(n_proto_default, 1)
    if _grp < _k_ref:
        print(f"    ! mean group size ({_grp}) < k ({_k_ref}): retrieval will "
              f"fall back across groups.")

    # ── PLE bin edges (only when num_embedding=ple) ────────────────
    # Same logic as reproduce.py. Computed once outside objective(): the
    # edges derive from the data, not from any trial hyperparameter.
    num_bin_edges = None
    if args.num_embedding == "ple" and len(dataset.X_num) > 0:
        X_num_train = X_train[:, dataset.X_num]
        q = torch.linspace(0.0, 1.0, args.num_bins + 1, device=X_num_train.device)
        num_bin_edges = torch.quantile(X_num_train, q, dim=0).T.contiguous()

    # ── Reproducibility guard: was this study built with a different P? ──
    # ⚠ The auto-computed P does not appear in the study filename. If a code
    #   change moves it (ds=1493 went 35 -> 100), trials with different
    #   structures end up mixed into, or overwriting, the same file. A P=35
    #   study really was lost to a P=100 run. P is a structural variable and
    #   must not be mixed.
    _prev_P = {t.user_attrs.get("n_prototypes_actual")
               for t in study.trials
               if t.state == optuna.trial.TrialState.COMPLETE
               and t.user_attrs.get("n_prototypes_actual") is not None}
    if _prev_P and _prev_P != {n_proto_default}:
        raise SystemExit(
            f"\n[stopped] The existing study has n_prototypes="
            f"{sorted(_prev_P)}, this run would use {n_proto_default}.\n"
            f"  file: {fname}\n"
            f"  P is a structural variable; mixing values in one study makes "
            f"the run impossible to reproduce.\n"
            f"  -> pass --n_prototypes {n_proto_default} to get a separate "
            f"study (..P{n_proto_default}), or move the existing study aside.")

    global best_so_far
    best_so_far = study.best_value if completed_trials_count > 0 else None

    # ── Objective (upstream MultiTab structure) ────────────
    def objective(trial):
        params       = get_search_space(trial, num_features=X_train.size(1),
                                        data_id=args.openml_id,
                                        num_embedding=args.num_embedding)
        # n_prototypes comes from the sqrt(N) rule, not from the search.
        params["n_prototypes"] = n_proto_default
        trial.set_user_attr("n_prototypes_actual", n_proto_default)
        model_kwargs = params_to_model_kwargs(params, dataset.n_features, output_dim)

        model = TabERA(
            **model_kwargs,
            column_names=dataset.col_names,
            # tasktype is required so that neighbour labels are encoded as
            # nn.Embedding for classification and nn.Linear for regression,
            # matching upstream TabR. Without it, nominal class labels would
            # be fed in as raw integers.
            tasktype=tasktype,
            n_classes=(output_dim if tasktype == "multiclass" else (2 if tasktype == "binclass" else None)),
            # The old min(2N, 10_000) cap meant MemoryBank could not hold all
            # of X_train once N_train > 10,000, so sample_groups reflected as
            # little as 28% of the real group (id=41027). Holding the whole
            # split costs ~73MB at N=35,855 and D=256, so the cap is gone and
            # group-constrained retrieval works as designed.
            memory_size=len(y_train),
            # ⚠ This argument used to be omitted here, so the constructor
            #   default (False) applied while reproduce.py passed True. HPO and
            #   the reproduce run were using different retrieval rules.
            #   Prediction was unaffected (measured max difference 0.000e+00)
            #   because retrieval is outside the prediction path, but the
            #   mismatch was real, so both sides now agree.
            exclude_self_retrieval=(not args.allow_self_retrieval),
            # Dead-prototype recovery is disabled by setting patience far
            # above the epoch count, the same way reproduce.py does it, rather
            # than by adding a separate branch. The value (1e9) must match.
            **({"dead_reinit_patience": 10 ** 9} if args.disable_dead_reinit else {}),
            # Categorical / numeric encoding. Without these, cat_col_idx is
            # None and the model silently falls back to raw encoding
            # regardless of cat_combine and num_embedding -- HPO would then
            # tune a different architecture than the one reproduce.py trains.
            cat_col_idx=list(dataset.X_cat),
            num_col_idx=list(dataset.X_num),
            cat_cardinalities=list(dataset.X_cat_cardinality),
            cat_combine=args.cat_combine,
            cat_embed_dim=args.cat_embed_dim,
            num_embedding=args.num_embedding,
            num_bin_edges=num_bin_edges,
            # plr_* are deliberately not passed here: for plr_lite they are
            # already inside model_kwargs (searched per trial and routed
            # through params_to_model_kwargs), and passing them again would
            # be a duplicate keyword argument. For other encodings they are
            # never read.
        )

        wrapper = TabERAWrapper(model, params, tasktype,
                                  device=str(device), **HPO_TRAINING_SCHEDULE)
        wrapper._data_id = args.openml_id   # shown in the epoch progress bar
        wrapper.fit(X_train, y_train, X_val, y_val)

        # ── Evaluate: one logits pass, then preds and probs ────
        wrapper.model.eval()
        with torch.no_grad():
            val_logits  = wrapper._forward_batched(X_val)
            test_logits = wrapper._forward_batched(X_test)
        preds_val,  probs_val  = get_preds_and_probs(val_logits,  tasktype)
        preds_test, probs_test = get_preds_and_probs(test_logits, tasktype)

        # Regression metrics are computed after undoing the y standardisation.
        if tasktype == "regression":
            val_metrics  = calculate_metric(y_val  * y_std, preds_val  * y_std, probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test * y_std, preds_test * y_std, probs_test, tasktype, "test")
        else:
            val_metrics  = calculate_metric(y_val,  preds_val,  probs_val,  tasktype, "val")
            test_metrics = calculate_metric(y_test, preds_test, probs_test, tasktype, "test")

        for k, v in val_metrics.items():
            trial.set_user_attr(k, v)
        for k, v in test_metrics.items():
            trial.set_user_attr(k, v)

        # Same console output as upstream MultiTab
        print(device, env_info, args.openml_id,
              data_info.get(str(args.openml_id))["name"], "tabera", savepath)
        print(val_metrics)
        print(test_metrics)
        now      = datetime.datetime.now()
        duration = now - trial.datetime_start
        print(f"### Optimization time for trial {trial.number}: {duration.total_seconds():.0f} secs")
        trial.set_user_attr("training_time", duration.total_seconds())

        # Objective: minimise rmse_val for regression, maximise acc_val otherwise.
        result = val_metrics["rmse_val"] if tasktype == "regression" else val_metrics["acc_val"]

        # ── Centroid margin penalty (percentile-based, no threshold) ──
        # routing_scale does not affect the forward pass (STE makes the hard
        # assignment invariant to a positive scale), but it does affect how
        # peaked the STE backward gradient is. Measured: trials that landed on
        # a low routing_scale tended to end with a routing structure no better
        # than random (credit-g: scale 1.49, margin_percentile ~0%, against
        # socmob 19.8 and SpeedDating 13.77, both ~100%).
        #
        # Design choice: leave the search range alone and let Optuna learn to
        # avoid the side effect through the objective itself.
        #
        # Why a percentile rather than a z-score threshold: a threshold on the
        # z-score needs a magic number, and picking one kept going wrong. At
        # -2.0 it caught only actively-bad cases and missed mfeat-zernike
        # (z_margin = -0.15, indistinguishable from random). Raising it to 2.0
        # only borrowed a display convention. The penalty is continuous
        # anyway, so the binary "is this significant" judgement a threshold
        # exists for is not needed — the rank against the null is enough.
        #
        # So the penalty uses the measured margin's percentile against the 50
        # null samples already computed in supervised.py: 100% (better than
        # every random draw) gives no penalty, 50% gives half the cap, 0%
        # gives the full cap. No threshold appears anywhere.
        # penalty_cap (0.05) was set from credit-g / mfeat-zernike / jasmine;
        # margin_percentile keeps accumulating in the user attributes below,
        # so the cap itself can be re-derived from data later.
        diag = wrapper.centroid_geometry_diag
        if diag is not None:
            trial.set_user_attr("centroid_z_top1",            diag["z_top1"])
            trial.set_user_attr("centroid_z_margin",           diag["z_margin"])
            trial.set_user_attr("centroid_margin_percentile",  diag["margin_percentile"])
            # Logged only; not fed into the penalty. These two describe the
            # stability of the whole run rather than the final snapshot. The
            # three metrics above look at the end state and miss cases like
            # credit-g trial #47, where margin_percentile was 1.0 yet
            # reinitialisation never settled during training. Whether they
            # actually correlate with bad outcomes (reproducibility, test
            # score) should be checked across many trials via
            # study.trials_dataframe() before either enters the objective.
            if "reinit_per_epoch" in diag:
                trial.set_user_attr("centroid_reinit_per_epoch", diag["reinit_per_epoch"])
            if "active_ratio_std" in diag:
                trial.set_user_attr("centroid_active_ratio_std", diag["active_ratio_std"])
            penalty_cap  = 0.05
            penalty_frac = penalty_cap * (1.0 - diag["margin_percentile"])
            if penalty_frac > 0.0:
                if tasktype == "regression":
                    result = result * (1.0 + penalty_frac)   # higher rmse is worse
                else:
                    result = result * (1.0 - penalty_frac)   # higher acc is better
                trial.set_user_attr("centroid_penalty_frac", penalty_frac)

        # Release GPU memory between trials. Each trial builds a fresh model
        # and optimizer; if PyTorch keeps the previous trial's allocation in
        # its cache, the collapse guard in supervised.py reads "no free
        # memory" and misfires.
        # `del` alone is not enough: the autograd graph (tensor <-> grad_fn)
        # forms reference cycles that CPython's refcounting cannot reclaim
        # immediately, so the cyclic collector has to run. On small datasets
        # (id=41027, N=35,855) the per-trial residue is invisible, but on
        # larger ones (id=41150, N=104,050) with heavy hyperparameters the GPU
        # filled up gradually until the guard misfired around trial 7.
        del model, wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return result

    # ── Callbacks (same as upstream MultiTab) ──────────────
    def stop_when_reached_optimal(study, trial):
        if study.best_value >= 1.0:
            study.stop()

    if tasktype == "regression":
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[lambda study, trial: joblib.dump(study, fname)])
    else:
        study.optimize(objective, n_trials=remaining_trials,
                       callbacks=[stop_when_reached_optimal,
                                  lambda study, trial: joblib.dump(study, fname)])

    # ── Total training time (same as upstream MultiTab) ────
    total_training_time = sum([
        trial.user_attrs.get("training_time", 0)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ])
    study.set_user_attr("total_training_time", total_training_time)

    # ── Save results (same as upstream MultiTab) ───────────
    print("#############################################")
    print(env_info)
    print(study.best_trial.user_attrs)
    df = study.trials_dataframe()
    df.to_csv(os.path.join(savepath, f"data={args.openml_id}{_ablation_tag}..seed={args.seed}..model=tabera.csv"), index=False)
    joblib.dump(study, fname)
    print(fname)
    print("#############################################")