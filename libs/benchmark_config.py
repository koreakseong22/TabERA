"""Final architecture defaults; safe to import before configuring CUDA."""
# early_stop_metric="val_loss" is the MultiTab protocol (Lee et al.): stop on the
# batch-averaged validation loss with patience 20 and evaluate the model at
# the epoch training stopped, no best-checkpoint restore. The HPO objective
# stays validation accuracy / RMSE, as upstream. "accuracy" (best-val-acc
# checkpoint restore) is the earlier TabERA behaviour and is now an ablation
# arm; studies recorded under it carry no ..esm tag and are not reused here.
# num_bins=8 is the fixed dynamics2d fallback and joint HPO's initial value.
# Joint PLE trials override it; ple_d_embedding is searched with initial value 12.
FINAL_CONFIG = dict(correction_geometry="unit_tangent", head_input_scale="auto",
                    beta_param="sigmoid", tie_rule="first", early_stop_metric="val_loss",
                    cat_combine="onehot", num_embedding="ple", num_bins=8,
                    cat_embed_dim=16, disable_dead_reinit=False, allow_self_retrieval=False)
