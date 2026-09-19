"""Final architecture defaults; safe to import before configuring CUDA."""
# Default: tangent correction, gamma=1, MultiTab val_loss early stopping.
# Evaluate the terminal model without best-checkpoint restore, matching the
# completed 105-run benchmark protocol.
# Accuracy studies carry no ..esm tag; val_loss studies carry ..esm=val_loss.
# num_bins=8 is the fixed dynamics2d fallback and joint HPO's initial value.
# Joint PLE trials override it; ple_d_embedding is searched with initial value 12.
FINAL_CONFIG = dict(correction_geometry="tangent", head_input_scale="unit",
                    beta_param="sigmoid", tie_rule="first", early_stop_metric="val_loss",
                    cat_combine="onehot", num_embedding="ple", num_bins=8,
                    cat_embed_dim=16, disable_dead_reinit=False, allow_self_retrieval=False)
