"""Final architecture defaults; safe to import before configuring CUDA."""
FINAL_CONFIG = dict(correction_geometry="unit_tangent", head_input_scale="auto",
                    beta_param="sigmoid", tie_rule="first", early_stop_metric="accuracy",
                    cat_combine="onehot", num_embedding="ple", num_bins=8,
                    cat_embed_dim=16, disable_dead_reinit=False, allow_self_retrieval=False)
