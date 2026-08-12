"""
libs/tabera.py
============
TabERA — Prototype-conditioned prediction with local evidence retrieval.

Forward
───────
    x
    ↓  TabularEmbedder
    q                                    query embedding
    ↓  argmax cos(q, c)  (STE: forward=hard, backward=soft)
    a                                    prototype assignment
    ├──────────────────────────┐
    ▼                          ▼
  c = prototype[a]        G(a) = its members
    ▼                          ▼
  h = c + beta*normalize(q-c)  NN(q, G(a)), k
    ▼                          ▼
  z = W*h                    explanation layers (1)(2)(3)
 prediction

The single assignment `a` fixes both the prediction baseline `c` and the
retrieval pool `G(a)`.

Invariant
─────────
    logits = (W*c + b) + W*(beta*r),   r = normalize(q - c)

dev_head is a single Linear, so this decomposition is an identity rather than
an approximation (measured residual ~1e-08). Explanation layer (3) depends on
it.

⚠ Retrieval is not on the prediction path. Neighbours never appear in the
  `logits` expression. Retrieval is explanation-only, and that is a settled
  design decision rather than the outcome of a rejection
  (TABERA_V3_ARCHITECTURE.md sections 12 and 14).

⚠ `r` is **not** a deviation from the prototype. ||c|| is fixed at 1 while
  ||q|| ranges from 7 to 1197, so cos(r, q) ~ 1.000. The accurate name is
  "query direction", and beta is its relative contribution (section 9). Where
  the documentation and the code appear to disagree, this comment is the
  reference.

The only training signal is cross-entropy. The prototype layer carries no
gradient-based objective: assignment is straight-through, the update is EMA,
and maintenance is dead-prototype recovery -- none of which read labels
(sections 10-4, 15-1, 16-2).

The earlier fusion_mode / aggregator / L_nbr / commitment / residual-VQ paths
are frozen in legacy/v3ema2_full/ and are not reproduced here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.prototypes import CentroidLayer

class ResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )
    def forward(self, x): return x + self.net(x)


class TabularEmbedder(nn.Module):
    """Project numeric and categorical features into one embedding space.

    History and rationale
    ─────────────────────
    (1) Originally every feature went through a single LayerNorm + Linear with
        no numeric/categorical distinction. Categorical features are
        LabelEncoder integer codes -- nominal, unordered -- yet were treated as
        continuous scalars, so an accidental encoding order was read as real
        distance ("category 0 and 3 are farther apart than 0 and 1"). Across
        29 TabZilla datasets the share of categorical features correlated
        robustly with the AUROC gap against baselines (Spearman rho = -0.63,
        p = 0.0003).

    (2) Speed: instead of one nn.Embedding per categorical column inside a
        Python loop, the cardinalities are concatenated into a single table
        addressed by offsets (measured 4.17x faster on nomao).

    (3) sum vs concat: Guo & Berkhahn (2016) use concat, while this started
        with sum. `cat_combine="concat"` selects the original behaviour --
        per-column embeddings concatenated, then projected to embed_dim.

    (4) The TabM / ModernNCA / TabR line turned out to favour a different
        combination: plain **one-hot** for categorical features (no learned
        parameters) with **PLE** (piecewise linear encoding, Gorishniy et al.
        2022) for numeric ones. `cat_combine="onehot"` with
        `num_embedding="ple"` reproduces it. TabM (one-hot + PLE-family) and
        ModernNCA (one-hot + PLR) share essentially this preprocessing.

        PLE (Gorishniy et al. 2022): for per-column quantile boundaries
        b_0 < ... < b_T, z_t = clamp((x - b_{t-1}) / (b_t - b_{t-1}), 0, 1) --
        0 below bin t, 1 above it, and the relative position within it
        otherwise. The boundaries are computed on the training split and
        passed in as `num_bin_edges`, the same pattern as cat_cardinalities.

    (5) PLR (lite), what TabR (Gorishniy et al. 2024) and ModernNCA (Ye et al.
        2024) actually use: a learnable periodic function rather than bins,
        followed by a Linear + ReLU shared across columns. Verified against
        the official implementation (yandex-research/rtdl-num-embeddings):
        ReLU(Linear(CosSin(2*pi*Linear(x, bias=False)))), where "lite" means
        only the outer Linear is shared.

        Neither won outright on the data -- PLR was unstable on datasets with
        very few numeric features (profb: AUROC fell to chance in one run).
        The constructor default is nevertheless `cat_combine="onehot"` with
        `num_embedding="plr_lite"`, chosen for consistency with the
        TabR/ModernNCA line rather than for a per-dataset score.

    ⚠ The CLI and HPO default is `ple`, not `plr_lite`. The scripts always
      pass num_embedding explicitly, so this constructor default only applies
      to direct callers.

    With cat_col_idx=None the class behaves exactly as the earlier
    numeric-only path regardless of these options, so raw-encoding
    checkpoints stay loadable.
    """
    def __init__(
        self,
        n_features: int,
        embed_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
        cat_col_idx: Optional[List[int]] = None,
        num_col_idx: Optional[List[int]] = None,
        cat_cardinalities: Optional[List[int]] = None,
        cat_combine: str = "onehot",
        cat_embed_dim: int = 16,
        num_embedding: str = "plr_lite",
        num_bin_edges: Optional[torch.Tensor] = None,
        ple_d_embedding: int = 12,   # per-feature embedding width for
                                      # PiecewiseLinearEmbeddings. The starting
                                      # value the rtdl_num_embeddings docs
                                      # recommend (d_embedding=12 with
                                      # activation=False). Fixed rather than
                                      # searched: adding a search dimension
                                      # without evidence is the mistake the
                                      # three PLR parameters already made.
        plr_n_frequencies: int = 16,
        plr_freq_scale: float = 0.01,
        plr_out_dim: int = 8,
    ):
        super().__init__()
        self.cat_col_idx = list(cat_col_idx) if cat_col_idx else []
        self.num_col_idx = list(num_col_idx) if num_col_idx is not None else None
        self.cat_combine = cat_combine
        self.num_embedding = num_embedding
        n_num = len(self.num_col_idx) if self.num_col_idx is not None else 0

        if self.cat_col_idx or (n_num > 0 and num_embedding in ("ple", "plr_lite")):
            if cat_combine not in ("sum", "concat", "onehot", "none"):
                raise ValueError(f"cat_combine must be 'sum'/'concat'/'onehot'/'none': {cat_combine}")
            if num_embedding not in ("linear", "ple", "plr_lite"):
                raise ValueError(f"num_embedding must be 'linear'/'ple'/'plr_lite': {num_embedding}")

            # ── Categorical setup ──
            self.cat_embeddings = None  # placeholder from the old loop version
            self.cat_embed_table = None
            if self.cat_col_idx:
                if cat_cardinalities is None or len(cat_cardinalities) != len(self.cat_col_idx):
                    raise ValueError(
                        "cat_cardinalities must be given with the same length "
                        "as cat_col_idx (one cardinality per column)."
                    )
                cardinalities = [int(c) for c in cat_cardinalities]
                offsets = torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(cardinalities[:-1]), dim=0).tolist())
                    if len(cardinalities) > 1 else [0],
                    dtype=torch.long,
                )
                self.register_buffer("_cat_offsets", offsets, persistent=True)
                self.register_buffer(
                    "_cat_cardinalities", torch.tensor(cardinalities, dtype=torch.long), persistent=True
                )
                total_vocab = sum(cardinalities)
                if cat_combine == "sum":
                    self.cat_embed_table = nn.Embedding(total_vocab, embed_dim)
                elif cat_combine == "concat":
                    self.cat_embed_table = nn.Embedding(total_vocab, cat_embed_dim)
                elif cat_combine == "onehot":
                    self._onehot_total_vocab = total_vocab  # no learned parameters

            # ── Numeric setup ──
            self.num_proj = None
            self.ple_n_bins = 0
            self.ple_d_embedding = 0
            self.plr_out_dim = 0
            if n_num > 0:
                if num_embedding == "ple":
                    if num_bin_edges is None:
                        raise ValueError(
                            "num_embedding='ple' requires num_bin_edges: the "
                            "per-column quantile boundaries precomputed on the "
                            "training split, shape (n_num, n_bins+1)."
                        )
                    self.register_buffer("ple_edges", num_bin_edges.clone(), persistent=True)
                    self.ple_n_bins = num_bin_edges.shape[1] - 1
                    # This used to pass the raw bin vector z straight into
                    # concat -> final_proj (a Linear shared by all features),
                    # which is rtdl_num_embeddings' PiecewiseLinearEncoding
                    # rather than the PiecewiseLinearEmbeddings that TabM
                    # recommends by default (activation=False). The latter
                    # applies Linear(PLE(x_i)) independently per feature, so
                    # each feature i needs its own learned (n_bins,
                    # d_embedding) weight. That weight is created here and
                    # contracted with z (einsum) in forward to give a
                    # per-feature embedding.
                    # ⚠ The "version B" initialisation in TabM's appendix A.3
                    #   may differ in detail; this uses the standard nn.Linear
                    #   scheme (Kaiming uniform family). Bit-exact
                    #   reproduction would require checking against
                    #   rtdl_num_embeddings.PiecewiseLinearEmbeddings.
                    self.ple_d_embedding = ple_d_embedding
                    self.ple_emb_weight = nn.Parameter(torch.empty(n_num, self.ple_n_bins, ple_d_embedding))
                    self.ple_emb_bias   = nn.Parameter(torch.zeros(n_num, ple_d_embedding))
                    bound = 1.0 / (self.ple_n_bins ** 0.5)
                    nn.init.uniform_(self.ple_emb_weight, -bound, bound)
                elif num_embedding == "plr_lite":
                    # PLR(lite) — TabR(Gorishniy et al. 2024): periodic embedding
                    # Per-column learnable frequencies -> a Linear shared by
                    # all columns -> ReLU. A different mechanism from PLE:
                    # values are represented by a periodic function rather
                    # than by bins.
                    # The frequencies are learned per column, since the natural
                    # scale differs per column, while only the Linear + ReLU
                    # that follows is shared. That sharing is what "lite"
                    # means -- far fewer parameters than the original PLR with
                    # a Linear per column, reported in the TabR paper as
                    # lighter with no loss in performance.
                    self.plr_freq = nn.Parameter(torch.randn(n_num, plr_n_frequencies) * plr_freq_scale)
                    self.plr_linear = nn.Linear(2 * plr_n_frequencies, plr_out_dim)  # shared
                    self.plr_out_dim = plr_out_dim
                elif cat_combine == "sum":
                    self.num_proj = nn.Sequential(nn.LayerNorm(n_num), nn.Linear(n_num, embed_dim))
                # concat/onehot/none with linear numeric: raw x_num is
                # concatenated as-is (num_proj = None)

            # ── Final combination ──
            if cat_combine == "sum" and num_embedding == "linear":
                # the original sum path
                self.final_proj = None
            else:
                # concat, onehot and PLE all follow "concatenate, then Linear"
                concat_dim = 0
                if n_num > 0:
                    if num_embedding == "ple":
                        concat_dim += self.ple_d_embedding * n_num
                    elif num_embedding == "plr_lite":
                        concat_dim += self.plr_out_dim * n_num
                    else:
                        concat_dim += n_num
                if self.cat_col_idx:
                    if cat_combine == "concat":
                        concat_dim += len(self.cat_col_idx) * cat_embed_dim
                    elif cat_combine == "onehot":
                        concat_dim += self._onehot_total_vocab
                    elif cat_combine == "sum":
                        concat_dim += embed_dim  # the sum result joins as one block
                self.final_proj = nn.Sequential(nn.LayerNorm(concat_dim), nn.Linear(concat_dim, embed_dim))
        else:
            self.cat_embed_table = None
            self.cat_embeddings = None
            self.final_proj = None
            self.num_proj = None
            self.ple_edges = None
            # ⚠ The attribute must stay named `proj`, not `num_proj`: existing
            #   checkpoints store keys as "embedder.proj.0.weight" and
            #   --from_saved_state would fail to load otherwise.
            self.proj = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, embed_dim))

        self.blocks = nn.Sequential(*[ResidualMLP(embed_dim, embed_dim * 2, dropout) for _ in range(n_layers)])

    def _encode_categorical(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode the categorical columns per cat_combine. Returns (B, cat_dim)."""
        if not self.cat_col_idx:
            return None
        x_cat = x[:, self.cat_col_idx].round().long()
        x_cat = torch.clamp(x_cat, min=torch.zeros_like(self._cat_cardinalities),
                             max=self._cat_cardinalities - 1)
        x_cat_global = x_cat + self._cat_offsets            # (B, n_cat), offset per column

        if self.cat_combine == "onehot":
            # No learned parameters. The offsets keep each column's one-hot
            # span disjoint, so summing them is mathematically the same as
            # concatenating the per-column one-hots (a block-diagonal layout).
            onehot = F.one_hot(x_cat_global, num_classes=self._onehot_total_vocab).sum(dim=1)
            return onehot.float()
        else:
            cat_embs = self.cat_embed_table(x_cat_global)    # (B, n_cat, D), one gather
            if self.cat_combine == "sum":
                return cat_embs.sum(dim=1)                    # (B, embed_dim)
            else:  # concat
                B = x.shape[0]
                return cat_embs.reshape(B, -1)                # (B, n_cat * cat_embed_dim)

    def _encode_numeric(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode the numeric columns per num_embedding."""
        if self.num_col_idx is None or len(self.num_col_idx) == 0:
            return None
        x_num = x[:, self.num_col_idx]
        if self.num_embedding == "ple":
            # PiecewiseLinearEmbeddings(activation=False) — TabM(Gorishniy et
            # al. 2024) recommends by default. Building the raw encoding z
            # from the bin boundaries is unchanged, but instead of emitting z
            # directly it is contracted with a per-feature learned weight
            # (ple_emb_weight) to give each feature a vector of width
            # ple_d_embedding -- equivalent to applying Linear(PLE(x_i))
            # independently per feature.
            lo = self.ple_edges[:, :-1]                       # (n_num, n_bins)
            hi = self.ple_edges[:, 1:]                         # (n_num, n_bins)
            x_expand = x_num.unsqueeze(-1)                     # (B, n_num, 1)
            frac = (x_expand - lo) / (hi - lo + 1e-8)           # (B, n_num, n_bins)
            z = torch.clamp(frac, 0.0, 1.0)                     # (B, n_num, n_bins)
            # (B, n_num, n_bins) x (n_num, n_bins, d) → (B, n_num, d)
            emb = torch.einsum("bnk,nkd->bnd", z, self.ple_emb_weight) + self.ple_emb_bias
            return emb.reshape(x_num.shape[0], -1)              # (B, n_num*ple_d_embedding)
        elif self.num_embedding == "plr_lite":
            # PLR(lite) (TabR, Gorishniy et al. 2024): periodic embedding
            # Per-column learned frequencies -> a Linear shared by all
            # columns -> ReLU. No Python loop: nn.Linear broadcasts over the
            # last dimension of a (B, n_num, 2k) tensor, which is exactly what
            # "shared Linear" means here.
            x_expand = x_num.unsqueeze(-1)                       # (B, n_num, 1)
            v = 2 * torch.pi * self.plr_freq * x_expand           # (B, n_num, k)
            periodic = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)  # (B, n_num, 2k)
            out = F.relu(self.plr_linear(periodic))               # -> (B, n_num, plr_out_dim)
            return out.reshape(x_num.shape[0], -1)                # (B, n_num*plr_out_dim)
        else:
            return x_num  # linear mode passes raw values to num_proj/final_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_proj is not None:
            parts = []
            num_repr = self._encode_numeric(x)
            if num_repr is not None:
                if self.num_proj is not None:
                    parts.append(self.num_proj(num_repr))
                else:
                    parts.append(num_repr)
            cat_repr = self._encode_categorical(x)
            if cat_repr is not None:
                parts.append(cat_repr)
            combined = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
            return self.blocks(self.final_proj(combined))
        elif self.cat_embed_table is not None:
            # The sum (+ linear numeric) path: embed_dim directly, no final_proj
            emb = None
            if self.num_proj is not None:
                emb = self.num_proj(x[:, self.num_col_idx])
            cat_repr = self._encode_categorical(x)
            emb = cat_repr if emb is None else emb + cat_repr
            return self.blocks(emb)
        else:
            return self.blocks(self.proj(x))


class MemoryBank(nn.Module):
    """Store of training embeddings, used for k-NN retrieval."""
    def __init__(self, max_size: int, embed_dim: int, n_size_buckets: int = 4,
                 group_round_unit: int = 256, vectorized_fallback: bool = True):
        super().__init__()
        self.max_size = max_size
        # Number of group-size buckets; a constant loop count in retrieve()
        self.n_size_buckets = n_size_buckets
        # The unit retrieve() rounds local_max_g to. It was hard-coded to 256
        # with no stated reason; 256 remains the default so behaviour is
        # unchanged, and it is exposed only so a sweep can measure it.
        self._group_round_unit = group_round_unit
        # Whether the cross-group fallback runs as batched tensor operations
        # (bmm + topk) instead of a per-sample Python loop with .item() syncs.
        # Added after torch.profiler showed the original path spending 51% of
        # its time in cudaStreamSynchronize and only 2% on actual GPU work.
        # True by default. 
        self._vectorized_fallback = vectorized_fallback
        self.register_buffer("keys",   torch.zeros(max_size, embed_dim))
        self.register_buffer("labels", torch.zeros(max_size))
        self.register_buffer("ptr",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("filled", torch.tensor(0, dtype=torch.long))
        # ── Normalisation cache: removes the repeated F.normalize inside
        # retrieve(). Updated incrementally in O(B) on update(); retrieve()
        # only gathers from it.
        self.register_buffer("_keys_norm", torch.zeros(max_size, embed_dim))
        # Which X_train row each slot holds. Comparing this against
        # FeatureStore.sample_ids settles "do the MemoryBank and FeatureStore
        # slots point at the same sample" by exact equality rather than by a
        # statistical approximation. -1 marks an unfilled slot, distinguishable
        # because real row numbers are always non-negative. It is a buffer, so
        # it travels with the best_state snapshot and restore automatically.
        self.register_buffer("sample_ids", torch.full((max_size,), -1, dtype=torch.long))

        # The threshold retrieve() uses to decide whether one unusually large
        # group makes it wasteful to pad everything else to its width.
        # ⚠ The default 4096 has no computational or literature basis. It is
        #   only a safe fallback for initialisation (epoch 0, before the GPU
        #   can be queried) and for CPU runs. During training,
        #   update_outlier_threshold() recomputes it from the actual free GPU
        #   memory once per epoch. Querying the GPU inside retrieve() -- once
        #   per batch -- would reintroduce the sync overhead that was removed
        #   earlier, so the query stays in supervised.py at epoch granularity.
        self._outlier_threshold = 4096

    def update_outlier_threshold(
        self,
        n_prototypes: int,
        free_bytes: "Optional[int]" = None,
        device: "Optional[torch.device]" = None,
        safety_fraction: float = 0.3,
    ) -> None:
        """
        Solve for the local_max_g threshold that keeps the tensor retrieve()
        would build on its normal (single-tensor) path below safety_fraction
        of the currently free GPU memory.

        This ties the threshold to an actual resource constraint instead of an
        unjustified constant (4096) -- the same principle as the collapse
        guard in supervised.py. It assumes one call per epoch: calling it per
        batch would reintroduce the sync overhead of querying the GPU.

        Parameters
        ──────────
        n_prototypes : total centroid count P, used as the worst-case bound on
                       the number of unique centroids U in one batch.
        free_bytes   : free GPU memory in bytes, if already queried. When None
                       the function queries it itself, costing one sync.
        device       : device to query when free_bytes is not supplied.
        safety_fraction : the normal-path tensor is treated as risky once it
                       exceeds this fraction of free memory (0.3 by default,
                       leaving room for auxiliary tensors such as Q_pad and
                       sim_u beyond keys_u).
        """
        if free_bytes is None:
            if device is None or not torch.cuda.is_available() or not str(device).startswith("cuda"):
                return  # no GPU memory concept (CPU): keep the 4096 fallback
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device)
            except Exception:
                return  # query failed: keep the fallback

        D = self.keys.shape[1]
        U_pad_worst = ((n_prototypes + 7) // 8) * 8  # worst-case unique centroids per batch
        # A factor of 3 roughly covers the auxiliary tensors (keys_u, Q_pad,
        # sim_u). This is an order-of-magnitude judgement, not an exact figure,
        # and the factor itself is as unverified as the one in the
        # supervised.py guard.
        denom = U_pad_worst * D * 4 * 3
        if denom <= 0:
            return
        new_threshold = int((free_bytes * safety_fraction) / denom)
        # Rounded down to retrieve()'s rounding unit (self._group_round_unit).
        # This used to be an independent hard-coded 256; coupling them means a
        # sweep over round_unit moves this together with it.
        _ru = self._group_round_unit
        new_threshold = max((new_threshold // _ru) * _ru, max(2 * _ru, 512))
        self._outlier_threshold = new_threshold

    @torch.no_grad()
    def update(self, keys, labels, sample_ids=None):
        B   = keys.shape[0]
        ptr = self.ptr.item()
        end = min(ptr + B, self.max_size)
        n   = end - ptr
        self.keys[ptr:end]   = keys[:n].detach()
        self.labels[ptr:end] = labels[:n].float().detach()
        # Normalise once here in O(B), instead of recomputing in every retrieve
        self._keys_norm[ptr:end] = F.normalize(keys[:n].detach(), dim=-1)
        # Without sample_ids the slots stay -1 (older call sites). The current
        # training loop always passes the X_train row numbers.
        if sample_ids is not None:
            self.sample_ids[ptr:end] = sample_ids[:n].detach().to(self.sample_ids.device)
        self.ptr    = torch.tensor(end % self.max_size, dtype=torch.long)
        self.filled = torch.tensor(min(self.filled.item() + n, self.max_size), dtype=torch.long)

    @torch.no_grad()
    def cache_sample_groups(
        self,
        sample_groups: "List[List[int]]",
        device: "torch.device",
        centroid_emb: "Optional[torch.Tensor]" = None,  # (P, D), for cross-group
    ) -> None:
        """
        Convert sample_groups to GPU tensors once and cache them.

        Called once per epoch after regroup_update, which removes the
        conversion cost from retrieve(). Groups are padded with -1 to the
        largest group size.

        Cross-group fallback
        ────────────────────
        When centroid_emb is given, the nearest centroid of each centroid is
        precomputed and cached. A group smaller than k then extends its search
        into that neighbouring group rather than falling back to a full
        search.
        """
        P   = len(sample_groups)
        max_g = max((len(g) for g in sample_groups), default=0)
        if max_g == 0:
            self._cached_groups      = None
            self._cached_group_sizes = None
            self._cached_extended    = None
            return

        # (P, max_g) padded tensor, -1 marks padding
        padded = torch.full((P, max_g), -1, dtype=torch.long)
        for p, g in enumerate(sample_groups):
            if g:
                padded[p, :len(g)] = torch.tensor(g, dtype=torch.long)

        self._cached_groups      = padded.to(device)         # (P, max_g)
        self._cached_group_sizes = torch.tensor(
            [len(g) for g in sample_groups], dtype=torch.long, device=device
        )                                                     # (P,)

        # ── Cross-group: cache of each group extended with its neighbour ──
        # For centroid p, its own group merged with the nearest centroid's.
        if centroid_emb is not None and P > 1:
            c = F.normalize(centroid_emb.detach(), dim=-1)    # (P, D)
            sim = c @ c.T                                     # (P, P)
            sim.fill_diagonal_(-1.0)  # exclude self
            nearest = sim.argmax(dim=-1)                      # (P,) nearest centroid

            extended_groups = []
            for p in range(P):
                # own group merged with the nearest group
                own    = sample_groups[p]
                near_p = nearest[p].item()
                neighbor = sample_groups[near_p]
                merged = own + neighbor
                extended_groups.append(merged)

            # cache the extended groups as a padded tensor too
            max_eg = max((len(g) for g in extended_groups), default=0)
            if max_eg > 0:
                padded_ext = torch.full((P, max_eg), -1, dtype=torch.long)
                for p, g in enumerate(extended_groups):
                    if g:
                        padded_ext[p, :len(g)] = torch.tensor(g, dtype=torch.long)
                self._cached_extended      = padded_ext.to(device)    # (P, max_eg)
                self._cached_extended_sizes = torch.tensor(
                    [len(g) for g in extended_groups], dtype=torch.long, device=device
                )
            else:
                self._cached_extended = None
        else:
            self._cached_extended = None

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,                       # (B, D)
        k: int,
        hard_assignment: "Optional[torch.Tensor]" = None,
        sample_groups:   "Optional[List[List[int]]]" = None,  # unused; the cache wins
        exclude_ids: "Optional[torch.Tensor]" = None,  # (B,) MemoryBank slots
            # whose sample_id equals the query's are dropped from the
            # candidates, preventing self-retrieval. None disables the
            # exclusion.
            # ⚠ The outlier path (the rare case where one group exceeds
            #   self._outlier_threshold, the else branch below) does not
            #   honour this argument yet -- the triggering condition could not
            #   be reproduced to verify a fix. On a dataset with a centroid
            #   large enough to take that branch, self-retrieval remains
            #   possible.
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fully vectorised k-NN retrieval.

        Returns (nk, neighbour_labels, top_k_idx):
          nk               (B, k, D)  neighbour key embeddings
          neighbour_labels (B, k)     neighbour labels
          top_k_idx        (B, k)     MemoryBank indices, for FeatureStore lookups

        This used to return nv as well -- the context_emb a neighbour held
        when it was stored. No consumer used it (values were built from
        label_emb and the query-neighbour offset only), and a probe across
        mfeat-zernike, vehicle and credit-approval found it statistically
        indistinguishable from a noise control once nk and the label were
        accounted for. It cost storage and lookup for no measurable benefit.
        """
        n   = self.filled.item()
        B   = query.shape[0]
        D   = query.shape[1]
        dev = query.device

        keys_full   = self.keys[:n]    # (n, D)
        labels_full = self.labels[:n]  # (n,)
        q_norm      = F.normalize(query, dim=-1)  # (B, D)

        # ── No cache, or before initialisation: fall back to a full search ──
        cached = getattr(self, '_cached_groups', None)
        if hard_assignment is None or cached is None or n < k:
            keys_all = self._keys_norm[:n]  # reuse the normalisation cache
            sim      = q_norm @ keys_all.T
            if exclude_ids is not None:
                # (B, n) where a slot's sample_id equals the query's own,
                # push the similarity to -inf so topk cannot select it
                _self_mask = self.sample_ids[:n].unsqueeze(0) == exclude_ids.unsqueeze(1).to(dev)
                sim = sim.masked_fill(_self_mask, -1e9)
            _, idx   = sim.topk(min(k, n), dim=-1)
            idx      = idx.clamp(0, n - 1)
            neighbour_labels = labels_full[idx]              # (B, k)
            return keys_full[idx], neighbour_labels, idx

        # ── Fully vectorised: no Python loop ──────────────────
        ha        = hard_assignment.to(dev)             # (B,)
        grp_sizes = self._cached_group_sizes[ha]        # (B,)

        # Which samples need the fallback: those whose group is smaller than k.
        # ⚠ With self-exclusion on, a group needs k+1 members to yield k
        #   neighbours after removing the query itself. The old condition
        #   (< k) sent groups of **exactly k** down the normal path, where
        #   masking self to -1e9 leaves only k-1 valid candidates and topk(k)
        #   returns the masked query as the k-th neighbour.
        #   Measured on ds=1489: all 8 samples of centroid 30, whose group is
        #   exactly 8 (= k), retrieved themselves. The bug predates this code.
        # ⚠ At inference sample_ids is not passed, so exclude_ids is None and
        #   the threshold stays k -- the --explain path is unchanged.
        _min_grp = k + (1 if exclude_ids is not None else 0)
        fallback_mask = grp_sizes < _min_grp            # (B,) bool
        normal_mask   = ~fallback_mask

        # ── Self-exclusion on the fallback path ────────────────────
        # The normal path (group >= k) and the global fallback (n < k) both
        # masked exclude_ids, but the **cross-group fallback (group < k) did
        # not**. Every sample assigned to a small group therefore retrieved
        # itself as the top-1 neighbour at similarity 1.000.
        #   Measured on credit-g: all 12 samples in groups smaller than k, and
        #   none of the 788 in larger groups.
        #
        # Scope: retrieval is outside the prediction path, so **predictions
        #   and the reported numbers are unaffected**. What it corrupted was
        #   the training-side diagnostics -- retrieval_label_purity
        #   overestimated and neighbor_label_entropy underestimated -- because
        #   the query's own label was always among its neighbours.
        #
        # The fallback computes similarities in seven places (extended group
        # vs global, times vectorised vs loop). Scattering the masking across
        # all of them is how one gets missed, so the self slot is resolved once
        # and applied through a shared helper.
        _self_slot = None
        if exclude_ids is not None:
            _eq  = self.sample_ids[:n].unsqueeze(0) == exclude_ids.unsqueeze(1).to(dev)
            _hit = _eq.any(dim=1)                                     # (B,)
            _self_slot = torch.where(
                _hit, _eq.float().argmax(dim=1),
                torch.full((_eq.shape[0],), -1, dtype=torch.long, device=dev))

        def _drop_self(sim, cand, rows):
            """Mask each row's own slot in sim (M, C) to -inf.

            cand (M, C) or (1, C) gives the memory slot each column refers
            to; for a global search, broadcasting arange(n) is enough.
            When the query is not in memory (_self_slot = -1) nothing is
            masked.
            """
            if _self_slot is None:
                return sim
            s = _self_slot[rows].reshape(-1, 1)                       # (M, 1)
            return sim.masked_fill((cand == s) & (s >= 0), -1e9)

        # Output buffers; fallback samples keep their zeros
        out_nk    = torch.zeros(B, k, D,          device=dev)
        out_labels = torch.zeros(B, k,            device=dev)
        top_k_idx = torch.zeros(B, k, dtype=torch.long, device=dev)

        # ── Normal samples: one centroid dedup pass, no size tiering ──────
        # At N=35,855, D<=256, B=256 the actual GPU work is under 1ms, while
        # the earlier size-tiered version spent most of its 25.4ms of CPU time
        # on bookkeeping: aten::index (71 calls), aten::nonzero (21),
        # aten::index_put_ (30) and aten::repeat_interleave (728us per call),
        # because each tier re-ran argsort / unique_consecutive / remap /
        # nonzero / repeat_interleave. The real topk and bmm accounted for
        # 260us of 933us.
        #
        # At this scale, cutting kernel launches matters far more than saving
        # FLOPs, so the tiering is gone and only the dedup remains:
        #   - centroid dedup stays, since preventing queries that point at the
        #     same centroid from gathering the candidates repeatedly removes
        #     the largest waste on its own
        #   - the width (local_max_g) is bounded by the largest group among
        #     the centroids that actually appear in this batch, still far
        #     below the global max_g
        #   - repeat_interleave is replaced by bucketize: same result, much
        #     cheaper
        #   - nonzero is called once for the whole batch rather than per tier
        if normal_mask.any():
            nm_idx = normal_mask.nonzero(as_tuple=True)[0]  # (Bn,)
            ha_nm  = ha[nm_idx]                              # (Bn,)
            q_nm   = q_norm[nm_idx]                          # (Bn, D)
            Bn = nm_idx.shape[0]                              # a Python int; no sync

            # ── centroid dedup, once for the batch ──
            csort_idx   = torch.argsort(ha_nm)                # (Bn,)
            ha_c_sorted = ha_nm[csort_idx]
            q_c_sorted  = q_nm[csort_idx]

            uniq, counts = torch.unique_consecutive(ha_c_sorted, return_counts=True)  # (U,)
            U = uniq.shape[0]

            offsets = counts.cumsum(0)                          # (U,) exclusive end of each group
            group_id = torch.bucketize(
                torch.arange(Bn, device=dev), offsets, right=True
            )                                                    # (Bn,) 0..U-1
            rank = torch.arange(Bn, device=dev) - (offsets[group_id] - counts[group_id])  # rank within the group

            grp_sizes_u = self._cached_group_sizes[uniq]         # (U,) group sizes in this batch
            local_max_g_raw = max(int(grp_sizes_u.max()), k)

            # When local_max_g_raw is unusually large -- one centroid having
            # absorbed a large share of the data -- padding all U centroids to
            # that width blows up memory and compute. Measured on id=41150
            # (N=104,050), where max_cluster_size grew from 3,526 to 34,195.
            # Most centroids stayed healthy and only a few became huge, so
            # active_ratio-based collapse detection could not catch it.
            #
            # The threshold is self._outlier_threshold, computed by
            # update_outlier_threshold() from the actual free GPU memory once
            # per epoch (driven from supervised.py) -- not a fixed constant.
            # Before the first update, or on CPU, the __init__ fallback of
            # 4096 applies, which is itself unjustified; that remains a
            # documented limitation.
            # ⚠ exclude_ids reordered into the csort coordinate system. It
            #   used to be built inside the `if` branch (group below the
            #   threshold) only, yet the self-mask in the `else` (tier
            #   expansion) branch reads it too -- crossing the threshold died
            #   with UnboundLocalError. It is hoisted so both branches share it.
            #   ⚠ The tier branch runs only when the largest group exceeds
            #     _outlier_threshold (4096 by default). The ten benchmark
            #     datasets top out at 2,615, so it never executed and the bug
            #     stayed hidden.
            ids_c_sorted = (exclude_ids[nm_idx].to(dev)[csort_idx]
                            if exclude_ids is not None else None)
            _OUTLIER_THRESHOLD = self._outlier_threshold

            if local_max_g_raw <= _OUTLIER_THRESHOLD:
                # ── Normal path ────────────────────────────────────────
                _round_u = 8
                U_pad = ((U + _round_u - 1) // _round_u) * _round_u
                if U_pad > U:
                    pad_ids = uniq[:1].expand(U_pad - U)
                    uniq_p  = torch.cat([uniq, pad_ids], dim=0)
                else:
                    uniq_p = uniq

                max_q_raw = int(counts.max())
                max_q = ((max_q_raw + 15) // 16) * 16
                max_q = min(max_q, Bn)

                _ru = self._group_round_unit
                local_max_g = ((local_max_g_raw + _ru - 1) // _ru) * _ru
                local_max_g = min(local_max_g, self._cached_groups.shape[1])

                Q_pad = torch.zeros(U_pad, max_q, D, device=dev)
                Q_pad[group_id, rank] = q_c_sorted

                cand_u  = self._cached_groups[uniq_p, :local_max_g]
                valid_u = cand_u >= 0
                safe_u  = cand_u.clamp(min=0, max=n - 1)

                keys_u = self._keys_norm[:n][safe_u.reshape(-1)].view(U_pad, local_max_g, D)

                sim_u = torch.bmm(Q_pad, keys_u.transpose(1, 2))
                sim_u = sim_u.masked_fill(~valid_u.unsqueeze(1), -1e9)

                if exclude_ids is not None:
                    # Rearrange exclude_ids into the same (U_pad, max_q)
                    # layout as Q_pad. Padding positions -- slots outside the
                    # real (group_id, rank) pairs -- keep the sentinel -1,
                    # which never occurs as a real sample_id (the same
                    # convention as the sample_ids buffer initial value), so
                    # they cannot match any candidate. Those slots are filtered
                    # out later by i_final_u[group_id, rank] anyway; the
                    # sentinel is defensive.
                    Ids_pad = torch.full((U_pad, max_q), -1, dtype=ids_c_sorted.dtype, device=dev)
                    Ids_pad[group_id, rank] = ids_c_sorted
                    cand_ids_u = self.sample_ids[safe_u.reshape(-1)].view(U_pad, local_max_g)  # (U_pad, local_max_g)
                    _self_mask_u = cand_ids_u.unsqueeze(1) == Ids_pad.unsqueeze(-1)  # (U_pad, max_q, local_max_g)
                    sim_u = sim_u.masked_fill(_self_mask_u, -1e9)

                k_eff = min(k, local_max_g)
                _, top_u  = sim_u.topk(k_eff, dim=-1)
                i_final_u = safe_u.unsqueeze(1).expand(-1, max_q, -1).gather(2, top_u)
                i_final_c_sorted = i_final_u[group_id, rank]

                final_pos = nm_idx[csort_idx]
                top_k_idx[final_pos, :k_eff]  = i_final_c_sorted
                out_nk[final_pos, :k_eff]     = keys_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff, D)
                out_labels[final_pos, :k_eff] = labels_full[i_final_c_sorted.reshape(-1)].view(Bn, k_eff)

            else:
                # ── Outlier path: split large and small groups (rare) ──
                big_mask = grp_sizes_u > _OUTLIER_THRESHOLD          # (U,) bool
                for tier_mask in (~big_mask, big_mask):
                    if not tier_mask.any():
                        continue
                    query_in_tier = tier_mask[group_id]              # (Bn,) bool
                    if not query_in_tier.any():
                        continue
                    sel_pos = query_in_tier.nonzero(as_tuple=True)[0]  # (Bt,) csorted coords

                    tier_uniq_local = tier_mask.nonzero(as_tuple=True)[0]  # (Ut,) 0..U-1 coords
                    Ut = tier_uniq_local.shape[0]
                    remap = torch.full((U,), -1, dtype=torch.long, device=dev)
                    remap[tier_uniq_local] = torch.arange(Ut, device=dev)

                    local_gid  = remap[group_id[sel_pos]]            # (Bt,) 0..Ut-1
                    local_rank = rank[sel_pos]                        # (Bt,)
                    q_sel      = q_c_sorted[sel_pos]                  # (Bt, D)

                    tier_centroid_ids = uniq[tier_uniq_local]         # (Ut,) actual centroid ids
                    tier_counts       = counts[tier_uniq_local]       # (Ut,)

                    Ut_pad = ((Ut + 7) // 8) * 8
                    if Ut_pad > Ut:
                        pad_ids2 = tier_centroid_ids[:1].expand(Ut_pad - Ut)
                        tier_centroid_ids_p = torch.cat([tier_centroid_ids, pad_ids2], dim=0)
                    else:
                        tier_centroid_ids_p = tier_centroid_ids

                    max_q_tier_raw = int(tier_counts.max())
                    max_q_tier = ((max_q_tier_raw + 15) // 16) * 16
                    max_q_tier = min(max_q_tier, Bn)

                    local_max_g_tier_raw = max(
                        int(self._cached_group_sizes[tier_centroid_ids].max()), k
                    )
                    _ru_t = self._group_round_unit
                    local_max_g_tier = ((local_max_g_tier_raw + _ru_t - 1) // _ru_t) * _ru_t
                    local_max_g_tier = min(local_max_g_tier, self._cached_groups.shape[1])

                    Q_pad_t = torch.zeros(Ut_pad, max_q_tier, D, device=dev)
                    Q_pad_t[local_gid, local_rank] = q_sel

                    cand_t  = self._cached_groups[tier_centroid_ids_p, :local_max_g_tier]
                    valid_t = cand_t >= 0
                    safe_t  = cand_t.clamp(min=0, max=n - 1)
                    keys_t  = self._keys_norm[:n][safe_t.reshape(-1)].view(Ut_pad, local_max_g_tier, D)

                    sim_t = torch.bmm(Q_pad_t, keys_t.transpose(1, 2))
                    sim_t = sim_t.masked_fill(~valid_t.unsqueeze(1), -1e9)

                    k_eff_t = min(k, local_max_g_tier)
                    # The tier expansion path had no self-exclusion either.
                    # Its layout matches the main path (the Ids_pad block
                    # above) -- both use [group, rank] coordinates -- so it is
                    # built the same way.
                    # ⚠ This path never executed on credit-g, so the fix could
                    #   not be verified by measurement. It rests only on the
                    #   structural match with the main path; confirm zero
                    #   self-retrieval on a dataset where the tier actually
                    #   runs.
                    if exclude_ids is not None:
                        Ids_pad_t = torch.full((Ut_pad, max_q_tier), -1,
                                                dtype=ids_c_sorted.dtype, device=dev)
                        Ids_pad_t[local_gid, local_rank] = ids_c_sorted[sel_pos]
                        cand_ids_t = self.sample_ids[safe_t.reshape(-1)].view(
                            Ut_pad, local_max_g_tier)
                        sim_t = sim_t.masked_fill(
                            cand_ids_t.unsqueeze(1) == Ids_pad_t.unsqueeze(-1), -1e9)
                    _, top_t  = sim_t.topk(k_eff_t, dim=-1)
                    i_final_t = safe_t.unsqueeze(1).expand(-1, max_q_tier, -1).gather(2, top_t)
                    i_final_sel = i_final_t[local_gid, local_rank]      # (Bt, k_eff_t)

                    final_pos_t = nm_idx[csort_idx[sel_pos]]  # (Bt,) original batch coords
                    top_k_idx[final_pos_t, :k_eff_t]  = i_final_sel
                    out_nk[final_pos_t, :k_eff_t]     = keys_full[i_final_sel.reshape(-1)].view(-1, k_eff_t, D)
                    out_labels[final_pos_t, :k_eff_t] = labels_full[i_final_sel.reshape(-1)].view(-1, k_eff_t)

        # Fallback samples: extend the search into the neighbouring centroid's
        # group (cross-group). The earlier behaviour left zeros here, which
        # effectively abandoned retrieval for these samples.
        if fallback_mask.any():
            fb_idx = fallback_mask.nonzero(as_tuple=True)[0]   # (Bf,)
            ha_fb  = ha[fb_idx]                                 # (Bf,)
            q_fb   = q_norm[fb_idx]                             # (Bf, D)
            Bf     = fb_idx.shape[0]

            # Use the cross-group cache when present, otherwise search globally
            ext = getattr(self, '_cached_extended', None)

            if self._vectorized_fallback:
                # ── Vectorised path ────────────────────────────────────
                # The original handled each fallback sample in a Python loop
                # with .item() syncs; torch.profiler measured
                # cudaStreamSynchronize at 51% of self CPU time against 2% of
                # actual GPU work. This computes the same candidate set and
                # the same topk selection in batched tensor operations
                # (bmm/gather plus a masked topk).
                # ⚠ That the two paths produce identical results is supported
                #   by the benchmark script's correctness check, not by this
                #   comment.
                if ext is not None:
                    cand_ext  = ext[ha_fb]                          # (Bf, max_eg)
                    ext_sizes = self._cached_extended_sizes[ha_fb]  # (Bf,)
                    valid_ext = (cand_ext >= 0)
                    safe_ext  = cand_ext.clamp(min=0, max=n - 1)

                    still_small = ext_sizes < k
                    use_ext     = ~still_small

                    if use_ext.any():
                        ext_idx   = use_ext.nonzero(as_tuple=True)[0]      # (Bs,)
                        max_eg    = safe_ext.shape[1]
                        k_eff_ext = min(k, max_eg)

                        q_sel     = q_fb[ext_idx]                          # (Bs, D)
                        safe_sel  = safe_ext[ext_idx]                      # (Bs, max_eg)
                        valid_sel = valid_ext[ext_idx]                     # (Bs, max_eg)

                        keys_sel = self._keys_norm[:n][safe_sel.reshape(-1)] \
                                       .view(ext_idx.shape[0], max_eg, D)
                        sim_sel  = torch.bmm(
                            q_sel.unsqueeze(1), keys_sel.transpose(1, 2)
                        ).squeeze(1)                                      # (Bs, max_eg)
                        sim_sel  = sim_sel.masked_fill(~valid_sel, -1e9)

                        sim_sel = _drop_self(sim_sel, safe_sel, fb_idx[ext_idx])
                        _, top_sel   = sim_sel.topk(k_eff_ext, dim=-1)     # (Bs, k_eff_ext)
                        real_idx_sel = safe_sel.gather(1, top_sel).clamp(0, n - 1)

                        final_pos_sel = fb_idx[ext_idx]
                        top_k_idx[final_pos_sel, :k_eff_ext]  = real_idx_sel
                        out_nk[final_pos_sel, :k_eff_ext]     = keys_full[real_idx_sel.reshape(-1)] \
                            .view(-1, k_eff_ext, D)
                        out_labels[final_pos_sel, :k_eff_ext] = labels_full[real_idx_sel.reshape(-1)] \
                            .view(-1, k_eff_ext)

                    if still_small.any():
                        ss_idx   = still_small.nonzero(as_tuple=True)[0]   # (Bt,)
                        q_ss     = q_fb[ss_idx]                            # (Bt, D)
                        keys_all = self._keys_norm[:n]
                        k_eff_ss = min(k, n)

                        sim_all    = q_ss @ keys_all.T                     # (Bt, n)
                        sim_all = _drop_self(sim_all, torch.arange(n, device=dev).unsqueeze(0), fb_idx[ss_idx])
                        _, idx_all = sim_all.topk(k_eff_ss, dim=-1)
                        idx_all    = idx_all.clamp(0, n - 1)

                        final_pos_ss = fb_idx[ss_idx]
                        top_k_idx[final_pos_ss, :k_eff_ss]  = idx_all
                        out_nk[final_pos_ss, :k_eff_ss]     = keys_full[idx_all.reshape(-1)] \
                            .view(-1, k_eff_ss, D)
                        out_labels[final_pos_ss, :k_eff_ss] = labels_full[idx_all.reshape(-1)] \
                            .view(-1, k_eff_ss)
                else:
                    keys_all  = self._keys_norm[:n]
                    k_eff_all = min(k, n)
                    sim_all    = q_fb @ keys_all.T                         # (Bf, n)
                    sim_all = _drop_self(sim_all, torch.arange(n, device=dev).unsqueeze(0), fb_idx)
                    _, idx_all = sim_all.topk(k_eff_all, dim=-1)
                    idx_all    = idx_all.clamp(0, n - 1)

                    top_k_idx[fb_idx, :k_eff_all]  = idx_all
                    out_nk[fb_idx, :k_eff_all]     = keys_full[idx_all.reshape(-1)] \
                        .view(-1, k_eff_all, D)
                    out_labels[fb_idx, :k_eff_all] = labels_full[idx_all.reshape(-1)] \
                        .view(-1, k_eff_all)

            elif ext is not None:
                cand_ext   = ext[ha_fb]                         # (Bf, max_eg)
                ext_sizes  = self._cached_extended_sizes[ha_fb] # (Bf,)
                valid_ext  = (cand_ext >= 0)
                safe_ext   = cand_ext.clamp(min=0, max=n - 1)

                # Extended group still smaller than k: fall back to a full search
                still_small = ext_sizes < k
                use_ext     = ~still_small

                if use_ext.any():
                    ext_idx   = use_ext.nonzero(as_tuple=True)[0]
                    max_eg    = safe_ext.shape[1]
                    k_eff_ext = min(k, max_eg)

                    for i in ext_idx:
                        i = i.item()
                        b_pos = fb_idx[i]
                        si_e  = safe_ext[i]                     # (max_eg,)
                        vm_e  = valid_ext[i]                    # (max_eg,)
                        q_e   = q_fb[i:i+1]                     # (1, D)

                        keys_e = self._keys_norm[:n][si_e[vm_e]]  # (valid, D) from the cache
                        if keys_e.shape[0] < k:
                            # still short: search globally
                            keys_all = self._keys_norm[:n]
                            sim_all  = q_e @ keys_all.T
                            sim_all = _drop_self(sim_all, torch.arange(n, device=dev).unsqueeze(0), b_pos.reshape(1))
                            _, idx_all = sim_all.topk(min(k, n), dim=-1)
                            idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                            out_nk[b_pos]     = keys_full[idx_all]
                            out_labels[b_pos] = labels_full[idx_all]
                            top_k_idx[b_pos]  = idx_all
                        else:
                            sim_e = q_e @ keys_e.T                           # (1, valid)
                            sim_e = _drop_self(sim_e, si_e[vm_e].unsqueeze(0), b_pos.reshape(1))
                            _, top_e = sim_e.topk(min(k, keys_e.shape[0]), dim=-1)
                            real_idx = si_e[vm_e][top_e.squeeze(0)]          # (k,)
                            real_idx = real_idx.clamp(0, n - 1)
                            kk = real_idx.shape[0]
                            out_nk[b_pos, :kk]     = keys_full[real_idx]
                            out_labels[b_pos, :kk] = labels_full[real_idx]
                            top_k_idx[b_pos, :kk]  = real_idx

                # Samples the extension could not fill: search globally
                if still_small.any():
                    ss_idx = still_small.nonzero(as_tuple=True)[0]
                    for i in ss_idx:
                        i = i.item()
                        b_pos = fb_idx[i]
                        q_s   = q_fb[i:i+1]
                        keys_all = self._keys_norm[:n]
                        sim_all  = q_s @ keys_all.T
                        sim_all = _drop_self(sim_all, torch.arange(n, device=dev).unsqueeze(0), b_pos.reshape(1))
                        _, idx_all = sim_all.topk(min(k, n), dim=-1)
                        idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                        out_nk[b_pos]     = keys_full[idx_all]
                        out_labels[b_pos] = labels_full[idx_all]
                        top_k_idx[b_pos]  = idx_all
            else:
                # No extension cache: search globally
                keys_all = self._keys_norm[:n]
                for i in range(Bf):
                    b_pos = fb_idx[i]
                    q_s   = q_fb[i:i+1]
                    sim_all = q_s @ keys_all.T
                    sim_all = _drop_self(sim_all, torch.arange(n, device=dev).unsqueeze(0), b_pos.reshape(1))
                    _, idx_all = sim_all.topk(min(k, n), dim=-1)
                    idx_all = idx_all.squeeze(0).clamp(0, n - 1)
                    out_nk[b_pos]     = keys_full[idx_all]
                    out_labels[b_pos] = labels_full[idx_all]
                    top_k_idx[b_pos]  = idx_all

        return out_nk, out_labels, top_k_idx

class FeatureStore:
    """
    Explanation-only store, fully independent of retrieval.

    Design
    ──────
    - Kept in sync with MemoryBank's ptr: same order, same slots.
    - Stores exactly the X that forward() hands to embedder(X). An earlier
      note claimed these were inverse-transformed values; they are not.
      `self._feature_store.update(X)` passes the same tensor the embedder
      saw, so numeric columns remain [0,1] quantiles and converting them to
      readable units is the display layer's job.
    """

    def __init__(
        self,
        max_size: int,
        n_features: int,
        col_names: Optional[List[str]] = None,
    ) -> None:
        self.max_size   = max_size
        self.n_features = n_features
        self.col_names  = col_names or [f"f{i}" for i in range(n_features)]
        self._store  = torch.zeros(max_size, n_features)
        self._ptr    = 0
        self._filled = 0
        # X_train row numbers, paired with MemoryBank.sample_ids. This class
        # is not an nn.Module, so they cannot be buffers and must travel in the
        # feature_store_state tuple at save and restore time -- in the
        # reproduce.py checkpoint, in --from_saved_state, and in the
        # best_feature_store snapshot in supervised.py.
        self._sample_ids = torch.full((max_size,), -1, dtype=torch.long)

    @torch.no_grad()
    def update(self, X_raw: torch.Tensor, sample_ids: Optional[torch.Tensor] = None) -> None:
        B   = X_raw.shape[0]
        end = min(self._ptr + B, self.max_size)
        n   = end - self._ptr
        self._store[self._ptr:end] = X_raw[:n].detach().cpu().float()
        if sample_ids is not None:
            self._sample_ids[self._ptr:end] = sample_ids[:n].detach().cpu()
        self._ptr    = end % self.max_size
        self._filled = min(self._filled + n, self.max_size)

    @torch.no_grad()
    def retrieve(self, indices: torch.Tensor) -> List[Dict[str, float]]:
        idx_cpu = indices.detach().cpu().clamp(0, self._filled - 1)
        if idx_cpu.dim() == 1:
            rows = self._store[idx_cpu]
            return [
                {self.col_names[fi]: float(rows[ki, fi]) for fi in range(self.n_features)}
                for ki in range(rows.shape[0])
            ]
        else:
            B, k = idx_cpu.shape
            result = []
            for b in range(B):
                rows = self._store[idx_cpu[b]]
                result.append([
                    {self.col_names[fi]: float(rows[ki, fi]) for fi in range(self.n_features)}
                    for ki in range(k)
                ])
            return result

    def top_features(self, sample_dict: Dict[str, float], n: int = 6) -> Dict[str, float]:
        return dict(sorted(sample_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:n])

    def __repr__(self) -> str:
        return f"FeatureStore(max_size={self.max_size}, n_features={self.n_features}, filled={self._filled})"


# ─────────────────────────────────────────────────────────────
# TabERA
# ─────────────────────────────────────────────────────────────



class TabERA(nn.Module):
    """Prototype-conditioned tabular classifier.

    Component roles follow TABERA_V3_ARCHITECTURE.md section 2.

        TabularEmbedder   input -> q. It does not decide the class.
        CentroidLayer     assignment, EMA update, dead recovery
        dev_head (W)      **shared** between c and beta*r. Splitting it
                          reopens the old evasion where ||W_q|| grew to
                          bypass the beta constraint.
        MemoryBank        NN(q, G(a)). Contributes nothing to prediction.
        FeatureStore      raw features for explanation only; no role in
                          training or prediction.
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 128,
        n_prototypes: int = 8,
        k: int = 8,
        # ⚠ k is an explanation budget. It does not affect prediction -- the
        #   retrieval result never enters the logits -- so it is not searched.
        prototype_labels: Optional[List[str]] = None,
        n_output: int = 1,
        memory_size: int = 10_000,
        embedder_layers: int = 2,
        dropout: float = 0.1,
        column_names: Optional[List[str]] = None,
        exclude_self_retrieval: bool = True,
        tasktype: str = "regression",
        n_classes: Optional[int] = None,
        routing_scale: float = 1.0,
        use_ema_codebook: bool = True,
        ema_decay: float = 0.99,
        vectorized_fallback: bool = True,
        cat_col_idx: Optional[List[int]] = None,
        num_col_idx: Optional[List[int]] = None,
        cat_cardinalities: Optional[List[int]] = None,
        cat_combine: str = "onehot",
        cat_embed_dim: int = 16,
        num_embedding: str = "plr_lite",
        num_bin_edges: Optional[torch.Tensor] = None,
        ple_d_embedding: int = 12,
        plr_n_frequencies: int = 16,
        plr_freq_scale: float = 0.01,
        plr_out_dim: int = 8,
        freeze_centroid_after: "int | None" = None,
        regroup_warmup_epochs: int = 0,
        dead_reinit_patience: int = 5,
        dead_reinit_noise_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.k            = k
        self.embed_dim    = embed_dim
        self.n_output     = n_output
        self.tasktype     = tasktype
        if tasktype in ("binclass", "multiclass"):
            # For multiclass, n_output is n_classes. For binclass, n_output
            # is 1 while the labels are two classes (0/1), so n_classes is
            # taken explicitly or defaults to 2.
            self._n_classes_for_labels = n_classes if n_classes is not None else (
                n_output if tasktype == "multiclass" else 2
            )
        else:
            self._n_classes_for_labels = None
        self.n_features   = n_features

        self.column_names = column_names
        self.exclude_self_retrieval = exclude_self_retrieval

        self.embedder = TabularEmbedder(
            n_features, embed_dim, embedder_layers, dropout,
            cat_col_idx=cat_col_idx, num_col_idx=num_col_idx,
            cat_cardinalities=cat_cardinalities,
            cat_combine=cat_combine, cat_embed_dim=cat_embed_dim,
            num_embedding=num_embedding, num_bin_edges=num_bin_edges,
            ple_d_embedding=ple_d_embedding,
            plr_n_frequencies=plr_n_frequencies, plr_freq_scale=plr_freq_scale,
            plr_out_dim=plr_out_dim,
        )

        self.prototype_layer = CentroidLayer(
            n_prototypes=n_prototypes,
            embed_dim=embed_dim,
            n_features=n_features,
            prototype_labels=prototype_labels,
            dropout=dropout,
            col_names=column_names,
            routing_scale=routing_scale,
            regroup_warmup_epochs=regroup_warmup_epochs,
            freeze_centroid_after=freeze_centroid_after,
            dead_reinit_patience=dead_reinit_patience,
            dead_reinit_noise_scale=dead_reinit_noise_scale,
            use_ema_codebook=use_ema_codebook,
            ema_decay=ema_decay,
        )

        # ── Memory bank (retrieval only) ──────────────
        self.memory = MemoryBank(memory_size, embed_dim,
                                  vectorized_fallback=vectorized_fallback)

        # ── FeatureStore (explanation only) ───────────
        self._feature_store: Optional[FeatureStore] = None
        if column_names and n_features > 0:
            self._feature_store = FeatureStore(
                max_size=memory_size,
                n_features=n_features,
                col_names=column_names,
            )

        # ── Prediction: h = c + beta*normalize(q-c), logits = W*h ──
        # ⚠ W is **shared** between the two terms. With a separate matrix per
        #   term, the optimizer grows the query-side norm (measured 24x) and
        #   evades the beta constraint. Sharing removes that route: growing W
        #   grows the c term as well.
        # ⚠ It must be a single Linear for the decomposition to be an
        #   identity. Adding a hidden layer turns explanation layer (3) into
        #   an approximation like SHAP or IG, with a baseline to choose.
        self.dev_head = nn.Linear(embed_dim, n_output)
        # Starts at sigma(-2.197) ~ 0.1. beta must be learned: fixing it
        # collapses wherever W*c cannot reach every class (section 12-6).
        self.dev_beta_raw = nn.Parameter(torch.tensor([-2.197]))

    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _hard_pred(logits: torch.Tensor) -> torch.Tensor:
        """Map logits to a predicted class.

        With n_output=1 the task is binary and the sign decides the class, so
        argmax would always return 0. Callers do not use this for regression.
        """
        if logits.shape[-1] == 1:
            return (logits.squeeze(-1) > 0).long()
        return logits.argmax(dim=-1)

    def forward(
        self,
        X: torch.Tensor,                            # (B, F)
        labels: Optional[torch.Tensor] = None,      # (B,) for the memory update
        sample_ids: Optional[torch.Tensor] = None,  # (B,) X_train row numbers
        # ⚠ sample_ids is the only basis for the MemoryBank / FeatureStore
        #   slot correspondence and for self-exclusion. Batches are shuffled,
        #   so storage order is not train row order.
        return_explanations: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # 1. Embed
        query_emb = self.embedder(X)                                   # (B, D)

        # 2. Route: one assignment fixes both the prediction baseline and G(a)
        context_emb, hard_assignment, routing_probs, _, _, top1_confidence = \
            self.prototype_layer(query_emb)

        # 3. Retrieve, within the assigned group only. Outside the prediction path.
        _neighbor_mask = None
        if self.memory.filled.item() >= self.k:
            nk, neighbour_labels, topk_idx = self.memory.retrieve(
                query_emb, self.k,
                hard_assignment=hard_assignment,
                exclude_ids=(sample_ids if self.exclude_self_retrieval else None),
            )
            with torch.no_grad():
                # A slot retrieval failed to fill has topk_idx = 0, which is
                # indistinguishable from "neighbour 0" outside the model. That
                # is the only reason forward emits this.
                _neighbor_mask = nk.detach().norm(dim=-1) > 1e-8        # (B, k)
        else:
            # Warmup: memory holds fewer than k entries.
            topk_idx = torch.zeros(X.shape[0], self.k, dtype=torch.long,
                                   device=X.device)
            _neighbor_mask = torch.zeros(X.shape[0], self.k, dtype=torch.bool,
                                         device=X.device)

        # 4. Predict
        _beta  = torch.sigmoid(self.dev_beta_raw)
        _dev   = query_emb - context_emb
        _dev_n = F.normalize(_dev, dim=-1)
        _h     = context_emb + _beta * _dev_n
        logits = self.dev_head(_h)

        # 5. Decomposition diagnostics. The logit decomposition is the ground
        #    truth: what the prototype said and how much the correction moved
        #    it is observed directly, on every forward, without retraining.
        with torch.no_grad():
            _lg_c = self.dev_head(context_emb)          # W*c + b, prototype only
            _diag = {
                "dev_beta": float(_beta),
                "dev_residual_ratio": float(
                    (_beta * _dev_n).norm(dim=-1).mean()
                    / context_emb.norm(dim=-1).mean().clamp_min(1e-8)),
                "dev_raw_norm": float(_dev.norm(dim=-1).mean()),
                # W*c can produce at most P distinct predictions. When that
                # count falls below the number of classes an accuracy ceiling
                # appears, and only then does the beta term do any work
                # (section 12-3).
                "dev_unique_logits": float(
                    len(torch.unique(logits.round(decimals=4), dim=0))),
                "dev_unique_logits_c": float(
                    len(torch.unique(_lg_c.round(decimals=4), dim=0))),
                "dev_changed_rate": float(
                    (self._hard_pred(logits)
                     != self._hard_pred(_lg_c)).float().mean()),
            }
            if labels is not None and self.tasktype != "regression":
                _diag["dev_acc_c"] = float(
                    (self._hard_pred(_lg_c) == labels.long()).float().mean())
                _diag["dev_acc_full"] = float(
                    (self._hard_pred(logits) == labels.long()).float().mean())
                _diag["dev_delta_acc"] = _diag["dev_acc_full"] - _diag["dev_acc_c"]

        # 6. Update memory (training only)
        if self.training and labels is not None:
            self.memory.update(query_emb.detach(), labels.float(), sample_ids)
            if self._feature_store is not None:
                # ⚠ Must be updated in the same batch as MemoryBank to keep
                #   the slot indices 1:1 (asserted in refresh_memory_keys).
                self._feature_store.update(X, sample_ids)

        if self.training and self.prototype_layer.use_ema_codebook:
            self.prototype_layer.ema_update(query_emb.detach(), hard_assignment)

        # 7. Auxiliary losses
        # ⚠ All zero under EMA: commitment was removed (section 10), the EMA
        #   replaces the codebook update, and diversity has nowhere to send a
        #   gradient because centroid_emb has requires_grad=False. The only
        #   training signal is cross-entropy. The key remains for caller
        #   compatibility.
        aux_loss = torch.zeros((), device=X.device, dtype=logits.dtype)

        out = {
            "logits":             logits,
            "aux_loss":           aux_loss,
            "routing":            routing_probs,
            "centroid_id":        hard_assignment,
            "hard_group":         hard_assignment,
            "routing_confidence": top1_confidence,
            "topk_idx":           topk_idx,
            "neighbor_mask":      _neighbor_mask,
            "query_emb":          query_emb,
            "context_emb":        context_emb,
            "dev_diag":           _diag,
        }

        if return_explanations:
            with torch.no_grad():
                q_norm  = F.normalize(query_emb.detach(), dim=-1)          # (B, D)
                c_norm  = F.normalize(
                    self.prototype_layer.centroid_emb.detach(), dim=-1)    # (P, D)
                cos_sim = q_norm @ c_norm.T                                # (B, P)
                soft_probs = F.softmax(
                    cos_sim * self.prototype_layer.routing_scale, dim=-1)

            proto_exp = self.prototype_layer.explain_routing(
                hard_assignment, soft_probs, cos_sim=cos_sim)

            with torch.no_grad():
                _beta_scalar = float(torch.sigmoid(self.dev_beta_raw.detach()).mean())
                _query_norm_ps = query_emb.detach().norm(dim=-1).cpu().numpy()

            out["explanations"] = [
                {
                    "prototype": proto_exp[b],
                    # ⚠ Neighbour similarities, labels, raw features and the
                    #   logit decomposition are not built here. diagnostics.py
                    #   reconstructs them externally from topk_idx, memory and
                    #   dev_head, so that enabling explanations cannot change
                    #   the prediction (section 2-7).
                    "retrieval_signal": {
                        "query_norm": float(_query_norm_ps[b]),
                        "beta": _beta_scalar,
                    },
                }
                for b in range(X.shape[0])
            ]

        return out

    # ─────────────────────────────────────────────────────────
    @property
    def feature_store(self) -> Optional[FeatureStore]:
        return self._feature_store

    @torch.no_grad()
    def refresh_memory_keys(self, batch_size: int = 1024) -> Optional[Dict[str, float]]:
        """Called once after training, right after best_state and
        feature_store are restored.

        The raw features in feature_store are pushed through the embedder
        again with the now-frozen weights, overwriting memory.keys and
        _keys_norm. memory.keys[i] then becomes a deterministic function of
        the raw features under the current weights, rather than a one-off
        snapshot taken with whatever dropout mask applied during training.

        Upstream TabR follows the same principle: it never stores candidate
        embeddings permanently, re-encoding them each epoch and recomputing
        from scratch after eval() at inference. Training noise must not leak
        into inference or explanation.

        ⚠ Deliberately not called during training. Doing it every step would
          cost E x N_train extra forward passes; this way it costs N_train
          only when a new best appears.
        """
        if self._feature_store is None:
            return None

        n_mem  = int(self.memory.filled.item())
        n_feat = self._feature_store._filled
        assert n_mem == n_feat, (
            f"refresh_memory_keys(): memory.filled({n_mem}) != "
            f"feature_store._filled({n_feat}): memory and feature_store may "
            f"have been restored to different points. Check that the "
            f"best_state and feature_store restore completed before this call."
        )
        if n_mem == 0:
            return {"n_refreshed": 0}

        was_training = self.training
        self.eval()
        device = self.memory.keys.device
        for start in range(0, n_mem, batch_size):
            end = min(start + batch_size, n_mem)
            raw   = self._feature_store._store[start:end].to(device)
            clean = self.embedder(raw)
            self.memory.keys[start:end]       = clean
            self.memory._keys_norm[start:end] = F.normalize(clean, dim=-1)
        if was_training:
            self.train()
        return {"n_refreshed": n_mem}

    def anneal(self, factor: float = 0.95) -> None:
        self.prototype_layer.anneal(factor)

    def summary(self, n_train: Optional[int] = None) -> str:
        total = sum(p.numel() for p in self.parameters())
        beta  = float(torch.sigmoid(self.dev_beta_raw.detach()).mean())
        lines = [
            "=" * 48, "TabERA", "=" * 48,
            f"  Parameters     : {total:,}",
            f"  Embed dim      : {self.embed_dim}",
            f"  Prototypes     : {self.prototype_layer.P}",
            f"  Retrieval k    : {self.k}  (explanation only)",
            f"  Prediction     : h = c + b*normalize(q-c),  z = W*h   (W shared)",
            f"  beta           : {beta:.4f}  (learned)",
            f"  Prototype mem  : EMA (decay={self.prototype_layer.ema_decay})"
            if self.prototype_layer.use_ema_codebook else
            f"  Prototype mem  : gradient codebook",
            f"  Training signal: CE only",
        ]
        if n_train is not None and self.prototype_layer.P > 0:
            avg_group_size = n_train / self.prototype_layer.P
            lines.append(
                f"  Avg group size : {avg_group_size:.1f}  "
                f"(N_train={n_train:,} / P={self.prototype_layer.P})")
            if self.k / avg_group_size > 1.0:
                # A group close to k makes within-group retrieval meaningless.
                lines.append(
                    f"  !  k({self.k}) > mean group size ({avg_group_size:.1f}): "
                    f"cross-group fallback may fire constantly")
        lines.append(self.prototype_layer.centroid_summary(top_n=3))
        lines.append("=" * 48)
        return "\n".join(lines)