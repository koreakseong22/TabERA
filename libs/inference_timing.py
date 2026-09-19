"""Inference-latency protocol shared by the TabERA and TabR runners.

Standalone on purpose: no imports from either repository, so the multitab
fork can load this file by path and both models are timed by the *same*
code. Any change here changes both sides.

Protocol (what a number in the JSON means)
  * A "call" is one forward pass over a pre-materialised batch that already
    sits on the device. Data loading, host-device copies of the inputs,
    model loading and index / memory construction are all outside the timer.
  * CUDA is asynchronous, so the timer synchronises before and after each
    call. Without that the GPU work would be attributed to the next call.
  * ``warmup`` untimed calls first (kernel compilation, allocator growth,
    faiss lazy init), then ``repeats`` timed calls on distinct batches.
  * Statistics are over the per-call latencies; ``median`` is the headline,
    ``p10`` / ``p90`` show the spread. Throughput = batch / median.
  * batch_size 1 is the online / interactive setting; larger batches show
    the batched-throughput setting. If the split has fewer rows than the
    batch, the batch is the whole split and ``batch_rows`` records that.
"""
import math
import platform
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

import torch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batches(X: torch.Tensor, batch_size: int, count: int, seed: int = 0,
                 tile: bool = False) -> List[torch.Tensor]:
    """``count`` batches of ``batch_size`` rows drawn from X in a fixed random
    row order, materialised up front so that indexing is never inside the
    timed region.

    ``tile`` decides what happens when the split has fewer rows than
    ``batch_size``. False (the default) shrinks the batch to the split, so a
    small dataset is timed on a smaller batch -- the per-dataset ratio between
    two models stays valid, but samples/s is NOT comparable across datasets.
    True repeats rows to reach ``batch_size`` exactly, which makes a
    fixed-batch throughput comparison across datasets meaningful; the work per
    call is identical whether or not rows repeat. Callers record which was
    used, so a result is self-describing.
    """
    n = X.shape[0]
    rows = batch_size if tile else min(batch_size, n)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    batches = []
    for r in range(count):
        start = (r * rows) % n
        idx = torch.arange(start, start + rows) % n
        batches.append(X[perm[idx].to(X.device)].contiguous())
    return batches


def _stats(ms: Sequence[float], rows: int, tiled: bool = False) -> Dict[str, float]:
    s = sorted(ms)
    q = lambda p: s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]
    med = statistics.median(s)
    return {
        "n_calls": len(s),
        "batch_rows": rows,
        # True when rows were repeated to reach the requested batch size
        # because the split was smaller. Only then is samples/s comparable
        # across datasets.
        "tiled": bool(tiled),
        "ms_median": med,
        "ms_mean": statistics.fmean(s),
        "ms_p10": q(0.10),
        "ms_p90": q(0.90),
        "ms_min": s[0],
        "ms_max": s[-1],
        "ms_per_sample_median": med / rows,
        "samples_per_s": rows / (med / 1000.0),
    }


@torch.no_grad()
def time_batches(fn: Callable[[torch.Tensor], object], X: torch.Tensor, batch_size: int,
                 repeats: int = 100, warmup: int = 10, seed: int = 0,
                 tile: bool = False) -> Dict[str, float]:
    """Per-call latency of ``fn`` on batches of ``batch_size`` rows."""
    device = X.device
    batches = make_batches(X, batch_size, repeats + warmup, seed=seed, tile=tile)
    for xb in batches[:warmup]:
        fn(xb)
    _sync(device)
    ms = []
    for xb in batches[warmup:]:
        _sync(device)
        t0 = time.perf_counter()
        fn(xb)
        _sync(device)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return _stats(ms, batches[0].shape[0], tiled=tile and batch_size > X.shape[0])


@torch.no_grad()
def time_full_pass(fn: Callable[[torch.Tensor], object], X: torch.Tensor, batch_size: int,
                   repeats: int = 20, warmup: int = 3) -> Dict[str, float]:
    """Latency of predicting the whole split in consecutive ``batch_size``
    chunks. ``fn`` receives one chunk at a time; the chunks are sliced once
    outside the timer."""
    device = X.device
    chunks = [X[s:s + batch_size].contiguous() for s in range(0, X.shape[0], batch_size)]

    def one_pass():
        for xb in chunks:
            fn(xb)

    for _ in range(warmup):
        one_pass()
    _sync(device)
    ms = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        one_pass()
        _sync(device)
        ms.append((time.perf_counter() - t0) * 1000.0)
    out = _stats(ms, X.shape[0])
    out["n_chunks"] = len(chunks)
    out["chunk_size"] = batch_size
    return out


@torch.no_grad()
def peak_memory_mb(fn: Callable[[torch.Tensor], object], xb: torch.Tensor,
                   device: torch.device) -> Optional[float]:
    """Peak allocated CUDA memory during one call, on top of what is already
    resident (weights, memory bank / candidate index). None on CPU."""
    if device.type != "cuda":
        return None
    _sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    fn(xb)
    _sync(device)
    return (torch.cuda.max_memory_allocated(device) - base) / 2**20


def resident_memory_mb(device: torch.device) -> Optional[float]:
    """Currently allocated CUDA memory: weights plus whatever the model keeps
    resident for inference (TabR's candidate keys and index, TabERA's
    prototypes and memory bank)."""
    if device.type != "cuda":
        return None
    _sync(device)
    return torch.cuda.memory_allocated(device) / 2**20


def environment(device: torch.device) -> Dict[str, object]:
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
        "node": platform.node(),
        "device": str(device),
        "threads": torch.get_num_threads(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info


def run_protocol(fns: Dict[str, Callable[[torch.Tensor], object]], X: torch.Tensor,
                 batch_sizes: Sequence[int], repeats: int, warmup: int,
                 full_pass_batch: Optional[int] = None, full_pass_repeats: int = 20,
                 seed: int = 0, tile: bool = False) -> Dict[str, object]:
    """Time every mode in ``fns`` under the same batches. Returns
    {mode: {"batch=<b>": stats, "full_pass": stats, "peak_call_mb": ...}}."""
    device = X.device
    out: Dict[str, object] = {}
    for name, fn in fns.items():
        rec: Dict[str, object] = {}
        for b in batch_sizes:
            rec[f"batch={b}"] = time_batches(fn, X, b, repeats=repeats, warmup=warmup,
                                             seed=seed, tile=tile)
        if full_pass_batch:
            rec["full_pass"] = time_full_pass(fn, X, full_pass_batch, repeats=full_pass_repeats)
        probe = make_batches(X, max(batch_sizes), 1, seed=seed)[0]
        rec["peak_call_mb"] = peak_memory_mb(fn, probe, device)
        out[name] = rec
    return out
