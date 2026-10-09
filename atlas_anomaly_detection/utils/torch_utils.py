"""Shared PyTorch scaffolding for the deep unsupervised detectors.

Keeps the deep scripts (autoencoder, LSTM-AE, Transformer, VAE, USAD,
OmniAnomaly, GAN) short and *identical* in their data contract:

* :func:`prepare_sequences` -- load -> feature-engineer -> select a compact
  sequence feature set -> build sliding windows (train windows are strided to
  bound CPU cost; every window is scored) -> return a tidy bundle.
* :func:`fit_reconstruction` -- a generic MSE reconstruction training loop for
  the autoencoder-family models.
* :func:`batch_iter`, :func:`set_seed`, :data:`DEVICE` -- small utilities the
  models with custom losses (VAE/USAD/GAN/OmniAnomaly) build on.

Everything runs on CPU by default (this is a prototype); set the ``TORCH_DEVICE``
env var to ``cuda`` to use a GPU on the full archive.

Cost notes
----------
* The seven deep detectors all window the *same* frame, so the dense stride-1
  tensor is built once per (window, seed) and memoised; each model only slices
  its own strided training subset out of it.
* The best CPU thread count differs per architecture, so it is set per model via
  :func:`torch_threads` rather than globally -- the transformer is 1.43x faster
  on 2 threads while the LSTM is 1.44x slower there.  ``TORCH_NUM_THREADS``
  overrides the baseline.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional

import numpy as np

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

DEVICE = os.environ.get("TORCH_DEVICE",
                        "cuda" if (_HAS_TORCH and torch.cuda.is_available())
                        else "cpu")


#: Thread count for attention-style models -- see :func:`torch_threads`.
FEW_THREADS: int = max(1, (os.cpu_count() or 4) // 4)

if _HAS_TORCH and DEVICE == "cpu" and "TORCH_NUM_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["TORCH_NUM_THREADS"]))


@contextmanager
def torch_threads(n: int):
    """Temporarily run with ``n`` intra-op threads, restoring the previous count.

    The right thread count is *architecture-dependent*, not global, so this is
    applied per model rather than once at import.  Measured on an 8-core CPU,
    one epoch over 8k windows, interleaved and best-of-3:

    ==============  =========  =========  =========
    model           2 threads  4 threads  8 threads
    ==============  =========  =========  =========
    transformer         4.03s      5.78s      6.73s
    LSTM-AE             1.15s      0.80s      0.74s
    dense AE            0.18s      0.15s      0.17s
    ==============  =========  =========  =========

    The transformer issues many small attention matmuls, so per-op thread
    barriers dominate and *fewer* threads win by 1.43x; the LSTM's per-timestep
    gemms are large enough to keep more cores busy.  Capping threads globally
    would speed the transformer up and slow the LSTM down by about as much.

    A ``TORCH_NUM_THREADS`` env var (applied at import) still wins overall,
    since this only shifts the count for the duration of the block.
    """
    if not _HAS_TORCH or DEVICE != "cpu":
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, int(n)))
    try:
        yield
    finally:
        torch.set_num_threads(previous)

#: Compact, informative feature set for the sequence models (level + dynamics +
#: health). Kept small to bound the window tensor size on CPU. Note: the raw
#: ``alarm`` bit is deliberately NOT here -- it is the detector's own decision
#: (circular as an input) and is used only as a baseline. ``sensor_health`` is
#: computed partly from ``alarm``; it is retained as a soft health summary but
#: could be dropped for a fully alarm-free input if desired.
SEQUENCE_FEATURES: List[str] = [
    "temperature", "pressure", "voltage", "current", "error_rate",
    "quality_flag", "heartbeat_delta",
    "temperature_roc", "current_roc", "sensor_health", "temp_curr_corr",
]


def require_torch() -> None:
    """Raise a friendly error if PyTorch is not installed."""
    if not _HAS_TORCH:
        raise SystemExit("PyTorch not installed -- run `pip install torch`.")


def set_seed(seed: int = 42) -> None:
    """Seed numpy and torch for reproducibility."""
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)


@dataclass
class SequenceData:
    """Bundle returned by :func:`prepare_sequences`."""

    data: object                 # UnsupervisedData (per-row df / X / y)
    features: List[str]
    X_train: np.ndarray          # (n_train_windows, window, n_feat) float32
    X_all: np.ndarray            # (n_windows, window, n_feat) float32
    last_index: np.ndarray       # row index of each window's final step
    y_windows: np.ndarray        # label of each window's final step
    window: int

    @property
    def n_features(self) -> int:
        return self.X_all.shape[2]


#: (window, seed) -> the dense stride-1 window tensor and its scaled frame.
#: Shared by all seven deep detectors within a run, which each used to rebuild
#: it from the CSV.
_WINDOW_CACHE: dict = {}


def _dense_windows(window: int, seed: int):
    """Scaled frame + stride-1 windows for ``window``, built at most once."""
    key = (window, seed)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None:
        return cached

    from sklearn.preprocessing import StandardScaler
    from utils.pipeline import prepare_unsupervised
    from utils.windowing import make_windows

    data = prepare_unsupervised(engineered=True, scale=True)
    feats = [f for f in SEQUENCE_FEATURES if f in data.df.columns]

    # Standardize the chosen columns on the frame so windows are on the same
    # scale as the per-row matrix used elsewhere.
    df = data.df.copy()
    df[feats] = StandardScaler().fit_transform(df[feats].to_numpy(float))

    X_all, last_index, y_windows = make_windows(
        df, feats, window=window, stride=1, label_col="anomaly")
    cached = (data, feats, df, X_all.astype(np.float32, copy=False),
              last_index, y_windows)
    _WINDOW_CACHE[key] = cached
    return cached


def prepare_sequences(window: int = 30, train_stride: int = 3,
                      seed: int = 42) -> "SequenceData":
    """Build train/all sliding-window tensors from the unlabeled dataset.

    The stride-1 tensor is memoised across detectors; the strided training set
    is sliced out of it (identical to windowing the frame a second time, see
    :func:`utils.windowing.stride_subset`) rather than rebuilt.
    """
    require_torch()
    set_seed(seed)
    from utils.windowing import stride_subset

    data, feats, df, X_all, last_index, y_windows = _dense_windows(window, seed)
    if train_stride == 1:
        X_train = X_all
    else:
        X_train = X_all[stride_subset(df, window=window, stride=train_stride)]
    return SequenceData(data=data, features=feats, X_train=X_train,
                        X_all=X_all, last_index=last_index,
                        y_windows=y_windows, window=window)


def finalize_sequences(name: str, seq: "SequenceData", win_scores: np.ndarray,
                       train_time: float, predict_time: float) -> dict:
    """Scatter per-window scores onto rows and run the standard evaluation tail.

    Rows never covered by a window (the leading ``window-1`` steps of each
    element -- an unavoidable cold start) receive a clearly-normal fill value.
    """
    from utils.pipeline import finalize_unsupervised
    from utils.windowing import scores_to_rows

    fill = float(np.percentile(win_scores, 1)) if len(win_scores) else 0.0
    row_scores = scores_to_rows(len(seq.data.df), seq.last_index, win_scores,
                                fill=fill)
    return finalize_unsupervised(name, seq.data, row_scores,
                                 train_time, predict_time)


def batch_iter(*arrays: np.ndarray, batch: int = 256, shuffle: bool = True,
               seed: int = 0) -> Iterator[tuple]:
    """Yield aligned mini-batches (as torch tensors on ``DEVICE``).

    Each array is wrapped as a tensor once (zero-copy for contiguous float32
    numpy) and sliced per batch, so a training loop no longer pays a numpy
    fancy-index *and* a host->tensor conversion on every step.
    """
    require_torch()
    n = len(arrays[0])
    tensors = [torch.as_tensor(np.ascontiguousarray(a)) for a in arrays]
    if shuffle:
        idx = torch.from_numpy(
            np.random.default_rng(seed).permutation(n).astype(np.int64))
        for start in range(0, n, batch):
            sl = idx[start:start + batch]
            yield tuple(t.index_select(0, sl).to(DEVICE) for t in tensors)
    else:
        for start in range(0, n, batch):
            sl = slice(start, start + batch)
            yield tuple(t[sl].to(DEVICE) for t in tensors)


def fit_reconstruction(model: "nn.Module", X_train: np.ndarray,
                       epochs: int = 8, batch: int = 256, lr: float = 1e-3,
                       flatten: bool = False, verbose: bool = True) -> None:
    """Generic MSE reconstruction training loop for AE-family models.

    ``flatten`` reshapes each window to a vector first (for dense AEs).  The
    model's ``forward`` must return a tensor the same shape as its input.
    """
    require_torch()
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        total, nb = 0.0, 0
        for (xb,) in batch_iter(X_train, batch=batch, shuffle=True, seed=ep):
            if flatten:
                xb = xb.reshape(xb.shape[0], -1)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(xb); nb += len(xb)
        if verbose:
            print(f"    epoch {ep + 1:>2}/{epochs}  loss={total / nb:.5f}")


def reconstruction_scores(model: "nn.Module", X_all: np.ndarray,
                          batch: int = 512, flatten: bool = False
                          ) -> np.ndarray:
    """Per-window mean squared reconstruction error (higher == more anomalous)."""
    require_torch()
    model.to(DEVICE).eval()
    out = np.empty(len(X_all), dtype=np.float32)
    pos = 0
    with torch.inference_mode():
        for (xb,) in batch_iter(X_all, batch=batch, shuffle=False):
            inp = xb.reshape(xb.shape[0], -1) if flatten else xb
            rec = model(inp)
            err = ((rec - inp) ** 2).reshape(len(xb), -1).mean(dim=1)
            out[pos:pos + len(xb)] = err.cpu().numpy(); pos += len(xb)
    return out
