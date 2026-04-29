# %%
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import time
from tkinter import W
from tracemalloc import start

import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Canonical ROI list (37 bilateral regions, Allen Mouse Brain Atlas)
# ---------------------------------------------------------------------------
ROI_NAMES = [
    "Both_AI", "Both_ORB", "Both_ILA", "Both_PL", "Both_ACA",
    "Both_RSP", "Both_VIS", "Both_PTLp", "Both_TEa", "Both_MOp",
    "Both_MOs", "Both_SSp", "Both_SSs", "Both_AUD", "Both_GU_VISC",
    "Both_PERI", "Both_ENT", "Both_ECT", "Both_dhc", "Both_vhc",
    "Both_SUB", "Both_EP", "Both_CLA", "Both_PIR", "Both_PALd",
    "Both_PALv", "Both_ACB", "Both_CP", "Both_LSX", "Both_AAA_CEA_MEA",
    "Both_HY", "Both_PO_PF", "Both_VPL_VPM", "Both_VTA", "Both_MBmot",
    "Both_MBsen", "Both_MBsta",
]

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so both packages are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path("__file__").resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("REPO_ROOT:", REPO_ROOT)
print("Python:", sys.version)
# %%
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
WINDOW = 25  # sliding window size (frames)
# LAG        = WINDOW      # lag between windows (frames)
LAG = 1  # lag between windows (frames)
VSTEP = 1  # speed step
SEED = 42
THRESHOLD = 1e-6  # discrepancy flag threshold

TAU_RANGE = (0,)
# TAU_RANGE = (-2,-1,0,1,2)

rng = np.random.default_rng(SEED)


# %%
# ===========================================================================
# Functions to compute dFC stream
# ===========================================================================


# @njit(fastmath=True)
def fast_corrcoef(ts):
    """
    Numba-accelerated Pearson correlation matrix using z-score and dot product.
    ts: np.ndarray (timepoints, features)
    """
    n_samples, n_features = ts.shape
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0, ddof=1)
    # Avoid division by zero for constant columns
    std[std == 0] = 1.0
    z = (ts - mean) / std
    return np.dot(z.T, z) / (n_samples - 1)


def ts2dfc_stream(ts, window_size, lag=None, format_data="2D", method="pearson"):
    """
    Compute dynamic functional connectivity (DFC) stream using a sliding window approach.

    Parameters:
        ts (np.ndarray): Time series data (timepoints x regions).
        window_size (int): Size of the sliding window.
        lag (int): Step size between windows (default = window_size).
        format_data (str): '2D' for vectorized FC, '3D' for FC matrices.
        method (str): Correlation method (currently only 'pearson').

    Returns:
        np.ndarray: DFC stream, either in 2D (n_pairs x frames) or 3D (n_regions x n_regions x frames).
    """
    t_total, n = ts.shape
    lag = lag or window_size
    frames = (t_total - window_size) // lag + 1
    n_pairs = n * (n - 1) // 2

    # Preallocate DFC stream
    dfc_stream = None
    tril_idx = None

    if format_data == "2D":
        dfc_stream = np.empty((n_pairs, frames))
        tril_idx = np.tril_indices(n, k=-1)  # Precompute once
    elif format_data == "3D":
        dfc_stream = np.empty((n, n, frames))
    else:
        raise ValueError(f"Unsupported format_data '{format_data}'. Use '2D' or '3D'")

    for k in range(frames):
        wstart = k * lag
        wstop = wstart + window_size
        window = ts[wstart:wstop, :]
        fc = fast_corrcoef(window)

        if format_data == "2D":
            dfc_stream[:, k] = fc[tril_idx]
        else:
            dfc_stream[:, :, k] = fc

    return dfc_stream


# %%
# ===========================================================================
# Functions to compute speed dFC
# ===========================================================================


def pearson_speed_vectorized(fc1: np.ndarray, fc2: np.ndarray) -> np.ndarray:
    """Vectorized Pearson-based speed via einsum.

    Parameters
    ----------
    fc1, fc2 : np.ndarray, shape (n_pairs, n_frames)

    Returns
    -------
    speeds : np.ndarray, shape (n_frames,)
        1 - corr(fc1[:, t], fc2[:, t]) for each t.
    """
    fc1c = fc1 - fc1.mean(axis=0, keepdims=True)
    fc2c = fc2 - fc2.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", fc1c, fc2c)
    ss1 = np.einsum("ij,ij->j", fc1c, fc1c)
    ss2 = np.einsum("ij,ij->j", fc2c, fc2c)
    denom = np.sqrt(ss1 * ss2)
    corr = np.where(denom > np.finfo(float).eps, num / denom, 0.0)
    return 1.0 - corr


def dfc_speed_split(
    dfc_stream: np.ndarray,
    *,
    window_size: int,
    lag: int = 1,
    vstep: int = 1,
    tau_range: Sequence[int] | np.ndarray = (0,),
    method: str = "pearson",
    return_fc2: bool = False,
) -> np.ndarray:
    """Compute dFC speed across multiple tau offsets (window oversampling).

    For each tau in tau_range and each FC1 frame t, speed is:

        speed[t, tau] = 1 - corr(FC[t], FC[t + time_window + tau])

    where:
        time_window = ceil(window_size / lag)   — minimum frame separation
                                                  guaranteeing no overlap between
                                                  the windows of FC1 and FC2.

    vstep controls how FC1 frames are sampled:
        vstep=1            → dense: every frame (FC1s overlap each other)
        vstep=time_window  → sparse: FC1s are fully non-overlapping

    In both cases the FC1→FC2 separation is always exactly (time_window + tau)
    frames — vstep does NOT affect that gap.

    tau=0   → FC1 and FC2 are adjacent (minimum non-overlap).
    tau>0   → FC2 is further in time (longer-scale oversampling).
    tau<0   → FC2 partially overlaps FC1 (controlled overlap regime).
              tau must satisfy tau > -time_window to avoid full collapse (FC1==FC2).

    Parameters
    ----------
    dfc_stream : np.ndarray
        2D (n_pairs, n_frames) vectorized dFC stream, lower triangle, as produced
        by ts2dfc_stream(..., format_data='2D').
        3D (n_rois, n_rois, n_frames) is also accepted; the lower triangle is
        extracted automatically, consistent with ts2dfc_stream.
    window_size : int
        Size of the sliding window used to build the dFC stream (in TRs).
    lag : int
        Lag between consecutive windows used to build the dFC stream (in TRs).
        Default is 1.
    vstep : int
        Step between consecutive FC1 frames (in stream frames).
        Recommended values:
            1            → dense speed series (default)
            time_window  → fully non-overlapping FC1s
        Default is 1.
    tau_range : sequence of int
        Frame offsets applied on top of time_window. May include negative values
        for partial overlap. Must satisfy:
            tau > -ceil(window_size / lag)   for all tau in tau_range.
    method : {"pearson"}
        Similarity metric. Currently only "pearson" is supported.
    return_fc2 : bool
        If True, return the FC2 index array instead of speed values (useful
        for debugging index construction).

    Returns
    -------
    np.ndarray
        Shape (len(tau_range), T_eff) — one row of speed values per tau.
        T_eff = len(np.arange(0, indices_max, vstep)) - 1
        If return_fc2 is True, returns a 1D int array instead.

    Raises
    ------
    ValueError
        If any tau causes full collapse (FC1 == FC2), or if there are not enough
        frames for the requested parameters.
    TypeError
        If dfc_stream is not a numpy array.
    """
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    if dfc_stream.ndim not in (2, 3):
        raise ValueError(
            "dfc_stream must be 2D (n_pairs, n_frames) or 3D (n_rois, n_rois, n_frames)"
        )
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if vstep < 1:
        raise ValueError("vstep must be >= 1")
    if method != "pearson":
        raise ValueError(
            f"Unsupported method '{method}'. Currently only 'pearson' is supported."
        )

    if dfc_stream.ndim == 3:
        n_rois = dfc_stream.shape[0]
        tril_idx = np.tril_indices(n_rois, k=-1)
        fc_stream = dfc_stream[tril_idx[0], tril_idx[1], :]
    else:
        fc_stream = dfc_stream

    n_frames = fc_stream.shape[1]

    time_window = math.ceil(window_size / lag)

    tau_arr = np.atleast_1d(np.asarray(tau_range, dtype=int))
    if tau_arr.size == 0:
        tau_arr = np.array([0], dtype=int)

    invalid = tau_arr[tau_arr <= -time_window]
    if invalid.size > 0:
        raise ValueError(
            f"tau values {invalid.tolist()} would cause FC1 == FC2 (full collapse). "
            f"All tau must be > {-time_window} (i.e., >= {-time_window + 1})."
        )

    tau_max = int(tau_arr.max())
    indices_max = n_frames - (time_window + tau_max)

    if indices_max <= vstep:
        raise ValueError(
            f"Not enough frames for the requested parameters "
            f"(n_frames={n_frames}, window_size={window_size}, lag={lag}, "
            f"vstep={vstep}, tau_max={tau_max}). "
            f"Need at least {time_window + tau_max + vstep + 1} frames."
        )

    head = np.arange(0, indices_max - vstep, vstep, dtype=int)

    if head.size < 1:
        raise ValueError("Not enough FC1 frames after sub-sampling. Reduce vstep.")

    fc1_idx = np.tile(head, (tau_arr.size, 1))
    fc2_idx = np.vstack([head + time_window + int(tau) for tau in tau_arr])

    if return_fc2:
        return fc2_idx.flatten().astype(int)

    fc1_mat = fc_stream[:, fc1_idx.flatten()]
    fc2_mat = fc_stream[:, fc2_idx.flatten()]

    speeds_flat = pearson_speed_vectorized(fc1_mat, fc2_mat)
    speeds = np.clip(speeds_flat, 0.0, 2.0).reshape(tau_arr.size, -1)

    return speeds


def load_one_subject_mat(file_tc):
    mat = loadmat(file_tc)
    ts = mat["tc"]
    # roi_name = mat["ROI_name"]
    if ts.shape[0] < ts.shape[1]:
        ts = ts.T

    roi_npy = Path(__file__).parent.parent / "data" / "dataset" / "roi_names.npy"
    roi_list = list(np.load(roi_npy))
    return ts, roi_list


def pearson_speed_nodal_vectorized(fc1: np.ndarray, fc2: np.ndarray) -> np.ndarray:
    fc1c = fc1 - fc1.mean(axis=0, keepdims=True)
    fc2c = fc2 - fc2.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", fc1c, fc2c)
    ss1 = np.einsum("ij,ij->j", fc1c, fc1c)
    ss2 = np.einsum("ij,ij->j", fc2c, fc2c)
    denom = np.sqrt(ss1 * ss2)
    corr = np.where(denom > np.finfo(float).eps, num / denom, 0.0)
    return 1.0 - corr


def dfc_speed_nodal(
    dfc_stream: np.ndarray,
    *,
    window_size: int,
    lag: int = 1,
    vstep: int = 1,
    tau_range: Sequence[int] | np.ndarray = (0,),
    method: str = "pearson",
):
    if not isinstance(dfc_stream, np.ndarray):
        raise TypeError("dfc_stream must be a numpy array")
    if dfc_stream.ndim != 3:
        raise ValueError("dfc_stream must be 3D (n_rois, n_rois, n_frames)")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if vstep < 1:
        raise ValueError("vstep must be >= 1")
    if method != "pearson":
        raise ValueError(
            f"Unsupported method '{method}'. Currently only 'pearson' is supported."
        )

    n_rois, _, n_frames = dfc_stream.shape
    time_window = math.ceil(window_size / lag)

    tau_arr = np.atleast_1d(np.asarray(tau_range, dtype=int))
    if tau_arr.size == 0:
        tau_arr = np.array([0], dtype=int)

    invalid = tau_arr[tau_arr <= -time_window]
    if invalid.size > 0:
        raise ValueError(
            f"tau values {invalid.tolist()} would cause FC1 == FC2 (full collapse). "
            f"All tau must be > {-time_window} (i.e., >= {-time_window + 1})."
        )

    tau_max = int(tau_arr.max())
    indices_max = n_frames - (time_window + tau_max)

    if indices_max <= vstep:
        raise ValueError(
            f"Not enough frames for the requested parameters "
            f"(n_frames={n_frames}, window_size={window_size}, lag={lag}, "
            f"vstep={vstep}, tau_max={tau_max}). "
            f"Need at least {time_window + tau_max + vstep + 1} frames."
        )

    head = np.arange(0, indices_max - vstep, vstep, dtype=int)

    if head.size < 1:
        raise ValueError("Not enough FC1 frames after sub-sampling. Reduce vstep.")

    fc1_idx = np.tile(head, (tau_arr.size, 1))
    fc2_idx = np.vstack([head + time_window + int(tau) for tau in tau_arr])

    nodal_speeds = np.empty((n_rois, tau_arr.size, head.size), dtype=float)

    for roi in range(n_rois):
        mask = np.ones(n_rois, dtype=bool)
        mask[roi] = False

        roi_stream = dfc_stream[roi, mask, :]

        fc1_mat = np.empty((n_rois - 1, tau_arr.size * head.size), dtype=float)
        fc2_mat = np.empty((n_rois - 1, tau_arr.size * head.size), dtype=float)

        for itau in range(tau_arr.size):
            fc1_mat[:, itau * head.size:(itau + 1) * head.size] = roi_stream[:, fc1_idx[itau]]
            fc2_mat[:, itau * head.size:(itau + 1) * head.size] = roi_stream[:, fc2_idx[itau]]

        speeds_flat = pearson_speed_nodal_vectorized(fc1_mat, fc2_mat)
        nodal_speeds[roi] = np.clip(speeds_flat.reshape(tau_arr.size, head.size), 0.0, 2.0)

    return nodal_speeds


def compute_subject_nodal_speed(file_tc, window_size, lag=1, vstep=1, tau_range=(0,), method="pearson"):
    ts, roi_list = load_one_subject_mat(file_tc)

    dfc_stream_3d = ts2dfc_stream(
        ts,
        window_size=window_size,
        lag=lag,
        format_data="3D",
        method=method,
    )

    nodal_speeds = dfc_speed_nodal(
        dfc_stream_3d,
        window_size=window_size,
        lag=lag,
        vstep=vstep,
        tau_range=tau_range,
        method=method,
    )
    
    percentile_grid = np.linspace(0, 100, 100)

    nodal_speed_percentiles = np.percentile(
        nodal_speeds,
        percentile_grid,
        axis=(1, 2)
    ).T

    return nodal_speed_percentiles, percentile_grid, roi_list, nodal_speeds, ts.shape, dfc_stream_3d.shape    

def compute_subject_global_speed(file_tc, window_size, lag=1, vstep=1, tau_range=(0,), method="pearson"):
    ts, roi_list = load_one_subject_mat(file_tc)

    dfc_stream_2d = ts2dfc_stream(
        ts,
        window_size=window_size,
        lag=lag,
        format_data="2D",
        method=method,
    )

    speed_split = dfc_speed_split(
        dfc_stream_2d,
        window_size=window_size,
        lag=lag,
        vstep=vstep,
        tau_range=tau_range,
        method=method,
        return_fc2=False,
    )

    speed_mean = speed_split.mean()

    percentiles = np.percentile(speed_split.flatten(), np.arange(101))

    return speed_mean, percentiles, speed_split.flatten()
# %%
