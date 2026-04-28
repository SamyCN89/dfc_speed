"""
dfc_speed_compute.py
--------------------
Compute dynamic functional connectivity (dFC) speed from a sequence of
sliding-window correlation matrices (or any dFC estimator output).

Speed is defined as the frame-to-frame Euclidean (or cosine) distance
between consecutive FC matrices in the upper-triangle vectorised space.

Usage
-----
    from dfc_speed.dfc_speed_compute import compute_dfc_speed
    speed = compute_dfc_speed(fc_timeseries)   # (T-1,) array

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.spatial.distance import euclidean, cosine


# ── Utility ──────────────────────────────────────────────────────────────────

def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle values (excluding diagonal) as a 1-D vector."""
    idx = np.triu_indices(matrix.shape[-1], k=1)
    return matrix[..., idx[0], idx[1]]


# ── Core ─────────────────────────────────────────────────────────────────────

def compute_dfc_speed(
    fc_timeseries: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute dFC speed from a sequence of FC matrices.

    Parameters
    ----------
    fc_timeseries : np.ndarray, shape (T, N, N)
        Sequence of T symmetric FC matrices of size N×N.
    metric : {"euclidean", "cosine"}
        Distance metric between consecutive frames.

    Returns
    -------
    speed : np.ndarray, shape (T-1,)
        Frame-to-frame distance (speed) values.
    """
    if fc_timeseries.ndim != 3:
        raise ValueError(
            f"Expected 3-D array (T, N, N), got shape {fc_timeseries.shape}"
        )

    T = fc_timeseries.shape[0]
    vecs = upper_triangle(fc_timeseries)        # (T, n_pairs)

    dist_fn = {"euclidean": euclidean, "cosine": cosine}[metric]

    speed = np.array(
        [dist_fn(vecs[t], vecs[t + 1]) for t in range(T - 1)],
        dtype=float,
    )
    return speed


def compute_dfc_speed_batch(
    fc_batch: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute dFC speed for a batch of subjects/sessions.

    Parameters
    ----------
    fc_batch : np.ndarray, shape (S, T, N, N)
        S subjects × T timepoints × N×N FC matrices.
    metric : {"euclidean", "cosine"}

    Returns
    -------
    speeds : np.ndarray, shape (S, T-1)
    """
    if fc_batch.ndim != 4:
        raise ValueError(
            f"Expected 4-D array (S, T, N, N), got shape {fc_batch.shape}"
        )
    return np.stack(
        [compute_dfc_speed(fc_batch[s], metric=metric)
         for s in range(fc_batch.shape[0])],
        axis=0,
    )


# ── Summary statistics ────────────────────────────────────────────────────────

def speed_summary(speed: np.ndarray) -> dict:
    """Return mean, std, median, and IQR of a speed vector."""
    return {
        "mean":   float(np.mean(speed)),
        "std":    float(np.std(speed)),
        "median": float(np.median(speed)),
        "iqr":    float(np.percentile(speed, 75) - np.percentile(speed, 25)),
    }
