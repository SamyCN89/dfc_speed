#!/usr/bin/env python3
"""
Multiscale dFC speed — clean white figures for publication/sharing

Folder structure expected:
    data/tc_4M_veh_T21_55_seeds.mat
    data/tc_21F_veh_wt_55_seeds.mat
    fig/   (output folder, created automatically)

Figures produced:
    fig/fig1_speed_heatmap.png      — 2D heatmap of speed density vs window size (white bg)
    fig/fig2_band_distributions.png — KDE + histogram per timescale band, T21 vs WT
"""
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy import stats
from scipy.stats import gaussian_kde

os.makedirs("fig", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Core dFC functions
# ═══════════════════════════════════════════════════════════════════════════

def fast_corrcoef(ts):
    """Pearson correlation matrix via z-score dot product (timepoints x regions)."""
    mean = np.mean(ts, axis=0)
    std  = np.std(ts, axis=0, ddof=1)
    std[std == 0] = 1.0
    z = (ts - mean) / std
    return np.dot(z.T, z) / (ts.shape[0] - 1)


def ts2dfc_stream(ts, window_size, lag=1):
    """Sliding-window dFC stream -> 2D array (n_pairs x n_frames)."""
    t_total, n = ts.shape
    frames     = (t_total - window_size) // lag + 1
    n_pairs    = n * (n - 1) // 2
    dfc_stream = np.empty((n_pairs, frames))
    tril_idx   = np.tril_indices(n, k=-1)
    for k in range(frames):
        wstart = k * lag
        fc = fast_corrcoef(ts[wstart:wstart + window_size, :])
        dfc_stream[:, k] = fc[tril_idx]
    return dfc_stream


def pearson_speed_vectorized(fc1, fc2):
    """Vectorized 1 - Pearson(fc1[:,t], fc2[:,t]) for all t simultaneously."""
    fc1c  = fc1 - fc1.mean(axis=0, keepdims=True)
    fc2c  = fc2 - fc2.mean(axis=0, keepdims=True)
    num   = np.einsum("ij,ij->j", fc1c, fc2c)
    denom = np.sqrt(np.einsum("ij,ij->j", fc1c, fc1c) *
                    np.einsum("ij,ij->j", fc2c, fc2c))
    corr  = np.where(denom > np.finfo(float).eps, num / denom, 0.0)
    return 1.0 - corr


def dfc_speed_split(dfc_stream, *, window_size, lag=1, vstep=1, tau_range=(0,)):
    """
    Compute dFC speed (1 - Pearson correlation between successive FC frames).

    Parameters
    ----------
    dfc_stream  : 2D array (n_pairs, n_frames) from ts2dfc_stream
    window_size : sliding window size used to build dfc_stream (TRs)
    lag         : lag used to build dfc_stream (TRs); default 1
    vstep       : step between FC1 frames (1 = dense; time_window = non-overlapping)
    tau_range   : additional frame offsets on top of the minimum non-overlap gap

    Returns
    -------
    speeds : array (len(tau_range), T_eff), or None if not enough frames
    """
    n_frames    = dfc_stream.shape[1]
    time_window = math.ceil(window_size / lag)   # min frame separation for non-overlap
    tau_arr     = np.atleast_1d(np.asarray(tau_range, dtype=int))
    tau_max     = int(tau_arr.max())
    indices_max = n_frames - (time_window + tau_max)

    if indices_max <= vstep:
        return None

    head    = np.arange(0, indices_max - vstep, vstep, dtype=int)
    fc1_idx = np.tile(head, (tau_arr.size, 1))
    fc2_idx = np.vstack([head + time_window + int(tau) for tau in tau_arr])

    speeds = pearson_speed_vectorized(
        dfc_stream[:, fc1_idx.flatten()],
        dfc_stream[:, fc2_idx.flatten()],
    )
    return np.clip(speeds, 0.0, 2.0).reshape(tau_arr.size, -1)


# ═══════════════════════════════════════════════════════════════════════════
# Load data  <- update paths here if needed
# ═══════════════════════════════════════════════════════════════════════════

TR = 0.72   # repetition time (seconds)

groups = {
    "T21": loadmat("data/tc_4M_veh_T21_55_seeds.mat")["tc"],
    "WT":  loadmat("data/tc_21F_veh_wt_55_seeds.mat")["tc"],
}

# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_SIZES = np.arange(10, 85, 5)   # 10, 15, ..., 80 TRs

BANDS = {
    "Fast":         (10, 25),    # ~7-18 s
    "Intermediate": (30, 55),    # ~22-40 s
    "Slow":         (60, 80),    # ~43-58 s
}

# ═══════════════════════════════════════════════════════════════════════════
# Compute speed
# ═══════════════════════════════════════════════════════════════════════════

results = {}   # results[group][ws] = 1D speed array (tau=0)
for name, ts in groups.items():
    results[name] = {}
    for ws in WINDOW_SIZES:
        dfc = ts2dfc_stream(ts, window_size=int(ws), lag=1)
        spd = dfc_speed_split(dfc, window_size=int(ws), lag=1, vstep=1, tau_range=(0,))
        if spd is not None:
            results[name][ws] = spd[0]

def pool_band(rgroup, lo, hi):
    arrays = [v for ws, v in rgroup.items() if lo <= ws <= hi]
    return np.concatenate(arrays) if arrays else np.array([])

band_pools = {
    label: {name: pool_band(results[name], lo, hi) for name in groups}
    for label, (lo, hi) in BANDS.items()
}

# ═══════════════════════════════════════════════════════════════════════════
# Shared style
# ═══════════════════════════════════════════════════════════════════════════

C = {"T21": "#d62728", "WT": "#1f77b4"}
BAND_COLORS = {"Fast": "#e07b00", "Intermediate": "#2ca02c", "Slow": "#7b2d8b"}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.6,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "font.size":         11,
})

BINS_HIST = np.linspace(0.3, 1.4, 70)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Heatmap: speed density vs window size  (white background)
# ═══════════════════════════════════════════════════════════════════════════
#
# Each row = one window size; colour = probability density of v_dFC.
# Two panels side-by-side (T21 | WT) share the same colour scale.
# Dashed horizontal lines mark the timescale band boundaries.
# Band names annotated on the left margin.

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                         gridspec_kw={"right": 0.88})
fig.suptitle("dFC speed distribution across window sizes",
             fontsize=14, fontweight="bold")

BINS_HM = np.linspace(0.3, 1.4, 80)
bc_hm   = 0.5 * (BINS_HM[:-1] + BINS_HM[1:])

# Pre-compute histograms and find shared vmax
vmax_global = 0
hist_data   = {}
for name in groups:
    ws_list = sorted(results[name].keys())
    mat = np.zeros((len(ws_list), len(bc_hm)))
    for i, ws in enumerate(ws_list):
        h, _ = np.histogram(results[name][ws], bins=BINS_HM, density=True)
        mat[i] = h
    hist_data[name] = (ws_list, mat)
    vmax_global = max(vmax_global, mat.max())

for ax, (name, color) in zip(axes, C.items()):
    ws_list, mat = hist_data[name]
    im = ax.imshow(
        mat, aspect="auto", origin="lower", cmap="YlOrRd",
        vmin=0, vmax=vmax_global,
        extent=[BINS_HM[0], BINS_HM[-1], 0, len(ws_list)],
    )
    # Y-axis: window size labels (every other tick to avoid crowding)
    tick_pos    = np.arange(len(ws_list)) + 0.5
    tick_labels = [f"{int(ws)}" for ws in ws_list]
    ax.set_yticks(tick_pos[::2])
    ax.set_yticklabels(tick_labels[::2], fontsize=8)

    # Dashed lines at band boundaries
    for lo, hi in BANDS.values():
        for bound in (lo, hi):
            if bound in ws_list:
                idx = ws_list.index(bound)
                ax.axhline(idx + 0.5, color="steelblue", lw=0.9, ls="--", alpha=0.7)

    ax.set_title(f"Group: {name}", fontsize=12, fontweight="bold", color=color)
    ax.set_xlabel(r"$v_{dFC}$  (1 − Pearson corr)", fontsize=11)

axes[0].set_ylabel("Window size (TRs)", fontsize=11)

# Band labels — placed in figure coordinates to the left of the y-axis label
ax0      = axes[0]
ws_list0 = hist_data["T21"][0]
for label, (lo, hi) in BANDS.items():
    if lo in ws_list0 and hi in ws_list0:
        y_mid = (ws_list0.index(lo) + ws_list0.index(hi)) / 2 + 0.5
        # Use axis transform for x (data) but keep y in axis (row) units
        ax0.annotate(
            label,
            xy=(BINS_HM[0], y_mid),
            xytext=(BINS_HM[0] - 0.18, y_mid),   # further left, well past y-axis label
            fontsize=9, color=BAND_COLORS[label], fontweight="bold",
            ha="right", va="center", annotation_clip=False,
        )

# Colourbar in its own axis to the right of both panels (never overlaps data)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])   # [left, bottom, width, height]
cbar    = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Density", fontsize=10)

plt.tight_layout()
plt.savefig("fig/fig1_speed_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig/fig1_speed_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — KDE + histogram per timescale band, T21 vs WT
# ═══════════════════════════════════════════════════════════════════════════
#
# One column per band.  Each panel:
#   - filled histogram (semi-transparent) showing raw pooled data
#   - smooth KDE curve overlaid
#   - dotted vertical line at each group's median
#   - significance stars (Mann-Whitney U) in top-right corner

x_kde   = np.linspace(0.3, 1.4, 500)
n_bands = len(BANDS)

fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5), sharey=False)
fig.suptitle("Pooled dFC speed by timescale — T21 vs WT",
             fontsize=14, fontweight="bold")

for ax, (label, pools) in zip(axes, band_pools.items()):
    bcol = BAND_COLORS[label]
    lo, hi = BANDS[label]

    for name in ("WT", "T21"):
        arr = pools[name]
        if arr.size < 5:
            continue
        col = C[name]
        med = np.median(arr)
        ls  = "-" if name == "T21" else "--"

        # Raw histogram (transparent fill, no edge)
        ax.hist(arr, bins=BINS_HIST, density=True,
                color=col, alpha=0.18, linewidth=0)

        # KDE curve
        kde = gaussian_kde(arr, bw_method=0.15)
        ax.plot(x_kde, kde(x_kde), lw=2.5, ls=ls, color=col,
                label=f"{name}  (med = {med:.3f})")

        # Median dotted line
        ax.axvline(med, lw=1.3, ls=":", color=col, alpha=0.8)

    # Mann-Whitney significance stars
    a, b = pools["T21"], pools["WT"]
    if a.size > 0 and b.size > 0:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax.text(0.97, 0.96, stars, transform=ax.transAxes,
                ha="right", va="top", fontsize=14, fontweight="bold", color="black")

    ax.set_title(
        f"{label}\n{lo}–{hi} TRs  (~{lo*TR:.0f}–{hi*TR:.0f} s)",
        fontsize=11, fontweight="bold", color=bcol,
    )
    ax.set_xlabel(r"$v_{dFC}$  (1 − Pearson corr)", fontsize=11)
    ax.set_xlim(0.4, 1.3)
    ax.legend(fontsize=9, framealpha=0.85, loc="upper left")

axes[0].set_ylabel("Density", fontsize=11)

plt.tight_layout()
plt.savefig("fig/fig2_band_distributions.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("Saved fig/fig2_band_distributions.png")
print("\nDone.")
