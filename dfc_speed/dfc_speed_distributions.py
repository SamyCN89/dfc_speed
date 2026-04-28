"""
dfc_speed_distributions.py
--------------------------
Group-level analysis of dFC speed distributions:
  - Per-condition pooling and summary
  - Permutation-based group comparisons
  - Bootstrap confidence intervals
  - Plotting helpers (returns fig objects; no plt.show() calls)

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


# ── Pooling & summary ─────────────────────────────────────────────────────────

def pool_speeds(
    speeds_per_subject: List[np.ndarray],
) -> np.ndarray:
    """Concatenate speed vectors across subjects into a single array."""
    return np.concatenate(speeds_per_subject)


def group_summary(
    speeds_dict: Dict[str, List[np.ndarray]],
) -> Dict[str, dict]:
    """
    Compute mean ± SEM and median speed for each condition.

    Parameters
    ----------
    speeds_dict : dict
        Keys = condition labels, values = list of per-subject speed arrays.

    Returns
    -------
    summary : dict  {condition: {mean, sem, median, n_subjects}}
    """
    summary = {}
    for cond, speed_list in speeds_dict.items():
        subject_means = np.array([s.mean() for s in speed_list])
        summary[cond] = {
            "mean":       float(subject_means.mean()),
            "sem":        float(subject_means.std(ddof=1) / np.sqrt(len(subject_means))),
            "median":     float(np.median(subject_means)),
            "n_subjects": len(speed_list),
        }
    return summary


# ── Permutation test ──────────────────────────────────────────────────────────

def permutation_test_speed(
    group_a: List[np.ndarray],
    group_b: List[np.ndarray],
    n_permutations: int = 5000,
    random_state: int = 42,
) -> Tuple[float, float]:
    """
    Permutation test on the difference in mean speed between two groups.

    Returns
    -------
    observed_diff : float
    p_value : float (two-tailed)
    """
    rng = np.random.default_rng(random_state)
    a_means = np.array([s.mean() for s in group_a])
    b_means = np.array([s.mean() for s in group_b])
    observed_diff = a_means.mean() - b_means.mean()

    combined = np.concatenate([a_means, b_means])
    n_a = len(a_means)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(combined)
        null_dist[i] = perm[:n_a].mean() - perm[n_a:].mean()

    p_value = np.mean(np.abs(null_dist) >= np.abs(observed_diff))
    return float(observed_diff), float(p_value)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_mean_ci(
    speed_list: List[np.ndarray],
    n_boot: int = 5000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI around the group mean speed.

    Returns
    -------
    mean, ci_low, ci_high
    """
    rng = np.random.default_rng(random_state)
    subject_means = np.array([s.mean() for s in speed_list])
    n = len(subject_means)

    boot_means = np.array([
        rng.choice(subject_means, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])

    alpha = (1 - ci) / 2
    return (
        float(subject_means.mean()),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_speed_distributions(
    speeds_dict: Dict[str, List[np.ndarray]],
    colors: Optional[Dict[str, str]] = None,
    title: str = "dFC Speed Distributions",
) -> plt.Figure:
    """
    Violin + strip plot of per-subject mean speeds per condition.

    Returns a matplotlib Figure (caller handles saving/display).
    """
    conditions = list(speeds_dict.keys())
    subject_means = [
        [s.mean() for s in speeds_dict[c]] for c in conditions
    ]
    colors = colors or {c: f"C{i}" for i, c in enumerate(conditions)}

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(conditions)), 5))

    parts = ax.violinplot(subject_means, showmedians=True)
    for i, (pc, cond) in enumerate(zip(parts["bodies"], conditions)):
        pc.set_facecolor(colors[cond])
        pc.set_alpha(0.4)

    for i, (vals, cond) in enumerate(zip(subject_means, conditions)):
        ax.scatter(
            np.full(len(vals), i + 1) + np.random.uniform(-0.05, 0.05, len(vals)),
            vals,
            color=colors[cond],
            s=30,
            zorder=3,
            label=cond,
        )

    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean dFC speed (a.u.)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_speed_timecourse(
    speeds: np.ndarray,
    tr: float = 1.0,
    label: str = "speed",
    color: str = "steelblue",
) -> plt.Figure:
    """
    Plot a single subject's dFC speed timecourse.

    Parameters
    ----------
    speeds : (T-1,) array
    tr : repetition time in seconds (for x-axis)
    """
    t = np.arange(len(speeds)) * tr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, speeds, color=color, lw=1.2, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dFC speed")
    ax.set_title("dFC Speed Timecourse")
    ax.legend()
    fig.tight_layout()
    return fig
