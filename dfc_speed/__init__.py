"""dfc_speed — Dynamic FC speed computation and distribution analysis."""

from .dfc_speed_compute import (
    compute_dfc_speed,
    compute_dfc_speed_batch,
    speed_summary,
)
from .dfc_speed_distributions import (
    pool_speeds,
    group_summary,
    permutation_test_speed,
    bootstrap_mean_ci,
    plot_speed_distributions,
    plot_speed_timecourse,
)

__all__ = [
    "compute_dfc_speed",
    "compute_dfc_speed_batch",
    "speed_summary",
    "pool_speeds",
    "group_summary",
    "permutation_test_speed",
    "bootstrap_mean_ci",
    "plot_speed_distributions",
    "plot_speed_timecourse",
]
