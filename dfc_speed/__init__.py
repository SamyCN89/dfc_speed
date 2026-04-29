"""dfc_speed — Dynamic Functional Connectivity Speed."""

from .dfc_speed_nodal import (
    ts2dfc_stream,
    dfc_speed_split,
    dfc_speed_nodal,
    compute_subject_nodal_speed,
    compute_subject_global_speed,
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
    # core
    "ts2dfc_stream",
    "dfc_speed_split",
    "dfc_speed_nodal",
    # subject wrappers
    "compute_subject_nodal_speed",
    "compute_subject_global_speed",
    # group analysis
    "pool_speeds",
    "group_summary",
    "permutation_test_speed",
    "bootstrap_mean_ci",
    # plots
    "plot_speed_distributions",
    "plot_speed_timecourse",
]