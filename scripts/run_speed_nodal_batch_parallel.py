"""
run_speed_nodal_batch_parallel.py
----------------------------------
Parallelized version of run_speed_nodal_batch.py using joblib.
Each subject is processed on a separate CPU core.

Usage:
    python scripts/run_speed_nodal_batch_parallel.py

Outputs (same as serial version):
    data/outputs/nodal_speed_all_subjects_windows_5_100.csv
    data/outputs/nodal_speed_all_subjects_windows_5_100.npz
    data/outputs/global_speed_mean_all_subjects_windows_5_100.csv
    data/outputs/global_speed_percentiles_all_subjects_windows_5_100.csv
    data/outputs/global_speed_raw.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from dfc_speed.dfc_speed_nodal import (
    WINDOW,
    LAG,
    VSTEP,
    SEED,
    THRESHOLD,
    TAU_RANGE,
    compute_subject_nodal_speed,
    compute_subject_global_speed,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Replace the hardcoded paths with these
PATH_TC  = Path(__file__).parent.parent / "data" / "dataset" / "time_courses"
PATH_OUT = Path(__file__).parent.parent / "data" / "outputs"
PATH_OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
WINDOW_LIST = list(range(5, 101))
N_JOBS      = 8   # number of parallel workers (one per subject)

SUBJECT_FILES = [
    "tc_Coimagine_EDM92_0136_37_seeds.mat",
    "tc_Coimagine_EDM92_0137_37_seeds.mat",
    "tc_Coimagine_EDM92_100_37_seeds.mat",
    "tc_Coimagine_EDM92_101_37_seeds.mat",
    "tc_Coimagine_EDM92_114_37_seeds.mat",
    "tc_Coimagine_EDM92_128_37_seeds.mat",
    "tc_Coimagine_EDM92_133_37_seeds.mat",
    "tc_Coimagine_EDM92_134_37_seeds.mat",
    "tc_Coimagine_EDM92_137_37_seeds.mat",
    "tc_Coimagine_EDM92_138_37_seeds.mat",
    "tc_Coimagine_EDM92_147_37_seeds.mat",
    "tc_Coimagine_EDM92_155_37_seeds.mat",
    "tc_Coimagine_EDM92_158_37_seeds.mat",
    "tc_Coimagine_EDM92_159_37_seeds.mat",
    "tc_Coimagine_EDM92_177_37_seeds.mat",
    "tc_Coimagine_EDM92_36_37_seeds.mat",
    "tc_Coimagine_EDM92_38_37_seeds.mat",
    "tc_Coimagine_EDM92_41_37_seeds.mat",
    "tc_Coimagine_EDM92_50_37_seeds.mat",
    "tc_Coimagine_EDM92_60_37_seeds.mat",
    "tc_Coimagine_EDM92_68_37_seeds.mat",
    "tc_Coimagine_EDM92_75_37_seeds.mat",
    "tc_Coimagine_EDM92_76_37_seeds.mat",
    "tc_Coimagine_EDM92_84_37_seeds.mat",
    "tc_Coimagine_EDM92_86_37_seeds.mat",
    "tc_Coimagine_EDM92_88_37_seeds.mat",
    "tc_Coimagine_VEH_0133_37_seeds.mat",
    "tc_Coimagine_VEH_0134_37_seeds.mat",
    "tc_Coimagine_VEH_0138_37_seeds.mat",
    "tc_Coimagine_VEH_108_37_seeds.mat",
    "tc_Coimagine_VEH_120_37_seeds.mat",
    "tc_Coimagine_VEH_127_37_seeds.mat",
    "tc_Coimagine_VEH_131_37_seeds.mat",
    "tc_Coimagine_VEH_132_37_seeds.mat",
    "tc_Coimagine_VEH_135_37_seeds.mat",
    "tc_Coimagine_VEH_148_37_seeds.mat",
    "tc_Coimagine_VEH_149_37_seeds.mat",
    "tc_Coimagine_VEH_174_37_seeds.mat",
    "tc_Coimagine_VEH_176_37_seeds.mat",
    "tc_Coimagine_VEH_35_37_seeds.mat",
    "tc_Coimagine_VEH_40_37_seeds.mat",
    "tc_Coimagine_VEH_49_37_seeds.mat",
    "tc_Coimagine_VEH_59_37_seeds.mat",
    "tc_Coimagine_VEH_67_37_seeds.mat",
    "tc_Coimagine_VEH_69_37_seeds.mat",
    "tc_Coimagine_VEH_70_37_seeds.mat",
    "tc_Coimagine_VEH_74_37_seeds.mat",
    "tc_Coimagine_VEH_83_37_seeds.mat",
    "tc_Coimagine_VEH_87_37_seeds.mat",
    "tc_Coimagine_VEH_95_37_seeds.mat",
    "tc_Coimagine_VEH_97_37_seeds.mat",
]


# ---------------------------------------------------------------------------
# Per-subject worker function
# ---------------------------------------------------------------------------
def process_subject(subject_file):
    """
    Process one subject across all window sizes.
    Returns four lists of dicts (nodal, global_mean, global_pct, global_raw)
    plus a dict of npz arrays keyed by subject_w{window}.
    """
    file_tc = PATH_TC / subject_file
    subject = subject_file.replace("tc_", "").replace("_37_seeds.mat", "")

    rows_nodal      = []
    rows_gmean      = []
    rows_gpct       = []
    rows_graw       = []
    npz_entries     = {}

    for window_size in WINDOW_LIST:

        # --- nodal speed ---
        try:
            nodal_speed_percentiles, percentile_grid, roi_list, nodal_speeds, _, _ = \
                compute_subject_nodal_speed(
                    file_tc,
                    window_size=window_size,
                    lag=LAG,
                    vstep=VSTEP,
                    tau_range=TAU_RANGE,
                    method="pearson",
                )

            npz_entries[f"{subject}_w{window_size}"] = nodal_speeds

            for roi, vals in zip(roi_list, nodal_speed_percentiles):
                row = {"subject": subject, "ROI": roi, "window": window_size}
                for p, v in zip(percentile_grid, vals):
                    row[f"p{p:.5f}"] = float(v)
                rows_nodal.append(row)

        except Exception as e:
            print(f"[ERROR NODAL] {subject} | window={window_size} | {e}")

        # --- global speed ---
        try:
            speed_mean, percentiles, speed_values = compute_subject_global_speed(
                file_tc,
                window_size=window_size,
                lag=LAG,
                vstep=VSTEP,
                tau_range=TAU_RANGE,
                method="pearson",
            )

            rows_gmean.append({
                "subject": subject,
                "window":  window_size,
                "speed":   float(speed_mean),
            })

            row_pct = {"subject": subject, "window": window_size}
            for i, p in enumerate(percentiles):
                row_pct[f"p{i}"] = float(p)
            rows_gpct.append(row_pct)

            window_bin = (
                "short" if window_size <= 31
                else "mid" if window_size <= 62
                else "long"
            )
            for val in speed_values:
                rows_graw.append({
                    "subject":    subject,
                    "window":     window_size,
                    "window_bin": window_bin,
                    "speed":      float(val),
                })

        except Exception as e:
            print(f"[ERROR GLOBAL] {subject} | window={window_size} | {e}")

    return rows_nodal, rows_gmean, rows_gpct, rows_graw, npz_entries


# ---------------------------------------------------------------------------
# Main: parallel execution + aggregation
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print(f"Running {len(SUBJECT_FILES)} subjects on {N_JOBS} cores ...")
    print(f"Window range: {WINDOW_LIST[0]}–{WINDOW_LIST[-1]} ({len(WINDOW_LIST)} sizes)\n")

    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=0)(
        delayed(process_subject)(sf)
        for sf in tqdm(SUBJECT_FILES, desc="Subjects", unit="subj")
    )

    # --- aggregate ---
    all_nodal, all_gmean, all_gpct, all_graw, all_npz = [], [], [], [], {}

    for rows_nodal, rows_gmean, rows_gpct, rows_graw, npz_entries in results:
        all_nodal.extend(rows_nodal)
        all_gmean.extend(rows_gmean)
        all_gpct.extend(rows_gpct)
        all_graw.extend(rows_graw)
        all_npz.update(npz_entries)

    # --- save ---
    csv_nodal = PATH_OUT / "nodal_speed_all_subjects_windows_5_100.csv"
    npz_nodal = PATH_OUT / "nodal_speed_all_subjects_windows_5_100.npz"
    csv_gmean = PATH_OUT / "global_speed_mean_all_subjects_windows_5_100.csv"
    csv_gpct  = PATH_OUT / "global_speed_percentiles_all_subjects_windows_5_100.csv"
    csv_graw  = PATH_OUT / "global_speed_raw.csv"

    pd.DataFrame(all_nodal).to_csv(csv_nodal, index=False)
    np.savez_compressed(npz_nodal, **all_npz)
    pd.DataFrame(all_gmean).to_csv(csv_gmean, index=False)
    pd.DataFrame(all_gpct).to_csv(csv_gpct,  index=False)
    pd.DataFrame(all_graw).to_csv(csv_graw,  index=False)

    print("\nDone.")
    print(f"  Nodal CSV : {csv_nodal}")
    print(f"  Nodal NPZ : {npz_nodal}")
    print(f"  Global mean CSV        : {csv_gmean}")
    print(f"  Global percentiles CSV : {csv_gpct}")
    print(f"  Global raw CSV         : {csv_graw}")
