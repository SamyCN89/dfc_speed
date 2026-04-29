from pathlib import Path
import pandas as pd
import numpy as np

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

PATH_TC = Path("/media/samy/Elements2/Proyectos/LauraHarsan/dataset/julien_caillette/time_courses_2/")
PATH_OUT = Path("/home/samy/Bureau/vscode/dfc_speed/data/outputs")
PATH_OUT.mkdir(parents=True, exist_ok=True)

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

WINDOW_LIST = list(range(5, 101))

rows = []
rows_global_mean = []
rows_global_percentiles = []
rows_global_raw = []
npz_dict = {}

for subject_file in SUBJECT_FILES:
    file_tc = PATH_TC / subject_file


    subject = subject_file.replace("tc_", "").replace("_37_seeds.mat", "")

    print(f"\n=== {subject} ===")

    for window_size in WINDOW_LIST:
        print(f"window = {window_size}")

        try:
            nodal_speed_percentiles, percentile_grid, roi_list, nodal_speeds, ts_shape, dfc_shape = compute_subject_nodal_speed(
                file_tc,
                window_size=window_size,
                lag=LAG,
                vstep=VSTEP,
                tau_range=TAU_RANGE,
                method="pearson",
            )

            npz_key = f"{subject}_w{window_size}"
            npz_dict[npz_key] = nodal_speeds


            for roi, vals in zip(roi_list, nodal_speed_percentiles):
                row = {
                    "subject": subject,
                    "ROI": roi,
                    "window": window_size,
                }

                for p, v in zip(percentile_grid, vals):
                    row[f"p{p:.5f}"] = float(v)

                rows.append(row)
                

        except Exception as e:
            print(f"[ERROR NODAL] {subject} | window={window_size} | {e}")

        try:
            speed_mean, percentiles, speed_values = compute_subject_global_speed(
                file_tc,
                window_size=window_size,
                lag=LAG,
                vstep=VSTEP,
                tau_range=TAU_RANGE,
                method="pearson",
            )

            rows_global_mean.append(
                {
                    "subject": subject,
                    "window": window_size,
                    "speed": float(speed_mean),
                }
            )

            row_percentiles = {
                "subject": subject,
                "window": window_size,
            }

            for i, p in enumerate(percentiles):
                row_percentiles[f"p{i}"] = float(p)

            rows_global_percentiles.append(row_percentiles)


            # Définir le bin de fenêtre
            if window_size <= 31:
                window_bin = "short"
            elif window_size <= 62:
                window_bin = "mid"
            else:
                window_bin = "long"

            # Ajouter toutes les valeurs de speed (distribution brute)
            for val in speed_values:
                rows_global_raw.append(
                    {
                        "subject": subject,
                        "window": window_size,
                        "window_bin": window_bin,
                        "speed": float(val),
                    }
                )

        except Exception as e:
            print(f"[ERROR GLOBAL] {subject} | window={window_size} | {e}")

df = pd.DataFrame(rows)
csv_file = PATH_OUT / "nodal_speed_all_subjects_windows_5_100.csv"
df.to_csv(csv_file, index=False)

npz_file = PATH_OUT / "nodal_speed_all_subjects_windows_5_100.npz"
np.savez_compressed(npz_file, **npz_dict)

df_global_mean = pd.DataFrame(rows_global_mean)
csv_global_mean = PATH_OUT / "global_speed_mean_all_subjects_windows_5_100.csv"
df_global_mean.to_csv(csv_global_mean, index=False)

df_global_percentiles = pd.DataFrame(rows_global_percentiles)
csv_global_percentiles = PATH_OUT / "global_speed_percentiles_all_subjects_windows_5_100.csv"
df_global_percentiles.to_csv(csv_global_percentiles, index=False)

df_raw = pd.DataFrame(rows_global_raw)
df_raw.to_csv(PATH_OUT / "global_speed_raw.csv", index=False)

print("\nDone.")
print(f"Nodal CSV saved: {csv_file}")
print(f"Nodal NPZ saved: {npz_file}")
print(f"Global mean CSV saved: {csv_global_mean}")
print(f"Global percentiles CSV saved: {csv_global_percentiles}")
