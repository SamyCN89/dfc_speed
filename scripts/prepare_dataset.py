"""
prepare_dataset.py
------------------
Copies all Coimagine .mat files to the local dataset folder and saves
the canonical ROI list as a companion .txt and .npy file.

Usage:
    python scripts/prepare_dataset.py

Output structure:
    data/dataset/
        roi_names.txt          ← one ROI name per line
        roi_names.npy          ← numpy array of strings
        time_courses/
            tc_Coimagine_EDM92_0136_37_seeds.mat
            tc_Coimagine_EDM92_0137_37_seeds.mat
            ... (all 51 subjects)
"""

from pathlib import Path
import shutil
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PATH_SRC = Path("/media/samy/Elements2/Proyectos/LauraHarsan/dataset/julien_caillette/time_courses_2")
PATH_DST = Path("/home/samy/Bureau/vscode/dfc_speed/data/dataset")
PATH_TC  = PATH_DST / "time_courses"

PATH_TC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Canonical ROI list (37 regions, bilateral)
# ---------------------------------------------------------------------------
ROI_NAMES = [
    "Both_AI",
    "Both_ORB",
    "Both_ILA",
    "Both_PL",
    "Both_ACA",
    "Both_RSP",
    "Both_VIS",
    "Both_PTLp",
    "Both_TEa",
    "Both_MOp",
    "Both_MOs",
    "Both_SSp",
    "Both_SSs",
    "Both_AUD",
    "Both_GU_VISC",
    "Both_PERI",
    "Both_ENT",
    "Both_ECT",
    "Both_dhc",
    "Both_vhc",
    "Both_SUB",
    "Both_EP",
    "Both_CLA",
    "Both_PIR",
    "Both_PALd",
    "Both_PALv",
    "Both_ACB",
    "Both_CP",
    "Both_LSX",
    "Both_AAA_CEA_MEA",
    "Both_HY",
    "Both_PO_PF",
    "Both_VPL_VPM",
    "Both_VTA",
    "Both_MBmot",
    "Both_MBsen",
    "Both_MBsta",
]

assert len(ROI_NAMES) == 37, f"Expected 37 ROIs, got {len(ROI_NAMES)}"

# ---------------------------------------------------------------------------
# Save ROI names
# ---------------------------------------------------------------------------
roi_txt = PATH_DST / "roi_names.txt"
roi_npy = PATH_DST / "roi_names.npy"

with open(roi_txt, "w") as f:
    for name in ROI_NAMES:
        f.write(name + "\n")

np.save(roi_npy, np.array(ROI_NAMES))

print(f"ROI names saved: {roi_txt}")
print(f"ROI names saved: {roi_npy}")

# ---------------------------------------------------------------------------
# Copy .mat files
# ---------------------------------------------------------------------------
mat_files = sorted(PATH_SRC.glob("tc_Coimagine_*.mat"))

if not mat_files:
    print(f"\n[ERROR] No .mat files found in {PATH_SRC}")
    print("Check that the external drive is mounted.")
else:
    print(f"\nFound {len(mat_files)} .mat files — copying to {PATH_TC} ...")
    for i, src in enumerate(mat_files):
        dst = PATH_TC / src.name
        if dst.exists():
            print(f"  [{i+1:02d}/{len(mat_files)}] SKIP (already exists): {src.name}")
        else:
            shutil.copy2(src, dst)
            print(f"  [{i+1:02d}/{len(mat_files)}] COPIED: {src.name}")

    print(f"\nDone. {len(list(PATH_TC.glob('*.mat')))} files in {PATH_TC}")

# ---------------------------------------------------------------------------
# Quick sanity check on first file
# ---------------------------------------------------------------------------
from scipy.io import loadmat

first = sorted(PATH_TC.glob("*.mat"))[0]
mat   = loadmat(first)
tc    = mat["tc"]
if tc.shape[0] < tc.shape[1]:
    tc = tc.T

print(f"\nSanity check on: {first.name}")
print(f"  tc shape (timepoints x ROIs): {tc.shape}")
print(f"  ROI list length             : {len(ROI_NAMES)}")

if tc.shape[1] == len(ROI_NAMES):
    print("  ✓ ROI count matches perfectly.")
else:
    print(f"  ✗ MISMATCH — tc has {tc.shape[1]} columns but ROI list has {len(ROI_NAMES)} names.")
