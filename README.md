# dfc_speed

**Dynamic Functional Connectivity Speed — Computation & Group Analysis**

A standalone Python toolkit for computing and analysing the *speed* of dynamic functional connectivity (dFC) — quantifying how rapidly the brain's functional organisation reconfigures over time.

Extracted and refactored from [`net_fluidity`](https://github.com/SamyCN89/net_fluidity) for modular, collaborator-ready use.

---

## What is dFC speed?

dFC speed is the frame-to-frame distance between consecutive FC matrices in upper-triangle vector space:

```
speed(t) = dist( FC(t), FC(t+1) )   t = 1 … T-1
```

High speed → rapid functional reconfiguration.
Low speed → stable, slowly evolving connectivity.

---

## Repository structure

```
dfc_speed/
├── dfc_speed/
│   ├── __init__.py
│   ├── dfc_speed_compute.py       # Core speed computation (single / batch)
│   └── dfc_speed_distributions.py # Group analysis, permutation test, plots
├── notebooks/
│   └── 01_dfc_speed_demo.ipynb    # End-to-end demo with simulated data
├── data/
│   └── example/                   # Example outputs (figures, NPZ)
├── tests/
│   └── test_speed.py
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/SamyCN89/dfc_speed.git
cd dfc_speed
pip install -r requirements.txt
```

Or install directly:
```bash
pip install -e .
```

---

## Quick start

```python
import numpy as np
from dfc_speed import compute_dfc_speed, permutation_test_speed

# fc_timeseries: (T, N, N) array of sliding-window FC matrices
speed = compute_dfc_speed(fc_timeseries)   # → (T-1,) speed vector

# Group comparison
obs_diff, p_val = permutation_test_speed(speeds_group_a, speeds_group_b)
print(f"Δ speed = {obs_diff:.4f}, p = {p_val:.4f}")
```

See `notebooks/01_dfc_speed_demo.ipynb` for a full walkthrough.

---

## Input format

| Variable | Shape | Description |
|---|---|---|
| `fc_timeseries` | `(T, N, N)` | Sequence of symmetric FC matrices |
| `fc_batch` | `(S, T, N, N)` | Batch: S subjects × T timepoints |
| `speeds_group_x` | `list of (T-1,)` | Per-subject speed arrays |

FC matrices can be produced by any sliding-window, tapered, or point-process dFC estimator (e.g. via `net_fluidity`).

---

## Key functions

| Function | Description |
|---|---|
| `compute_dfc_speed(fc, metric)` | Single-session speed vector |
| `compute_dfc_speed_batch(fc_batch)` | Batch version across subjects |
| `speed_summary(speed)` | Mean, std, median, IQR |
| `group_summary(speeds_dict)` | Per-condition mean ± SEM |
| `permutation_test_speed(a, b)` | Two-group permutation test (two-tailed) |
| `bootstrap_mean_ci(speed_list)` | Bootstrap 95% CI on group mean |
| `plot_speed_distributions(...)` | Violin + strip plot |
| `plot_speed_timecourse(...)` | Single-subject timecourse |

---

## Requirements

```
numpy>=1.22
scipy>=1.8
matplotlib>=3.5
```

---

## License

MIT License. See `LICENSE`.

---

## Contact

Samy Castro · FunSy team, LNCA, University of Strasbourg
GitHub: [@SamyCN89](https://github.com/SamyCN89)
