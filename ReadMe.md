# Ridiculously Simple Data-Driven Air Pollution Interpolation Method (RSIM)

This repository contains MATLAB code and example synthetic data for reproducing and comparing air pollution interpolation methods, including the **Ridiculously Simple Interpolation Method (RSIM)** described in:

> Feldman, A., Kendler, S., Pisoni, E., Fishbain, B. *Ridiculously Simple Data-Driven Air Pollution Interpolation Method*. *Environmental Modelling & Software* (submitted, 2025).

---

## Quick Start
```matlab
addpath('src');
RSIM_main;         % train & evaluate RSIM on the selected dataset
% or
compare_with_ui;   % run visual + numeric comparison (guided UI)
```

---

## Repository Structure
```
.
├── data/                 # Example .mat files (synthetic data)
│   ├── 1.mat
│   ├── 2.mat
│   └── ... (20 total)
├── src/                  # MATLAB source code
│   ├── RSIM_main.m       # RSIM training & evaluation pipeline
│   └── compare_with_ui.m # Visual & numeric comparison script
└── README.md
```

---

## Requirements
- **MATLAB** R2021a or newer
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) (for RSIM)
- *(Optional)* Python 3.x with [PyKrige](https://github.com/GeoStat-Framework/PyKrige) installed (for Kriging methods)

---

## Data Format
Each `.mat` file in `data/` contains:
- `X` — ground truth pollution map (`H × W` array, e.g., `100 × 100`)
- `sensors` — `(row, col)` coordinates of available sensors (1-based indexing)

The provided files are **synthetic data** for demonstration and testing.

---

## How to Use

### 1) Run the comparison UI
```matlab
addpath('src');
compare_with_ui;
```
- Select the `data/` folder when prompted.
- Optionally load a trained RSIM network (`net`) for comparison.
- Choose which interpolation methods to include.
- Choose whether to compute numeric metrics (RMSE & correlation).
- Decide whether to show visual panels and/or save selected panels.

**Outputs (if saving enabled):**
- `panel_###.png` — visual panel for selected samples
- `summary_metrics.csv` — average RMSE and correlation per method

### 2) Train and evaluate RSIM
```matlab
addpath('src');
RSIM_main;
```
- Trains a linear RSIM model on the selected dataset and evaluates performance.

---

## Notes
- Kriging uses PyKrige via MATLAB–Python integration. If PyKrige is not available, Kriging panes are skipped (marked N/A) but other methods still run.
- RMSE and correlation are computed without toolbox dependencies (inline helper functions).
- Only **synthetic** data is included in `data/`. If you use real-world datasets (e.g., Antwerp), host them separately to comply with privacy/licensing.

---

## License
MIT License (recommended). If you adopt MIT, add a `LICENSE` file to the repo.
