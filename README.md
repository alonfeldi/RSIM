
# RSIM (with user synthetic data)

Minimal MATLAB pipeline using your `.mat` files (`X`, `sensors`) for training and evaluation.
- Run `src/RSIM_main.m` in MATLAB and choose the folder with `.mat` files.
- CSV exports for GitHub preview are under `data/maps/`.

## Quick start
```matlab
addpath('src');
RSIM_main;
```

## Data
The `data/maps/` folder contains CSVs converted from your uploaded MAT files for easy preview:
- `map_1.csv`, `map_2.csv`, `map_3.csv`
- `sensors.csv` (from the first file)

You can add more `.mat` files to your own repo and the code will use them directly.

## Notes
- This structure is ready for GitHub and for EMS "Software & Data Availability".
- For large datasets, consider publishing on Zenodo and linking the DOI here.
