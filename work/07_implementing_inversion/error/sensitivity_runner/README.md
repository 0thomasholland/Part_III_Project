# sensitivity_runner

This folder mirrors the error runner workflow for sensitivity analysis.

## What it produces

For each truth setup, the runner executes all three sensitivity sweeps from
`07 - Inversion Sensitivity.ipynb`:

- length scale: `[0.05, 0.15, 0.4] * mean_sea_floor_radius`
- prior mean offset (mm): `[1, 10, 50]`
- covariance amplitude multiplier: `[0.5, 2]`

Each setup yields 9 case records and is written to one pickle file in
`inversion_results/`.

## Scripts

- `sensitivity_runner.py`: parallel case generation and pickle output
- `data_globber.py`: consolidate pickles into `master_results_wide.csv` (wide)
- `sensitivity_plots.py`: quick diagnostics from `master_results_wide.csv`

## Typical usage

Run from this directory.

```bash
# 1) Edit variables at the top of sensitivity_runner.py
#    (TOTAL_SETUPS, START_INDEX, N_JOBS, etc.)
python sensitivity_runner.py
python data_globber.py
python sensitivity_plots.py
```

## Expected row count

`rows = total_setups * 9`

Examples:

- `total_setups=2` -> `18` rows
- `total_setups=200` -> `1800` rows

For the wide table (`master_results_wide.csv`):

- rows = `total_setups`
- columns = setup-level truth/altimetry fields + one set of metric columns
	per sensitivity case

## Core CSV columns

- setup-level fields: `setup_index`, truth/altimetry metadata
- case columns: `<metric>__<sweep_type>__<sweep_value>`
	- example: `posterior_bias_mm__mean_offset__10`
