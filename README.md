# Part III Project: Bayesian Inversion Methods for Sea Level Change Estimation

![GitHub last commit](https://img.shields.io/github/last-commit/0thomasholland/Part_III_Project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/0thomasholland/Part_III_Project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/0thomasholland/Part_III_Project)
![GitHub top language](https://img.shields.io/github/languages/top/0thomasholland/Part_III_Project)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/0thomasholland/Part_III_Project)
![GitHub repo size](https://img.shields.io/github/repo-size/0thomasholland/Part_III_Project)
![GitHub License](https://img.shields.io/github/license/0thomasholland/Part_III_Project)


A research project investigating the application of Bayesian inversion methods to estimate sea level change from satellite altimetry observations, incorporating sea level physics to improve upon traditional spatial averaging approaches.
<!-- 
**Author:** Thomas Holland  
**Project Supervisor:** David Al-Attar -->

## Key Bibliography

- [Lickley et al., ‘Bias in Estimates of Global Mean Sea Level Change Inferred from Satellite Altimetry’](https://doi.org/10.1175/JCLI-D-18-0024.1)
- [Al-Attar et al., ‘Reciprocity and Sensitivity Kernels for Sea Level Fingerprints’](https://doi.org/10.1093/gji/ggad434)
- [Al-Attar and Heathcote, pygeoinf](https://github.com/da380/pygeoinf)
- [Al-Attar and Heathcote, PySLFP](https://github.com/da380/pyslfp)

## Project Overview

This project aims to develop and compare methods for estimating global mean sea level (GMSL) change from satellite altimetry data. The research progresses from implementing traditional methods to developing Bayesian inversion approaches that incorporate sea level physics.

### Research Goals

1. **Traditional Methods Implementation** (Weeks 1-4 of Michaelmas Term)
   - Implement conventional methods for estimating sea level change from satellite altimetry
   - Investigate accuracy and error characteristics
   - Spatial averaging of sea surface height changes over oceans

2. **Bayesian Inversion Methods** (Weeks 5-8 of Michaelmas Term)
   - Apply Bayesian inversion incorporating sea level physics
   - Single-time (~ 1 month data averages) estimates initially
   - Compare new methods against traditional approaches
   - Explore error space for conventional methods across satellite availability bands

3. **Time-Dependent Extensions** (Christmas Break)
   - Extend to time-dependent estimates
   - Consider feed-forward mechanisms (Kalman filter-like approaches)

4. **Possible Extensions** (Lent Term)
   - Integration of ice altimetry data and other data types
   - Network robustness analysis (simulating data gaps)
   - Comparative analysis of data types
   - Feature resolution capabilities across different observation types

## Project Structure

```text
Part_III_Project/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
│
├── src/Part_III_Project/       # Core library code
│   ├── analysis.py             # Analysis utilities
│   ├── gaussian_measures.py    # Gaussian measure implementations
│   ├── ice_loads.py            # Ice load models
│   ├── measure_utilities.py    # Measure utility functions
│   ├── plotting.py             # Plotting functions
│   ├── ssh_operators.py        # Sea surface height operators
│   ├── utilities.py            # General utilities
│   └── visualization.py        # Visualization tools
│
├── scripts/                    # Analysis scripts
│   ├── bayesian_inversion/     # Bayesian inversion methods
│   │   ├── bayesian_inversion_initial.py
│   │   ├── bayesian_inversion_nicer.py
│   │   └── bayesian_inversion_lmax64_1000runs.ipynb
│   ├── gmsl_error/             # GMSL error analysis
│   │   ├── deterministic_latitude_sweep/   # Latitude coverage analysis
│   │   ├── deterministic_ternary/          # Ternary ice sheet mixture analysis
│   │   └── gaussian_operator_composition/  # Error via operator composition
│   ├── gmsl_estimation/        # GMSL estimation methods
│   │   ├── lmax_convergence/   # L_max convergence studies
│   │   ├── parameter_sweep/    # Parameter sweep analysis
│   │   └── single_case/        # Single case examples
│   └── figures/                # Figure generation scripts
│       └── poster/
│
├── outputs/                    # Report and bibliography
│   ├── refs.bib                # Bibliography
│   ├── report/                 # Project report (LaTeX)
│   └── poster/                 # Poster materials
│
├── work_da/                    # Working directory from supervisor
│   └── random_fields*.py       # Random field examples
│
└── log/                        # Project documentation and notes
    ├── Progress.md             # Weekly progress tracking
    ├── Maths.md                # Mathematical notes
    ├── Literature.md           # Literature review
    └── Code notes.md           # Code documentation
```

## Poster
![Poster](outputs/poster/Poster-page001.svg)
