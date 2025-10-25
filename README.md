# Part III Project: Bayesian Inversion Methods for Sea Level Change Estimation

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
├── src/                        # Code for my project that I have written
│   └── Part_III_Project/
│
├── work/                       # Working notebooks and analysis
│   ├── Sea Level to Sea Surface.ipynb
│   └── traditional_methods/        # Traditional methods analysis                     
│
├── work_DA/                       # Working directory of my supervisor
├── outputs/                    # Report and bibliography
│   ├── refs.bib
│   ├── report/
│   ├── poster/
│   └── presentation/
│
└── work_log/                   # Project documentation and notes with weekly progress
```
