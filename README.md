# Part III Project: Bayesian Inversion Methods for Sea Level Change Estimation

A research project investigating the application of Bayesian inversion methods to estimate sea level change from satellite altimetry observations, incorporating sea level physics to improve upon traditional spatial averaging approaches.
<!-- 
**Author:** Thomas Holland  
**Project Supervisor:** David Al-Attar -->

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
├── src/                        # Source code package
│   └── Part_III_Project/
│       ├── __init__.py
│       ├── load_generators.py           # Load generation utilities
│       ├── plot_methods.py              # Visualization functions
│       ├── sea_surface_height_change.py # SSH change calculations
│       └── sea_surface_height.py        # SSH fingerprint class (deprecated)
│
├── work/                       # Working notebooks and analysis
│   ├── Sea Level to Sea Surface.ipynb
│   └── traditional_methods/        # Traditional methods analysis
│       ├── load_lat.ipynb               # Load-latitude analysis
│       ├── ternary.ipynb                # Ternary diagrams for ice sheet contributions
│       ├── traditional_methods.ipynb    # Main traditional methods implementation  
│       └── *.csv                        
│
├── outputs/                    # Report and bibliography
│   ├── refs.bib
│   └── report/
│       ├── report.tex
│       └── parts/               # Report sections
│
└── work_log/                   # Project documentation and notes
    ├── Code notes.md
    ├── Introduction.md
    ├── Literature.md
    ├── Progress.md              # Weekly progress and meeting notes
    └── Thoughts.md
```
