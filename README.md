# Part III Project: Bayesian Inference for Ice-Driven Sea Level Change

This repository contains my Part III research project on estimating ice-driven sea level change from satellite altimetry. The work tests the bias introduced by the standard sea-surface-height proxy approach, then replaces that proxy with a gravitationally consistent Bayesian inversion framework built on top of `pygeoinf` and `pyslfp`.

The project develops from simple error quantification experiments through to joint inversion over ice thickness change, firn compaction, and ocean dynamic topography, with later extensions including GRACE-informed joint experiments.

**Author:** Thomas Holland  
**Supervisor:** David Al-Attar

## Main Findings

- Treating sea surface height change as a direct proxy for sea level change introduces source-dependent bias.
- That bias cannot be removed with a single scalar correction, because it depends on where ice mass is changing.
- An infinite-dimensional Bayesian inversion gives substantially better calibrated uncertainty than the proxy method in synthetic twin experiments.
- Joint inversion over ice, firn, and ocean dynamic topography can separate signals that are entangled in the observations.
- A factored composite forward operator reduces the cost of the joint problem without changing the modelled physics.

For the full write-up, see `outputs/report/report.pdf` and the LaTeX source in `outputs/report/`.

## Method In Brief

Traditional global mean sea level estimates from satellite altimetry often average sea surface height change over the ocean and apply approximate corrections. This project studies the error in that approach and instead uses the full sea-level fingerprint physics linking ice mass loss, gravitational redistribution, solid Earth deformation, and the sea surface.

The central inversion framework is Bayesian and function-space based:

- priors are defined as Gaussian measures over spatial fields
- forward operators map ice and ocean signals into satellite-style observations
- posteriors are propagated through the full sea-level physics
- uncertainty is treated as part of the inference, not as an afterthought

Later stages extend the model space to infer multiple coupled signals jointly:

- ice thickness change
- firn compaction
- ocean dynamic topography
- GRACE-informed mass constraints in follow-on experiments

## Repository Structure

### Core Code

- `src/project/`: project-specific reusable operators, plotting helpers, projections, and the factored forward operator used in joint inversion experiments
- `src/pyslfp_extras/`: extensions around `pyslfp`, including altimetry sampling, GMSL operators, ice-sheet change models, and ocean-dynamics models
- `src/pygeoinf_extras/`: smaller helper operators, plotting, and statistical utilities built around `pygeoinf`

### Experiments And Analysis

- `work/`: numbered experiment directories, roughly following the scientific development of the project
- `notebooks/`: chapter-style notebooks that present the main ideas and figures more readably than the raw experiment scripts
- `da_work/`: supervisor-provided exploratory code and discussion material

### Outputs

- `outputs/report/`: final report source, figures, and compiled PDF
- `outputs/poster/`: poster materials
- `outputs/poster_sups/`: supplementary poster figures

## How The Work Is Organised

The `work/` directory is best read as a research log in code form rather than as a single polished pipeline.

Representative stages:

- `work/00_whole_ice_sheets/`: early whole-ice-sheet error experiments
- `work/01_major_source/`: source-specific bias analysis for major ice sheets
- `work/03_major_source_mixing/`: mixed-source experiments
- `work/04_other_signal_effects/`: ocean-dynamic and related signal effects
- `work/06_altimetry_sampling/`: converting gridded fields into satellite-style sampling
- `work/07_implementing_inversion/`: core inversion experiments, including sensitivity work
- `work/08_joint_inversion/` and `work/08a_small_joint_inversion/`: coupled inversion over multiple signals
- `work/10_time_series_inversion/`: early time-series experiments
- `work/11_realistic_ice/`: more physically motivated ice priors
- `work/13_knockout_test/` and `work/14_grace_joint_inversion/`: information-content and GRACE-coupled tests
- `work/15_regional_comparison/`: later comparison work
- `work/99_plotting/`: plotting scripts used to produce report figures

## Best Entry Points

If you want the clearest route through the project, start here:

1. `outputs/report/report.pdf` for the complete scientific narrative.
2. `notebooks/01 - Sea Surface Height.ipynb` through `notebooks/09 - Truth-Prior Cross Test.ipynb` for the main computational story in notebook form.
3. `src/pyslfp_extras/` and `src/project/` for the reusable implementation.
4. `work/07_implementing_inversion/`, `work/08_joint_inversion/`, and `work/14_grace_joint_inversion/` for the main inversion scripts.

## Getting Started

This project is configured as a Python package with source under `src/`.

Requirements:

- Python `>=3.14`
- `uv` for environment and dependency management

Install dependencies:

```bash
uv sync
```

Optional development tools are defined in the `dev` dependency group.

## Reproducing Parts Of The Project

This repository does not expose one single end-to-end command for all results. Instead, reproduction is split across notebooks and experiment scripts.

Practical routes:

1. Use the notebooks in `notebooks/` for the most readable walkthrough of the main methods and figures.
2. Use `work/99_plotting/` to regenerate many report-ready figures from saved experiment outputs.
3. Use scripts in `work/07_implementing_inversion/` and `work/08_joint_inversion/` for the main synthetic inversion workflows.
4. Use `work/07_implementing_inversion/error/sensitivity_runner/README.md` for the sensitivity batch workflow.

Because many scripts reflect iterative research development, some are intended for targeted reruns rather than turnkey execution from a clean checkout.

## Notebooks And Report

The notebooks roughly mirror the conceptual progression of the project:

- `01 - Sea Surface Height`: sea level versus sea surface height
- `02 - Major Sources`: source-dependent fingerprint errors
- `03 - Ocean Dynamics`: non-ice ocean signals
- `04 - Ice and Firn`: separating thickness and firn effects
- `05 - Altimetry Sampling`: observation modelling
- `06 - Simple Inversion`: first inversion examples
- `07 - Inversion Sensitivity`: prior and parameter sensitivity
- `08 - Joint Inversion`: coupled multi-signal inversion
- `09 - Truth-Prior Cross Test`: robustness checks

Report source is organised under `outputs/report/parts/` with separate files for abstract, introduction, methods, results, discussion, and conclusion.

## Dependencies And Related Packages

This work builds directly on:

- [`pygeoinf`](https://github.com/da380/pygeoinf): Bayesian inverse problems on Hilbert spaces
- [`pyslfp`](https://github.com/da380/pyslfp): sea-level fingerprint physics and operators

Other core scientific dependencies are listed in `pyproject.toml` and include `numpy`, `matplotlib`, `pandas`, `xarray`, `netcdf4`, `seaborn`, `pygmt`, and related tooling.
