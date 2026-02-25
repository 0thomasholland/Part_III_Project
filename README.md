# Part III Project: Bayesian Inversion Methods for Sea Level Change Estimation

![Tests](https://github.com/0thomasholland/Part_III_Project/actions/workflows/tests.yml/badge.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/0thomasholland/Part_III_Project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/0thomasholland/Part_III_Project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/t/0thomasholland/Part_III_Project)
![GitHub top language](https://img.shields.io/github/languages/top/0thomasholland/Part_III_Project)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/0thomasholland/Part_III_Project)
![GitHub repo size](https://img.shields.io/github/repo-size/0thomasholland/Part_III_Project)
![GitHub License](https://img.shields.io/github/license/0thomasholland/Part_III_Project)


A research project investigating the application of Bayesian inversion methods to estimate sea level change from satellite altimetry observations, incorporating sea level physics to improve upon traditional spatial averaging approaches.
 
**Author:** Thomas Holland  
**Project Supervisor:** David Al-Attar 


## Project Overview

This project aims to develop and compare methods for estimating global mean sea level (GMSL) change from satellite altimetry data. The research progresses from implementing traditional methods to developing Bayesian inversion approaches that incorporate sea level physics.

### Research Goals

1. **Traditional Methods Implementation**
   - Implement conventional methods for estimating sea level change from satellite altimetry
   - Investigate accuracy and error characteristics
   - Spatial averaging of sea surface height changes over oceans

2. **Bayesian Inversion Methods**
   - Apply Bayesian inversion incorporating sea level physics
   - Single-time (~ 1 month data averages) estimates initially
   - Compare new methods against traditional approaches

4. **Extensions**
   - Integration of ice altimetry data and other data types (tide gauage data)
   - Modeling ice thickness vs firn thickness changes
   - Comparative analysis of data types
   - Altimetry resolution comparison
   - Generating realistic priors

## Repository structure

- `da_work/` - example code used by supervisor to discuss key points
- `outputs/` - written outputs of the project:
  - `poster` - project poster created at the halfway point of the project
  - `poster_sups` - the poster suplimentals created to support the poster
  - `presentation` - the presentation given at the three-quarter point of the project
  - `report` - the final report of the project
- `src/` - reused "package" code for the project, split into:
  - `project` - specific code reused within this project
  - `pygeoinf_extras` - code that specifically extends the [pygeoinf (Al-Attar and Heathcote)](https://github.com/da380/pygeoinf) package
  - `pyslfp_extras` - code that specifically extends the [PySLFP (Al-Attar and Heathcote)](https://github.com/da380/pyslfp) package
- `work/` - the science done
- `notebooks/` - notebooks that specifically showcase parts of the custom packages or the work done in `work/`, referenced in the report

## Work completed


### Structure

Folders are structured in terms of "questions":

- 00 - all ice field error: `with uniform/uniform distribution of ice thickness change across ice fields what is the error?`
- 01 - major sources: `how does the error vary across different ice sheets?`
- 02 - ice load bands: `how does changing the load latitude vs altimetry sampling latitude affect the error?`
- 03 - major source mixing: `how does mixing different major sources affect the error?`
- 04 - signal effects: `how do other signals (e.g. ODT, signal noise, etc) affect the error associated?`
- 05 - bayesian inversions: `what are bayesian inversions?`
- 06 - altimetry sampling: `how can altimeter data be sampled to generate a point field?`
- 07 - inversion: `how can we use baysian inversion with sea altimetry data?`
- XX - other data: `can we use other data sources to improve accuracy?`

### Order of work

Deterministic:
- 00 - all ice field error
- 01 - major sources in scalar fields
- 03 - looking at mixing of major sources

Gaussian framework:
- 00 - gaussian all ice field error
- 01 - major sources in gaussian fields
- 05 - learning about inversions using pygeoinf
- 06 - satellite altimetry sampling
- 07 - implementing inversions
- 04 - adding in other signals (ODT, etc)
- 04 - refineing ODT fields
- 10 - having a look at time series inversions (simplistic)
- 11 - realistic ice fields
