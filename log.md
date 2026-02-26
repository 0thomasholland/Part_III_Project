# Development Log

## 10 October 2025

- Initialised project repository with configuration files, notebook structure, and automated PDF generation workflow.
- Created initial project notebook with introduction, literature review, progress tracking, and research notes sections.
- Set up Poetry for dependency management and configured the development environment.

## 11 October 2025

- Added preliminary LaTeX report structure and expanded notebook content with introductory material.

## 13 October 2025

- Implemented the initial sea level to sea surface height conversion code in `sea_surface_height.py`, establishing the core SSH computation module.
- Created the first working notebook demonstrating the SL-to-SSH transformation using pyslfp fingerprint data.
- Reorganised project files: moved notebooks to dedicated directory and work logs to `work_log/`.

## 14 October 2025

- Extended the FingerPrint class with a `SeaSurfaceFingerPrint` subclass, enabling direct SSH computation from sea level fingerprints.
- Added SLC and SSHC comparison plots to the notebook, visualising the distinction between sea level change and sea surface height change.
- Wrote initial error quantification code computing relative errors for Greenland, West Antarctic, and East Antarctic ice sheet loads across a range of satellite latitude coverages using the pyslfp fingerprint library.
- This establishes the foundation of the empirical error quantification — assessing how inaccurate the traditional altimetry-as-proxy approach is as a function of ice source location and satellite coverage band.
- Set up Google Docs synchronisation workflow for progress tracking.

## 15 October 2025

- Refactored the `SeaSurfaceFingerPrint` class methods and removed the original standalone conversion function in favour of the class-based approach.

## 17 October 2025

- Refactored the per-ice-sheet error loop into a dictionary-based structure and corrected the placement of the mean sea level change computation outside the inner latitude loop (a correctness fix as well as a performance improvement).
- Added sea level fingerprint calculation script with random surface mass load fractions for testing.
- Implemented initial error estimation functions and added ternary error analysis with CSV output.

## 18 October 2025

- Introduced parallelisation with joblib for the latitude/ice-sheet error computation, necessary given the number of configurations being swept.
- Added ternary plot functions to the plotting module and split traditional methods code into more manageable components.
- Transferred traditional methods analysis to notebook format for better documentation.
- Drafted initial project report structure with abstract, frontmatter, introduction, and previous methods sections in LaTeX.
- Added unit tests for the sea surface height change module and plot methods (later removed to streamline the CI workflow).

## 19 October 2025

- Implemented latitude band vs error analysis: computing how GMSL estimation error varies with the latitudinal extent of satellite coverage.
- Generated updated relative error plots and exported results to CSV for further analysis.

## 20 October 2025

- Re-ran ternary error analysis at higher resolution for improved accuracy.
- Added error distribution characterisation, including probability density function fitting for the 50–70° satellite coverage band.
- Increased computational resolution across multiple analysis scripts for publication-quality results.

## 21 October 2025

- Added p-value plots to the statistical analysis of error distributions.
- Reorganised traditional methods work into a `best_case/` subdirectory for clearer project structure.
- Created exploration area for random fields investigation using pygeoinf.

## 22 October 2025

- Added a simple random fields example demonstrating Gaussian random field generation on the sphere.
- Converted analysis scripts between notebook and Python file formats to find the optimal workflow.

## 23 October 2025

- Expanded random fields exploration with satellite range analysis and error field computations.

## 24 October 2025

- Added second random field example exploring different covariance structures.
- Ran random field error analysis at higher resolution.
- Expanded report with frontmatter and previous methods content.

## 25 October 2025

- Refined random field visualisation colour schemes for clarity.
- Updated README with key references.

## 31 October 2025

- Started work on third random fields example, building towards the full forward problem formulation.

## 1 November 2025

- Implemented pushforward workflow code for mapping distributions through the forward model.
- Added mathematical notes on measure theory foundations and distribution diagrams.
- Updated preamble formatting for improved PDF generation.

## 2 November 2025

- Added sampling methods to the random fields framework.

## 3 November 2025

- Updated pyslfp dependency and added fourth random field example incorporating ocean dynamic topography (ODT) noise.
- Began working on the full error quantification methods and forward problem data pipeline in `main.py`.

## 4 November 2025

- Reorganised project directory structure for better logical layout: renamed folders, moved files, and cleaned up scratch pads.
- Implemented measure space formulation code, laying out the mathematical structure for combining ice sheet and ocean dynamic measures.
- Updated methods documentation and whiteboard notes.

## 5 November 2025

- Continued implementing measure space functions, working towards a complete formulation of the direct sum of Gaussian measures.

## 6 November 2025

- Implemented direct sums of measures, enabling composition of ice sheet and ODT Gaussian measures in the joint model space.

## 7 November 2025

- Fixed measure sampling implementation and got the measure space code working correctly.
- Added parallel computation code for running multiple realisations.
- Debugged and resolved non-dimensionalisation issues in the forward model — identified and fixed a scaling bug affecting the error quantification.
- Updated variable inputs for optimised computation and implemented target GMSL functionality.

## 8 November 2025

- Updated plot methods for improved visualisation of error distributions.

## 10 November 2025

- Added error space exploration analysis, investigating the structure of errors across the parameter space.
- Merged parallel distribution runs branch into main.

## 11 November 2025

- Continued refining the parallel error computation framework.

## 13 November 2025

- Fixed computational issues in the error quantification pipeline that had been producing incorrect results.
- Added metrics computation script for systematic evaluation of estimation performance.
- Implemented altimetry estimate methods and integrated them into the parallel workflow.
- Set up plotting infrastructure for visualising overnight computation results.

## 14 November 2025

- Processed overnight parallel computation data and generated updated plots.
- Added ODT-to-SSH conversion in the forward model.
- Refactored single-value example and reorganised analysis files.

## 15 November 2025

- Major code refactoring session: built the `setup_altimetry_inversion_components` function, resolving variable naming conventions (`_op` and `_measure` suffixes) and formalising the physics of the RowLinearOperator structure with ODT entering the data error model.
- Developed the preconditioner approach — the key insight being that calling the same setup function at two different lmax values resolves the data space dimension mismatch that would otherwise make a naïve lower-resolution preconditioner unworkable.
- Added CG solver callbacks to monitor convergence iteration by iteration and worked out serialisation of pygeoinf objects (pickle vs dill). Integrated the preconditioner into the posterior solve call.
- Generated initial tutorial-style plotting code adapted for the altimetry setup rather than tide gauges.
- Renamed folders for clarity and restructured the inversion code.

## 18 November 2025

- Updated project dependencies and .gitignore configuration.
- Reorganised analysis code: renamed functions for clarity, converted notebooks to Python scripts, and added plot output generation.
- Added detailed data analysis scripts and additional error plots.
- Fixed forward model implementation and generated updated figures.

## 20 November 2025

- Added ternary error analysis plots for the Gaussian error model.
- Generated high-resolution ternary plots and explored different visualisation approaches.

## 21 November 2025

- Conducted parameter sweep regression analysis across seven parameters (ice GMSL std, ice length scale, net ice thickness change, ODT length scale, ODT std, altimetry error length scale, altimetry error amplitude) producing nearly 250,000 configurations.
- Linear regression showed that net_ice_thickness_change and altimetry_range dominate the error mean, whilst most other parameters become significant only through interaction terms. The error standard deviation is almost perfectly predicted by linear terms alone — a substantial quantitative result for the error quantification section.
- Added GMSL error calculations and generated associated analysis plots.
- Enhanced output documentation with new figures and model descriptions.

## 22 November 2025

- Implemented scaled version of the error analysis and added new dataset files.

## 25 November 2025

- Updated error quantification code and added new analysis plots.
- Started work on error distribution visualisation for the poster.

## 27 November 2025

- Tidied project structure: renamed files for clarity, improved function naming, and removed Bayesian inversion and poster figures from main branch (moved to separate branches).
- Configured Git LFS for tracking large PNG files.
- Began conference poster preparation with initial layout and figures.

## 29 November 2025

- Added additional figures for the poster and refined whitespace in plots.

## 30 November 2025

- Built plots comparing SSH and SLC sensitivity kernels — a physically important distinction since SSH is what altimetry actually measures, whilst SLC is what matters for coastal communities.
- Resolved the SSH/SLC sign discrepancy: the source was identified as originating from the SLC = N − U decomposition and which component of the fingerprint output was being projected onto.
- Prepared explainer figures showing the physical distinction between sea level change and sea surface height change.
- Converted units to millimetres and adjusted to more physically sensible ice loss scenarios.
- Added distribution plots and updated poster layout.

## 1 December 2025

- Generated combined distribution plots for the poster with consistent colour schemes.
- Finalised poster figures: GMSL plots, background sensitivity plots, and combined error distributions.
- Set transparent backgrounds as default for all figure output.
- Investigated sensitivity kernels for the forward model.

## 2 December 2025

- Continued poster refinement: updated figures, plot styling, and layout.
- Merged poster branch into main.

## 3 December 2025

- Added improved inset plot for ODT showing both pointwise (Dirac delta function) and distributed representations.
- Finalised conference poster with print-ready settings (bleed marks removed).

## 26 January 2026

- Resumed work on Bayesian inversion using pygeoinf: generated inversion demonstration plots.
- Began write-up planning, structuring the introduction and error quantification sections within the 7,500-word paper-style limit. Focus on integrating the deterministic analysis (ternary plots, latitude sweeps) with the analytical Gaussian operator approach into a coherent narrative.

## 27 January 2026

- Explored pygeoinf inversion functionality, experimenting with different prior and observation configurations.

## 30 January 2026

- Staged intermediate work for repository restructuring.

## 31 January 2026

- Cleaned and restructured the repository in preparation for the final development phase.

## 1 February 2026

- Staged intermediate computational results.

## 2 February 2026

- Updated major source mixing figures and refactored source code for improved modularity.

## 3 February 2026

- Implemented Gaussian error model work: added Gaussian error analysis for whole ice sheet configurations.
- Generated new plots and ran high-resolution computations across all Gaussian latitudes.
- Cleaned up deprecated code and reorganised plotting functions.

## 4 February 2026

- Added new analysis file and updated figure plotting code.

## 6 February 2026

- Implemented altimetry sampling module with whole ice sheet replication for comparison of error behaviour.
- Implemented the first complete Bayesian inversion pipeline.
- Added ocean coordinate helper functions and removed duplicated code.

## 7 February 2026

- Added uniform and spatially variable ODT models, characterising where uncertainties arise due to ODT variability.
- Implemented altimetry error model integration.
- Updated analysis for Gaussian error over major ice sheet sources.

## 9 February 2026

- Began work on a realistic inversion case using a rotationally variant error field.

## 10 February 2026

- Created ice sheet density profile for realistic ice thickness calculations.
- Added observational datasets for real-data inversions.
- Continued developing the inversion framework.

## 12 February 2026

- Parallelised the grid point computation for improved performance.

## 17 February 2026

- Added time series inversion capability, extending the framework to handle temporal sequences.

## 20 February 2026

- Added time series visualisation to analysis plots.
- Began initial report writing.

## 23 February 2026

- Implemented `from_samples` method for constructing Gaussian measures from empirical data.

## 24 February 2026

- Processed DUACS real data: loaded monthly SSH anomaly dataset with xarray, regridded to the pyslfp spherical harmonic grid, expanded to SH coefficients, and fitted a GaussianMeasure from the sample collection. This provides the empirical ocean dynamic topography (ODT) characterisation for the joint inversion prior.

## 25 February 2026

- Major class architecture session: designed `IceSheetChange` and `OceanDynamics` classes. Key design decisions include nested melt pattern classes (uniform vs thickness-weighted logistic activator with configurable parameters), the complement relationship between ice and firn weights, independent firn GMSL std (defaulting to 20% of ice GMSL std), and independent ice/firn priors since the inversion is what disentangles them.
- Established unified spatial pattern contract for OceanDynamics where all pattern types expose normalised [0,1] weights with amplitude residing on the class.
- Replaced standalone ice thickness functions with an `IceSheetChange` class and began converting ocean dynamics to a class-based architecture.
- Added fingerprint response and sea surface height computation to the ice thickness class.
- Created unit tests for the ice thickness class.
- Added placeholder subsubsections to the methods section and began writing mathematical content.
- Restructured the `src/` directory and moved file generators to `work/`.

## 26 February 2026

- Began structuring the methods section in earnest, with subsections for the mathematics, ice melt modelling, load operators, SSH, ocean dynamics, error quantification, and Bayesian inversion. The Richards curve activator function was already conceptually in place.
- Cleaned up repository structure and updated configuration.

