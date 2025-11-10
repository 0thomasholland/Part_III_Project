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

