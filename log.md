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

