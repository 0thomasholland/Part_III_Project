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

