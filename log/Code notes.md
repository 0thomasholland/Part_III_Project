
# Coding work

## Week 1 (Oct 10-13, 2025) — Project Setup & Sea Level Conversion
- **Project initialisation**: Poetry configuration and dependency management
- **CI/CD setup**: Created notebook-to-PDF workflow for automatic documentation
- **Google Docs sync**: Built automatic sync for progress tracking
- **Core library started**: 
  - Sea level to sea surface height conversion functions
  - Extended `FingerPrint` class with `SeaSurfaceFingerPrint` class
- **First notebook**: SLc and SSHc plots

## Week 2 (Oct 14-20, 2025) — Traditional Methods & Error Analysis
- **New modules created**:
  - Ice load generation functions
  - Plotting utilities
- **Traditional method error analysis**:
  - Main traditional methods implementation
  - Ternary plots for ice sheet mixture error analysis
  - 50-70° satellite coverage analysis
  - Latitude band vs error relationship
- **Parallel processing**: Added parallel error function calculation
- **Unit tests**: Created tests for sea surface height change (later removed)
- **Report setup**: Started LaTeX report structure
- **Error characterisation**: PDF characterisation, p-value plots, higher resolution runs

## Week 3 (Oct 21-25, 2025) — Random Fields & Code Organisation
- **Random fields exploration**: Worked through examples from supervisor
- **Code organisation**: Cleaned up project structure
- **Report progress**: Updated previous methods and frontmatter sections

## Week 4 (Oct 31 - Nov 7, 2025) — Gaussian Measures Framework
- **New measure space module**:
  - Gaussian measure implementations
  - Linear operators on measures
  - Utility functions
- **Key bug fixes**:
  - Fixed measures sampling issues
  - Sorted out non-dimensionalisation problems
  - Discovered and fixed scaling bug in GMSL calculations
- **Distribution mapping work**:
  - Single case GMSL estimation
  - Parallel parameter sweeps
  - Generated GMSL comparison plots at lmax 32, 64, 128, 256
- **Target GMSL**: Implemented target GMSL functionality
- **Parallel code**: Added parallel sweep capabilities

## Week 5 (Nov 8-10, 2025) — Error Space Exploration
- **Error space work**:
  - Parameter space exploration
  - Data analysis and corner plots
  - Error metrics calculations
- **Outputs**: Corner plots, error metrics vs inputs plots
- **Parallel distributions**: Merged parallel run distributions

## Week 6 (Nov 11-14, 2025) — Altimetry Methods & Bayesian Inversion Start
- **Altimetry estimates**: Added altimetry estimation methods
- **Bayesian inversion started**: Initial implementation
- **Minimum error examples**: Series of minimal examples progressing from basic → fingerprinted → SSH → GMSL, with and without shift corrections
- **ODT to SSH integration**: Added ocean dynamic topography handling
- **L_max convergence study**: Testing and error space data for different L_max values

## Week 7 (Nov 15-18, 2025) — Refactoring & Inversion Methods
- **Bayesian inversion development**:
  - Initial implementation
  - Cleaner refactored version
  - High sample runs (1000 runs at lmax 64)
- **Code cleanup**: Reformatted and tidied codebase

## Week 8 (Nov 20-27, 2025) — GMSL Error Quantification
- **GMSL error quantification**:
  - Error calculation via operator composition methods
  - Parallel error computation
  - Noise-aware error analysis
  - Visualization and statistics
- **Ternary error analysis**:
  - Ice sheet fraction error analysis
  - Generated high-resolution ternary plots with shift corrections
- **Parameter scaling study**:
  - Scaling analysis
  - Generated mosaic plots for ice change vs altimetry range


