#!/bin/bash

#uv run 01_sea_surface_height_plots.py
echo Starting 2
uv run 02_major_sources_plots.py
echo Starting 3
uv run 03_ocean_dynamics_plots.py
echo Starting 4
uv run 04_ice_and_firn_plots.py
echo "Starting 5"
uv run 05_altimetry_sampling_plots.py
echo Starting 6
uv run 06_simple_inversion_plots.py
echo Starting 7
uv run 07_inversion_sensitivity_plots.py
echo Starting 9
uv run 09_truth_prior_cross_test_plots.py

echo Starting 10
uv run 10_summary_plots.py

echo Moving files
../../copy_report_pdfs.sh
