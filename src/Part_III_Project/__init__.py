"""Part III Project - Sea level change analysis tools.

This package provides tools for analyzing sea level changes using
Gaussian measures and pyslfp FingerPrint objects.

Modules:
    analysis: Statistical distance metrics for distribution comparison
    gaussian_measures: Construct Gaussian measures for uncertainty quantification
    ice_loads: Generate ice load distributions
    measure_utilities: Helper functions for working with Gaussian measures
    plotting: Ternary plot visualization
    ssh_operators: Sea surface height operators
    utilities: Common utility functions
    visualization: Plot Gaussian measure distributions
"""

# Core SSH operators
# Analysis tools
from Part_III_Project.analysis import (
    cohens_d_effect_size,
    kullback_leibler_divergence,
    mean_squared_error,
    wasserstein_distance,
)

# Gaussian measures
from Part_III_Project.gaussian_measures import (
    get_altimetry_gmsl_measure,
    get_gmsl_measure,
    ice_thickness_change_measures,
    load_measure,
    ocean_dynamic_topography_measures,
    sea_level_change_measure,
    sea_surface_height_measure,
    sensor_error_measure,
)

# Ice load generation
from Part_III_Project.ice_loads import create_ice_load_latitude_band

# Measure utilities
from Part_III_Project.measure_utilities import (
    get_stats_from_measure,
    plot_measure,
)

# Ternary plotting
from Part_III_Project.plotting import (
    plot_ternary_heatmap,
    plot_ternary_heatmap_subplots,
)
from Part_III_Project.ssh_operators import (
    SeaSurfaceHeightFingerPrint,
    compute_sea_surface_height_change,
)

# Utilities
from Part_III_Project.utilities import (
    compute_altimetry_weighting_function,
    compute_ocean_altimetry_weighting_function,
    compute_relative_error,
    extract_gmsl_statistics,
)

# Visualization
from Part_III_Project.visualization import (
    plot_gaussian_measure_distribution,
    plot_gmsl_comparison,
)

__all__ = [
    "SeaSurfaceHeightFingerPrint",
    "cohens_d_effect_size",
    "compute_altimetry_weighting_function",
    "compute_ocean_altimetry_weighting_function",
    "compute_relative_error",
    "compute_sea_surface_height_change",
    "create_ice_load_latitude_band",
    "extract_gmsl_statistics",
    "get_altimetry_gmsl_measure",
    "get_gmsl_measure",
    "get_stats_from_measure",
    "ice_thickness_change_measures",
    "kullback_leibler_divergence",
    "load_measure",
    "mean_squared_error",
    "ocean_dynamic_topography_measures",
    "plot_gaussian_measure_distribution",
    "plot_gmsl_comparison",
    "plot_measure",
    "plot_ternary_heatmap",
    "plot_ternary_heatmap_subplots",
    "sea_level_change_measure",
    "sea_surface_height_measure",
    "sensor_error_measure",
    "wasserstein_distance",
]
