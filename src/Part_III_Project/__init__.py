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

# Backward compatibility aliases (new names for clarity)
SeaSurfaceFingerPrint = SeaSurfaceHeightFingerPrint
sea_surface_height_change = compute_sea_surface_height_change
create_ice_band = create_ice_load_latitude_band
create_ice_thickness_measures = ice_thickness_change_measures
create_ocean_dynamic_topography_measures = (
    ocean_dynamic_topography_measures
)
create_combined_load_measure = load_measure
create_gmsl_measure = get_gmsl_measure
create_altimetry_gmsl_measure = get_altimetry_gmsl_measure
create_sea_level_change_measure = sea_level_change_measure
create_sea_surface_height_measures = sea_surface_height_measure
create_sensor_error_measure = sensor_error_measure
extract_measure_statistics = get_stats_from_measure
plot_gaussian_measures = plot_measure

__all__ = [
    # Core operators
    "SeaSurfaceHeightFingerPrint",
    "compute_sea_surface_height_change",
    # Ice loads
    "create_ice_load_latitude_band",
    # Plotting
    "plot_ternary_heatmap",
    "plot_ternary_heatmap_subplots",
    # Gaussian measures (original names)
    "ice_thickness_change_measures",
    "ocean_dynamic_topography_measures",
    "load_measure",
    "get_gmsl_measure",
    "get_altimetry_gmsl_measure",
    "sea_level_change_measure",
    "sea_surface_height_measure",
    "sensor_error_measure",
    # Measure utilities
    "get_stats_from_measure",
    "plot_measure",
    # Visualization
    "plot_gaussian_measure_distribution",
    "plot_gmsl_comparison",
    # Analysis
    "kullback_leibler_divergence",
    "mean_squared_error",
    "cohens_d_effect_size",
    "wasserstein_distance",
    # Utilities
    "compute_altimetry_weighting_function",
    "compute_ocean_altimetry_weighting_function",
    "compute_relative_error",
    "extract_gmsl_statistics",
    # New clearer names (aliases)
    "create_ice_band",
    "create_ice_thickness_measures",
    "create_ocean_dynamic_topography_measures",
    "create_combined_load_measure",
    "create_gmsl_measure",
    "create_altimetry_gmsl_measure",
    "create_sea_level_change_measure",
    "create_sea_surface_height_measures",
    "create_sensor_error_measure",
    "extract_measure_statistics",
    "plot_gaussian_measures",
    # Backward compatibility
    "SeaSurfaceFingerPrint",
    "sea_surface_height_change",
]
