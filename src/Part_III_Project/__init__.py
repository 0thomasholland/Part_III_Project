"""
Unified public imports for the library
"""

from Part_III_Project.deterministic_space.load_generators import create_ice_band
from Part_III_Project.deterministic_space.plot_methods import (
    plot_ternary_heatmap,
    plot_ternary_heatmap_subplots,
    # plot_ternary_heatmap_subplots_parallel,
)
from Part_III_Project.deterministic_space.sea_surface_height_change import (
    SeaSurfaceFingerPrint,
    sea_surface_height_change,
)
from Part_III_Project.measure_space.measure_helpers import (
    return_1D_variance,
    return_expectation,
)
from Part_III_Project.measure_space.measures import (
    altimetry_observed_sea_surface_change_measure,
    calculate_measures,
    sea_level_change_measure,
    sea_surface_change_measure,
)

__all__ = [
    "SeaSurfaceFingerPrint",
    "sea_surface_height_change",
    "plot_ternary_heatmap",
    "plot_ternary_heatmap_subplots",
    # "plot_ternary_heatmap_subplots_parallel",
    "create_ice_band",
    "sea_level_change_measure",
    "sea_surface_change_measure",
    "altimetry_observed_sea_surface_change_measure",
    "calculate_measures",
    "return_expectation",
    "return_1D_variance",
]
