"""
Unified public imports for the library
"""

from Part_III_Project.plot_methods import (
    plot_ternary_heatmap,
    plot_ternary_heatmap_subplots,
    plot_ternary_heatmap_subplots_parallel,
)
from Part_III_Project.sea_surface_height import SeaSurfaceFingerPrint
from Part_III_Project.sea_surface_height_change import sea_surface_height_change

__all__ = [
    "SeaSurfaceFingerPrint",
    "sea_surface_height_change",
    "plot_ternary_heatmap",
    "plot_ternary_heatmap_subplots",
    "plot_ternary_heatmap_subplots_parallel",
]
