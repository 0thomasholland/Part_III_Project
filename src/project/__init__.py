"""Part III Project - Sea level change analysis tools.

This package provides tools for analyzing sea level changes using
Gaussian measures and pyslfp FingerPrint objects.
"""

from project.factored_forward_operator import (
    build_factored_forward_operator,
)
from project.operators import (
    ice_thickness_to_slc_operator,
)
from project.plots import error_plot
from project.stats import (
    cohens_d_effect_size,
    kullback_leibler_divergence,
    mean_squared_error,
    wasserstein_distance,
)

__all__ = [
    "cohens_d_effect_size",
    "kullback_leibler_divergence",
    "wasserstein_distance",
    "mean_squared_error",
    "error_plot",
    "ice_thickness_to_slc_operator",
    "build_factored_forward_operator",
]
