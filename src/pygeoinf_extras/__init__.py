"""pygeoinf_extras

Extensions to pygeoinf.
"""

from pygeoinf_extras.plots import get_1D_stats, plot_measure
from pygeoinf_extras.stats import (
    absolute_error,
    expectation,
    numeric_error,
    relative_error,
    standard_dev,
    variance,
)

__all__ = [
    "plot_measure",
    "get_1D_stats",
    "absolute_error",
    "expectation",
    "numeric_error",
    "relative_error",
    "variance",
    "standard_dev",
]
