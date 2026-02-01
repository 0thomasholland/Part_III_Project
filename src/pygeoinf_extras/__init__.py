"""pygeoinf_extras

Extensions to pygeoinf.
"""

from pygeoinf_extas.stats import absolute_error, numeric_error, relative_error

from pygeoinf_extras.plots import get_1D_stats, plot_measure

__all__ = [
    "plot_measure",
    "get_1D_stats",
]
