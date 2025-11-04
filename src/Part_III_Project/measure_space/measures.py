from joblib import Parallel, delayed
import numpy as np
from numpy.typing import NDArray, ArrayLike
from pygeoinf.gaussian_measure import GaussianMeasure

from Part_III_Project.measure_space.operators import 


def sea_level_change_measure() -> GaussianMeasure:
    pass

def sea_surface_change_measure() -> GaussianMeasure:
    pass

def altimetry_observed_sea_surface_change_measure(
        altimetry_range: float || ArrayLike
) -> NDArray:
    if type(altimetry_range) is float:
        altimetry_range = np.array([altimetry_range])
    pass

def calculate_measures() -> tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure]:
    pass