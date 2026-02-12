import numpy as np
from pygeoinf import HilbertSpace, LinearOperator
from pyslfp import FingerPrint

from pyslfp_extras.helpers import (
    get_ocean_point_coordinates,
)


def ocean_point_evaluation_operator(
    finger_print: FingerPrint,
    measurement_space: HilbertSpace,
    point_degree_spacing: float = 5.0,
    altimetry_latitude_range: float = 66.0,
    parallel_workers: None | int = None,
) -> LinearOperator:
    """
    Constructs a linear operator that evaluates the ocean surface height at
    specific points on the Earth's surface, as determined by the provided
    `FingerPrint`. The operator is designed to only evaluate points that are
    classified as ocean according to the `FingerPrint`'s ocean and altimetry
    projections.
    """
    ocean_coords = get_ocean_point_coordinates(
        finger_print,
        point_degree_spacing=point_degree_spacing,
        altimetry_latitude_range=altimetry_latitude_range,
        parallel_workers=parallel_workers,
    )

    ocean_coords = list(zip(*ocean_coords))

    _op = measurement_space.point_evaluation_operator(
        ocean_coords
    )
    return _op
