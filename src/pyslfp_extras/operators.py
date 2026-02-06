import numpy as np
from pygeoinf import HilbertSpace, LinearOperator
from pyslfp import FingerPrint


def ocean_point_evaluation_operator(
    finger_print: FingerPrint,
    measurement_space: HilbertSpace,
    point_degree_spacing: float = 5.0,
) -> LinearOperator:
    """
    Constructs a linear operator that evaluates the ocean surface height at
    specific points on the Earth's surface, as determined by the provided
    `FingerPrint`. The operator is designed to only evaluate points that are
    classified as ocean according to the `FingerPrint`'s ocean and altimetry
    projections.
    """
    mask = (
        finger_print.ocean_projection(value=0)
        * finger_print.altimetry_projection(value=0)
    ).to_array()
    nlat, nlon = mask.shape

    mask_lats = np.linspace(90, -90, nlat)
    mask_lons = (
        np.linspace(0, 360, nlon, endpoint=True)
        if nlon > nlat
        else np.linspace(0, 360, nlon, endpoint=False)
    )

    target_lats = np.arange(
        90,
        -90 - point_degree_spacing,
        -point_degree_spacing,
    )
    target_lons = np.arange(0, 360, point_degree_spacing)

    ocean_coords = []
    for lat in target_lats:
        for lon in target_lons:
            mask_lat_idx = np.argmin(
                np.abs(mask_lats - lat)
            )
            mask_lon_idx = np.argmin(
                np.abs(mask_lons - lon)
            )

            if mask[mask_lat_idx, mask_lon_idx] == 1:
                ocean_coords.append([lat, lon])

    _op = measurement_space.point_evaluation_operator(
        ocean_coords
    )
    return _op
