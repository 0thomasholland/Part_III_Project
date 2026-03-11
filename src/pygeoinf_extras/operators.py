import numpy as np
from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
    MatrixLinearOperator,
)


def point_averaging_operator(space) -> LinearOperator:
    _n_points = space.dim
    _op: MatrixLinearOperator = LinearOperator.from_matrix(
        EuclideanSpace(_n_points),
        EuclideanSpace(1),
        np.array([[1.0 / _n_points] * _n_points]),
    )
    return _op


def point_averaging_area_weighted_operator(
    space, latitudes: np.ndarray
) -> LinearOperator:
    """Area-weighted averaging operator for a set of points on the sphere.

    Computes a weighted mean of point values where each point is weighted
    by cos(latitude), the standard area correction for a regular lat/lon
    grid. This correctly accounts for the fact that grid cells shrink
    toward the poles, so a simple arithmetic mean would over-represent
    low-latitude points.

    The weights are normalised to sum to 1, so the operator returns a
    scalar estimate of the spatial mean:

        GMSL = sum_i w_i * SSH(x_i)
        where w_i = cos(lat_i) / sum_j cos(lat_j)

    This is equivalent to a rectangular-cell area weighting
    (integrating cos(lat) over a latitude band of width degree_spacing)
    for any regular grid that does not reach the poles, which is always
    the case within the altimetry coverage band.

    Parameters
    ----------
    space:
        The EuclideanSpace whose dimension equals the number of points.
        This is typically the codomain of a point_evaluation_operator.
    latitudes:
        1-D array of latitudes in degrees, in the same order as the
        points used to build the point_evaluation_operator.

    Returns
    -------
    LinearOperator
        Maps EuclideanSpace(n_points) -> EuclideanSpace(1).
    """
    _latitudes = np.asarray(latitudes, dtype=float)
    if _latitudes.shape != (space.dim,):
        raise ValueError(
            f"latitudes length {_latitudes.shape[0]} does not match "
            f"space dimension {space.dim}."
        )
    _cos_weights = np.cos(np.deg2rad(_latitudes))
    _cos_weights_sum = _cos_weights.sum()
    if _cos_weights_sum == 0:
        raise ValueError(
            "Sum of cos(latitude) weights is zero — check that latitudes "
            "are in degrees and that the point set is non-empty."
        )
    _normalised_weights = _cos_weights / _cos_weights_sum
    _op: MatrixLinearOperator = LinearOperator.from_matrix(
        EuclideanSpace(space.dim),
        EuclideanSpace(1),
        _normalised_weights[np.newaxis, :],
    )
    return _op
