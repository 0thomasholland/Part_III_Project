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
