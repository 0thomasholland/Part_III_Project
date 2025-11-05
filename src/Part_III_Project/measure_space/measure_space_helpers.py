from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.linear_operators import LinearOperator


def get_stats_from_measure(
    measure: GaussianMeasure,
) -> tuple[float, float]:
    expectation = measure.expectation[0]
    variance = measure.covariance.matrix(dense=True)[0, 0]
    return expectation, variance
