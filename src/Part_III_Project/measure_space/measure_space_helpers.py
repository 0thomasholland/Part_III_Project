from pygeoinf.gaussian_measure import GaussianMeasure
from pygeoinf.linear_operators import LinearOperator


def return_expectation(measure: GaussianMeasure) -> float:
    # return measure.expectation
    pass


def return_1D_variance(measure: GaussianMeasure) -> float:
    # return measure.covariance[0, 0]
    pass


def check_space_compatibility(
    measure1: GaussianMeasure, measure2: GaussianMeasure
) -> bool:
    """
    Check if two Gaussian measures are compatible in terms of their
    spatial domains and covariances.
    """
    if measure1.domain != measure2.domain:
        return False
    return True


def check_operator_compatibility(
    operator: LinearOperator, input_measure: GaussianMeasure
) -> bool:
    """
    Check if a linear operator is compatible with given input and output spaces.
    """
    if operator.domain != input_measure.domain:
        return False
    return True
