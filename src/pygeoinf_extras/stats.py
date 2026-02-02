from pygeoinf import GaussianMeasure


def absolute_error() -> GaussianMeasure:
    pass


def numeric_error() -> None:
    pass


def relative_error() -> None:
    pass


def variance(measure: GaussianMeasure) -> float:
    return measure.covariance.matrix(dense=True)[0, 0]


def expectation(measure: GaussianMeasure) -> float:
    return measure.expectation[0]


def standard_dev(measure: GaussianMeasure) -> float:
    var = variance(measure)
    return var**0.5
