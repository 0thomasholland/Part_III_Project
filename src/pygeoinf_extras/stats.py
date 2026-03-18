from pygeoinf import GaussianMeasure


def absolute_error() -> GaussianMeasure:
    pass


def numeric_error() -> None:
    pass


def relative_error() -> None:
    pass


def variance(
    measure: GaussianMeasure,
    /,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
) -> float:
    return measure.covariance.matrix(
        dense=True, parallel=parallel, n_jobs=n_jobs
    )[0, 0]


def expectation(measure: GaussianMeasure) -> float:
    return measure.expectation[0]


def standard_dev(
    measure: GaussianMeasure,
    /,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
) -> float:
    var = variance(
        measure, parallel=parallel, n_jobs=n_jobs
    )
    return var**0.5
