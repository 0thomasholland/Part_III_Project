from typing import Literal

from numpy import ndarray
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import (
    Lebesgue,
    Sobolev,
)
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
)


def gmsl_from_ice_thickness_operator(
    load_space: Lebesgue | Sobolev | HilbertSpace,
    fp: FingerPrint,
) -> LinearOperator:
    _op: LinearOperator = averaging_operator(
        load_space,
        [
            -fp.ice_density
            * fp.one_minus_ocean_function
            * fp.ice_projection(value=0)
            * fp.length_scale
            / (fp.water_density * fp.ocean_area),
        ],
    )
    return _op


def gmsl_from_ice_load_operator(
    load_space: Lebesgue | Sobolev | HilbertSpace,
    fp: FingerPrint,
) -> LinearOperator:
    _thickness_to_load_op: LinearOperator = (
        ice_thickness_change_to_load_operator(
            finger_print=fp, load_space=load_space
        )
    )
    _op: LinearOperator = (
        gmsl_from_ice_load_operator @ _thickness_to_load_op
    )
    return _op


def altimetry_gmsl(
    ssh: SHGrid,
    fp: FingerPrint,
    latitude: float = 66.0,
) -> float:
    _alt_projection = fp.altimetry_projection(
        latitude_max=latitude,
        latitude_min=-latitude,
        value=0,
    )
    _alt_projection_integral = fp.integrate(_alt_projection)
    _alt_weighting_func = (
        _alt_projection / _alt_projection_integral
    )
    _estimated_gmsl: float = fp.integrate(
        _alt_weighting_func * ssh
    )
    return _estimated_gmsl


def gmsl_error(
    true_gmsl: float | ndarray | GaussianMeasure,
    estimated_gmsl: float | ndarray | GaussianMeasure,
    error_type: Literal["numeric", "relative"] = "relative",
) -> float | ndarray | GaussianMeasure:
    if isinstance(true_gmsl, float) and isinstance(
        estimated_gmsl, float
    ):
        error: float = estimated_gmsl - true_gmsl
        return (
            error / true_gmsl
            if error_type == "relative"
            else error
        )

    if isinstance(true_gmsl, ndarray) and isinstance(
        estimated_gmsl, ndarray
    ):
        error: ndarray = estimated_gmsl - true_gmsl
        return (
            error / true_gmsl
            if error_type == "relative"
            else error
        )

    if isinstance(
        true_gmsl, GaussianMeasure
    ) and isinstance(estimated_gmsl, GaussianMeasure):
        error_measure: GaussianMeasure = (
            estimated_gmsl - true_gmsl
        )
        if error_type == "relative":
            raise NotImplementedError(
                "Relative error not implemented for GaussianMeasure."
            )
        return error_measure

    raise TypeError(
        f"Incompatible types: true_gmsl ({type(true_gmsl).__name__}) and "
        f"estimated_gmsl ({type(estimated_gmsl).__name__}) must be the same type."
    )
