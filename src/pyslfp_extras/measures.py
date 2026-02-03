from pandas import read_iceberg
from pygeoinf import GaussianMeasure, LinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import FingerPrint, ice_projection_operator

from pygeoinf_extras import expectation, standard_dev
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)


def ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    gmsl_target_std: float = 0.001,  # 1 mm
    gmsl_target_mean: float = 0.0,
) -> GaussianMeasure:
    """Create a Gaussian measure for ice thickness change, normalized to a target GMSL standard deviation and mean.

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping FingerPrint to a Sobolev space.
        length_scale: Length scale for the heat kernel.
        gmsl_target_std: Target standard deviation for GMSL.
        gmsl_target_mean: Target mean for GMSL (shifts the distribution).

    Returns:
        A GaussianMeasure for ice thickness change.
    """

    ## MEASURES AND OPS ##
    _load_space: Sobolev = finger_print_operator.domain
    _base_measure: GaussianMeasure = (
        _load_space.heat_kernel_gaussian_measure(
            length_scale
        )
    )
    _ice_projection_op: LinearOperator = (
        ice_projection_operator(finger_print, _load_space)
    )
    _gmsl_op: LinearOperator = (
        gmsl_from_ice_thickness_operator(
            finger_print, finger_print_operator
        )
    )

    ## STD SCALING ##

    _gmsl_std = standard_dev(
        _base_measure.affine_mapping(operator=_gmsl_op)
    )

    _std_scale = gmsl_target_std / _gmsl_std

    ## SHIFT ##

    _gmsl_per_unit = finger_print.integrate(
        -finger_print.ice_density
        * finger_print.one_minus_ocean_function
        * finger_print.ice_projection(value=0)
        * finger_print.length_scale
        / (
            finger_print.water_density
            * finger_print.ocean_area
        )
    )

    _ice_shift_needed = (
        gmsl_target_mean / _gmsl_per_unit
        if _gmsl_per_unit != 0
        else 0.0
    )

    ## MEASURE ##

    _shift_vector = _load_space.project_function(
        lambda _: _ice_shift_needed
    )

    ice_thickness_measure = _base_measure.affine_mapping(
        operator=_std_scale * _ice_projection_op,
        translation=_shift_vector,
    ).affine_mapping(operator=_ice_projection_op)

    return ice_thickness_measure
