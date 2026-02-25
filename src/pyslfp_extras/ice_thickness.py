from typing import Callable

import numpy as np
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    spatial_mutliplication_operator,
)

from pygeoinf_extras import standard_dev
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)


def _ice_melt_activator(x, x_min, x_max):
    # Standardize input: 0 at min thickness, 1 at max thickness
    _x = (x - x_min) / (x_max - x_min)

    # Parameters for a clean 0-to-1 probability curve
    a = 0.1  # Lower asymptote (Thick ice = 0 probability)
    k = 1.0  # Upper asymptote (Thin ice = 1 probability)
    b = 10.0  # Steepness
    m = 0.45  # Threshold (where the drop-off happens)
    nu = 0.75  # Asymmetry (adjusts how 'sharp' the turn is)

    # Note: We use (_x - M) to make probability drop as thickness increases
    _x = a + (k - a) / (1 + np.exp(b * (_x - m))) ** (
        1 / nu
    )
    return _x


def ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    gmsl_target_std: float = 0.001,  # 1 mm
    gmsl_target_mean: float = 0.0,
    spatial_melt: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for ice thickness change, normalized to a target GMSL standard deviation and mean.
    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping FingerPrint to a Sobolev space.
        length_scale: Length scale for the heat kernel.
        gmsl_target_std: Target standard deviation for GMSL.
        gmsl_target_mean: Target mean for GMSL (shifts the distribution). Defaults to 0.0
            for a zero-mean prior; set to a non-zero value only if you have an independent
            estimate of GMSL change that is not derived from the altimetry data being inverted.
        spatial_melt: If True, scale the measure by a function derived from ice thickness,
            giving higher pointwise std near ice margins.
    Returns:
        A GaussianMeasure for ice thickness change.
    """
    ## MEASURES AND OPS ##
    _load_space: Sobolev = finger_print_operator.domain
    _base_measure: GaussianMeasure = _load_space.point_value_scaled_heat_kernel_gaussian_measure(
        length_scale
    )
    melt_likelihood = 1.0
    if spatial_melt:
        melt_likelihood: SHGrid = (
            finger_print.ice_thickness.copy()
        )
        melt_likelihood.data = _ice_melt_activator(
            melt_likelihood.data,
            melt_likelihood.data.min(),
            melt_likelihood.data.max(),
        )
        f = melt_likelihood * finger_print.ice_projection(
            value=0
        )
        _base_measure = _base_measure.affine_mapping(
            operator=spatial_mutliplication_operator(
                f, _load_space
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
    if gmsl_target_mean != 0.0:
        _gmsl_per_unit = finger_print.integrate(
            -finger_print.ice_density
            * finger_print.one_minus_ocean_function
            * finger_print.ice_projection(value=0)
            * melt_likelihood
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
        _shift_vector = (
            _load_space.project_function(
                lambda _: _ice_shift_needed
            )
            * melt_likelihood
        )
    else:
        _shift_vector = None

    ## MEASURE ##
    ice_thickness_measure = _base_measure.affine_mapping(
        operator=_std_scale * _ice_projection_op,
        translation=_shift_vector,
    ).affine_mapping(operator=_ice_projection_op)
    return ice_thickness_measure


def _source_ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    source_projection_method: Callable,
    gmsl_target_std: float = 0.001,
    gmsl_target_mean: float = 0.0,
) -> GaussianMeasure:
    """Create a Gaussian measure for ice thickness change from a specific source region.

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping FingerPrint to a Sobolev space.
        length_scale: Length scale for the heat kernel.
        source_projection_method: A method on finger_print that returns a source-specific
            projection grid (e.g. finger_print.greenland_projection).
        gmsl_target_std: Target standard deviation for GMSL.
        gmsl_target_mean: Target mean for GMSL.

    Returns:
        A GaussianMeasure for ice thickness change restricted to the source region.
    """
    _load_space: Sobolev = finger_print_operator.domain
    _base_measure: GaussianMeasure = (
        _load_space.heat_kernel_gaussian_measure(
            length_scale
        )
    )

    _source_projection_grid = source_projection_method(
        value=0
    )
    _source_projection_op: LinearOperator = (
        spatial_mutliplication_operator(
            _source_projection_grid, _load_space
        )
    )

    _gmsl_op: LinearOperator = (
        gmsl_from_ice_thickness_operator(
            finger_print, finger_print_operator
        )
    )

    # STD SCALING
    _gmsl_std = standard_dev(
        _base_measure.affine_mapping(operator=_gmsl_op)
    )
    _std_scale = gmsl_target_std / _gmsl_std

    # SHIFT
    _gmsl_per_unit = finger_print.integrate(
        -finger_print.ice_density
        * finger_print.one_minus_ocean_function
        * _source_projection_grid
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

    # MEASURE
    _shift_vector = _load_space.project_function(
        lambda _: _ice_shift_needed
    )

    ice_thickness_measure = _base_measure.affine_mapping(
        operator=_std_scale * _source_projection_op,
        translation=_shift_vector,
    ).affine_mapping(operator=_source_projection_op)

    return ice_thickness_measure


def greenland_ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    gmsl_target_std: float = 0.001,
    gmsl_target_mean: float = 0.0,
) -> GaussianMeasure:
    """Create a Gaussian measure for Greenland ice thickness change."""
    return _source_ice_thickness_gaussian_measure(
        finger_print=finger_print,
        finger_print_operator=finger_print_operator,
        length_scale=length_scale,
        source_projection_method=finger_print.greenland_projection,
        gmsl_target_std=gmsl_target_std,
        gmsl_target_mean=gmsl_target_mean,
    )


def west_antarctic_ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    gmsl_target_std: float = 0.001,
    gmsl_target_mean: float = 0.0,
) -> GaussianMeasure:
    """Create a Gaussian measure for West Antarctic ice thickness change."""
    return _source_ice_thickness_gaussian_measure(
        finger_print=finger_print,
        finger_print_operator=finger_print_operator,
        length_scale=length_scale,
        source_projection_method=finger_print.west_antarctic_projection,
        gmsl_target_std=gmsl_target_std,
        gmsl_target_mean=gmsl_target_mean,
    )


def east_antarctic_ice_thickness_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    length_scale: float,
    gmsl_target_std: float = 0.001,
    gmsl_target_mean: float = 0.0,
) -> GaussianMeasure:
    """Create a Gaussian measure for East Antarctic ice thickness change."""
    return _source_ice_thickness_gaussian_measure(
        finger_print=finger_print,
        finger_print_operator=finger_print_operator,
        length_scale=length_scale,
        source_projection_method=finger_print.east_antarctic_projection,
        gmsl_target_std=gmsl_target_std,
        gmsl_target_mean=gmsl_target_mean,
    )
