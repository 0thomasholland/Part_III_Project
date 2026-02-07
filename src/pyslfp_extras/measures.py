from typing import Callable

import numpy as np
from pandas import read_iceberg
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ocean_projection_operator,
    remove_ocean_average_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

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


def altimetry_error_gaussian_measure(
    measurement_space: Sobolev | HilbertSpace,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    finger_print: FingerPrint | None = None,
    altimetry_operator: LinearOperator | None = None,
) -> GaussianMeasure:
    """Create a Gaussian measure for altimetry observation errors on the SSH measurement space.

    Args:
        measurement_space: The SSH space (e.g. from sea_surface_height_operator.codomain).
        order: Sobolev kernel order. Default 1.5.
        length_scale: Correlation length scale. Default 0.005 * fp.mean_sea_floor_radius.
        amplitude: Pointwise standard deviation. Default 0.0001 / fp.length_scale.
        finger_print: FingerPrint instance used for default parameter scaling.
            Required if length_scale or amplitude are not provided.
        altimetry_operator: Optional operator for spatial masking (e.g. latitude bounds).

    Returns:
        A GaussianMeasure for altimetry observation errors.
    """
    if length_scale is None or amplitude is None:
        if finger_print is None:
            raise ValueError(
                "finger_print is required when length_scale or amplitude are not provided."
            )
        if length_scale is None:
            length_scale = (
                0.005 * finger_print.mean_sea_floor_radius
            )
        if amplitude is None:
            amplitude = 0.0001 / finger_print.length_scale

    base_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, length_scale, amplitude
    )

    if altimetry_operator is not None:
        return base_measure.affine_mapping(
            operator=altimetry_operator
        )

    return base_measure


def odt_variability_field(
    finger_print: FingerPrint,
    base_amplitude: float = 0.8,
    current_amplitude: float = 10,
    tropical_amplitude: float = 7,
) -> SHGrid:
    """Generate a synthetic spatial variability field for ODT based on observed patterns.

    Creates a field with variability values (in meters) reflecting where ODT variance
    is expected to be high (western boundary currents, ACC) vs low (deep basins).

    Args:
        finger_print: The FingerPrint instance (used for grid creation).
        current_amplitude: Peak amplitude in major current systems, multiplier, default 1.3
        tropical_amplitude: Amplitude in tropical regions, multiplier, default 1.15

    Returns:
        An SHGrid with variability values in meters.
    """

    def _gaussian_blob(
        lat_grid, lon_grid, lat0, lon0, lat_width, lon_width
    ):
        """Smooth Gaussian-like spatial weight centered at (lat0, lon0)."""
        dlat = lat_grid - lat0
        dlon = lon_grid - lon0
        # Wrap longitude differences to [-180, 180]
        dlon = np.mod(dlon + 180, 360) - 180
        return np.exp(
            -0.5
            * (
                (dlat / lat_width) ** 2
                + (dlon / lon_width) ** 2
            )
        )

    def _latitude_band(lat_grid, lat_center, lat_width):
        """Smooth zonal band centered at lat_center."""
        return np.exp(
            -0.5
            * ((lat_grid - lat_center) / lat_width) ** 2
        )

    grid = finger_print.zero_grid()
    lats = finger_print.lats()
    lons = finger_print.lons()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Start with base amplitude everywhere
    field = np.full_like(lat_grid, base_amplitude)

    current_extra = current_amplitude - base_amplitude

    # Gulf Stream
    field += (
        current_extra
        * 0.75
        * _gaussian_blob(lat_grid, lon_grid, 35, 305, 8, 25)
    )
    field += (
        current_extra
        * 0.5
        * _gaussian_blob(lat_grid, lon_grid, 40, 330, 8, 10)
    )
    field += (
        current_extra
        * 0.75
        * _gaussian_blob(lat_grid, lon_grid, 45, 350, 8, 10)
    )

    # Kuroshio
    field += current_extra * _gaussian_blob(
        lat_grid, lon_grid, 35, 155, 8, 25
    )

    # Agulhas
    field += current_extra * _gaussian_blob(
        lat_grid, lon_grid, -35, 35, 12, 25
    )

    # East Australian Current
    field += current_extra * _gaussian_blob(
        lat_grid, lon_grid, -30, 155, 8, 10
    )

    acc_amplitude = 0.5 * (
        current_amplitude + tropical_amplitude
    )
    acc_extra = acc_amplitude - base_amplitude
    field += acc_extra * _latitude_band(lat_grid, -60, 10)

    tropical_extra = tropical_amplitude - base_amplitude

    # Equatorial Pacific
    field += tropical_extra * _gaussian_blob(
        lat_grid, lon_grid, -10, 230, 15, 50
    )
    field += tropical_extra * _gaussian_blob(
        lat_grid, lon_grid, -10, 230, 10, 40
    )

    # Indonesian Throughflow
    field += tropical_extra * _gaussian_blob(
        lat_grid, lon_grid, -5, 120, 8, 15
    )

    # Caribbean
    field += tropical_extra * _gaussian_blob(
        lat_grid, lon_grid, 15, 280, 8, 15
    )

    # Mask to ocean only
    ocean_mask = finger_print.ocean_projection(
        value=0
    ).to_array()
    field = field * ocean_mask

    grid.data[:, :] = field
    return grid


def odt_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    order: float = 1.5,
    length_scale: float | None = 10000,  # 10km
    amplitude: float | None = 0.003,
    use_spatial_variability: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for Ocean Dynamic Topography as a height field on the load space.

    The measure is projected onto the ocean and constrained to have zero ocean average.
    Optional spatially-varying amplitude can be applied.

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        order: Sobolev kernel order. Default 1.5.
        length_scale: Correlation length scale. Default 0.1 * fp.mean_sea_floor_radius.
        amplitude: Pointwise standard deviation. Default 0.001 / fp.length_scale.
        spatial_variability: An SHGrid or callable providing spatially-varying amplitude.
            Applied via spatial multiplication after ocean projection.
        use_synthetic_variability: If True and spatial_variability is None, generate
            the synthetic variability pattern from odt_variability_field().

    Returns:
        A GaussianMeasure for ODT on the load space.
    """
    load_space: Sobolev = finger_print_operator.domain

    base_measure = load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, length_scale, amplitude
    )

    ocean_proj = ocean_projection_operator(
        finger_print, load_space
    )
    remove_avg = remove_ocean_average_operator(
        finger_print, load_space
    )

    if use_spatial_variability:
        spatial_variability = odt_variability_field(
            finger_print
        )
        spatial_op = spatial_mutliplication_operator(
            spatial_variability, load_space
        )
        combined_op = remove_avg @ spatial_op @ ocean_proj
    else:
        combined_op = remove_avg @ ocean_proj

    return base_measure.affine_mapping(operator=combined_op)


def odt_fingerprint_ssh_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    odt_measure: GaussianMeasure | None = None,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    use_synthetic_variability: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for the SSH contribution from ODT's fingerprint response.

    This represents only the gravitationally-induced sea surface height change caused
    by the ocean load redistribution from ODT, not the ODT height itself.

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        odt_measure: Pre-built ODT measure. If None, one is created with the given parameters.
        order: Sobolev kernel order for ODT measure creation. Default 1.5.
        length_scale: Correlation length scale for ODT measure creation.
        amplitude: Pointwise standard deviation for ODT measure creation.
        use_synthetic_variability: Whether to use synthetic spatial variability for ODT.

    Returns:
        A GaussianMeasure on the SSH space representing the fingerprint response to ODT.
    """
    if odt_measure is None:
        odt_measure = odt_gaussian_measure(
            finger_print,
            finger_print_operator,
            order=order,
            length_scale=length_scale,
            amplitude=amplitude,
            use_synthetic_variability=use_synthetic_variability,
        )

    load_space: Sobolev = finger_print_operator.domain
    load_op = sea_level_change_to_load_operator(
        finger_print, load_space
    )
    ssh_op = sea_surface_height_operator(
        finger_print, finger_print_operator.codomain
    )

    fingerprint_ssh_op = (
        ssh_op @ finger_print_operator @ load_op
    )
    return odt_measure.affine_mapping(
        operator=fingerprint_ssh_op
    )


def odt_total_ssh_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    odt_measure: GaussianMeasure | None = None,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    use_synthetic_variability: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for the total SSH contribution from ODT.

    This is the sum of the ODT height field itself and its gravitationally-induced
    fingerprint response in SSH (see sea_surface_height_operator docstring).

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        odt_measure: Pre-built ODT measure. If None, one is created with the given parameters.
        order: Sobolev kernel order for ODT measure creation. Default 1.5.
        length_scale: Correlation length scale for ODT measure creation.
        amplitude: Pointwise standard deviation for ODT measure creation.
        use_synthetic_variability: Whether to use synthetic spatial variability for ODT.

    Returns:
        A GaussianMeasure on the SSH space representing the total ODT contribution.
    """
    if odt_measure is None:
        odt_measure = odt_gaussian_measure(
            finger_print,
            finger_print_operator,
            order=order,
            length_scale=length_scale,
            amplitude=amplitude,
            use_synthetic_variability=use_synthetic_variability,
        )

    load_space: Sobolev = finger_print_operator.domain
    load_op = sea_level_change_to_load_operator(
        finger_print, load_space
    )
    ssh_op = sea_surface_height_operator(
        finger_print, finger_print_operator.codomain
    )

    fingerprint_ssh_op = (
        ssh_op @ finger_print_operator @ load_op
    )
    identity_op = load_space.identity_operator()
    total_op = fingerprint_ssh_op + identity_op
    return odt_measure.affine_mapping(operator=total_op)
