import numpy as np
import pyslfp as sl
from pygeoinf import GaussianMeasure, LinearOperator
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)


def old_ice_thickness_change_measures(
    fingerprint: FingerPrint = None,
    fingerprint_operator: LinearOperator = None,
    length_scale: float = 60,
    thickness_95_range: float = 100,
    net_thickness_change: float = 0,
) -> tuple[GaussianMeasure, GaussianMeasure]:
    """-> ice_thickness_measure, ice_thickness_load_measure
    Takes a length scale for the ice thickness changes and either a target GMSL std or a 95% range for the ice thickness changes to set the amplitude of the ocean dynamic topography measure.
    All parameters should be passed in non-dimensionalized form (already divided by fingerprint.length_scale).
    """
    ice_measure = fingerprint_operator.domain.point_value_scaled_heat_kernel_gaussian_measure(
        scale=length_scale,  # controls correlation length between nearby points
        amplitude=thickness_95_range
        / 3.92,  # the standard deviation of melt at each point
    )
    # project over ice only
    # remove mean shift now
    # adjust for any net thickness change specified
    if net_thickness_change != 0.0:
        shift_vector = ice_measure.domain.project_function(
            lambda point: net_thickness_change,
        )

        # shift_vector = np.zeros(fingerprint_operator.domain.dim)
        # shift_vector[0] = net_thickness_change
        # shift_vector = fingerprint_operator.domain.from_components(
        # shift_vector,
        # )
        ice_measure = ice_measure.affine_mapping(
            translation=shift_vector,
        )
    ice_measure = ice_measure.affine_mapping(
        operator=ice_projection_operator(
            fingerprint,
            fingerprint_operator.domain,
        ),
    )
    ice_load_measure = ice_measure.affine_mapping(
        operator=ice_thickness_change_to_load_operator(
            finger_print=fingerprint,
            load_space=fingerprint_operator.domain,
        ),
    )
    return ice_measure, ice_load_measure


def ice_thickness_change_measures(
    fingerprint: FingerPrint,
    fingerprint_operator: LinearOperator,
    length_scale: float,
    ice_gmsl_target_std: float,
    net_thickness_change: float,
) -> tuple[GaussianMeasure, GaussianMeasure]:
    """-> ice_thickness_measure, ice_thickness_load_measure
    Takes a length scale for the ice thickness changes and either a target GMSL std or a 95% range for the ice thickness changes to set the amplitude of the ocean dynamic topography measure.
    All parameters should be passed in non-dimensionalized form (already divided by fingerprint.length_scale).
    """
    initial_ice_thickness_measure = (
        fingerprint_operator.domain.heat_kernel_gaussian_measure(
            length_scale,
        )
    )
    ice_projection = sl.ice_projection_operator(
        fingerprint,
        fingerprint_operator.domain,
    )
    ice_thickness_measure = (
        initial_ice_thickness_measure.affine_mapping(
            operator=ice_projection,
        )
    )
    GMSL_weighting_function = (
        -fingerprint.ice_density
        * fingerprint.one_minus_ocean_function
        * fingerprint.ice_projection(value=0)
        * fingerprint.length_scale
        / (fingerprint.water_density * fingerprint.ocean_area)
    )
    GMSL_operator = sl.averaging_operator(
        fingerprint_operator.domain,
        [GMSL_weighting_function],
    )
    GMSL_measure = ice_thickness_measure.affine_mapping(
        operator=GMSL_operator,
    )
    GMSL_variance = GMSL_measure.covariance.matrix(dense=True)[0, 0]
    GMSL_std = np.sqrt(GMSL_variance)

    # Normalise the ice load thickness measure and then recompute the load measure
    ice_thickness_measure *= ice_gmsl_target_std / GMSL_std
    if net_thickness_change != 0.0:
        shift_vector = ice_thickness_measure.domain.project_function(
            lambda point: net_thickness_change,
        )
        ice_thickness_measure = ice_thickness_measure.affine_mapping(
            translation=shift_vector,
        )
    ice_thickness_measure = ice_thickness_measure.affine_mapping(
        operator=ice_projection_operator(
            fingerprint,
            fingerprint_operator.domain,
        ),
    )
    ice_load_measure = ice_thickness_measure.affine_mapping(
        operator=ice_thickness_change_to_load_operator(
            finger_print=fingerprint,
            load_space=fingerprint_operator.domain,
        ),
    )
    return ice_thickness_measure, ice_load_measure


def ocean_dynamic_topography_measures(
    fingerprint: FingerPrint = None,
    fingerprint_operator: LinearOperator = None,
    length_scale: float = 60,
    amplitude_95_range: float = 0.001,
) -> tuple[GaussianMeasure, GaussianMeasure]:
    """-> ODT_measure, ODT_load_measure
    All parameters should be passed in non-dimensionalized form (already divided by fingerprint.length_scale).
    """
    _initial_odt_measure = fingerprint_operator.domain.point_value_scaled_sobolev_kernel_gaussian_measure(
        order=1.5,
        scale=length_scale,
        amplitude=amplitude_95_range / 3.92,
    )
    # set the ODT to be zero mean over oceans
    _ocean_projection = sl.ocean_projection_operator(
        fingerprint,
        fingerprint_operator.domain,
    )
    _remove_ocean_average_operator = sl.remove_ocean_average_operator(
        fingerprint,
        fingerprint_operator.domain,
    )
    odt_measure = _initial_odt_measure.affine_mapping(
        operator=_remove_ocean_average_operator @ _ocean_projection,
    )
    # calculate the corresponding load measure
    odt_load_measure = odt_measure.affine_mapping(
        operator=sea_level_change_to_load_operator(
            finger_print=fingerprint,
            load_space=fingerprint_operator.domain,
        ),
    )
    return odt_measure, odt_load_measure


def load_measure(
    ice_thickness_load_measure: GaussianMeasure
    | tuple[GaussianMeasure, GaussianMeasure],
    odt_load_measure: GaussianMeasure
    | tuple[GaussianMeasure, GaussianMeasure],
) -> GaussianMeasure:
    """Returns a direct load measure"""
    if isinstance(ice_thickness_load_measure, tuple):
        ice_thickness_load_measure = ice_thickness_load_measure[1]
    if isinstance(odt_load_measure, tuple):
        odt_load_measure = odt_load_measure[1]

    if type(ice_thickness_load_measure) != type(odt_load_measure):
        raise ValueError(
            "Both inputs must be GaussianMeasures or tuples of GaussianMeasures",
        )
    if ice_thickness_load_measure.domain != odt_load_measure.domain:
        raise ValueError(
            "Both input measures must be defined on the same domain",
        )
    try:
        direct_load_measure = (
            ice_thickness_load_measure + odt_load_measure
        )
    except:
        raise ValueError(
            "Ya code broke boss tryna combine those two load measures to one",
        )
    return direct_load_measure


def sensor_error_measure(
    *,
    error_scale_std: float = 0.01,
    error_lengthscale: float = 1.0,
    fingerprint_operator: LinearOperator,
    fingerprint: FingerPrint,
) -> GaussianMeasure:
    """-> error measure over entire sphere surface
    All parameters should be passed in non-dimensionalized form (already divided by fingerprint.length_scale).
    """
    try:
        _error_measure = fingerprint_operator.codomain.point_value_scaled_sobolev_kernel_gaussian_measure(
            order=1.5,
            scale=error_lengthscale,
            amplitude=error_scale_std,
        )
    except:
        raise ValueError(
            "Failed to create error measure",
        )

    return _error_measure


def sea_level_change_measure(
    fingerprint_operator: LinearOperator,
    fingerprint: FingerPrint,
    load_measure: GaussianMeasure,
) -> GaussianMeasure:
    """Returns a single sea level change measure"""
    if load_measure.domain != fingerprint_operator.domain:
        raise ValueError(
            "load_measure and fingerprint_operator must be defined on the same domain",
        )
    response_measure = load_measure.affine_mapping(
        operator=fingerprint_operator,
    )

    response_space = response_measure.domain
    projection_operator = response_space.subspace_projection(
        0,
    )

    slc_measure = response_measure.affine_mapping(
        operator=projection_operator,
    )

    slc_measure = slc_measure.affine_mapping(
        operator=ocean_projection_operator(
            fingerprint,
            fingerprint_operator.domain,
        ),
    )

    return slc_measure


def sea_surface_height_measure(
    fingerprint: FingerPrint,
    fingerprint_operator: LinearOperator,
    load_measure: GaussianMeasure,
    odt_measure: GaussianMeasure
    | tuple[GaussianMeasure, GaussianMeasure]
    | None = None,
    noise_measure: GaussianMeasure | None = None,
) -> tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure]:
    """-> SSH, SSH+ODT, SSH+ODT+NOISE
    takes in a load measure and optionally an ODT measure and optional noise measure to return three sea surface height measures.
    If no ODT measure is provided, the SSH+ODT measure is just the SSH measure.
    If no noise measure is provided, the SSH+ODT+NOISE measure is just the SSH+ODT measure.
    """
    if isinstance(odt_measure, tuple):
        odt_measure = odt_measure[0]
    try:
        ssh_measure = load_measure.affine_mapping(
            operator=sea_surface_height_operator(
                fingerprint,
                fingerprint_operator.codomain,
            )
            @ fingerprint_operator,
        )
    except:
        raise ValueError(
            "Failed to create SSH measure",
        )
    # Create the SSH+ODT measure
    ssh_odt_measure = ssh_measure
    if odt_measure is not None:
        # Check if odt_measure is passed as a tuple and extract the measure (the ODT signal itself)
        if isinstance(odt_measure, tuple):
            _odt_measure_to_add = odt_measure[0]
        else:
            _odt_measure_to_add = odt_measure

        # Get the identity operator for the target domain (the observation space)
        target_identity_operator = (
            ssh_measure.domain.identity_operator()
        )

        try:
            # Map the ODT measure to the target observation space (ssh_measure.domain)
            # using the identity operator to match the domain object required for addition.
            odt_measure_obs_space = (
                _odt_measure_to_add.affine_mapping(
                    operator=target_identity_operator,
                )
            )
        except Exception as e:
            # This catch is for any issue with the affine mapping itself (like domain incompatibility)
            raise ValueError(
                f"Failed to map ODT measure to the target observation space. Original error: {e}",
            )

        try:
            # Now the domains should match for addition
            ssh_odt_measure = ssh_measure + odt_measure_obs_space
        except Exception as e:
            raise ValueError(
                f"Failed to create SSH+ODT measure via addition. Check domain compatibility. Original error: {e}",
            )

    ssh_odt_noise_measure = ssh_odt_measure
    if noise_measure is not None:
        try:
            ssh_odt_noise_measure = ssh_odt_measure + noise_measure
        except:
            raise ValueError(
                "Failed to create SSH+ODT+NOISE measure",
            )
    # extract the subspace corresponding to the sea surface height measures

    return ssh_measure, ssh_odt_measure, ssh_odt_noise_measure


def altimetry_measurements_measure(
    ssh_measure: tuple[
        GaussianMeasure,
        GaussianMeasure,
        GaussianMeasure,
    ],
    altimetry_range: float = 66,
) -> tuple[GaussianMeasure, GaussianMeasure, GaussianMeasure]:
    """-> SSH_alt range, SSH+ODT_alt range, SSH+ODT+NOISE_alt range"""


def get_gmsl_measure(
    measure: GaussianMeasure,
    fingerprint: FingerPrint,
) -> GaussianMeasure:
    weighting_function = (
        fingerprint.ocean_function / fingerprint.ocean_area
    )

    altimetry_estimate_operator = sl.averaging_operator(
        measure.domain,
        [weighting_function],
    )
    gmsl_measure = measure.affine_mapping(
        operator=altimetry_estimate_operator,
    )
    return gmsl_measure
