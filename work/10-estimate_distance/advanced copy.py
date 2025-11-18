# %%
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
    RowLinearOperator,
)
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

lmax = 256
fp = FingerPrint(
    lmax=lmax,
)
fp.set_state_from_ice_ng()
fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

load_space: HilbertSpace = fingerprint_operator.domain
response_space: HilbertSpace = fingerprint_operator.codomain
sea_surface_height_op: LinearOperator = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space: HilbertSpace = sea_surface_height_op.codomain

# %%
###### VARIABLES
ice_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.0004 / fp.length_scale  # in meters
net_ice_thickness_change = -100.0 / fp.length_scale  # in meters

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_standard_deviation = 0.005 / fp.length_scale  # in meters

altimetry_range = 66  # in degrees
altimetry_error_length_scale = 0.005 * fp.mean_sea_floor_radius
altimetry_error_amplitude = 0.001 / fp.length_scale  # in meters
######

# %%

### MEASURES
ice_thickness_change: GaussianMeasure
ice_thickness_change, _ = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=net_ice_thickness_change,
)
odt_change: GaussianMeasure
odt_change, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    standard_deviation=odt_standard_deviation,
)
measurement_error: GaussianMeasure = (
    measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        1.5,
        altimetry_error_length_scale,
        altimetry_error_amplitude,
    )
)

# %%

### OPERATORS

GMSL_from_ice_op: LinearOperator = averaging_operator(
    load_space,
    [
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        * fp.length_scale
        / (fp.water_density * fp.ocean_area),
    ],
)

Altimetry_op: LinearOperator = averaging_operator(
    measurement_space,
    [
        (
            (
                _a := fp.ocean_projection(value=0)
                * fp.altimetry_projection(
                    latitude_min=-altimetry_range,
                    latitude_max=altimetry_range,
                    value=0,
                )
            )
            / fp.integrate(_a)
        ),
    ],
) @ RowLinearOperator(
    [
        measurement_space.identity_operator(),
        measurement_space.zero_operator(),
        measurement_space.zero_operator(),
    ],
)

Load_w_op: LinearOperator = sea_level_change_to_load_operator(
    fp,
    load_space,
)

Load_i_op: LinearOperator = ice_thickness_change_to_load_operator(
    fp,
    load_space,
)

combined_load_operator: LinearOperator = RowLinearOperator(
    [Load_i_op, Load_w_op, load_space.zero_operator()],
)

Fingerprint_ssh_op: LinearOperator = sea_surface_height_op @ fingerprint_operator

altimetry_error_operator: LinearOperator = RowLinearOperator(
    [
        measurement_space.zero_operator(),
        measurement_space.identity_operator(),
        measurement_space.identity_operator(),
    ],
)


total_measure = GaussianMeasure.from_direct_sum(
    [
        ice_thickness_change,
        odt_change,
        measurement_error,
    ],
)


estimation_operator: LinearOperator = Altimetry_op @ (
    (Fingerprint_ssh_op @ combined_load_operator) + altimetry_error_operator
)

estimation_measure = total_measure.affine_mapping(
    operator=estimation_operator,
)

true_measure = total_measure.affine_mapping(
    operator=estimation_operator,
) @ RowLinearOperator(
    [
        measurement_space.identity_operator(),
        measurement_space.zero_operator(),
        measurement_space.zero_operator(),
    ],
)

# %%

print(estimation_measure.expectation)
print(true_measure.expectation)
