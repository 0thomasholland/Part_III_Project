# %%
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import GaussianMeasure, HilbertSpace, LinearOperator
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
)

Load_w_op: LinearOperator = sea_level_change_to_load_operator(
    fp,
    load_space,
)

Load_i_op: LinearOperator = ice_thickness_change_to_load_operator(
    fp,
    load_space,
)

Fingerprint_ssh_op: LinearOperator = sea_surface_height_op @ fingerprint_operator

# %%

error_expectation: float = (
    (Altimetry_op @ Fingerprint_ssh_op @ Load_i_op)(
        ice_thickness_change.expectation,
    )
    - GMSL_from_ice_op(ice_thickness_change.expectation)
)[0] * fp.length_scale

# %%

error_covariance = (
    Altimetry_op
    @ Fingerprint_ssh_op
    @ Load_i_op
    @ ice_thickness_change.covariance
    @ Load_i_op.adjoint
    @ Fingerprint_ssh_op.adjoint
    @ Altimetry_op.adjoint
    + Altimetry_op
    @ Fingerprint_ssh_op
    @ Load_w_op
    @ odt_change.covariance
    @ Load_w_op.adjoint
    @ Fingerprint_ssh_op.adjoint
    @ Altimetry_op.adjoint
    + Altimetry_op @ odt_change.covariance @ Altimetry_op.adjoint
    + Altimetry_op @ measurement_error.covariance @ Altimetry_op.adjoint
    - GMSL_from_ice_op @ ice_thickness_change.covariance @ GMSL_from_ice_op.adjoint
)

error_measure = (
    ice_thickness_change.affine_mapping(
        operator=Altimetry_op @ Fingerprint_ssh_op @ Load_i_op - GMSL_from_ice_op,
    )
    + odt_change.affine_mapping(
        operator=Altimetry_op @ Fingerprint_ssh_op @ Load_w_op,
    )
    + odt_change.affine_mapping(operator=Altimetry_op)
    + measurement_error.affine_mapping(operator=Altimetry_op)
)


# %%

print(error_measure.expectation)
print(error_expectation)

print(error_measure.covariance.matrix(dense=True))
print(error_covariance.matrix(dense=True))
# %%

error_covariance = (error_covariance.matrix(dense=True)[0, 0]) * (fp.length_scale**2)
print(error_covariance)

# %%

error_std_dev: float = np.sqrt(error_covariance)

print(error_std_dev)
# %%
print("Error expectation:", error_expectation)
# print("Error std dev:", error_std_dev)
#

# %%

true_gmsl_expectation: float = (
    GMSL_from_ice_op(ice_thickness_change.expectation)[0] * fp.length_scale
)
true_gmsl_std: float = (
    np.sqrt(
        (
            GMSL_from_ice_op
            @ ice_thickness_change.covariance
            @ GMSL_from_ice_op.adjoint
        ).matrix(
            dense=True,
        )[0, 0],
    )
    * fp.length_scale
)

print("True GMSL expectation:", true_gmsl_expectation)
print("True GMSL std dev:", true_gmsl_std)

adjusted_error_expectation = error_expectation + true_gmsl_expectation
# %%
# plot these distributions on the same plot
x_min = min(
    adjusted_error_expectation - 4 * np.sqrt(error_covariance),
    true_gmsl_expectation - 4 * true_gmsl_std,
)
x_max = max(
    adjusted_error_expectation + 4 * np.sqrt(error_covariance),
    true_gmsl_expectation + 4 * true_gmsl_std,
)
x = np.linspace(x_min, x_max, 1000)
fig, ax = plt.subplots(figsize=(10, 6))
y_error = norm.pdf(
    x,
    adjusted_error_expectation,
    np.sqrt(error_covariance),
)
y_true = norm.pdf(
    x,
    true_gmsl_expectation,
    true_gmsl_std,
)
ax.plot(
    x,
    y_error,
    label=f"Estimation Error Distribution (μ={error_expectation:.2e}, σ={error_std_dev:.2e})",
)
ax.plot(
    x,
    y_true,
    label=f"True GMSL Distribution (μ={true_gmsl_expectation:.2e}, σ={true_gmsl_std:.2e})",
)
ax.set_title(
    f"Comparison of Estimation Error and True GMSL Distributions\nIce Thickness Change {net_ice_thickness_change * fp.length_scale:.2e} m; ODT Std Dev {odt_standard_deviation * fp.length_scale:.2e} m;\nAltimetry Error Std Dev {altimetry_error_amplitude * fp.length_scale:.2e} m; Altimetry Lat Range {altimetry_range}°",
)
ax.set_xlabel("GMSL Change (m)")
ax.set_ylabel("Probability Density")
ax.legend()
plt.show()

# %%
print("Expected GMSL Change:", true_gmsl_expectation)
print(
    "Estimated GMSL Change:",
    true_gmsl_expectation + error_expectation,
)
print("Estimation GMSL Change error:", error_expectation)
