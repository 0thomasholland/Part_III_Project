# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

mpl.rcParams["figure.dpi"] = 600

# %%
# Setup
lmax = 256
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()
fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain
sea_surface_height_op = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space = sea_surface_height_op.codomain
# %%
# Parameters
ice_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.004 / fp.length_scale
net_ice_thickness_change = -5.0 / fp.length_scale

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_standard_deviation = 0.08 / fp.length_scale

altimetry_range = 66
altimetry_error_length_scale = 0.005 * fp.mean_sea_floor_radius
altimetry_error_amplitude = 0.003 / fp.length_scale
# %%
# Measures
ice_thickness_change, _ = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=net_ice_thickness_change,
)

odt_change, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    standard_deviation=odt_standard_deviation,
)

measurement_error = (
    measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        1.5,
        altimetry_error_length_scale,
        altimetry_error_amplitude,
    )
)
# %%
# Operators
GMSL_from_ice_op = averaging_operator(
    load_space,
    [
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        * fp.length_scale
        / (fp.water_density * fp.ocean_area),
    ],
)

altimetry_weight = fp.ocean_projection(
    value=0,
) * fp.altimetry_projection(
    latitude_min=-altimetry_range,
    latitude_max=altimetry_range,
    value=0,
)
Altimetry_op = averaging_operator(
    measurement_space,
    [altimetry_weight / fp.integrate(altimetry_weight)],
)

Load_w_op = sea_level_change_to_load_operator(fp, load_space)
Load_i_op = ice_thickness_change_to_load_operator(fp, load_space)
Fingerprint_ssh_op = sea_surface_height_op @ fingerprint_operator
# %%
# Compute distributions using affine mappings
true_gmsl = ice_thickness_change.affine_mapping(
    operator=GMSL_from_ice_op,
)

estimated_gmsl = (
    ice_thickness_change.affine_mapping(
        operator=Altimetry_op @ Fingerprint_ssh_op @ Load_i_op,
    )
    + odt_change.affine_mapping(
        operator=Altimetry_op @ Fingerprint_ssh_op @ Load_w_op,
    )
    + odt_change.affine_mapping(operator=Altimetry_op)
    + measurement_error.affine_mapping(operator=Altimetry_op)
)

error = estimated_gmsl - true_gmsl
# %%
# Extract statistics (convert to meters)
true_mean = true_gmsl.expectation[0] * fp.length_scale
true_std = np.sqrt(true_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale

est_mean = estimated_gmsl.expectation[0] * fp.length_scale
est_std = np.sqrt(estimated_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale

error_mean = error.expectation[0] * fp.length_scale
error_std = np.sqrt(error.covariance.matrix(dense=True)[0, 0]) * fp.length_scale

print(f"Error expectation: {error_mean:.6f} m")
print(f"Error std dev: {error_std:.6f} m")
# %%
# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# GMSL distributions
x_range = 4
x_min = min(
    true_mean - x_range * true_std,
    est_mean - x_range * est_std,
)
x_max = max(
    true_mean + x_range * true_std,
    est_mean + x_range * est_std,
)
x = np.linspace(x_min, x_max, 1000)

ax1.plot(
    x,
    norm.pdf(x, true_mean, true_std),
    label="True GMSL",
    linewidth=2,
)
ax1.plot(
    x,
    norm.pdf(x, est_mean, est_std),
    label="Estimated GMSL",
    linewidth=2,
)
ax1.set_xlabel("GMSL (m)")
ax1.set_ylabel("Probability Density")
ax1.set_title("GMSL Distributions")
ax1.legend()
ax1.grid(alpha=0.3)

# Error distribution
error_x = np.linspace(
    error_mean - 4 * error_std,
    error_mean + 4 * error_std,
    1000,
)
ax2.plot(
    error_x,
    norm.pdf(error_x, error_mean, error_std),
    "r",
    linewidth=2,
)
ax2.axvline(
    0,
    color="k",
    linestyle="--",
    alpha=0.3,
    # label="Zero error",
)
ax2.set_xlabel("Error (m)")
# ax2.set_ylabel("Probability Density")
ax2.set_title("Error Distribution")
# ax2.legend()
ax2.grid(alpha=0.3)
fig.suptitle("GMSL Estimation Errors")
fig.text(
    0.5,
    0.01,
    f"Ice Thickness Change: {net_ice_thickness_change * fp.length_scale:.1f} m; Ocean Dynamic Topography Std Dev: {odt_standard_deviation * fp.length_scale:.3f} m\nAltimetry Error Amplitude Standard Deviation: {altimetry_error_amplitude * fp.length_scale:.3f} m; Calculated for Altimetry over +/- {altimetry_range:.0f} deg\nValues are approximate for monthly binned measurements.",
    fontsize=10,
    ha="center",
    va="bottom",
)
fig.subplots_adjust(bottom=0.25)  # Increase bottom margin

# plt.tight_layout()
# %%
plt.show()

# %%

iteration = 1
fig.savefig(f"output_{iteration}.pdf")
fig.savefig(f"output_{iteration}.png")
