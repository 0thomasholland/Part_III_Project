# %%
import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

lmax = 256
fp = sl.FingerPrint(
    lmax=lmax,
)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

# %%

###### VARIABLES
ice_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.004 / fp.length_scale  # in meters
net_ice_thickness_change = -10.0 / fp.length_scale  # in meters

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_amplitude_95_range = 0.01 / fp.length_scale  # in
altimetry_range = 70  # in meters

altimetry_error_legth_scale = 0.005 * fp.mean_sea_floor_radius
altimetry_error_amplitude = 0.001 / fp.length_scale  # in meters

######

ice_thickness_change, _ = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=net_ice_thickness_change,
)

odt_change, odt_load = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    amplitude_95_range=odt_amplitude_95_range,
)


measuremeant_error = sl.sea_surface_height_operator(
    fp,
    fingerprint_operator.codomain,
).codomain.point_value_scaled_sobolev_kernel_gaussian_measure(
    1.5,
    altimetry_error_legth_scale,
    altimetry_error_amplitude,
)


# %%
true_gmsl_operator: inf.LinearOperator = sl.averaging_operator(
    fingerprint_operator.domain,
    [
        -fp.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        * fp.length_scale
        / (fp.water_density * fp.ocean_area),
    ],
)

altimetry_averaging_operator: inf.LinearOperator = (
    sl.averaging_operator(
        sl.sea_surface_height_operator(
            fp,
            fingerprint_operator.codomain,
        ).codomain,
        [
            (
                fp.ocean_projection(value=0)
                * fp.altimetry_projection(
                    latitude_min=-altimetry_range,
                    latitude_max=altimetry_range,
                    value=0,
                )
            )
            / fp.integrate(
                fp.ocean_projection(value=0)
                * fp.altimetry_projection(
                    latitude_min=-altimetry_range,
                    latitude_max=altimetry_range,
                    value=0,
                ),
            ),
        ],
    )
)

estimated_gmsl_operator = (
    altimetry_averaging_operator
    @ sl.sea_surface_height_operator(
        fp,
        fingerprint_operator.codomain,
    )
    @ fingerprint_operator
    @ sl.ice_thickness_change_to_load_operator(
        fp,
        fingerprint_operator.domain,
    )
)

# %%
combined_operator = estimated_gmsl_operator - true_gmsl_operator

# %%
error_covariance = (
    combined_operator
    @ ice_thickness_change.covariance
    @ combined_operator.adjoint
    + altimetry_averaging_operator
    @ measuremeant_error.covariance
    @ altimetry_averaging_operator.adjoint
).matrix(dense=True)[0, 0]

error_expectation = combined_operator(
    ice_thickness_change.expectation,
)[0]
# %%
print(error_expectation)
print(error_covariance)

# %%

true_gmsl_covariance = (
    true_gmsl_operator
    @ ice_thickness_change.covariance
    @ true_gmsl_operator.adjoint
).matrix(dense=True)[0, 0]

true_gmsl_expectation = true_gmsl_operator(
    ice_thickness_change.expectation,
)[0]

# %%
print(true_gmsl_expectation)
print(true_gmsl_covariance)

# %%

# plot these distributions on the same plot
x_min = min(
    error_expectation - 4 * np.sqrt(error_covariance),
    true_gmsl_expectation - 4 * np.sqrt(true_gmsl_covariance),
)
x_max = max(
    error_expectation + 4 * np.sqrt(error_covariance),
    true_gmsl_expectation + 4 * np.sqrt(true_gmsl_covariance),
)
x = np.linspace(x_min, x_max, 1000)
fig, ax = plt.subplots(figsize=(10, 6))
y_error = norm.pdf(
    x,
    error_expectation,
    np.sqrt(error_covariance),
)
y_true = norm.pdf(
    x,
    true_gmsl_expectation,
    np.sqrt(true_gmsl_covariance),
)
ax.plot(
    x,
    y_error,
    label=f"Estimation Error (μ={error_expectation:.2e}, σ={np.sqrt(error_covariance):.2e})",
)
ax.plot(
    x,
    y_true,
    label=f"True GMSL (μ={true_gmsl_expectation:.2e}, σ={np.sqrt(true_gmsl_covariance):.2e})",
)
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.set_xlabel("GMSL (m)")
ax.set_ylabel("Probability Density")
ax.set_title(
    f"GMSL Estimation Error vs True GMSL Distribution\nIce thickness change: {net_ice_thickness_change:.2e} m, ODT range: {odt_amplitude_95_range:.2e} m, \nAltimetry error std: {altimetry_error_amplitude:.2e} m, Altimetry lat range: {altimetry_range}°",
)
ax.legend()
plt.show()

fig.savefig("/home/th/Downloads/fig.png")
