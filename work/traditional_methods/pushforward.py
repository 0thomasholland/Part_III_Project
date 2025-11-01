## Workflow
# Define sea level topology distribution and ice sheet change distributions using point_value_scaled_sobolev_kernel_gaussian_measure and heat_kernel_gaussian_measure respectively
# Use this to calculate the sea level change (SLC fingerprint) due to the direct load from ice sheet changes and ocean dynamic topography changes
# Add in the sea topography change from ocean dynamic topography changes
# calculate the sea surface height change
# add in a noise model distribution to represent measurement noise from the satalite to get observed_ssh
# caculate the GMSL change from SLC fingerprint and from observed_ssh,

# %% import libraries
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
from pyslfp.physical_parameters import GRAVITATIONAL_ACCELERATION
from scipy.stats import norm

from Part_III_Project import SeaSurfaceFingerPrint

# %% set up variables

lmax = 128
sobolev_order = 2
sobolev_length_scaler = 0.1

ice_thickness_length_scale_scaler = 0.2
ice_thickness_gmsl_target = 0.001  # in units of fp.length

ocean_dynamic_topography_order = 1.5
ocean_dynamic_topography_length_scale_scaler = 0.05  # relative to mean sea floor radius
ocean_dynamic_topography_amplitude = 0.005  # in units of fp.length

noise_length_scaler = 0.0001
noise_amplitude = 0.0001  # in units of fp.length

# %% set up fingerprint instance
fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    sobolev_order, sobolev_length_scaler * fp.mean_sea_floor_radius
)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# %% define field characteristics for ice_thickness, ocean_dynamic_topography and noise

ice_thickness_length_scale = (
    ice_thickness_length_scale_scaler * fp.mean_sea_floor_radius
)
ice_thickness_gmsl_target = ice_thickness_gmsl_target / fp.length_scale

ocean_dynamic_topography_length_scale = (
    ocean_dynamic_topography_length_scale_scaler * fp.mean_sea_floor_radius
)


# Create ice thickness measure with projection
ice_thickness_measure = load_space.heat_kernel_gaussian_measure(
    ice_thickness_length_scale
).affine_mapping(operator=sl.ice_projection_operator(fp, load_space))

# Calculate GMSL standard deviation and normalize
GMSL_std = np.sqrt(
    ice_thickness_measure.affine_mapping(
        operator=sl.averaging_operator(
            load_space,
            [
                -fp.ice_density
                * fp.one_minus_ocean_function
                * fp.ice_projection(value=0)
                * fp.length_scale
                / (fp.water_density * fp.ocean_area)
            ],
        )
    ).covariance.matrix(dense=True)[0, 0]
)

ice_thickness_measure *= ice_thickness_gmsl_target / GMSL_std

# Push forward rotationally invariant field to ocean-only measure
ocean_dynamic_topography_measure = (
    load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        ocean_dynamic_topography_order,
        ocean_dynamic_topography_length_scale,
        ocean_dynamic_topography_amplitude,
    ).affine_mapping(operator=sl.ocean_projection_operator(fp, load_space))
)

# %% define load measures

ice_load_measure = ice_thickness_measure.affine_mapping(
    operator=sl.ice_thickness_change_to_load_operator(fp, load_space)
)

ocean_dynamic_topography_load_measure = ocean_dynamic_topography_measure.affine_mapping(
    operator=sl.spatial_mutliplication_operator(
        fp.water_density * fp.ocean_function, load_space
    )
)

direct_load_measure = ice_load_measure + ocean_dynamic_topography_load_measure

# %% plot direct load measure samples
fig1, ax1, im1 = sl.plot(direct_load_measure.sample(), symmetric=True)
ax1.set_title("Direct Load Sample")

# %% define sea level fingerprint measure

sea_level_fingerprint_measure = direct_load_measure.affine_mapping(
    operator=fingerprint_operator
)


# %% plot sea level fingerprint measure samples

(sea_level_fingerprint, _, _, _) = sea_level_fingerprint_measure.sample()

fig2, ax2, im2 = sl.plot(sea_level_fingerprint * fp.ocean_projection(), symmetric=True)
ax2.set_title("Sea Level Change Sample")

# %% define sea surface height change using the full formula
# The fingerprint operator returns a tuple: (sea_level_change, displacement, gravitational_potential_change, angular_velocity_change)
# We need to compute: SSH = sea_level_change + displacement + (centrifugal_potential_change / g)

# Get the first subspace (sea level change component) - this will be the codomain
sea_level_space = response_space.subspace(0)


# Create operator that computes sea surface height change from the full fingerprint tuple
def compute_sea_surface_height_change(fingerprint_tuple):
    """
    Compute sea surface height change from fingerprint operator output.

    Args:
        fingerprint_tuple: (sea_level_change, displacement, gravitational_potential_change, angular_velocity_change)

    Returns:
        sea_surface_height_change: SHGrid
    """
    (
        sea_level_change,
        displacement,
        gravitational_potential_change,
        angular_velocity_change,
    ) = fingerprint_tuple

    # Compute centrifugal potential change
    centrifugal_potential_change = fp.centrifugal_potential_change(
        angular_velocity_change=angular_velocity_change
    )

    # Compute sea surface height change using the full formula
    ssh_change = (
        sea_level_change
        + displacement
        + (centrifugal_potential_change / GRAVITATIONAL_ACCELERATION)
    )

    return ssh_change


# Create the linear operator that maps from response_space to sea_level_space
ssh_operator = inf.LinearOperator(
    response_space, sea_level_space, compute_sea_surface_height_change
)

# Push forward the measure through the operator to get sea surface height measure
sea_surface_height_measure = sea_level_fingerprint_measure.affine_mapping(
    operator=ssh_operator
)

# %% plot sea surface height change sample
fig3, ax3, im3 = sl.plot(
    sea_surface_height_measure.sample() * fp.ocean_projection(), symmetric=True
)
ax3.set_title("Sea Surface Height Change Sample")

# %% set up observered sea surface height by adding in topographic change
sea_surface_height_measure_obs = sea_surface_height_measure

#################
# This block needs fixing - current issue is that can't generate noise measures as they are too "smooth" for static noise and the .random() method only works in sampled space...
#################
# # %% add noise as random field with no spatial correlation with very small length scales that simulates static noise

# # noise_measure = sea_level_space.point_value_scaled_heat_kernel_gaussian_measure(
# #     scale=noise_length_scaler * fp.mean_sea_floor_radius,
# #     amplitude=noise_amplitude,
# # )

# noise_measure = sea_level_space.random()

# fig3, ax3, im3 = sl.plot(noise_measure, symmetric=True)
# ax3.set_title("Noise Sample")

# # %% define observed sea surface height measure
# sea_surface_height_measure_obs = sea_surface_height_measure.sample() + (
#     0.000003 * noise_measure
# )

# fig4, ax4, im4 = sl.plot(
#     sea_surface_height_measure_obs * fp.ocean_projection(), symmetric=True
# )
# ax4.set_title("Observed Sea Surface Height Sample")

###### END OF BROKEN SECTION #########


# %% calculate GMSL change from sea level fingerprint and observed sea surface height

# GMSL change from ice thickness measure (using the same calculation as earlier normalization)
true_gmsl_change = ice_thickness_measure.affine_mapping(
    operator=sl.averaging_operator(
        load_space,
        [
            -fp.ice_density
            * fp.one_minus_ocean_function
            * fp.ice_projection(value=0)
            * fp.length_scale
            / (fp.water_density * fp.ocean_area)
        ],
    )
)

# GMSL change from observed sea surface height measure
observed_gmsl_change = sea_surface_height_measure_obs.affine_mapping(
    operator=sl.averaging_operator(
        sea_level_space,
        [
            -fp.ice_density
            * fp.one_minus_ocean_function
            * fp.ice_projection(value=0)
            * fp.length_scale
            / (fp.water_density * fp.ocean_area)
        ],
    )
)

# project to ±66 degrees latitude range using
observed_gmsl_change_sat_range_66 = sea_surface_height_measure_obs.affine_mapping(
    operator=sl.averaging_operator(
        sea_level_space,
        [
            -fp.ice_density
            * fp.altimetry_projection(latitude_min=-66, latitude_max=66, value=0)
            * fp.ice_projection(value=0)
            * fp.length_scale
            / (fp.water_density * fp.ocean_area)
        ],
    )
)

# %% plot GMSL changes distribution of true and observed

# Extract mean and standard deviation for each measure
# For Gaussian measures, these are available from the covariance structure
true_mean = true_gmsl_change.expectation[0]
true_std = np.sqrt(true_gmsl_change.covariance.matrix(dense=True)[0, 0])

observed_mean = observed_gmsl_change.expectation[0]
observed_std = np.sqrt(observed_gmsl_change.covariance.matrix(dense=True)[0, 0])

# sat_range_mean = observed_gmsl_change_sat_range_66.expectation[0]
# sat_range_std = np.sqrt(
#     observed_gmsl_change_sat_range_66.covariance.matrix(dense=True)[0, 0]
# )

# Create x-axis range that covers all three distributions
x_min = min(
    true_mean - 4 * true_std,
    observed_mean - 4 * observed_std,
    #     sat_range_mean - 4 * sat_range_std,
)
x_max = max(
    true_mean + 4 * true_std,
    observed_mean + 4 * observed_std,
    #     sat_range_mean + 4 * sat_range_std,
)

x = np.linspace(x_min, x_max, 500)

# Calculate probability density functions for each distribution
true_pdf = norm.pdf(x, loc=true_mean, scale=true_std)
observed_pdf = norm.pdf(x, loc=observed_mean, scale=observed_std)
# sat_range_pdf = norm.pdf(x, loc=sat_range_mean, scale=sat_range_std)

# Create the plot
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.plot(
    x * fp.length_scale * 1000,
    true_pdf / (fp.length_scale * 1000),
    label="True GMSL Change",
    linewidth=2,
    color="blue",
)
ax5.plot(
    x * fp.length_scale * 1000,
    observed_pdf / (fp.length_scale * 1000),
    label="Observed GMSL Change",
    linewidth=2,
    color="red",
    linestyle="--",
)
# ax5.plot(
#     x * fp.length_scale * 1000,
#     sat_range_pdf / (fp.length_scale * 1000),
#     label="Observed GMSL (±66° lat)",
#     linewidth=2,
#     color="green",
#     linestyle=":",
# )

ax5.set_xlabel("GMSL Change (mm)", fontsize=12)
ax5.set_ylabel("Probability Density (1/mm)", fontsize=12)
ax5.set_title("Distribution of GMSL Changes", fontsize=14, fontweight="bold")
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# Add vertical lines at the means
ax5.axvline(true_mean * fp.length_scale * 1000, color="blue", alpha=0.3, linestyle="-")
ax5.axvline(
    observed_mean * fp.length_scale * 1000, color="red", alpha=0.3, linestyle="--"
)
# ax5.axvline(
#     sat_range_mean * fp.length_scale * 1000, color="green", alpha=0.3, linestyle=":"
# )

# Print statistics
print("\n" + "=" * 60)
print("GMSL Change Statistics")
print("=" * 60)
print("True GMSL Change:")
print(f"  Mean: {true_mean * fp.length_scale * 1000:.4f} mm")
print(f"  Std:  {true_std * fp.length_scale * 1000:.4f} mm")
print("\nObserved GMSL Change:")
print(f"  Mean: {observed_mean * fp.length_scale * 1000:.4f} mm")
print(f"  Std:  {observed_std * fp.length_scale * 1000:.4f} mm")
# print("\nObserved GMSL (±66° lat):")
# print(f"  Mean: {sat_range_mean * fp.length_scale * 1000:.4f} mm")
# print(f"  Std:  {sat_range_std * fp.length_scale * 1000:.4f} mm")
# print("=" * 60 + "\n")

# %% show all plots

plt.show()
