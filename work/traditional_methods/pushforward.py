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

plt.rcParams["axes.formatter.useoffset"] = False

# %% set up variables
lmax = 32
sobolev_order = 2
sobolev_length_scaler = 0.1

ice_thickness_length_scale_scaler = 200.0 * 1e3  # in metres
ice_thickness_change_std = 200  # in metres (spatial variability std dev)

ocean_dynamic_topography_order = 1.5
ocean_dynamic_topography_length_scale_scaler = 0.05
ocean_dynamic_topography_amplitude = 0.005
# %% set up fingerprint instance
fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    sobolev_order, sobolev_length_scaler * fp.mean_sea_floor_radius
)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# %% define ice thickness measure

# Convert to normalized units
ice_thickness_length_scale = ice_thickness_length_scale_scaler / fp.length_scale
ice_thickness_change_std_norm = ice_thickness_change_std / fp.length_scale
# set up distribution
ice_thickness_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale=ice_thickness_length_scale,
    amplitude=ice_thickness_change_std_norm,  # This sets the std
)

fig0, ax0, im0 = sl.plot(
    ice_thickness_measure.sample() * fp.ice_projection(), symmetric=False
)
ax0.set_title("Ice Thickness Change Sample")
fig0.colorbar(im0, ax=ax0, orientation="horizontal", label="Ice Thickness Change (m)")


# %%

# Calculate resulting GMSL (for verification)
gmsl_from_ice = ice_thickness_measure.affine_mapping(
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

print(
    f"Input ice thickness change: {ice_thickness_change_mean:.1f} ± {ice_thickness_change_std:.1f} m"
)
print(
    f"Expected GMSL change: {gmsl_from_ice.expectation[0] * fp.length_scale * 1000:.2f} mm"
)
print(
    f"GMSL std deviation: {np.sqrt(gmsl_from_ice.covariance.matrix(dense=True)[0, 0]) * fp.length_scale * 1000:.2f} mm"
)

# %% define ocean dynamic topography measure

ocean_dynamic_topography_length_scale = (
    ocean_dynamic_topography_length_scale_scaler * fp.mean_sea_floor_radius
)

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

fig0, ax0, im0 = sl.plot(
    ice_load_measure.sample() * fp.ice_projection(), symmetric=False, cmap="coolwarm"
)
ax0.set_title("Ice Load Sample")
fig0.colorbar(im0, ax=ax0, orientation="horizontal", label="Load (normalized units)")

ocean_dynamic_topography_load_measure = ocean_dynamic_topography_measure.affine_mapping(
    operator=sl.spatial_mutliplication_operator(
        fp.water_density * fp.ocean_function, load_space
    )
)

direct_load_measure = ice_load_measure + ocean_dynamic_topography_load_measure

# %% plot direct load measure samples
fig1, ax1, im1 = sl.plot(direct_load_measure.sample(), symmetric=False)
ax1.set_title("Direct Load Sample")
# scale bar
fig1.colorbar(im1, ax=ax1, orientation="horizontal", label="Load (normalized units)")

# %% define sea level fingerprint measure

sea_level_fingerprint_measure = direct_load_measure.affine_mapping(
    operator=fingerprint_operator
)


# %% plot sea level fingerprint measure samples

(sea_level_fingerprint, _, _, _) = sea_level_fingerprint_measure.sample()

fig2, ax2, im2 = sl.plot(sea_level_fingerprint * fp.ocean_projection(), symmetric=True)
ax2.set_title("Sea Level Change Sample")
fig2.colorbar(im2, ax=ax2, orientation="horizontal", label="SLC (m)")

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
fig3.colorbar(im3, ax=ax3, orientation="horizontal", label="SSH Change (m)")

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
print("Calculating GMSL changes...")
# global mean sea level change from ice thickness measure (using the same calculation as earlier normalization)
# true_mean_sea_level_change = fp.mean_sea_level_change(direct_load) but as a measure where mean_sea_level_change = -self.integrate(direct_load) / (self.water_density * self.ocean_area)


# First, extract just the sea level change component (first element of the tuple)
# Create an operator that extracts the first component
def extract_sea_level_change(fingerprint_tuple):
    return fingerprint_tuple[0]


sea_level_extraction_operator = inf.LinearOperator(
    response_space, sea_level_space, extract_sea_level_change
)

# Get just the sea level change measure
sea_level_change_measure = sea_level_fingerprint_measure.affine_mapping(
    operator=sea_level_extraction_operator
)

# Now compute GMSL from the extracted sea level change
true_gmsl_change = sea_level_change_measure.affine_mapping(
    operator=sl.averaging_operator(
        sea_level_space, [fp.ocean_function * fp.length_scale]
    )
)
print("Calculated true GMSL change.")

print("Calculating observed GMSL change...")
observed_gmsl_change = sea_surface_height_measure_obs.affine_mapping(
    operator=sl.averaging_operator(
        sea_level_space, [fp.ocean_function * fp.length_scale]
    )
)
print("Calculated observed GMSL change.")

print("Calculating observed GMSL change (±66° lat)...")
observed_gmsl_change_sat_range_66 = sea_surface_height_measure_obs.affine_mapping(
    operator=sl.averaging_operator(
        sea_level_space,
        [
            fp.altimetry_projection(latitude_min=-66, latitude_max=66, value=0.0)
            * fp.ocean_function
            * fp.length_scale
        ],
    )
)
print("Calculated observed GMSL change (±66° lat).")


# %% extract mean and standard deviation for each measure using covariance matrix

print("checkpoint 0")
# Extract mean and standard deviation for each measure
# For Gaussian measures, these are available from the covariance structure
true_mean = true_gmsl_change.expectation[0]
# Direct covariance evaluation - apply operator once and take the result
basis_vector = np.array([1.0])  # Unit vector in the 1D space
cov_basis = true_gmsl_change.covariance(basis_vector)
true_std = np.sqrt(np.dot(cov_basis, basis_vector))

observed_mean = observed_gmsl_change.expectation[0]
cov_basis_obs = observed_gmsl_change.covariance(basis_vector)
observed_std = np.sqrt(np.dot(cov_basis_obs, basis_vector))

observed_66_mean = observed_gmsl_change_sat_range_66.expectation[0]
cov_basis_obs_66 = observed_gmsl_change_sat_range_66.covariance(basis_vector)
observed_66_std = np.sqrt(np.dot(cov_basis_obs_66, basis_vector))

# %%
# Create x-axis range that covers all three distributions
x_min = min(
    true_mean - 4 * true_std,
    observed_mean - 4 * observed_std,
    observed_66_mean - 4 * observed_66_std,
)
x_max = max(
    true_mean + 4 * true_std,
    observed_mean + 4 * observed_std,
    observed_66_mean + 4 * observed_66_std,
)

x = np.linspace(x_min, x_max, 500)

# Calculate probability density functions for each distribution
true_pdf = norm.pdf(x, loc=true_mean, scale=true_std)
observed_pdf = norm.pdf(x, loc=observed_mean, scale=observed_std)
observed_66_pdf = norm.pdf(x, loc=observed_66_mean, scale=observed_66_std)

# Create the plot
fig5, ax5 = plt.subplots(figsize=(10, 6))

ax5.plot(
    x * fp.length_scale,
    true_pdf,
    label="True GMSL Change",
    linewidth=2,
    color="blue",
)
ax5.plot(
    x * fp.length_scale,
    observed_pdf,
    label="Observed GMSL Change",
    linewidth=2,
    color="red",
    linestyle="--",
)
ax5.plot(
    x * fp.length_scale,
    observed_66_pdf,
    label="Observed GMSL (±66° lat)",
    linewidth=2,
    color="green",
    linestyle=":",
)

ax5.set_xlabel("GMSL Change (m)", fontsize=12)
ax5.set_ylabel("Probability Density (1/m)", fontsize=12)
ax5.set_title("Distribution of GMSL Changes", fontsize=14, fontweight="bold")
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)

# Add vertical lines at the means
ax5.axvline(true_mean, color="blue", alpha=0.3, linestyle="-")
ax5.axvline(observed_mean, color="red", alpha=0.3, linestyle="--")
ax5.axvline(observed_66_mean, color="green", alpha=0.3, linestyle=":")

# Print statistics
print("\n" + "=" * 60)
print("GMSL Change Statistics")
print("=" * 60)
print("True GMSL Change:")
print(f"  Mean: {true_mean:.4f} m")
print(f"  Std:  {true_std:.4f} m")
print("\nObserved GMSL Change:")
print(f"  Mean: {observed_mean:.4f} m")
print(f"  Std:  {observed_std:.4f} m")
print("\nObserved GMSL (±66° lat):")
print(f"  Mean: {observed_66_mean:.4f} m")
print(f"  Std:  {observed_66_std:.4f} m")
print("=" * 60 + "\n")

# %% show all plots

plt.show()
