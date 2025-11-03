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

# %% params
lmax = 32
sobolev_order = 2
sobolev_length_scale = 0.1

ice_thickness_length_scale = 200.0 * 1e3  # in metres
ice_thickness_change_std = 20  # in metres (spatial variability std dev)

ocean_dynamic_topography_order = 1.5
ocean_dynamic_topography_length_scale = 0.05
ocean_dynamic_topography_amplitude = 0.005

fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()
fingerprint_operator = fp.as_sobolev_linear_operator(
    sobolev_order, sobolev_length_scale * fp.mean_sea_floor_radius
)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# %% ice thickness change measure

ice_thickness_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale=ice_thickness_length_scale, amplitude=ice_thickness_change_std
).affine_mapping(operator=sl.ice_projection_operator(fp, load_space))

fig1, ax1, im1 = sl.plot(ice_thickness_measure.sample(), symmetric=True)
fig1.colorbar(im1, ax=ax1, orientation="horizontal", label="Ice Thickness Change (m)")

# %% ocean dynamic topography measure

ocean_dynamic_topography_measure = (
    load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        ocean_dynamic_topography_order,
        ocean_dynamic_topography_length_scale,
        ocean_dynamic_topography_amplitude,
    ).affine_mapping(operator=sl.ocean_projection_operator(fp, load_space))
)

fig2, ax2, im2 = sl.plot(ocean_dynamic_topography_measure.sample(), symmetric=True)
fig2.colorbar(
    im2, ax=ax2, orientation="horizontal", label="Ocean Dynamic Topography (m)"
)

# %% load measures

ice_load_measure = ice_thickness_measure.affine_mapping(
    operator=sl.ice_thickness_change_to_load_operator(fp, load_space)
)

ocean_load_measure = ocean_dynamic_topography_measure.affine_mapping(
    operator=sl.spatial_mutliplication_operator(
        fp.water_density * fp.ocean_function, load_space
    )
)

total_load_measure = ice_load_measure + ocean_load_measure

fig3, ax3, im3 = sl.plot(total_load_measure.sample(), symmetric=True)
fig3.colorbar(im3, ax=ax3, orientation="horizontal", label="Total Load (Pa)")


# %% Sea Level Change

sea_level_change_measure = total_load_measure.affine_mapping(
    operator=fingerprint_operator
)

sea_surface_height_change_measure = sea_level_change_measure.affine_mapping(
    operator=sl.sea_surface_height_operator(fp, response_space)
)


fig4, ax4, im4 = sl.plot(
    sea_level_change_measure.sample()[0] * fp.ocean_function, symmetric=True
)
fig4.colorbar(im4, ax=ax4, orientation="horizontal", label="Sea Level Change (m)")

fig5, ax5, im5 = sl.plot(
    sea_surface_height_change_measure.sample() * fp.ocean_function, symmetric=True
)
fig5.colorbar(
    im5, ax=ax5, orientation="horizontal", label="Sea Surface Height Change (m)"
)

# %% observed, including noise and topography

### TODO NOISE FUNCTION ###

### TODO ADD topography ###
observed_ssh_measure = (
    sea_surface_height_change_measure  # + noise_measure + topography_measure
)

# %% create altimetry weighting function


# %% calculate gmsl change estimate, by creating an affine map operator that can be applied to measures
# Helper function to compute GMSL from a sample
def compute_gmsl(sample):
    """Compute GMSL as a scalar from a spatial field."""
    return fp.integrate(sample * fp.ocean_function) / fp.integrate(fp.ocean_function)


# To get GMSL statistics from the measure, sample many times
n_samples = 100
gmsl_samples = np.array(
    [compute_gmsl(observed_ssh_measure.sample()) for _ in range(n_samples)]
)

true_gmsl_samples = np.array(
    [compute_gmsl(sea_level_change_measure.sample()[0]) for _ in range(n_samples)]
)

gmsl_mean = np.mean(gmsl_samples)
gmsl_std = np.std(gmsl_samples)
true_gmsl_mean = np.mean(true_gmsl_samples)
true_gmsl_std = np.std(true_gmsl_samples)

print(f"GMSL mean: {gmsl_mean:.4f} m")
print(f"GMSL std: {gmsl_std:.4f} m")
print(f"True GMSL mean: {true_gmsl_mean:.4f} m")
print(f"True GMSL std: {true_gmsl_std:.4f} m")

# Plot histogram of GMSL distribution
fig6, ax6 = plt.subplots()
ax6.hist(
    gmsl_samples, bins=50, density=True, alpha=0.7, label="SSH Estimate", color="blue"
)
ax6.hist(
    true_gmsl_samples,
    bins=50,
    density=True,
    alpha=0.7,
    label="SLC True",
    color="orange",
)
ax6.set_xlabel("GMSL (m)")
ax6.set_ylabel("Probability Density")
ax6.legend()

# %%

### TODO get actual distribution...
