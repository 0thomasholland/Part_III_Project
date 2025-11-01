## Workflow
# Define sea level topology distribution and ice sheet change distributions using point_value_scaled_sobolev_kernel_gaussian_measure and heat_kernel_gaussian_measure respectively
# Use this to calculate the sea level change (SLC fingerprint) due to the direct load from ice sheet changes and ocean dynamic topography changes
# Add in the sea topography change from ocean dynamic topography changes
# calculate the sea surface height change
# add in a noise model distribution to represent measurement noise from the satalite to get observed_ssh
# caculate the GMSL change from SLC fingerprint and from observed_ssh,


### ISSUE: is there a way of consistently accessing the same sampling of a measure after pushing it through an operator? i.e. so that the noise and the sea level fingerprint sample correspond to the same spatial locations?

# %% import libraries
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl

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

noise_length_scaler = 0.01
noise_amplitude = 0.001  # in units of fp.length

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

# %% define sea surface height change (for now just have has SLC fingerprint)

sea_surface_height_measure = sea_level_fingerprint_measure

# %% add noise as random field with no spatial correlation with very small length scales that simulates static noise
noise_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale=noise_length_scaler * fp.mean_sea_floor_radius,
    amplitude=noise_amplitude,
)

fig3, ax3, im3 = sl.plot(noise_measure.sample(), symmetric=True)
ax3.set_title("Noise Sample")

# %% define observed sea surface height measure
print(sea_surface_height_measure)
print(sea_surface_height_measure.domain.subspace(0))
print(noise_measure.domain)
### ISSUE: cannot add measures defined on different spaces, becuase having pushed through the load to sea level fingerprint operator, it contains the four subspaces... how to access this?
# %% show all plots

plt.show()
