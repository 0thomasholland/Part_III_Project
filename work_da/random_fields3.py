import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl

# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=256)
fp.set_state_from_ice_ng()

# --- Get the representation as a Linear operator between Sobolev spaces ---
fingerprint_operator = fp.as_sobolev_linear_operator(2, 0.1 * fp.mean_sea_floor_radius)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# --- Generate random fields for the ice thickness change ---

ice_thickness_length_scale = 0.2 * fp.mean_sea_floor_radius
ice_thickness_gmsl_target = 0.005 / fp.length_scale

# Set an intial rotationally invariant measure
initial_ice_thickness_measure = load_space.heat_kernel_gaussian_measure(
    ice_thickness_length_scale
)

# Set a projection operator for the ice sheets and push forward the
# initial measure so that fields are non-zero only over the ice sheets
ice_projection = sl.ice_projection_operator(fp, load_space)
ice_thickness_measure = initial_ice_thickness_measure.affine_mapping(
    operator=ice_projection
)

# Set up an operator that maps direct loads to GMSL change
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
GMSL_operator = sl.averaging_operator(load_space, [GMSL_weighting_function])

# Push foward the load measure to one for GMSL and get its standard deviation
GMSL_measure = ice_thickness_measure.affine_mapping(operator=GMSL_operator)
GMSL_variance = GMSL_measure.covariance.matrix(dense=True)[0, 0]
GMSL_std = np.sqrt(GMSL_variance)

# Normalise the ice load thickness measure and then recompute the load measure
ice_thickness_measure *= ice_thickness_gmsl_target / GMSL_std


# Define operator that maps ice thickness changes to loads
ice_thickness_to_load_operator = sl.ice_thickness_change_to_load_operator(
    fp, load_space
)
ice_load_measure = ice_thickness_measure.affine_mapping(
    operator=ice_thickness_to_load_operator
)


# --- Set up a random field for ocean dynamic topography ---

ocean_dynamic_topography_order = 1.5
ocean_dynamic_topography_length_scale = 0.05 * fp.mean_sea_floor_radius
ocean_dynamic_topography_amplitude = 0.005 / fp.length_scale

# Start with a rotationally invariant random field
initial_ocean_dynamic_topography_measure = (
    load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        ocean_dynamic_topography_order,
        ocean_dynamic_topography_length_scale,
        ocean_dynamic_topography_amplitude,
    )
)

# Push forward to a measure that is non-zero only in the oceans.
ocean_projection = sl.ocean_projection_operator(fp, load_space)
ocean_dynamic_topography_measure = (
    initial_ocean_dynamic_topography_measure.affine_mapping(operator=ocean_projection)
)

# Push forward to a ocean load measure
ocean_dynamic_topography_to_load_operator = sl.spatial_mutliplication_operator(
    fp.water_density * fp.ocean_function, load_space
)
ocean_dynamic_topography_load_measure = ocean_dynamic_topography_measure.affine_mapping(
    operator=ocean_dynamic_topography_to_load_operator
)

# --- form the total load measure ---
direct_load_measure = ice_load_measure + ocean_dynamic_topography_load_measure

direct_load = direct_load_measure.sample()


sea_level_change, _, _, _ = fingerprint_operator(direct_load)

fig1, ax1, im1 = sl.plot(
    direct_load,
    symmetric=True,
)
fig1.colorbar(
    im1,
    ax=ax1,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="direct load (kg m$^{-2}$))",
)


fig2, ax2, im2 = sl.plot(
    sea_level_change * fp.ocean_projection(),
    symmetric=True,
)

fig2.colorbar(
    im2,
    ax=ax2,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="sea level change (m)",
)


plt.show()


"""

sea_level_change, _, _, _ = (fingerprint_operator @ ice_thickness_to_load_operator)(
    ice_thickness_change
)
fig1, ax1, im1 = sl.plot(
    ice_thickness_change * fp.ice_projection(),
)
fig1.colorbar(
    im1,
    ax=ax1,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="ice thickness change (m)",
)


fig2, ax2, im2 = sl.plot(
    sea_level_change * fp.ocean_projection(),
)
fig2.colorbar(
    im2,
    ax=ax2,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="sea level change (m)",
)

plt.show()
"""
