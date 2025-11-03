import numpy as np
import matplotlib.pyplot as plt
import pygeoinf as inf
import pyslfp as sl


# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=256)
fp.set_state_from_ice_ng()

# --- Get the representation as a Linear operator between Sobolev spaces ---
fingerprint_operator = fp.as_sobolev_linear_operator(2, 0.1 * fp.mean_sea_floor_radius)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# --- form the sea surface height operator ---
sea_surface_height_operator = sl.sea_surface_height_operator(fp, response_space)


# --- Set up a random field for the ice thickness change and associated load ---

ice_thickness_length_scale = 0.05 * fp.mean_sea_floor_radius
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

# --- Set up a random field for the ocean dynamic topography ---

ocean_dynamic_topography_order = 1.5
ocean_dynamic_topography_length_scale = 0.005 * fp.mean_sea_floor_radius
ocean_dynamic_topography_amplitude = 0.0005 / fp.length_scale

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

# --- Set up joint distribution for ice thickness change and ocean dynamic topography ---

joint_measure = inf.GaussianMeasure.from_direct_sum(
    [ice_thickness_measure, ocean_dynamic_topography_measure]
)


# --- Set up a random field for the total load ---

# Define operators that maps ice thickness changes and sea level change to loads
ice_thickness_to_load_operator = sl.ice_thickness_change_to_load_operator(
    fp, load_space
)
sea_level_change_to_load_operator = sl.sea_level_change_to_load_operator(fp, load_space)


# Set up the linear operator that maps to the direct load ---
direct_load_operator = inf.RowLinearOperator(
    [ice_thickness_to_load_operator, sea_level_change_to_load_operator]
)

ice_thickness_change, ocean_dynamic_topography = joint_measure.sample()


# Push forward the joint measure under this operator
direct_load_measure = joint_measure.affine_mapping(operator=direct_load_operator)

# --- Set up the linear operator that maps to the total sea surface height change ---
joint_space = direct_load_operator.domain
total_sea_surface_height_operator = (
    sea_surface_height_operator @ fingerprint_operator @ direct_load_operator
    + joint_space.subspace_projection(1)
)

ice_thickness_change, ocean_dynamic_topography = joint_measure.sample()

sea_surface_height_change = total_sea_surface_height_operator(
    [ice_thickness_change, ocean_dynamic_topography]
)

fig1, ax1, im1 = sl.plot(
    ice_thickness_change * fp.ice_projection(),
    symmetric=True,
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
    ocean_dynamic_topography * fp.ocean_projection(),
    symmetric=True,
)
fig2.colorbar(
    im2,
    ax=ax2,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="ocean dynamic topography (m)",
)


fig3, ax3, im3 = sl.plot(
    sea_surface_height_change * fp.ocean_projection(),
    symmetric=True,
)
fig3.colorbar(
    im3,
    ax=ax3,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="sea surface height change (m)",
)

plt.show()
