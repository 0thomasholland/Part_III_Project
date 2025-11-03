import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
import scipy.stats as stats

# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()

# --- Get the representation as a Linear operator between Sobolev spaces ---
fingerprint_operator = fp.as_sobolev_linear_operator(2, 0.1 * fp.mean_sea_floor_radius)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain

# --- form the sea surface height operator ---
response_to_sea_surface_height_operator = sl.sea_surface_height_operator(
    fp, response_space
)

# --- Set up a random field for the ice thickness change and associated load ---

ice_thickness_length_scale = 0.1 * fp.mean_sea_floor_radius
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
ocean_dynamic_topography_amplitude = 0.001 / fp.length_scale


# Start with a rotationally invariant random field
initial_ocean_dynamic_topography_measure = (
    load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        ocean_dynamic_topography_order,
        ocean_dynamic_topography_length_scale,
        ocean_dynamic_topography_amplitude,
    )
)


# Push forward to a measure that is non-zero only in the oceans and which averages to zero
ocean_projection = sl.ocean_projection_operator(fp, load_space)
remove_ocean_average_operator = sl.remove_ocean_average_operator(fp, load_space)
ocean_dynamic_topography_measure = (
    initial_ocean_dynamic_topography_measure.affine_mapping(
        operator=remove_ocean_average_operator @ ocean_projection
    )
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


# Push forward the joint measure under this operator
direct_load_measure = joint_measure.affine_mapping(operator=direct_load_operator)

# --- Set up the linear operator that maps to the total sea surface height change ---

total_sea_surface_height_operator = (
    response_to_sea_surface_height_operator
    @ fingerprint_operator
    @ direct_load_operator
    + inf.RowLinearOperator(
        [load_space.zero_operator(), load_space.identity_operator()]
    )
)


# --- Set up an altimetry operator ---

sea_surface_height_space = total_sea_surface_height_operator.codomain

# Set the range for the altimetry measurements
latitude_min = -66
latitude_max = 66

altimetry_projection = fp.altimetry_projection(
    latitude_min=latitude_min, latitude_max=latitude_max, value=0
)

altimetry_operator = sl.spatial_mutliplication_operator(
    altimetry_projection,
    sea_surface_height_space,
)

# Set up a observational error field for the altimetry observations

altimetry_error_order = 1.5
altimetry_error_length_scale = 0.005 * fp.mean_sea_floor_radius
altimetry_error_order_amplitude = 0.0001 / fp.length_scale


initial_altimetry_error_measure = (
    sea_surface_height_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        altimetry_error_order,
        altimetry_error_length_scale,
        altimetry_error_order_amplitude,
    )
)

altimetry_error_measure = initial_altimetry_error_measure.affine_mapping(
    operator=altimetry_operator
)


# --- make an instance of the inputs and plot the results ---


ice_thickness_change, ocean_dynamic_topography = joint_measure.sample()

sea_surface_height_change = total_sea_surface_height_operator(
    [ice_thickness_change, ocean_dynamic_topography]
)

altimetry_error = altimetry_error_measure.sample()

altimetry_observation = altimetry_operator(sea_surface_height_change) + altimetry_error


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

fig4, ax4, im4 = sl.plot(
    altimetry_observation * fp.ocean_projection(),
    symmetric=True,
)

fig4.colorbar(
    im4,
    ax=ax4,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
    label="altimetry observation (m)",
)

plt.show()


# --- Set up an operator that maps the true ice thickness change to GMSL ---
# TBD


# --- Set up an operator that maps the altimetry observation to an estimate of GMSL ---

altimetry_normalisation = fp.integrate(altimetry_projection)

altimetry_estimate_operator = sl.averaging_operator(
    sea_surface_height_space, [altimetry_projection / altimetry_normalisation]
)

# --- Push forward the various measures to get one for the altimetry estimate ---

altimetry_estimate_measure = joint_measure.affine_mapping(
    operator=altimetry_estimate_operator @ total_sea_surface_height_operator
) + altimetry_error_measure.affine_mapping(operator=altimetry_estimate_operator)


# --- extract information on the final measure for plotting ---


# 1. Get statistics for the POSTERIOR distribution
altimetry_estimate_mean = altimetry_estimate_measure.expectation[0]
altimetry_estimate_var = altimetry_estimate_measure.covariance.matrix(dense=True)[0, 0]
altimetry_estimate_std = np.sqrt(altimetry_estimate_var)

print(altimetry_estimate_mean)
print(altimetry_estimate_std)

# 2. Define an x-axis that covers both distributions
x_min = altimetry_estimate_mean - 6 * altimetry_estimate_std

x_max = altimetry_estimate_mean + 6 * altimetry_estimate_std

x_axis = np.linspace(x_min, x_max, 1000)

# 3. Calculate the PDF values manually using the mean and std
posterior_pdf_values = stats.norm.pdf(
    x_axis, loc=altimetry_estimate_mean, scale=altimetry_estimate_std
)

# 4. Create the plot with two y-axes
fig5, ax5 = plt.subplots(figsize=(12, 7))

# Plot the POSTERIOR on the second axis (ax2)
ax5.plot(
    x_axis,
    posterior_pdf_values,
)


plt.show()


# inf.plot_1d_distributions([altimetry_estimate_measure])
# plt.show()
