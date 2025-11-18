# %% imports

import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
import scipy.stats as stats

# %% Parameters setting

lmax = 32


standard_nondim = sl.EarthModelParameters.from_standard_non_dimensionalisation()
fp = sl.FingerPrint(lmax=lmax, earth_model_parameters=standard_nondim)
fp.set_state_from_ice_ng()


## ice space parameters
ice_sobolev_order = 2
ice_sobolev_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_thickness_change_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_thickness_change_standard_dev = 100 / fp.length_scale  # in non-dimensional meters
ice_thickness_change_expectation = -100.0 / fp.length_scale  # in non-dimensional meters

# ocean dynamic topography parameters
ocean_dynamic_topography_sobolev_order = 1.5
ocean_dynamic_topography_sobolev_length_scale = (
    np.array([0.005, 0.1]) * fp.mean_sea_floor_radius
)
ocean_dynamic_topography_amplitude_standard_dev = (
    np.array([5, 1]) / fp.length_scale / 1000
)  # in non-dimensional millimeters

# satellite measurement
sat_max, sat_min = 66, -66  # degrees


# %% Set up a operators spaces

fingerprint_operator = fp.as_sobolev_linear_operator(
    ice_sobolev_order, ice_sobolev_length_scale
)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain
response_to_sea_surface_height_operator = sl.sea_surface_height_operator(
    fp, response_space
)

# %% Set up ice thickness change measure
shift_vector = np.zeros(load_space.dim)
shift_vector[0] = ice_thickness_change_expectation
shift_vector = load_space.from_components(shift_vector)

ice_thickness_measure = load_space.point_value_scaled_heat_kernel_gaussian_measure(
    scale=ice_thickness_change_length_scale, amplitude=ice_thickness_change_standard_dev
).affine_mapping(
    operator=sl.ice_projection_operator(fp, load_space), translation=shift_vector
)


# %%

fig1, ax1, im1 = sl.plot(
    ice_thickness_measure.sample() * fp.length_scale, symmetric=True
)
fig1.colorbar(im1, ax=ax1, orientation="horizontal", label="Ice Thickness Change (m)")


# %% ocean dynamic topography measure

ocean_dynamic_topography_measure = (
    load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        ocean_dynamic_topography_sobolev_order,
        ocean_dynamic_topography_sobolev_length_scale[0],
        ocean_dynamic_topography_amplitude_standard_dev[0],
    ).affine_mapping(operator=sl.ocean_projection_operator(fp, load_space))
)

for ODT_LS, ODT_AMP in zip(
    ocean_dynamic_topography_sobolev_length_scale[1:],
    ocean_dynamic_topography_amplitude_standard_dev[1:],
):
    ocean_dynamic_topography_measure += (
        load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            ocean_dynamic_topography_sobolev_order,
            ODT_LS,
            ODT_AMP,
        ).affine_mapping(operator=sl.ocean_projection_operator(fp, load_space))
    )

ocean_dynamic_topography_measure = ocean_dynamic_topography_measure.affine_mapping(
    operator=sl.remove_ocean_average_operator(fp, load_space)
)

fig2, ax2, im2 = sl.plot(
    ocean_dynamic_topography_measure.sample() * fp.length_scale * 1000, symmetric=True
)
fig2.colorbar(
    im2, ax=ax2, orientation="horizontal", label="Ocean Dynamic Topography (mm)"
)

# %% converting to direct load

# joint measure

joint_measure = inf.GaussianMeasure.from_direct_sum(
    [ice_thickness_measure, ocean_dynamic_topography_measure]
)

# direct load operator

direct_load_operator = inf.RowLinearOperator(
    [
        sl.ice_thickness_change_to_load_operator(fp, load_space),
        sl.sea_level_change_to_load_operator(fp, load_space),
    ]
)

direct_load_measure = joint_measure.affine_mapping(operator=direct_load_operator)

fig3, ax3, im3 = sl.plot(direct_load_measure.sample(), symmetric=True)
fig3.colorbar(im3, ax=ax3, orientation="horizontal", label="Direct Load (Pa)")
fig4, ax4, im4 = sl.plot(
    direct_load_measure.sample() * fp.ocean_projection(), symmetric=True
)
fig4.colorbar(
    im4, ax=ax4, orientation="horizontal", label="Direct Load over Oceans (Pa)"
)

# %% calculate total sea surface height

# total_sea_level_change_operator = (
#     fingerprint_operator @ direct_load_operator
#     + inf.RowLinearOperator(
#         [load_space.identity_operator(), load_space.zero_operator()]
#     )
# )

total_sea_surface_height_operator = (
    response_to_sea_surface_height_operator
    @ fingerprint_operator
    @ direct_load_operator
    + inf.RowLinearOperator(
        [load_space.identity_operator(), load_space.zero_operator()]
    )
)

# sea_level_change_measure = joint_measure.affine_mapping(
#     operator=total_sea_level_change_operator
# )

sea_surface_height_measure = joint_measure.affine_mapping(
    operator=total_sea_surface_height_operator
)

sea_level_change_space = total_sea_level_change_operator.codomain
sea_surface_height_space = total_sea_surface_height_operator.codomain

fig5, ax5, im5 = sl.plot(
    sea_surface_height_measure.sample() * fp.ocean_function, symmetric=True
)
fig5.colorbar(
    im5, ax=ax5, orientation="horizontal", label="Sea Surface Height Change (m)"
)

# fig6, ax6, im6 = sl.plot(sea_level_change_measure.sample(), symmetric=True)
# fig6.colorbar(im6, ax=ax6, orientation="horizontal", label="Sea Level Change (m)")

# %% operator that maps SSH to GMSL

altimetry_projection_operator = sl.averaging_operator(
    sea_surface_height_space,
    [
        fp.altimetry_projection(latitude_min=sat_min, latitude_max=sat_max, value=0)
        / fp.integrate(
            fp.altimetry_projection(latitude_min=sat_min, latitude_max=sat_max, value=0)
        )
    ],
)

true_gmsl_operator = sl.averaging_operator(
    sea_level_change_measure.space,
    [fp.ocean_projection() / fp.integrate(fp.ocean_projection())],
)

sshc_gmsl_operator = sl.averaging_operator(
    sea_surface_height_space,
    [
        fp.altimetry_projection(latitude_min=-90, latitude_max=90, value=0)
        / fp.integrate(
            fp.altimetry_projection(latitude_min=-90, latitude_max=90, value=0)
        )
    ],
)

altimetry_estimate_measure = joint_measure.affine_mapping(
    operator=altimetry_projection_operator @ total_sea_surface_height_operator
)
true_gmsl = joint_measure.affine_mapping(
    operator=true_gmsl_operator @ total_sea_surface_height_operator
)
sshc_gmsl = joint_measure.affine_mapping(
    operator=sshc_gmsl_operator @ total_sea_surface_height_operator
)

alt_mean = altimetry_estimate_measure.expectation[0]
alt_std = np.sqrt(altimetry_estimate_measure.covariance.matrix(dense=True)[0, 0])
true_mean = true_gmsl.expectation[0]
true_std = np.sqrt(true_gmsl.covariance.matrix(dense=True)[0, 0])
sshc_mean = sshc_gmsl.expectation[0]
sshc_std = np.sqrt(sshc_gmsl.covariance.matrix(dense=True)[0, 0])

print(f"Altimetry GMSL Change: {alt_mean:.4f} ± {alt_std:.4f} m")
print(f"True GMSL Change: {true_mean:.4f} ± {true_std:.4f} m")
print(f"SSH GMSL Change: {sshc_mean:.4f} ± {sshc_std:.4f} m")

# plot distributions of GMSL estimates
xmin = min(alt_mean - 4 * alt_std, true_mean - 4 * true_std, sshc_mean - 4 * sshc_std)
xmax = max(alt_mean + 4 * alt_std, true_mean + 4 * true_std, sshc_mean + 4 * sshc_std)

x_axis = np.linspace(xmin, xmax, 100)

alt_pdf = stats.norm.pdf(x_axis, loc=alt_mean, scale=alt_std)
true_pdf = stats.norm.pdf(x_axis, loc=true_mean, scale=true_std)
sshc_pdf = stats.norm.pdf(x_axis, loc=sshc_mean, scale=sshc_std)

fig6, ax6 = plt.subplots()
ax6.plot(x_axis, alt_pdf, label="Altimetry GMSL Estimate")
ax6.plot(x_axis, true_pdf, label="True GMSL")
ax6.plot(x_axis, sshc_pdf, label="SSH GMSL Estimate")
ax6.set_xlabel("GMSL Change (m)")
ax6.set_ylabel("Probability Density")
ax6.legend()

# %% plot true gmsl
# %%
plt.show()
