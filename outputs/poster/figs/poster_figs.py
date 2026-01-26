# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

# %%
# Setup
lmax = 64
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

ice_thickness_sample = ice_thickness_change.sample()
odt_sample = odt_change.sample()

ice_load_sample = fp.direct_load_from_ice_thickness_change(ice_thickness_sample)
odt_load_sample = fp.direct_load_from_sea_level_change(odt_sample)


altimetry_error_sample = measurement_error.sample()

odt_fingerprint_sampled = fp(direct_load=odt_load_sample)
ice_fingerprint_sampled = fp(direct_load=ice_load_sample)

true_slc_sample = ice_fingerprint_sampled[0]

odt_ssh_sampled = fp.sea_surface_height_change(
    odt_fingerprint_sampled[0], odt_fingerprint_sampled[1], odt_fingerprint_sampled[3]
)
ice_ssh_sampled = fp.sea_surface_height_change(
    ice_fingerprint_sampled[0], ice_fingerprint_sampled[1], ice_fingerprint_sampled[3]
)

error_sample = odt_sample + altimetry_error_sample + odt_ssh_sampled

# total_ssh_sample = (
#     odt_sample + altimetry_error_sample + odt_ssh_sampled + ice_ssh_sampled
# )


def plot_map(data, title, file_name=None):
    _fig, _ax, _im = plot(data, cmap="coolwarm")
    _ax.set_title(title)
    if file_name is not None:
        plt.savefig(f"{file_name}.pdf", dpi=600)
        plt.savefig(f"{file_name}.png", dpi=600)


plot_map(
    ice_thickness_sample * fp.ice_projection(),
    "Ice Thickness Change Sample (m)",
)
plot_map(
    odt_sample * fp.ocean_projection(),
    "Ocean Dynamic Topography Change Sample (m)",
)
plot_map(
    true_slc_sample * fp.ocean_projection(),
    "True Sea Level Change Sample (m)",
)
plot_map(
    error_sample * fp.ocean_projection(),
    "Altimetry Observation Error Sample (m)",
)
# plot_map(
#    total_ssh_sample * fp.ocean_projection(),
#   "Total Sea Surface Height Change Sample (m)",
# )
