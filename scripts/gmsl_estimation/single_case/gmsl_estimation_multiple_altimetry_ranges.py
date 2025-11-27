# %%

import matplotlib.pyplot as plt
import pyslfp as sl

from Part_III_Project import (
    get_altimetry_gmsl_measure,
    get_gmsl_measure,
    ice_thickness_change_measures,
    load_measure,
    ocean_dynamic_topography_measures,
    plot_measure,
    sea_level_change_measure,
    sea_surface_height_measure,
)

# --- Control switches ---
projection_plots = False  # Set to False to disable projection plots
lmax = 256
# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(
    lmax=lmax,
    # earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)
# %%


#### VARIABLES

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_amplitude_95_range = (
    1 / fp.length_scale
)  # in units of sea level, non-dimensionalized

ice_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.004 / fp.length_scale  # in meters, non-dimensionalized
net_ice_thickness_change = -10.0 / fp.length_scale  # in meters, non-dimensionalized


# %%

ocean_dynamic_measure, ocean_dynamic_load_measure = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    amplitude_95_range=odt_amplitude_95_range,
)

ice_thickness_measure, ice_load_measure = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=net_ice_thickness_change,
)

direct_load_measure = load_measure(
    ice_thickness_load_measure=ice_load_measure,
    odt_load_measure=ocean_dynamic_load_measure,
)

slc = sea_level_change_measure(
    fingerprint_operator=fingerprint_operator,
    fingerprint=fp,
    load_measure=direct_load_measure,
)
# %%

gmsl_alt_66_measure = get_altimetry_gmsl_measure(
    measure=slc,
    fingerprint=fp,
    altimetry_range=66,
)
gmsl_measure = get_gmsl_measure(
    measure=slc,
    fingerprint=fp,
)
gmsl_alt_55_measure = get_altimetry_gmsl_measure(
    measure=slc,
    fingerprint=fp,
    altimetry_range=55,
)
gmsl_alt_77_measure = get_altimetry_gmsl_measure(
    measure=slc,
    fingerprint=fp,
    altimetry_range=77,
)
# %%

fig, ax, im = sl.plot(
    slc.sample()
    * fp.length_scale
    * fp.ocean_function
    * fp.altimetry_projection(latitude_min=-77, latitude_max=77),
)
fig.colorbar(
    im,
    ax=ax,
    label="Sea Level Change (m)",
    orientation="horizontal",
)
plt.show()


# %%
plot_measure(
    measures=[
        gmsl_measure,
        gmsl_alt_66_measure,
        gmsl_alt_55_measure,
        gmsl_alt_77_measure,
    ],
    names=["GMSL", "Altimetry 66°", "Altimetry 55°", "Altimetry 77°"],
    args={},
    fingerprint=fp,
)
