# %%

import matplotlib.pyplot as plt
import pyslfp as sl

from Part_III_Project import (
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

# %%
ice_thickness_measure, ice_load_measure = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=net_ice_thickness_change,
)

# %%
direct_load_measure = load_measure(
    ice_thickness_load_measure=ice_load_measure,
    odt_load_measure=ocean_dynamic_load_measure,
)

# %%
slc = sea_level_change_measure(
    fingerprint_operator=fingerprint_operator,
    fingerprint=fp,
    load_measure=direct_load_measure,
)

# %%

ssh_measure, ssh_odt_measure, _ = sea_surface_height_measure(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    load_measure=direct_load_measure,
    odt_measure=ocean_dynamic_measure,
)

# %%

gmsl_true = get_gmsl_measure(
    measure=slc,
    fingerprint=fp,
)
gmsl_ssh_estimate = get_gmsl_measure(
    measure=ssh_measure,
    fingerprint=fp,
)
gmsl_ssh_odt_estimate = get_gmsl_measure(
    measure=ssh_odt_measure,
    fingerprint=fp,
)

# %% --- ALL PLOTTING BELOW ---

if projection_plots:
    fig1, ax1, im1 = sl.plot(
        ocean_dynamic_load_measure.sample() * fp.load_scale,
        symmetric=False,
    )
    fig1.colorbar(
        im1,
        ax=ax1,
        label="ODT Load",
        orientation="horizontal",
    )

    fig1a, ax1a, im1a = sl.plot(
        ocean_dynamic_measure.sample() * fp.length_scale,
        symmetric=False,
    )
    fig1a.colorbar(
        im1a,
        ax=ax1a,
        label="ODT Height",
        orientation="horizontal",
    )

if projection_plots:
    fig2, ax2, im2 = sl.plot(
        ice_load_measure.sample() * fp.load_scale * fp.ice_projection(),
        symmetric=False,
    )
    fig2.colorbar(
        im2,
        ax=ax2,
        label="Ice Load",
        orientation="horizontal",
    )
    fig2a, ax2a, im2a = sl.plot(
        ice_thickness_measure.sample() * fp.length_scale * fp.ice_projection(),
        symmetric=False,
    )
    fig2a.colorbar(
        im2a,
        ax=ax2a,
        label="Ice Thickness Change",
        orientation="horizontal",
    )

if projection_plots:
    fig3, ax3, im3 = sl.plot(
        direct_load_measure.sample() * fp.load_scale,
        # * fp.ocean_projection(), # option to check that there is ocean load,
        symmetric=True,
    )
    fig3.colorbar(
        im3,
        ax=ax3,
        label="Total Load",
        orientation="horizontal",
    )

if projection_plots:
    fig4, ax4, im4 = sl.plot(
        slc.sample() * fp.length_scale,
        symmetric=True,
    )
    fig4.colorbar(
        im4,
        ax=ax4,
        label="SLC",
        orientation="horizontal",
    )

# Plot SSH
fig5, ax5, im5 = sl.plot(
    ssh_measure.sample() * fp.length_scale,
    symmetric=False,
)
fig5.colorbar(
    im5,
    ax=ax5,
    label="SSH",
    orientation="horizontal",
)

fig6, ax6, im6 = sl.plot(
    ssh_odt_measure.sample() * fp.length_scale,
    symmetric=False,
)
fig6.colorbar(
    im6,
    ax=ax6,
    label="SSH + ODT",
    orientation="horizontal",
)

# %%

plot_fig, plot_ax = plot_measure(
    measures=[
        gmsl_true,
        gmsl_ssh_estimate,
        gmsl_ssh_odt_estimate,
    ],
    names=[
        "True GMSL",
        "SSH Estimated GMSL",
        "SSH+ODT Estimated GMSL",
    ],
    args={
        "Ice thickness Change": net_ice_thickness_change * fp.length_scale,
        "GMSL Ice target": ice_gmsl_target_std * fp.length_scale,
        "Ice Length Scale": ice_length_scale * fp.length_scale,
        "ODT Amplitude 95% Range": odt_amplitude_95_range * fp.length_scale,
        "ODT Length Scale": odt_length_scale * fp.length_scale,
        "Lmax": lmax,
    },
    fingerprint=fp,
)
# %%
# save plot with fstring for parameters and in "work/5-distribution_mapping/outputs/single_value_example/"
filename = (
    "work/5-distribution_mapping/outputs/single_value_params/"
    f"gmsl_comparison_"
    # f"iceLS_{ice_length_scale * fp.length_scale:.2e}_"
    # f"iceAR_{ice_gmsl_target_std * fp.length_scale:.2e}_"
    # f"netIce_{net_ice_thickness_change * fp.length_scale:.2e}_"
    # f"odtLS_{odt_length_scale * fp.length_scale:.2e}_"
    # f"odtAR_{odt_amplitude_95_range * fp.length_scale:.2e}_"
    f"lmax_{lmax}.png"
)
plot_fig.savefig(filename, dpi=300)
plt.show()

# %%
