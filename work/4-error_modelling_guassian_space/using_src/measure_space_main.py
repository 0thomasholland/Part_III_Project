# %%

import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
from scipy import stats

from Part_III_Project import (
    ice_thickness_change_measures,
    load_measure,
    ocean_dynamic_topography_measures,
    plot_measure,
    sea_level_change_measure,
    sea_surface_height_measure,
)

# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

# %%

ocean_dynamic_measure, ocean_dynamic_load_measure = (
    ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=0.1 * fp.mean_sea_floor_radius,
        amplitude_95_range=0.001,
    )
)

fig1, ax1, im1 = sl.plot(
    ocean_dynamic_load_measure.sample() * fp.load_scale,
)
fig1.colorbar(
    im1,
    ax=ax1,
    label="ODT Load",
    orientation="horizontal",
)

# %%
ice_thickness_measure, ice_load_measure = (
    ice_thickness_change_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=0.1 * fp.mean_sea_floor_radius,
        thickness_95_range=400,
        net_thickness_change=-100.0,
    )
)

fig2, ax2, im2 = sl.plot(
    ice_load_measure.sample() * fp.load_scale,
)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Ice Load",
    orientation="horizontal",
)

# %%
direct_load_measure = load_measure(
    ice_thickness_load_measure=ice_load_measure,
    odt_load_measure=ocean_dynamic_load_measure,
)

fig3, ax3, im3 = sl.plot(
    direct_load_measure.sample() * fp.load_scale,
    # * fp.ocean_projection(), # option to check that there is ocean load
)
fig3.colorbar(
    im3,
    ax=ax3,
    label="Total Load",
    orientation="horizontal",
)

# %%

sea_level_change_measure = sea_level_change_measure(
    fingerprint_operator=fingerprint_operator,
    load_measure=direct_load_measure,
)

fig4, ax4, im4 = sl.plot(
    sea_level_change_measure.sample() * fp.length_scale,
)
fig3.colorbar(
    im3,
    ax=ax3,
    label="SLC",
    orientation="horizontal",
)

# %%

ssh_measure, ssh_odt_measure, ssh_odt_noise_measure = (
    sea_surface_height_measure(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        odt_measure=ocean_dynamic_measure,
    )
)

plot_fig, plot_ax = plot_measure(
    [
        sea_level_change_measure,
        ssh_measure,
        ssh_odt_measure,
        ssh_odt_noise_measure,
    ],
)
# %%

plt.show()
