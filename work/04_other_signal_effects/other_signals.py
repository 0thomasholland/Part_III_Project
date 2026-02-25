# %%

import cartopy.feature as cfeature
import numpy as np
from cartopy import crs as ccrs
from matplotlib import pyplot as plt
from pygeoinf import GaussianMeasure, LinearOperator
from pyslfp import (
    FingerPrint,
    IceModel,
    plot,
    sea_surface_height_operator,
)
from pyslfp_extras.measures import (
    non_ice_ssh_variability_field,
    non_ice_ssh_variability_fingerprint_ssh_measure,
    non_ice_ssh_variability_gaussian_measure,
    non_ice_ssh_variability_total_ssh_measure,
)

# %%

DATA_FILE = (
    "../../data/noise_file/non_ice_ssh_variability.nc"
)

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%
# Plot the empirically-derived spatial variability field
variability_field = non_ice_ssh_variability_field(
    finger_print=fp,
    variability_path=DATA_FILE,
)

fig, ax, im = plot(
    variability_field,
    coasts=True,
    cmap="Greys",
    colorbar_label="Relative variability (normalised, mean=1)",
)
ax.set_title(
    "Non-ice SSH variability field (DUACS-derived)"
)


# %%
# Uniform (no spatial variability)
non_ice_ssh_variability_uniform: GaussianMeasure = (
    non_ice_ssh_variability_gaussian_measure(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=fp.mean_sea_floor_radius * 0.2,
        amplitude=0.0002,
        point_multiplier=3000,
    )
)

non_ice_ssh_variability_uniform_sample = (
    non_ice_ssh_variability_uniform.sample()
)

# %%
# Spatially varying (empirically derived from DUACS)
non_ice_ssh_variability_variable: GaussianMeasure = (
    non_ice_ssh_variability_gaussian_measure(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=fp.mean_sea_floor_radius * 0.2,
        use_spatial_variability=True,
        amplitude=0.0002,
        point_multiplier=3000,
        variability_path=DATA_FILE,
    )
)

non_ice_ssh_variability_variable_sample = (
    non_ice_ssh_variability_variable.sample()
)


fig2, axes = plt.subplots(
    2,
    3,
    figsize=(18, 10),
    subplot_kw={"projection": ccrs.Robinson()},
)

for i in range(3):
    s_uniform = (
        non_ice_ssh_variability_uniform.sample()
        * fp.ocean_projection()
        * 1000
        * fp.length_scale
    )
    s_variable = (
        non_ice_ssh_variability_variable.sample()
        * fp.ocean_projection()
        * 1000
        * fp.length_scale
    )

    vmax = max(
        np.nanmax(np.abs(s_uniform.to_array())),
        np.nanmax(np.abs(s_variable.to_array())),
    )

    lons = s_uniform.lons()
    lats = s_uniform.lats()

    ax1 = axes[0, i]
    im1 = ax1.pcolormesh(
        lons,
        lats,
        s_uniform.to_array(),
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    ax1.add_feature(cfeature.LAND, color="lightgray")
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax1.set_title("Uniform variability sample")

    ax2 = axes[1, i]
    im2 = ax2.pcolormesh(
        lons,
        lats,
        s_variable.to_array(),
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )
    ax2.add_feature(cfeature.LAND, color="lightgray")
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.4)
    ax2.set_title(
        "Spatially variable (DUACS-derived) variability sample"
    )

    # colorbar

    cbar = fig2.colorbar(
        im2,
        ax=[ax1, ax2],
        orientation="horizontal",
        pad=0.05,
        label="Non-ice SSH variability (mm)",
    )


plt.show()
