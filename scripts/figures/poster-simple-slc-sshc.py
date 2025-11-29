# %%
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    plot,
)
from scipy.stats import norm

mpl.rcParams["figure.dpi"] = 600

# %%
# Setup
lmax = 256
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()

# %%

# Create a disk load approx over Greenland (lat: 65N, lon: -40E) with radius of 10 degrees and height change of -100 m
ice_change = fp.disk_load(
    7.0,
    65.0,
    -45.0,
    -100.0,
) * fp.ice_projection(value=0)
direct_load = fp.direct_load_from_ice_thickness_change(ice_change)

# Compute the sea level change fingerprint

(
    sea_level_change,
    displacement,
    gravity_potential_change,
    angular_velocity_change,
) = fp(
    direct_load=direct_load,
)

sea_surface_height_change = fp.sea_surface_height_change(
    sea_level_change,
    displacement,
    angular_velocity_change,
)

sea_level_change *= fp.ocean_function
sea_surface_height_change *= fp.ocean_function
error = sea_surface_height_change - sea_level_change

gmsl = fp.integrate(sea_level_change) / fp.ocean_area
gmsl_estimate = (
    fp.integrate(sea_surface_height_change) / fp.ocean_area
)
error_calc = fp.integrate(error) / fp.ocean_area
alt_error = gmsl_estimate - gmsl

print(f"GMSL: {gmsl:.6f} m")
print(f"GMSL Estimate: {gmsl_estimate:.6f} m")
print(f"Error: {error_calc:.6f} m")
print(f"Alternative Error: {alt_error:.6f} m")


# %%

# for SLC and SSHC find the min and max and set a universal colorbar range
data_min = min(
    sea_level_change.min(),
    sea_surface_height_change.min(),
)
data_max = max(
    sea_level_change.max(),
    sea_surface_height_change.max(),
)
data_abs = 2.5

print(f"SLC min: {(sea_level_change).min():.2e} m")
print(f"SLC max: {sea_level_change.max():.2e} m")
print(f"SSHC min: {sea_surface_height_change.min():.2e} m")
print(f"SSHC max: {sea_surface_height_change.max():.2e} m")
print(f"Universal colorbar range: ±{data_abs:.2e} m")

# %%

norm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=0.3)

# %%
# Plotting


fig_ice, ax_ice, im_ice = plot(
    ice_change * fp.ice_projection(),
    symmetric=True,
)
cbar_ice = fig_ice.colorbar(
    im_ice,
    ax=ax_ice,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_ice.set_label("Ice Thickness Change (m)")

fig_load, ax_load, im_load = plot(
    direct_load * 1e-6 * fp.ice_projection(),
    symmetric=True,
)
cbar_load = fig_load.colorbar(
    im_load,
    ax=ax_load,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_load.set_label("Load Change (MPa)")

fig_displacement, ax_displacement, im_displacement = plot(
    displacement,
    symmetric=True,
)
cbar_displacement = fig_displacement.colorbar(
    im_displacement,
    ax=ax_displacement,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_displacement.set_label("Vertical Displacement (m)")


fig_slc, ax_slc, im_slc = plot(
    sea_level_change * fp.ocean_projection(),
    norm=norm,
)
cbar_slc = fig_slc.colorbar(
    im_slc,
    ax=ax_slc,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_slc.set_ticks([-2.5, -1.25, 0, 0.3])
cbar_slc.set_label("Sea Level Change (m)")

fig_sshc, ax_sshc, im_sshc = plot(
    sea_surface_height_change * fp.ocean_projection(),
    norm=norm,
)
cbar_sshc = fig_sshc.colorbar(
    im_sshc,
    ax=ax_sshc,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_sshc.set_ticks([-2.5, -1.25, 0, 0.3])
cbar_sshc.set_label("Sea Surface Height Change (m)")

# %%
fig_error, ax_error, im_error = plot(
    error * fp.ocean_projection(),
    symmetric=True,
    vmax=1.5,
    vmin=-1.5,
)
cbar_error = fig_error.colorbar(
    im_error,
    ax=ax_error,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cbar_error.set_label("Error (m) [SSHC - SLC]")

# %%
# save all the figures at 600 dpi in ../../outputs/posters/AutomatedFigures/
output_path = "../../outputs/poster/AutomatedFigures/Simple/"
fig_ice.savefig(output_path + "1-ice_change.png", dpi=600)
fig_load.savefig(output_path + "2-direct_load.png", dpi=600)
fig_displacement.savefig(
    output_path + "3-vertical_displacement.png",
    dpi=600,
)
fig_slc.savefig(output_path + "4-sea_level_change.png", dpi=600)
fig_sshc.savefig(
    output_path + "5-sea_surface_height_change.png",
    dpi=600,
)
fig_error.savefig(output_path + "6-error_sshc_slc.png", dpi=600)
