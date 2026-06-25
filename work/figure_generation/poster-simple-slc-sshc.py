# %%
from pyslfp import LinearSeaLevelEquation
from pyslfp.linear_operators.physics import (
    centrifugal_potential_operator,
)
from pyslfp.state import EarthState
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
from scipy.stats import norm

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.size"] = 24

# %%
# Setup
lmax = 256
fp = EarthState.from_defaults(lmax=lmax)

# %%

# Create a disk load approx over Greenland (lat: 65N, lon: -40E) with radius of 10 degrees and height change of -100 m
ice_change = fp.disk_load(
    7.0,
    65.0,
    -45.0,
    -10.0,
) * fp.ice_projection(value=0)
direct_load = fp.direct_load_from_ice_thickness_change(ice_change)

# Compute the sea level change fingerprint

(
    sea_level_change,
    displacement,
    gravity_potential_change,
    angular_velocity_change,
) = LinearSeaLevelEquation(fp).solve_sea_level_equation(direct_load,
)

sea_surface_height_change = (sea_level_change + displacement + centrifugal_potential_operator(fp.model)(angular_velocity_change,
) / fp.model.parameters.gravitational_acceleration)

sea_level_change *= fp.ocean_function
sea_surface_height_change *= fp.ocean_function
error = sea_surface_height_change - sea_level_change

gmsl = fp.model.integrate(sea_level_change) / fp.ocean_area
gmsl_estimate = (
    fp.model.integrate(sea_surface_height_change) / fp.ocean_area
)
error_calc = fp.model.integrate(error) / fp.ocean_area
alt_error = gmsl_estimate - gmsl

print(f"GMSL: {gmsl * 1000:.4f} mm")
print(f"GMSL Estimate: {gmsl_estimate * 1000:.4f} mm")
print(f"Error: {error_calc * 1000:.4f} mm")

absolute_error_mean = (
    fp.model.integrate(np.abs(error) * fp.ocean_function) / fp.ocean_area
)

print(f"Mean Absolute Error: {absolute_error_mean * 1000:.4f} mm")

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
data_abs = 0.25 * 1000

print(f"SLC min: {(sea_level_change).min() * 1000:.2e} mm")
print(f"SLC max: {sea_level_change.max() * 1000:.2e} mm")
print(f"SSHC min: {sea_surface_height_change.min() * 1000:.2e} mm")
print(f"SSHC max: {sea_surface_height_change.max() * 1000:.2e} mm")
print(f"Universal colorbar range: ±{data_abs:.2e} mm")

# %%

norm = mcolors.TwoSlopeNorm(
    vmin=-0.25 * 1000,
    vcenter=0,
    vmax=0.03 * 1000,
)

# %%
# Plotting

fig_ice, ax_ice, im_ice = plot(
    ice_change * fp.ice_projection(),
    symmetric=True,
    gridlines=False,
    figsize=(10 * 0.8, 8 * 0.8),
)
fig_ice.set_facecolor((1, 1, 1, 0.0))
cbar_ice = fig_ice.colorbar(
    im_ice,
    ax=ax_ice,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_ice.set_label("Ice Thickness Change (m)")

fig_load, ax_load, im_load = plot(
    direct_load * 1e-6 * fp.ice_projection(),
    symmetric=True,
    gridlines=False,
)
fig_load.set_facecolor((1, 1, 1, 0.0))
cbar_load = fig_load.colorbar(
    im_load,
    ax=ax_load,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_load.set_label("Load Change (MPa)")

fig_displacement, ax_displacement, im_displacement = plot(
    displacement,
    symmetric=True,
    gridlines=False,
)
fig_displacement.set_facecolor((1, 1, 1, 0.0))
cbar_displacement = fig_displacement.colorbar(
    im_displacement,
    ax=ax_displacement,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_displacement.set_label("Vertical Displacement (m)")

fig_slc, ax_slc, im_slc = plot(
    sea_level_change * fp.ocean_projection() * 1000,
    norm=norm,
    gridlines=False,
)
fig_slc.set_facecolor((1, 1, 1, 0.0))
cbar_slc = fig_slc.colorbar(
    im_slc,
    ax=ax_slc,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_slc.set_ticks(
    [-0.25 * 1000, -0.125 * 1000, 0, 0.015 * 1000, 0.03 * 1000],
)
cbar_slc.set_label(
    f"Sea Level Change (mm)\nGMSL Change = {gmsl * 1000:.1f} mm",
)

fig_sshc, ax_sshc, im_sshc = plot(
    sea_surface_height_change * fp.ocean_projection() * 1000,
    norm=norm,
    gridlines=False,
)
fig_sshc.set_facecolor((1, 1, 1, 0.0))
cbar_sshc = fig_sshc.colorbar(
    im_sshc,
    ax=ax_sshc,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_sshc.set_ticks(
    [-0.25 * 1000, -0.125 * 1000, 0, 0.015 * 1000, 0.03 * 1000],
)
cbar_sshc.set_label(
    f"Sea Surface Height Change (mm)\nGMSL Estimated Change = {gmsl_estimate * 1000:.1f} mm",
)
# %%
fig_error, ax_error, im_error = plot(
    error * fp.ocean_projection() * 1000,
    symmetric=True,
    vmax=150,
    vmin=-150,
    gridlines=False,
)
fig_error.set_facecolor((1, 1, 1, 0.0))
cbar_error = fig_error.colorbar(
    im_error,
    ax=ax_error,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
)
cbar_error.set_label(
    "Error: SSHC - SLC (mm)",
)

# %%
# save all the figures at 600 dpi in ../../outputs/posters/AutomatedFigures/
output_path = "../../outputs/poster/AutomatedFigures/Simple/"
fig_ice.savefig(
    output_path + "1-ice_change.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
fig_load.savefig(
    output_path + "2-direct_load.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
fig_displacement.savefig(
    output_path + "3-vertical_displacement.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
fig_slc.savefig(
    output_path + "4-sea_level_change.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
fig_sshc.savefig(
    output_path + "5-sea_surface_height_change.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
fig_error.savefig(
    output_path + "6-error_sshc_slc.png",
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
)
