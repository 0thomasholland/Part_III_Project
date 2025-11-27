# %%
import matplotlib as mpl
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

gmsl = (
    fp.integrate(sea_level_change * fp.ocean_function) / fp.ocean_area
)
gmsl_estimate = (
    fp.integrate(sea_surface_height_change * fp.ocean_function)
    / fp.ocean_area
)
error_calc = fp.integrate(error * fp.ocean_function) / fp.ocean_area
alt_error = gmsl_estimate - gmsl

print(f"GMSL: {gmsl:.6f} m")
print(f"GMSL Estimate: {gmsl_estimate:.6f} m")
print(f"Error: {error_calc:.6f} m")
print(f"Alternative Error: {alt_error:.6f} m")

error *= fp.ocean_function

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


# %%
# Plotting

plots = [
    {
        "data": ice_change,
        "title": "Ice Thickness Change",
        "mask": fp.ice_projection(),
        "cbar_label": "m",
        "symmetric_cbar": True,
    },
    {
        "data": direct_load,
        "title": "Direct Load from Ice Thickness Change",
        "mask": fp.land_projection(),
        "cbar_label": "kg/m$^2$",
        "symmetric_cbar": True,
    },
    {
        "data": sea_level_change * 1000,
        "title": "Sea Level Change (SLC) Fingerprint",
        "cbar_label": "mm",
        "mask": fp.ocean_projection(),
        "vmax": data_max * 1000,
        "vmin": data_min * 1000,
    },
    {
        "data": sea_surface_height_change * 1000,
        "title": "Sea Surface Height Change (SSHC) Fingerprint",
        "mask": fp.ocean_projection(),
        "cbar_label": "mm",
        "vmax": data_max * 1000,
        "vmin": data_min * 1000,
    },
    {
        "data": error * 1000,
        "title": "Error (SSHC - SLC)",
        "mask": fp.ocean_projection(),
        "cbar_label": "mm",
        "log_norm": True,
    },
]


for i, plot_info in enumerate(plots):
    if plot_info.get("vmax", None) is None:
        fig, ax, im = plot(
            plot_info["data"] * plot_info.get("mask", 1),
            symmetric=plot_info.get("symmetric_cbar", False),
        )
    else:
        fig, ax, im = plot(
            plot_info["data"] * plot_info.get("mask", 1),
            symmetric=plot_info.get("symmetric_cbar", False),
            vmax=plot_info["vmax"],
            vmin=plot_info["vmin"],
        )
    if plot_info.get("log_norm", False):
        im.set_norm(
            mpl.colors.SymLogNorm(
                linthresh=0.01 * np.max(np.abs(plot_info["data"])),
                linscale=0.1,
                vmin=np.min(plot_info["data"]),
                vmax=np.max(plot_info["data"]),
                base=10,
            ),
        )
    fig.colorbar(
        im,
        ax=ax,
        label=plot_info["cbar_label"],
        orientation="horizontal",
        pad=0.1,
    )
    plt.title(plot_info["title"])
    plt.tight_layout()
    plt.savefig(
        f"../../outputs/poster/AutomatedFigures/poster_simple_slc_sshc_{i + 1}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()
