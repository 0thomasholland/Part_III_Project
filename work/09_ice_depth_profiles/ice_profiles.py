# %%
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# %%


def depth_plot(
    formula,
):
    depths: NDArray = np.linspace(0, 5000, 100)  # m
    density: NDArray = formula(depths=depths)
    integrated_mass = np.cumsum(
        density * np.diff(depths, prepend=0)
    )  # kg/m^2
    pressure = density * 9.81 * depths  # Pa

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(11, 6),
        sharey=True,
        width_ratios=[1, 1, 1, 0.1],
    )
    axes[0].plot(density, depths)
    axes[0].set_xlabel("Density (kg/m^3)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()

    axes[1].plot(pressure, depths)
    axes[1].set_xlabel("Pressure (Pa)")
    axes[1].invert_yaxis()

    axes[2].plot(integrated_mass, depths)
    axes[2].set_xlabel("Integrated Mass (kg/m^2)")
    axes[2].invert_yaxis()

    # add a line next to the others that showns: GIS avg ~ 1.6km, GIS max ~ 3.5km, WAIS avg ~ 1.1km, WAIS max ~ 2km, EAIS avg ~2.2km, EAIS max ~ 4.8km
    axes[3].axhline(
        1600, color="blue", linestyle="--", label="GIS avg"
    )
    axes[3].axhline(
        3500, color="blue", linestyle="-", label="GIS max"
    )
    axes[3].axhline(
        1100,
        color="orange",
        linestyle="--",
        label="WAIS avg",
    )
    axes[3].axhline(
        2000,
        color="orange",
        linestyle="-",
        label="WAIS max",
    )
    axes[3].axhline(
        2200,
        color="green",
        linestyle="--",
        label="EAIS avg",
    )
    axes[3].axhline(
        4800, color="green", linestyle="-", label="EAIS max"
    )
    # move the ledgend for axes 2 to the axes 1
    axes[3].legend(loc="upper left", bbox_to_anchor=(1, 1))
    axes[3].invert_yaxis()
    return fig, axes


# %%

# test with linear density profile


def linear_density_profile(depths: NDArray) -> NDArray:
    return 1000 + 0.3 * depths


fig, axes = depth_plot(linear_density_profile)

# %%
# power-law curve


def power_law_density_profile(depths: NDArray) -> NDArray:
    _ice_density = 917  # kg/m^3
    _snow_density = 350  # kg/m^3
    _transition_depth = 40  # m
    return _ice_density - (
        _ice_density - _snow_density
    ) * np.exp(-depths / _transition_depth)


fig, axes = depth_plot(power_law_density_profile)

# %%


def herron_langway(depths: NDArray) -> NDArray:
    # Constants
    rho_i = 917.0  # Density of pure ice
    rho_w = 1000.0  # Density of water
    rho_s = 300.0  # Surface snow density (kg/m^3)
    R = 8.314  # Gas constant
    T_c = -30  # Mean annual temperature in Celsius
    A_we = 0.3  # Mean annual accumulation in m water equivalent / year
    T_k = T_c + 273.15  # Temperature in Kelvin

    k0 = 11 * np.exp(-10160 / (R * T_k))
    k1 = 575 * np.exp(-21400 / (R * T_k))

    rho_550 = 550.0
    term1_550 = np.log(rho_550 / (rho_i - rho_550))
    term2_550 = np.log(rho_s / (rho_i - rho_s))
    z_550 = (term1_550 - term2_550) / (k0 * rho_i / rho_w)

    C = term1_550 - (rho_i / rho_w) * (
        k1 * z_550 / np.sqrt(A_we)
    )

    rho = np.zeros_like(depths, dtype=float)

    mask1 = depths <= z_550
    mask2 = depths > z_550

    K1 = (rho_i / rho_w) * k0 * depths[mask1] + np.log(
        rho_s / (rho_i - rho_s)
    )
    rho[mask1] = (rho_i * np.exp(K1)) / (1 + np.exp(K1))

    K2 = (rho_i / rho_w) * (
        k1 * depths[mask2] / np.sqrt(A_we)
    ) + C
    rho[mask2] = (rho_i * np.exp(K2)) / (1 + np.exp(K2))

    return rho


fig, axes = depth_plot(herron_langway)
