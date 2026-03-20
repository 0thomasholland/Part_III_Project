# Auto-generated from notebook code cells.
# Source: notebooks/04 - Ice and Firn.ipynb

from pathlib import Path

import cartopy.crs as ccrs
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
from pyshtools import SHGrid
from pyslfp import FingerPrint, IceModel

from project import colors
from project.projections import (
    EXTENT_ANTARCTICA,
    EXTENT_GREENLAND,
    PROJ_ANTARCTICA,
    PROJ_GREENLAND,
)
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.plotting import plot

np.random.seed(423991)


SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _noop(*_args, **_kwargs):
    return None


plt.show = _noop


def plot_shgrid_robinson_on_ax(
    shgrid: SHGrid,
    ax,
    *,
    cmap: str = "RdBu",
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    data = np.asarray(shgrid.data)
    lons = np.asarray(shgrid.lons())
    lats = np.asarray(shgrid.lats())

    if symmetric and vmin is None and vmax is None:
        max_abs_value = np.nanmax(np.abs(data))
        vmin = -max_abs_value
        vmax = max_abs_value
    elif vmin is None or vmax is None:
        raise ValueError(
            "If symmetric=False, provide vmin and vmax."
        )

    im = ax.pcolormesh(
        lons,
        lats,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.coastlines(linewidth=0.8)
    ax.set_global()
    return im


def plot_shgrid_polar_on_ax(
    shgrid: SHGrid,
    ax,
    extent: list[float],
    *,
    cmap: str = "RdBu",
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    data = np.asarray(shgrid.data)
    lons = np.asarray(shgrid.lons())
    lats = np.asarray(shgrid.lats())

    if symmetric and vmin is None and vmax is None:
        max_abs_value = np.nanmax(np.abs(data))
        vmin = -max_abs_value
        vmax = max_abs_value
    elif vmin is None or vmax is None:
        raise ValueError(
            "If symmetric=False, provide vmin and vmax."
        )

    im = ax.pcolormesh(
        lons,
        lats,
        data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.coastlines(linewidth=0.8)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return im


def activator_richards(x, x_min, x_max):
    """Richards activation function for ice/firn melt probability.

    Standardizes input: 0 at min thickness, 1 at max thickness.
    Parameters define a clean 0-to-1 probability curve based on ice thickness.
    """
    _x = (x - x_min) / (x_max - x_min)

    # Parameters for a clean 0-to-1 probability curve
    a = 0.1  # Lower asymptote (Thick ice = 0 probability)
    k = 0.9  # Upper asymptote (Thin ice = 1 probability)
    b = 10.0  # Steepness
    m = 0.45  # Threshold (where the drop-off happens)
    nu = 0.75  # Asymmetry (adjusts how 'sharp' the turn is)

    # Note: We use (_x - m) to make probability drop as thickness increases
    _x = a + (k - a) / (1 + np.exp(b * (_x - m))) ** (
        1 / nu
    )
    return _x


def setup_ice_models():
    lmax = 256
    fp = FingerPrint(lmax=lmax)
    fp.set_state_from_ice_ng(
        version=IceModel.ICE7G, date=0.0
    )
    fp_op = fp.as_sobolev_linear_operator(
        2, fp.mean_sea_floor_radius * 0.1
    )

    ice_change_uniform = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=0.01 * fp.mean_sea_floor_radius,
        pattern=IceSheetChange.UniformPattern(),
        include_firn=False,
        ice_gmsl_std=0.02,
    )

    weighted_pattern = (
        IceSheetChange.ThicknessWeightedPattern(
            lower_asymptote=0.1,
            upper_asymptote=0.9,
            steepness=10.0,
            threshold=0.45,
            asymmetry=0.75,
        )
    )

    ice_change_spatial = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=0.01 * fp.mean_sea_floor_radius,
        pattern=weighted_pattern,
        ice_gmsl_std=0.02,
        firn_gmsl_std=0.015,
        include_firn=True,
        ice_density=fp.ice_density,
        firn_density=0.3 * fp.ice_density,
    )

    return (
        fp,
        weighted_pattern,
        ice_change_uniform,
        ice_change_spatial,
    )


def save_uniform_field_plot(uniform_sample):
    fig, ax, _ = plot(
        uniform_sample.ice_thickness,
        symmetric=True,
        colorbar_label="Uniform-prior ice thickness change (m)",
    )
    ax.set_title("Sample from Uniform Ice Prior")
    fig.savefig(
        FIGURES_DIR / "04_uniform_field.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_variability_side_by_side(weighted_pattern, fp):
    ice_std_field = weighted_pattern.spatial_weights(fp)
    firn_std_field = weighted_pattern.firn_weights(fp)

    fig = plt.figure(figsize=(6.5, 5))
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.4,
        wspace=0.2,
        left=0.08,
        right=0.95,
        top=0.93,
        bottom=0.15,
    )

    # Top left: Activator function
    ax_activator = fig.add_subplot(gs[0, 0])
    ax_activator.set_box_aspect(1)

    # Top right: Ice thickness field
    ax_ice_thickness = fig.add_subplot(
        gs[0, 1], projection=ccrs.Robinson()
    )

    # Bottom left: Firn std field
    ax_firn = fig.add_subplot(
        gs[1, 0], projection=ccrs.Robinson()
    )

    # Bottom right: Ice std field
    ax_ice = fig.add_subplot(
        gs[1, 1], projection=ccrs.Robinson()
    )

    # Plot activator function
    data = fp.ice_thickness.data.flatten()
    input_range = np.linspace(data.min(), data.max(), 100)
    ice_melt = activator_richards(
        input_range, data.min(), data.max()
    )
    firn_melt = 1 - ice_melt

    ax_activator.plot(
        input_range,
        ice_melt,
        label="Ice function",
        color="black",
    )
    ax_activator.plot(
        input_range,
        firn_melt,
        label="Firn function",
        color="black",
        linestyle="dashed",
    )
    ax_activator.legend(loc="upper right")
    ax_activator.set_xlabel("Ice thickness (m)")
    ax_activator.set_ylim(-0.0, 1.0)
    ax_activator.set_ylabel(
        "Change field std dev multiplier"
    )
    ax_activator.set_title("Activation Function")

    # Plot ice thickness field
    im_thickness = plot_shgrid_robinson_on_ax(
        fp.ice_thickness,
        ax_ice_thickness,
        cmap=cc.cm.blues,
        symmetric=False,
        vmin=0.0,
        vmax=4000.0,
    )
    ax_ice_thickness.set_title("Present-day Ice Thickness")
    fig.colorbar(
        im_thickness,
        ax=ax_ice_thickness,
        orientation="horizontal",
        pad=0.04,
        shrink=0.85,
        label="Ice thickness (m)",
    )

    # Plot firn std field
    im_firn = plot_shgrid_robinson_on_ax(
        firn_std_field * fp.ice_projection(),
        ax_firn,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )
    ax_firn.set_title(
        "Firn change pointwise standard deviation field"
    )
    fig.colorbar(
        im_firn,
        ax=ax_firn,
        orientation="horizontal",
        pad=0.04,
        shrink=0.85,
        label="Firn variability multiplier",
    )

    # Plot ice std field
    im_ice = plot_shgrid_robinson_on_ax(
        ice_std_field * fp.ice_projection(),
        ax_ice,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )
    ax_ice.set_title(
        "Ice change pointwise standard deviation field"
    )
    fig.colorbar(
        im_ice,
        ax=ax_ice,
        orientation="horizontal",
        pad=0.04,
        shrink=0.85,
        label="Ice variability multiplier",
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variability_fields_side_by_side.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_variability_polar_grid(weighted_pattern, fp):
    ice_std_field = weighted_pattern.spatial_weights(fp)
    firn_std_field = weighted_pattern.firn_weights(fp)

    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(
        2,
        2,
        hspace=0.2,
        wspace=0.1,
        left=0.08,
        right=0.95,
        top=0.93,
        bottom=0.15,
    )

    proj_greenland = PROJ_GREENLAND
    proj_antarctica = PROJ_ANTARCTICA

    ax_firn_gl = fig.add_subplot(
        gs[0, 0], projection=proj_greenland
    )
    ax_firn_ant = fig.add_subplot(
        gs[0, 1], projection=proj_antarctica
    )
    ax_ice_gl = fig.add_subplot(
        gs[1, 0], projection=proj_greenland
    )
    ax_ice_ant = fig.add_subplot(
        gs[1, 1], projection=proj_antarctica
    )

    extent_gl = EXTENT_GREENLAND
    extent_ant = EXTENT_ANTARCTICA

    im_firn_gl = plot_shgrid_polar_on_ax(
        firn_std_field * fp.ice_projection(),
        ax_firn_gl,
        extent_gl,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )
    im_firn_ant = plot_shgrid_polar_on_ax(
        firn_std_field * fp.ice_projection(),
        ax_firn_ant,
        extent_ant,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )

    im_ice_gl = plot_shgrid_polar_on_ax(
        ice_std_field * fp.ice_projection(),
        ax_ice_gl,
        extent_gl,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )
    im_ice_ant = plot_shgrid_polar_on_ax(
        ice_std_field * fp.ice_projection(),
        ax_ice_ant,
        extent_ant,
        cmap=cc.cm.kbc,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )

    ax_firn_gl.set_title("Firn variability (Greenland)")
    ax_firn_ant.set_title("Firn variability (Antarctica)")
    ax_ice_gl.set_title("Ice variability (Greenland)")
    ax_ice_ant.set_title("Ice variability (Antarctica)")

    fig.colorbar(
        im_firn_gl,
        ax=[ax_firn_gl, ax_firn_ant],
        orientation="horizontal",
        pad=0.04,
        shrink=0.5,
        label="Firn variability multiplier",
    )
    fig.colorbar(
        im_ice_gl,
        ax=[ax_ice_gl, ax_ice_ant],
        orientation="horizontal",
        pad=0.04,
        shrink=0.5,
        label="Ice variability multiplier",
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variability_fields_polar_grid.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_variable_thickness_load_grid(samples):
    thickness_max = max(
        np.abs(samples.firn_thickness).max(),
        np.abs(samples.ice_thickness).max(),
        np.abs(samples.total_thickness).max(),
    )
    load_max = max(
        np.abs(samples.firn_load).max(),
        np.abs(samples.ice_load).max(),
        np.abs(samples.total_load).max(),
    )

    thickness_fields = [
        samples.ice_thickness,
        samples.firn_thickness,
        samples.total_thickness,
    ]
    load_fields = [
        samples.ice_load,
        samples.firn_load,
        samples.total_load,
    ]
    row_labels = ["Ice", "Firn", "Total"]

    fig = plt.figure(figsize=(6.5, 6.0))
    gs = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.0, 1.0, 0.085],
        wspace=0.02,
        hspace=0.08,
    )

    axes = []
    for row in range(3):
        for col in range(2):
            axes.append(
                fig.add_subplot(
                    gs[row, col], projection=ccrs.Robinson()
                )
            )

    cax_th = fig.add_subplot(gs[3, 0])
    cax_ld = fig.add_subplot(gs[3, 1])

    thickness_ims = []
    load_ims = []

    for row, label in enumerate(row_labels):
        ax_th = axes[2 * row]
        ax_ld = axes[2 * row + 1]

        im_th = plot_shgrid_robinson_on_ax(
            thickness_fields[row],
            ax_th,
            cmap="RdBu",
            symmetric=False,
            vmin=-thickness_max,
            vmax=thickness_max,
        )
        im_ld = plot_shgrid_robinson_on_ax(
            load_fields[row],
            ax_ld,
            cmap="RdBu",
            symmetric=False,
            vmin=-load_max,
            vmax=load_max,
        )

        thickness_ims.append(im_th)
        load_ims.append(im_ld)

        ax_th.set_title(
            f"{label} thickness change", fontsize=10, pad=2
        )
        ax_ld.set_title(
            f"{label} load change", fontsize=10, pad=2
        )

    fig.colorbar(
        thickness_ims[-1],
        cax=cax_th,
        orientation="horizontal",
        label="Thickness change (m)",
        shrink=0.85,
    )
    fig.colorbar(
        load_ims[-1],
        cax=cax_ld,
        orientation="horizontal",
        label="Load change (kg/m²)",
        shrink=0.85,
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variable_fields_thickness_load_grid.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_variable_thickness_load_polar_grid(samples):
    thickness_max = max(
        np.abs(samples.firn_thickness).max(),
        np.abs(samples.ice_thickness).max(),
        np.abs(samples.total_thickness).max(),
    )
    load_max = max(
        np.abs(samples.firn_load).max(),
        np.abs(samples.ice_load).max(),
        np.abs(samples.total_load).max(),
    )

    thickness_fields = [
        samples.ice_thickness,
        samples.firn_thickness,
        samples.total_thickness,
    ]
    load_fields = [
        samples.ice_load,
        samples.firn_load,
        samples.total_load,
    ]
    row_labels = ["Ice", "Firn", "Total"]

    fig = plt.figure(figsize=(10.0, 8.0))
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.0, 1.0, 1.0, 0.085],
        wspace=0.05,
        hspace=0.1,
    )

    proj_greenland = PROJ_GREENLAND
    proj_antarctica = PROJ_ANTARCTICA

    axes = []
    for row in range(3):
        row_axes = []
        for col in range(2):
            row_axes.append(
                fig.add_subplot(
                    gs[row, col * 2],
                    projection=proj_greenland,
                )
            )
            row_axes.append(
                fig.add_subplot(
                    gs[row, col * 2 + 1],
                    projection=proj_antarctica,
                )
            )
        axes.append(row_axes)

    cax_th = fig.add_subplot(gs[3, 0:2])
    cax_ld = fig.add_subplot(gs[3, 2:4])

    thickness_ims = []
    load_ims = []

    for row, label in enumerate(row_labels):
        ax_th_gl = axes[row][0]
        ax_th_ant = axes[row][1]
        ax_ld_gl = axes[row][2]
        ax_ld_ant = axes[row][3]

        im_th_gl = plot_shgrid_polar_on_ax(
            thickness_fields[row],
            ax_th_gl,
            EXTENT_GREENLAND,
            cmap="RdBu",
            symmetric=False,
            vmin=-thickness_max,
            vmax=thickness_max,
        )
        im_th_ant = plot_shgrid_polar_on_ax(
            thickness_fields[row],
            ax_th_ant,
            EXTENT_ANTARCTICA,
            cmap="RdBu",
            symmetric=False,
            vmin=-thickness_max,
            vmax=thickness_max,
        )
        im_ld_gl = plot_shgrid_polar_on_ax(
            load_fields[row],
            ax_ld_gl,
            EXTENT_GREENLAND,
            cmap="RdBu",
            symmetric=False,
            vmin=-load_max,
            vmax=load_max,
        )
        im_ld_ant = plot_shgrid_polar_on_ax(
            load_fields[row],
            ax_ld_ant,
            EXTENT_ANTARCTICA,
            cmap="RdBu",
            symmetric=False,
            vmin=-load_max,
            vmax=load_max,
        )

        thickness_ims.append(im_th_gl)
        load_ims.append(im_ld_gl)

        if row == 0:
            ax_th_gl.set_title(
                "Thickness (Greenland)", fontsize=10, pad=2
            )
            ax_th_ant.set_title(
                "Thickness (Antarctica)", fontsize=10, pad=2
            )
            ax_ld_gl.set_title(
                "Load (Greenland)", fontsize=10, pad=2
            )
            ax_ld_ant.set_title(
                "Load (Antarctica)", fontsize=10, pad=2
            )

        ax_th_gl.text(
            -0.1,
            0.5,
            label,
            va="center",
            ha="right",
            rotation="vertical",
            transform=ax_th_gl.transAxes,
            fontsize=12,
            fontweight="bold",
        )

    fig.colorbar(
        thickness_ims[-1],
        cax=cax_th,
        orientation="horizontal",
        label="Thickness change (m)",
        shrink=0.85,
    )
    fig.colorbar(
        load_ims[-1],
        cax=cax_ld,
        orientation="horizontal",
        label="Load change (kg/m²)",
        shrink=0.85,
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variable_fields_thickness_load_polar_grid.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_activator_function_plot(fp):
    """Save plot of ice and firn melt activation functions."""
    data = fp.ice_thickness.data.flatten()

    input_range = np.linspace(data.min(), data.max(), 100)
    ice_melt = activator_richards(
        input_range, data.min(), data.max()
    )
    firn_melt = 1 - ice_melt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        input_range,
        ice_melt,
        label="Ice function",
        color="black",
    )
    ax.plot(
        input_range,
        firn_melt,
        label="Firn function",
        color="black",
        linestyle="dashed",
    )
    ax.legend(loc="upper right")
    ax.set_xlabel("Input (ice thickness in m)")
    ax.set_ylim(-0.0, 1.0)
    ax.set_ylabel(
        "Output (change field standard deviation multiplier)"
    )

    fig.savefig(
        FIGURES_DIR / "04_activator_function.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    (
        fp,
        weighted_pattern,
        ice_change_uniform,
        ice_change_spatial,
    ) = setup_ice_models()

    uniform_sample = ice_change_uniform.sample()
    samples = ice_change_spatial.sample()

    save_uniform_field_plot(uniform_sample)
    save_variability_side_by_side(weighted_pattern, fp)
    save_variability_polar_grid(weighted_pattern, fp)
    save_variable_thickness_load_grid(samples)
    save_variable_thickness_load_polar_grid(samples)
    save_activator_function_plot(fp)


if __name__ == "__main__":
    main()
