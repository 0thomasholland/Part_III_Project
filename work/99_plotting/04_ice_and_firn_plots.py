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

from pyslfp_extras.ice_thickness import IceSheetChange

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
    fig = plt.figure(figsize=(11.5, 6.5))
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.Robinson()
    )

    im = plot_shgrid_robinson_on_ax(
        uniform_sample.ice_thickness,
        ax,
        cmap=cc.cm.blues,
        symmetric=False,
        vmin=0.15,
        vmax=0.90,
    )
    ax.set_title("Uniform-prior ice thickness sample")

    fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.06,
        shrink=0.75,
        label="Ice thickness (m)",
    )
    fig.savefig(
        FIGURES_DIR / "04_uniform_field.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_variability_side_by_side(weighted_pattern, fp):
    ice_std_field = weighted_pattern.spatial_weights(fp)
    firn_std_field = weighted_pattern.firn_weights(fp)

    fig = plt.figure(figsize=(14.0, 5.6))
    ax_left = fig.add_subplot(
        1, 2, 1, projection=ccrs.Robinson()
    )
    ax_right = fig.add_subplot(
        1, 2, 2, projection=ccrs.Robinson()
    )

    im_left = plot_shgrid_robinson_on_ax(
        firn_std_field,
        ax_left,
        cmap=cc.cm.blues,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )
    im_right = plot_shgrid_robinson_on_ax(
        ice_std_field,
        ax_right,
        cmap=cc.cm.blues,
        symmetric=False,
        vmin=0.1,
        vmax=0.9,
    )

    ax_left.set_title(
        "Firn melt pointwise standard deviation field"
    )
    ax_right.set_title(
        "Ice melt pointwise standard deviation field"
    )

    fig.colorbar(
        im_left,
        ax=ax_left,
        orientation="horizontal",
        pad=0.06,
        shrink=0.9,
        label="Firn variability multiplier",
    )
    fig.colorbar(
        im_right,
        ax=ax_right,
        orientation="horizontal",
        pad=0.06,
        shrink=0.9,
        label="Ice variability multiplier",
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variability_fields_side_by_side.pdf",
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
    )
    fig.colorbar(
        load_ims[-1],
        cax=cax_ld,
        orientation="horizontal",
        label="Load change (kg/m²)",
    )

    fig.savefig(
        FIGURES_DIR
        / "04_variable_fields_thickness_load_grid.pdf",
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
    save_variable_thickness_load_grid(samples)


if __name__ == "__main__":
    main()
