"""Combined SSH figure: expectations and samples in a 2-2-1 grid layout.

Produces two figures (6.5" × 7"), each with 5 panels replicating the layout
from the report's SSH section:
  - 10_ssh_expectations.pdf  — expectations of the IceSheetChange measures
  - 10_ssh_samples.pdf       — a single sample from ice_change.sample()
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pyshtools import SHGrid
from pyslfp import FingerPrint, IceModel

from project import (
    colors,  # noqa: F401 — sets default fonts/style
)
from pyslfp_extras.ice_thickness import IceSheetChange

np.random.seed(120101)

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.show = lambda *args, **kwargs: None


# ---------------------------------------------------------------------------
# Helper: plot an SHGrid onto an existing Robinson-projection axes
# ---------------------------------------------------------------------------


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
            "If symmetric=False, provide both vmin and vmax."
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


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def make_ssh_grid_figure(panels, title: str) -> plt.Figure:
    """Create the 2-2-1 grid figure.

    Parameters
    ----------
    panels : list of (SHGrid, dict) tuples
        Each tuple is (field, plot_kwargs) where plot_kwargs is passed to
        plot_shgrid_robinson_on_ax and must include 'colorbar_label'.
    title : str
        Figure suptitle.
    """
    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.05)

    axes = [
        fig.add_subplot(
            gs[0, 0:2], projection=ccrs.Robinson()
        ),  # top-left
        fig.add_subplot(
            gs[0, 2:4], projection=ccrs.Robinson()
        ),  # top-right
        fig.add_subplot(
            gs[1, 0:2], projection=ccrs.Robinson()
        ),  # mid-left
        fig.add_subplot(
            gs[1, 2:4], projection=ccrs.Robinson()
        ),  # mid-right
        fig.add_subplot(
            gs[2, 1:3], projection=ccrs.Robinson()
        ),  # bottom-center
    ]

    for ax, (field, kwargs) in zip(axes, panels):
        label = kwargs.pop("colorbar_label", None)
        im = plot_shgrid_robinson_on_ax(field, ax, **kwargs)
        fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.7,
            label=label,
        )

    fig.suptitle(title, y=1.01, fontsize=10)
    return fig


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.001,
    gmsl_target_mean=0.01,
)

altimetry_mask = fp.altimetry_projection(
    latitude_min=-66.0, latitude_max=66.0
)

# ---------------------------------------------------------------------------
# Figure 1: Expectations
# ---------------------------------------------------------------------------

exp_panels = [
    (
        ice_change.ice_load.expectation,
        {
            "colorbar_label": "Load (kg/m²)",
            "symmetric": True,
        },
    ),
    (
        ice_change.ice_slc.expectation
        * fp.ocean_projection()
        * 1000,
        {
            "colorbar_label": "Sea Level Change (mm)",
            "symmetric": True,
        },
    ),
    (
        ice_change.ice_ssh.expectation
        * fp.ocean_projection()
        * 1000,
        {
            "colorbar_label": "Sea Surface Height\nChange (mm)",
            "symmetric": True,
        },
    ),
    (
        ice_change.ice_ssh.expectation
        * altimetry_mask
        * 1000,
        {
            "colorbar_label": "Observed Sea Surface Height\nChange (mm)",
            "symmetric": True,
        },
    ),
    (
        (
            ice_change.ice_slc - ice_change.ice_ssh
        ).expectation
        * fp.ocean_projection()
        * 1000,
        {
            "colorbar_label": "Error Expectation: SLC − SSH (mm)",
            "symmetric": True,
        },
    ),
]

fig_exp = make_ssh_grid_figure(
    exp_panels, "Sea Surface Height — Expectations"
)
fig_exp.savefig(
    FIGURES_DIR / "10_ssh_expectations.pdf",
    dpi=600,
    bbox_inches="tight",
)
plt.close(fig_exp)

# ---------------------------------------------------------------------------
# Figure 2: Sample
# ---------------------------------------------------------------------------

sample = ice_change.sample()
# Note: sample.ice_slc is already ocean-projected (ocean_projection_operator
# is baked in during IceSheetChange.sample()); sample.ice_ssh is not.

s_load = sample.ice_load * fp.ice_projection()
s_slc = sample.ice_slc * fp.ocean_projection() * 1000
s_ssh = sample.ice_ssh * fp.ocean_projection() * 1000
s_obs = sample.ice_ssh * altimetry_mask * 1000
s_err = (
    (sample.ice_slc - sample.ice_ssh)
    * fp.ocean_projection()
) * 1000


def _bounds(field):
    d = np.asarray(field.data)
    return {
        "symmetric": False,
        "vmin": float(np.nanmin(d)),
        "vmax": float(np.nanmax(d)),
    }


samp_panels = [
    (
        s_load,
        {
            "colorbar_label": "Load (kg/m²)",
            **_bounds(s_load),
        },
    ),
    (
        s_slc,
        {
            "colorbar_label": "Sea Level Change (mm)",
            **_bounds(s_slc),
        },
    ),
    (
        s_ssh,
        {
            "colorbar_label": "Sea Surface Height\nChange (mm)",
            **_bounds(s_slc),
        },
    ),
    (
        s_obs,
        {
            "colorbar_label": "Observed Sea Surface Height\nChange (mm)",
            **_bounds(s_slc),
        },
    ),
    (
        s_err,
        {
            "colorbar_label": "Bias: SLC − SSH (mm)",
            **_bounds(s_err),
        },
    ),
]

fig_samp = make_ssh_grid_figure(
    samp_panels, "Sea Surface Height — Sample"
)
fig_samp.savefig(
    FIGURES_DIR / "10_ssh_samples.pdf",
    dpi=600,
    bbox_inches="tight",
)
plt.close(fig_samp)

print("Saved: 10_ssh_expectations.pdf, 10_ssh_samples.pdf")
