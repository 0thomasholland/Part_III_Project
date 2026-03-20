import re

with open('work/99_plotting/06_simple_inversion_plots.py', 'r') as f:
    content = f.read()

append = """
from project.projections import PROJ_ANTARCTICA, EXTENT_ANTARCTICA, PROJ_GREENLAND, EXTENT_GREENLAND

fig_ant_true, ax_ant_true, im_ant_true = plot(
    1000 * model_true * fp.length_scale * fp.ice_projection(),
    projection=PROJ_ANTARCTICA,
    map_extent=EXTENT_ANTARCTICA,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_ant_true.set_title("True Ice Thickness Change (Antarctica)")
fig_ant_true.savefig(
    FIGURES_DIR / f"6-8_true_ice_thickness_antarctica.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_ant_post, ax_ant_post, im_ant_post = plot(
    1000 * model_posterior_expectation * fp.length_scale * fp.ice_projection(),
    projection=PROJ_ANTARCTICA,
    map_extent=EXTENT_ANTARCTICA,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_ant_post.set_title("Posterior Expectation (Antarctica)")
fig_ant_post.savefig(
    FIGURES_DIR / f"6-9_posterior_ice_thickness_antarctica.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_grn_true, ax_grn_true, im_grn_true = plot(
    1000 * model_true * fp.length_scale * fp.ice_projection(),
    projection=PROJ_GREENLAND,
    map_extent=EXTENT_GREENLAND,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_grn_true.set_title("True Ice Thickness Change (Greenland)")
fig_grn_true.savefig(
    FIGURES_DIR / f"6-10_true_ice_thickness_greenland.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_grn_post, ax_grn_post, im_grn_post = plot(
    1000 * model_posterior_expectation * fp.length_scale * fp.ice_projection(),
    projection=PROJ_GREENLAND,
    map_extent=EXTENT_GREENLAND,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_grn_post.set_title("Posterior Expectation (Greenland)")
fig_grn_post.savefig(
    FIGURES_DIR / f"6-11_posterior_ice_thickness_greenland.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

_save_all_figures("06_simple_inversion")
"""

content = content.replace('_save_all_figures("06_simple_inversion")', append)

with open('work/99_plotting/06_simple_inversion_plots.py', 'w') as f:
    f.write(content)

