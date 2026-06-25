# Auto-generated from notebook code cells.
# Source: notebooks/06 - Simple Inversion.ipynb
# %%
# ---- Notebook code cell 1 ----
from pyslfp.linear_operators import (
    FingerPrintOperator,
    l2_products_operator,
)
from pyslfp.state import EarthState
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(120106)
from pathlib import Path

from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
)
from tqdm import tqdm

from project import colors
from pygeoinf_extras import standard_dev
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.plotting import plot

fig_format = "pdf"
# SCRIPT_DIR = Path(__file__).resolve().parent
# FIGURES_DIR = SCRIPT_DIR / "figures"
# FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.show = lambda *args, **kwargs: None
print = lambda *args, **kwargs: None

def _save_all_figures(prefix):
    for index, figure_number in enumerate(
        plt.get_fignums(), start=1
    ):
        fig = plt.figure(figure_number)
        fig.savefig(
            FIGURES_DIR / f"{prefix}_{index:02d}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close("all")

# ---- Notebook code cell 2 ----
lmax = 128
altimetry_degree_density = 5.0

fp = EarthState.from_defaults(lmax=lmax)
fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.model.parameters.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.01,
    point_degree_spacing=altimetry_degree_density,
)

ice_thickness_measure = ice_change.ice_thickness
ice_thickness_to_ssh_point_estimations_op = (
    ice_change.load_to_ssh_point_estimations_operator
    @ ice_change.ice_thickness_to_load_operator
)

model_space = ice_thickness_measure.domain
data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)

print(
    f"Forward operator: {model_space.dim} -> {data_space.dim}"
)

fig, ax, im = plot(
    ice_thickness_measure.sample(),
    symmetric=True,
    colorbar_label="Prior sample (non-dimensional)",
)
ax.set_title("Sample from Prior Ice-Thickness Measure")
plt.show()

# ---- Notebook code cell 3 ----
altimetry_std_dev = 0.001
data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, altimetry_std_dev
    )
)

forward_problem = LinearForwardProblem(
    ice_thickness_to_ssh_point_estimations_op,
    data_error_measure=data_error_measure,
)

model_true, data = forward_problem.synthetic_model_and_data(
    ice_thickness_measure
)

print(f"Model dimension: {model_true.data.size}")
print(f"Data dimension: {data.shape[0]}")

# ---- Notebook code cell 4 ----
bayesian_inversion = LinearBayesianInversion(
    forward_problem, ice_thickness_measure
)

residuals = []
pbar = tqdm(desc="CG solve")

def progress_callback(xk):
    residuals.append(np.linalg.norm(xk))
    pbar.set_postfix({"||x||": f"{residuals[-1]:.2e}"})
    pbar.update(1)

model_posterior_measure = (
    bayesian_inversion.model_posterior_measure(
        data, CGMatrixSolver(callback=progress_callback)
    )
)
pbar.close()

model_posterior_expectation = (
    model_posterior_measure.expectation
)
print("Inversion complete.")

# ---- Notebook code cell 5 ----
max_abs_ice_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    model_true.data.flatten(),
                    model_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.model.parameters.length_scale
)

fig1, ax1, im1 = plot(
    1000
    * model_true
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.set_title("a) True Ice Thickness Change")
fig1.savefig(
    FIGURES_DIR / f"6-1_true_ice_thickness.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig2, ax2, im2 = plot(
    1000
    * model_posterior_expectation
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.set_title(
    "b) Posterior Expectation (Inferred from Data)"
)
fig2.savefig(
    FIGURES_DIR
    / f"6-2_posterior_ice_thickness.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

# ---- Notebook code cell 6 ----
ice_thickness_to_slc_op = (
    ice_change.load_to_slc_operator
    @ ice_change.ice_thickness_to_load_operator
)

sea_level_true = ice_thickness_to_slc_op(model_true)
sea_level_posterior = ice_thickness_to_slc_op(
    model_posterior_expectation
)

if isinstance(sea_level_true, list):
    sea_level_true = sea_level_true[0]
if isinstance(sea_level_posterior, list):
    sea_level_posterior = sea_level_posterior[0]

def as_array(grid):
    if hasattr(grid, "to_array"):
        return grid.to_array()
    if hasattr(grid, "data"):
        return grid.data
    return np.asarray(grid)

max_abs_sl_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    as_array(sea_level_true),
                    as_array(sea_level_posterior),
                ]
            )
        )
    )
    * 1000
    * fp.model.parameters.length_scale
)

fig4, ax4, im4 = plot(
    1000
    * sea_level_true
    * fp.model.parameters.length_scale
    * fp.ocean_projection(),
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax4.set_title("a) True Sea-Level Fingerprint")
fig4.savefig(
    FIGURES_DIR / f"6-4_true_sea_level.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig5, ax5, im5 = plot(
    1000
    * sea_level_posterior
    * fp.model.parameters.length_scale
    * fp.ocean_projection(),
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax5.set_title("b) Posterior Sea-Level Fingerprint")
fig5.savefig(
    FIGURES_DIR / f"6-5_posterior_sea_level.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

# ---- Notebook code cell 7 ----
GMSL_weighting_function = (
    -fp.model.parameters.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.model.parameters.length_scale
    / (fp.model.parameters.water_density * fp.ocean_area)
)
B = l2_products_operator(
    model_space, [GMSL_weighting_function]
)

GMSL_true = B(model_true)[0]
GMSL_prior_measure = ice_thickness_measure.affine_mapping(
    operator=B
)
GMSL_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=B)
)

ssh_estimation_alt = (
    ice_change.load_to_point_estimated_gmsl_operator(
        ice_change.ice_thickness_to_load_operator(
            model_true
        )
    )[0]
    * 1000
)

F = point_l2_products_operator(data_space)
averaged_error = data_error_measure.affine_mapping(
    operator=F
)
ssh_std = standard_dev(averaged_error) * 1000

prior_expectation = GMSL_prior_measure.expectation[0]
posterior_expectation = GMSL_posterior_measure.expectation[
    0
]
prior_std_dev = standard_dev(GMSL_prior_measure)
posterior_std_dev = standard_dev(GMSL_posterior_measure)
# %%
x_prior = np.linspace(
    prior_expectation - 6 * prior_std_dev,
    prior_expectation + 6 * prior_std_dev,
    1000,
)
xmin = min(
    GMSL_true - 6 * posterior_std_dev,
    posterior_expectation - 6 * posterior_std_dev,
    ssh_estimation_alt - 6 * ssh_std,
)
xmax = max(
    GMSL_true + 6 * posterior_std_dev,
    posterior_expectation + 6 * posterior_std_dev,
    ssh_estimation_alt + 6 * ssh_std,
)

x_post = np.linspace(xmin, xmax, 1000)
# %%

def gaussian(x, mean, std_dev):
    return (
        1
        / (std_dev * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )

prior_pdf = gaussian(
    x_prior, prior_expectation, prior_std_dev
)
posterior_pdf = gaussian(
    x_post, posterior_expectation, posterior_std_dev
)
ssh_pdf = gaussian(x_post, ssh_estimation_alt, ssh_std)

y_max = max(
    prior_pdf.max(),
    posterior_pdf.max(),
    ssh_pdf.max(),
)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.axvline(
    GMSL_true,
    color=colors.true,
    linestyle="--",
    linewidth=3,
    label=f"True GMSL ({GMSL_true:.2e} mm)",
)

ax.plot(
    x_post,
    ssh_pdf,
    label=f"Altimetry (mean={ssh_estimation_alt:.2e} mm, std={ssh_std:.2e} mm)",
    color=colors.old_method,
    linewidth=3,
)
ax.plot(
    x_post,
    posterior_pdf,
    label=f"Posterior (mean={posterior_expectation:.2e} mm, std={posterior_std_dev:.2e} mm)",
    color=colors.new_method,
    linewidth=3,
)

ax.set_xlim(xmin, xmax)

ax.get_yaxis().set_visible(False)
ax.set_ylim(-0.1, y_max * 1.1)
ax.set_title("GMSL Contribution: Posterior vs Altimetry")
ax.legend()
plt.tight_layout()
fig.savefig(
    FIGURES_DIR
    / f"6-7_gmsl_distribution_comparison.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)
plt.show()

print(
    f"Posterior is {(GMSL_true - posterior_expectation) / posterior_std_dev:.2e} sigma away from true value."
)
print(
    f"Altimetry estimation is {(GMSL_true - ssh_estimation_alt) / ssh_std:.2e} sigma away from true value."
)

# %%
from project.projections import (
    EXTENT_ANTARCTICA,
    EXTENT_GREENLAND,
    PROJ_ANTARCTICA,
    PROJ_GREENLAND,
)

fig_ant_true, ax_ant_true, im_ant_true = plot(
    1000
    * model_true
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
    projection=PROJ_ANTARCTICA,
    map_extent=EXTENT_ANTARCTICA,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_ant_true.set_title(
    "True Ice Thickness Change (Antarctica)"
)
fig_ant_true.savefig(
    FIGURES_DIR
    / f"6-8_true_ice_thickness_antarctica.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_ant_post, ax_ant_post, im_ant_post = plot(
    1000
    * model_posterior_expectation
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
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
    FIGURES_DIR
    / f"6-9_posterior_ice_thickness_antarctica.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_grn_true, ax_grn_true, im_grn_true = plot(
    1000
    * model_true
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
    projection=PROJ_GREENLAND,
    map_extent=EXTENT_GREENLAND,
    figsize=(3.25, 3.25),
    colorbar=False,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
)
ax_grn_true.set_title(
    "True Ice Thickness Change (Greenland)"
)
fig_grn_true.savefig(
    FIGURES_DIR
    / f"6-10_true_ice_thickness_greenland.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

fig_grn_post, ax_grn_post, im_grn_post = plot(
    1000
    * model_posterior_expectation
    * fp.model.parameters.length_scale
    * fp.ice_projection(),
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
    FIGURES_DIR
    / f"6-11_posterior_ice_thickness_greenland.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)

plt.show()

_save_all_figures("06_simple_inversion")
