# Auto-generated from notebook code cells.
# Source: notebooks/07 - Inversion Sensitivity.ipynb

# ---- Notebook code cell 1 ----
import cartopy.crs as ccrs
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(349549)
import seaborn as sns
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from pygeoinf import (
    CholeskySolver,
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
)
from pygeoinf.symmetric_space.sphere import SphereHelper
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
)

from project import colors
from pygeoinf_extras import standard_dev
from pygeoinf_extras.operators import (
    point_averaging_area_weighted_operator,
)
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyshtools import SHGrid

from pyslfp_extras import plot

fig_format = "pdf"
SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script directory: {SCRIPT_DIR}")
print(f"Figures will be saved to: {FIGURES_DIR}")

# ---- Notebook code cell 2 ----
lmax = 128
altimetry_degree_density = 5.0

# --- Initialise fingerprint model ---
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# --- Truth prior: moderate length scale, small GMSL std ---
truth_length_scale = 0.1 * fp.mean_sea_floor_radius
truth_gmsl_std = 0.01

truth_ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=truth_length_scale,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=truth_gmsl_std,
    point_degree_spacing=altimetry_degree_density,
)

# --- Operators used everywhere ---
model_space = truth_ice_change.ice_thickness.domain
data_space = (
    truth_ice_change.load_to_ssh_point_estimations_operator
    @ truth_ice_change.ice_thickness_to_load_operator
).codomain

# Sea-level conversion (for spatial plots)
ice_thickness_to_slc_op = (
    truth_ice_change.load_to_slc_operator
    @ truth_ice_change.ice_thickness_to_load_operator
)

# GMSL property operator (mm)
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
B = averaging_operator(
    model_space, [GMSL_weighting_function]
)

# Area-weighted operator for altimetric GMSL comparison
altimetry_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=altimetry_degree_density,
)
F = point_averaging_area_weighted_operator(
    data_space, np.asarray(altimetry_points.lats)
)

print(
    f"Forward operator: {model_space.dim} -> {data_space.dim}"
)

# ---- Notebook code cell 3 ----
# --- Data error ---
altimetry_std_dev = 0.003
data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, altimetry_std_dev
    )
)

# --- Build the forward operator from the truth instance ---
truth_forward_op = (
    truth_ice_change.load_to_ssh_point_estimations_operator
    @ truth_ice_change.ice_thickness_to_load_operator
)

# --- Forward problem ---
forward_problem = LinearForwardProblem(
    truth_forward_op, data_error_measure=data_error_measure
)

# --- Draw truth ---
truth_prior_measure = truth_ice_change.ice_thickness
model_true, data = forward_problem.synthetic_model_and_data(
    truth_prior_measure
)


# --- True GMSL ---
GMSL_true = B(model_true)[0]
print(f"True GMSL contribution: {GMSL_true:.4f} mm")

# --- Area-weighted altimetry point estimate ---
ssh_point_values = truth_forward_op(model_true)
ssh_estimation_alt = F(ssh_point_values)[0] * 1000
print(
    f"Area-weighted altimetry estimation: {ssh_estimation_alt:.4f} mm"
)

# ---- Notebook code cell 4 ----
fig, ax, im = plot(
    model_true * fp.length_scale * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    symmetric=True,
    colorbar_label="Ice Thickness Change (m)",
)
ax.set_title("Synthetic True Ice Thickness Change")
fig.savefig(
    FIGURES_DIR / f"7-0_true_ice_thickness.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)
plt.show()


# ---- Notebook code cell 5 ----
def run_inversion(ice_change: IceSheetChange):
    """Run a Bayesian inversion for the given IceSheetChange prior and
    return (model_posterior_measure, prior_measure)."""
    fwd_op = (
        ice_change.load_to_ssh_point_estimations_operator
        @ ice_change.ice_thickness_to_load_operator
    )
    fwd_problem = LinearForwardProblem(
        fwd_op, data_error_measure=data_error_measure
    )
    prior = ice_change.ice_thickness
    inversion = LinearBayesianInversion(fwd_problem, prior)
    residuals = []

    pbar = tqdm(desc="CG solve")

    def progress_callback(xk):
        residuals.append(np.linalg.norm(xk))
        pbar.set_postfix({"||x||": f"{residuals[-1]:.2e}"})
        pbar.update(1)

    returnable = (
        inversion.model_posterior_measure(
            data,
            CGMatrixSolver(callback=progress_callback),
        ),
        prior,
    )

    pbar.close()

    return returnable


# ---- Notebook code cell 6 ----
from jsonschema.benchmarks.subcomponents import v


def plot_shgrid_robinson_on_ax(
    shgrid: SHGrid,
    ax,
    *,
    cmap: str = "seismic",
    symmetric: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot an SHGrid on an existing GeoAxes using Robinson projection."""
    if not isinstance(shgrid, SHGrid):
        raise TypeError(
            "Expected a pyshtools.SHGrid instance."
        )

    data = np.asarray(shgrid.data)
    lons = np.asarray(shgrid.lons())
    lats = np.asarray(shgrid.lats())

    if symmetric and vmin is None and vmax is None:
        max_abs_value = np.nanmax(np.abs(data))
        vmin = -max_abs_value
        vmax = max_abs_value
    else:
        if vmin is None or vmax is None:
            raise ValueError(
                "If symmetric=False, both vmin and vmax must be provided."
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
    )
    ax.coastlines(linewidth=0.6)
    ax.set_global()
    return im


def scalar_z_score(
    estimate: float, truth: float, std_dev: float
) -> float:
    """Return the z-score for a scalar estimate."""
    if std_dev <= 0:
        raise ValueError(
            "Standard deviation must be positive when computing a z-score."
        )
    return (estimate - truth) / std_dev


def gaussian_measure_summary(
    measure: GaussianMeasure,
    truth: float,
) -> tuple[float, float, float]:
    """Return (z_score, mean, std_dev) for a scalar Gaussian measure."""
    mean = measure.expectation[0]
    std_dev = np.sqrt(
        measure.covariance.matrix(dense=True)[0, 0]
    )
    z_score = scalar_z_score(mean, truth, std_dev)
    return z_score, mean, std_dev


def plot_gmsl_sensitivity(
    prior_measures,
    posterior_measures,
    param_values,
    param_label: str,
    panel_labels: list[str],
    suptitle: str,
    filename: str,
    inversion_results: list[GaussianMeasure],
    fixed_x_range: tuple[float, float] | None = None,
):
    """Plot GMSL PDFs and posterior maps with one row per inversion setting."""
    prior_gmsl_measures = [
        m.affine_mapping(operator=B) for m in prior_measures
    ]
    GMSL_posts = [
        m.affine_mapping(operator=B)
        for m in posterior_measures
    ]
    # Calculate altimetry point estimation error
    averaged_error = data_error_measure.affine_mapping(
        operator=F
    )
    ssh_std = standard_dev(averaged_error) * 1000
    altimetry_z = scalar_z_score(
        ssh_estimation_alt, GMSL_true, ssh_std
    )

    n_panels = len(param_values)
    fig = plt.figure(figsize=(7, 9))
    gs = fig.add_gridspec(
        n_panels, 2, width_ratios=[1.0, 1.75]
    )
    axes_pdf = [
        fig.add_subplot(gs[idx, 0])
        for idx in range(n_panels)
    ]
    axes_map = [
        fig.add_subplot(
            gs[idx, 1], projection=ccrs.Robinson()
        )
        for idx in range(n_panels)
    ]

    # map absolute max value, apply to all panels for consistent colorbar
    max_abs_value = max(
        np.abs(inversion_result.expectation.data).max()
        for inversion_result in inversion_results
    )

    z_score_rows = []

    for idx in range(n_panels):
        ax_pdf = axes_pdf[idx]
        ax_map = axes_map[idx]

        prior_z, prior_mean, prior_std = (
            gaussian_measure_summary(
                prior_gmsl_measures[idx], GMSL_true
            )
        )
        post_z, post_mean, post_std = (
            gaussian_measure_summary(
                GMSL_posts[idx], GMSL_true
            )
        )
        z_score_rows.append(
            (
                param_values[idx],
                prior_z,
                prior_mean,
                prior_std,
                post_z,
                post_mean,
                post_std,
            )
        )

        if fixed_x_range is not None:
            x = np.linspace(*fixed_x_range, 500)
        else:
            x_lo = min(
                post_mean - 4 * post_std,
                ssh_estimation_alt - 4 * ssh_std,
                GMSL_true - 0.2 * abs(GMSL_true),
            )
            x_hi = max(
                post_mean + 4 * post_std,
                ssh_estimation_alt + 4 * ssh_std,
                GMSL_true + 0.2 * abs(GMSL_true),
            )
            x = np.linspace(x_lo, x_hi, 500)

        ax_pdf.get_yaxis().set_visible(False)

        alt_pdf = stats.norm.pdf(
            x, ssh_estimation_alt, ssh_std
        )
        ax_pdf.fill_between(
            x, alt_pdf, color=colors.old_method, alpha=0.25
        )
        ax_pdf.plot(
            x,
            alt_pdf,
            color=colors.old_method,
            lw=2,
            label="Altimetry",
        )

        post_pdf = stats.norm.pdf(x, post_mean, post_std)
        ax_pdf.fill_between(
            x, post_pdf, color=colors.new_method, alpha=0.25
        )
        ax_pdf.plot(
            x,
            post_pdf,
            color=colors.new_method,
            lw=2,
            label="Posterior",
        )

        ax_pdf.axvline(
            GMSL_true,
            color=colors.true,
            ls="--",
            lw=2,
            label=f"True ({GMSL_true:.2f} mm)",
        )

        row_label = f"{panel_labels[idx]} {param_label}: {param_values[idx]}"
        ax_pdf.set_title(row_label)
        ax_pdf.set_xlabel("GMSL Contribution (mm)")
        if idx == 0:
            ax_pdf.legend(loc="best", fontsize=8)

        im = plot_shgrid_robinson_on_ax(
            inversion_results[idx].expectation,
            ax_map,
            cmap="seismic",
            vmin=-max_abs_value,
            vmax=max_abs_value,
            symmetric=True,
        )
        ax_map.set_title("Posterior Mean Map")
        fig.colorbar(
            im,
            ax=ax_map,
            orientation="horizontal",
            pad=0.05,
            shrink=0.7,
            label="Posterior Mean Ice Thickness Change (mm)",
        )

    for ax_pdf in axes_pdf:
        sns.despine(ax=ax_pdf, left=True)
    fig.suptitle(suptitle, fontsize=13, y=1.005)
    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{filename}.{fig_format}",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()

    print(f"\nZ-scores for {suptitle}")
    print(
        "  "
        f"Altimetry: z = {altimetry_z:.2f} "
        f"(estimate = {ssh_estimation_alt:.2f} mm, std = {ssh_std:.2f} mm)"
    )
    for (
        param_value,
        prior_z,
        prior_mean,
        prior_std,
        post_z,
        post_mean,
        post_std,
    ) in z_score_rows:
        print(
            "  "
            f"{param_label} = {param_value}: "
            f"prior z = {prior_z:.2f} "
            f"(mean = {prior_mean:.2f} mm, std = {prior_std:.2f} mm), "
            f"posterior z = {post_z:.2f} "
            f"(mean = {post_mean:.2f} mm, std = {post_std:.2f} mm)"
        )


# ---- Notebook code cell 7 ----
# Baseline inversion with the same prior settings used to generate truth
posterior_true, prior_true = run_inversion(truth_ice_change)

# Posterior in GMSL space
GMSL_post_true = posterior_true.affine_mapping(operator=B)
post_mean_true = GMSL_post_true.expectation[0]
post_std_true = np.sqrt(
    GMSL_post_true.covariance.matrix(dense=True)[0, 0]
)

# ---- Notebook code cell 8 ----
averaged_error = data_error_measure.affine_mapping(
    operator=F
)
ssh_std = standard_dev(averaged_error) * 1000
altimetry_z_true = scalar_z_score(
    ssh_estimation_alt, GMSL_true, ssh_std
)
prior_gmsl_true = prior_true.affine_mapping(operator=B)
prior_z_true, prior_mean_true, prior_std_true = (
    gaussian_measure_summary(prior_gmsl_true, GMSL_true)
)
(
    posterior_z_true,
    posterior_mean_true,
    posterior_std_true,
) = gaussian_measure_summary(GMSL_post_true, GMSL_true)

x = np.linspace(
    min(
        GMSL_true - 4 * ssh_std,
        ssh_estimation_alt - 4 * ssh_std,
        post_mean_true - 4 * post_std_true,
    ),
    max(
        GMSL_true + 4 * ssh_std,
        ssh_estimation_alt + 4 * ssh_std,
        post_mean_true + 4 * post_std_true,
    ),
    500,
)

post_pdf = stats.norm.pdf(x, post_mean_true, post_std_true)
alt_pdf = stats.norm.pdf(x, ssh_estimation_alt, ssh_std)

fig = plt.figure(figsize=(7, 5), constrained_layout=True)
gs = fig.add_gridspec(
    4,
    2,
    width_ratios=[1, 1.85],
    height_ratios=[1, 1, 1, 1],
    hspace=0.5,
)

# middle two rows on the left for pdf, leaving top and bottom rows empty for spacing
ax_pdf = fig.add_subplot(gs[1:3, 0])

ax_true = fig.add_subplot(
    gs[0:2, 1], projection=ccrs.Robinson()
)
ax_post = fig.add_subplot(
    gs[2:4, 1], projection=ccrs.Robinson()
)

ax_pdf.fill_between(
    x, alt_pdf, color=colors.old_method, alpha=0.25
)
ax_pdf.plot(
    x,
    alt_pdf,
    color=colors.old_method,
    lw=2,
    label="Altimetry",
)
ax_pdf.fill_between(
    x, post_pdf, color=colors.new_method, alpha=0.25
)
ax_pdf.plot(
    x,
    post_pdf,
    color=colors.new_method,
    lw=2,
    label="Posterior",
)
ax_pdf.axvline(
    GMSL_true,
    color=colors.true,
    ls="--",
    lw=2,
    label=f"True ({GMSL_true:.2f} mm)",
)
ax_pdf.set_xlabel("GMSL Contribution (mm)")
ax_pdf.set_ylabel("Density")
ax_pdf.set_title("Accurate-Prior GMSL Posterior")
# legend to the bottom left of the figure
ax_pdf.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.3,
    ),  # (x, y) coordinates relative to the ax_pdf bounding box
    fontsize=8,
    ncol=1,  # Set to 3 if you want the legend items in a single horizontal row
)
sns.despine(ax=ax_pdf)

true_thickness = (
    model_true * fp.length_scale * fp.ice_projection()
)
posterior_thickness = (
    posterior_true.expectation
    * fp.length_scale
    * fp.ice_projection()
)

im_true = plot_shgrid_robinson_on_ax(
    true_thickness,
    ax_true,
    cmap="seismic",
    symmetric=True,
)
ax_true.set_title("True Ice Thickness Change")

im_post = plot_shgrid_robinson_on_ax(
    posterior_thickness,
    ax_post,
    cmap="seismic",
    symmetric=True,
)
ax_post.set_title(
    "Posterior Mean Ice Thickness",
)

vmax = max(
    np.nanmax(np.abs(np.asarray(true_thickness.data))),
    np.nanmax(np.abs(np.asarray(posterior_thickness.data))),
)
im_true.set_clim(-vmax, vmax)
im_post.set_clim(-vmax, vmax)

fig.colorbar(
    im_post,
    ax=[ax_true, ax_post],
    orientation="horizontal",
    pad=0.06,
    shrink=0.9,
    label="Ice Thickness Change (m)",
)

fig.savefig(
    FIGURES_DIR / f"7-0_baseline_inversion.{fig_format}",
    dpi=600,
    bbox_inches="tight",
)
plt.show()

print("\nZ-scores for accurate-prior inversion")
print(
    "  "
    f"Altimetry: z = {altimetry_z_true:.2f} "
    f"(estimate = {ssh_estimation_alt:.2f} mm, std = {ssh_std:.2f} mm)"
)
print(
    "  "
    f"Prior: z = {prior_z_true:.2f} "
    f"(mean = {prior_mean_true:.2f} mm, std = {prior_std_true:.2f} mm)"
)
print(
    "  "
    f"Posterior: z = {posterior_z_true:.2f} "
    f"(mean = {posterior_mean_true:.2f} mm, std = {posterior_std_true:.2f} mm)"
)

# ---- Notebook code cell 9 ----
length_scales = (
    np.array([0.05, 0.15, 0.4]) * fp.mean_sea_floor_radius
)

prior_measures_ls = []
posterior_measures_ls = []

for scale in length_scales:
    print(f"\nLength scale: {scale} km")

    ic = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=scale,
        pattern=IceSheetChange.ThicknessWeightedPattern(),
        ice_gmsl_std=truth_gmsl_std,
        point_degree_spacing=altimetry_degree_density,
    )

    posterior, prior = run_inversion(ic)
    prior_measures_ls.append(prior)
    posterior_measures_ls.append(posterior)
    print("  Inversion complete")

print("\nAll length-scale inversions complete.")

# ---- Notebook code cell 10 ----
plot_gmsl_sensitivity(
    prior_measures_ls,
    posterior_measures_ls,
    param_values=[
        f"{v / (0.1 * fp.mean_sea_floor_radius):.1f}x ({v / 1000:.2f} km)"
        for v in length_scales
    ],
    param_label="Length scale",
    panel_labels=["a)", "b) Prior", "c)"],
    suptitle=f"GMSL Sensitivity to Prior Length Scale, true length scale: {truth_length_scale / 1000:.2f} km",
    filename="7-1_length_scale",
    inversion_results=posterior_measures_ls,
)

# ---- Notebook code cell 11 ----
offsets_mm = np.array([1.0, 10.0, 50.0])

prior_measures_off = []
posterior_measures_off = []

for offset_mm in offsets_mm:
    print(f"\nGMSL target mean: {offset_mm} mm")
    target_nd = offset_mm / (1000 * fp.length_scale)

    ic = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=truth_length_scale,
        pattern=IceSheetChange.ThicknessWeightedPattern(),
        ice_gmsl_std=truth_gmsl_std,
        gmsl_target_mean=target_nd,
        point_degree_spacing=altimetry_degree_density,
    )

    posterior, prior = run_inversion(ic)
    prior_measures_off.append(prior)
    posterior_measures_off.append(posterior)
    print("  Inversion complete")

print("\nAll mean-offset inversions complete.")

# ---- Notebook code cell 12 ----
plot_gmsl_sensitivity(
    prior_measures_off,
    posterior_measures_off,
    param_values=[f"{v} mm" for v in offsets_mm],
    param_label="GMSL offset",
    panel_labels=["a) ", "b)", "c)", "d)"],
    suptitle="GMSL Sensitivity to Prior Mean Offset",
    filename="7-2_mean_translation",
    inversion_results=posterior_measures_off,
)

# ---- Notebook code cell 13 ----
std_multipliers = np.array([0.5, 2.0, 5.0])

prior_measures_cov = []
posterior_measures_cov = []

for mult in std_multipliers:
    print(f"\nStd multiplier: {mult}")

    ic = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=truth_length_scale,
        pattern=IceSheetChange.ThicknessWeightedPattern(),
        ice_gmsl_std=truth_gmsl_std * mult,
        point_degree_spacing=altimetry_degree_density,
    )

    posterior, prior = run_inversion(ic)
    prior_measures_cov.append(prior)
    posterior_measures_cov.append(posterior)
    print("  Inversion complete")

print("\nAll covariance-scaling inversions complete.")

# ---- Notebook code cell 14 ----
plot_gmsl_sensitivity(
    prior_measures_cov,
    posterior_measures_cov,
    param_values=[f"x{v}" for v in std_multipliers],
    param_label="Std multiplier",
    panel_labels=["a)", "b)", "c)", "d)"],
    suptitle="GMSL Sensitivity to Prior Covariance Amplitude",
    filename="7-3_covariance_scaling",
    inversion_results=posterior_measures_cov,
)
