# %%
# =============================================================================
# Knockout Test: Joint Inversion with Ice, Firn, and Ocean Dynamics
#
# Runs four inversions from the same synthetic true model:
#   1. Full     - SSH altimetry + tide gauges + ice altimetry
#   2. No SSH   - tide gauges + ice altimetry only
#   3. No TG    - SSH altimetry + ice altimetry only
#   4. No ice   - SSH altimetry + tide gauges only
#
# Produces:
#   - GMSL posterior comparison plot (all 4 variants)
#   - Component grid: true vs posterior for ice, firn, and ODT (3 rows × 5 cols)
# =============================================================================
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    BlockLinearOperator,
    CGMatrixSolver,
    EigenSolver,
    GaussianMeasure,
    HilbertSpaceDirectSum,
    LinearBayesianInversion,
    LinearForwardProblem,
    RowLinearOperator,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
)
from tqdm import tqdm

from project import colors
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.ocean_dynamics import OceanDynamics

# %%
# =============================================================================
# Full-resolution model setup  (lmax=128)
# =============================================================================

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

measure_error_std = 0.0005

ice = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    include_firn=True,
)

OD_pattern = OceanDynamics.DataPattern()
odt = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.004,
    length_scale=10000.0,
    pattern=OD_pattern,
)

# %%
# =============================================================================
# Model space and prior
# =============================================================================

model_space = HilbertSpaceDirectSum(
    [
        ice.ice_thickness.domain,
        ice.firn_thickness.domain,
        odt.height_measure.domain,
    ]
)
model_prior = GaussianMeasure.from_direct_sum(
    [
        ice.ice_thickness,
        ice.firn_thickness,
        odt.height_measure,
    ]
)

# %%
# =============================================================================
# Observation points
# =============================================================================

ssh_altimetry = GridPoints.ocean_altimetry(fp, 10.0, 66.0)
ice_altimetry = GridPoints.ice(fp, 10.0)

lats, lons = read_gloss_tide_gauge_data()

filtered_lats = lats.copy()
filtered_lons = lons.copy()
for i in range(len(lats)):
    for j in range(i + 1, len(lats)):
        if (
            abs(lats[i] - lats[j]) < 8.0
            and abs(lons[i] - lons[j]) < 8.0
        ):
            filtered_lats[j] = None
            filtered_lons[j] = None
filtered_lats = [
    lat for lat in filtered_lats if lat is not None
]
filtered_lons = [
    lon for lon in filtered_lons if lon is not None
]

tide_gauge_points = list(zip(filtered_lats, filtered_lons))
tide_sampling_op = tide_gauge_operator(
    ice.load_to_slc_operator.codomain, tide_gauge_points
)

# %%
# =============================================================================
# Full-resolution forward operator blocks
# =============================================================================

# Row 1: Ocean SSH altimetry
f11 = (
    ssh_altimetry.point_evaluation_operator(
        ice.load_to_ssh_operator.codomain
    )
    @ ice.load_to_ssh_operator
    @ ice.ice_thickness_to_load_operator
)
f12 = (
    ssh_altimetry.point_evaluation_operator(
        ice.load_to_ssh_operator.codomain
    )
    @ ice.load_to_ssh_operator
    @ ice.firn_thickness_to_load_operator
)
f13 = (
    ssh_altimetry.point_evaluation_operator(
        odt._height_to_ssh_op.codomain
    )
    @ odt._height_to_ssh_op
)

# Row 2: Tide gauges (SLC)
f21 = (
    tide_sampling_op
    @ ice.load_to_slc_operator
    @ ice.ice_thickness_to_load_operator
)
f22 = (
    tide_sampling_op
    @ ice.load_to_slc_operator
    @ ice.firn_thickness_to_load_operator
)
f23 = tide_sampling_op @ odt._height_to_slc_op

# Row 3: Ice height altimetry
f31 = ice_altimetry.point_evaluation_operator(
    ice.ice_thickness.domain
)
f32 = ice_altimetry.point_evaluation_operator(
    ice.firn_thickness.domain
)
f33 = ice_altimetry.point_evaluation_operator(
    odt.height_measure.domain
).domain.zero_operator(
    codomain=ice_altimetry.point_evaluation_operator(
        odt.height_measure.domain
    ).codomain
)

# %%
# =============================================================================
# Build the four forward operators (one per variant)
# =============================================================================

forward_op_full = BlockLinearOperator(
    [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]]
)
forward_op_no_ssh = BlockLinearOperator(
    [[f21, f22, f23], [f31, f32, f33]]
)
forward_op_no_tg = BlockLinearOperator(
    [[f11, f12, f13], [f31, f32, f33]]
)
forward_op_no_ice = BlockLinearOperator(
    [[f11, f12, f13], [f21, f22, f23]]
)

# %%
# =============================================================================
# Model-to-SLC operator (used for SLC maps)
# =============================================================================

model_space_to_slc_operator = RowLinearOperator(
    [
        ice.load_to_slc_operator
        @ ice.ice_thickness_to_load_operator,
        ice.load_to_slc_operator
        @ ice.firn_thickness_to_load_operator,
        odt._height_to_slc_op,
    ]
)

# %%
# =============================================================================
# Generate model_true once from the prior; then generate variant data
#
# Use synthetic_model_and_data (not model_prior.sample()) so that model_true
# is a properly structured direct-sum element and model_true[i] returns the
# i-th component as an SHGrid.
# =============================================================================

data_error_full = GaussianMeasure.from_standard_deviation(
    forward_op_full.codomain, measure_error_std
)
_full_problem_for_sampling = LinearForwardProblem(
    forward_op_full, data_error_measure=data_error_full
)
print("Sampling true model from prior...")
model_true, data_full = (
    _full_problem_for_sampling.synthetic_model_and_data(
        model_prior
    )
)
print("True model sampled.")


def make_data(forward_op):
    """Apply forward_op to model_true and add independent noise."""
    data_error = GaussianMeasure.from_standard_deviation(
        forward_op.codomain, measure_error_std
    )
    return forward_op(
        model_true
    ) + data_error.sample(), data_error


data_no_ssh, data_error_no_ssh = make_data(
    forward_op_no_ssh
)
data_no_tg, data_error_no_tg = make_data(forward_op_no_tg)
data_no_ice, data_error_no_ice = make_data(
    forward_op_no_ice
)

# %%
# =============================================================================
# Low-resolution preconditioner setup (lmax=32)
# =============================================================================

lmax_precon = 32

precon_fp = FingerPrint(lmax=lmax_precon)
precon_fp.set_state_from_ice_ng(
    version=IceModel.ICE7G, date=0.0
)
precon_fp_op = precon_fp.as_sobolev_linear_operator(
    2, precon_fp.mean_sea_floor_radius * 0.1
)

precon_ice = IceSheetChange.global_ice(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    length_scale=0.1 * precon_fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    include_firn=True,
)

precon_odt = OceanDynamics(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    std=0.004,
    length_scale=10000.0,
    pattern=OD_pattern,
)

precon_model_prior = GaussianMeasure.from_direct_sum(
    [
        precon_ice.ice_thickness,
        precon_ice.firn_thickness,
        precon_odt.height_measure,
    ]
)

precon_tide_sampling_op = tide_gauge_operator(
    precon_ice.load_to_slc_operator.codomain,
    tide_gauge_points,
)

# Preconditioner blocks: low-res operators sampled at full-res observation points
pf11 = (
    precon_ice.load_to_ssh_operator.codomain.point_evaluation_operator(
        ssh_altimetry.coords
    )
    @ precon_ice.load_to_ssh_operator
    @ precon_ice.ice_thickness_to_load_operator
)
pf12 = (
    precon_ice.load_to_ssh_operator.codomain.point_evaluation_operator(
        ssh_altimetry.coords
    )
    @ precon_ice.load_to_ssh_operator
    @ precon_ice.firn_thickness_to_load_operator
)
pf13 = (
    precon_odt._height_to_ssh_op.codomain.point_evaluation_operator(
        ssh_altimetry.coords
    )
    @ precon_odt._height_to_ssh_op
)
pf21 = (
    precon_tide_sampling_op
    @ precon_ice.load_to_slc_operator
    @ precon_ice.ice_thickness_to_load_operator
)
pf22 = (
    precon_tide_sampling_op
    @ precon_ice.load_to_slc_operator
    @ precon_ice.firn_thickness_to_load_operator
)
pf23 = (
    precon_tide_sampling_op @ precon_odt._height_to_slc_op
)
pf31 = precon_ice.ice_thickness.domain.point_evaluation_operator(
    ice_altimetry.coords
)
pf32 = precon_ice.firn_thickness.domain.point_evaluation_operator(
    ice_altimetry.coords
)
pf33 = precon_odt.height_measure.domain.point_evaluation_operator(
    ice_altimetry.coords
).domain.zero_operator(
    codomain=precon_odt.height_measure.domain.point_evaluation_operator(
        ice_altimetry.coords
    ).codomain
)

# Four preconditioner forward operators (matching the four full-res variants)
precon_op_full = BlockLinearOperator(
    [
        [pf11, pf12, pf13],
        [pf21, pf22, pf23],
        [pf31, pf32, pf33],
    ]
)
precon_op_no_ssh = BlockLinearOperator(
    [[pf21, pf22, pf23], [pf31, pf32, pf33]]
)
precon_op_no_tg = BlockLinearOperator(
    [[pf11, pf12, pf13], [pf31, pf32, pf33]]
)
precon_op_no_ice = BlockLinearOperator(
    [[pf11, pf12, pf13], [pf21, pf22, pf23]]
)

# %%
# =============================================================================
# Build preconditioners via eigen-decomposition
# =============================================================================


def build_preconditioner(
    precon_forward_op, full_res_data_space, label
):
    """Build an approximate inverse normal operator at low resolution."""
    precon_data_error = (
        GaussianMeasure.from_standard_deviation(
            full_res_data_space, measure_error_std
        )
    )
    precon_problem = LinearForwardProblem(
        precon_forward_op,
        data_error_measure=precon_data_error,
    )
    precon_inversion = LinearBayesianInversion(
        precon_problem, precon_model_prior
    )
    print(f"Building preconditioner: {label}...")
    solver = EigenSolver(parallel=False)
    precon_inv = solver(precon_inversion.normal_operator)
    print(f"  Done: {label}")
    return precon_inv


precon_inv_full = build_preconditioner(
    precon_op_full, forward_op_full.codomain, "full"
)
precon_inv_no_ssh = build_preconditioner(
    precon_op_no_ssh,
    forward_op_no_ssh.codomain,
    "no SSH altimetry",
)
precon_inv_no_tg = build_preconditioner(
    precon_op_no_tg,
    forward_op_no_tg.codomain,
    "no tide gauges",
)
precon_inv_no_ice = build_preconditioner(
    precon_op_no_ice,
    forward_op_no_ice.codomain,
    "no ice altimetry",
)

# %%
# =============================================================================
# Run all four inversions
# =============================================================================


def run_inversion(
    forward_op, data_error, data, precon_inv, label
):
    """Run a preconditioned CG Bayesian inversion and return the posterior."""
    problem = LinearForwardProblem(
        forward_op, data_error_measure=data_error
    )
    inversion = LinearBayesianInversion(
        problem, model_prior
    )
    residuals = []
    pbar = tqdm(desc=f"CG ({label})")
    is_solving_mean = [True]

    def callback(xk):
        if is_solving_mean[0]:
            residuals.append(np.linalg.norm(xk))
            pbar.set_postfix(
                {"||x||": f"{residuals[-1]:.2e}"}
            )
            pbar.update(1)

    posterior = inversion.model_posterior_measure(
        data,
        CGMatrixSolver(
            callback=callback, maxiter=300, rtol=1e-5
        ),
        preconditioner=precon_inv,
    )
    pbar.close()
    is_solving_mean[0] = False
    print(f"  Inversion complete: {label}")
    return posterior, residuals


print("\nRunning inversions...")
posterior_full, res_full = run_inversion(
    forward_op_full,
    data_error_full,
    data_full,
    precon_inv_full,
    "full",
)
posterior_no_ssh, res_no_ssh = run_inversion(
    forward_op_no_ssh,
    data_error_no_ssh,
    data_no_ssh,
    precon_inv_no_ssh,
    "no SSH",
)
posterior_no_tg, res_no_tg = run_inversion(
    forward_op_no_tg,
    data_error_no_tg,
    data_no_tg,
    precon_inv_no_tg,
    "no TG",
)
posterior_no_ice, res_no_ice = run_inversion(
    forward_op_no_ice,
    data_error_no_ice,
    data_no_ice,
    precon_inv_no_ice,
    "no ice",
)
print("All inversions complete.")

# %%
# =============================================================================
# CG convergence plot
# =============================================================================

variant_residuals = [
    ("Full (SSH+TG+ice)", res_full, colors.new_method),
    ("No SSH altimetry", res_no_ssh, colors.ice_altimetry),
    ("No tide gauges", res_no_tg, colors.firn),
    (
        "No ice altimetry",
        res_no_ice,
        colors.ocean_altimetry,
    ),
]

fig_cg, ax_cg = plt.subplots(figsize=(7, 4))
for label, residuals, color in variant_residuals:
    ax_cg.semilogy(
        residuals, label=label, color=color, linewidth=1.5
    )
ax_cg.set_xlabel("Iteration")
ax_cg.set_ylabel(r"$\|x_k\|$")
ax_cg.set_title("CG Convergence by Inversion Variant")
ax_cg.legend(fontsize=8)
ax_cg.grid(True, which="both", ls="-", alpha=0.4)
fig_cg.tight_layout()
fig_cg.savefig("figs/knockout_cg_convergence.pdf", dpi=600)

# %%
# =============================================================================
# GMSL operators
# =============================================================================

ice_gmsl_op = ice.ice_thickness_to_gmsl_operator
firn_gmsl_op = ice.firn_thickness_to_gmsl_operator
# ODT doesn't contribute to GMSL (zero operator for the 3rd component)
odt_zero_gmsl_op = odt.height_measure.domain.zero_operator(
    codomain=ice_gmsl_op.codomain
)
total_gmsl_op = RowLinearOperator(
    [ice_gmsl_op, firn_gmsl_op, odt_zero_gmsl_op]
)

# True total GMSL (mm)
total_gmsl_true_mm = total_gmsl_op(model_true)[0] * 1000


def compute_gmsl_posterior(posterior):
    """Return (posterior mean in mm, posterior std in mm)."""
    post_measure = posterior.affine_mapping(
        operator=total_gmsl_op
    )
    exp_mm = post_measure.expectation[0] * 1000
    # covariance.matrix returns a 1×1 matrix; guard against tiny
    # negative values from floating-point rounding
    var = float(
        post_measure.covariance.matrix(dense=True)[0, 0]
    )
    std_mm = np.sqrt(max(var, 0.0)) * 1000
    return exp_mm, std_mm


gmsl_full = compute_gmsl_posterior(posterior_full)
gmsl_no_ssh = compute_gmsl_posterior(posterior_no_ssh)
gmsl_no_tg = compute_gmsl_posterior(posterior_no_tg)
gmsl_no_ice = compute_gmsl_posterior(posterior_no_ice)

# %%
# =============================================================================
# GMSL comparison: posterior distributions for all four variants
# =============================================================================


def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * np.sqrt(2 * np.pi)
    )


variant_gmsl = [
    ("Full (SSH+TG+ice)", gmsl_full, colors.new_method),
    ("No SSH altimetry", gmsl_no_ssh, colors.ice_altimetry),
    ("No tide gauges", gmsl_no_tg, colors.ocean_dynamics),
    (
        "No ice altimetry",
        gmsl_no_ice,
        colors.ocean_altimetry,
    ),
]

# x-axis: centre on the true value, span 4σ of the widest *finite* std
finite_stds = [
    s for _, (_, s), _ in variant_gmsl if s > 1e-6
]
x_half = 4 * max(finite_stds) if finite_stds else 5.0
x_range = np.linspace(
    total_gmsl_true_mm - x_half,
    total_gmsl_true_mm + x_half,
    1000,
)

fig_gmsl, ax_gmsl = plt.subplots(figsize=(8, 4))
ax_gmsl.axvline(
    total_gmsl_true_mm,
    color=colors.true,
    linestyle="--",
    linewidth=2,
    label=f"True GMSL ({total_gmsl_true_mm:.2f} mm)",
)
for label, (exp_mm, std_mm), color in variant_gmsl:
    if std_mm < 1e-6:
        # Posterior is essentially a delta function — draw a vertical line
        ax_gmsl.axvline(
            exp_mm,
            color=color,
            linestyle="-",
            linewidth=2,
            label=f"{label}\n(mean={exp_mm:.2f} mm, std≈0)",
        )
    else:
        pdf = gaussian(x_range, exp_mm, std_mm)
        ax_gmsl.plot(
            x_range,
            pdf,
            label=f"{label}\n(mean={exp_mm:.2f}, std={std_mm:.2e} mm)",
            color=color,
            linewidth=1.8,
        )
        ax_gmsl.axvline(
            exp_mm,
            color=color,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

ax_gmsl.get_yaxis().set_visible(False)
ax_gmsl.set_xlabel("GMSL Contribution (mm)")
ax_gmsl.set_title(
    "Knockout Test: GMSL Posterior Distributions"
)
ax_gmsl.legend(fontsize=8, loc="upper left")
fig_gmsl.tight_layout()
fig_gmsl.savefig("figs/knockout_gmsl.pdf", dpi=600)

# Print summary table
print("\nGMSL Summary")
print(f"  True GMSL:           {total_gmsl_true_mm:.4f} mm")
for label, (exp_mm, std_mm), _ in variant_gmsl:
    sigma = abs(exp_mm - total_gmsl_true_mm) / std_mm
    print(
        f"  {label:<25}: mean={exp_mm:.4f} mm, std={std_mm:.2e} mm, "
        f"{sigma:.2f} sigma from truth"
    )

# %%
# =============================================================================
# Component grid: true vs posterior expectations
# Rows: ice thickness, firn thickness, ODT height
# Columns: true | full | no SSH | no TG | no ice
# =============================================================================

posteriors_ordered = [
    ("Full\n(SSH+TG+ice)", posterior_full),
    ("No SSH\naltimetry", posterior_no_ssh),
    ("No tide\ngauges", posterior_no_tg),
    ("No ice\naltimetry", posterior_no_ice),
]


# Helper: extract SHGrid as a 2D array in mm, masked by the projection.
# Non-masked pixels will be near-zero (not NaN), appearing as the neutral
# colour on the seismic colormap — matching the existing plotting pattern.
def field_mm(shgrid, projection_mask):
    scale = fp.length_scale * 1000  # m → mm
    return (shgrid * projection_mask * scale).data.astype(
        float
    )


# Helper: compute symmetric colour limit across a list of 2D arrays
def sym_clim(*arrays):
    vals = np.concatenate(
        [a[np.isfinite(a)].ravel() for a in arrays]
    )
    return np.nanmax(np.abs(vals))


# Pre-compute expectations and build arrays for all cells
ice_true = model_true[0]
firn_true = model_true[1]
odt_true = model_true[2]

component_rows = []
for comp_key, proj, unit_label in [
    (
        "ice",
        fp.ice_projection(),
        "Ice Thickness Change (mm)",
    ),
    (
        "firn",
        fp.ice_projection(),
        "Firn Thickness Change (mm)",
    ),
    (
        "odt",
        fp.ocean_projection(),
        "Ocean Dyn. Height (mm)",
    ),
]:
    true_field = {
        "ice": ice_true,
        "firn": firn_true,
        "odt": odt_true,
    }[comp_key]
    idx = {"ice": 0, "firn": 1, "odt": 2}[comp_key]

    row_arrays = [field_mm(true_field, proj)]
    for _, post in posteriors_ordered:
        row_arrays.append(
            field_mm(post.expectation[idx], proj)
        )

    clim = sym_clim(*row_arrays)
    component_rows.append(
        (comp_key, unit_label, row_arrays, clim)
    )

# Get lon/lat for pcolormesh from the SHGrid
# SHGrid.lats() returns latitudes (°), lons() returns 0..360
_sample_grid = ice_true
_raw_lats = _sample_grid.lats()
_raw_lons = _sample_grid.lons()
# Shift longitudes to -180..180 for cartopy
_lons_shifted = np.where(
    _raw_lons > 180, _raw_lons - 360, _raw_lons
)
_sort_idx = np.argsort(_lons_shifted)
_lons_plot = _lons_shifted[_sort_idx]
_lon_grid, _lat_grid = np.meshgrid(_lons_plot, _raw_lats)

n_rows = len(component_rows)
n_cols = 1 + len(posteriors_ordered)  # true + 4 posteriors

fig_grid, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(9, 6.5),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=True,
)

col_titles = ["True"] + [
    lbl for lbl, _ in posteriors_ordered
]

for row_idx, (
    comp_key,
    unit_label,
    row_arrays,
    clim,
) in enumerate(component_rows):
    for col_idx, (arr, col_title) in enumerate(
        zip(row_arrays, col_titles)
    ):
        ax = axes[row_idx, col_idx]
        data_sorted = arr[:, _sort_idx]
        im = ax.pcolormesh(
            _lon_grid,
            _lat_grid,
            data_sorted,
            transform=ccrs.PlateCarree(),
            cmap="seismic",
            vmin=-clim,
            vmax=clim,
        )
        ax.coastlines(linewidth=0.4, color="k")

        if row_idx == 0:
            ax.set_title(col_title, fontsize=8, pad=4)
        if col_idx == 0:
            ax.text(
                -0.04,
                0.5,
                unit_label,
                va="center",
                ha="right",
                rotation=90,
                fontsize=7.5,
                transform=ax.transAxes,
            )

    # One colorbar per row, spanning all columns
    cb = fig_grid.colorbar(
        im,
        ax=axes[row_idx, :],
        orientation="horizontal",
        shrink=0.6,
        label=unit_label,
    )
    cb.ax.tick_params(labelsize=7)

fig_grid.suptitle(
    "Knockout Test: True vs Posterior Expectations\n"
    "(rows: ice / firn / ocean dynamics; "
    "columns: true and each inversion variant)",
    fontsize=9,
)
fig_grid.savefig(
    "figs/knockout_component_grid.pdf",
    dpi=300,
    bbox_inches="tight",
)
fig_grid.savefig(
    "figs/knockout_component_grid.png",
    dpi=200,
    bbox_inches="tight",
)

print("\nAll figures saved to figs/")
