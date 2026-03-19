# %%
# =============================================================================
# Knockout Test: Joint Inversion with Ice, Firn, Ocean Dynamics, and GRACE
#
# Runs four inversions from the same synthetic true model:
#   1. Full      - SSH altimetry + ice altimetry + GRACE
#   2. No SSH    - ice altimetry + GRACE
#   3. No ice    - SSH altimetry + GRACE
#   4. No GRACE  - SSH altimetry + ice altimetry
#
# Uses the factored forward operator (P_left @ F_middle @ L_right) so that
# the expensive fingerprint evaluation is shared across variants.
#
# Produces:
#   - CG convergence plot (all 5 variants)
#   - GMSL posterior comparison plot (all 5 variants)
#   - Component grid: true vs posterior for ice, firn, and ODT (3 rows × 6 cols)
# =============================================================================
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from pygeoinf import (
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
    CGMatrixSolver,
    EigenSolver,
    EuclideanSpace,
    GaussianMeasure,
    HilbertSpaceDirectSum,
    LinearBayesianInversion,
    LinearForwardProblem,
    RowLinearOperator,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)
from pyslfp.operators import grace_operator
from scipy import stats
from tqdm import tqdm

from project import colors
from pygeoinf_extras import standard_dev
from pygeoinf_extras.plots import plot_bivariate_corner
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.ocean_dynamics import OceanDynamics

# %%
# =============================================================================
# Full-resolution model setup  (lmax=128)
# =============================================================================

np.random.seed(42)  # for reproducibility

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

measure_error_std = 0.001
grace_std_dev_m = 0.0027
grace_std = grace_std_dev_m / fp.length_scale
grace_observation_degree = 96

ice = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    firn_gmsl_std=0.002,
    firn_density=0.3 * fp.ice_density,
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
ice_altimetry = GridPoints.ice(fp, 15.0)

# %%
# =============================================================================
# Build shared intermediate operators (F_middle, L_right, and row components)
#
# The factored structure is:  forward_op = P_left @ F_middle @ L_right
#
# F_middle and L_right are shared across all knockout variants.
# Only P_left changes (by selecting which observation rows to include).
# =============================================================================

load_space = fp_op.domain
response_space = fp_op.codomain
ice_space = ice.ice_thickness.domain
firn_space = ice.firn_thickness.domain
odt_space = odt.height_measure.domain

F = fp_op
S = sea_surface_height_operator(fp, response_space)
slc_proj = response_space.subspace_projection(0)
slc_space = slc_proj.codomain

L_I = ice.ice_thickness_to_load_operator
L_F = ice.firn_thickness_to_load_operator
L_W = sea_level_change_to_load_operator(fp, load_space)

# Point evaluation operators (full resolution)
P_S_ssh = ssh_altimetry.point_evaluation_operator(
    S.codomain
)
P_S_odt = ssh_altimetry.point_evaluation_operator(odt_space)
P_I_ice = ice_altimetry.point_evaluation_operator(ice_space)
P_I_firn = ice_altimetry.point_evaluation_operator(
    firn_space
)
grace_op = grace_operator(
    response_space, grace_observation_degree
)

id_odt = odt_space.identity_operator()
id_ice = ice_space.identity_operator()
id_firn = firn_space.identity_operator()

ssh_obs = P_S_ssh.codomain
ice_obs = P_I_ice.codomain
grace_obs = grace_op.codomain

# == L_right (4×3): shared ==
L_right = BlockLinearOperator(
    [
        [L_I, L_F, L_W],
        [
            ice_space.zero_operator(codomain=odt_space),
            firn_space.zero_operator(codomain=odt_space),
            id_odt,
        ],
        [
            id_ice,
            firn_space.zero_operator(codomain=ice_space),
            odt_space.zero_operator(codomain=ice_space),
        ],
        [
            ice_space.zero_operator(codomain=firn_space),
            id_firn,
            odt_space.zero_operator(codomain=firn_space),
        ],
    ]
)

# == F_middle (4×4 block diagonal): shared ==
F_middle = BlockDiagonalLinearOperator(
    [F, id_odt, id_ice, id_firn]
)

# == Individual rows of P_left ==
# Each row is a list of 4 column entries mapping
# [response_space, odt_space, ice_space, firn_space] -> obs_space
row_ssh = [
    P_S_ssh @ S,
    P_S_odt,
    ice_space.zero_operator(codomain=ssh_obs),
    firn_space.zero_operator(codomain=ssh_obs),
]
row_ice = [
    response_space.zero_operator(codomain=ice_obs),
    odt_space.zero_operator(codomain=ice_obs),
    P_I_ice,
    P_I_firn,
]
row_grace = [
    grace_op,
    odt_space.zero_operator(codomain=grace_obs),
    ice_space.zero_operator(codomain=grace_obs),
    firn_space.zero_operator(codomain=grace_obs),
]

# Per-row data error measures
err_ssh = GaussianMeasure.from_standard_deviation(
    ssh_obs, measure_error_std
)
err_ice = GaussianMeasure.from_standard_deviation(
    ice_obs, measure_error_std
)
err_grace = GaussianMeasure.from_standard_deviation(
    grace_obs, grace_std
)


def build_variant(selected_rows, selected_errors):
    """
    Build a forward operator and data error for the given subset of rows.

    Parameters
    ----------
    selected_rows : list of lists
        Each element is a P_left row (list of 4 column entries).
    selected_errors : list of GaussianMeasure
        Per-row data error measures matching selected_rows.

    Returns
    -------
    forward_op : LinearOperator
    data_error : GaussianMeasure
    """
    p_left = BlockLinearOperator(selected_rows)
    forward_op = p_left @ F_middle @ L_right
    data_error = GaussianMeasure.from_direct_sum(
        selected_errors
    )
    return forward_op, data_error


# Build the five variants
forward_op_full, data_error_full = build_variant(
    [row_ssh, row_ice, row_grace],
    [err_ssh, err_ice, err_grace],
)
forward_op_no_ssh, data_error_no_ssh = build_variant(
    [row_ice, row_grace],
    [err_ice, err_grace],
)
forward_op_no_ice, data_error_no_ice = build_variant(
    [row_ssh, row_grace],
    [err_ssh, err_grace],
)
forward_op_no_grace, data_error_no_grace = build_variant(
    [row_ssh, row_ice],
    [err_ssh, err_ice],
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
# Generate model_true from the full problem; derive variant data from it
# =============================================================================

print("Sampling true model from prior...")
full_problem_for_sampling = LinearForwardProblem(
    forward_op_full, data_error_measure=data_error_full
)
model_true, data_full = (
    full_problem_for_sampling.synthetic_model_and_data(
        model_prior
    )
)
print("True model sampled.")


def make_data(forward_op, data_error):
    """Apply forward_op to model_true and add independent noise."""
    return (
        forward_op(model_true) + data_error.sample(),
        data_error,
    )


data_no_ssh, _ = make_data(
    forward_op_no_ssh, data_error_no_ssh
)
data_no_ice, _ = make_data(
    forward_op_no_ice, data_error_no_ice
)
data_no_grace, _ = make_data(
    forward_op_no_grace, data_error_no_grace
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

# Low-res shared operators
precon_load_space = precon_fp_op.domain
precon_response_space = precon_fp_op.codomain
precon_ice_space = precon_ice.ice_thickness.domain
precon_firn_space = precon_ice.firn_thickness.domain
precon_odt_space = precon_odt.height_measure.domain

precon_F = precon_fp_op
precon_S = sea_surface_height_operator(
    precon_fp, precon_response_space
)
precon_slc_proj = precon_response_space.subspace_projection(
    0
)
precon_slc_space = precon_slc_proj.codomain

precon_L_I = precon_ice.ice_thickness_to_load_operator
precon_L_F = precon_ice.firn_thickness_to_load_operator
precon_L_W = sea_level_change_to_load_operator(
    precon_fp, precon_load_space
)

# Evaluate at full-resolution coordinates
precon_P_S_ssh = (
    precon_S.codomain.point_evaluation_operator(
        ssh_altimetry.coords
    )
)
precon_P_S_odt = precon_odt_space.point_evaluation_operator(
    ssh_altimetry.coords
)
precon_P_I_ice = precon_ice_space.point_evaluation_operator(
    ice_altimetry.coords
)
precon_P_I_firn = (
    precon_firn_space.point_evaluation_operator(
        ice_altimetry.coords
    )
)
precon_grace_op = grace_operator(
    precon_response_space, grace_observation_degree
)

precon_id_odt = precon_odt_space.identity_operator()
precon_id_ice = precon_ice_space.identity_operator()
precon_id_firn = precon_firn_space.identity_operator()

precon_L_right = BlockLinearOperator(
    [
        [precon_L_I, precon_L_F, precon_L_W],
        [
            precon_ice_space.zero_operator(
                codomain=precon_odt_space
            ),
            precon_firn_space.zero_operator(
                codomain=precon_odt_space
            ),
            precon_id_odt,
        ],
        [
            precon_id_ice,
            precon_firn_space.zero_operator(
                codomain=precon_ice_space
            ),
            precon_odt_space.zero_operator(
                codomain=precon_ice_space
            ),
        ],
        [
            precon_ice_space.zero_operator(
                codomain=precon_firn_space
            ),
            precon_id_firn,
            precon_odt_space.zero_operator(
                codomain=precon_firn_space
            ),
        ],
    ]
)

precon_F_middle = BlockDiagonalLinearOperator(
    [precon_F, precon_id_odt, precon_id_ice, precon_id_firn]
)

# Low-res P_left rows (sampling at full-res coords)
precon_row_ssh = [
    precon_P_S_ssh @ precon_S,
    precon_P_S_odt,
    precon_ice_space.zero_operator(
        codomain=precon_P_S_ssh.codomain
    ),
    precon_firn_space.zero_operator(
        codomain=precon_P_S_ssh.codomain
    ),
]
precon_row_ice = [
    precon_response_space.zero_operator(
        codomain=precon_P_I_ice.codomain
    ),
    precon_odt_space.zero_operator(
        codomain=precon_P_I_ice.codomain
    ),
    precon_P_I_ice,
    precon_P_I_firn,
]
precon_row_grace = [
    precon_grace_op,
    precon_odt_space.zero_operator(
        codomain=precon_grace_op.codomain
    ),
    precon_ice_space.zero_operator(
        codomain=precon_grace_op.codomain
    ),
    precon_firn_space.zero_operator(
        codomain=precon_grace_op.codomain
    ),
]


def build_preconditioner(
    selected_precon_rows, full_res_data_error, label
):
    """Build an approximate inverse normal operator at low resolution."""
    p_left = BlockLinearOperator(selected_precon_rows)
    precon_forward_op = (
        p_left @ precon_F_middle @ precon_L_right
    )
    precon_problem = LinearForwardProblem(
        precon_forward_op,
        data_error_measure=full_res_data_error,
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
    [precon_row_ssh, precon_row_ice, precon_row_grace],
    data_error_full,
    "full",
)
precon_inv_no_ssh = build_preconditioner(
    [precon_row_ice, precon_row_grace],
    data_error_no_ssh,
    "no SSH altimetry",
)
precon_inv_no_ice = build_preconditioner(
    [precon_row_ssh, precon_row_grace],
    data_error_no_ice,
    "no ice altimetry",
)
precon_inv_no_grace = build_preconditioner(
    [precon_row_ssh, precon_row_ice],
    data_error_no_grace,
    "no GRACE",
)

# %%
# =============================================================================
# Run all five inversions
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
            callback=callback, maxiter=500, rtol=1e-5
        ),
        preconditioner=precon_inv,
    )
    pbar.close()
    is_solving_mean[0] = False
    print(f"  Inversion complete: {label}")
    return posterior, residuals


print("\nRunning inversions...")

# =============================================================================
# GMSL operators
# =============================================================================

ice_gmsl_op = ice.ice_thickness_to_gmsl_operator
firn_gmsl_op = ice.firn_thickness_to_gmsl_operator
odt_zero_gmsl_op = odt.height_measure.domain.zero_operator(
    codomain=ice_gmsl_op.codomain
)
total_gmsl_op = RowLinearOperator(
    [ice_gmsl_op, firn_gmsl_op, odt_zero_gmsl_op]
)

total_gmsl_true_mm = total_gmsl_op(model_true)[0] * 1000


def compute_gmsl_posterior(posterior):
    """Return (posterior mean in mm, posterior std in mm)."""
    post_measure = posterior.affine_mapping(
        operator=total_gmsl_op
    )
    exp_mm = post_measure.expectation[0] * 1000
    var = float(
        post_measure.covariance.matrix(
            dense=True, parallel=False
        )[0, 0]
    )
    std_mm = np.sqrt(max(var, 0.0)) * 1000
    return exp_mm, std_mm


# Row operators mapping the full 3-component model space to a scalar GMSL (mm)
ice_gmsl_row = RowLinearOperator(
    [
        1000 * ice_gmsl_op,
        ice.firn_thickness.domain.zero_operator(
            codomain=ice_gmsl_op.codomain
        ),
        odt.height_measure.domain.zero_operator(
            codomain=ice_gmsl_op.codomain
        ),
    ]
)
firn_gmsl_row = RowLinearOperator(
    [
        ice.ice_thickness.domain.zero_operator(
            codomain=firn_gmsl_op.codomain
        ),
        1000 * firn_gmsl_op,
        odt.height_measure.domain.zero_operator(
            codomain=firn_gmsl_op.codomain
        ),
    ]
)

true_ice_gmsl_mm = ice_gmsl_row(model_true)[0]
true_firn_gmsl_mm = firn_gmsl_row(model_true)[0]
true_values_2d = np.array(
    [true_ice_gmsl_mm, true_firn_gmsl_mm]
)


def gmsl_2d_posterior(posterior):
    """
    Compute the 2D joint Gaussian measure for
    (ice GMSL [mm], firn GMSL [mm]) from a full posterior.

    Uses the polarization identity to extract the cross-covariance
    without computing the full dense covariance of the posterior.
    """
    ice_post = posterior.affine_mapping(
        operator=ice_gmsl_row
    )
    firn_post = posterior.affine_mapping(
        operator=firn_gmsl_row
    )
    sum_post = posterior.affine_mapping(
        operator=ice_gmsl_row + firn_gmsl_row
    )
    var_ice = standard_dev(ice_post, parallel=False) ** 2
    var_firn = standard_dev(firn_post, parallel=False) ** 2
    var_sum = standard_dev(sum_post, parallel=False) ** 2
    cross_cov = 0.5 * (var_sum - var_ice - var_firn)

    mu_ice = ice_post.expectation[0]
    mu_firn = firn_post.expectation[0]
    return GaussianMeasure.from_covariance_matrix(
        EuclideanSpace(2),
        np.array(
            [[var_ice, cross_cov], [cross_cov, var_firn]]
        ),
        expectation=np.array([mu_ice, mu_firn]),
    )


def run_inversion_task(args):
    posterior, residuals = run_inversion(*args)
    gmsl = compute_gmsl_posterior(posterior)
    posterior_2d = gmsl_2d_posterior(posterior)
    return posterior, residuals, gmsl, posterior_2d


tasks = [
    (
        forward_op_full,
        data_error_full,
        data_full,
        precon_inv_full,
        "full",
    ),
    (
        forward_op_no_ssh,
        data_error_no_ssh,
        data_no_ssh,
        precon_inv_no_ssh,
        "no SSH",
    ),
    (
        forward_op_no_ice,
        data_error_no_ice,
        data_no_ice,
        precon_inv_no_ice,
        "no ice",
    ),
    (
        forward_op_no_grace,
        data_error_no_grace,
        data_no_grace,
        precon_inv_no_grace,
        "no GRACE",
    ),
]

results = Parallel(n_jobs=-1, backend="multiprocessing")(
    delayed(run_inversion_task)(args) for args in tasks
)

(
    (posterior_full, res_full, gmsl_full, post2d_full),
    (
        posterior_no_ssh,
        res_no_ssh,
        gmsl_no_ssh,
        post2d_no_ssh,
    ),
    (
        posterior_no_ice,
        res_no_ice,
        gmsl_no_ice,
        post2d_no_ice,
    ),
    (
        posterior_no_grace,
        res_no_grace,
        gmsl_no_grace,
        post2d_no_grace,
    ),
) = results

print("All inversions complete.")

# %%
# =============================================================================
# CG convergence plot
# =============================================================================

variant_residuals = [
    (
        "Full (SSH+ice+GRACE)",
        res_full,
        colors.new_method,
    ),
    ("No SSH altimetry", res_no_ssh, colors.ice_altimetry),
    (
        "No ice altimetry",
        res_no_ice,
        colors.ocean_altimetry,
    ),
    ("No GRACE", res_no_grace, colors.ocean_dynamics),
]

fig_cg, ax_cg = plt.subplots(figsize=(7, 4))
for label, residuals, color in variant_residuals:
    ax_cg.semilogy(
        residuals, label=label, color=color, linewidth=1.5
    )
ax_cg.set_xlabel("Iteration")
ax_cg.set_ylabel(r"$\|x_k\|$")
ax_cg.set_title(
    "CG Convergence by Inversion Variant (with GRACE)"
)
ax_cg.legend(fontsize=8)
ax_cg.grid(True, which="both", ls="-", alpha=0.4)
fig_cg.tight_layout()
fig_cg.savefig(
    "figs/knockout_grace_cg_convergence.pdf", dpi=600
)

# %%
# =============================================================================
# GMSL comparison: posterior distributions for all five variants
# =============================================================================


def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (
        std * np.sqrt(2 * np.pi)
    )


variant_gmsl = [
    (
        "Full (SSH+ice+GRACE)",
        gmsl_full,
        colors.new_method,
    ),
    (
        "No SSH altimetry",
        gmsl_no_ssh,
        colors.ice_altimetry,
    ),
    (
        "No ice altimetry",
        gmsl_no_ice,
        colors.ocean_altimetry,
    ),
    ("No GRACE", gmsl_no_grace, colors.firn),
]

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
            label=(
                f"{label}\n"
                f"(mean={exp_mm:.2f}, std={std_mm:.2e} mm)"
            ),
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
    "Knockout Test: GMSL Posterior Distributions (with GRACE)"
)
ax_gmsl.legend(fontsize=8, loc="upper left")
fig_gmsl.tight_layout()
fig_gmsl.savefig("figs/knockout_grace_gmsl.pdf", dpi=600)

# Print summary table
print("\nGMSL Summary")
print(f"  True GMSL:           {total_gmsl_true_mm:.4f} mm")
for label, (exp_mm, std_mm), _ in variant_gmsl:
    sigma = abs(exp_mm - total_gmsl_true_mm) / std_mm
    print(
        f"  {label:<30}: mean={exp_mm:.4f} mm, "
        f"std={std_mm:.2e} mm, {sigma:.2f} sigma from truth"
    )

# %%
# =============================================================================
# Component grid: true vs posterior expectations
# Rows: ice thickness, firn thickness, ODT height
# Columns: true | full | no SSH | no ice | no GRACE
# =============================================================================

posteriors_ordered = [
    ("Full\n(SSH+ice+GRACE)", posterior_full),
    ("No SSH\naltimetry", posterior_no_ssh),
    ("No ice\naltimetry", posterior_no_ice),
    ("No GRACE", posterior_no_grace),
]


def field_mm(shgrid, projection_mask):
    scale = fp.length_scale * 1000  # m → mm
    return (shgrid * projection_mask * scale).data.astype(
        float
    )


def sym_clim(*arrays):
    vals = np.concatenate(
        [a[np.isfinite(a)].ravel() for a in arrays]
    )
    return np.nanmax(np.abs(vals))


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

# Lon/lat for pcolormesh
_sample_grid = ice_true
_raw_lats = _sample_grid.lats()
_raw_lons = _sample_grid.lons()
_lons_shifted = np.where(
    _raw_lons > 180, _raw_lons - 360, _raw_lons
)
_sort_idx = np.argsort(_lons_shifted)
_lons_plot = _lons_shifted[_sort_idx]
_lon_grid, _lat_grid = np.meshgrid(_lons_plot, _raw_lats)

n_rows = len(component_rows)
n_cols = 1 + len(posteriors_ordered)  # true + 5 posteriors

fig_grid, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(11, 6.5),
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
            ax.set_title(col_title, fontsize=7, pad=4)
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

    cb = fig_grid.colorbar(
        im,
        ax=axes[row_idx, :],
        orientation="horizontal",
        shrink=0.6,
        label=unit_label,
    )
    cb.ax.tick_params(labelsize=7)

fig_grid.suptitle(
    "Knockout Test: True vs Posterior Expectations (with GRACE)\n"
    "(rows: ice / firn / ocean dynamics; "
    "columns: true and each inversion variant)",
    fontsize=9,
)
fig_grid.savefig(
    "figs/knockout_grace_component_grid.pdf",
    dpi=600,
    bbox_inches="tight",
)

print("\nAll figures saved to figs/")

# %%
# =============================================================================
# Bivariate corner: Ice GMSL vs Firn GMSL for all knockout variants
#
# Two outputs:
#   1. Overlay figure  — all 5 variants on shared axes (1-sigma ellipses +
#      marginal PDFs) so sensitivity to each dropped data type is directly
#      visible.
#   2. Individual plot_bivariate_corner figures — one per variant, saved
#      separately for detailed inspection.
# =============================================================================

# Compute 2D posteriors for all variants
variants_2d = [
    (
        "Full (SSH+ice+GRACE)",
        post2d_full,
        colors.new_method,
    ),
    (
        "No SSH altimetry",
        post2d_no_ssh,
        colors.ice_altimetry,
    ),
    (
        "No ice altimetry",
        post2d_no_ice,
        colors.ocean_altimetry,
    ),
    (
        "No GRACE",
        post2d_no_grace,
        colors.firn,
    ),
]

# %%
# =============================================================================
# 1. Overlay bivariate corner figure
#
# Layout mirrors plot_bivariate_corner:
#   top-left  — marginal PDFs for ice GMSL (all variants)
#   bottom-left — 2D 1-sigma ellipses (all variants)
#   bottom-right — marginal PDFs for firn GMSL, rotated (all variants)
#   top-right — legend
# =============================================================================

fig_ov, axes_ov = plt.subplots(
    2,
    2,
    figsize=(8, 8),
    gridspec_kw={
        "width_ratios": [2, 1],
        "height_ratios": [1, 2],
    },
)
ax_top = axes_ov[0, 0]  # ice GMSL marginals
ax_main = axes_ov[1, 0]  # 2D joint
ax_right = axes_ov[1, 1]  # firn GMSL marginals (rotated)
ax_legend = axes_ov[0, 1]
ax_legend.axis("off")

for label, measure_2d, color in variants_2d:
    mu = measure_2d.expectation
    cov = measure_2d.covariance.matrix(
        dense=True, parallel=False
    )

    sigma0 = np.sqrt(cov[0, 0])
    sigma1 = np.sqrt(cov[1, 1])

    # -- Top: ice GMSL marginal --
    x0 = np.linspace(
        mu[0] - 4 * sigma0, mu[0] + 4 * sigma0, 300
    )
    ax_top.plot(
        x0,
        stats.norm.pdf(x0, mu[0], sigma0),
        color=color,
        linewidth=1.6,
        label=label,
    )

    # -- Right: firn GMSL marginal (rotated) --
    x1 = np.linspace(
        mu[1] - 4 * sigma1, mu[1] + 4 * sigma1, 300
    )
    ax_right.plot(
        stats.norm.pdf(x1, mu[1], sigma1),
        x1,
        color=color,
        linewidth=1.6,
    )

    # -- Main: 1-sigma and 2-sigma ellipses --
    rv = stats.multivariate_normal(mu, cov)
    sigma_level = rv.pdf(mu) * np.exp(-0.5)
    sigma_level_2 = rv.pdf(mu) * np.exp(-2.0)
    x_grid = np.linspace(
        mu[0] - 3.75 * sigma0,
        mu[0] + 3.75 * sigma0,
        120,
    )
    y_grid = np.linspace(
        mu[1] - 3.75 * sigma1,
        mu[1] + 3.75 * sigma1,
        120,
    )
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = rv.pdf(np.dstack((X, Y)))
    ax_main.contour(
        X,
        Y,
        Z,
        levels=[sigma_level],
        colors=[color],
        linewidths=1.8,
        linestyles="-",
    )
    ax_main.contour(
        X,
        Y,
        Z,
        levels=[sigma_level_2],
        colors=[color],
        linewidths=1.8,
        linestyles=":",
    )
    ax_main.plot(
        mu[0],
        mu[1],
        "+",
        color=color,
        markersize=8,
        mew=2,
    )

# True values
ax_top.axvline(
    true_ice_gmsl_mm,
    color=colors.true,
    linestyle="--",
    linewidth=1.5,
    label="True",
)
ax_right.axhline(
    true_firn_gmsl_mm,
    color=colors.true,
    linestyle="--",
    linewidth=1.5,
)
ax_main.plot(
    true_ice_gmsl_mm,
    true_firn_gmsl_mm,
    "kx",
    markersize=10,
    mew=2,
    label="True",
    zorder=5,
)

ax_top.set_ylabel("Density")
ax_top.set_xticklabels([])
ax_top.set_yticklabels([])
ax_right.set_xlabel("Density")
ax_right.set_yticklabels([])
ax_main.set_xlabel("Ice GMSL (mm)")
ax_main.set_ylabel("Firn GMSL (mm)")

handles, labels_leg = ax_top.get_legend_handles_labels()
h_main, l_main = ax_main.get_legend_handles_labels()
handles += h_main
labels_leg += l_main
ax_legend.legend(
    handles,
    labels_leg,
    loc="center",
    fontsize=9,
    frameon=False,
)

fig_ov.suptitle(
    "Knockout Sensitivity: Ice vs Firn GMSL\n"
    "(1-sigma and 2-sigma ellipses, all variants)",
    fontsize=13,
)
plt.tight_layout()
fig_ov.savefig(
    "figs/knockout_grace_gmsl_bivariate_overlay.pdf",
    dpi=600,
)
fig_ov.savefig(
    "figs/knockout_grace_gmsl_bivariate_overlay.png",
    dpi=200,
)

# %%
# =============================================================================
fig_ov.savefig(
    "figs/knockout_grace_gmsl_bivariate_overlay.png",
    dpi=200,
)

# %%
# =============================================================================
print("Bivariate corner figures saved to figs/")
