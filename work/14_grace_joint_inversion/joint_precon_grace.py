# %%
import matplotlib.pyplot as plt
import numpy as np
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
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)
from pyslfp.operators import grace_operator
from tqdm import tqdm

from project import colors
from project.operators import (
    ice_thickness_to_estimated_gmsl_operator,
)
from pygeoinf_extras import standard_dev
from pygeoinf_extras.plots import plot_bivariate_corner
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.ocean_dynamics import OceanDynamics

# %%
# =============================================================================
# Full-resolution model setup
# =============================================================================

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

dir = "figs1"
measure_error_std = 0.001

# GRACE-specific error (in non-dimensionalised units)
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
# Full-resolution model space and prior
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
# Full-resolution forward operator with GRACE
# =============================================================================
#
# Extends the factored block structure (P_left @ F_middle @ L_right)
# from joint_precon_factored.py by adding GRACE as a 4th observation row.
#
#   L_right (4×3):  load operators + routing permutation  [unchanged]
#   F_middle(4×4):  block_diag(F, I_odt, I_ice, I_firn)  [unchanged]
#   P_left  (3×4):  rows = SSH altimetry / ice altimetry / GRACE
#
# The GRACE row maps from the fingerprint response_space via the spherical
# harmonic sampling operator; all other intermediate spaces contribute zero.
#
# =============================================================================


def _build_forward_operator(
    fp,
    fp_op,
    ice,
    odt,
    ssh_altimetry,
    ice_altimetry,
    grace_observation_degree,
):
    """
    Build the factored forward operator extended with GRACE observations.

    Observation rows (in order):
      0 — SSH altimetry
      1 — Ice altimetry
      2 — GRACE spherical harmonic coefficients
    """
    # -- Spaces --
    load_space = fp_op.domain
    response_space = fp_op.codomain
    ice_space = ice.ice_thickness.domain
    firn_space = ice.firn_thickness.domain
    odt_space = odt.height_measure.domain

    # -- Component operators --
    F = fp_op
    S = sea_surface_height_operator(fp, response_space)
    slc_proj = response_space.subspace_projection(0)
    slc_space = slc_proj.codomain

    L_I = ice.ice_thickness_to_load_operator
    L_F = ice.firn_thickness_to_load_operator
    L_W = sea_level_change_to_load_operator(fp, load_space)

    # -- Point evaluation operators --
    P_S_ssh = ssh_altimetry.point_evaluation_operator(
        S.codomain
    )
    P_S_odt = ssh_altimetry.point_evaluation_operator(
        odt_space
    )

    P_I_ice = ice_altimetry.point_evaluation_operator(
        ice_space
    )
    P_I_firn = ice_altimetry.point_evaluation_operator(
        firn_space
    )

    # -- GRACE operator --
    grace_op = grace_operator(
        response_space, grace_observation_degree
    )

    # -- Identities --
    id_odt = odt_space.identity_operator()
    id_ice = ice_space.identity_operator()
    id_firn = firn_space.identity_operator()

    # -- Observation spaces --
    ssh_obs = P_S_ssh.codomain
    ice_obs = P_I_ice.codomain
    grace_obs = grace_op.codomain

    # == L_right (4×3) ==
    L_right = BlockLinearOperator(
        [
            [L_I, L_F, L_W],
            [
                ice_space.zero_operator(codomain=odt_space),
                firn_space.zero_operator(
                    codomain=odt_space
                ),
                id_odt,
            ],
            [
                id_ice,
                firn_space.zero_operator(
                    codomain=ice_space
                ),
                odt_space.zero_operator(codomain=ice_space),
            ],
            [
                ice_space.zero_operator(
                    codomain=firn_space
                ),
                id_firn,
                odt_space.zero_operator(
                    codomain=firn_space
                ),
            ],
        ]
    )

    # == F_middle (4×4 block diagonal) ==
    F_middle = BlockDiagonalLinearOperator(
        [F, id_odt, id_ice, id_firn]
    )

    # == P_left (3×4) ==
    # [[P_S·S,    P_S_odt,  0,       0        ],
    #  [0,        0,        P_I_ice, P_I_firn  ],
    #  [grace_op, 0,        0,       0         ]]
    P_left = BlockLinearOperator(
        [
            [
                P_S_ssh @ S,
                P_S_odt,
                ice_space.zero_operator(codomain=ssh_obs),
                firn_space.zero_operator(codomain=ssh_obs),
            ],
            [
                response_space.zero_operator(
                    codomain=ice_obs
                ),
                odt_space.zero_operator(codomain=ice_obs),
                P_I_ice,
                P_I_firn,
            ],
            [
                grace_op,
                odt_space.zero_operator(codomain=grace_obs),
                ice_space.zero_operator(codomain=grace_obs),
                firn_space.zero_operator(
                    codomain=grace_obs
                ),
            ],
        ]
    )

    return P_left @ F_middle @ L_right, grace_op.codomain


forward_operator, grace_obs = _build_forward_operator(
    fp,
    fp_op,
    ice,
    odt,
    ssh_altimetry,
    ice_altimetry,
    grace_observation_degree,
)

data_space = forward_operator.codomain

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
# Data error (component-wise: altimetry/ice share measure_error_std;
# GRACE uses its own grace_std)
# =============================================================================

# Extract observation sub-spaces from the data_space order:
# 0: SSH altimetry, 1: ice altimetry, 2: GRACE
ssh_obs_space = data_space.subspaces[0]
ice_obs_space = data_space.subspaces[1]

data_error_measure = GaussianMeasure.from_direct_sum(
    [
        GaussianMeasure.from_standard_deviation(
            ssh_obs_space, measure_error_std
        ),
        GaussianMeasure.from_standard_deviation(
            ice_obs_space, measure_error_std
        ),
        GaussianMeasure.from_standard_deviation(
            grace_obs, grace_std
        ),
    ]
)

forward_problem = LinearForwardProblem(
    forward_operator,
    data_error_measure=data_error_measure,
)

model_true, data = forward_problem.synthetic_model_and_data(
    model_prior
)

# %%
# =============================================================================
# Preconditioner setup (lower-resolution joint model with GRACE)
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

# %%
# =============================================================================
# Check ocean / ice point consistency between full and preconditioner grids
# =============================================================================

precon_ssh_altimetry = GridPoints.ocean_altimetry(
    precon_fp, 10.0, 66.0
)
precon_ice_altimetry = GridPoints.ice(precon_fp, 15.0)

precon_ssh_ocean_set = set(precon_ssh_altimetry.coords)
full_ssh_ocean_set = set(ssh_altimetry.coords)
ssh_points_not_in_precon = (
    full_ssh_ocean_set - precon_ssh_ocean_set
)
print(
    f"Full-resolution SSH ocean points: {len(full_ssh_ocean_set)}"
)
print(
    f"Preconditioner SSH ocean points: {len(precon_ssh_ocean_set)}"
)
print(
    f"Full-res SSH points NOT in preconditioner ocean: "
    f"{len(ssh_points_not_in_precon)}"
)
if ssh_points_not_in_precon:
    print(
        "WARNING: Some SSH ocean points from the full grid "
        "are not ocean on the preconditioner grid:"
    )
    for lat, lon in sorted(ssh_points_not_in_precon):
        print(f"  lat={lat:.1f}, lon={lon:.1f}")

precon_ice_set = set(precon_ice_altimetry.coords)
full_ice_set = set(ice_altimetry.coords)
ice_points_not_in_precon = full_ice_set - precon_ice_set
print(
    f"\nFull-resolution ice altimetry points: {len(full_ice_set)}"
)
print(
    f"Preconditioner ice altimetry points: {len(precon_ice_set)}"
)
print(
    f"Full-res ice points NOT in preconditioner: "
    f"{len(ice_points_not_in_precon)}"
)
if ice_points_not_in_precon:
    print(
        "WARNING: Some ice altimetry points from the full grid "
        "are not ice on the preconditioner grid:"
    )
    for lat, lon in sorted(ice_points_not_in_precon):
        print(f"  lat={lat:.1f}, lon={lon:.1f}")

# %%
# =============================================================================
# Preconditioner forward operator (low-res, same 4-row structure,
# sampling at full-resolution coordinates)
# =============================================================================

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

# GRACE operator at low resolution — same observation_degree
precon_grace_op = grace_operator(
    precon_response_space, grace_observation_degree
)

precon_id_odt = precon_odt_space.identity_operator()
precon_id_ice = precon_ice_space.identity_operator()
precon_id_firn = precon_firn_space.identity_operator()

precon_ssh_obs = precon_P_S_ssh.codomain
precon_ice_obs = precon_P_I_ice.codomain
precon_grace_obs = precon_grace_op.codomain

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

precon_P_left = BlockLinearOperator(
    [
        [
            precon_P_S_ssh @ precon_S,
            precon_P_S_odt,
            precon_ice_space.zero_operator(
                codomain=precon_ssh_obs
            ),
            precon_firn_space.zero_operator(
                codomain=precon_ssh_obs
            ),
        ],
        [
            precon_response_space.zero_operator(
                codomain=precon_ice_obs
            ),
            precon_odt_space.zero_operator(
                codomain=precon_ice_obs
            ),
            precon_P_I_ice,
            precon_P_I_firn,
        ],
        [
            precon_grace_op,
            precon_odt_space.zero_operator(
                codomain=precon_grace_obs
            ),
            precon_ice_space.zero_operator(
                codomain=precon_grace_obs
            ),
            precon_firn_space.zero_operator(
                codomain=precon_grace_obs
            ),
        ],
    ]
)

precon_forward_operator = (
    precon_P_left @ precon_F_middle @ precon_L_right
)

# %%
# =============================================================================
# Form the preconditioner inverse via eigen-decomposition
# =============================================================================

precon_forward_problem = LinearForwardProblem(
    precon_forward_operator,
    data_error_measure=data_error_measure,
)

precon_bayesian_inversion = LinearBayesianInversion(
    precon_forward_problem, precon_model_prior
)

precon_normal_operator = (
    precon_bayesian_inversion.normal_operator
)

print(
    "Forming the preconditioner via eigen-decomposition..."
)
solver = EigenSolver(parallel=False)
precon_inverse_normal_operator = solver(
    precon_normal_operator
)
print("Preconditioner ready.")

# %%
# =============================================================================
# Full inversion with preconditioner
# =============================================================================

bayesian_inversion = LinearBayesianInversion(
    forward_problem, model_prior
)

print("Starting inversion...")
residuals = []
pbar = tqdm(desc="CG solve")


def progress_callback(xk):
    residuals.append(np.linalg.norm(xk))
    pbar.set_postfix({"||x||": f"{residuals[-1]:.2e}"})
    pbar.update(1)


model_posterior_measure = (
    bayesian_inversion.model_posterior_measure(
        data,
        CGMatrixSolver(
            callback=progress_callback, maxiter=500
        ),
        preconditioner=precon_inverse_normal_operator,
    )
)
pbar.close()
print("")
print("Inversion complete.")

plt.figure(figsize=(3, 2))
plt.semilogy(
    residuals, marker="o", linestyle="-", markersize=3
)
plt.title("Convergence of CG Solver")
plt.xlabel("Iteration")
plt.ylabel("Norm of Solution ($||x_k||$)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig(
    f"{dir}/joint_precon_grace_cg_convergence.pdf", dpi=600
)

model_posterior_expectation = (
    model_posterior_measure.expectation
)

# %%
# =============================================================================
# Extract components
# =============================================================================

ice_thickness_true = model_true[0]
ice_thickness_posterior_expectation = (
    model_posterior_expectation[0]
)
firn_thickness_true = model_true[1]
firn_thickness_posterior_expectation = (
    model_posterior_expectation[1]
)
odt_height_true = model_true[2]
odt_height_posterior_expectation = (
    model_posterior_expectation[2]
)

# %%
# =============================================================================
# Plotting — ice thickness
# =============================================================================

max_abs_ice_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    ice_thickness_true.data.flatten(),
                    ice_thickness_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

fig1, ax1, im1 = plot(
    1000
    * ice_thickness_true
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.set_title("a) True Ice Thickness Change")
fig1.tight_layout()

fig2, ax2, im2 = plot(
    1000
    * ice_thickness_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.set_title(
    "b) Posterior Expectation (GRACE + altimetry)"
)
fig2.tight_layout()

# %%
# =============================================================================
# Plotting — firn thickness
# =============================================================================

max_abs_firn_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    firn_thickness_true.data.flatten(),
                    firn_thickness_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

fig3, ax3, im3 = plot(
    1000
    * firn_thickness_true
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_firn_change,
    vmax=max_abs_firn_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Firn Thickness Change (mm)",
)
ax3.set_title("c) True Firn Thickness Change")
fig3.tight_layout()

fig4, ax4, im4 = plot(
    1000
    * firn_thickness_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_firn_change,
    vmax=max_abs_firn_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Firn Thickness Change (mm)",
)
ax4.set_title(
    "d) Posterior Expectation (Inferred from Data)"
)
fig4.tight_layout()

# %%
# =============================================================================
# Plotting — ocean dynamics
# =============================================================================

max_abs_odt_height_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    odt_height_true.data.flatten(),
                    odt_height_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

fig5, ax5, im5 = plot(
    1000
    * odt_height_true
    * fp.length_scale
    * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_odt_height_change,
    vmax=max_abs_odt_height_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="ODT Height Change (mm)",
)
ax5.set_title("e) True Ocean Height Change")
fig5.tight_layout()

fig6, ax6, im6 = plot(
    1000
    * odt_height_posterior_expectation
    * fp.length_scale
    * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_odt_height_change,
    vmax=max_abs_odt_height_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="ODT Height Change (mm)",
)
ax6.set_title(
    "f) Posterior Expectation (Inferred from Data)"
)
fig6.tight_layout()

# %%
# =============================================================================
# Plotting — sea level change
# =============================================================================

slc_true = model_space_to_slc_operator(model_true)[0]
slc_posterior_expectation = model_space_to_slc_operator(
    model_posterior_expectation
)[0]

max_abs_sl_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (
                        slc_true * fp.ocean_projection()
                    ).data.flatten(),
                    (
                        slc_posterior_expectation
                        * fp.ocean_projection()
                    ).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

fig7, ax7, im7 = plot(
    1000
    * slc_true
    * fp.length_scale
    * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Sea Level Change (mm)",
)
ax7.set_title("g) True Sea-Level Change")
fig7.tight_layout()

fig8, ax8, im8 = plot(
    1000
    * slc_posterior_expectation
    * fp.length_scale
    * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Sea Level Change (mm)",
)
ax8.set_title(
    "h) Posterior Expectation (Inferred from Data)"
)
fig8.tight_layout()

# %%
# =============================================================================
# Plotting — total ice + firn load
# =============================================================================

total_load = ice.ice_thickness_to_load_operator(
    ice_thickness_true
) + ice.firn_thickness_to_load_operator(firn_thickness_true)

total_load_posterior = ice.ice_thickness_to_load_operator(
    ice_thickness_posterior_expectation
) + ice.firn_thickness_to_load_operator(
    firn_thickness_posterior_expectation
)

fig9, ax9, im9 = plot(
    total_load * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Total Ice+Firn Load Change (kg)",
)
ax9.set_title("i) True Total Ice+Firn Load Change")
fig9.tight_layout()

fig10, ax10, im10 = plot(
    total_load_posterior * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    figsize=(3, 2),
    gridlines=False,
    colorbar_label="Total Ice+Firn Load Change (kg)",
)
ax10.set_title(
    "j) Posterior Expectation of Total Ice+Firn Load Change"
)
fig10.tight_layout()

# %%
# =============================================================================
# Save figures
# =============================================================================

fig1.savefig(
    f"{dir}/joint_precon_grace_ice_thickness.pdf", dpi=600
)
fig2.savefig(
    f"{dir}/joint_precon_grace_ice_thickness_posterior.pdf",
    dpi=600,
)
fig3.savefig(
    f"{dir}/joint_precon_grace_firn_thickness.pdf", dpi=600
)
fig4.savefig(
    f"{dir}/joint_precon_grace_firn_thickness_posterior.pdf",
    dpi=600,
)
fig5.savefig(
    f"{dir}/joint_precon_grace_odt_height.pdf", dpi=600
)
fig6.savefig(
    f"{dir}/joint_precon_grace_odt_height_posterior.pdf",
    dpi=600,
)
fig7.savefig(f"{dir}/joint_precon_grace_slc.pdf", dpi=600)
fig8.savefig(
    f"{dir}/joint_precon_grace_slc_posterior.pdf", dpi=600
)
fig9.savefig(
    f"{dir}/joint_precon_grace_total_load.pdf", dpi=600
)
fig10.savefig(
    f"{dir}/joint_precon_grace_total_load_posterior.pdf",
    dpi=600,
)

# %%
# =============================================================================
# Ice vs firn GMSL covariance (corner plot)
# =============================================================================

ice_gmsl_op = ice.ice_thickness_to_gmsl_operator
firn_gmsl_op = ice.firn_thickness_to_gmsl_operator

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

ice_gmsl_post = model_posterior_measure.affine_mapping(
    operator=ice_gmsl_row
)
firn_gmsl_post = model_posterior_measure.affine_mapping(
    operator=firn_gmsl_row
)

sum_gmsl_post = model_posterior_measure.affine_mapping(
    operator=ice_gmsl_row + firn_gmsl_row
)
var_ice = standard_dev(ice_gmsl_post) ** 2
var_firn = standard_dev(firn_gmsl_post) ** 2
var_sum = standard_dev(sum_gmsl_post) ** 2
cross_cov = 0.5 * (var_sum - var_ice - var_firn)

mu_ice = ice_gmsl_post.expectation[0]
mu_firn = firn_gmsl_post.expectation[0]
mean_2d = np.array([mu_ice, mu_firn])
cov_2d = np.array(
    [[var_ice, cross_cov], [cross_cov, var_firn]]
)

joint_gmsl_posterior_measure = (
    GaussianMeasure.from_covariance_matrix(
        EuclideanSpace(2), cov_2d, expectation=mean_2d
    )
)

true_ice_gmsl = ice_gmsl_row(model_true)[0]
true_firn_gmsl = firn_gmsl_row(model_true)[0]

fig_cov, axes_cov = plot_bivariate_corner(
    joint_gmsl_posterior_measure,
    true_values=np.array([true_ice_gmsl, true_firn_gmsl]),
    labels=["Ice GMSL (mm)", "Firn GMSL (mm)"],
    title="Joint Posterior: Ice vs Firn GMSL Contributions",
    figsize=(6.5, 6.5),
    pdf_colors=[colors.ice, colors.firn],
)
fig_cov.savefig(
    f"{dir}/joint_precon_grace_ice_firn_gmsl_covariance.pdf",
    dpi=600,
)

import cartopy.crs as ccrs
from pyshtools import SHGrid


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


fig, axs = plt.subplots(
    3,
    2,
    figsize=(6.5, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

# Row 1: Ice
im1 = plot_shgrid_robinson_on_ax(
    1000
    * ice_thickness_true
    * fp.length_scale
    * fp.ice_projection(),
    axs[0, 0],
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    symmetric=False,
)
axs[0, 0].set_title("True Ice Change (mm)", fontsize=10)

im2 = plot_shgrid_robinson_on_ax(
    1000
    * ice_thickness_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    axs[0, 1],
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    symmetric=False,
)
axs[0, 1].set_title(
    "Posterior Ice Change (mm)", fontsize=10
)

# Row 2: Firn
im3 = plot_shgrid_robinson_on_ax(
    1000
    * firn_thickness_true
    * fp.length_scale
    * fp.ice_projection(),
    axs[1, 0],
    cmap="seismic",
    vmin=-max_abs_firn_change,
    vmax=max_abs_firn_change,
    symmetric=False,
)
axs[1, 0].set_title("True Firn Change (mm)", fontsize=10)

im4 = plot_shgrid_robinson_on_ax(
    1000
    * firn_thickness_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    axs[1, 1],
    cmap="seismic",
    vmin=-max_abs_firn_change,
    vmax=max_abs_firn_change,
    symmetric=False,
)
axs[1, 1].set_title(
    "Posterior Firn Change (mm)", fontsize=10
)

# Row 3: ODT
im5 = plot_shgrid_robinson_on_ax(
    1000
    * odt_height_true
    * fp.length_scale
    * fp.ocean_projection(),
    axs[2, 0],
    cmap="seismic",
    vmin=-max_abs_odt_height_change,
    vmax=max_abs_odt_height_change,
    symmetric=False,
)
axs[2, 0].set_title("True ODT Change (mm)", fontsize=10)

im6 = plot_shgrid_robinson_on_ax(
    1000
    * odt_height_posterior_expectation
    * fp.length_scale
    * fp.ocean_projection(),
    axs[2, 1],
    cmap="seismic",
    vmin=-max_abs_odt_height_change,
    vmax=max_abs_odt_height_change,
    symmetric=False,
)
axs[2, 1].set_title(
    "Posterior ODT Change (mm)", fontsize=10
)

plt.tight_layout()

# Add colorbars
cbar1 = fig.colorbar(
    im2,
    ax=axs[0, :],
    orientation="vertical",
    shrink=0.8,
    pad=0.02,
    aspect=20,
)
cbar1.set_label("Ice Change (mm)", fontsize=9)
cbar2 = fig.colorbar(
    im4,
    ax=axs[1, :],
    orientation="vertical",
    shrink=0.8,
    pad=0.02,
    aspect=20,
)
cbar2.set_label("Firn Change (mm)", fontsize=9)
cbar3 = fig.colorbar(
    im6,
    ax=axs[2, :],
    orientation="vertical",
    shrink=0.8,
    pad=0.02,
    aspect=20,
)
cbar3.set_label("ODT Change (mm)", fontsize=9)

fig.savefig(
    f"{dir}/joint_precon_grace_6panel.pdf",
    dpi=600,
    bbox_inches="tight",
)
