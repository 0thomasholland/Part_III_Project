# %%
"""
Regional Greenland vs Antarctica comparison.
Only GRACE case (ice + firn + ODT + GRACE) is considered.
Three bivariate plots are produced (one each for total / ice / firn thickness
GMSL contributions), showing the posterior.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    BlockDiagonalLinearOperator,
    BlockLinearOperator,
    CGMatrixSolver,
    EigenSolver,
    GaussianMeasure,
    HilbertSpaceDirectSum,
    LinearBayesianInversion,
    LinearForwardProblem,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
    read_gloss_tide_gauge_data,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    tide_gauge_operator,
)
from pyslfp.operators import grace_operator
from scipy import stats
from tqdm import tqdm

from project import colors
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.ocean_dynamics import OceanDynamics

os.makedirs("figs", exist_ok=True)

# %%
# =============================================================================
# Shared physical setup
# =============================================================================

lmax = 128
lmax_precon = 32
measure_error_std = 0.001

fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%
# --- Observation points ---

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
filtered_lats = [l for l in filtered_lats if l is not None]
filtered_lons = [l for l in filtered_lons if l is not None]
tide_gauge_points = list(zip(filtered_lats, filtered_lons))

OD_pattern = OceanDynamics.DataPattern()

# %%
# --- Preconditioner fingerprint ---

precon_fp = FingerPrint(lmax=lmax_precon)
precon_fp.set_state_from_ice_ng(
    version=IceModel.ICE7G, date=0.0
)
precon_fp_op = precon_fp.as_sobolev_linear_operator(
    2, precon_fp.mean_sea_floor_radius * 0.1
)

# %%
# =============================================================================
# Regional weighting functions (Greenland and Antarctica)
# The * 1000 * fp.length_scale converts non-dimensional thickness → mm GMSL.
# =============================================================================

gl_wf = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

ant_wf = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * (
        fp.west_antarctic_projection(value=0)
        + fp.east_antarctic_projection(value=0)
    )
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

# %%
# =============================================================================
# Overlay plotting helper
# =============================================================================


def plot_bivariate_overlay(
    measures_info,
    xlabel,
    ylabel,
    title,
    figsize=(8, 8),
):
    """
    Overlay multiple 2D posterior measures on a bivariate corner plot.
    """
    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [2, 1],
            "height_ratios": [1, 2],
        },
    )
    ax_gl = axes[0, 0]  # GL marginal (top-left, upright)
    ax_2d = axes[1, 0]  # 2D contour (bottom-left)
    ax_ant = axes[
        1, 1
    ]  # ANT marginal (bottom-right, rotated)
    axes[0, 1].axis("off")

    for info in measures_info:
        m = info["measure"]
        c = info["color"]
        tv = info.get("true_val")

        mean = m.expectation
        cov = m.covariance.matrix(dense=True)
        mu_gl, mu_ant = mean[0], mean[1]
        sig_gl = np.sqrt(cov[0, 0])
        sig_ant = np.sqrt(cov[1, 1])

        # --- GL marginal ---
        x_gl = np.linspace(
            mu_gl - 4.0 * sig_gl, mu_gl + 4.0 * sig_gl, 400
        )
        pdf_gl = stats.norm.pdf(x_gl, mu_gl, sig_gl)
        ax_gl.plot(
            x_gl, pdf_gl, color=c, label=info["label"]
        )
        ax_gl.fill_between(x_gl, pdf_gl, alpha=0.2, color=c)

        # --- ANT marginal (rotated: x=density, y=value) ---
        x_ant = np.linspace(
            mu_ant - 4.0 * sig_ant,
            mu_ant + 4.0 * sig_ant,
            400,
        )
        pdf_ant = stats.norm.pdf(x_ant, mu_ant, sig_ant)
        ax_ant.plot(pdf_ant, x_ant, color=c)
        ax_ant.fill_betweenx(
            x_ant, 0, pdf_ant, alpha=0.2, color=c
        )

        # --- 2D 1-sigma contour ---
        x2 = np.linspace(
            mu_gl - 4.0 * sig_gl, mu_gl + 4.0 * sig_gl, 120
        )
        y2 = np.linspace(
            mu_ant - 4.0 * sig_ant,
            mu_ant + 4.0 * sig_ant,
            120,
        )
        X, Y = np.meshgrid(x2, y2)
        rv = stats.multivariate_normal(
            [mu_gl, mu_ant],
            [
                [cov[0, 0], cov[0, 1]],
                [cov[1, 0], cov[1, 1]],
            ],
        )
        Z = rv.pdf(np.dstack((X, Y)))
        sigma_level = rv.pdf([mu_gl, mu_ant]) * np.exp(-0.5)
        ax_2d.contour(
            X,
            Y,
            Z,
            levels=[sigma_level],
            colors=[c],
            linewidths=1.5,
        )
        ax_2d.plot(
            mu_gl,
            mu_ant,
            "+",
            color=c,
            markersize=10,
            mew=2,
        )

        # --- True values ---
        if tv is not None:
            ax_gl.axvline(
                tv[0], color=c, linestyle="--", alpha=0.7
            )
            ax_ant.axhline(
                tv[1], color=c, linestyle="--", alpha=0.7
            )
            ax_2d.plot(
                tv[0],
                tv[1],
                "x",
                color=c,
                markersize=10,
                mew=2,
            )

    ax_gl.set_ylabel("Density")
    ax_gl.set_xticklabels([])
    ax_ant.set_xlabel("Density")
    ax_ant.set_yticklabels([])
    ax_2d.set_xlabel(xlabel)
    ax_2d.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13)
    ax_gl.legend(fontsize=8, loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axes


# %%
# =============================================================================
# INVERSION C — GRACE: ice + firn + ODT + GRACE
# =============================================================================

print("=" * 60)
print("INVERSION C: GRACE (ice + firn + ODT + GRACE)")
print("=" * 60)

grace_std_dev_m = 0.0027
grace_std_c = grace_std_dev_m / fp.length_scale
grace_obs_degree = 96

ice_c = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    firn_gmsl_std=0.002,
    firn_density=0.3 * fp.ice_density,
    include_firn=True,
)
odt_c = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.004,
    length_scale=10000.0,
    pattern=OD_pattern,
)

model_space_c = HilbertSpaceDirectSum(
    [
        ice_c.ice_thickness.domain,
        ice_c.firn_thickness.domain,
        odt_c.height_measure.domain,
    ]
)
model_prior_c = GaussianMeasure.from_direct_sum(
    [
        ice_c.ice_thickness,
        ice_c.firn_thickness,
        odt_c.height_measure,
    ]
)

tide_op_c = tide_gauge_operator(
    ice_c.load_to_slc_operator.codomain, tide_gauge_points
)

load_space_c = fp_op.domain
response_space_c = fp_op.codomain
S_c = sea_surface_height_operator(fp, response_space_c)
slc_proj_c = response_space_c.subspace_projection(0)
slc_space_c = slc_proj_c.codomain

L_I_c = ice_c.ice_thickness_to_load_operator
L_F_c = ice_c.firn_thickness_to_load_operator
L_W_c = sea_level_change_to_load_operator(fp, load_space_c)

P_S_ssh_c = ssh_altimetry.point_evaluation_operator(
    S_c.codomain
)
P_S_odt_c = ssh_altimetry.point_evaluation_operator(
    odt_c.height_measure.domain
)
P_T_slc_c = slc_space_c.point_evaluation_operator(
    tide_gauge_points
)
P_T_odt_c = (
    odt_c.height_measure.domain.point_evaluation_operator(
        tide_gauge_points
    )
)
P_I_ice_c = ice_altimetry.point_evaluation_operator(
    ice_c.ice_thickness.domain
)
P_I_firn_c = ice_altimetry.point_evaluation_operator(
    ice_c.firn_thickness.domain
)
grace_op_c = grace_operator(
    response_space_c, grace_obs_degree
)

ice_space_c = ice_c.ice_thickness.domain
firn_space_c = ice_c.firn_thickness.domain
odt_space_c = odt_c.height_measure.domain
ssh_obs_c = P_S_ssh_c.codomain
tg_obs_c = P_T_slc_c.codomain
ice_obs_c = P_I_ice_c.codomain
grace_obs_c = grace_op_c.codomain

id_odt_c = odt_space_c.identity_operator()
id_ice_c = ice_space_c.identity_operator()
id_firn_c = firn_space_c.identity_operator()

L_right_c = BlockLinearOperator(
    [
        [L_I_c, L_F_c, L_W_c],
        [
            ice_space_c.zero_operator(codomain=odt_space_c),
            firn_space_c.zero_operator(
                codomain=odt_space_c
            ),
            id_odt_c,
        ],
        [
            id_ice_c,
            firn_space_c.zero_operator(
                codomain=ice_space_c
            ),
            odt_space_c.zero_operator(codomain=ice_space_c),
        ],
        [
            ice_space_c.zero_operator(
                codomain=firn_space_c
            ),
            id_firn_c,
            odt_space_c.zero_operator(
                codomain=firn_space_c
            ),
        ],
    ]
)
F_middle_c = BlockDiagonalLinearOperator(
    [fp_op, id_odt_c, id_ice_c, id_firn_c]
)
P_left_c = BlockLinearOperator(
    [
        [
            P_S_ssh_c @ S_c,
            P_S_odt_c,
            ice_space_c.zero_operator(codomain=ssh_obs_c),
            firn_space_c.zero_operator(codomain=ssh_obs_c),
        ],
        [
            P_T_slc_c @ slc_proj_c,
            P_T_odt_c,
            ice_space_c.zero_operator(codomain=tg_obs_c),
            firn_space_c.zero_operator(codomain=tg_obs_c),
        ],
        [
            response_space_c.zero_operator(
                codomain=ice_obs_c
            ),
            odt_space_c.zero_operator(codomain=ice_obs_c),
            P_I_ice_c,
            P_I_firn_c,
        ],
        [
            grace_op_c,
            odt_space_c.zero_operator(codomain=grace_obs_c),
            ice_space_c.zero_operator(codomain=grace_obs_c),
            firn_space_c.zero_operator(
                codomain=grace_obs_c
            ),
        ],
    ]
)
forward_op_c = P_left_c @ F_middle_c @ L_right_c
data_space_c = forward_op_c.codomain

ssh_obs_space_c = data_space_c.subspaces[0]
tg_obs_space_c = data_space_c.subspaces[1]
ice_obs_space_c = data_space_c.subspaces[2]

data_error_c = GaussianMeasure.from_direct_sum(
    [
        GaussianMeasure.from_standard_deviation(
            ssh_obs_space_c, measure_error_std
        ),
        GaussianMeasure.from_standard_deviation(
            tg_obs_space_c, measure_error_std
        ),
        GaussianMeasure.from_standard_deviation(
            ice_obs_space_c, measure_error_std
        ),
        GaussianMeasure.from_standard_deviation(
            grace_obs_c, grace_std_c
        ),
    ]
)

forward_problem_c = LinearForwardProblem(
    forward_op_c, data_error_measure=data_error_c
)
model_true_c, data_c = (
    forward_problem_c.synthetic_model_and_data(
        model_prior_c
    )
)

# --- Preconditioner C ---

precon_ice_c = IceSheetChange.global_ice(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    length_scale=0.1 * precon_fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    include_firn=True,
)
precon_odt_c = OceanDynamics(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    std=0.004,
    length_scale=10000.0,
    pattern=OD_pattern,
)
precon_model_prior_c = GaussianMeasure.from_direct_sum(
    [
        precon_ice_c.ice_thickness,
        precon_ice_c.firn_thickness,
        precon_odt_c.height_measure,
    ]
)
precon_tide_c = tide_gauge_operator(
    precon_ice_c.load_to_slc_operator.codomain,
    tide_gauge_points,
)

precon_load_space_c = precon_fp_op.domain
precon_response_space_c = precon_fp_op.codomain
precon_S_c = sea_surface_height_operator(
    precon_fp, precon_response_space_c
)
precon_slc_proj_c = (
    precon_response_space_c.subspace_projection(0)
)
precon_slc_space_c = precon_slc_proj_c.codomain
precon_L_I_c = precon_ice_c.ice_thickness_to_load_operator
precon_L_F_c = precon_ice_c.firn_thickness_to_load_operator
precon_L_W_c = sea_level_change_to_load_operator(
    precon_fp, precon_load_space_c
)
precon_ice_space_c = precon_ice_c.ice_thickness.domain
precon_firn_space_c = precon_ice_c.firn_thickness.domain
precon_odt_space_c = precon_odt_c.height_measure.domain
precon_id_odt_c = precon_odt_space_c.identity_operator()
precon_id_ice_c = precon_ice_space_c.identity_operator()
precon_id_firn_c = precon_firn_space_c.identity_operator()

precon_P_S_ssh_c = (
    precon_S_c.codomain.point_evaluation_operator(
        ssh_altimetry.coords
    )
)
precon_P_S_odt_c = (
    precon_odt_space_c.point_evaluation_operator(
        ssh_altimetry.coords
    )
)
precon_P_T_slc_c = (
    precon_slc_space_c.point_evaluation_operator(
        tide_gauge_points
    )
)
precon_P_T_odt_c = (
    precon_odt_space_c.point_evaluation_operator(
        tide_gauge_points
    )
)
precon_P_I_ice_c = (
    precon_ice_space_c.point_evaluation_operator(
        ice_altimetry.coords
    )
)
precon_P_I_firn_c = (
    precon_firn_space_c.point_evaluation_operator(
        ice_altimetry.coords
    )
)
precon_grace_op_c = grace_operator(
    precon_response_space_c, grace_obs_degree
)

precon_ssh_obs_c = precon_P_S_ssh_c.codomain
precon_tg_obs_c = precon_P_T_slc_c.codomain
precon_ice_obs_c = precon_P_I_ice_c.codomain
precon_grace_obs_c = precon_grace_op_c.codomain

precon_L_right_c = BlockLinearOperator(
    [
        [precon_L_I_c, precon_L_F_c, precon_L_W_c],
        [
            precon_ice_space_c.zero_operator(
                codomain=precon_odt_space_c
            ),
            precon_firn_space_c.zero_operator(
                codomain=precon_odt_space_c
            ),
            precon_id_odt_c,
        ],
        [
            precon_id_ice_c,
            precon_firn_space_c.zero_operator(
                codomain=precon_ice_space_c
            ),
            precon_odt_space_c.zero_operator(
                codomain=precon_ice_space_c
            ),
        ],
        [
            precon_ice_space_c.zero_operator(
                codomain=precon_firn_space_c
            ),
            precon_id_firn_c,
            precon_odt_space_c.zero_operator(
                codomain=precon_firn_space_c
            ),
        ],
    ]
)
precon_F_middle_c = BlockDiagonalLinearOperator(
    [
        precon_fp_op,
        precon_id_odt_c,
        precon_id_ice_c,
        precon_id_firn_c,
    ]
)
precon_P_left_c = BlockLinearOperator(
    [
        [
            precon_P_S_ssh_c @ precon_S_c,
            precon_P_S_odt_c,
            precon_ice_space_c.zero_operator(
                codomain=precon_ssh_obs_c
            ),
            precon_firn_space_c.zero_operator(
                codomain=precon_ssh_obs_c
            ),
        ],
        [
            precon_P_T_slc_c @ precon_slc_proj_c,
            precon_P_T_odt_c,
            precon_ice_space_c.zero_operator(
                codomain=precon_tg_obs_c
            ),
            precon_firn_space_c.zero_operator(
                codomain=precon_tg_obs_c
            ),
        ],
        [
            precon_response_space_c.zero_operator(
                codomain=precon_ice_obs_c
            ),
            precon_odt_space_c.zero_operator(
                codomain=precon_ice_obs_c
            ),
            precon_P_I_ice_c,
            precon_P_I_firn_c,
        ],
        [
            precon_grace_op_c,
            precon_odt_space_c.zero_operator(
                codomain=precon_grace_obs_c
            ),
            precon_ice_space_c.zero_operator(
                codomain=precon_grace_obs_c
            ),
            precon_firn_space_c.zero_operator(
                codomain=precon_grace_obs_c
            ),
        ],
    ]
)
precon_forward_op_c = (
    precon_P_left_c @ precon_F_middle_c @ precon_L_right_c
)

precon_inv_c = LinearBayesianInversion(
    LinearForwardProblem(
        precon_forward_op_c,
        data_error_measure=data_error_c,
    ),
    precon_model_prior_c,
)
print("Forming preconditioner C...")
precon_inv_normal_c = EigenSolver(parallel=False)(
    precon_inv_c.normal_operator
)

pbar_c = tqdm(desc="CG C")
posterior_c = LinearBayesianInversion(
    forward_problem_c, model_prior_c
).model_posterior_measure(
    data_c,
    CGMatrixSolver(
        callback=lambda xk: pbar_c.update(1), maxiter=500
    ),
    preconditioner=precon_inv_normal_c,
)
pbar_c.close()
print("Inversion C complete.\n")

# %%
# =============================================================================
# Regional 2D (Greenland, Antarctica) operators
# =============================================================================


def make_3comp_regional_ops(ice_obj, odt_obj):
    i_sp = ice_obj.ice_thickness.domain
    f_sp = ice_obj.firn_thickness.domain
    o_sp = odt_obj.height_measure.domain

    gl_i = averaging_operator(i_sp, [gl_wf])
    ant_i = averaging_operator(i_sp, [ant_wf])
    gl_f = averaging_operator(f_sp, [gl_wf])
    ant_f = averaging_operator(f_sp, [ant_wf])
    sc = gl_i.codomain

    C_tot = BlockLinearOperator(
        [
            [gl_i, gl_f, o_sp.zero_operator(codomain=sc)],
            [ant_i, ant_f, o_sp.zero_operator(codomain=sc)],
        ]
    )
    C_ice = BlockLinearOperator(
        [
            [
                gl_i,
                f_sp.zero_operator(codomain=sc),
                o_sp.zero_operator(codomain=sc),
            ],
            [
                ant_i,
                f_sp.zero_operator(codomain=sc),
                o_sp.zero_operator(codomain=sc),
            ],
        ]
    )
    C_firn = BlockLinearOperator(
        [
            [
                i_sp.zero_operator(codomain=sc),
                gl_f,
                o_sp.zero_operator(codomain=sc),
            ],
            [
                i_sp.zero_operator(codomain=sc),
                ant_f,
                o_sp.zero_operator(codomain=sc),
            ],
        ]
    )
    return C_tot, C_ice, C_firn


C_total_c, C_ice_c, C_firn_c = make_3comp_regional_ops(
    ice_c, odt_c
)

# %%
# =============================================================================
# Compute 2D regional posterior measures and true values
# =============================================================================


def regional_measure_and_true(posterior, model_true, C):
    """Map posterior and true model to (Greenland, Antarctica) 2D measure."""
    measure_2d = posterior.affine_mapping(operator=C)
    true_2d = C(model_true)
    return measure_2d, true_2d


post_total_c, true_total_c = regional_measure_and_true(
    posterior_c, model_true_c, C_total_c
)
post_ice_c, true_ice_c = regional_measure_and_true(
    posterior_c, model_true_c, C_ice_c
)
post_firn_c, true_firn_c = regional_measure_and_true(
    posterior_c, model_true_c, C_firn_c
)

# %%
# =============================================================================
# Bivariate overlay plots
# =============================================================================

col_c = "tab:orange"  # GRACE


def _tv(true_2d):
    """Extract true [gl, ant] values from a 2D field vector."""
    return [float(true_2d[0]), float(true_2d[1])]


# --- Total thickness (ice + firn GMSL contribution) ---

fig_total, _ = plot_bivariate_overlay(
    [
        {
            "measure": post_total_c,
            "label": "GRACE (ice+firn+ODT+GRACE)",
            "color": col_c,
            "true_val": _tv(true_total_c),
        },
    ],
    xlabel="Greenland Total Contribution (mm)",
    ylabel="Antarctica Total Contribution (mm)",
    title="Regional Comparison: Total (Ice + Firn) Thickness",
)
fig_total.savefig(
    "figs/regional_bivariate_total.pdf", dpi=600
)

# --- Ice-only ---

fig_ice, _ = plot_bivariate_overlay(
    [
        {
            "measure": post_ice_c,
            "label": "GRACE (ice+firn+ODT+GRACE)",
            "color": col_c,
            "true_val": _tv(true_ice_c),
        },
    ],
    xlabel="Greenland Ice Contribution (mm)",
    ylabel="Antarctica Ice Contribution (mm)",
    title="Regional Comparison: Ice Thickness",
)
fig_ice.savefig("figs/regional_bivariate_ice.pdf", dpi=600)

# --- Firn-only ---

fig_firn, _ = plot_bivariate_overlay(
    [
        {
            "measure": post_firn_c,
            "label": "GRACE (ice+firn+ODT+GRACE)",
            "color": col_c,
            "true_val": _tv(true_firn_c),
        },
    ],
    xlabel="Greenland Firn Contribution (mm)",
    ylabel="Antarctica Firn Contribution (mm)",
    title="Regional Comparison: Firn Thickness",
)
fig_firn.savefig(
    "figs/regional_bivariate_firn.pdf", dpi=600
)

plt.show()
print("Done. Figures saved to figs/")
