# %%
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    BlockLinearOperator,
    CGMatrixSolver,
    GaussianMeasure,
    HilbertSpaceDirectSum,
    LinearBayesianInversion,
    LinearForwardProblem,
    RowLinearOperator,
    plot_1d_distributions,
    ColumnLinearOperator,
    plot_corner_distributions,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
    plot,
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
)
from tqdm import tqdm
from xarray.plot.utils import _infer_xy_labels

from project.factored_forward_operator import (
    build_factored_forward_operator,
)
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

dir = "figs_fac_no_precon"
measure_error_std = 0.0005

ice = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.007,
    firn_gmsl_std=0.006,
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

ssh_altimetry = GridPoints.ocean_altimetry(fp, 3.0, 66.0)
ice_altimetry = GridPoints.ice(fp, 5.0)

lats, lons = read_gloss_tide_gauge_data()

tide_gauge_points = list(zip(lats, lons))
tide_sampling_op = tide_gauge_operator(
    ice.load_to_slc_operator.codomain, tide_gauge_points
)

# %%
# =============================================================================
# Full-resolution forward operator (factored block structure)
# =============================================================================

forward_operator = build_factored_forward_operator(
    fp,
    fp_op,
    ice,
    odt,
    ssh_altimetry,
    ice_altimetry,
    tide_gauge_points,
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
# Data error and forward problem
# =============================================================================

data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, measure_error_std
    )
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
# Bayesian inversion (no preconditioner)
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
            callback=progress_callback,
            rtol=5e-3,
            maxiter=1000,
        ),
    )
)
pbar.close()
print("")
print("Inversion complete.")

# %%
# =============================================================================
# Convergence plot
# =============================================================================

plt.figure(figsize=(8, 5))
plt.semilogy(
    residuals, marker="o", linestyle="-", markersize=3
)
plt.title("Convergence of CG Solver")
plt.xlabel("Iteration")
plt.ylabel("Norm of Solution ($||x_k||$)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()
plt.savefig(
    f"{dir}/joint_inversion_cg_convergence.png", dpi=600
)

# %%
# =============================================================================
# Extract components
# =============================================================================

model_posterior_expectation = (
    model_posterior_measure.expectation
)

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
# Plotting
# =============================================================================

# --- Ice thickness ---
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
    colorbar_label="Ice Thickness Change (mm)",
)
ax2.set_title(
    "b) Posterior Expectation (Inferred from Data)"
)
fig2.tight_layout()

# --- Firn thickness ---
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
    colorbar_label="Firn Thickness Change (mm)",
)
ax4.set_title(
    "d) Posterior Expectation (Inferred from Data)"
)
fig4.tight_layout()

# --- Ocean dynamics ---
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
    colorbar_label="ODT Height Change (mm)",
)
ax6.set_title(
    "f) Posterior Expectation (Inferred from Data)"
)
fig6.tight_layout()

# %%
# =============================================================================
# Sea-level change maps
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
    colorbar_label="Sea Level Change (mm)",
)
ax8.set_title(
    "h) Posterior Expectation (Inferred from Data)"
)
fig8.tight_layout()

# %%
# =============================================================================
# Total ice + firn load maps
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
    f"{dir}/joint_inversion_ice_thickness.png", dpi=600
)
fig2.savefig(
    f"{dir}/joint_inversion_ice_thickness_posterior.png",
    dpi=600,
)
fig3.savefig(
    f"{dir}/joint_inversion_firn_thickness.png", dpi=600
)
fig4.savefig(
    f"{dir}/joint_inversion_firn_thickness_posterior.png",
    dpi=600,
)
fig5.savefig(
    f"{dir}/joint_inversion_odt_height.png", dpi=600
)
fig6.savefig(
    f"{dir}/joint_inversion_odt_height_posterior.png",
    dpi=600,
)
fig7.savefig(f"{dir}/joint_inversion_slc.pdf", dpi=600)
fig8.savefig(
    f"{dir}/joint_inversion_slc_posterior.png", dpi=600
)
fig9.savefig(
    f"{dir}/joint_inversion_total_load.png", dpi=600
)
fig10.savefig(
    f"{dir}/joint_inversion_total_load_posterior.png",
    dpi=600,
)

# %%


# model_space = HilbertSpaceDirectSum(
#     [
#         ice.ice_thickness.domain,
#         ice.firn_thickness.domain,
#         odt.height_measure.domain,
#     ]
# )
# model_prior = GaussianMeasure.from_direct_sum(
#     [
#         ice.ice_thickness,
#         ice.firn_thickness,
#         odt.height_measure,
#     ]
# )

ice_gmsl_weighting_function = (
    -ice.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

firn_gmsl_weighting_function = (
    -ice.firn_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

ice_avg_op = averaging_operator(
    model_space.subspace(0), [ice_gmsl_weighting_function]
)
firn_avg_op = averaging_operator(
    model_space.subspace(1), [firn_gmsl_weighting_function]
)

ice_gmsl_true = ice_avg_op(ice_thickness_true)[0]
ice_gmsl_prior = model_prior.affine_mapping(
    operator=ice_avg_op
    @ model_prior.domain.subspace_projection(0)
)
ice_gmsl_posterior = model_posterior_measure.affine_mapping(
    operator=ice_avg_op
    @ model_posterior_measure.domain.subspace_projection(0)
)

firn_gmsl_true = firn_avg_op(firn_thickness_true)[0]
firn_gmsl_prior = model_prior.affine_mapping(
    operator=firn_avg_op
    @ model_prior.domain.subspace_projection(1)
)
firn_gmsl_posterior = model_posterior_measure.affine_mapping(
    operator=firn_avg_op
    @ model_posterior_measure.domain.subspace_projection(1)
)

print(ice_gmsl_true)

# %%

fig, (ax1, ax2) = plot_1d_distributions(
    ice_gmsl_posterior,
    prior_measures=ice_gmsl_prior,
    true_value=ice_gmsl_true,
    xlabel="Global Mean Sea Level Change from Ice Thickness (mm)",
    title="a) Global Mean Sea Level Change from Ice Thickness",
)

fig2, (ax3, ax4) = plot_1d_distributions(
    firn_gmsl_posterior,
    prior_measures=firn_gmsl_prior,
    true_value=firn_gmsl_true,
    xlabel="Global Mean Sea Level Change from Firn Thickness (mm)",
    title="b) Global Mean Sea Level Change from Firn Thickness",
)

fig1.savefig(
    f"{dir}/joint_inversion_ice_gmsl_distribution.png",
    dpi=600,
)
fig2.savefig(
    f"{dir}/joint_inversion_firn_gmsl_distribution.png",
    dpi=600,
)

# %%

prior_thickness_measures = GaussianMeasure.from_direct_sum(
    [ice.ice_thickness, ice.firn_thickness]
)

posterior_thickness_measure = GaussianMeasure.from_direct_sum(
    [
        model_posterior_measure.affine_mapping(
            operator=model_posterior_measure.domain.subspace_projection(
                0
            )
        ),
        model_posterior_measure.affine_mapping(
            operator=model_posterior_measure.domain.subspace_projection(
                1
            )
        ),
    ]
)

averaging_op = BlockLinearOperator(
    [
        [
            averaging_operator(
                thickness_measures.domain.subspace(0),
                [ice_gmsl_weighting_function],
            )
        ],
        [
            averaging_operator(
                thickness_measures.domain.subspace(1),
                [firn_gmsl_weighting_function],
            )
        ],
    ]
)

posterior = posterior_thickness_measure.affine_mapping(
    operator=averaging_op
)

prior = prior_thickness_measures.affine_mapping(
    operator=averaging_op
)

plot_corner_distributions(
    posterior,
    labels=[
        "Ice GMSL Change (mm)",
        "Firn GMSL Change (mm)",
    ],
)


# %%
# Looking at global averages, and regional averages for greenland, WAIS, EAIS
#
# For ice and firn thickness posteiror vs prior
#
# For corner plots of ice vs firn for each


print(firn_gmsl_true + ice_gmsl_true)
print(
    1000
    * (
        ice.ice_thickness_to_gmsl_operator(
            model_posterior_expectation[0]
        )
        + ice.firn_thickness_to_gmsl_operator(
            model_posterior_expectation[1]
        )
    )
)
