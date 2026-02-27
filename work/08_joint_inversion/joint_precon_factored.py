# %%
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
    plot,
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
)
from tqdm import tqdm

from project.factored_forward_operator import (
    build_factored_forward_operator,
    build_factored_forward_operator_precon,
)
from project.operators import (
    ice_thickness_to_estimated_gmsl_operator,
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

dir = "figs_fac"
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
            # Remove the second point (j) if it's too close to the first point (i)
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
# Full-resolution forward operator (block structure)
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
# Preconditioner setup (lower-resolution joint model)
# =============================================================================

lmax_precon = 32

precon_fp = FingerPrint(lmax=lmax_precon)
precon_fp.set_state_from_ice_ng(
    version=IceModel.ICE7G, date=0.0
)
precon_fp_op = precon_fp.as_sobolev_linear_operator(
    2, precon_fp.mean_sea_floor_radius * 0.1
)

# --- Preconditioner ice model ---
precon_ice = IceSheetChange.global_ice(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    length_scale=0.1 * precon_fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.003,
    include_firn=True,
)

# --- Preconditioner ocean dynamics ---
precon_odt = OceanDynamics(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    std=0.004,
    length_scale=10000.0,
    pattern=OD_pattern,
)


# --- Preconditioner prior ---
precon_model_prior = GaussianMeasure.from_direct_sum(
    [
        precon_ice.ice_thickness,
        precon_ice.firn_thickness,
        precon_odt.height_measure,
    ]
)

# %%
# =============================================================================
# Check ocean point consistency between full and preconditioner grids
# =============================================================================

precon_ssh_altimetry = GridPoints.ocean_altimetry(
    precon_fp, 10.0, 66.0
)
precon_ice_altimetry = GridPoints.ice(precon_fp, 10.0)

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
    f"Full-res SSH points NOT in preconditioner ocean: {len(ssh_points_not_in_precon)}"
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
    f"Full-res ice points NOT in preconditioner: {len(ice_points_not_in_precon)}"
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
# Preconditioner forward operator (block structure matching full problem,
# but using low-resolution operators and sampling at full-res points)
# =============================================================================

# Tide gauge operator for the preconditioner
precon_tide_sampling_op = tide_gauge_operator(
    precon_ice.load_to_slc_operator.codomain,
    tide_gauge_points,
)

precon_forward_operator = (
    build_factored_forward_operator_precon(
        precon_fp,
        precon_fp_op,
        precon_ice,
        precon_odt,
        ssh_altimetry.coords,
        ice_altimetry.coords,
        tide_gauge_points,
    )
)


# %%
# =============================================================================
# Form the preconditioner inverse via eigen-decomposition
# =============================================================================


precon_data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, measure_error_std
    )
)

precon_forward_problem = LinearForwardProblem(
    precon_forward_operator,
    data_error_measure=precon_data_error_measure,
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
            callback=progress_callback, maxiter=300
        ),
        preconditioner=precon_inverse_normal_operator,
    )
)
pbar.close()
print("")
print("Inversion complete.")

plt.figure(figsize=(8, 5))
plt.semilogy(
    residuals, marker="o", linestyle="-", markersize=3
)
plt.title("Convergence of CG Solver")
plt.xlabel("Iteration")
plt.ylabel("Norm of Solution ($||x_k||$)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig(
    f"{dir}/joint_precon_cg_convergence.png", dpi=600
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
# operator that maps from the model space to slc

slc_true = model_space_to_slc_operator(model_true)[0]
slc_posterior_expectation = model_space_to_slc_operator(
    model_posterior_expectation
)[0]

# plot
#
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

# map from model space to total ice and firn load

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

fig1.savefig(
    f"{dir}/joint_precon_ice_thickness.png", dpi=600
)
fig2.savefig(
    f"{dir}/joint_precon_ice_thickness_posterior.png",
    dpi=600,
)
fig3.savefig(
    f"{dir}/joint_precon_firn_thickness.png", dpi=600
)
fig4.savefig(
    f"{dir}/joint_precon_firn_thickness_posterior.png",
    dpi=600,
)
fig5.savefig(f"{dir}/joint_precon_odt_height.png", dpi=600)
fig6.savefig(
    f"{dir}/joint_precon_odt_height_posterior.png", dpi=600
)
fig7.savefig(f"{dir}/joint_precon_slc.png", dpi=600)
fig8.savefig(
    f"{dir}/joint_precon_slc_posterior.png", dpi=600
)
fig9.savefig(f"{dir}/joint_precon_total_load.png", dpi=600)
fig10.savefig(
    f"{dir}/joint_precon_total_load_posterior.png", dpi=600
)
