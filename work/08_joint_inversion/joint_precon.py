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
    std=0.002,
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

ssh_altimetry = GridPoints.ocean_altimetry(fp, 5.0, 66.0)
ice_altimetry = GridPoints.ice(fp, 2.5)

lats, lons = read_gloss_tide_gauge_data()
tide_gauge_points = list(zip(lats, lons))
tide_sampling_op = tide_gauge_operator(
    ice.load_to_slc_operator.codomain, tide_gauge_points
)

# %%
# =============================================================================
# Full-resolution forward operator (block structure)
# =============================================================================

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

forward_operator = BlockLinearOperator(
    [[f11, f12, f13], [f21, f22, f23], [f31, f32, f33]]
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

model_space_to_ice_thickness_operator = RowLinearOperator(
    [
        ice.ice_thickness_to_load_operator,
        ice.firn_thickness_to_load_operator,
        odt.height_measure.domain.zero_operator(),
    ]
)

# %%
# =============================================================================
# Data error and forward problem
# =============================================================================

std_dev = 0.005
data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, std_dev
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
    std=0.002,
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
    precon_fp, 5.0, 66.0
)
precon_ice_altimetry = GridPoints.ice(precon_fp, 2.5)

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

# Row 1: SSH altimetry observations (sampled at full-res ocean points)
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

# Row 2: Tide gauge observations
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

# Row 3: Ice altimetry observations (sampled at full-res ice points)
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

precon_forward_operator = BlockLinearOperator(
    [
        [pf11, pf12, pf13],
        [pf21, pf22, pf23],
        [pf31, pf32, pf33],
    ]
)

# %%
# =============================================================================
# Form the preconditioner inverse via eigen-decomposition
# =============================================================================

precon_std_dev = 0.01

precon_data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, std_dev
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
            callback=progress_callback, maxiter=200
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
plt.show()

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
    colorbar_label="Ocean Height Change (mm)",
)
ax5.set_title("e) True Ocean Height Change")

fig6, ax6, im6 = plot(
    1000
    * odt_height_posterior_expectation
    * fp.length_scale
    * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_odt_height_change,
    vmax=max_abs_odt_height_change,
    colorbar_label="Ocean Height Change (mm)",
)
ax6.set_title(
    "f) Posterior Expectation (Inferred from Data)"
)

plt.show()
