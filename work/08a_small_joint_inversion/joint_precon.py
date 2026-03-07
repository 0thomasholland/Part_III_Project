# %%
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
    RowLinearOperator,
    plot_corner_distributions,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    read_gloss_tide_gauge_data,
    tide_gauge_operator,
)
from tqdm import tqdm

from project import colors
from pygeoinf_extras import expectation, standard_dev
from pygeoinf_extras.operators import (
    point_averaging_operator,
)
from pyslfp_extras import plot
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange

# %%
# =============================================================================
# Full-resolution model setup
# =============================================================================

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

measure_error_std = 0.001
ice_measurement_angle = 5.0
ocean_measurement_angle = 10.0

ice = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.01,
    firn_gmsl_std=0.008,
    include_firn=True,
    firn_density=fp.ice_density * 0.4,
    ice_density=fp.ice_density,
    point_degree_spacing=ocean_measurement_angle,  # Ensure ocean points are included in the model space
)

# %%
# =============================================================================
# Full-resolution model space and prior
# =============================================================================

model_space = HilbertSpaceDirectSum(
    [
        ice.ice_thickness.domain,
        ice.firn_thickness.domain,
    ]
)
model_prior = GaussianMeasure.from_direct_sum(
    [
        ice.ice_thickness,
        ice.firn_thickness,
    ]
)

# %%
# =============================================================================
# Observation points
# =============================================================================

ssh_altimetry = GridPoints.ocean_altimetry(
    fp, ocean_measurement_angle, 66.0
)
ice_altimetry = GridPoints.ice(fp, ice_measurement_angle)

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
f31 = ice_altimetry.point_evaluation_operator(
    ice.ice_thickness.domain
)
f32 = ice_altimetry.point_evaluation_operator(
    ice.firn_thickness.domain
)

forward_operator = BlockLinearOperator(
    [[f11, f12], [f21, f22], [f31, f32]]
)

data_space = forward_operator.codomain

model_space_to_slc_operator = RowLinearOperator(
    [
        ice.load_to_slc_operator
        @ ice.ice_thickness_to_load_operator,
        ice.load_to_slc_operator
        @ ice.firn_thickness_to_load_operator,
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


# --- Preconditioner prior ---
precon_model_prior = GaussianMeasure.from_direct_sum(
    [
        precon_ice.ice_thickness,
        precon_ice.firn_thickness,
    ]
)

# %%
# =============================================================================
# Check ocean point consistency between full and preconditioner grids
# =============================================================================

precon_ssh_altimetry = GridPoints.ocean_altimetry(
    precon_fp, ocean_measurement_angle, 66.0
)
precon_ice_altimetry = GridPoints.ice(
    precon_fp, ice_measurement_angle
)

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
# Row 3: Ice altimetry observations (sampled at full-res ice points)
pf31 = precon_ice.ice_thickness.domain.point_evaluation_operator(
    ice_altimetry.coords
)
pf32 = precon_ice.firn_thickness.domain.point_evaluation_operator(
    ice_altimetry.coords
)

precon_forward_operator = BlockLinearOperator(
    [
        [pf11, pf12],
        [pf21, pf22],
        [pf31, pf32],
    ]
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

# %%
plt.figure(figsize=(8, 5))
plt.semilogy(
    residuals, marker="o", linestyle="-", markersize=3
)
plt.title("Convergence of CG Solver")
plt.xlabel("Iteration")
plt.ylabel("Norm of Solution ($||x_k||$)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig("figs/joint_precon_cg_convergence.png", dpi=600)

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

fig1.savefig("figs/joint_precon_ice_thickness.png", dpi=600)
fig2.savefig(
    "figs/joint_precon_ice_thickness_posterior.png", dpi=600
)
fig3.savefig(
    "figs/joint_precon_firn_thickness.png", dpi=600
)
fig4.savefig(
    "figs/joint_precon_firn_thickness_posterior.png",
    dpi=600,
)
fig7.savefig("figs/joint_precon_slc.png", dpi=600)
fig8.savefig("figs/joint_precon_slc_posterior.png", dpi=600)

# %%

total_thickness_true = (
    ice_thickness_true + firn_thickness_true
)
total_thickness_posterior_expectation = (
    ice_thickness_posterior_expectation
    + firn_thickness_posterior_expectation
)

total_load_true = ice.ice_thickness_to_load_operator(
    ice_thickness_true
) + ice.firn_thickness_to_load_operator(firn_thickness_true)
total_load_posterior_expectation = (
    ice.ice_thickness_to_load_operator(
        ice_thickness_posterior_expectation
    )
    + ice.firn_thickness_to_load_operator(
        firn_thickness_posterior_expectation
    )
)

max_total_thickness_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    total_thickness_true.data.flatten(),
                    total_thickness_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

max_total_load_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    total_load_true.data.flatten(),
                    total_load_posterior_expectation.data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)


fig9, ax9, im9 = plot(
    1000
    * total_thickness_true
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_total_thickness_change,
    vmax=max_total_thickness_change,
    colorbar_label="Total Thickness Change (mm)",
)
ax9.set_title("e) True Total Thickness Change")
fig9.tight_layout()

fig10, ax10, im10 = plot(
    1000
    * total_thickness_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_total_thickness_change,
    vmax=max_total_thickness_change,
    colorbar_label="Total Thickness Change (mm)",
)
ax10.set_title(
    "f) Posterior Expectation (Inferred from Data)"
)
fig10.tight_layout()

fig9.savefig(
    "figs/joint_precon_total_thickness.png", dpi=600
)
fig10.savefig(
    "figs/joint_precon_total_thickness_posterior.png",
    dpi=600,
)

fig11, ax11, im11 = plot(
    1000 * total_load_true * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_total_load_change,
    vmax=max_total_load_change,
    colorbar_label="Total Load Change (kg/m$^2$)",
)
ax11.set_title("i) True Total Load Change")
fig11.tight_layout()

fig12, ax12, im12 = plot(
    1000
    * total_load_posterior_expectation
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_total_load_change,
    vmax=max_total_load_change,
    colorbar_label="Total Load Change (kg/m$^2$)",
)
ax12.set_title(
    "j) Posterior Expectation (Inferred from Data)"
)
fig12.tight_layout()

fig11.savefig("figs/joint_precon_total_load.png", dpi=600)
fig12.savefig(
    "figs/joint_precon_total_load_posterior.png", dpi=600
)

# %%
# =============================================================================
# GMSL comparison: new method (joint inversion) vs old method (altimetry)
# =============================================================================

# --- GMSL operators for each component ---

# True GMSL from ice thickness (maps ice_thickness_space -> R^1)
ice_gmsl_op = ice.ice_thickness_to_gmsl_operator
# True GMSL from firn thickness (maps firn_thickness_space -> R^1)
firn_gmsl_op = ice.firn_thickness_to_gmsl_operator

# Total GMSL operator on the joint model space [ice, firn] -> R^1
total_gmsl_op = RowLinearOperator(
    [ice_gmsl_op, firn_gmsl_op]
)

# --- True GMSL values (mm) ---
total_gmsl_true = total_gmsl_op(model_true)[0] * 1000
ice_gmsl_true = ice_gmsl_op(ice_thickness_true)[0] * 1000
firn_gmsl_true = firn_gmsl_op(firn_thickness_true)[0] * 1000

# --- Posterior GMSL distributions (new method) ---
total_gmsl_posterior_measure = (
    model_posterior_measure.affine_mapping(
        operator=total_gmsl_op
    )
)

ice_gmsl_posterior_measure = model_posterior_measure.affine_mapping(
    operator=ice_gmsl_op
    @ model_posterior_measure.domain.subspace_projection(0)
)
firn_gmsl_posterior_measure = model_posterior_measure.affine_mapping(
    operator=firn_gmsl_op
    @ model_posterior_measure.domain.subspace_projection(1)
)

# %%
total_posterior_exp = (
    total_gmsl_posterior_measure.expectation[0] * 1000
)
# %%
total_gmsl_posterior_measure_std = (
    total_gmsl_posterior_measure.covariance.matrix(
        dense=True
    )
)
# %%
ice_posterior_exp = (
    ice_gmsl_posterior_measure.expectation[0] * 1000
)
ice_posterior_std = (
    standard_dev(ice_gmsl_posterior_measure) * 1000
)
firn_posterior_exp = (
    firn_gmsl_posterior_measure.expectation[0] * 1000
)
firn_posterior_std = (
    standard_dev(firn_gmsl_posterior_measure) * 1000
)
# %%
# --- Old method: altimetry point estimation of GMSL ---
# The old method averages SSH at ocean altimetry points.
# For total: use total load (ice + firn) through load_to_point_estimated_gmsl_operator
total_load_true = ice.ice_thickness_to_load_operator(
    ice_thickness_true
) + ice.firn_thickness_to_load_operator(firn_thickness_true)
total_alt_gmsl = (
    ice.load_to_point_estimated_gmsl_operator(
        total_load_true
    )[0]
    * 1000
)

# For ice only: use ice load only
ice_load_true = ice.ice_thickness_to_load_operator(
    ice_thickness_true
)
ice_alt_gmsl = (
    ice.load_to_point_estimated_gmsl_operator(
        ice_load_true
    )[0]
    * 1000
)

# For firn only: use firn load only
firn_load_true = ice.firn_thickness_to_load_operator(
    firn_thickness_true
)
firn_alt_gmsl = (
    ice.load_to_point_estimated_gmsl_operator(
        firn_load_true
    )[0]
    * 1000
)
# %%
# --- Old method error standard deviation ---
# The altimetry SSH points each have measurement error std = measure_error_std.
# Averaging n points with equal weights gives std = measure_error_std / sqrt(n).
ssh_point_space = (
    ice.load_to_ssh_point_estimations_operator.codomain
)
F_avg = point_averaging_operator(ssh_point_space)
ssh_error_measure = GaussianMeasure.from_standard_deviation(
    ssh_point_space, measure_error_std
)
averaged_ssh_error = ssh_error_measure.affine_mapping(
    operator=F_avg
)
alt_std = standard_dev(averaged_ssh_error) * 1000  # mm

# %%


# --- Helper: Gaussian PDF ---
def gaussian(x, mean, std_dev):
    return (
        1
        / (std_dev * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )


# --- Plot function for old vs new comparison ---
def plot_gmsl_comparison(
    true_val,
    posterior_exp,
    posterior_std,
    alt_est,
    alt_std_val,
    title,
    save_name,
):
    x_range = np.linspace(
        min(posterior_exp, alt_est, true_val)
        - 6 * max(posterior_std, alt_std_val),
        max(posterior_exp, alt_est, true_val)
        + 6 * max(posterior_std, alt_std_val),
        1000,
    )

    posterior_pdf = gaussian(
        x_range, posterior_exp, posterior_std
    )
    alt_pdf = gaussian(x_range, alt_est, alt_std_val)
    y_max = max(posterior_pdf.max(), alt_pdf.max())

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.axvline(
        true_val,
        color=colors.true,
        linestyle="--",
        label=f"True GMSL ({true_val:.2f} mm)",
    )

    ax.plot(
        x_range,
        posterior_pdf,
        label=(
            f"Joint Inversion (new)\n"
            f"(mean={posterior_exp:.2f} mm, std={posterior_std:.2e} mm)"
        ),
        color=colors.new_method,
    )

    ax.plot(
        x_range,
        alt_pdf,
        label=(
            f"Altimetry Point Estimation (old)\n"
            f"(mean={alt_est:.2f} mm, std={alt_std_val:.2e} mm)"
        ),
        color=colors.old_method,
    )

    ax.axvline(
        posterior_exp,
        color=colors.new_method,
        linestyle="--",
    )
    ax.axvline(
        alt_est,
        color=colors.old_method,
        linestyle="--",
    )

    ax.get_yaxis().set_visible(False)
    ax.set_ylim(-0.1, y_max * 1.1)
    ax.set_xlabel("GMSL Contribution (mm)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_name, dpi=600)
    return fig, ax


# --- 1) Total GMSL change ---
fig_gmsl_total, _ = plot_gmsl_comparison(
    total_gmsl_true,
    total_posterior_exp,
    total_gmsl_posterior_measure_std,
    total_alt_gmsl,
    alt_std,
    "Total GMSL: Joint Inversion vs Altimetry",
    "figs/joint_precon_gmsl_total.png",
)

# --- 2) Ice GMSL change ---
fig_gmsl_ice, _ = plot_gmsl_comparison(
    ice_gmsl_true,
    ice_posterior_exp,
    ice_posterior_std,
    ice_alt_gmsl,
    alt_std,
    "Ice GMSL: Joint Inversion vs Altimetry",
    "figs/joint_precon_gmsl_ice.png",
)

# --- 3) Firn GMSL change ---
fig_gmsl_firn, _ = plot_gmsl_comparison(
    firn_gmsl_true,
    firn_posterior_exp,
    firn_posterior_std,
    firn_alt_gmsl,
    alt_std,
    "Firn GMSL: Joint Inversion vs Altimetry",
    "figs/joint_precon_gmsl_firn.png",
)

# --- Print sigma diagnostics ---
total_post_sigma = (
    total_gmsl_true - total_posterior_exp
) / total_gmsl_posterior_measure_std
total_alt_sigma = (
    total_gmsl_true - total_alt_gmsl
) / alt_std
ice_post_sigma = (
    ice_gmsl_true - ice_posterior_exp
) / ice_posterior_std
ice_alt_sigma = (ice_gmsl_true - ice_alt_gmsl) / alt_std
firn_post_sigma = (
    firn_gmsl_true - firn_posterior_exp
) / firn_posterior_std
firn_alt_sigma = (firn_gmsl_true - firn_alt_gmsl) / alt_std

print(
    f"\nTotal GMSL:"
    f"\n  Posterior is {total_post_sigma:.2f}"
    f" sigma from true."
    f"\n  Altimetry is {total_alt_sigma:.2f}"
    f" sigma from true."
)
print(
    f"\nIce GMSL:"
    f"\n  Posterior is {ice_post_sigma:.2f}"
    f" sigma from true."
    f"\n  Altimetry is {ice_alt_sigma:.2f}"
    f" sigma from true."
)
print(
    f"\nFirn GMSL:"
    f"\n  Posterior is {firn_post_sigma:.2f}"
    f" sigma from true."
    f"\n  Altimetry is {firn_alt_sigma:.2f}"
    f" sigma from true."
)

# %%
# Trade off analysis

joint_gmsl_measure_op = BlockDiagonalLinearOperator(
    [ice_gmsl_op, firn_gmsl_op]
)

joint_gmsl_posterior_measure = (
    model_posterior_measure.affine_mapping(
        operator=joint_gmsl_measure_op
    )
)

fig, axes = plot_corner_distributions(
    joint_gmsl_posterior_measure,
    labels=["Ice GMSL (mm)", "Firn GMSL (mm)"],
    title="Joint Posterior Distribution of Ice vs Firn GMSL Contributions",
)
fig.savefig("figs/joint_precon_gmsl_tradeoff.png", dpi=600)
