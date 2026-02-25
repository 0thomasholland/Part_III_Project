# %%
import cartopy.crs as ccrs
import numpy as np
from pygeoinf import (
    CGMatrixSolver,
    EigenSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    plot_1d_distributions,
    plot_corner_distributions,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
    plot,
)
from pyslfp_extras.helpers import (
    get_ocean_point_coordinates,
)
from pyslfp_extras.operators import (
    ocean_point_evaluation_operator,
)
from tqdm import tqdm

from project import (
    error_plot,
    ice_thickness_to_slc_operator,
)
from project.operators import (
    ice_thickness_to_point_estimated_gmsl_operator,
    ice_thickness_to_ssh_operator,
    ice_thickness_to_ssh_point_estimations_operator,
)
from pyslfp_extras.gmsl import (
    altimetry_gmsl,
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)
from pyslfp_extras.ocean_dynamics import (
    OceanDynamics,
)

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%

# generate prior dataset

altimetry_degree_density = 5.0

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.01,
    gmsl_target_mean=0.08,
)
ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness_measure
)

ice_thickness_to_ssh_point_estimations_op: LinearOperator = ice_thickness_to_ssh_point_estimations_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
    altimetry_latitude_range=66.0,
    point_degree_spacing=altimetry_degree_density,
    parallel_workers=-1,
)

points: tuple[list[float], list[float]] = (
    get_ocean_point_coordinates(
        finger_print=fp,
        point_degree_spacing=altimetry_degree_density,
        altimetry_latitude_range=66.0,
        parallel_workers=-1,
    )
)


# %%

data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)

# %%

odt_error = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.003,
    pattern=OceanDynamics.SyntheticPattern(
        point_multiplier=20,
    ),
)
error_field_measure: GaussianMeasure = (
    odt_error.load_measure
)

error_sampling_points = error_field_measure.affine_mapping(
    operator=ocean_point_evaluation_operator(
        finger_print=fp,
        measurement_space=error_field_measure.domain,
        point_degree_spacing=altimetry_degree_density,
        altimetry_latitude_range=66.0,
    )
)


# %%
sample_ice_thickness = ice_thickness_measure.sample()
sample_error_field = error_field_measure.sample()
sample_ssh = ice_thickness_to_ssh_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
    altimetry_latitude_range=66.0,
)(sample_ice_thickness)
sample_combined = sample_ssh + sample_error_field

plot(sample_ice_thickness, symmetric=True)
plot(sample_ssh, symmetric=True)
plot(sample_error_field, symmetric=True)
plot(sample_combined, symmetric=True)


# %%

forward_problem = LinearForwardProblem(
    ice_thickness_to_ssh_point_estimations_op,
    data_error_measure=error_sampling_points,
)

model_true, data = forward_problem.synthetic_model_and_data(
    ice_thickness_measure
)

# %%
# Preconditioner — low-resolution version of the full problem

lmax_precon = 32

precon_fp = FingerPrint(lmax=lmax_precon)
precon_fp.set_state_from_ice_ng(
    version=IceModel.ICE7G, date=0.0
)

precon_fp_op = precon_fp.as_sobolev_linear_operator(
    2, precon_fp.mean_sea_floor_radius * 0.1
)

precon_ice_change = IceSheetChange.global_ice(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    length_scale=0.1 * precon_fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.01,
    gmsl_target_mean=0.08,
)
precon_ice_thickness_measure: GaussianMeasure = (
    precon_ice_change.ice_thickness_measure
)

# Check that the full-resolution ocean points are also ocean points
# on the lower-resolution preconditioner grid.
precon_ocean_points = get_ocean_point_coordinates(
    finger_print=precon_fp,
    point_degree_spacing=altimetry_degree_density,
    altimetry_latitude_range=66.0,
    parallel_workers=-1,
)
precon_ocean_set = set(
    zip(precon_ocean_points[0], precon_ocean_points[1])
)
full_ocean_set = set(zip(points[0], points[1]))
points_not_in_precon_ocean = (
    full_ocean_set - precon_ocean_set
)
print(
    f"Full-resolution ocean points: {len(full_ocean_set)}"
)
print(
    f"Preconditioner ocean points: {len(precon_ocean_set)}"
)
print(
    f"Full-res points NOT in preconditioner ocean: {len(points_not_in_precon_ocean)}"
)
if points_not_in_precon_ocean:
    print(
        "WARNING: Some ocean points from the full grid are not ocean on the preconditioner grid:"
    )
    for lat, lon in sorted(points_not_in_precon_ocean):
        print(f"  lat={lat:.1f}, lon={lon:.1f}")

# Build the precon forward operator manually so it maps to the same
# data space as the full problem (same ocean points from the full fp).
precon_ssh_op = ice_thickness_to_ssh_operator(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    altimetry_latitude_range=66.0,
)
precon_point_eval_op = (
    precon_ssh_op.codomain.point_evaluation_operator(
        list(zip(points[0], points[1]))
    )
)
precon_forward_op: LinearOperator = (
    precon_point_eval_op @ precon_ssh_op
)

# Build the low-resolution realistic error measure, using a point
# evaluation operator from the same SSH codomain so the error measure
# lands in the same data space as precon_forward_op.
precon_odt_error = OceanDynamics(
    finger_print=precon_fp,
    finger_print_operator=precon_fp_op,
    std=0.003,
    pattern=OceanDynamics.SyntheticPattern(
        point_multiplier=20,
    ),
)
precon_error_field_measure: GaussianMeasure = (
    precon_odt_error.load_measure
)

precon_error_point_eval_op = precon_error_field_measure.domain.point_evaluation_operator(
    list(zip(points[0], points[1]))
)

precon_error_sampling_points = (
    precon_error_field_measure.affine_mapping(
        operator=precon_error_point_eval_op,
    )
)

precon_forward_problem = LinearForwardProblem(
    precon_forward_op,
    data_error_measure=precon_error_sampling_points,
)

# Set up the inversion for the preconditioning system
precon_bayesian_inversion = LinearBayesianInversion(
    precon_forward_problem, precon_ice_thickness_measure
)

# Get the normal operator for the preconditioning system.
precon_normal_operator = (
    precon_bayesian_inversion.normal_operator
)

# Form its inverse using Eigen-decomposition.
print("Forming the preconditioner...")
solver = EigenSolver(parallel=True, n_jobs=12)
precon_inverse_normal_operator = solver(
    precon_normal_operator
)

# %%

# Set up the Bayesian inversion method
bayesian_inversion = LinearBayesianInversion(
    forward_problem, ice_thickness_measure
)

# Solve for the posterior distribution
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
        CGMatrixSolver(callback=progress_callback),
        preconditioner=precon_inverse_normal_operator,
    )
)
pbar.close()
print("")
print("Inversion complete.")

# Get the posterior expectation
model_posterior_expectation = (
    model_posterior_measure.expectation
)

print(
    f"Number of solutions of the fingerprint problem = {fp.solver_counter}"
)

# %%

max_abs_ice_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (model_true).data.flatten(),
                    (
                        model_posterior_expectation
                    ).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

# --- Plot 1: The "Ground Truth" Model ---
fig1, ax1, im1 = plot(
    1000
    * model_true
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.set_title("a) True Ice Thickness Change")

# --- Plot 2: The Posterior Expectation (Our Best Estimate) ---
fig2, ax2, im2 = plot(
    1000
    * model_posterior_expectation
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

# %%

ice_thickness_to_slc_op = ice_thickness_to_slc_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
)

sea_level_posterior = ice_thickness_to_slc_op(
    model_posterior_expectation
)

sea_level_true = ice_thickness_to_slc_op(model_true)

ocean_mask = fp.ocean_projection()
max_abs_sl_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (
                        sea_level_true * ocean_mask
                    ).data.flatten(),
                    (
                        sea_level_posterior * ocean_mask
                    ).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)


# --- Plot 3: The "True" Sea-Level Field ---
fig3, ax3, im13 = plot(
    1000 * sea_level_true * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax3.set_title("a) True Sea-Level Fingerprint")

# --- Plot 4: The Sea-Level Field Predicted by the Inversion ---
fig4, ax4, im4 = plot(
    1000
    * sea_level_posterior
    * fp.ocean_projection()
    * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax4.set_title("b) Predicted Sea-Level Fingerprint")
ax4.plot(
    points[1],  # longitudes
    points[0],  # latitudes
    "kx",
    label="Altimetry Point Estimations",
    transform=ccrs.PlateCarree(),
)
# %%


model_space = ice_thickness_measure.domain

# Set the weighting function for GMSL estimates  - Note that length scale factor to dimensionalise the result into mm
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

# Form the mapping to GSML.
B = averaging_operator(
    model_space, [GMSL_weighting_function]
)

# Get the true GMSL
GMSL_true = B(model_true)

# Push forward the posterior to the GMSL space.
GMSL_prior_measure = ice_thickness_measure.affine_mapping(
    operator=B
)
GMSL_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=B)
)

# Plot the PDFs
fig, ax = plot_1d_distributions(
    GMSL_posterior_measure,
    # prior_measures=GMSL_prior_measure,
    true_value=GMSL_true[0],
    xlabel="GMSL Change (mm)",
    title="Global Mean Sea Level Change Inference from GRACE Data",
)

# %%

GLI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
WAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.west_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
EAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.east_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

C = averaging_operator(
    model_space,
    [
        GLI_weighting_function,
        WAI_weighting_function,
        EAI_weighting_function,
    ],
)

property_true = C(model_true)
property_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=C)
)

# Visualise the distribution using a corner plot
plot_corner_distributions(
    property_posterior_measure,
    true_values=property_true,
    labels=[
        "Greenland Contribution (mm)",
        "West Antarctica Contribution (mm)",
        "East Antarctica Contribution (mm)",
    ],
    title="Joint Posterior Distributions of GMSL Contributions from Major Ice Sheets",
)
