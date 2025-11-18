# %%
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    HilbertSpace,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    RowLinearOperator,
)
from pygeoinf.linear_solvers import ScipyIterativeSolver
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

# %%
# CONFIGURATION VARIABLES

lmax = 128

# Model space parameters
model_sobolev_order = 2
model_length_scale_factor = 0.1  # fraction of mean sea floor radius

# Ice thickness change prior parameters
ice_length_scale_factor = 0.1  # fraction of mean sea floor radius
ice_gmsl_target_std_m = 0.01  # metres
ice_net_thickness_change_m = 0.0  # metres

# Ocean dynamic topography prior parameters
meso_scale_lengthscale = 250e3
meso_scale_std = 0.001
sub_meso_scale_lengthscale = 2e3
sub_meso_scale_std = 0.001

# Altimetry observation parameters
along_track_error = 0.002  # metres
passes = 3  # typically ~ 3 passes every 30 days


altimetry_latitude_max = 66  # degrees
altimetry_latitude_min = -66  # degrees
altimetry_noise_sobolev_order = 1.5
altimetry_noise_length_scale_factor = 0.001  # fraction of mean sea floor radius
altimetry_noise_std_m = along_track_error / np.sqrt(passes)  # metres

# Solver parameters
solver_rtol = 1e-4
solver_maxiter = 5000

# %%
# FINGERPRINT MODEL INITIALISATION

fp = FingerPrint(
    lmax=lmax,
    earth_model_parameters=FingerPrint.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

# %%
# UNIT CONVERSIONS (physical to non-dimensional)

model_length_scale = model_length_scale_factor * fp.mean_sea_floor_radius
ice_length_scale = ice_length_scale_factor * fp.mean_sea_floor_radius
ice_gmsl_target_std = ice_gmsl_target_std_m / fp.length_scale
ice_net_thickness_change = ice_net_thickness_change_m / fp.length_scale
meso_scale_lengthscale = meso_scale_lengthscale / fp.length_scale
sub_meso_scale_lengthscale = sub_meso_scale_lengthscale / fp.length_scale
meso_scale_std = meso_scale_std / fp.length_scale
sub_meso_scale_std = sub_meso_scale_std / fp.length_scale
altimetry_noise_length_scale = (
    altimetry_noise_length_scale_factor * fp.mean_sea_floor_radius
)
altimetry_noise_std = altimetry_noise_std_m / fp.length_scale

# %%
# HILBERT SPACES

model_space = Sobolev(
    fp.lmax,
    model_sobolev_order,
    model_length_scale,
    radius=fp.mean_sea_floor_radius,
)

fingerprint_op: LinearOperator = fp.as_sobolev_linear_operator(
    model_sobolev_order,
    model_length_scale,
)
load_space: HilbertSpace = fingerprint_op.domain
response_space: HilbertSpace = fingerprint_op.codomain

sea_surface_height_op: LinearOperator = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space: HilbertSpace = sea_surface_height_op.codomain

# %%
# LINEAR OPERATORS

# Load conversion operators
water_to_load_op: LinearOperator = sea_level_change_to_load_operator(
    fp,
    load_space,
)
ice_to_load_op: LinearOperator = ice_thickness_change_to_load_operator(fp, load_space)

# Altimetry spatial mask operator
altimetry_mask_op: LinearOperator = spatial_mutliplication_operator(
    fp.altimetry_projection(
        latitude_max=altimetry_latitude_max,
        latitude_min=altimetry_latitude_min,
        value=0,
    )
    * fp.ocean_function,
    measurement_space,
)

# Error/noise model operator
# Maps [odt_gravitational, odt_direct, instrument_noise] → observed error
altimetry_noise_op = altimetry_mask_op @ RowLinearOperator(
    [
        sea_surface_height_op @ fingerprint_op @ water_to_load_op,
        measurement_space.identity_operator(),
        measurement_space.identity_operator(),
    ],
)

# Signal forward operator: ice thickness change → altimetry observations
ice_to_altimetry_op = (
    altimetry_mask_op
    @ sea_surface_height_op
    @ fingerprint_op
    @ ice_to_load_op
    @ ice_projection_operator(fp, load_space)
)

# %%
# GAUSSIAN MEASURES (PRIORS AND ERROR MODELS)

# Ice thickness change prior
ice_thickness_change_measure, _ = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_op,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=ice_net_thickness_change,
)

# Ocean dynamic topography prior
meso, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_op,
    length_scale=meso_scale_lengthscale,
    standard_deviation=meso_scale_std,
)
sub_meso, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_op,
    length_scale=sub_meso_scale_lengthscale,
    standard_deviation=sub_meso_scale_std,
)
odt_change_measure = meso + sub_meso

# Measurement noise
measurement_noise_measure = (
    measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        altimetry_noise_sobolev_order,
        altimetry_noise_length_scale,
        altimetry_noise_std,
    )
)

# Combined data error measure
error_input_measure = GaussianMeasure.from_direct_sum(
    [
        odt_change_measure,
        odt_change_measure,
        measurement_noise_measure,
    ],
)
data_error_measure = error_input_measure.affine_mapping(
    operator=altimetry_noise_op,
)

# Visualise a sample from the data error measure
fig, ax, im = plot(odt_change_measure.sample() * fp.length_scale)
fig.colorbar(
    im,
    ax=ax,
    label="ODT change measure sample (m)",
    orientation="horizontal",
)

# %%
# FORWARD PROBLEM AND SYNTHETIC DATA

forward_problem = LinearForwardProblem(
    ice_to_altimetry_op,
    data_error_measure=data_error_measure,
)

model_true, data = forward_problem.synthetic_model_and_data(
    ice_thickness_change_measure,
)

# %%
# BAYESIAN INVERSION

inversion = LinearBayesianInversion(
    forward_problem,
    ice_thickness_change_measure,
)


# %%


class ConvergenceMonitor:
    def __init__(self):
        self.iteration = 0
        self.x_norms = []
        self.x_changes = []
        self.prev_x = None

    def __call__(self, xk):
        self.iteration += 1
        x_norm = np.linalg.norm(xk)
        self.x_norms.append(x_norm)

        if self.prev_x is not None:
            change = np.linalg.norm(xk - self.prev_x)
            relative_change = change / x_norm if x_norm > 0 else change
            self.x_changes.append(relative_change)

            if self.iteration % 5 == 0:
                print(
                    f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}, "
                    f"relative change = {relative_change:.6e}",
                )
        elif self.iteration % 5 == 0:
            print(f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}")

        self.prev_x = xk.copy()


monitor = ConvergenceMonitor()
solver = CGMatrixSolver(
    callback=monitor,
    rtol=solver_rtol,
    maxiter=solver_maxiter,
)


model_posterior_measure = inversion.model_posterior_measure(
    data,
    solver,
)

print(f"Total iterations: {monitor.iteration}")
print(f"Number of fingerprint problem solutions: {fp.solver_counter}")


fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(range(1, monitor.iteration + 1), monitor.x_norms, marker="o")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Norm of solution ||x||")
ax1.set_title("Convergence of Solution Norm")

# %%
# RESULTS

model_posterior_expectation = model_posterior_measure.expectation


# %%
# VISUALISATION

# Ice thickness change recovery
# ------------------------------------------

# True ice thickness change
fig1, ax1, im1 = plot(
    model_true * fp.length_scale * fp.ice_projection(),
    symmetric=True,
)
fig1.colorbar(
    im1,
    ax=ax1,
    label="True ice thickness change (m)",
    orientation="horizontal",
)

# Posterior expectation (recovered)
fig2, ax2, im2 = plot(
    model_posterior_expectation * fp.length_scale * fp.ice_projection(),
    symmetric=True,
)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Posterior ice thickness change (m)",
    orientation="horizontal",
)
# %%
# Difference (error in recovery)
fig3, ax3, im3 = plot(
    (model_posterior_expectation - model_true) * fp.length_scale * fp.ice_projection(),
)
fig3.colorbar(
    im3,
    ax=ax3,
    label="Difference in ice thickness change (m)",
    orientation="horizontal",
)


# 2. Sea level change recovery
# --------------------------------------
# Map ice thickness to sea surface height
sea_level_true = (
    sea_surface_height_op
    @ fingerprint_op
    @ ice_to_load_op
    @ ice_projection_operator(fp, load_space)
)(model_true)

sea_level_posterior = (
    sea_surface_height_op
    @ fingerprint_op
    @ ice_to_load_op
    @ ice_projection_operator(fp, load_space)
)(model_posterior_expectation)

# Mask to ocean only
ocean_mask = fp.ocean_projection()

# True sea level fingerprint
fig4, ax4, im4 = plot(
    sea_level_true * ocean_mask * fp.length_scale * 1000,
)
fig4.colorbar(
    im4,
    ax=ax4,
    label="True sea level change (mm)",
    orientation="horizontal",
)

# Predicted sea level fingerprint
fig5, ax5, im5 = plot(
    sea_level_posterior * ocean_mask * fp.length_scale * 1000,
)
fig5.colorbar(
    im5,
    ax=ax5,
    label="Posterior sea level change (mm)",
    orientation="horizontal",
)


# 3. OBSERVED DATA (what the altimeter actually "sees")
# -----------------------------------------------------
# The data includes the altimetry mask and noise
fig6, ax6, im6 = plot(
    data
    * fp.length_scale
    * 1000
    * fp.ocean_projection()
    * fp.altimetry_projection(latitude_min=-66, latitude_max=66),
)


fig6.colorbar(
    im6,
    ax=ax6,
    label="Observed sea surface height change (mm)",
    orientation="horizontal",
)


# 4. DATA RESIDUAL
# -----------------
# How well does the posterior prediction fit the data?
predicted_data = ice_to_altimetry_op(model_posterior_expectation)
data_residual = data - predicted_data

fig7, ax7, im7 = plot(
    data_residual
    * fp.length_scale
    * 1000
    * fp.ocean_projection()
    * fp.altimetry_projection(latitude_min=-66, latitude_max=66),
)
fig7.colorbar(
    im7,
    ax=ax7,
    label="Residual (mm)",
    orientation="horizontal",
)

# %%

plt.show()


# %%

# Your existing setup
lat_greenland, lon_greenland = 65.0, -45.0
radius_deg = 5.0
thickness_change_meters = -100.0  # Negative = ice LOSS

ice_thickness_change = fp.disk_load(
    delta=radius_deg,
    latitude=lat_greenland,
    longitude=lon_greenland,
    amplitude=thickness_change_meters,
)

# Plot the input
# fig1, ax1, im1 = plot(ice_thickness_change, symmetric=True)
# fig1.colorbar(
#     im1,
#     ax=ax1,
#     label="Ice thickness change (m)",
#     orientation="horizontal",
# )

# ax1.set_title("Input: -100m ice loss in Greenland")

# Convert to Sobolev space element
ice_sh_coeffs = fp.expand_field(ice_thickness_change)
coeffs_flat = ice_sh_coeffs.coeffs[0].flatten()  # Use real coefficients
test_in_load_space = load_space.from_components(coeffs_flat)

# Push through your forward operator
test_ssh = ice_to_altimetry_op(test_in_load_space)

# Convert output back to grid
test_ssh_components = measurement_space.to_components(test_ssh)
print(f"Output components shape: {test_ssh_components.shape}")
print(f"measurement_space dimension: {measurement_space.dim}")

# Reshape back to SHCoeffs format
test_ssh_coeffs_array = test_ssh_components.reshape(
    (lmax + 1, lmax + 1),
)

# Create an SHCoeffs object to use expand_coefficient
# We need to put it back in the (2, lmax+1, lmax+1) format
full_coeffs = np.zeros((2, lmax + 1, lmax + 1))
full_coeffs[0] = test_ssh_coeffs_array
test_ssh_coeffs = fp.zero_coefficients()
test_ssh_coeffs.coeffs = full_coeffs

# Convert back to grid
test_ssh_grid = fp.expand_coefficient(test_ssh_coeffs)

# Plot the output
fig2, ax2, im2 = plot(test_ssh_grid * fp.length_scale, symmetric=True)
fig2.colorbar(
    im2,
    ax=ax2,
    label="Sea surface height change (m)",
    orientation="horizontal",
)
ax2.set_title("Output: SSH from -100m ice loss in Greenland")
# %%

a, b, c, d = fp(
    direct_load=fp.direct_load_from_ice_thickness_change(
        ice_thickness_change,
    ),
)
ssh_change_true = fp.sea_surface_height_change(
    a,
    b,
    d,
)
fig3, ax3, im3 = plot(
    ssh_change_true * fp.length_scale * fp.ocean_projection(),
    symmetric=True,
)
fig3.colorbar(
    im3,
    ax=ax3,
    label="Sea surface height change (m)",
    orientation="horizontal",
)
# Check the pattern:
# Ice LOSS should cause:
# - Negative SSH near Greenland (less gravity, water flows away)
# - Positive SSH in far field (global sea level rise)
print(
    f"\nMax SSH: {np.nanmax(test_ssh_grid.data * fp.length_scale):.4f} m",
)
print(
    f"Min SSH: {np.nanmin(test_ssh_grid.data * fp.length_scale):.4f} m",
)

# %%
