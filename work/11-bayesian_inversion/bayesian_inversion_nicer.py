# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
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

lmax = 64

# Model space parameters
model_sobolev_order = 2
model_length_scale = 0.1  # fraction of mean sea floor radius

# Ice thickness change prior parameters
ice_length_scale = 0.1  # fraction of mean sea floor radius
ice_gmsl_target_std = 0.1  # metres
ice_net_thickness_change = -1.0  # metres

# Ocean dynamic topography prior parameters
odt_lengthscale = 250e3
odt_std = 0.001

# Altimetry observation parameters
along_track_error = 0.002  # metres
passes = 3  # typically ~ 3 passes every 30 days


altimetry_latitude_max = 66  # degrees
altimetry_latitude_min = -66  # degrees

altimetry_noise_sobolev_order = 1.5
altimetry_noise_length_scale = 2e4  # fraction of mean sea floor radius
altimetry_noise_std = along_track_error / np.sqrt(passes)  # metres

# Solver parameters
solver_rtol = 1e-8
solver_maxiter = 100

# %%
# FINGERPRINT MODEL INITIALISATION

fp = FingerPrint(
    lmax=lmax,
    # earth_model_parameters=FingerPrint.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

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
    # [
    #     sea_surface_height_op @ fingerprint_op @ water_to_load_op,
    #     measurement_space.identity_operator(),
    #     measurement_space.identity_operator(),
    # ],
    [
        measurement_space.zero_operator(),
        measurement_space.zero_operator(),
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
    length_scale=ice_length_scale * fp.mean_sea_floor_radius,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=ice_net_thickness_change,
)

# Ocean dynamic topography prior
odt_change_measure, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_op,
    length_scale=odt_lengthscale,
    standard_deviation=odt_std,
)

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
fig, ax, im = plot(
    data_error_measure.sample()
    * fp.length_scale
    * fp.ocean_projection()
    * fp.altimetry_projection(latitude_max=66, latitude_min=-66)
)
fig.colorbar(
    im,
    ax=ax,
    label="Noise",
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
# Combined subplot grid: Ice and Sea Level (True, Posterior, Residual)
# ---------------------------------------------------------------------
# %%
# VISUALISATION
# Combined subplot grid: Ice and Sea Level (True, Posterior, Residual)
# ---------------------------------------------------------------------

# Calculate all quantities first
# Ice thickness change
ice_true = model_true * fp.length_scale * fp.ice_projection()
ice_posterior = model_posterior_expectation * fp.length_scale * fp.ice_projection()
ice_residual = (
    (model_posterior_expectation - model_true) * fp.length_scale * fp.ice_projection()
)

# Sea level change
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
sea_level_true_masked = sea_level_true * ocean_mask * fp.length_scale
sea_level_posterior_masked = sea_level_posterior * ocean_mask * fp.length_scale
sea_level_residual = (
    (sea_level_true - sea_level_posterior) * fp.length_scale * ocean_mask
)


# Create subplot grid with Robinson projection
projection = ccrs.Robinson()
fig, axes = plt.subplots(3, 2, figsize=(16, 20), subplot_kw={"projection": projection})
axes = axes.flatten()

# Data and parameters for each subplot
data_list = [
    ice_true,
    sea_level_true_masked,
    ice_posterior,
    sea_level_posterior_masked,
    ice_residual,
    sea_level_residual,
]

titles = [
    "True Ice Thickness Change",
    "True Sea Level Change",
    "Posterior Ice Thickness Change",
    "Posterior Sea Level Change",
    "Residual Ice Thickness Change",
    "Residual Sea Level Change",
]

labels = [
    "True ice thickness change (m)",
    "True sea level change (mm)",
    "Posterior ice thickness change (m)",
    "Posterior sea level change (mm)",
    "Residual ice thickness change (m)",
    "Residual sea level (mm)",
]

symmetric_flags = [False, False, False, False, False, False]

# Plot each subplot
for idx, (data, title, label, symmetric) in enumerate(
    zip(data_list, titles, labels, symmetric_flags)
):
    ax = axes[idx]

    # Get lons and lats
    lons = data.lons()
    lats = data.lats()

    # Set up plot parameters
    cmap = "RdBu"
    kwargs = {}

    if symmetric:
        data_max = 1.2 * np.nanmax(np.abs(data.data))
        kwargs["vmin"] = -data_max
        kwargs["vmax"] = data_max

    # Create pcolormesh plot
    im = ax.pcolormesh(
        lons, lats, data.data, transform=ccrs.PlateCarree(), cmap=cmap, **kwargs
    )

    # Add coastlines
    ax.coastlines(linewidth=0.5)

    # Add gridlines
    gl = ax.gridlines(
        linestyle="--",
        draw_labels=True,
        dms=True,
        x_inline=False,
        y_inline=False,
    )
    gl.xlocator = mticker.MultipleLocator(30)
    gl.ylocator = mticker.MultipleLocator(30)
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    # Set title
    ax.set_title(title, fontsize=12, pad=10)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
    cbar.set_label(label, fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig("bayesian_inversion_results.png", dpi=600)
