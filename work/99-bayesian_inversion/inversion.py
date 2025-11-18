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
from pygeoinf.nonlinear_operators import NonLinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)
from Part_III_Project.measure_space.measures import (
    altimetry_measurements_measure,
)

lmax = 64
fp = FingerPrint(
    lmax=lmax,
)
fp.set_state_from_ice_ng()

# %%
# model space

scale = 0.1 * fp.mean_sea_floor_radius

model_space = Sobolev(
    fp.lmax,
    2,
    scale,
    radius=fp.mean_sea_floor_radius,
)

# %%

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

load_space: HilbertSpace = fingerprint_operator.domain
response_space: HilbertSpace = fingerprint_operator.codomain
sea_surface_height_op: LinearOperator = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space: HilbertSpace = sea_surface_height_op.codomain


#### OPERATORS
Load_w_op: LinearOperator = sea_level_change_to_load_operator(
    fp,
    load_space,
)

Load_i_op: LinearOperator = ice_thickness_change_to_load_operator(
    fp,
    load_space,
)

altimetry_operator: LinearOperator = spatial_mutliplication_operator(
    fp.altimetry_projection(
        latitude_max=66,
        latitude_min=-66,
        value=0,
    )
    * fp.ocean_function,
    measurement_space,
)

error_altimetry_operator: LinearOperator = altimetry_operator @ RowLinearOperator(
    [
        sea_surface_height_op @ fingerprint_operator @ Load_w_op,
        measurement_space.identity_operator(),
        measurement_space.identity_operator(),
    ],
)

forward_operator = (
    altimetry_operator
    @ sea_surface_height_op
    @ fingerprint_operator
    @ Load_i_op
    @ ice_projection_operator(fp, load_space)
)
# %%
### VARIABLES
ice_length_scale = 0.1 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.0004 / fp.length_scale  # in meters
ice_net_thickness_change = 0 / fp.length_scale  # in meters

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_standard_deviation = 0.005 / fp.length_scale  # in meters

altimetry_range = 66  # in degrees
altimetry_error_length_scale = 0.005 * fp.mean_sea_floor_radius
altimetry_error_amplitude = 0.001 / fp.length_scale  # in meters

### MEASURES

ice_thickness_change: GaussianMeasure
ice_thickness_change, _ = ice_thickness_change_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=ice_length_scale,
    ice_gmsl_target_std=ice_gmsl_target_std,
    net_thickness_change=ice_net_thickness_change,
)
odt_change: GaussianMeasure
odt_change, _ = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    standard_deviation=odt_standard_deviation,
)
measurement_error: GaussianMeasure = (
    measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        1.5,
        altimetry_error_length_scale,
        altimetry_error_amplitude,
    )
)

error_input_measures = GaussianMeasure.from_direct_sum(
    [odt_change, odt_change, measurement_error],
)

data_error_measure = error_input_measures.affine_mapping(
    operator=error_altimetry_operator,
)

fig, ax, im = plot(odt_change.sample() * fp.length_scale)
fig.colorbar(
    im,
    ax=ax,
    label="Data error measure sample",
    orientation="horizontal",
)

# %%
forward_problem = LinearForwardProblem(
    forward_operator,
    data_error_measure=data_error_measure,
)

model_prior, data = forward_problem.synthetic_model_and_data(
    ice_thickness_change,
)

# %%
inversion = LinearBayesianInversion(
    forward_problem,
    ice_thickness_change,
)


# %%
class ConvergenceMonitor:
    def __init__(self):
        self.iteration = 0
        self.x_norms = []

    def __call__(self, xk):
        self.iteration += 1
        x_norm = np.linalg.norm(xk)
        self.x_norms.append(x_norm)

        if self.iteration % 5 == 0:  # Print every 5 iterations
            print(f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}")


monitor = ConvergenceMonitor()
solver: ScipyIterativeSolver = CGMatrixSolver(
    callback=monitor,
    rtol=1e-4,
    maxiter=500,
)

model_posterior_measure = inversion.model_posterior_measure(
    data,
    solver,
)

print(f"Total iterations: {monitor.iteration}")

# %%

model_posterior_expectation = model_posterior_measure.expectation

print(
    f"number of solutions of fingerprint problme = {fp.solver_counter}",
)
