# %%
import pickle
import uuid
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
)

from pygeoinf_extras import expectation, standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%
# --- Configuration ---
# Shift applied to the truth relative to the prior mean (in mm GMSL).
# The prior remains zero-mean; the synthetic truth is drawn from the
# prior and then offset by this amount.
shift_mm = 0.5  # <-- vary this between runs

altimetry_degree_density = 5.0
ice_gmsl_target = 0.01

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=ice_gmsl_target,
    point_degree_spacing=altimetry_degree_density,
)

ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness
)
model_space = ice_thickness_measure.domain

ice_thickness_to_ssh_point_estimations_op: LinearOperator = (
    ice_change.load_to_ssh_point_estimations_operator
    @ ice_change.ice_thickness_to_load_operator
)

data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)

grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=altimetry_degree_density,
    latitude_range=66.0,
)
number_grid_points = grid_points.coords

# %%
# --- Build the ice-thickness shift vector ---
# We want to add a uniform ice-thickness offset over the ice extent so
# that the resulting GMSL changes by exactly `shift_mm`.
#
# GMSL (mm) = integrate[ -rho_ice * (1 - O) * P_ice * h * L / (rho_w * A_o) ] * 1000
#
# For a spatially uniform shift h_0 over the ice projection the integral
# reduces to:
#   GMSL_per_unit = integrate[ -rho_ice * (1-O) * P_ice * L / (rho_w * A_o) ] * 1000
#
# so h_0 = shift_mm / GMSL_per_unit  (in non-dimensional length units).

_ice_mask = fp.ice_projection(value=0)
_gmsl_per_unit = (
    fp.integrate(
        -fp.ice_density
        * fp.one_minus_ocean_function
        * _ice_mask
        * fp.length_scale
        / (fp.water_density * fp.ocean_area)
    )
    * 1000
)  # now in mm per unit non-dim thickness

_h0 = (
    shift_mm / _gmsl_per_unit
    if _gmsl_per_unit != 0
    else 0.0
)

# Project the uniform offset into the model space (Sobolev coefficients)
_shift_field = _ice_mask * _h0
shift_vector = model_space.project_function(_shift_field)

# %%
# --- GMSL property operator (mm) ---
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
B = averaging_operator(
    model_space, [GMSL_weighting_function]
)

# Verify the shift vector gives the expected GMSL offset
_check_gmsl = B(shift_vector)[0]
print(
    f"Requested shift: {shift_mm:.4f} mm, "
    f"achieved shift: {_check_gmsl:.4f} mm"
)

# %%
altimetry_std_dev = 0.001
data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, altimetry_std_dev
    )
)

forward_problem = LinearForwardProblem(
    ice_thickness_to_ssh_point_estimations_op,
    data_error_measure=data_error_measure,
)

bayesian_inversion = LinearBayesianInversion(
    forward_problem, ice_thickness_measure
)


# %%
def inversion_func() -> str:
    output_dir = Path("inversion_results")
    output_dir.mkdir(exist_ok=True)
    unique_id = uuid.uuid4().hex
    file_path = output_dir / f"inversion_{unique_id}.pkl"

    try:
        # Draw from the prior, then shift the truth
        model_true_unshifted = (
            ice_thickness_measure.sample()
        )
        model_true = model_true_unshifted + shift_vector

        # Generate data from the shifted truth
        data = (
            ice_thickness_to_ssh_point_estimations_op(
                model_true
            )
            + data_error_measure.sample()
        )

        iteration = 0

        def callback(residual_norm):
            nonlocal iteration
            iteration += 1

        model_posterior_measure = (
            bayesian_inversion.model_posterior_measure(
                data,
                CGMatrixSolver(
                    maxiter=200, callback=callback
                ),
            )
        )

        model_posterior_expectation = (
            model_posterior_measure.expectation
        )

        GMSL_true = B(model_true)

        GMSL_posterior_measure = (
            model_posterior_measure.affine_mapping(
                operator=B
            )
        )

        ssh_estimation_alt = (
            ice_change.load_to_point_estimated_gmsl_operator(
                ice_change.ice_thickness_to_load_operator(
                    model_true
                )
            )[0]
            * 1000
        )

        posterior_mean = expectation(GMSL_posterior_measure)
        posterior_std_dev = standard_dev(
            GMSL_posterior_measure
        )

        results = {
            "posterior_mean": posterior_mean,
            "posterior_std_dev": posterior_std_dev,
            "gmsl_true": GMSL_true[0],
            "ssh_estimation": ssh_estimation_alt,
            "ice_gmsl_target": ice_gmsl_target,
            "altimetry_error_std_dev": altimetry_std_dev,
            "altimetry_gridding": altimetry_degree_density,
            "iterations_to_solve": iteration,
            "number_of_grid_points": len(
                number_grid_points
            ),
            "shift_mm": shift_mm,
        }

        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        return f"Saved {file_path.name}"

    except Exception as e:
        return f"Error in worker: {str(e)}"


# %%
TOTAL_SAMPLES = 70

if __name__ == "__main__":
    results = Parallel(
        n_jobs=-1, backend="multiprocessing", verbose=11
    )(
        delayed(inversion_func)()
        for _ in range(TOTAL_SAMPLES)
    )
    print(results)
