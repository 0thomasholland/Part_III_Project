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
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%

# generate prior dataset

shift = 0.0
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


data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
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


def inversion_func() -> str:
    output_dir = Path("inversion_results")
    output_dir.mkdir(exist_ok=True)

    # Generate a unique name for this specific random run
    unique_id = uuid.uuid4().hex
    file_path = output_dir / f"inversion_{unique_id}.pkl"

    try:
        model_true, data = (
            forward_problem.synthetic_model_and_data(
                ice_thickness_measure
            )
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

        GMSL_true = B(model_true)

        GMSL_posterior_measure = (
            model_posterior_measure.affine_mapping(
                operator=B
            )
        )

        ssh_esimation_alt = (
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
            "ssh_estimation": ssh_esimation_alt,
            "ice_gmsl_target": ice_gmsl_target,
            "altimetry_error_std_dev": altimetry_std_dev,
            "altimetry_gridding": altimetry_degree_density,
            "iterations_to_solve": iteration,
            "number_of_grid_points": len(
                number_grid_points
            ),
            "shift": shift,
        }

        with open(file_path, "wb") as f:
            pickle.dump(results, f)

        return f"Saved {file_path.name}"

    except Exception as e:
        return f"Error in worker: {str(e)}"


# Total number of random samples you want to collect
TOTAL_SAMPLES = 70

if __name__ == "__main__":
    # n_jobs should be roughly (Total RAM / Max RAM used by one inversion)
    results = Parallel(
        n_jobs=-1, backend="multiprocessing", verbose=11
    )(
        delayed(inversion_func)()
        for _ in range(TOTAL_SAMPLES)
    )
    print(results)
