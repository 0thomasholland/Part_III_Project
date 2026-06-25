from pyslfp.linear_operators import (
    FingerPrintOperator,
    l2_products_operator,
)
from pyslfp.state import EarthState
import pickle
import random
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
)

from pygeoinf_extras import standard_dev
from pygeoinf_extras.operators import (
    point_averaging_area_weighted_operator,
)
from pyslfp.state import EarthState
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange

# ---- Runner configuration (edit these variables directly) ----
OUTPUT_DIR = (
    Path(__file__).resolve().parent / "inversion_results"
)
TOTAL_SETUPS = 160
START_INDEX = 161
N_JOBS = -1
# Recycle worker processes after this many setup tasks.
# Use 1 to restart after each setup; use None to disable recycling.
MAX_TASKS_PER_CHILD = 1

L_MAX = 128
ALTIMETRY_DEGREE_DENSITY = 5.0
ALTIMETRY_STD_DEV = 0.003

TRUTH_LENGTH_SCALE_FACTOR = 0.1
TRUTH_GMSL_STD = 0.01

LENGTH_SCALE_FACTORS = np.array(
    [0.05, 0.15, 0.4], dtype=float
)
OFFSETS_MM = np.array([1.0, 10.0, 50.0], dtype=float)
STD_MULTIPLIERS = np.array([0.5, 2.0], dtype=float)
ACCURATE_PRIOR_MARKER = 1.0

CG_MAXITER = 1000
BASE_RANDOM_SEED = 20260313

def _seed_worker_rng(setup_index: int) -> int:
    """Seed Python and NumPy RNGs per setup for reproducible diversity."""
    seed = BASE_RANDOM_SEED + int(setup_index)
    random.seed(seed)
    np.random.seed(seed)
    return seed

def scalar_z_score(
    estimate: float, truth: float, std_dev: float
) -> float:
    if std_dev <= 0:
        raise ValueError(
            "Standard deviation must be positive."
        )
    return (estimate - truth) / std_dev

def gaussian_measure_summary(
    measure: GaussianMeasure,
    truth: float,
) -> tuple[float, float, float]:
    mean = float(measure.expectation[0])
    std_dev = float(
        np.sqrt(measure.covariance.matrix(dense=True)[0, 0])
    )
    z_score = scalar_z_score(mean, truth, std_dev)
    return z_score, mean, std_dev

def build_truth_setup(
    *, setup_index: int
) -> dict[str, Any]:
    fp = EarthState.from_defaults(lmax=L_MAX)
    fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
    ), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
    ))

    truth_length_scale = (
        TRUTH_LENGTH_SCALE_FACTOR * fp.model.parameters.mean_sea_floor_radius
    )

    truth_ice_change = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=truth_length_scale,
        pattern=IceSheetChange.ThicknessWeightedPattern(),
        ice_gmsl_std=TRUTH_GMSL_STD,
        point_degree_spacing=ALTIMETRY_DEGREE_DENSITY,
    )

    truth_forward_op = (
        truth_ice_change.load_to_ssh_point_estimations_operator
        @ truth_ice_change.ice_thickness_to_load_operator
    )
    model_space = truth_ice_change.ice_thickness.domain
    data_space = truth_forward_op.codomain

    data_error_measure = (
        GaussianMeasure.from_standard_deviation(
            data_space, ALTIMETRY_STD_DEV
        )
    )
    forward_problem = LinearForwardProblem(
        truth_forward_op,
        data_error_measure=data_error_measure,
    )

    truth_prior_measure = truth_ice_change.ice_thickness
    model_true, data = (
        forward_problem.synthetic_model_and_data(
            truth_prior_measure
        )
    )

    gmsl_weighting_function = (
        -fp.model.parameters.ice_density
        * fp.one_minus_ocean_function
        * fp.ice_projection(value=0)
        * 1000
        * fp.model.parameters.length_scale
        / (fp.model.parameters.water_density * fp.ocean_area)
    )
    B = l2_products_operator(
        model_space, [gmsl_weighting_function]
    )
    gmsl_true = float(B(model_true)[0])

    altimetry_points = GridPoints.ocean_altimetry(
        fp,
        degree_spacing=ALTIMETRY_DEGREE_DENSITY,
    )
    F = point_averaging_area_weighted_operator(
        data_space, np.asarray(altimetry_points.lats)
    )

    ssh_point_values = truth_forward_op(model_true)
    ssh_estimation_alt = float(
        F(ssh_point_values)[0] * 1000
    )

    averaged_error = data_error_measure.affine_mapping(
        operator=F
    )
    ssh_std = float(standard_dev(averaged_error) * 1000)
    altimetry_z = float(
        scalar_z_score(
            ssh_estimation_alt, gmsl_true, ssh_std
        )
    )

    return {
        "fp": fp,
        "fp_op": fp_op,
        "B": B,
        "data": data,
        "data_error_measure": data_error_measure,
        "gmsl_true": gmsl_true,
        "ssh_estimation_alt": ssh_estimation_alt,
        "ssh_std": ssh_std,
        "altimetry_z": altimetry_z,
        "truth_length_scale": truth_length_scale,
    }

def build_ice_change_for_case(
    *,
    fp: EarthState,
    fp_op,
    sweep_type: str,
    sweep_value: float,
    truth_length_scale: float,
) -> IceSheetChange:
    kwargs = {
        "finger_print": fp,
        "finger_print_operator": fp_op,
        "length_scale": truth_length_scale,
        "pattern": IceSheetChange.ThicknessWeightedPattern(),
        "ice_gmsl_std": TRUTH_GMSL_STD,
        "point_degree_spacing": ALTIMETRY_DEGREE_DENSITY,
    }

    if sweep_type == "length_scale":
        kwargs["length_scale"] = (
            sweep_value * fp.model.parameters.mean_sea_floor_radius
        )
    elif sweep_type == "mean_offset":
        target_nd = sweep_value / (1000 * fp.model.parameters.length_scale)
        kwargs["gmsl_target_mean"] = target_nd
    elif sweep_type == "std_multiplier":
        kwargs["ice_gmsl_std"] = (
            TRUTH_GMSL_STD * sweep_value
        )
    elif sweep_type == "accurate_prior":
        # Defaults in kwargs already match the truth prior.
        pass
    else:
        raise ValueError(
            f"Unknown sweep_type: {sweep_type}"
        )

    return IceSheetChange.global_ice(**kwargs)

def run_inversion_case(
    *,
    ice_change: IceSheetChange,
    data,
    data_error_measure: GaussianMeasure,
    B,
    gmsl_true: float,
) -> dict[str, float | int]:
    fwd_op = (
        ice_change.load_to_ssh_point_estimations_operator
        @ ice_change.ice_thickness_to_load_operator
    )
    fwd_problem = LinearForwardProblem(
        fwd_op, data_error_measure=data_error_measure
    )
    prior = ice_change.ice_thickness
    inversion = LinearBayesianInversion(fwd_problem, prior)

    iterations = 0

    def callback(_):
        nonlocal iterations
        iterations += 1

    started = time.perf_counter()
    posterior = inversion.model_posterior_measure(
        data,
        CGMatrixSolver(
            maxiter=CG_MAXITER,
            callback=callback,
        ),
    )
    runtime_s = float(time.perf_counter() - started)

    prior_gmsl = prior.affine_mapping(operator=B)
    posterior_gmsl = posterior.affine_mapping(operator=B)

    prior_z, prior_mean, prior_std = (
        gaussian_measure_summary(prior_gmsl, gmsl_true)
    )
    posterior_z, posterior_mean, posterior_std = (
        gaussian_measure_summary(posterior_gmsl, gmsl_true)
    )

    return {
        "prior_z": float(prior_z),
        "prior_mean_mm": float(prior_mean),
        "prior_std_mm": float(prior_std),
        "posterior_z": float(posterior_z),
        "posterior_mean_mm": float(posterior_mean),
        "posterior_std_mm": float(posterior_std),
        "prior_bias_mm": float(prior_mean - gmsl_true),
        "posterior_bias_mm": float(
            posterior_mean - gmsl_true
        ),
        "cg_iterations": int(iterations),
        "runtime_s": runtime_s,
    }

def _worker(setup_index: int) -> str:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = _seed_worker_rng(setup_index)

    file_path = OUTPUT_DIR / (
        f"sensitivity_{setup_index:05d}_{uuid.uuid4().hex}.pkl"
    )
    print(
        "Worker "
        f"{setup_index} starting (seed={seed}), "
        f"will save to {file_path.name}"
    )
    try:
        setup = build_truth_setup(setup_index=setup_index)

        sweep_defs = [
            ("length_scale", LENGTH_SCALE_FACTORS),
            ("mean_offset", OFFSETS_MM),
            ("std_multiplier", STD_MULTIPLIERS),
            (
                "accurate_prior",
                np.array(
                    [ACCURATE_PRIOR_MARKER], dtype=float
                ),
            ),
        ]
        print(
            f"Worker {setup_index} completed truth setup, running sweeps..."
        )
        records = []
        for sweep_type, sweep_values in sweep_defs:
            for sweep_value in sweep_values:
                ice_change = build_ice_change_for_case(
                    fp=setup["fp"],
                    fp_op=setup["fp_op"],
                    sweep_type=sweep_type,
                    sweep_value=float(sweep_value),
                    truth_length_scale=setup[
                        "truth_length_scale"
                    ],
                )

                case_metrics = run_inversion_case(
                    ice_change=ice_change,
                    data=setup["data"],
                    data_error_measure=setup[
                        "data_error_measure"
                    ],
                    B=setup["B"],
                    gmsl_true=setup["gmsl_true"],
                )
                print(
                    f"Setup {setup_index}: {sweep_type} = {sweep_value}"
                )
                record = {
                    "setup_index": setup_index,
                    "sweep_type": sweep_type,
                    "sweep_value": float(sweep_value),
                    "gmsl_true_mm": float(
                        setup["gmsl_true"]
                    ),
                    "altimetry_estimate_mm": float(
                        setup["ssh_estimation_alt"]
                    ),
                    "altimetry_std_mm": float(
                        setup["ssh_std"]
                    ),
                    "altimetry_z": float(
                        setup["altimetry_z"]
                    ),
                    "truth_length_scale_m": float(
                        setup["truth_length_scale"]
                    ),
                    "truth_gmsl_std_nd": float(
                        TRUTH_GMSL_STD
                    ),
                    "altimetry_degree_density": float(
                        ALTIMETRY_DEGREE_DENSITY
                    ),
                    "altimetry_error_std_nd": float(
                        ALTIMETRY_STD_DEV
                    ),
                }
                record.update(case_metrics)
                records.append(record)
        print(
            f"Worker {setup_index} completed sweeps, saving results..."
        )
        payload = {
            "setup_index": setup_index,
            "records": records,
            "n_records": len(records),
        }
        with open(file_path, "wb") as handle:
            pickle.dump(payload, handle)

        return f"Saved {file_path.name} with {len(records)} records"
    except Exception as exc:
        return f"Worker {setup_index} failed: {exc}"

def main() -> None:
    stop_index = START_INDEX + TOTAL_SETUPS

    backend_kwargs: dict[str, int] = {}
    if MAX_TASKS_PER_CHILD is not None:
        backend_kwargs["maxtasksperchild"] = int(
            MAX_TASKS_PER_CHILD
        )

    results = Parallel(
        n_jobs=N_JOBS,
        backend="loky",
	batch_size=1,
        verbose=11,
        backend_kwargs=backend_kwargs,
    )(
        delayed(_worker)(setup_index)
        for setup_index in range(START_INDEX, stop_index)
    )

    for message in results:
        print(message)

if __name__ == "__main__":
    main()
