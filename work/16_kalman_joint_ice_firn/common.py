from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygeoinf import (
    BlockLinearOperator,
    EuclideanSpace,
    GaussianMeasure,
    RowLinearOperator,
)
from pygeoinf_extras.plots import plot_bivariate_corner
from pyslfp.linear_operators import (
    FingerPrintOperator,
    grace_observation_operator,
)
from pyslfp.state import EarthState
from scipy import stats

from pyslfp_extras import plot
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange


@dataclass
class ExperimentConfig:
    lmax: int = 128
    n_epochs: int = 8
    seed: int = 7
    length_scale_fraction: float = 0.10
    ice_gmsl_std: float = 0.003
    firn_gmsl_std: float = 0.002
    firn_density_fraction: float = 0.30
    altimetry_latitude_range: float = 66.0
    ssh_degree_spacing: float = 10.0
    ice_degree_spacing: float = 10.0
    bore_degree_spacing: float = 5.0
    n_bores_per_epoch: int = 6
    bore_revisit_probability: float = 0.35
    ssh_noise_std: float = 0.0010
    ice_noise_std: float = 0.0010
    bore_noise_std: float = 0.0004
    grace_noise_std_m: float = 0.0027
    grace_observation_degree: int = 96
    ice_process_scale: float = 0.18
    firn_process_scale: float = 0.25
    state_ar1: float = 1.0
    ensemble_size: int = 64
    output_dir: Path = Path("work/16_kalman_joint_ice_firn/outputs")


@dataclass
class TemporalSetup:
    config: ExperimentConfig
    fp: EarthState
    ice: IceSheetChange
    model_space: object
    ice_space: object
    firn_space: object
    ssh_matrix: np.ndarray
    ice_matrix: np.ndarray
    grace_matrix: np.ndarray
    gmsl_matrix: np.ndarray
    ice_gmsl_matrix: np.ndarray
    firn_gmsl_matrix: np.ndarray
    ice_dim: int
    firn_dim: int
    ice_projection: np.ndarray
    bore_candidate_coords: list[tuple[float, float]]
    bore_matrix_cache: dict[tuple[tuple[float, float], ...], np.ndarray] = (
        field(default_factory=dict)
    )


@dataclass
class ObservationSet:
    name: str
    H: np.ndarray
    y: np.ndarray
    noise_std: np.ndarray
    bore_coords: list[tuple[float, float]]


def _point_operator(space, coords):
    sig = inspect.signature(space.point_evaluation_operator)
    if "matrix_free" in sig.parameters:
        return space.point_evaluation_operator(
            coords,
            matrix_free=True,
            parallel=False,
        )
    return space.point_evaluation_operator(coords)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_setup(config: ExperimentConfig) -> TemporalSetup:
    fp = EarthState.from_defaults(lmax=config.lmax)
    fp_op = FingerPrintOperator(
        fp,
        load_parameters=(
            2,
            fp.model.parameters.mean_sea_floor_radius
            * config.length_scale_fraction,
        ),
        response_parameters=(
            3,
            fp.model.parameters.mean_sea_floor_radius
            * config.length_scale_fraction,
        ),
    )
    ice = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=(
            config.length_scale_fraction
            * fp.model.parameters.mean_sea_floor_radius
        ),
        pattern=IceSheetChange.ThicknessWeightedPattern(),
        ice_gmsl_std=config.ice_gmsl_std,
        firn_gmsl_std=config.firn_gmsl_std,
        firn_density=(
            config.firn_density_fraction
            * fp.model.parameters.ice_density
        ),
        include_firn=True,
    )
    model_prior = GaussianMeasure.from_direct_sum(
        [ice.ice_thickness, ice.firn_thickness]
    )

    ice_space = ice.ice_thickness.domain
    firn_space = ice.firn_thickness.domain
    ice_dim = ice_space.dim
    firn_dim = firn_space.dim

    ssh_altimetry = GridPoints.ocean_altimetry(
        fp,
        degree_spacing=config.ssh_degree_spacing,
        latitude_range=config.altimetry_latitude_range,
    )
    ice_altimetry = GridPoints.ice(
        fp,
        degree_spacing=config.ice_degree_spacing,
    )
    bore_candidates = GridPoints.ice(
        fp,
        degree_spacing=config.bore_degree_spacing,
    )
    grace_op = grace_observation_operator(
        fp_op.codomain,
        config.grace_observation_degree,
    )

    ssh_point_operator = _point_operator(
        ice.load_to_ssh_operator.codomain,
        ssh_altimetry.coords,
    )
    ice_point_operator_ice = _point_operator(
        ice_space,
        ice_altimetry.coords,
    )
    ice_point_operator_firn = _point_operator(
        firn_space,
        ice_altimetry.coords,
    )

    f11 = (
        ssh_point_operator
        @ ice.load_to_ssh_operator
        @ ice.ice_thickness_to_load_operator
    )
    f12 = (
        ssh_point_operator
        @ ice.load_to_ssh_operator
        @ ice.firn_thickness_to_load_operator
    )
    f21 = ice_point_operator_ice
    f22 = ice_point_operator_firn
    f31 = grace_op @ fp_op @ ice.ice_thickness_to_load_operator
    f32 = grace_op @ fp_op @ ice.firn_thickness_to_load_operator

    ssh_matrix = BlockLinearOperator([[f11, f12]]).matrix(dense=True)
    ice_matrix = BlockLinearOperator([[f21, f22]]).matrix(dense=True)
    grace_matrix = BlockLinearOperator([[f31, f32]]).matrix(dense=True)
    gmsl_matrix = RowLinearOperator(
        [
            ice.ice_thickness_to_gmsl_operator,
            ice.firn_thickness_to_gmsl_operator,
        ]
    ).matrix(dense=True)
    ice_gmsl_matrix = RowLinearOperator(
        [
            ice.ice_thickness_to_gmsl_operator,
            firn_space.zero_operator(
                codomain=ice.ice_thickness_to_gmsl_operator.codomain
            ),
        ]
    ).matrix(dense=True)
    firn_gmsl_matrix = RowLinearOperator(
        [
            ice_space.zero_operator(
                codomain=ice.firn_thickness_to_gmsl_operator.codomain
            ),
            ice.firn_thickness_to_gmsl_operator,
        ]
    ).matrix(dense=True)
    ice_projection = fp.ice_projection(value=0).data.copy()

    return TemporalSetup(
        config=config,
        fp=fp,
        ice=ice,
        model_space=model_prior.domain,
        ice_space=ice_space,
        firn_space=firn_space,
        ssh_matrix=ssh_matrix,
        ice_matrix=ice_matrix,
        grace_matrix=grace_matrix,
        gmsl_matrix=gmsl_matrix,
        ice_gmsl_matrix=ice_gmsl_matrix,
        firn_gmsl_matrix=firn_gmsl_matrix,
        ice_dim=ice_dim,
        firn_dim=firn_dim,
        ice_projection=ice_projection,
        bore_candidate_coords=bore_candidates.coords,
    )


def _seed_numpy_from_rng(rng: np.random.Generator) -> None:
    np.random.seed(int(rng.integers(0, 2**32 - 1)))


def _sample_component_vector(space, measure, rng: np.random.Generator) -> np.ndarray:
    _seed_numpy_from_rng(rng)
    return np.asarray(space.to_components(measure.sample()), dtype=float)


def sample_state_vector(
    setup: TemporalSetup,
    rng: np.random.Generator,
) -> np.ndarray:
    ice_vector = _sample_component_vector(
        setup.ice_space,
        setup.ice.ice_thickness,
        rng,
    )
    firn_vector = _sample_component_vector(
        setup.firn_space,
        setup.ice.firn_thickness,
        rng,
    )
    return np.concatenate([ice_vector, firn_vector])


def sample_process_vector(
    setup: TemporalSetup,
    rng: np.random.Generator,
) -> np.ndarray:
    innovation = sample_state_vector(setup, rng)
    innovation[: setup.ice_dim] *= setup.config.ice_process_scale
    innovation[setup.ice_dim :] *= setup.config.firn_process_scale
    return innovation


def sample_ensemble(
    setup: TemporalSetup,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.column_stack(
        [
            sample_state_vector(setup, rng)
            for _ in range(setup.config.ensemble_size)
        ]
    )


def sample_process_ensemble(
    setup: TemporalSetup,
    rng: np.random.Generator,
) -> np.ndarray:
    return np.column_stack(
        [
            sample_process_vector(setup, rng)
            for _ in range(setup.config.ensemble_size)
        ]
    )


def generate_bore_schedule(
    candidate_coords: list[tuple[float, float]],
    n_epochs: int,
    n_bores_per_epoch: int,
    revisit_probability: float,
    rng: np.random.Generator,
) -> list[list[tuple[float, float]]]:
    schedule: list[list[tuple[float, float]]] = []
    previous: list[tuple[float, float]] = []
    n_bores = min(n_bores_per_epoch, len(candidate_coords))
    for _ in range(n_epochs):
        retained = [
            coord
            for coord in previous
            if rng.random() < revisit_probability
        ]
        retained = retained[:n_bores]
        remaining = [
            coord
            for coord in candidate_coords
            if coord not in retained
        ]
        needed = n_bores - len(retained)
        if needed > 0 and remaining:
            indices = rng.choice(
                len(remaining),
                size=needed,
                replace=False,
            )
            retained.extend(remaining[index] for index in indices)
        schedule.append(retained)
        previous = retained
    return schedule


def build_bore_matrix(
    setup: TemporalSetup,
    coords: list[tuple[float, float]],
) -> np.ndarray:
    cache_key = tuple(coords)
    cached = setup.bore_matrix_cache.get(cache_key)
    if cached is not None:
        return cached

    point_op = _point_operator(setup.firn_space, coords)
    firn_matrix = point_op.matrix(dense=True)
    zeros = np.zeros((firn_matrix.shape[0], setup.ice_dim))
    bore_matrix = np.hstack([zeros, firn_matrix])
    setup.bore_matrix_cache[cache_key] = bore_matrix
    return bore_matrix


def sample_truth(
    setup: TemporalSetup,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    states = [sample_state_vector(setup, rng)]
    for _ in range(1, setup.config.n_epochs):
        innovation = sample_process_vector(setup, rng)
        states.append(setup.config.state_ar1 * states[-1] + innovation)
    return states


def build_observations(
    setup: TemporalSetup,
    truth_states: list[np.ndarray],
    bore_schedule: list[list[tuple[float, float]]],
    include_ssh: bool,
    include_ice: bool,
    include_bores: bool,
    include_grace: bool,
    rng: np.random.Generator,
) -> list[ObservationSet]:
    observations: list[ObservationSet] = []
    for epoch, truth in enumerate(truth_states):
        matrices: list[np.ndarray] = []
        noises: list[np.ndarray] = []
        labels: list[str] = []

        if include_ssh:
            matrices.append(setup.ssh_matrix)
            noises.append(
                np.full(
                    setup.ssh_matrix.shape[0],
                    setup.config.ssh_noise_std,
                )
            )
            labels.append("ssh")
        if include_ice:
            matrices.append(setup.ice_matrix)
            noises.append(
                np.full(
                    setup.ice_matrix.shape[0],
                    setup.config.ice_noise_std,
                )
            )
            labels.append("ice")
        if include_bores:
            bore_matrix = build_bore_matrix(setup, bore_schedule[epoch])
            matrices.append(bore_matrix)
            noises.append(
                np.full(
                    bore_matrix.shape[0],
                    setup.config.bore_noise_std,
                )
            )
            labels.append("bore")
        if include_grace:
            matrices.append(setup.grace_matrix)
            noises.append(
                np.full(
                    setup.grace_matrix.shape[0],
                    setup.config.grace_noise_std_m
                    / setup.fp.model.parameters.length_scale,
                )
            )
            labels.append("grace")

        H = np.vstack(matrices)
        noise_std = np.concatenate(noises)
        y = H @ truth + rng.normal(scale=noise_std)
        observations.append(
            ObservationSet(
                name="+".join(labels),
                H=H,
                y=y,
                noise_std=noise_std,
                bore_coords=bore_schedule[epoch]
                if include_bores
                else [],
            )
        )
    return observations


def _ensemble_mean(ensemble: np.ndarray) -> np.ndarray:
    return np.mean(ensemble, axis=1)


def _ensemble_anomalies(ensemble: np.ndarray) -> np.ndarray:
    return ensemble - _ensemble_mean(ensemble)[:, None]


def _ensemble_std_vector(ensemble: np.ndarray) -> np.ndarray:
    return np.std(ensemble, axis=1, ddof=1)


def _ensemble_rank(ensemble: np.ndarray) -> int:
    anomalies = _ensemble_anomalies(ensemble)
    gram = anomalies.T @ anomalies
    return int(np.linalg.matrix_rank(gram, tol=1e-10))


def filter_and_smooth(
    setup: TemporalSetup,
    observations: list[ObservationSet],
    rng: np.random.Generator,
) -> dict[str, list[np.ndarray] | list[int] | list[float]]:
    ensemble = sample_ensemble(setup, rng)

    filtered_means = []
    filtered_ensembles = []
    filtered_std_vectors = []
    predicted_means = []
    predicted_ensembles = []
    predicted_std_vectors = []
    filtered_ranks = []
    predicted_ranks = []

    for obs in observations:
        predicted = (
            setup.config.state_ar1 * ensemble
            + sample_process_ensemble(setup, rng)
        )
        predicted_means.append(_ensemble_mean(predicted))
        predicted_ensembles.append(predicted.copy())
        predicted_std_vectors.append(_ensemble_std_vector(predicted))
        predicted_ranks.append(_ensemble_rank(predicted))

        anomalies = _ensemble_anomalies(predicted)
        observed_ensemble = obs.H @ predicted
        observed_anomalies = _ensemble_anomalies(observed_ensemble)
        n_members = predicted.shape[1]
        innovation_covariance = (
            observed_anomalies @ observed_anomalies.T / (n_members - 1)
        )
        innovation_covariance.flat[
            :: innovation_covariance.shape[0] + 1
        ] += obs.noise_std**2
        gain = (
            anomalies
            @ observed_anomalies.T
            / (n_members - 1)
            @ np.linalg.pinv(innovation_covariance)
        )
        perturbed_obs = obs.y[:, None] + rng.normal(
            scale=obs.noise_std[:, None],
            size=observed_ensemble.shape,
        )
        ensemble = predicted + gain @ (perturbed_obs - observed_ensemble)

        filtered_means.append(_ensemble_mean(ensemble))
        filtered_ensembles.append(ensemble.copy())
        filtered_std_vectors.append(_ensemble_std_vector(ensemble))
        filtered_ranks.append(_ensemble_rank(ensemble))

    smoothed_means = [mean.copy() for mean in filtered_means]
    smoothed_ensembles = [ens.copy() for ens in filtered_ensembles]
    smoothed_std_vectors = [vec.copy() for vec in filtered_std_vectors]

    return {
        "predicted_means": predicted_means,
        "predicted_ensembles": predicted_ensembles,
        "predicted_std_vectors": predicted_std_vectors,
        "filtered_means": filtered_means,
        "filtered_ensembles": filtered_ensembles,
        "filtered_std_vectors": filtered_std_vectors,
        "smoothed_means": smoothed_means,
        "smoothed_ensembles": smoothed_ensembles,
        "smoothed_std_vectors": smoothed_std_vectors,
        "predicted_ranks": predicted_ranks,
        "filtered_ranks": filtered_ranks,
        "predicted_variance": [1.0] * len(predicted_ranks),
        "filtered_variance": [1.0] * len(filtered_ranks),
    }


def vector_to_grids(setup: TemporalSetup, state_vector: np.ndarray):
    return setup.model_space.from_components(state_vector)


def _component_masks(setup: TemporalSetup) -> np.ndarray:
    return setup.ice_projection > 0.5


def _grid_error_and_z(
    setup: TemporalSetup,
    truth_grid,
    mean_grid,
    std_grid,
) -> tuple[np.ndarray, np.ndarray]:
    mask = _component_masks(setup)
    scale = 1000.0 * setup.fp.model.parameters.length_scale
    error_mm = scale * (mean_grid.data[mask] - truth_grid.data[mask])
    std_mm = scale * std_grid.data[mask]
    safe_std_mm = np.where(std_mm > 0.0, std_mm, np.nan)
    return error_mm, error_mm / safe_std_mm


def _ensemble_scalar_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1))


def summarise_solution(
    setup: TemporalSetup,
    truth_states: list[np.ndarray],
    filtered_means: list[np.ndarray],
    smoothed_means: list[np.ndarray],
    filtered_std_vectors: list[np.ndarray],
    smoothed_std_vectors: list[np.ndarray],
    filtered_ensembles: list[np.ndarray],
    smoothed_ensembles: list[np.ndarray],
    observations: list[ObservationSet],
) -> tuple[pd.DataFrame, list[dict[str, np.ndarray]]]:
    rows = []
    diagnostics = []
    for epoch, truth_state in enumerate(truth_states):
        true_gmsl_mm = float(1000.0 * (setup.gmsl_matrix @ truth_state)[0])

        filtered_gmsl_samples_mm = (
            1000.0 * (setup.gmsl_matrix @ filtered_ensembles[epoch]).ravel()
        )
        smoothed_gmsl_samples_mm = (
            1000.0 * (setup.gmsl_matrix @ smoothed_ensembles[epoch]).ravel()
        )

        filtered_gmsl_mm = float(np.mean(filtered_gmsl_samples_mm))
        smoothed_gmsl_mm = float(np.mean(smoothed_gmsl_samples_mm))
        filtered_gmsl_std_mm = _ensemble_scalar_std(
            filtered_gmsl_samples_mm
        )
        smoothed_gmsl_std_mm = _ensemble_scalar_std(
            smoothed_gmsl_samples_mm
        )

        truth_ice, truth_firn = vector_to_grids(setup, truth_state)
        filtered_ice, filtered_firn = vector_to_grids(
            setup, filtered_means[epoch]
        )
        smoothed_ice, smoothed_firn = vector_to_grids(
            setup, smoothed_means[epoch]
        )
        filtered_std_ice, filtered_std_firn = vector_to_grids(
            setup, filtered_std_vectors[epoch]
        )
        smoothed_std_ice, smoothed_std_firn = vector_to_grids(
            setup, smoothed_std_vectors[epoch]
        )

        filtered_ice_error_mm, filtered_ice_z = _grid_error_and_z(
            setup,
            truth_ice,
            filtered_ice,
            filtered_std_ice,
        )
        filtered_firn_error_mm, filtered_firn_z = _grid_error_and_z(
            setup,
            truth_firn,
            filtered_firn,
            filtered_std_firn,
        )
        smoothed_ice_error_mm, smoothed_ice_z = _grid_error_and_z(
            setup,
            truth_ice,
            smoothed_ice,
            smoothed_std_ice,
        )
        smoothed_firn_error_mm, smoothed_firn_z = _grid_error_and_z(
            setup,
            truth_firn,
            smoothed_firn,
            smoothed_std_firn,
        )

        diagnostics.append(
            {
                "filtered_ice_error_mm": filtered_ice_error_mm,
                "filtered_ice_z": filtered_ice_z,
                "filtered_firn_error_mm": filtered_firn_error_mm,
                "filtered_firn_z": filtered_firn_z,
                "smoothed_ice_error_mm": smoothed_ice_error_mm,
                "smoothed_ice_z": smoothed_ice_z,
                "smoothed_firn_error_mm": smoothed_firn_error_mm,
                "smoothed_firn_z": smoothed_firn_z,
            }
        )

        rows.append(
            {
                "epoch": epoch,
                "n_bores": len(observations[epoch].bore_coords),
                "true_gmsl_mm": true_gmsl_mm,
                "filtered_gmsl_mm": filtered_gmsl_mm,
                "smoothed_gmsl_mm": smoothed_gmsl_mm,
                "filtered_gmsl_error_mm": filtered_gmsl_mm - true_gmsl_mm,
                "smoothed_gmsl_error_mm": smoothed_gmsl_mm - true_gmsl_mm,
                "filtered_gmsl_abs_z": abs(
                    (filtered_gmsl_mm - true_gmsl_mm)
                    / filtered_gmsl_std_mm
                ),
                "smoothed_gmsl_abs_z": abs(
                    (smoothed_gmsl_mm - true_gmsl_mm)
                    / smoothed_gmsl_std_mm
                ),
                "filtered_gmsl_z": (
                    filtered_gmsl_mm - true_gmsl_mm
                )
                / filtered_gmsl_std_mm,
                "smoothed_gmsl_z": (
                    smoothed_gmsl_mm - true_gmsl_mm
                )
                / smoothed_gmsl_std_mm,
                "filtered_gmsl_std_mm": filtered_gmsl_std_mm,
                "smoothed_gmsl_std_mm": smoothed_gmsl_std_mm,
                "filtered_ice_mean_abs_z": float(
                    np.nanmean(np.abs(filtered_ice_z))
                ),
                "smoothed_ice_mean_abs_z": float(
                    np.nanmean(np.abs(smoothed_ice_z))
                ),
                "filtered_firn_mean_abs_z": float(
                    np.nanmean(np.abs(filtered_firn_z))
                ),
                "smoothed_firn_mean_abs_z": float(
                    np.nanmean(np.abs(smoothed_firn_z))
                ),
                "filtered_ice_rms_z": float(
                    np.sqrt(np.nanmean(filtered_ice_z**2))
                ),
                "smoothed_ice_rms_z": float(
                    np.sqrt(np.nanmean(smoothed_ice_z**2))
                ),
                "filtered_firn_rms_z": float(
                    np.sqrt(np.nanmean(filtered_firn_z**2))
                ),
                "smoothed_firn_rms_z": float(
                    np.sqrt(np.nanmean(smoothed_firn_z**2))
                ),
                "filtered_state_std": float(
                    np.sqrt(np.mean(filtered_std_vectors[epoch] ** 2))
                ),
                "smoothed_state_std": float(
                    np.sqrt(np.mean(smoothed_std_vectors[epoch] ** 2))
                ),
                "filtered_state_trace": float(
                    np.sum(filtered_std_vectors[epoch] ** 2)
                ),
                "smoothed_state_trace": float(
                    np.sum(smoothed_std_vectors[epoch] ** 2)
                ),
            }
        )

    return pd.DataFrame(rows), diagnostics


def run_variant(
    config: ExperimentConfig,
    name: str,
    include_ssh: bool,
    include_ice: bool,
    include_bores: bool,
    include_grace: bool,
    truth_states: list[np.ndarray] | None = None,
    bore_schedule: list[list[tuple[float, float]]] | None = None,
    setup: TemporalSetup | None = None,
) -> dict[str, object]:
    setup = build_setup(config) if setup is None else setup
    truth_rng = np.random.default_rng(config.seed)
    bore_rng = np.random.default_rng(config.seed + 1)
    observation_rng = np.random.default_rng(config.seed + 2)
    filter_rng = np.random.default_rng(config.seed + 3)

    if truth_states is None:
        truth_states = sample_truth(setup, truth_rng)
    if bore_schedule is None:
        bore_schedule = generate_bore_schedule(
            setup.bore_candidate_coords,
            config.n_epochs,
            config.n_bores_per_epoch,
            config.bore_revisit_probability,
            bore_rng,
        )

    observations = build_observations(
        setup,
        truth_states,
        bore_schedule,
        include_ssh=include_ssh,
        include_ice=include_ice,
        include_bores=include_bores,
        include_grace=include_grace,
        rng=observation_rng,
    )
    solution = filter_and_smooth(setup, observations, filter_rng)
    summary, diagnostics = summarise_solution(
        setup,
        truth_states,
        solution["filtered_means"],
        solution["smoothed_means"],
        solution["filtered_std_vectors"],
        solution["smoothed_std_vectors"],
        solution["filtered_ensembles"],
        solution["smoothed_ensembles"],
        observations,
    )
    summary["variant"] = name
    summary["mean_predicted_rank"] = float(
        np.mean(solution["predicted_ranks"])
    )
    summary["mean_filtered_rank"] = float(
        np.mean(solution["filtered_ranks"])
    )
    summary["mean_predicted_variance_fraction"] = 1.0
    summary["mean_filtered_variance_fraction"] = 1.0
    return {
        "setup": setup,
        "truth_states": truth_states,
        "bore_schedule": bore_schedule,
        "observations": observations,
        "solution": solution,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def plot_zscore_timeseries(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(
        summary["epoch"],
        summary["filtered_ice_mean_abs_z"],
        label="Filtered ice",
    )
    axes[0].plot(
        summary["epoch"],
        summary["smoothed_ice_mean_abs_z"],
        label="Smoothed ice",
    )
    axes[0].set_ylabel("Mean |z|")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        summary["epoch"],
        summary["filtered_firn_mean_abs_z"],
        label="Filtered firn",
    )
    axes[1].plot(
        summary["epoch"],
        summary["smoothed_firn_mean_abs_z"],
        label="Smoothed firn",
    )
    axes[1].set_ylabel("Mean |z|")
    axes[1].set_xlabel("Epoch")
    axes[1].set_xticks(summary["epoch"])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_gmsl_timeseries(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(summary["epoch"], summary["true_gmsl_mm"], label="True")
    ax.plot(summary["epoch"], summary["filtered_gmsl_mm"], label="Filtered")
    ax.plot(summary["epoch"], summary["smoothed_gmsl_mm"], label="Smoothed")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total GMSL (mm)")
    ax.set_xticks(summary["epoch"])
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_variant_comparison(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for variant, group in summary.groupby("variant"):
        ax.plot(group["epoch"], group[metric], label=variant)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["epoch"].unique()))
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_gmsl_zscore_timeseries(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(summary["epoch"], summary["filtered_gmsl_z"], label="Filtered")
    axes[0].plot(summary["epoch"], summary["smoothed_gmsl_z"], label="Smoothed")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_ylabel("GMSL z")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        summary["epoch"],
        summary["filtered_gmsl_abs_z"],
        label="Filtered |z|",
    )
    axes[1].plot(
        summary["epoch"],
        summary["smoothed_gmsl_abs_z"],
        label="Smoothed |z|",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("|GMSL z|")
    axes[1].set_xticks(summary["epoch"])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_state_uncertainty_timeseries(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(summary["epoch"], summary["filtered_state_std"], label="Filtered")
    axes[0].plot(summary["epoch"], summary["smoothed_state_std"], label="Smoothed")
    axes[0].set_ylabel("Mean state std")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        summary["epoch"],
        summary["filtered_state_trace"],
        label="Filtered",
    )
    axes[1].plot(
        summary["epoch"],
        summary["smoothed_state_trace"],
        label="Smoothed",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Trace(P)")
    axes[1].set_xticks(summary["epoch"])
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_rank_timeseries(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(summary["epoch"], summary["mean_predicted_rank"], label="Predicted rank")
    ax.plot(summary["epoch"], summary["mean_filtered_rank"], label="Filtered rank")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective rank")
    ax.set_xticks(summary["epoch"])
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_bore_network(
    bore_schedule: list[list[tuple[float, float]]],
    output_path: Path,
    title: str,
) -> None:
    n_epochs = len(bore_schedule)
    fig, axes = plt.subplots(
        n_epochs,
        1,
        figsize=(8, max(2 * n_epochs, 6)),
        sharex=True,
        constrained_layout=True,
    )
    if n_epochs == 1:
        axes = [axes]

    previous = set()
    for epoch, coords in enumerate(bore_schedule):
        coords_set = set(coords)
        revisits = [coord for coord in coords if coord in previous]
        new_coords = [coord for coord in coords if coord not in previous]
        axis = axes[epoch]
        if new_coords:
            axis.scatter(
                [coord[1] for coord in new_coords],
                [coord[0] for coord in new_coords],
                label="New",
                color="tab:blue",
            )
        if revisits:
            axis.scatter(
                [coord[1] for coord in revisits],
                [coord[0] for coord in revisits],
                label="Revisit",
                color="tab:orange",
                marker="x",
            )
        axis.set_ylabel(f"Epoch {epoch}\nlat")
        axis.set_xlim(0, 360)
        axis.set_ylim(-90, 90)
        axis.set_yticks([-60, 0, 60])
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        previous = coords_set

    axes[-1].set_xlabel("Longitude (deg)")
    axes[-1].set_xticks([0, 60, 120, 180, 240, 300, 360])
    fig.suptitle(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_variant_improvement(
    summary: pd.DataFrame,
    reference_variant: str,
    comparison_variant: str,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    reference = summary[summary["variant"] == reference_variant].sort_values("epoch")
    comparison = summary[summary["variant"] == comparison_variant].sort_values("epoch")
    delta = comparison[metric].to_numpy() - reference[metric].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(reference["epoch"], delta)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_xticks(reference["epoch"])
    ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_error_distributions(
    diagnostics: list[dict[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    ice_error = np.concatenate([row["smoothed_ice_error_mm"] for row in diagnostics])
    ice_z = np.concatenate([row["smoothed_ice_z"] for row in diagnostics])
    firn_error = np.concatenate([row["smoothed_firn_error_mm"] for row in diagnostics])
    firn_z = np.concatenate([row["smoothed_firn_z"] for row in diagnostics])

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].hist(ice_error[np.isfinite(ice_error)], bins=50, color="tab:blue", alpha=0.8)
    axes[0, 0].set_title("Ice posterior error")
    axes[0, 0].set_xlabel("Error (mm)")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(ice_z[np.isfinite(ice_z)], bins=50, color="tab:purple", alpha=0.8)
    axes[0, 1].set_title("Ice posterior z-score")
    axes[0, 1].set_xlabel("z")

    axes[1, 0].hist(firn_error[np.isfinite(firn_error)], bins=50, color="tab:green", alpha=0.8)
    axes[1, 0].set_title("Firn posterior error")
    axes[1, 0].set_xlabel("Error (mm)")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(firn_z[np.isfinite(firn_z)], bins=50, color="tab:red", alpha=0.8)
    axes[1, 1].set_title("Firn posterior z-score")
    axes[1, 1].set_xlabel("z")

    fig.suptitle(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def posterior_std_grids(
    setup: TemporalSetup,
    std_vector: np.ndarray,
):
    return vector_to_grids(setup, std_vector)


def plot_uncertainty_reduction_maps(
    setup: TemporalSetup,
    full_std_vectors: list[np.ndarray],
    reference_std_vectors: list[np.ndarray],
    output_path: Path,
    component: str,
    title: str,
) -> None:
    epochs = [0, len(full_std_vectors) // 2, len(full_std_vectors) - 1]
    scale = 1000.0 * setup.fp.model.parameters.length_scale
    ice_mask = setup.fp.ice_projection()
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10, 10),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson()},
    )

    ref_vmax = 0.0
    red_vmax = 0.0
    grids = []
    for epoch in epochs:
        full_ice, full_firn = posterior_std_grids(setup, full_std_vectors[epoch])
        ref_ice, ref_firn = posterior_std_grids(setup, reference_std_vectors[epoch])
        full_grid = full_firn if component == "firn" else full_ice
        ref_grid = ref_firn if component == "firn" else ref_ice
        scaled_ref_grid = scale * ref_grid * ice_mask
        reduction_grid = scale * (ref_grid - full_grid) * ice_mask
        grids.append((scaled_ref_grid, reduction_grid))
        ref_vmax = max(ref_vmax, float(np.nanmax(scaled_ref_grid.data)))
        red_vmax = max(
            red_vmax,
            float(np.nanmax(np.abs(reduction_grid.data))),
        )

    for row, epoch in enumerate(epochs):
        ref_grid, reduction_grid = grids[row]
        ref_image = plot(
            ref_grid,
            ax=axes[row, 0],
            cmap="viridis",
            vmin=0.0,
            vmax=ref_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )[2]
        red_image = plot(
            reduction_grid,
            ax=axes[row, 1],
            cmap="seismic",
            vmin=-red_vmax,
            vmax=red_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )[2]
        if row == 0:
            axes[row, 0].set_title("Reference posterior std")
            axes[row, 1].set_title("Std reduction")
        axes[row, 0].text(
            -0.12,
            0.5,
            f"Epoch {epoch}",
            rotation=90,
            va="center",
            ha="right",
            transform=axes[row, 0].transAxes,
            fontsize=10,
        )

    fig.colorbar(ref_image, ax=axes[:, 0], shrink=0.85, label="Posterior std (mm)")
    fig.colorbar(red_image, ax=axes[:, 1], shrink=0.85, label="Std reduction (mm)")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def gmsl_2d_posterior(
    result: dict[str, object],
    epoch: int = -1,
) -> tuple[GaussianMeasure, np.ndarray]:
    setup = result["setup"]
    truth_state = result["truth_states"][epoch]
    ensemble = result["solution"]["smoothed_ensembles"][epoch]

    ice_samples = 1000.0 * (setup.ice_gmsl_matrix @ ensemble).ravel()
    firn_samples = 1000.0 * (setup.firn_gmsl_matrix @ ensemble).ravel()
    samples = np.vstack([ice_samples, firn_samples])
    mean = np.mean(samples, axis=1)
    covariance = np.cov(samples)
    measure = GaussianMeasure.from_covariance_matrix(
        EuclideanSpace(2),
        covariance,
        expectation=mean,
    )
    true_values = np.array(
        [
            1000.0 * (setup.ice_gmsl_matrix @ truth_state)[0],
            1000.0 * (setup.firn_gmsl_matrix @ truth_state)[0],
        ]
    )
    return measure, true_values


def plot_gmsl_bivariate_overlay(
    results: dict[str, dict[str, object]],
    output_path: Path,
    title: str,
    epoch: int = -1,
) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8, 8),
        gridspec_kw={"width_ratios": [2, 1], "height_ratios": [1, 2]},
    )
    ax_top = axes[0, 0]
    ax_main = axes[1, 0]
    ax_right = axes[1, 1]
    ax_legend = axes[0, 1]
    ax_legend.axis("off")

    true_values = None
    cmap = plt.get_cmap("tab10")
    for index, (label, result) in enumerate(results.items()):
        measure, true_values = gmsl_2d_posterior(result, epoch=epoch)
        mean = measure.expectation
        cov = measure.covariance.matrix(dense=True)
        sigma_ice = np.sqrt(cov[0, 0])
        sigma_firn = np.sqrt(cov[1, 1])
        color = cmap(index)

        x0 = np.linspace(mean[0] - 4 * sigma_ice, mean[0] + 4 * sigma_ice, 300)
        x1 = np.linspace(mean[1] - 4 * sigma_firn, mean[1] + 4 * sigma_firn, 300)
        ax_top.plot(x0, stats.norm.pdf(x0, mean[0], sigma_ice), color=color, linewidth=1.6, label=label)
        ax_right.plot(stats.norm.pdf(x1, mean[1], sigma_firn), x1, color=color, linewidth=1.6)

        rv = stats.multivariate_normal(mean, cov)
        sigma_level = rv.pdf(mean) * np.exp(-0.5)
        X, Y = np.meshgrid(
            np.linspace(mean[0] - 3.75 * sigma_ice, mean[0] + 3.75 * sigma_ice, 120),
            np.linspace(mean[1] - 3.75 * sigma_firn, mean[1] + 3.75 * sigma_firn, 120),
        )
        Z = rv.pdf(np.dstack((X, Y)))
        ax_main.contour(X, Y, Z, levels=[sigma_level], colors=[color], linewidths=1.8)
        ax_main.plot(mean[0], mean[1], "+", color=color, markersize=8, mew=2)

    if true_values is not None:
        ax_top.axvline(true_values[0], color="black", linestyle="--", linewidth=1.5, label="True")
        ax_right.axhline(true_values[1], color="black", linestyle="--", linewidth=1.5)
        ax_main.plot(true_values[0], true_values[1], "kx", markersize=10, mew=2, label="True")

    ax_top.set_ylabel("Density")
    ax_top.set_xticklabels([])
    ax_top.set_yticklabels([])
    ax_right.set_xlabel("Density")
    ax_right.set_yticklabels([])
    ax_main.set_xlabel("Ice GMSL (mm)")
    ax_main.set_ylabel("Firn GMSL (mm)")

    handles, labels = ax_top.get_legend_handles_labels()
    more_handles, more_labels = ax_main.get_legend_handles_labels()
    handles += more_handles
    labels += more_labels
    ax_legend.legend(handles, labels, loc="center", fontsize=9, frameon=False)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_gmsl_bivariate_corners(
    results: dict[str, dict[str, object]],
    output_dir: Path,
    title_prefix: str,
    epoch: int = -1,
) -> None:
    for label, result in results.items():
        measure, true_values = gmsl_2d_posterior(result, epoch=epoch)
        fig, _ = plot_bivariate_corner(
            measure,
            true_values=true_values,
            labels=["Ice GMSL (mm)", "Firn GMSL (mm)"],
            title=f"{title_prefix} - {label}",
            figsize=(6.5, 6.5),
            pdf_colors=["tab:blue", "tab:blue"],
        )
        fig.savefig(
            output_dir / f"{label}_gmsl_corner.png",
            dpi=300,
        )
        plt.close(fig)
