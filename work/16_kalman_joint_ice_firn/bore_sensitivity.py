from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ExperimentConfig, ensure_output_dir, run_variant


BASE_CONFIG = ExperimentConfig(
    output_dir=Path(
        "work/16_kalman_joint_ice_firn/outputs/bore_sensitivity"
    ),
)
SEEDS = list(range(5))
BORE_COUNTS = [2, 4, 6, 8]
REVISIT_PROBABILITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def _evaluate_case(
    seed: int,
    n_bores: int,
    revisit_probability: float,
) -> dict[str, float]:
    config = replace(
        BASE_CONFIG,
        seed=seed,
        n_bores_per_epoch=n_bores,
        bore_revisit_probability=revisit_probability,
    )
    full = run_variant(
        config=config,
        name="full",
        include_ssh=True,
        include_ice=True,
        include_bores=True,
        include_grace=True,
    )
    no_bores = run_variant(
        config=config,
        name="no_bores",
        include_ssh=True,
        include_ice=True,
        include_bores=False,
        include_grace=True,
    )
    full_summary = full["summary"]
    no_bores_summary = no_bores["summary"]
    return {
        "seed": seed,
        "n_bores_per_epoch": n_bores,
        "revisit_probability": revisit_probability,
        "full_firn_mean_abs_z": float(
            full_summary["smoothed_firn_mean_abs_z"].mean()
        ),
        "no_bores_firn_mean_abs_z": float(
            no_bores_summary["smoothed_firn_mean_abs_z"].mean()
        ),
        "firn_mean_abs_z_gain": float(
            no_bores_summary["smoothed_firn_mean_abs_z"].mean()
            - full_summary["smoothed_firn_mean_abs_z"].mean()
        ),
        "ice_mean_abs_z_gain": float(
            no_bores_summary["smoothed_ice_mean_abs_z"].mean()
            - full_summary["smoothed_ice_mean_abs_z"].mean()
        ),
        "gmsl_abs_z_gain": float(
            no_bores_summary["smoothed_gmsl_abs_z"].mean()
            - full_summary["smoothed_gmsl_abs_z"].mean()
        ),
    }


def plot_heatmap(
    frame: pd.DataFrame,
    value: str,
    output_path: Path,
    title: str,
    cbar_label: str,
) -> None:
    pivot = frame.pivot(
        index="revisit_probability",
        columns="n_bores_per_epoch",
        values=value,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(
        pivot.to_numpy(),
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Bores per epoch")
    ax.set_ylabel("Revisit probability")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ensure_output_dir(BASE_CONFIG.output_dir)
    cases = [
        (seed, n_bores, revisit_probability)
        for seed in SEEDS
        for n_bores in BORE_COUNTS
        for revisit_probability in REVISIT_PROBABILITIES
    ]
    max_workers = min(int(os.environ.get("WORK16_MAX_WORKERS", "2")), len(cases))
    rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_evaluate_case, seed, n_bores, revisit_probability)
            for seed, n_bores, revisit_probability in cases
        ]
        for future in as_completed(futures):
            rows.append(future.result())

    frame = pd.DataFrame(rows)
    frame.to_csv(
        BASE_CONFIG.output_dir / "bore_sensitivity_seedwise.csv",
        index=False,
    )
    aggregate = (
        frame.groupby(
            ["n_bores_per_epoch", "revisit_probability"]
        )
        .agg(
            firn_mean_abs_z_gain=("firn_mean_abs_z_gain", "mean"),
            ice_mean_abs_z_gain=("ice_mean_abs_z_gain", "mean"),
            gmsl_abs_z_gain=(
                "gmsl_abs_z_gain",
                "mean",
            ),
        )
        .reset_index()
    )
    aggregate.to_csv(
        BASE_CONFIG.output_dir / "bore_sensitivity_aggregate.csv",
        index=False,
    )

    plot_heatmap(
        aggregate,
        value="firn_mean_abs_z_gain",
        output_path=(
            BASE_CONFIG.output_dir / "bore_sensitivity_firn_gain.png"
        ),
        title="Firn mean |z| gain from bores",
        cbar_label="No-bores minus full firn mean |z|",
    )
    plot_heatmap(
        aggregate,
        value="ice_mean_abs_z_gain",
        output_path=(
            BASE_CONFIG.output_dir / "bore_sensitivity_ice_gain.png"
        ),
        title="Ice mean |z| gain from bores",
        cbar_label="No-bores minus full ice mean |z|",
    )
    plot_heatmap(
        aggregate,
        value="gmsl_abs_z_gain",
        output_path=(
            BASE_CONFIG.output_dir / "bore_sensitivity_gmsl_gain.png"
        ),
        title="GMSL |z| gain from bores",
        cbar_label="No-bores minus full |GMSL z|",
    )
    print(aggregate.round(6))
    print("Saved outputs to", BASE_CONFIG.output_dir)


if __name__ == "__main__":
    main()
