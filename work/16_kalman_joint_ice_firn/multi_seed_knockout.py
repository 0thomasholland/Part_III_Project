from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ExperimentConfig, ensure_output_dir, run_variant
from parallel_utils import run_cases_in_pool


BASE_CONFIG = ExperimentConfig(
    output_dir=Path(
        "work/16_kalman_joint_ice_firn/outputs/multi_seed_knockout"
    ),
)
SEEDS = list(range(8))
VARIANTS = [
    ("full", True, True, True, True),
    ("no_bores", True, True, False, True),
    ("no_ice", True, False, True, True),
    ("no_ssh", False, True, True, True),
    ("no_grace", True, True, True, False),
]


def _evaluate_case(
    case: tuple[int, str, bool, bool, bool, bool],
) -> pd.DataFrame:
    seed, name, include_ssh, include_ice, include_bores, include_grace = case
    config = replace(BASE_CONFIG, seed=seed)
    result = run_variant(
        config=config,
        name=name,
        include_ssh=include_ssh,
        include_ice=include_ice,
        include_bores=include_bores,
        include_grace=include_grace,
    )
    summary = result["summary"].copy()
    summary["seed"] = seed
    return summary


def plot_aggregate_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped = (
        summary.groupby(["variant", "epoch"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    for variant, group in grouped.groupby("variant"):
        ax.plot(group["epoch"], group["mean"], label=variant)
        ax.fill_between(
            group["epoch"],
            group["mean"] - group["std"],
            group["mean"] + group["std"],
            alpha=0.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_seedwise_improvement(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    full = summary[summary["variant"] == "full"]
    no_bores = summary[summary["variant"] == "no_bores"]
    merged = full.merge(
        no_bores,
        on=["seed", "epoch"],
        suffixes=("_full", "_no_bores"),
    )
    merged["improvement"] = (
        merged[f"{metric}_no_bores"]
        - merged[f"{metric}_full"]
    )
    grouped = (
        merged.groupby("epoch")["improvement"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(grouped["epoch"], grouped["mean"])
    ax.fill_between(
        grouped["epoch"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    print(
        "Running multi_seed_knockout.py: repeat knockout study across multiple random seeds."
    )
    ensure_output_dir(BASE_CONFIG.output_dir)
    cases = [
        (seed, name, include_ssh, include_ice, include_bores, include_grace)
        for seed in SEEDS
        for name, include_ssh, include_ice, include_bores, include_grace in VARIANTS
    ]
    summaries = run_cases_in_pool(cases, _evaluate_case)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(
        BASE_CONFIG.output_dir / "multi_seed_knockout_summary.csv",
        index=False,
    )

    plot_aggregate_metric(
        summary,
        metric="smoothed_firn_mean_abs_z",
        output_path=(
            BASE_CONFIG.output_dir / "multi_seed_firn_mean_abs_z.png"
        ),
        title="Multi-seed smoothed firn mean |z|",
        ylabel="Firn mean |z|",
    )
    plot_aggregate_metric(
        summary,
        metric="smoothed_ice_mean_abs_z",
        output_path=(
            BASE_CONFIG.output_dir / "multi_seed_ice_mean_abs_z.png"
        ),
        title="Multi-seed smoothed ice mean |z|",
        ylabel="Ice mean |z|",
    )
    plot_aggregate_metric(
        summary,
        metric="smoothed_gmsl_abs_z",
        output_path=(
            BASE_CONFIG.output_dir / "multi_seed_gmsl_abs_z.png"
        ),
        title="Multi-seed smoothed |GMSL z|",
        ylabel="|GMSL z|",
    )
    plot_seedwise_improvement(
        summary,
        metric="smoothed_firn_mean_abs_z",
        output_path=(
            BASE_CONFIG.output_dir / "multi_seed_bore_firn_z_gain.png"
        ),
        title="Firn mean |z| gain from bores across seeds",
        ylabel="No-bores minus full firn mean |z|",
    )
    plot_seedwise_improvement(
        summary,
        metric="smoothed_ice_mean_abs_z",
        output_path=(
            BASE_CONFIG.output_dir / "multi_seed_bore_ice_z_gain.png"
        ),
        title="Ice mean |z| gain from bores",
        ylabel="No-bores minus full ice mean |z|",
    )

    aggregate = (
        summary.groupby("variant")
        .agg(
            mean_firn_mean_abs_z=("smoothed_firn_mean_abs_z", "mean"),
            std_firn_mean_abs_z=("smoothed_firn_mean_abs_z", "std"),
            mean_ice_mean_abs_z=("smoothed_ice_mean_abs_z", "mean"),
            mean_gmsl_abs_z=(
                "smoothed_gmsl_abs_z",
                "mean",
            ),
        )
        .reset_index()
    )
    aggregate.to_csv(
        BASE_CONFIG.output_dir / "multi_seed_variant_means.csv",
        index=False,
    )
    print("Saved multi-seed summary tables and aggregate comparison plots.")
    print(aggregate.round(6))
    print("Saved outputs to", BASE_CONFIG.output_dir)


if __name__ == "__main__":
    main()
