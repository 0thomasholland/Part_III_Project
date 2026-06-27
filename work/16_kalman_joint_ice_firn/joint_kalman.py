from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from common import (
    ExperimentConfig,
    ensure_output_dir,
    plot_bore_network,
    plot_error_distributions,
    plot_gmsl_timeseries,
    plot_gmsl_zscore_timeseries,
    plot_rank_timeseries,
    plot_state_uncertainty_timeseries,
    plot_zscore_timeseries,
    run_variant,
    vector_to_grids,
)
from pyslfp_extras import plot


CONFIG = ExperimentConfig(
    output_dir=Path("work/16_kalman_joint_ice_firn/outputs/joint_kalman"),
)


def plot_firn_snapshots(result: dict[str, object], output_path: Path) -> None:
    setup = result["setup"]
    truth_states = result["truth_states"]
    filtered_means = result["solution"]["filtered_means"]
    smoothed_means = result["solution"]["smoothed_means"]
    epochs = [0, len(truth_states) // 2, len(truth_states) - 1]

    scale = 1000.0 * setup.fp.model.parameters.length_scale
    ice_mask = setup.fp.ice_projection()

    truth_grids = []
    filtered_grids = []
    smoothed_grids = []
    error_grids = []
    for epoch in epochs:
        _, truth_firn = vector_to_grids(setup, truth_states[epoch])
        _, filter_firn = vector_to_grids(
            setup, filtered_means[epoch]
        )
        _, smooth_firn = vector_to_grids(
            setup, smoothed_means[epoch]
        )
        truth_grids.append(scale * truth_firn * ice_mask)
        filtered_grids.append(scale * filter_firn * ice_mask)
        smoothed_grids.append(scale * smooth_firn * ice_mask)
        error_grids.append(
            scale * (smooth_firn - truth_firn) * ice_mask
        )

    field_vmax = max(
        np.nanmax(np.abs(grid.data))
        for grid in truth_grids + filtered_grids + smoothed_grids
    )
    error_vmax = max(
        np.nanmax(np.abs(grid.data)) for grid in error_grids
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(13, 9),
        constrained_layout=True,
        subplot_kw={"projection": ccrs.Robinson()},
    )
    column_titles = [
        "True firn change",
        "Filtered firn change",
        "Smoothed firn change",
        "Smoothed error",
    ]
    for row, epoch in enumerate(epochs):
        field_image = plot(
            truth_grids[row],
            ax=axes[row, 0],
            cmap="seismic",
            vmin=-field_vmax,
            vmax=field_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )[2]
        plot(
            filtered_grids[row],
            ax=axes[row, 1],
            cmap="seismic",
            vmin=-field_vmax,
            vmax=field_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )
        plot(
            smoothed_grids[row],
            ax=axes[row, 2],
            cmap="seismic",
            vmin=-field_vmax,
            vmax=field_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )
        error_image = plot(
            error_grids[row],
            ax=axes[row, 3],
            cmap="seismic",
            vmin=-error_vmax,
            vmax=error_vmax,
            colorbar=False,
            coasts=True,
            gridlines=True,
            tight_layout=False,
        )[2]

        for col, axis in enumerate(axes[row]):
            if row == 0:
                axis.set_title(column_titles[col], fontsize=10)
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

    fig.colorbar(
        field_image,
        ax=axes[:, :3],
        shrink=0.85,
        label="Firn thickness change (mm)",
    )
    fig.colorbar(
        error_image,
        ax=axes[:, 3],
        shrink=0.85,
        label="Firn error (mm)",
    )

    fig.suptitle(
        "Joint Kalman firn snapshots: true, filtered, smoothed, and error"
    )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    print(
        "Running joint_kalman.py: full-observation baseline with filter and smoother outputs."
    )
    ensure_output_dir(CONFIG.output_dir)
    result = run_variant(
        config=CONFIG,
        name="full",
        include_ssh=True,
        include_ice=True,
        include_bores=True,
        include_grace=True,
    )
    summary = result["summary"]
    summary.to_csv(CONFIG.output_dir / "joint_kalman_summary.csv", index=False)

    plot_zscore_timeseries(
        summary,
        CONFIG.output_dir / "joint_kalman_zscore.png",
        title="Joint posterior field z-score diagnostics",
    )
    plot_gmsl_timeseries(
        summary,
        CONFIG.output_dir / "joint_kalman_gmsl.png",
        title="Kalman filter/smoother total GMSL",
    )
    plot_gmsl_zscore_timeseries(
        summary,
        CONFIG.output_dir / "joint_kalman_gmsl_zscore.png",
        title="Joint posterior GMSL z-score diagnostics",
    )
    plot_state_uncertainty_timeseries(
        summary,
        CONFIG.output_dir / "joint_kalman_state_uncertainty.png",
        title="State uncertainty diagnostics",
    )
    plot_rank_timeseries(
        summary,
        CONFIG.output_dir / "joint_kalman_rank.png",
        title="Truncation rank diagnostics",
    )
    plot_bore_network(
        result["bore_schedule"],
        CONFIG.output_dir / "joint_kalman_bore_network.png",
        title="Synthetic bore network geometry",
    )
    plot_firn_snapshots(
        result,
        CONFIG.output_dir / "joint_kalman_firn_snapshots.png",
    )
    plot_error_distributions(
        result["diagnostics"],
        CONFIG.output_dir / "joint_kalman_error_distributions.png",
        title="Joint posterior error and z-score distributions",
    )

    print("Saved baseline summary and diagnostic plots.")
    print(summary.round(4))
    print(
        "Saved outputs to",
        CONFIG.output_dir,
    )


if __name__ == "__main__":
    main()
