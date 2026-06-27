from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ExperimentConfig,
    build_setup,
    ensure_output_dir,
    generate_bore_schedule,
    plot_error_distributions,
    plot_bore_network,
    plot_gmsl_bivariate_corners,
    plot_gmsl_bivariate_overlay,
    plot_gmsl_zscore_timeseries,
    plot_uncertainty_reduction_maps,
    plot_variant_comparison,
    plot_variant_improvement,
    run_variant,
    sample_truth,
)


CONFIG = ExperimentConfig(
    output_dir=Path("work/16_kalman_joint_ice_firn/outputs/knockout"),
)

VARIANTS = [
    ("full", True, True, True, True),
    ("no_bores", True, True, False, True),
    ("no_ice", True, False, True, True),
    ("no_ssh", False, True, True, True),
    ("no_grace", True, True, True, False),
]


def main() -> None:
    ensure_output_dir(CONFIG.output_dir)
    setup = build_setup(CONFIG)
    rng = np.random.default_rng(CONFIG.seed)
    truth_states = sample_truth(setup, rng)
    bore_schedule = generate_bore_schedule(
        setup.bore_candidate_coords,
        CONFIG.n_epochs,
        CONFIG.n_bores_per_epoch,
        CONFIG.bore_revisit_probability,
        rng,
    )

    summaries = []
    results = {}
    for name, include_ssh, include_ice, include_bores, include_grace in VARIANTS:
        result = run_variant(
            config=CONFIG,
            name=name,
            include_ssh=include_ssh,
            include_ice=include_ice,
            include_bores=include_bores,
            include_grace=include_grace,
            truth_states=truth_states,
            bore_schedule=bore_schedule,
            setup=setup,
        )
        results[name] = result
        summaries.append(result["summary"])

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(CONFIG.output_dir / "knockout_summary.csv", index=False)

    plot_variant_comparison(
        summary,
        metric="smoothed_firn_mean_abs_z",
        output_path=CONFIG.output_dir / "knockout_firn_mean_abs_z.png",
        title="Knockout comparison: smoothed firn mean |z|",
        ylabel="Firn mean |z|",
    )
    plot_variant_comparison(
        summary,
        metric="smoothed_ice_mean_abs_z",
        output_path=CONFIG.output_dir / "knockout_ice_mean_abs_z.png",
        title="Knockout comparison: smoothed ice mean |z|",
        ylabel="Ice mean |z|",
    )
    plot_variant_comparison(
        summary,
        metric="smoothed_gmsl_abs_z",
        output_path=CONFIG.output_dir / "knockout_gmsl_abs_z.png",
        title="Knockout comparison: smoothed |GMSL z|",
        ylabel="|GMSL z|",
    )
    plot_variant_improvement(
        summary,
        reference_variant="full",
        comparison_variant="no_bores",
        metric="smoothed_firn_mean_abs_z",
        output_path=(
            CONFIG.output_dir / "knockout_bore_firn_z_improvement.png"
        ),
        title="Firn z-score penalty without bore constraints",
        ylabel="No-bores minus full firn mean |z|",
    )
    plot_variant_improvement(
        summary,
        reference_variant="full",
        comparison_variant="no_grace",
        metric="smoothed_gmsl_abs_z",
        output_path=(
            CONFIG.output_dir / "knockout_grace_gmsl_z_improvement.png"
        ),
        title="GMSL z-score penalty without GRACE",
        ylabel="No-GRACE minus full |GMSL z|",
    )
    plot_bore_network(
        bore_schedule,
        CONFIG.output_dir / "knockout_bore_network.png",
        title="Bore network used in knockout runs",
    )
    plot_gmsl_zscore_timeseries(
        results["full"]["summary"],
        CONFIG.output_dir / "knockout_full_gmsl_zscore.png",
        title="Full-observation GMSL z-score",
    )
    plot_error_distributions(
        results["full"]["diagnostics"],
        CONFIG.output_dir / "knockout_full_error_distributions.png",
        title="Full-observation posterior error and z-score distributions",
    )
    plot_uncertainty_reduction_maps(
        setup,
        results["full"]["solution"]["smoothed_std_vectors"],
        results["no_bores"]["solution"]["smoothed_std_vectors"],
        CONFIG.output_dir / "knockout_firn_std_reduction.png",
        component="firn",
        title="Firn uncertainty reduction from bore constraints",
    )
    plot_uncertainty_reduction_maps(
        setup,
        results["full"]["solution"]["smoothed_std_vectors"],
        results["no_bores"]["solution"]["smoothed_std_vectors"],
        CONFIG.output_dir / "knockout_ice_std_reduction.png",
        component="ice",
        title="Ice uncertainty reduction from bore constraints",
    )
    plot_uncertainty_reduction_maps(
        setup,
        results["full"]["solution"]["smoothed_std_vectors"],
        results["no_grace"]["solution"]["smoothed_std_vectors"],
        CONFIG.output_dir / "knockout_grace_firn_std_reduction.png",
        component="firn",
        title="Firn uncertainty reduction from GRACE constraints",
    )
    plot_gmsl_bivariate_overlay(
        results,
        CONFIG.output_dir / "knockout_gmsl_bivariate_overlay.png",
        title="Knockout sensitivity: ice vs firn GMSL posterior covariance",
    )
    plot_gmsl_bivariate_corners(
        results,
        CONFIG.output_dir,
        title_prefix="Knockout ice vs firn GMSL posterior",
    )

    pivot = summary.groupby("variant")[
        [
            "smoothed_ice_mean_abs_z",
            "smoothed_firn_mean_abs_z",
            "smoothed_gmsl_abs_z",
            "mean_filtered_rank",
            "mean_filtered_variance_fraction",
        ]
    ].mean()
    print(summary.round(4))
    print("\nVariant means:\n")
    print(pivot.round(4))
    print("\nSaved outputs to", CONFIG.output_dir)


if __name__ == "__main__":
    main()
