import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project import colors

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
MASTER_RESULTS_WIDE_PATH = (
    Path(__file__).resolve().parent
    / "master_results_wide.csv"
)


def _require_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError(
            "master_results_wide.csv has no rows."
        )


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".pdf")
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def _parse_sweep_value_token(token: str) -> float:
    return float(token.replace("m", "-").replace("p", "."))


def _wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    case_cols = [
        col for col in df_wide.columns if "__" in col
    ]
    id_cols = [
        col for col in df_wide.columns if "__" not in col
    ]

    records: list[dict] = []
    for _, row in df_wide.iterrows():
        base = {col: row[col] for col in id_cols}
        cases: dict[tuple[str, str], dict] = {}

        for col in case_cols:
            metric, sweep_type, value_token = col.split(
                "__", 2
            )
            key = (sweep_type, value_token)
            if key not in cases:
                cases[key] = {}
            cases[key][metric] = row[col]

        for (
            sweep_type,
            value_token,
        ), metrics in cases.items():
            record = dict(base)
            record.update(metrics)
            record["sweep_type"] = sweep_type
            record["sweep_value"] = (
                _parse_sweep_value_token(value_token)
            )
            records.append(record)

    return pd.DataFrame(records)


def _ordered_sweep_types(df: pd.DataFrame) -> list[str]:
    preferred = [
        "std_multiplier",
        "mean_offset",
        "length_scale",
    ]
    available = set(df["sweep_type"].unique())
    ordered = [s for s in preferred if s in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def _accurate_prior_z_scores_by_setup(
    df: pd.DataFrame,
) -> pd.Series:
    # Accurate prior: zero-mean and truth std.
    setup = (
        df[
            [
                "setup_index",
                "gmsl_true_mm",
                "truth_gmsl_std_nd",
            ]
        ]
        .drop_duplicates(subset=["setup_index"])
        .copy()
        .set_index("setup_index")
    )
    std_mm = setup["truth_gmsl_std_nd"] * 1000.0
    return (0.0 - setup["gmsl_true_mm"]) / std_mm


def plot_true_vs_case_z_kde_grid(
    df: pd.DataFrame,
    *,
    case_z_col: str = "prior_z",
    output_name: str = "kde_grid_true_vs_case_z.pdf",
    title: str = "KDE Grid: Accurate Prior z vs Tweaked-Parameter z",
) -> None:
    z_true_by_setup = _accurate_prior_z_scores_by_setup(df)

    plot_df = df.copy()
    plot_df["true_prior_z"] = plot_df["setup_index"].map(
        z_true_by_setup
    )
    plot_df = plot_df.dropna(
        subset=["true_prior_z", case_z_col]
    )
    if plot_df.empty:
        return

    sweep_order = _ordered_sweep_types(plot_df)
    cases: list[tuple[str, float]] = []
    for sweep_type in sweep_order:
        values = sorted(
            plot_df.loc[
                plot_df["sweep_type"] == sweep_type,
                "sweep_value",
            ].unique()
        )
        for value in values:
            cases.append((sweep_type, float(value)))

    if not cases:
        return

    ncols = 4
    nrows = math.ceil(len(cases) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    combined = pd.concat(
        [plot_df["true_prior_z"], plot_df[case_z_col]],
        ignore_index=True,
    )
    lim_low = float(combined.min())
    lim_high = float(combined.max())
    pad = (
        (lim_high - lim_low) * 0.08
        if lim_high > lim_low
        else 1.0
    )
    lim_low -= pad
    lim_high += pad

    for idx, (sweep_type, sweep_value) in enumerate(cases):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        subset = plot_df[
            (plot_df["sweep_type"] == sweep_type)
            & (plot_df["sweep_value"] == sweep_value)
        ]

        if len(subset) >= 3:
            sns.kdeplot(
                data=subset,
                x="true_prior_z",
                y=case_z_col,
                fill=True,
                thresh=0.05,
                levels=12,
                cmap="mako",
                bw_adjust=1.0,
                ax=ax,
            )

        sns.scatterplot(
            data=subset,
            x="true_prior_z",
            y=case_z_col,
            s=14,
            color="black",
            alpha=0.45,
            ax=ax,
            legend=False,
        )

        ax.plot(
            [lim_low, lim_high],
            [lim_low, lim_high],
            linestyle="--",
            color="black",
            linewidth=0.8,
            alpha=0.8,
        )
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_aspect("equal", "box")
        ax.set_title(
            f"{sweep_type}={sweep_value:g}",
            fontsize=9,
        )
        ax.set_xlabel("Accurate prior z-score")
        ax.set_ylabel("Tweaked-parameter z-score")

    total_axes = nrows * ncols
    for idx in range(len(cases), total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.suptitle(
        title,
        y=0.995,
    )
    fig.tight_layout()
    _save_figure(
        fig,
        FIGURES_DIR / output_name,
    )


def plot_grouped_ridge_kde(
    df: pd.DataFrame,
    *,
    metric_col: str = "posterior_bias_mm",
) -> None:
    sweep_order = _ordered_sweep_types(df)
    if not sweep_order:
        return

    intra_gap = 0.70
    group_gap = 1.45
    ridge_height = 0.85

    rows: list[dict] = []
    current_y = 0.0
    for sweep_type in sweep_order:
        values = sorted(
            df.loc[
                df["sweep_type"] == sweep_type,
                "sweep_value",
            ].unique()
        )
        for value in values:
            rows.append(
                {
                    "sweep_type": sweep_type,
                    "sweep_value": value,
                    "base_y": current_y,
                }
            )
            current_y += intra_gap
        current_y += group_gap

    if not rows:
        return

    x_data = df[metric_col].dropna()
    x_min = float(x_data.min())
    x_max = float(x_data.max())
    x_pad = (x_max - x_min) * 0.08 if x_max > x_min else 1.0

    fig, ax = plt.subplots(figsize=(10, 7))
    palette = sns.color_palette("crest", n_colors=len(rows))

    for idx, row in enumerate(rows):
        subset = df[
            (df["sweep_type"] == row["sweep_type"])
            & (df["sweep_value"] == row["sweep_value"])
        ][metric_col].dropna()

        if subset.empty:
            continue

        sns.kdeplot(
            x=subset,
            ax=ax,
            bw_adjust=1.0,
            fill=False,
            linewidth=1.0,
            color=palette[idx],
            clip=(x_min - x_pad, x_max + x_pad),
        )
        line = ax.lines[-1]
        x_vals, y_vals = line.get_data()
        line.remove()

        if len(y_vals) == 0:
            continue

        y_peak = float(max(y_vals))
        if y_peak <= 0:
            continue

        y_norm = (y_vals / y_peak) * ridge_height
        y_shifted = y_norm + row["base_y"]

        ax.fill_between(
            x_vals,
            row["base_y"],
            y_shifted,
            color=palette[idx],
            alpha=0.85,
            linewidth=0,
        )
        ax.plot(
            x_vals,
            y_shifted,
            color="black",
            linewidth=0.7,
        )

        label = (
            f"{row['sweep_type']}={row['sweep_value']:g}"
        )
        ax.text(
            x_min - x_pad * 0.4,
            row["base_y"] + 0.08,
            label,
            ha="right",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )

    for sweep_type in sweep_order[:-1]:
        group_rows = [
            r for r in rows if r["sweep_type"] == sweep_type
        ]
        group_end = (
            group_rows[-1]["base_y"] + intra_gap * 0.9
        )
        ax.axhline(
            group_end,
            color="black",
            alpha=0.2,
            linewidth=0.7,
        )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        color="black",
        alpha=0.6,
    )
    ax.set_xlim(x_min - x_pad * 0.6, x_max + x_pad)
    ax.set_ylim(-0.2, current_y - group_gap + ridge_height)
    ax.set_yticks([])
    ax.set_xlabel("Posterior bias (mm)")
    ax.set_ylabel("")
    ax.set_title("Ridge KDE Across All Sensitivity Cases")

    accurate_prior_z = _accurate_prior_z_scores_by_setup(df)
    ax.text(
        0.5,
        1.03,
        (
            "Accurate prior z-score "
            f"(mean={accurate_prior_z.mean():.3f}, "
            f"median={accurate_prior_z.median():.3f})"
        ),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    _save_figure(
        fig,
        FIGURES_DIR / "ridge_kde_all.pdf",
    )


def main() -> None:
    master_path = MASTER_RESULTS_WIDE_PATH
    if not master_path.exists():
        raise FileNotFoundError(
            f"Missing {master_path}. Run data_globber.py first."
        )

    df = _wide_to_long(pd.read_csv(master_path))
    _require_data(df)

    plot_grouped_ridge_kde(df)
    plot_true_vs_case_z_kde_grid(
        df,
        case_z_col="prior_z",
        output_name="kde_grid_accurate_prior_vs_tweaked_z.pdf",
        title="KDE Grid: Accurate Prior z vs Tweaked-Parameter z",
    )


if __name__ == "__main__":
    main()
