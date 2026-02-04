import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from pygeoinf import GaussianMeasure


def error_plot_from_metrics(
    true_measure: tuple[float, float],
    estimation_measure: tuple[float, float],
    error_measure: tuple[float, float],
    show_bias: bool = True,
    figsize: tuple[int, int] = (12, 6),
    true_label: str = "True Distribution",
    est_label: str = "Estimated Distribution",
    error_label: str = "Error Distribution",
    true_color: str = "blue",
    est_color: str = "orange",
    ax1_title: str = "",
    ax1_xlabel: str = "Value",
    ax2_title: str = "Error Distribution",
    ax2_xlabel: str = "Error",
    suptitle: str = "Comparison of True, Estimated, and Error Distributions",
    ax: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    # If axes provided, use them; otherwise create new figure
    if ax is None:
        fig, (ax1, ax2) = subplots(1, 2, figsize=figsize)
        fig.suptitle(suptitle)
    else:
        ax1, ax2 = ax
        fig = ax1.get_figure()

    true_mean = true_measure[0]
    true_std = true_measure[1]
    est_mean = estimation_measure[0]
    est_std = estimation_measure[1]
    error_mean = error_measure[0]
    error_std = error_measure[1]

    # calculate the x_1 axis via the most negative 4std of each and the most positive 4std of each
    x_1 = np.linspace(
        min(
            true_mean - 4 * true_std,
            est_mean - 4 * est_std,
        ),
        max(
            true_mean + 4 * true_std,
            est_mean + 4 * est_std,
        ),
        1000,
    )

    true_pdf = (
        1 / (true_std * (2 * np.pi) ** 0.5)
    ) * np.exp(-0.5 * ((x_1 - true_mean) / true_std) ** 2)
    est_pdf = (1 / (est_std * (2 * np.pi) ** 0.5)) * np.exp(
        -0.5 * ((x_1 - est_mean) / est_std) ** 2
    )

    ax1.plot(
        x_1, true_pdf, label=true_label, color=true_color
    )
    ax1.plot(x_1, est_pdf, label=est_label, color=est_color)
    ax1.set_title(ax1_title)
    ax1.set_xlabel(ax1_xlabel)

    x_2 = np.linspace(
        error_mean - 4 * error_std,
        error_mean + 4 * error_std,
        1000,
    )
    error_pdf = (
        1 / (error_std * (2 * np.pi) ** 0.5)
    ) * np.exp(-0.5 * ((x_2 - error_mean) / error_std) ** 2)
    ax2.plot(x_2, error_pdf, label="Error", color="red")
    if show_bias:
        ax2.axvline(
            error_mean,
            color="black",
            linestyle="--",
            label=f"Error Expectation ({error_mean:.4f})",
        )
    ax2.set_title(ax2_title)
    ax2.set_xlabel(ax2_xlabel)

    return fig, (ax1, ax2)


def error_plot(
    true_measure: GaussianMeasure,
    estimation_measure: GaussianMeasure,
    show_bias: bool = True,
    figsize: tuple[int, int] = (12, 6),
    true_label: str = "True Distribution",
    est_label: str = "Estimated Distribution",
    error_label: str = "Error Distribution",
    true_color: str = "blue",
    est_color: str = "orange",
    ax1_title: str = "",
    ax1_xlabel: str = "Value",
    ax2_title: str = "Error Distribution",
    ax2_xlabel: str = "Error",
    suptitle: str = "Comparison of True, Estimated, and Error Distributions",
    ax: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    _error_measure: GaussianMeasure = (
        estimation_measure - true_measure
    )

    true_mean = true_measure.expectation[0]
    true_std = (
        (true_measure.covariance.matrix(dense=True)[0, 0])
        ** 0.5
    )
    est_mean = estimation_measure.expectation[0]
    est_std = (
        (
            estimation_measure.covariance.matrix(
                dense=True
            )[0, 0]
        )
        ** 0.5
    )
    error_mean = _error_measure.expectation[0]
    error_std = (
        (_error_measure.covariance.matrix(dense=True)[0, 0])
        ** 0.5
    )

    fig, (ax1, ax2) = error_plot_from_metrics(
        (true_mean, true_std),
        (est_mean, est_std),
        (error_mean, error_std),
        show_bias,
        figsize,
        true_label,
        est_label,
        error_label,
        true_color,
        est_color,
        ax1_title,
        ax1_xlabel,
        ax2_title,
        ax2_xlabel,
        suptitle,
        ax,
    )

    return fig, (ax1, ax2)


def error_latitude_plot(
    latitude: list[float] | np.ndarray,
    true_mean: list[float] | np.ndarray,
    true_std: list[float] | np.ndarray,
    estimate_mean: list[float] | np.ndarray,
    estimate_std: list[float] | np.ndarray,
    error_mean: list[float] | np.ndarray,
    error_std: list[float] | np.ndarray,
    show_bias: bool = True,
    figsize: tuple[int, int] = (12, 6),
    true_label: str = "True Distribution",
    estimate_label: str = "Estimated Distribution",
    error_label: str = "Error Distribution",
    true_color: str = "tab:blue",
    estimate_color: str = "tab:orange",
    ax1_title: str = "",
    ax1_ylabel: str = "Value",
    ax2_title: str = "Error Distribution",
    ax2_ylabel: str = "Error",
    suptitle: str = "Comparison of True, Estimated, and Error Distributions across latitudes",
    error_100_value: float | None = None,
    error_100_value_name: str | None = None,
    ax: tuple[Axes, Axes] | None = None,
) -> tuple[Figure, tuple[Axes, Axes]]:
    # If axes provided, use them; otherwise create new figure
    if ax is None:
        fig, (ax1, ax2) = subplots(1, 2, figsize=(16, 6))
        fig.suptitle(suptitle)
    else:
        ax1, ax2 = ax
        fig = ax1.get_figure()

    # Left plot: True and Estimated GMSL
    ax1.plot(
        latitude,
        true_mean,
        label=true_label,
        color=true_color,
    )
    ax1.fill_between(
        latitude,
        (true_mean - 2 * true_std),
        (true_mean + 2 * true_std),
        color=true_color,
        alpha=0.3,
        label="±2 Std Dev",
    )
    ax1.plot(
        latitude,
        estimate_mean,
        label=estimate_label,
        color=estimate_color,
    )
    ax1.fill_between(
        latitude,
        estimate_mean - 2 * estimate_std,
        estimate_mean + 2 * estimate_std,
        color=estimate_color,
        alpha=0.3,
        label="±2 Std Dev",
    )
    ax1.set_xlabel("Latitude (˚)")
    ax1.set_ylabel(ax1_ylabel)
    ax1.set_title(ax1_title)
    ax1.legend()

    # Right plot: Estimation Error
    ax2.plot(
        latitude,
        error_mean,
        label=error_label,
        color="tab:red",
    )
    ax2.fill_between(
        latitude,
        error_mean - 2 * error_std,
        error_mean + 2 * error_std,
        color="tab:red",
        alpha=0.3,
        label="±2 Std Dev",
    )
    ax2.set_xlabel("Latitude (˚)")
    ax2.set_ylabel(ax2_ylabel)
    ax2.set_title(ax2_title)
    # if error_100_value is not None, add a second y axis on the right of the plot that is our error value as a percentage of the error_100_value
    # using matplotlib.axes.Axes.secondary_xaxis

    if error_100_value is not None:
        ax2_sec = ax2.secondary_yaxis(
            "right",
            functions=(
                lambda x: (x / error_100_value) * 100,
                lambda x: (x / 100) * error_100_value,
            ),
        )
        if error_100_value_name is None:
            ax2_sec.set_ylabel(
                "Error as % of Reference Value"
            )
        else:
            ax2_sec.set_ylabel(
                f"Error as % of {error_100_value_name}"
            )

    ax2.legend()

    return fig, (ax1, ax2)


def double_distribution_plot(
    latitude: list[float] | np.ndarray,
    true_mean: list[float] | np.ndarray,
    true_std: list[float] | np.ndarray,
    estimate_mean: list[float] | np.ndarray,
    estimate_std: list[float] | np.ndarray,
    error_mean: list[float] | np.ndarray,
    error_std: list[float] | np.ndarray,
    show_bias: bool = False,
    figsize: tuple[int, int] = (16, 18),
    sample_values: tuple[float, float] = (np.nan, np.nan),
    true_label: str = "True Distribution",
    estimate_label: str = "Estimated Distribution",
    error_label: str = "Error Distribution",
    true_color: str = "tab:blue",
    estimate_color: str = "tab:orange",
    ax1_title: str = "",
    ax1_ylabel: str = "Value",
    ax2_title: str = "Error Distribution",
    ax2_ylabel: str = "Error",
    suptitle: str = "Comparison of True, Estimated, and Error Distributions across latitudes",
) -> tuple[
    Figure, tuple[Axes, Axes, Axes, Axes, Axes, Axes]
]:
    fig, axes = subplots(3, 2, figsize=figsize)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    fig.suptitle(suptitle)

    # first and second axes are from error_latitude_plot
    error_latitude_plot(
        latitude,
        true_mean,
        true_std,
        estimate_mean,
        estimate_std,
        error_mean,
        error_std,
        show_bias,
        figsize,
        true_label,
        estimate_label,
        error_label,
        true_color,
        estimate_color,
        ax1_title,
        ax1_ylabel,
        ax2_title,
        ax2_ylabel,
        ax=(ax1, ax2),
    )

    # third and fourth axes are the distributions at sample_values[0]
    # Find the nearest latitude to sample_values[0]
    latitude_arr = np.array(latitude)
    idx_0 = np.argmin(
        np.abs(latitude_arr - sample_values[0])
    )
    actual_lat_0 = latitude_arr[idx_0]

    true_mean_sample = true_mean[idx_0]
    true_std_sample = true_std[idx_0]
    estimate_mean_sample = estimate_mean[idx_0]
    estimate_std_sample = estimate_std[idx_0]
    error_mean_sample = error_mean[idx_0]
    error_std_sample = error_std[idx_0]

    error_plot_from_metrics(
        (true_mean_sample, true_std_sample),
        (estimate_mean_sample, estimate_std_sample),
        (error_mean_sample, error_std_sample),
        show_bias,
        figsize,
        true_label,
        estimate_label,
        error_label,
        true_color,
        estimate_color,
        f"Distributions at Latitude {actual_lat_0:.1f}˚",
        "Value",
        f"Error Distribution at Latitude {actual_lat_0:.1f}˚",
        "Error",
        ax=(ax3, ax4),
    )

    # fifth and sixth axes are the distributions at sample_values[1]
    # Find the nearest latitude to sample_values[1]
    idx_1 = np.argmin(
        np.abs(latitude_arr - sample_values[1])
    )
    actual_lat_1 = latitude_arr[idx_1]

    true_mean_sample = true_mean[idx_1]
    true_std_sample = true_std[idx_1]
    estimate_mean_sample = estimate_mean[idx_1]
    estimate_std_sample = estimate_std[idx_1]
    error_mean_sample = error_mean[idx_1]
    error_std_sample = error_std[idx_1]

    error_plot_from_metrics(
        (true_mean_sample, true_std_sample),
        (estimate_mean_sample, estimate_std_sample),
        (error_mean_sample, error_std_sample),
        show_bias,
        figsize,
        true_label,
        estimate_label,
        error_label,
        true_color,
        estimate_color,
        f"Distributions at Latitude {actual_lat_1:.1f}˚",
        "Value",
        f"Error Distribution at Latitude {actual_lat_1:.1f}˚",
        "Error",
        ax=(ax5, ax6),
    )

    # add vertical lines on the first two axes for the sample values

    ax1.axvline(
        sample_values[0],
        color="black",
        linestyle="--",
        label=f"Sample Latitude {sample_values[0]}˚",
    )
    ax1.axvline(
        sample_values[1],
        color="black",
        linestyle="--",
        label=f"Sample Latitude {sample_values[1]}˚",
    )
    ax2.axvline(
        sample_values[0],
        color="black",
        linestyle="--",
        label=f"Sample Latitude {sample_values[0]}˚",
    )
    ax2.axvline(
        sample_values[1],
        color="black",
        linestyle="--",
        label=f"Sample Latitude {sample_values[1]}˚",
    )

    return fig, (ax1, ax2, ax3, ax4, ax5, ax6)
