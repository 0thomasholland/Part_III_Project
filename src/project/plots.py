import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots
from pygeoinf import GaussianMeasure


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
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, (ax1, ax2) = subplots(1, 2, figsize=figsize)

    _error_measure: GuassianMeasure = (
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
            label=f"Error Expectation ({1000 * error_mean:.1f} mm)",
        )
    ax2.set_title(ax2_title)
    ax2.set_xlabel(ax2_xlabel)

    fig.suptitle(suptitle)

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
) -> tuple[Figure, tuple[Axes, Axes]]:
    fig, (ax1, ax2) = subplots(1, 2, figsize=(16, 6))

    true_mean = np.array(true_mean)
    true_std = np.array(true_std)
    estimate_mean = np.array(estimate_mean)
    estimate_std = np.array(estimate_std)
    error_mean = np.array(error_mean)
    error_std = np.array(error_std)

    # Left plot: True and Estimated GMSL

    ax1.plot(
        latitude,
        np.full_like(latitude, true_mean),
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
    ax2.legend()

    fig.suptitle(suptitle)

    return fig, (ax1, ax2)
