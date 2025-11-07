import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygeoinf.gaussian_measure import GaussianMeasure
from scipy.stats import norm


def get_stats_from_measure(
    measure: GaussianMeasure,
) -> tuple[float, float]:
    expectation = measure.expectation[0]
    variance = measure.covariance.matrix(dense=True)[0, 0]
    return expectation, variance


def plot_measure(
    measures: GaussianMeasure | list[GaussianMeasure],
    names: str | list[str] = [None],
    args: dict = {},
) -> tuple[plt.Figure, plt.Axes]:
    # returns figures, axes
    if not isinstance(measures, list):
        measures = [measures]
    if not isinstance(names, list):
        names = [names]
    expectations = []
    variances = []
    for m in measures:
        expectation, variance = get_stats_from_measure(m)
        expectations.append(expectation)
        variances.append(variance)

    # plot the normal distributions for each case on the same axis, using the max and min of all stds to set x limits
    x_min = min(
        expectations - 4 * np.sqrt(variances),
    )
    x_max = max(
        expectations + 4 * np.sqrt(variances),
    )
    x = np.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, mu, sigma in zip(
        names,
        expectations,
        np.sqrt(variances),
    ):
        y = norm.pdf(x, mu, sigma)
        ax.plot(
            x,
            y,
            label=f"{name} (μ={mu:.2e}, σ={sigma:.2e})"
            if name is not None
            else f"μ={mu:.2e}, σ={sigma:.2e}",
        )
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(
            1.01,
            1.0,
        ),
    )

    # plotting arguments used
    text_lines = []
    for key, value in args.items():
        # Format the value to be a string, using scientific notation for small numbers
        formatted_value = (
            f"{value:.2e}"
            if abs(value) < 0.01 and value != 0
            else str(value)
        )
        text_lines.append(f"{key}: {formatted_value} m")

    # Combine all lines with newlines
    arg_string = "\n".join(text_lines)

    fig.text(
        x=1.01,  # Move the text box well to the right (increase this value if needed)
        y=0.65,
        s="Input Parameters:\n\n" + arg_string,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=9,
        # Optional: Add a bounding box for clarity
        bbox={"facecolor": "lightyellow", "alpha": 0.7, "pad": 5},
    )
    fig.tight_layout(rect=[0, 0, 1.4, 1])
    return fig, ax
