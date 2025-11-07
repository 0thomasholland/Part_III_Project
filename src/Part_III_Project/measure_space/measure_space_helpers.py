import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygeoinf.gaussian_measure import GaussianMeasure
from statsistics import norm


def get_stats_from_measure(
    measure: GaussianMeasure,
) -> tuple[float, float]:
    expectation = measure.expectation[0]
    variance = measure.covariance.matrix(dense=True)[0, 0]
    return expectation, variance


def plot_measure(
    measures: GaussianMeasure | list[GaussianMeasure],
) -> tuple[plt.Figure, plt.Axes]:
    # returns figures, axes
    if not isinstance(measures, list):
        measures = [measures]
    stats = pd.DataFrame(
        columns=["Measure", "Expectation", "Variance"],
    )
    for m in measures:
        expectation, variance = get_stats_from_measure(m)
        stats = pd.concat(
            [
                stats,
                pd.DataFrame(
                    {
                        "Measure": [str(m)],
                        "Expectation": [expectation],
                        "Variance": [variance],
                    },
                ),
            ],
            ignore_index=True,
        )

    # plot the normal distributions for each case on the same axis, using the max and min of all stds to set x limits
    x_min = min(
        stats["Expectation"] - 4 * np.sqrt(stats["Variance"]),
    )
    x_max = max(
        stats["Expectation"] + 4 * np.sqrt(stats["Variance"]),
    )
    x = np.linspace(x_min, x_max, 1000)
    fig, ax = plt.subplots()
    for _, row in stats.iterrows():
        ax.plot(
            x,
            norm.pdf(
                x,
                row["Expectation"],
                np.sqrt(row["Variance"]),
            ),
            label=row["Measure"],
        )
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend()
    return fig, ax
