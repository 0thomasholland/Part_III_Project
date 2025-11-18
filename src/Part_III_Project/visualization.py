"""Visualization utilities for Part_III_Project.

This module provides plotting functions for Gaussian measures and
probability distributions related to sea level change analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from pygeoinf import GaussianMeasure
from pyslfp import FingerPrint
from scipy.stats import norm


def plot_gaussian_measure_distribution(
    measures: GaussianMeasure | list[GaussianMeasure],
    labels: str | list[str] | None = None,
    parameters: dict | None = None,
    fingerprint: FingerPrint | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot Gaussian measure(s) as normal distributions.

    Args:
        measures: Single GaussianMeasure or list of measures to plot
        labels: Label(s) for the distributions
        parameters: Dictionary of parameters to display (e.g. input values)
        fingerprint: FingerPrint object (for future use)
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (Figure, Axes) matplotlib objects

    """
    # Normalize inputs to lists
    if not isinstance(measures, list):
        measures = [measures]
    if labels is None:
        labels = [None] * len(measures)
    elif not isinstance(labels, list):
        labels = [labels]

    # Extract statistics from measures
    expectations = []
    stds = []
    for measure in measures:
        expectation = measure.expectation[0]
        variance = measure.covariance.matrix(dense=True)[0, 0]
        expectations.append(expectation)
        stds.append(np.sqrt(variance))

    # Set up x-axis range (±4 standard deviations)
    x_min = min(np.array(expectations) - 4 * np.array(stds))
    x_max = max(np.array(expectations) + 4 * np.array(stds))
    x = np.linspace(x_min, x_max, 1000)

    # Create figure and plot distributions
    fig, ax = plt.subplots(figsize=figsize)
    for label, mu, sigma in zip(
        labels, expectations, stds, strict=False
    ):
        y = norm.pdf(x, mu, sigma)
        legend_label = (
            f"{label} (μ={mu:.2e}, σ={sigma:.2e})"
            if label is not None
            else f"μ={mu:.2e}, σ={sigma:.2e}"
        )
        ax.plot(x, y, label=legend_label)

    # Configure axes
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel("Global Mean Sea Level Change (m)")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="upper right")

    # Add parameter box if provided
    if parameters is not None:
        text_lines = []
        for key, value in parameters.items():
            # Format small numbers in scientific notation
            if isinstance(value, (int, float)):
                formatted_value = (
                    f"{value:.2e}"
                    if abs(value) < 0.01 and value != 0
                    else str(value)
                )
            else:
                formatted_value = str(value)
            text_lines.append(f"{key}: {formatted_value} m")

        arg_string = "\n".join(text_lines)
        fig.text(
            x=0.1,
            y=0.95,
            s="Input Parameters:\n\n" + arg_string,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=9,
            bbox={"facecolor": "lightyellow", "alpha": 0.7, "pad": 5},
        )

    fig.tight_layout()
    return fig, ax


def plot_gmsl_comparison(
    true_measure: GaussianMeasure,
    approximate_measures: list[GaussianMeasure],
    approximate_labels: list[str],
    title: str = "GMSL Distribution Comparison",
    figsize: tuple[float, float] = (12, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot comparison between true and approximate GMSL distributions.

    Args:
        true_measure: True GMSL distribution
        approximate_measures: List of approximate GMSL distributions
        approximate_labels: Labels for approximate distributions
        title: Plot title
        figsize: Figure size as (width, height)

    Returns:
        Tuple of (Figure, Axes) matplotlib objects

    """
    all_measures = [true_measure] + approximate_measures
    all_labels = ["True"] + approximate_labels

    fig, ax = plot_gaussian_measure_distribution(
        measures=all_measures,
        labels=all_labels,
        figsize=figsize,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    return fig, ax
