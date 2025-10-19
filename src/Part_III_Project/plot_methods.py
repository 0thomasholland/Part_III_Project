"""
Plotting utilities for Part_III_Project

This module contains helpers to visualize ternary heatmaps using mpltern.

Functions:
- plot_ternary_heatmap: Plot a single ternary heatmap.
- plot_ternary_heatmap_subplots: Plot multiple ternary heatmaps as subplots
    using mpltern.
- plot_ternary_heatmap_subplots_parallel: Plot multiple ternary heatmaps as subplots
    using mpltern with parallel data processing.

Helper functions are prefixed with an underscore (_) to indicate they are for internal use only:
- _create_single_ternary_subplot: Main helper function to create a single ternary subplot
    on an existing axis.
- _extract_segment_data: Extract ternary coordinates and error values for a specific segment.
- _configure_ternary_axis: Configure axis labels and title for a ternary plot.
- _add_colorbar_to_axis: Add a colorbar to a ternary axis.
- _plot_tripcolor: Create a tripcolor plot on a ternary axis.
- _get_global_error_range: Calculate global min/max error values across specified segments.
"""

from __future__ import annotations

import mpltern  # noqa: F401  # Needed to register the 'ternary' projection with matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

#####################
## HELPER FUNCTION ##
#####################


def _extract_segment_data(
    df: pd.DataFrame, segment_name: float, sources: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Extract ternary coordinates and error values for a specific segment.

    Parameters:
    - df: DataFrame with columns ['segment', 'error', '<contribution columns>']
    - segment_name: The segment to filter
    - sources: List of column names for the three contributions [top, left, right]

    Returns:
    - Tuple of (top, left, right, errors) as numpy arrays, or None if no data found
    """
    segment_data = df[df["segment"] == segment_name].copy()

    if len(segment_data) == 0:
        return None

    top = segment_data[f"{sources[0]}"].values
    left = segment_data[f"{sources[1]}"].values
    right = segment_data[f"{sources[2]}"].values
    errors = segment_data["error"].values

    return top, left, right, errors


def _configure_ternary_axis(
    ax,
    labels: list[str],
    title: str,
    fontsize_labels: int = 13,
    fontsize_title: int = 14,
):
    """
    Configure axis labels and title for a ternary plot.

    Parameters:
    - ax: The ternary axis to configure
    - labels: List of labels for [top, left, right] axes
    - title: Title for the plot
    - fontsize_labels: Font size for axis labels
    - fontsize_title: Font size for title
    """
    ax.set_tlabel(labels[0], fontsize=fontsize_labels, fontweight="bold")
    ax.set_llabel(labels[1], fontsize=fontsize_labels, fontweight="bold")
    ax.set_rlabel(labels[2], fontsize=fontsize_labels, fontweight="bold")

    ax.taxis.set_label_position("corner")
    ax.laxis.set_label_position("corner")
    ax.raxis.set_label_position("corner")

    ax.set_title(
        title,
        fontsize=fontsize_title,
        fontweight="bold",
        pad=15 if fontsize_title < 14 else 20,
    )


def _add_colorbar_to_axis(ax, cs, fontsize: int = 12):
    """
    Add a colorbar to a ternary axis.

    Parameters:
    - ax: The ternary axis
    - cs: The tripcolor artist
    - fontsize: Font size for colorbar label

    Returns:
    - The colorbar object
    """
    cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8], transform=ax.transAxes)
    fig = ax.get_figure()
    colorbar = fig.colorbar(cs, cax=cax)
    colorbar.set_label("Error", rotation=270, va="baseline", fontsize=fontsize)
    return colorbar


def _plot_tripcolor(
    ax,
    top: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    errors: np.ndarray,
    vmin: float,
    vmax: float,
):
    """
    Create a tripcolor plot on a ternary axis.

    Parameters:
    - ax: The ternary axis
    - top, left, right: Ternary coordinates
    - errors: Values to plot
    - vmin, vmax: Color scale limits

    Returns:
    - The tripcolor artist
    """
    return ax.tripcolor(
        top,
        left,
        right,
        errors,
        shading="gouraud",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )


def _get_global_error_range(
    df: pd.DataFrame, segment_list: list[float]
) -> tuple[float, float]:
    """
    Calculate global min/max error values across specified segments.

    Parameters:
    - df: DataFrame with columns ['segment', 'error']
    - segment_list: List of segment values to include

    Returns:
    - Tuple of (global_vmin, global_vmax)
    """
    mask = df["segment"].isin(segment_list)
    global_vmin = df.loc[mask, "error"].min()
    global_vmax = df.loc[mask, "error"].max()
    return global_vmin, global_vmax


def _create_single_ternary_subplot(
    ax,
    df: pd.DataFrame,
    segment_name: float,
    sources: list[str],
    labels: list[str],
    global_vmin: float,
    global_vmax: float,
    fontsize_labels: int = 10,
    fontsize_title: int = 11,
    fontsize_colorbar: int = 9,
):
    """
    Main helper function to create a single ternary subplot on an existing axis.

    This is the core plotting function that all other functions use.

    Parameters:
    - ax: The matplotlib axis to plot on (must have 'ternary' projection)
    - df: DataFrame with columns ['segment', 'error', '<contribution columns>']
    - segment_name: The segment to filter and plot
    - sources: List of column names for the three contributions [top, left, right]
    - labels: List of labels for the three contributions
    - global_vmin, global_vmax: Color scale limits
    - fontsize_labels: Font size for axis labels
    - fontsize_title: Font size for subplot title
    - fontsize_colorbar: Font size for colorbar label

    Returns:
    - True if plot was successful, False if no data found
    """
    # Extract data for the segment
    data = _extract_segment_data(df, segment_name, sources)

    if data is None:
        ax.text(
            0.5,
            0.5,
            f"No data for segment {segment_name}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return False

    top, left, right, errors = data

    # Plot tripcolor
    cs = _plot_tripcolor(
        ax, top, left, right, errors, vmin=global_vmin, vmax=global_vmax
    )

    # Configure axis
    title = f"Satellite Segment ±{segment_name}°"
    _configure_ternary_axis(
        ax,
        labels,
        title,
        fontsize_labels=fontsize_labels,
        fontsize_title=fontsize_title,
    )

    # Add colorbar
    _add_colorbar_to_axis(ax, cs, fontsize=fontsize_colorbar)

    return True


def _extract_data_for_parallel(
    df: pd.DataFrame,
    segment_name: float,
    sources: list[str],
):
    """
    Helper function for parallel processing - extracts data without plotting.
    Returns the data needed to reconstruct the plot later.
    """
    data = _extract_segment_data(df, segment_name, sources)
    return data, segment_name


#####################
## CALLED FUNCTION ##
#####################


def plot_ternary_heatmap(
    df: pd.DataFrame, segment_name: float, sources: list[str], labels: list[str]
):
    """
    Plot a single ternary heatmap using mpltern.

    Parameters:
    - df: DataFrame with columns ['segment', 'error', '<contribution columns>']
    - segment_name: The segment to filter and plot
    - sources: List of column names for the three contributions, in order [top, left, right]
    - labels: List of labels for the three contributions, in the same order as `sources`

    Returns the created matplotlib Figure.
    """
    # Check if data exists
    data = _extract_segment_data(df, segment_name, sources)

    if data is None:
        print(f"No data found for segment: {segment_name}")
        return None

    top, left, right, errors = data

    # Create figure with ternary projection
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(projection="ternary")

    # Use the main helper to create the plot
    _create_single_ternary_subplot(
        ax=ax,
        df=df,
        segment_name=segment_name,
        sources=sources,
        labels=labels,
        global_vmin=errors.min(),
        global_vmax=errors.max(),
        fontsize_labels=13,
        fontsize_title=14,
        fontsize_colorbar=12,
    )

    # Override title for single plot
    title = f"Sea Level Rise Approximation Error - Satellite Segment: ±{segment_name}°"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    return fig


def plot_ternary_heatmap_subplots(
    df: pd.DataFrame,
    segment_list: list[float],
    sources: list[str],
    labels: list[str],
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (6, 5),
):
    """
    Plot multiple ternary heatmaps as subplots using mpltern.

    Parameters:
    - df: DataFrame with columns ['segment', 'error', '<contribution columns>']
    - segment_list: List of segment values to plot
    - sources: List of column names for the three contributions, in order [top, left, right]
    - labels: List of labels for the three contributions, in the same order as `sources`
    - ncols: Number of columns in the subplot grid
    - figsize_per_plot: (width, height) for each individual plot

    Returns a tuple of (figure, print_statement).
    """
    n_segments = len(segment_list)
    nrows = int(np.ceil(n_segments / ncols))

    # Create figure
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    # Get global min/max error for consistent color scale
    global_vmin, global_vmax = _get_global_error_range(df, segment_list)

    # Create each subplot using the main helper
    for idx, segment_name in enumerate(segment_list):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="ternary")
        _create_single_ternary_subplot(
            ax=ax,
            df=df,
            segment_name=segment_name,
            sources=sources,
            labels=labels,
            global_vmin=global_vmin,
            global_vmax=global_vmax,
            fontsize_labels=10,
            fontsize_title=11,
            fontsize_colorbar=9,
        )

    plt.suptitle(
        "Sea Level Rise Approximation Error Across Satellite Segments",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    print_statement = "Max error across all segments: {:.2f}; Min error across all segments: {:.2f}".format(
        global_vmax, global_vmin
    )
    return fig, print_statement


def plot_ternary_heatmap_subplots_parallel(
    df: pd.DataFrame,
    segment_list: list[float],
    sources: list[str],
    labels: list[str],
    ncols: int = 3,
    figsize_per_plot: tuple[float, float] = (6, 5),
    n_jobs: int = -1,
):
    """
    Plot multiple ternary heatmaps as subplots using mpltern, with parallel data processing.

    Parameters:
    - df: DataFrame with columns ['segment', 'error', '<contribution columns>']
    - segment_list: List of segment values to plot
    - sources: List of column names for the three contributions, in order [top, left, right]
    - labels: List of labels for the three contributions, in the same order as `sources`
    - ncols: Number of columns in the subplot grid
    - figsize_per_plot: (width, height) for each individual plot
    - n_jobs: Number of parallel jobs (-1 uses all cores)

    Returns a tuple of (figure, print_statement).
    """
    n_segments = len(segment_list)
    nrows = int(np.ceil(n_segments / ncols))

    # Get global min/max error for consistent color scale
    global_vmin, global_vmax = _get_global_error_range(df, segment_list)

    # Extract data in parallel (this is the computationally expensive part)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_extract_data_for_parallel)(df, segment_name, sources)
        for segment_name in segment_list
    )

    # Create figure
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    # Create each subplot - plotting must be done serially due to matplotlib limitations
    for idx, (data, segment_name) in enumerate(results):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="ternary")

        if data is None:
            ax.text(
                0.5,
                0.5,
                f"No data for segment {segment_name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        top, left, right, errors = data

        # Plot tripcolor
        cs = _plot_tripcolor(
            ax, top, left, right, errors, vmin=global_vmin, vmax=global_vmax
        )

        # Configure axis
        title = f"Satellite Segment ±{segment_name}°"
        _configure_ternary_axis(
            ax, labels, title, fontsize_labels=10, fontsize_title=11
        )

        # Add colorbar
        _add_colorbar_to_axis(ax, cs, fontsize=9)

    plt.suptitle(
        "Sea Level Rise Approximation Error Across Satellite Segments",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    print_statement = "Max error across all segments: {:.2f}; Min error across all segments: {:.2f}".format(
        global_vmax, global_vmin
    )
    return fig, print_statement
