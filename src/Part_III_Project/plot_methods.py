"""
Plotting utilities for Part_III_Project

This module contains helpers to visualize ternary heatmaps using mpltern.
"""

from __future__ import annotations

import mpltern  # noqa: F401  # Needed to register the 'ternary' projection with matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


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

	# Filter data for the specific segment
	segment_data = df[df["segment"] == segment_name].copy()

	if len(segment_data) == 0:
		print(f"No data found for segment: {segment_name}")
		return None

	# Extract the three components and error values
	top = segment_data[f"{sources[0]}"].values
	left = segment_data[f"{sources[1]}"].values
	right = segment_data[f"{sources[2]}"].values
	errors = segment_data["error"].values

	# Create figure with ternary projection
	fig = plt.figure(figsize=(10, 9))
	ax = fig.add_subplot(projection="ternary")

	# Plot using tripcolor with gouraud shading for smooth interpolation
	cs = ax.tripcolor(
		top,
		left,
		right,
		errors,
		shading="gouraud",
		cmap="RdYlBu_r",
		rasterized=True,
		vmin=segment_data["error"].min(),
		vmax=segment_data["error"].max(),
	)

	# Set axis labels
	ax.set_tlabel(labels[0], fontsize=13, fontweight="bold")
	ax.set_llabel(labels[1], fontsize=13, fontweight="bold")
	ax.set_rlabel(labels[2], fontsize=13, fontweight="bold")

	ax.taxis.set_label_position("corner")
	ax.laxis.set_label_position("corner")
	ax.raxis.set_label_position("corner")

	# Set title
	ax.set_title(
		f"Sea Level Rise Approximation Error - Satellite Segment: ±{segment_name}°",
		fontsize=14,
		fontweight="bold",
		pad=20,
	)

	# Add colorbar
	cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8], transform=ax.transAxes)
	colorbar = fig.colorbar(cs, cax=cax)
	colorbar.set_label("Error", rotation=270, va="baseline", fontsize=12)

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
	mask = df["segment"].isin(segment_list)
	global_vmin = df.loc[mask, "error"].min()
	global_vmax = df.loc[mask, "error"].max()

	for idx, segment_name in enumerate(segment_list):
		# Create ternary subplot
		ax = fig.add_subplot(nrows, ncols, idx + 1, projection="ternary")

		# Filter data for the specific segment
		segment_data = df[df["segment"] == segment_name].copy()

		if len(segment_data) == 0:
			ax.text(
				0.5,
				0.5,
				f"No data for segment {segment_name}",
				ha="center",
				va="center",
				transform=ax.transAxes,
			)
			continue

		# Extract the three components and error values
		top = segment_data[f"{sources[0]}"].values
		left = segment_data[f"{sources[1]}"].values
		right = segment_data[f"{sources[2]}"].values
		errors = segment_data["error"].values

		# Plot using tripcolor with gouraud shading for smooth interpolation
		cs = ax.tripcolor(
			top,
			left,
			right,
			errors,
			shading="gouraud",
			cmap="RdYlBu_r",
			vmin=global_vmin,
			vmax=global_vmax,
			rasterized=True,
		)

		# Set axis labels
		ax.set_tlabel(labels[0], fontsize=10, fontweight="bold")
		ax.set_llabel(labels[1], fontsize=10, fontweight="bold")
		ax.set_rlabel(labels[2], fontsize=10, fontweight="bold")

		ax.taxis.set_label_position("corner")
		ax.laxis.set_label_position("corner")
		ax.raxis.set_label_position("corner")

		# Set title
		ax.set_title(
			f"Satellite Segment ±{segment_name}°",
			fontsize=11,
			fontweight="bold",
			pad=15,
		)

		# Add individual colorbar for this subplot
		cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8], transform=ax.transAxes)
		colorbar = fig.colorbar(cs, cax=cax)
		colorbar.set_label("Error", rotation=270, va="baseline", fontsize=9)

	plt.suptitle(
		"Sea Level Rise Approximation Error Across Satellite Segments",
		fontsize=16,
		fontweight="bold",
		y=0.98,
	)
	plt.tight_layout()
	print_statement = (
		"Max error across all segments: {:.2f}; Min error across all segments: {:.2f}".format(
			global_vmax, global_vmin
		)
	)
	return fig, print_statement


def _plot_single_ternary_subplot(
    df: pd.DataFrame,
    segment_name: float,
    sources: list[str],
    labels: list[str],
    global_vmin: float,
    global_vmax: float,
):
    """
    Helper function to plot a single ternary subplot.
    Returns the artists needed to reconstruct the plot.
    """
    # Filter data for the specific segment
    segment_data = df[df["segment"] == segment_name].copy()

    if len(segment_data) == 0:
        return None, segment_name

    # Extract the three components and error values
    top = segment_data[f"{sources[0]}"].values
    left = segment_data[f"{sources[1]}"].values
    right = segment_data[f"{sources[2]}"].values
    errors = segment_data["error"].values

    return (top, left, right, errors), segment_name


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
    mask = df["segment"].isin(segment_list)
    global_vmin = df.loc[mask, "error"].min()
    global_vmax = df.loc[mask, "error"].max()

    # Process segments in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_plot_single_ternary_subplot)(
            df, segment_name, sources, labels, global_vmin, global_vmax
        )
        for segment_name in segment_list
    )

    # Create figure
    fig = plt.figure(figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))

    for idx, (data, segment_name) in enumerate(results):
        # Create ternary subplot
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

        # Plot using tripcolor with gouraud shading for smooth interpolation
        cs = ax.tripcolor(
            top,
            left,
            right,
            errors,
            shading="gouraud",
            cmap="RdYlBu_r",
            vmin=global_vmin,
            vmax=global_vmax,
            rasterized=True,
        )

        # Set axis labels
        ax.set_tlabel(labels[0], fontsize=10, fontweight="bold")
        ax.set_llabel(labels[1], fontsize=10, fontweight="bold")
        ax.set_rlabel(labels[2], fontsize=10, fontweight="bold")

        ax.taxis.set_label_position("corner")
        ax.laxis.set_label_position("corner")
        ax.raxis.set_label_position("corner")

        # Set title
        ax.set_title(
            f"Satellite Segment ±{segment_name}°",
            fontsize=11,
            fontweight="bold",
            pad=15,
        )

        # Add individual colorbar for this subplot
        cax = ax.inset_axes([1.05, 0.1, 0.05, 0.8], transform=ax.transAxes)
        colorbar = fig.colorbar(cs, cax=cax)
        colorbar.set_label("Error", rotation=270, va="baseline", fontsize=9)

    plt.suptitle(
        "Sea Level Rise Approximation Error Across Satellite Segments",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    print_statement = (
        "Max error across all segments: {:.2f}; Min error across all segments: {:.2f}".format(
            global_vmax, global_vmin
        )
    )
    return fig, print_statement