from typing import List, Optional, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_measure():
    pass


def get_1D_stats():
    pass


def plot_corner_distributions(
    posterior_measure: object,
    /,
    *,
    true_values: Optional[
        Union[List[float], np.ndarray]
    ] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = None,
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
):
    """
    Create a corner plot for multi-dimensional posterior distributions.

    Args:
        posterior_measure: Multi-dimensional posterior measure (pygeoinf object)
        true_values: True values for each dimension (optional)
        labels: Labels for each dimension (optional)
        title: Title for the plot
        figsize: Figure size tuple (if None, calculated based on dimensions)
        show_plot: Whether to display the plot
        include_sigma_contours: Whether to include 1-sigma contour lines
        colormap: Colormap for 2D plots
        parallel: Compute dense covariance matrix in parallel, default False.
        n_jobs: Number of cores to use in parallel calculations, default -1.

    Returns:
        fig, axes: Figure and axes array
    """

    # Extract statistics from the measure
    if hasattr(
        posterior_measure, "expectation"
    ) and hasattr(posterior_measure, "covariance"):
        mean_posterior = posterior_measure.expectation
        cov_posterior = posterior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )
    else:
        raise ValueError(
            "posterior_measure must have 'expectation' and 'covariance' attributes"
        )

    n_dims = len(mean_posterior)

    # Set default labels if not provided
    if labels is None:
        labels = [
            f"Dimension {i + 1}" for i in range(n_dims)
        ]

    # Set figure size based on dimensions if not provided
    if figsize is None:
        figsize = (3 * n_dims, 3 * n_dims)

    # Create subplots
    fig, axes = plt.subplots(
        n_dims, n_dims, figsize=figsize
    )
    fig.suptitle(title, fontsize=16)

    # Ensure axes is always 2D array
    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_dims == 2:
        axes = axes.reshape(2, 2)

    # Initialize pcm variable for colorbar
    pcm = None

    for i in range(n_dims):
        for j in range(n_dims):
            ax = axes[i, j]

            if (
                i == j
            ):  # Diagonal plots (1D marginal distributions)
                mu = mean_posterior[i]
                sigma = np.sqrt(cov_posterior[i, i])

                # Create x-axis range
                x = np.linspace(
                    mu - 3.75 * sigma,
                    mu + 3.75 * sigma,
                    200,
                )
                pdf = stats.norm.pdf(x, mu, sigma)

                # Plot the PDF
                ax.plot(
                    x,
                    pdf,
                    "darkblue",
                    label="Posterior PDF",
                )
                ax.fill_between(
                    x, pdf, color="lightblue", alpha=0.6
                )

                # Add true value if provided
                if true_values is not None:
                    true_val = true_values[i]
                    ax.axvline(
                        true_val,
                        color="black",
                        linestyle="-",
                        label=f"True: {true_val:.2f}",
                    )

                ax.set_xlabel(labels[i])
                ax.set_ylabel("Density" if i == 0 else "")
                ax.set_yticklabels([])

            elif (
                i > j
            ):  # Lower triangle: 2D joint distributions
                # Extract 2D mean and covariance
                mean_2d = np.array(
                    [mean_posterior[j], mean_posterior[i]]
                )
                cov_2d = np.array(
                    [
                        [
                            cov_posterior[j, j],
                            cov_posterior[j, i],
                        ],
                        [
                            cov_posterior[i, j],
                            cov_posterior[i, i],
                        ],
                    ]
                )

                # Create 2D grid
                sigma_j = np.sqrt(cov_posterior[j, j])
                sigma_i = np.sqrt(cov_posterior[i, i])

                x_range = np.linspace(
                    mean_2d[0] - 3.75 * sigma_j,
                    mean_2d[0] + 3.75 * sigma_j,
                    100,
                )
                y_range = np.linspace(
                    mean_2d[1] - 3.75 * sigma_i,
                    mean_2d[1] + 3.75 * sigma_i,
                    100,
                )

                X, Y = np.meshgrid(x_range, y_range)
                pos = np.dstack((X, Y))

                # Calculate PDF values
                rv = stats.multivariate_normal(
                    mean_2d, cov_2d
                )
                Z = rv.pdf(pos)

                # Create filled contour plot using pcolormesh like the original
                pcm = ax.pcolormesh(
                    X,
                    Y,
                    Z,
                    shading="auto",
                    cmap=colormap,
                    norm=colors.LogNorm(
                        vmin=Z.min(), vmax=Z.max()
                    ),
                )

                # Add contour lines
                ax.contour(
                    X,
                    Y,
                    Z,
                    colors="black",
                    linewidths=0.5,
                    alpha=0.6,
                )

                # Add 1-sigma contour if requested
                if include_sigma_contours:
                    # Calculate 1-sigma level (approximately 39% of peak for 2D Gaussian)
                    sigma_level = rv.pdf(mean_2d) * np.exp(
                        -0.5
                    )
                    ax.contour(
                        X,
                        Y,
                        Z,
                        levels=[sigma_level],
                        colors="red",
                        linewidths=1,
                        linestyles="--",
                        alpha=0.8,
                    )

                # Plot mean point
                ax.plot(
                    mean_posterior[j],
                    mean_posterior[i],
                    "r+",
                    markersize=10,
                    mew=2,
                    label="Posterior Mean",
                )

                # Plot true value if provided
                if true_values is not None:
                    ax.plot(
                        true_values[j],
                        true_values[i],
                        "kx",
                        markersize=10,
                        mew=2,
                        label="True Value",
                    )

                ax.set_xlabel(labels[j])
                ax.set_ylabel(labels[i])

            else:  # Upper triangle: hide these plots
                ax.axis("off")

    # Create legend similar to the original
    handles, labels_leg = axes[
        0, 0
    ].get_legend_handles_labels()
    if n_dims > 1:
        handles2, labels2 = axes[
            1, 0
        ].get_legend_handles_labels()
        handles.extend(handles2)
        labels_leg.extend(labels2)

    # Clean up labels by removing values after colons
    cleaned_labels = [
        label.split(":")[0] for label in labels_leg
    ]

    fig.legend(
        handles,
        cleaned_labels,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.95),
    )

    # Adjust main plot layout to make room on the right for the colorbar
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])

    # Add a colorbar if we have 2D plots
    if n_dims > 1 and pcm is not None:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax)
        cbar.set_label("Probability Density", size=12)

    return fig, axes


def plot_bivariate_corner(
    posterior_measure: object,
    /,
    *,
    true_values: Optional[
        Union[List[float], np.ndarray]
    ] = None,
    labels: Optional[List[str]] = None,
    title: str = "Joint Posterior Distribution",
    figsize: Optional[tuple] = (8, 8),
    include_sigma_contours: bool = True,
    colormap: str = "Blues",
    parallel: bool = False,
    n_jobs: int = -1,
    pdf_colors: Optional[List[str]] = None,
):
    """
    Create a bivariate corner plot (2D) where the top plot is an upright Gaussian
    and the right plot is rotated.
    """
    # Extract statistics from the measure
    if hasattr(
        posterior_measure, "expectation"
    ) and hasattr(posterior_measure, "covariance"):
        mean_posterior = posterior_measure.expectation
        cov_posterior = posterior_measure.covariance.matrix(
            dense=True, parallel=parallel, n_jobs=n_jobs
        )
    else:
        raise ValueError(
            "posterior_measure must have 'expectation' and 'covariance' attributes"
        )

    n_dims = len(mean_posterior)
    if n_dims != 2:
        raise ValueError(
            "plot_bivariate_corner expects exactly 2 dimensions."
        )

    if labels is None:
        labels = [
            f"Dimension {i + 1}" for i in range(n_dims)
        ]

    if pdf_colors is None:
        pdf_colors = ["darkblue", "darkblue"]
    elif len(pdf_colors) != 2:
        raise ValueError(
            "pdf_colors must contain exactly 2 colors."
        )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [2, 1],
            "height_ratios": [1, 2],
        },
    )
    fig.suptitle(title, fontsize=16)

    # Top-Left: Upright Gaussian (Dim 0)
    ax00 = axes[0, 0]
    mu0 = mean_posterior[0]
    sigma0 = np.sqrt(cov_posterior[0, 0])
    x0 = np.linspace(
        mu0 - 3.75 * sigma0, mu0 + 3.75 * sigma0, 200
    )
    pdf0 = stats.norm.pdf(x0, mu0, sigma0)
    ax00.plot(
        x0, pdf0, color=pdf_colors[0], label="Posterior PDF"
    )
    ax00.fill_between(
        x0, pdf0, color=pdf_colors[0], alpha=0.4
    )
    if true_values is not None:
        ax00.axvline(
            true_values[0],
            color="black",
            linestyle="-",
            label=f"True: {true_values[0]:.2f}",
        )
    ax00.set_ylabel("Density")
    ax00.set_xticklabels([])

    # Bottom-Right: Rotated Gaussian (Dim 1)
    ax11 = axes[1, 1]
    mu1 = mean_posterior[1]
    sigma1 = np.sqrt(cov_posterior[1, 1])
    x1 = np.linspace(
        mu1 - 3.75 * sigma1, mu1 + 3.75 * sigma1, 200
    )
    pdf1 = stats.norm.pdf(x1, mu1, sigma1)
    ax11.plot(
        pdf1, x1, color=pdf_colors[1], label="Posterior PDF"
    )
    ax11.fill_betweenx(
        x1, 0, pdf1, color=pdf_colors[1], alpha=0.4
    )
    if true_values is not None:
        ax11.axhline(
            true_values[1],
            color="black",
            linestyle="-",
            label=f"True: {true_values[1]:.2f}",
        )
    ax11.set_xlabel("Density")
    ax11.set_yticklabels([])

    # Bottom-Left: 2D Contour
    ax10 = axes[1, 0]
    mean_2d = np.array(
        [mean_posterior[0], mean_posterior[1]]
    )
    cov_2d = np.array(
        [
            [cov_posterior[0, 0], cov_posterior[0, 1]],
            [cov_posterior[1, 0], cov_posterior[1, 1]],
        ]
    )
    X, Y = np.meshgrid(x0[::2], x1[::2])
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean_2d, cov_2d)
    Z = rv.pdf(pos)
    z_min = max(Z.min(), Z.max() * 1e-10)
    pcm = ax10.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        cmap=colormap,
        norm=colors.LogNorm(vmin=z_min, vmax=Z.max()),
    )
    ax10.contour(
        X, Y, Z, colors="black", linewidths=0.5, alpha=0.6
    )
    if include_sigma_contours:
        sigma_level = rv.pdf(mean_2d) * np.exp(-0.5)
        ax10.contour(
            X,
            Y,
            Z,
            levels=[sigma_level],
            colors="red",
            linewidths=1,
            linestyles="--",
            alpha=0.8,
        )
    ax10.plot(
        mean_posterior[0],
        mean_posterior[1],
        "r+",
        markersize=10,
        mew=2,
        label="Posterior Mean",
    )
    if true_values is not None:
        ax10.plot(
            true_values[0],
            true_values[1],
            "kx",
            markersize=10,
            mew=2,
            label="True Value",
        )
    ax10.set_xlabel(labels[0])
    ax10.set_ylabel(labels[1])

    # Top-Right: Hidden
    ax01 = axes[0, 1]
    ax01.axis("off")

    # Legend
    handles, labels_leg = ax10.get_legend_handles_labels()
    handles2, labels2 = ax00.get_legend_handles_labels()
    handles.extend(handles2)
    labels_leg.extend(labels2)

    cleaned_labels = [
        label.split(":")[0] for label in labels_leg
    ]
    # Remove duplicates while preserving order
    unique_handles = []
    unique_labels = []
    for h, l in zip(handles, cleaned_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)

    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.95),
    )

    plt.tight_layout(rect=[0, 0, 0.88, 0.96])

    cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label("Probability Density", size=12)

    return fig, axes
