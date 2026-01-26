# %%
import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import (
    CircleHelper,
    Sobolev,
)

# %% definitions
model_space = Sobolev.from_sobolev_parameters(2.0, 0.05)

n_data = 20
missing = 8  # number of missing observations
standard_deviation = 0.1

# Calculate number of observations for each case
n_data_high = n_data
n_data_low = n_data - missing

# Generate ALL observation points first (for the high case)
observation_points_all = model_space.random_points(
    n_data_high
)

# Convert to numpy array if it isn't already
observation_points_all = np.array(observation_points_all)

# For the low case, randomly select which points to keep
# First shuffle to make the selection random
indices = np.arange(n_data_high)
np.random.shuffle(indices)
selected_indices = np.sort(
    indices[:n_data_low]
)  # Sort to maintain order

observation_points_high = observation_points_all
observation_points_low = observation_points_all[
    selected_indices
]

# Create forward operators for both cases
forward_operator_high = (
    model_space.point_evaluation_operator(
        observation_points_high
    )
)
forward_operator_low = (
    model_space.point_evaluation_operator(
        observation_points_low
    )
)

data_space_high = forward_operator_high.codomain
data_space_low = forward_operator_low.codomain


standard_deviation_high = standard_deviation
standard_deviation_low = standard_deviation

data_error_measure_high = (
    inf.GaussianMeasure.from_standard_deviation(
        data_space_high, standard_deviation_high
    )
)
data_error_measure_low = (
    inf.GaussianMeasure.from_standard_deviation(
        data_space_low, standard_deviation_low
    )
)

forward_problem_high = inf.LinearForwardProblem(
    forward_operator_high,
    data_error_measure=data_error_measure_high,
)
forward_problem_low = inf.LinearForwardProblem(
    forward_operator_low,
    data_error_measure=data_error_measure_low,
)

model_prior_measure = model_space.point_value_scaled_heat_kernel_gaussian_measure(
    0.1, 1.0
)

true_model, data_high = (
    forward_problem_high.synthetic_model_and_data(
        model_prior_measure
    )
)

data_low = data_high[selected_indices]

# %% plotting helper for comparison


def plot_comparison_results(
    space: CircleHelper,
    true_model: np.ndarray,
    posterior_mean_high: np.ndarray,
    posterior_std_high: np.ndarray,
    posterior_mean_low: np.ndarray,
    posterior_std_low: np.ndarray,
    n_high: int,
    n_low: int,
    obs_points_high: np.ndarray,
    obs_points_low: np.ndarray,
    data_high: np.ndarray,
    data_low: np.ndarray,
    data_high_std: float,
    data_low_std: float,
):
    """Helper function to create a comparison plot of high vs low observations."""
    fig, ax = space.plot(
        true_model,
        color="k",
        linestyle="--",
        linewidth=2,
        label="True Model",
        figsize=(10, 6),
    )

    # Plot high observations solution (blue)
    space.plot(
        posterior_mean_high,
        fig=fig,
        ax=ax,
        color="b",
        label=f"All Data Obs Posterior Mean (n={n_high})",
    )
    space.plot_error_bounds(
        posterior_mean_high,
        2 * posterior_std_high,
        fig=fig,
        ax=ax,
        alpha=0.2,
        color="b",
    )

    # Plot low observations solution (red)
    space.plot(
        posterior_mean_low,
        fig=fig,
        ax=ax,
        color="r",
        label=f"Reduced Posterior Mean (n={n_low})",
    )
    space.plot_error_bounds(
        posterior_mean_low,
        2 * posterior_std_low,
        fig=fig,
        ax=ax,
        alpha=0.2,
        color="r",
    )

    ax.errorbar(
        obs_points_high,
        data_high,
        2 * data_high_std,
        fmt="bo",
        capsize=3,
        label=f"All Obs Data (n={n_high}, σ=%.2f)"
        % data_high_std,
    )
    ax.errorbar(
        obs_points_low,
        data_low,
        2 * data_low_std,
        fmt="ro",
        capsize=3,
        label=f"Reduced Obs Data (n={n_low}, σ=%.2f)"
        % data_low_std,
    )

    ax.set_title(
        f"Inversion Comparison: {n_high} vs {n_low} Observations ({missing} Missing Points)",
        fontsize=16,
    )
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    # plt.savefig(
    #     "pygeoinf_inversion_missing_case.png", dpi=600
    # )
    plt.show()


# %% perform Bayesian inversion for high observations

bayesian_inversion_high = inf.LinearBayesianInversion(
    forward_problem_high, model_prior_measure
)

solver = inf.CholeskySolver(galerkin=True)
model_posterior_measure_high = (
    bayesian_inversion_high.model_posterior_measure(
        data_high, solver
    )
)
posterior_mean_high = (
    model_posterior_measure_high.expectation
)

posterior_pointwise_variance_high = (
    model_posterior_measure_high.sample_pointwise_variance(
        200
    )
)
posterior_std_high = np.sqrt(
    posterior_pointwise_variance_high
)

# %% perform Bayesian inversion for low observations

bayesian_inversion_low = inf.LinearBayesianInversion(
    forward_problem_low, model_prior_measure
)

model_posterior_measure_low = (
    bayesian_inversion_low.model_posterior_measure(
        data_low, solver
    )
)
posterior_mean_low = model_posterior_measure_low.expectation

posterior_pointwise_variance_low = (
    model_posterior_measure_low.sample_pointwise_variance(
        200
    )
)
posterior_std_low = np.sqrt(
    posterior_pointwise_variance_low
)

# %% plot comparison

plot_comparison_results(
    model_space,
    true_model,
    posterior_mean_high,
    posterior_std_high,
    posterior_mean_low,
    posterior_std_low,
    n_data_high,
    n_data_low,
    observation_points_high,
    observation_points_low,
    data_high,
    data_low,
    standard_deviation_high,
    standard_deviation_low,
)
