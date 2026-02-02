# %%
import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
from pygeoinf.symmetric_space.circle import (
    CircleHelper,
    Sobolev,
)

# %% definitions
model_space = Sobolev.from_sobolev_parameters(2.0, 0.01)

n_data_high = 15
n_data_low = 10
standard_deviation_high = 0.1
standard_deviation_low = 0.05

# Generate observation points for both cases
observation_points_high = model_space.random_points(
    n_data_high
)
observation_points_low = model_space.random_points(
    n_data_low
)


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

data_error_measure_high = (
    inf.GaussianMeasure.from_standard_deviation(
        data_space_high, standard_deviation_high
    )
)
data_error_measure_low = (
    inf.GaussianMeasure.from_standard_deviation(
        data_space_low,
        standard_deviation_low,
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

# Generate the true model once using the high observation problem
true_model, data_high = (
    forward_problem_high.synthetic_model_and_data(
        model_prior_measure
    )
)

# Generate data for low observations using the same true model
# Apply the forward operator to the true model and add noise
data_low_clean = forward_operator_low(true_model)
noise_low = np.random.normal(
    0, standard_deviation_low, n_data_low
)
data_low = data_low_clean + noise_low

# %% plotting helper for comparison


def plot_comparison_results(
    space: CircleHelper,
    # true_model: np.ndarray,
    posterior_mean_high: np.ndarray,
    posterior_std_high: np.ndarray,
    # posterior_mean_low: np.ndarray,
    # posterior_std_low: np.ndarray,
    n_high: int,
    # n_low: int,
    obs_points_high: np.ndarray,
    # obs_points_low: np.ndarray,
    data_high: np.ndarray,
    # data_low: np.ndarray,
    data_high_std: float,
    # data_low_std: float,
):
    """Helper function to create a comparison plot of high vs low observations."""
    fig, ax = space.plot(
        true_model,
        color="white",
        linestyle="",
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
        label=f"Posterior Mean (n={n_high})",
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
    # space.plot(
    #     posterior_mean_low,
    #     fig=fig,
    #     ax=ax,
    #     color="r",
    #     label=f"Posterior Mean (n={n_low})",
    # )
    # space.plot_error_bounds(
    #     posterior_mean_low,
    #     2 * posterior_std_low,
    #     fig=fig,
    #     ax=ax,
    #     alpha=0.2,
    #     color="r",
    # )

    # ax.errorbar(obs_points, data, 2 * data_std, fmt="ko", capsize=3, label="Data")
    # use for high and low obs points

    ax.errorbar(
        obs_points_high,
        data_high,
        2 * data_high_std,
        fmt="bo",
        capsize=3,
        label="High Obs Data (σ=%.2f)" % data_high_std,
    )
    # ax.errorbar(
    #     obs_points_low,
    #     data_low,
    #     2 * data_low_std,
    #     fmt="ro",
    #     capsize=3,
    #     label="Low Obs Data (σ=%.2f)" % data_low_std,
    # )

    ax.set_title(
        "Inversion Comparison: High vs Low Observations",
        fontsize=16,
    )
    ax.set_xlabel("Angle (radians)")
    ax.set_ylabel("Function Value")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    # plt.savefig(
    #     "bayesian_inversion_comparison.png", dpi=600
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
    # true_model,
    posterior_mean_high,
    posterior_std_high,
    # posterior_mean_low,
    # posterior_std_low,
    n_data_high,
    # n_data_low,
    observation_points_high,
    # observation_points_low,
    data_high,
    # data_low,
    standard_deviation_high,
    # standard_deviation_low,
)
