"""Statistical analysis utilities for Part_III_Project.

This module provides functions for computing statistical distance metrics
between probability distributions, particularly for comparing approximate
and true distributions in sea level change estimation.

Functions:
    kullback_leibler_divergence: KL divergence between two normal distributions
    mean_squared_error: MSE between distribution means
    cohens_d_effect_size: Cohen's d effect size metric
    wasserstein_distance: 2-Wasserstein distance between normal distributions
"""

import numpy as np


def kullback_leibler_divergence(
    expectation_true: float,
    std_true: float,
    expectation_approx: float,
    std_approx: float,
) -> float:
    """Compute the Kullback-Leibler divergence between two univariate normal distributions.

    The KL divergence measures how one probability distribution diverges from a
    reference probability distribution. For normal distributions N(μ_true, σ_true²)
    and N(μ_approx, σ_approx²), this provides a measure of information lost when
    using the approximate distribution.

    Args:
        expectation_true: Mean of the true distribution
        std_true: Standard deviation of the true distribution
        expectation_approx: Mean of the approximate distribution
        std_approx: Standard deviation of the approximate distribution

    Returns:
        KL divergence from true to approximate distribution (non-negative)

    Note:
        Returns KL(P_true || P_approx) where P_true is the reference distribution.
        The metric is asymmetric: KL(P||Q) ≠ KL(Q||P) in general.

    """
    kl_divergence = (
        np.log(std_approx / std_true)
        + (std_true**2 + (expectation_true - expectation_approx) ** 2)
        / (2 * std_approx**2)
        - 0.5
    )
    return float(kl_divergence)


def mean_squared_error(
    expectation_true: float,
    expectation_approx: float,
) -> float:
    """Compute the Mean Squared Error between distribution means.

    MSE measures the squared difference between the true and approximate
    distribution means, providing a simple measure of bias.

    Args:
        expectation_true: Mean of the true distribution
        expectation_approx: Mean of the approximate distribution

    Returns:
        Mean squared error (non-negative)

    """
    mse = (expectation_true - expectation_approx) ** 2
    return float(mse)


def cohens_d_effect_size(
    expectation_true: float,
    std_true: float,
    expectation_approx: float,
    std_approx: float,
) -> float:
    """Compute Cohen's d effect size between two normal distributions.

    Cohen's d measures the standardized difference between two means,
    using the pooled standard deviation. This provides a scale-independent
    measure of the separation between distributions.

    Args:
        expectation_true: Mean of the true distribution
        std_true: Standard deviation of the true distribution
        expectation_approx: Mean of the approximate distribution
        std_approx: Standard deviation of the approximate distribution

    Returns:
        Cohen's d effect size (can be positive or negative)

    Note:
        |d| < 0.2: small effect
        |d| ≈ 0.5: medium effect
        |d| > 0.8: large effect

    """
    pooled_std = np.sqrt((std_true**2 + std_approx**2) / 2)
    d = (expectation_true - expectation_approx) / pooled_std
    return float(d)


def wasserstein_distance(
    expectation_true: float,
    std_true: float,
    expectation_approx: float,
    std_approx: float,
) -> float:
    """Compute the 2-Wasserstein distance between two univariate normal distributions.

    The Wasserstein distance (also called Earth Mover's Distance) measures the
    minimum "cost" of transforming one probability distribution into another.
    For normal distributions, this has a closed form.

    Args:
        expectation_true: Mean of the true distribution
        std_true: Standard deviation of the true distribution
        expectation_approx: Mean of the approximate distribution
        std_approx: Standard deviation of the approximate distribution

    Returns:
        2-Wasserstein distance (non-negative)

    Note:
        The Wasserstein distance is a proper metric (symmetric, satisfies triangle inequality).

    """
    wd = np.sqrt(
        (expectation_true - expectation_approx) ** 2
        + (std_true - std_approx) ** 2,
    )
    return float(wd)
