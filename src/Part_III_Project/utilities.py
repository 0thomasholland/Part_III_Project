"""Common utility functions for Part_III_Project.

This module provides helper functions for working with pyslfp FingerPrint
objects, computing altimetry-related quantities, and extracting statistics
from Gaussian measures.

Functions:
    compute_altimetry_weighting_function: Create normalized altimetry weighting
    extract_gmsl_statistics: Extract mean and std from GMSL measure
    compute_relative_error: Calculate relative percentage error
"""

import numpy as np
from pygeoinf import GaussianMeasure
from pyslfp import FingerPrint


def compute_altimetry_weighting_function(
    fingerprint: FingerPrint,
    latitude_min: float,
    latitude_max: float,
) -> "SHGrid":
    """Compute a normalized altimetry weighting function for a latitude range.

    Creates a weighting function that integrates to 1 over the specified
    satellite coverage region, suitable for computing weighted averages
    of sea surface height changes.

    Args:
        fingerprint: pyslfp FingerPrint object defining the geometry
        latitude_min: Minimum latitude in degrees (e.g., -66 for typical satellites)
        latitude_max: Maximum latitude in degrees (e.g., +66 for typical satellites)

    Returns:
        Normalized weighting function as an SHGrid that integrates to 1
        over the specified latitude band

    Example:
        >>> fp = FingerPrint()
        >>> weighting = compute_altimetry_weighting_function(fp, -66, 66)
        >>> fp.integrate(weighting)  # Should be approximately 1.0

    """
    altimetry_projection = fingerprint.altimetry_projection(
        latitude_min=latitude_min,
        latitude_max=latitude_max,
        value=0,
    )
    altimetry_projection_integral = fingerprint.integrate(
        altimetry_projection
    )
    altimetry_weighting_function = (
        altimetry_projection / altimetry_projection_integral
    )
    return altimetry_weighting_function


def extract_gmsl_statistics(
    gmsl_measure: GaussianMeasure,
) -> tuple[float, float]:
    """Extract mean and standard deviation from a GMSL Gaussian measure.

    Args:
        gmsl_measure: A 1D Gaussian measure representing GMSL distribution

    Returns:
        Tuple of (expectation, standard_deviation)

    Example:
        >>> mean, std = extract_gmsl_statistics(gmsl_measure)
        >>> print(f"GMSL: {mean:.2f} ± {std:.2e} m")

    """
    expectation = float(gmsl_measure.expectation[0])
    variance = gmsl_measure.covariance.matrix(dense=True)[0, 0]
    std = float(np.sqrt(variance))
    return expectation, std


def compute_relative_error(
    true_value: float,
    estimated_value: float,
    percentage: bool = True,
) -> float:
    """Compute the relative error between true and estimated values.

    Args:
        true_value: The true or reference value
        estimated_value: The estimated or approximate value
        percentage: If True, return as percentage; if False, return as fraction

    Returns:
        Relative error, optionally as percentage

    Example:
        >>> compute_relative_error(100.0, 95.0)
        5.0  # 5% error

    """
    if true_value == 0:
        raise ValueError(
            "True value cannot be zero for relative error calculation"
        )

    relative_error = np.abs(estimated_value - true_value) / np.abs(
        true_value
    )

    if percentage:
        relative_error *= 100

    return float(relative_error)


def compute_ocean_altimetry_weighting_function(
    fingerprint: FingerPrint,
    latitude_min: float,
    latitude_max: float,
) -> "SHGrid":
    """Compute normalized weighting function for altimetry over ocean regions only.

    Combines ocean projection with altimetry latitude coverage to create
    a weighting function for computing ocean-only averages within satellite range.

    Args:
        fingerprint: pyslfp FingerPrint object defining the geometry
        latitude_min: Minimum latitude in degrees
        latitude_max: Maximum latitude in degrees

    Returns:
        Normalized weighting function that integrates to 1 over ocean
        regions within the specified latitude band

    """
    ocean_projection = fingerprint.ocean_projection(value=0)
    altimetry_projection = fingerprint.altimetry_projection(
        latitude_min=latitude_min,
        latitude_max=latitude_max,
        value=0,
    )
    combined_projection = ocean_projection * altimetry_projection
    normalization_integral = fingerprint.integrate(
        combined_projection
    )
    weighting_function = combined_projection / normalization_integral
    return weighting_function
