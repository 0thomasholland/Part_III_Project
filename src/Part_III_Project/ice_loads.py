"""Ice load generation utilities for Part_III_Project.

This module provides functions for creating ice load distributions
on the sphere, suitable for use with pyslfp FingerPrint objects.

Functions:
    create_ice_load_latitude_band: Create ice load over a latitude band
"""

import pyshtools as pysh


def create_ice_load_latitude_band(
    lat_center: float,
    lat_width: float = 1.0,
    ice_thickness: float = 100.0,
    lmax: int = 360,
    grid: str = "DH",
    sampling: int = 1,
) -> pysh.SHGrid:
    """Create an ice load distributed uniformly over a latitude band.

    Generates a spherical harmonic grid representing ice thickness
    concentrated in a band around a specified latitude.

    Args:
        lat_center: Center latitude of the band in degrees
        lat_width: Half-width of band in degrees (total width is 2*lat_width)
        ice_thickness: Uniform ice thickness in meters within the band
        lmax: Maximum spherical harmonic degree for representation
        grid: Spherical harmonic grid type ('DH' or 'GLQ')
        sampling: Grid sampling (1 for standard, 2 for oversampling)

    Returns:
        SHGrid object with ice thickness in meters, zero outside the band

    Example:
        >>> ice = create_ice_load_latitude_band(70, lat_width=5, ice_thickness=100)
        >>> # Creates 100m ice load from 65°N to 75°N

    """
    # Create empty grid
    ice_grid = pysh.SHGrid.from_zeros(
        lmax, grid=grid, sampling=sampling
    )

    # Get latitude array
    lats = ice_grid.lats()

    # Create mask for the latitude band
    mask = (lats >= lat_center - lat_width) & (
        lats <= lat_center + lat_width
    )

    # Set ice thickness uniformly across longitudes in the band
    ice_grid.data[mask, :] = ice_thickness

    return ice_grid
