"""
Load generating utilities for Part_III_Project

Main functions:
- create_ice_band: Create an ice load over a latitude band


"""

import pyshtools as pysh


def create_ice_band(
    lat_center: float,
    lat_width: float = 1.0,
    ice_thickness: float = 100.0,
    lmax: int = 360,
    grid: str = "DH",
    sampling: int = 1,
) -> pysh.SHGrid:
    """
    Create an ice load over a latitude band.

    Args:
        lat_center: Center latitude of the band (degrees)
        lat_width: Half-width of the band (degrees). Default 1.0 for 2° total width
        ice_thickness: Ice thickness in meters
        lmax: Maximum spherical harmonic degree
        grid: pyshtools grid type
        sampling: pyshtools sampling

    Returns:
        SHGrid with ice load in meters
    """
    # Create empty grid
    ice_grid = pysh.SHGrid.from_zeros(lmax, grid=grid, sampling=sampling)

    # Get latitude array
    lats = ice_grid.lats()

    # Create mask for the latitude band
    mask = (lats >= lat_center - lat_width) & (lats <= lat_center + lat_width)

    # Set ice thickness uniformly across longitudes in the band
    ice_grid.data[mask, :] = ice_thickness

    return ice_grid
