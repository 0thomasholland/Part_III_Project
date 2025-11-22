"""Sea surface height operators for Part_III_Project.

This module extends pyslfp's FingerPrint class to compute sea surface height
(SSH) changes instead of sea level changes, and provides utility functions
for SSH calculations.

Classes:
    SeaSurfaceHeightFingerPrint: Extended FingerPrint returning SSH

Functions:
    compute_sea_surface_height_change: Calculate SSH from components
"""

from typing import Optional, Tuple

import numpy as np
import pyslfp as sl
from pygeoinf import LinearOperator
from pygeoinf.symmetric_space.sphere import Lebesgue, Sobolev
from pyshtools import SHGrid
from pyslfp import FingerPrint
from pyslfp.physical_parameters import GRAVITATIONAL_ACCELERATION


class SeaSurfaceHeightFingerPrint(sl.FingerPrint):
    """FingerPrint that computes sea surface height instead of sea level.

    This class extends pyslfp.FingerPrint to return sea surface height (SSH)
    by adding surface displacement to sea level change. SSH is the directly
    observable quantity measured by satellite altimetry.
    """

    def __call__(
        self,
        /,
        *,
        direct_load: SHGrid | None = None,
        displacement_load: SHGrid | None = None,
        gravitational_potential_load: SHGrid | None = None,
        angular_momentum_change: np.ndarray | None = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        verbose: bool = False,
    ) -> tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """Solve sea level equation and return sea surface height.

        Extends parent FingerPrint to add displacement to sea level change,
        yielding the directly observable sea surface height.

        Args:
            direct_load: Direct surface load (ice thickness change)
            displacement_load: Load affecting surface displacement
            gravitational_potential_load: Load affecting gravitational potential
            angular_momentum_change: Change in angular momentum vector
            rotational_feedbacks: Include rotational feedback effects
            rtol: Relative tolerance for iterative solver
            verbose: Print iteration information

        Returns:
            Tuple containing:
                - sea_surface_height_change: Observable SSH change
                - displacement: Vertical surface displacement
                - gravity_potential_change: Change in gravity potential
                - angular_velocity_change: Change in rotation vector

        """
        # Call parent class to get sea level change
        (
            sea_level_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        ) = super().__call__(
            direct_load=direct_load,
            displacement_load=displacement_load,
            gravitational_potential_load=gravitational_potential_load,
            angular_momentum_change=angular_momentum_change,
            rotational_feedbacks=rotational_feedbacks,
            rtol=rtol,
            verbose=verbose,
        )

        # Convert to sea surface height
        sea_surface_height_change = sea_level_change + displacement

        return (
            sea_surface_height_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        )


def compute_sea_surface_height_change(
    finger_print: sl.FingerPrint,
    sea_level_change: SHGrid,
    displacement: SHGrid,
    angular_velocity_change: np.ndarray,
) -> SHGrid:
    """Calculate sea surface height change from sea level equation components.

    Computes SSH by combining sea level change, surface displacement, and
    centrifugal potential change due to Earth rotation variations.

    Args:
        finger_print: FingerPrint object with Earth model parameters
        sea_level_change: Relative sea level change from sea level equation
        displacement: Vertical surface displacement
        angular_velocity_change: Change in Earth's angular velocity vector

    Returns:
        Sea surface height change as SHGrid

    Note:
        SSH = SLC + displacement + (centrifugal_potential / g)
        This is the quantity directly observable by satellite altimetry.

    """
    return (
        sea_level_change
        + displacement
        + (
            finger_print.centrifugal_potential_change(
                angular_velocity_change=angular_velocity_change,
            )
            / GRAVITATIONAL_ACCELERATION
        )
    )


#  def lat_long_altimetry_projection(
#         fingerprint: FingerPrint,
#         latitude_min: float = -66,
#         latitude_max: float = 66,
#         longitude_min: float | None = None,
#         longitude_max: float | None = None,
#         value: float = np.nan,
#     ) -> SHGrid:
#         """
#         Returns a grid that is 1 in the oceans between specified latitudes
#         (typical for satellite altimetry) and `value` elsewhere.

#         Parameters
#         ----------
#         latitude_min : float
#             Minimum latitude in degrees. Default is -66.
#         latitude_max : float
#             Maximum latitude in degrees. Default is 66.
#         longitude_min : float, optional
#             Minimum longitude in degrees [-180, 180]. If None, no longitude
#             restriction is applied.
#         longitude_max : float, optional
#             Maximum longitude in degrees [-180, 180]. If None, no longitude
#             restriction is applied.
#         value : float
#             Value to assign outside the region. Default is NaN.

#         Returns
#         -------
#         SHGrid
#             Grid with 1 in the specified ocean region and `value` elsewhere.
#         """
#         lats, lons = np.meshgrid(fingerprint.lats(), fingerprint.lons(), indexing="ij")
#         ocean_mask = fingerprint.ocean_function.data > 0
#         lat_mask = np.logical_and(lats > latitude_min, lats < latitude_max)

#         # Combine with longitude mask if specified
#         if longitude_min is not None and longitude_max is not None:
#             if longitude_min < longitude_max:
#                 lon_mask = np.logical_and(lons >= longitude_min, lons < longitude_max)
#             else:
#                 # Handle wrap-around (e.g., longitude_min=170, longitude_max=-170)
#                 lon_mask = np.logical_or(lons >= longitude_min, lons < longitude_max)
#             combined_mask = np.logical_and(ocean_mask, np.logical_and(lat_mask, lon_mask))
#         else:
#             combined_mask = np.logical_and(ocean_mask, lat_mask)

#         return SHGrid.from_array(
#             np.where(combined_mask, 1, value), grid=fingerprint.grid
#         )


# def binned_averaging_operator(
#     fingerprint: FingerPrint,
#     load_space: Union[Lebesgue, Sobolev],
#     /,
#     *,
#     grid_size_lat: float,
#     grid_size_lon: float,
#     latitude_min: float = -66,
#     latitude_max: float = 66,
#     longitude_min: float | None = None,
#     longitude_max: float | None = None,
# ) -> Tuple[LinearOperator, SHGrid, List[Tuple[float, float]]]:
#     """
#     Creates an operator that bins ocean data into grid cells and computes
#     spatial averages within each bin.

#     The operator computes the average value of a function over rectangular
#     ocean bins, using L2 integration. Each bin is defined by its lat/lon
#     extent, and only ocean areas within the specified region are included.

#     Args:
#         finger_print: The FingerPrint object.
#         load_space: The Hilbert space for the input function. Must be a
#             Lebesgue or Sobolev space.
#         grid_size_lat: The size of each bin in the latitudinal direction (degrees).
#         grid_size_lon: The size of each bin in the longitudinal direction (degrees).
#         latitude_min: Minimum latitude for bins (degrees). Defaults to -66.
#         latitude_max: Maximum latitude for bins (degrees). Defaults to 66.
#         longitude_min: Minimum longitude for bins (degrees [-180, 180]).
#             If None, no longitude restriction. Defaults to None.
#         longitude_max: Maximum longitude for bins (degrees [-180, 180]).
#             If None, no longitude restriction. Defaults to None.
#         exclude_ice_shelves: If True, exclude ice-shelved regions. Defaults to False.

#     Returns:
#         A tuple containing:
#         - operator: LinearOperator that maps from load_space to binned averages
#         - bin_centers_grid: SHGrid with NaN everywhere except at valid bin centers,
#           which are set to their linear index
#         - bin_centers_list: List of (lat, lon) tuples for each bin center
#     """
#     if not isinstance(load_space, (Lebesgue, Sobolev)):
#         raise TypeError("load_space must be a Lebesgue or Sobolev space.")

#     # Get the ocean mask for the specified region
#     ocean_mask = lat_long_altimetry_projection(
#         fingerprint,
#         latitude_min=latitude_min,
#         latitude_max=latitude_max,
#         longitude_min=longitude_min,
#         longitude_max=longitude_max,
#         value=0,
#     )

#     # Create bin edges
#     lat_edges = np.arange(latitude_min, latitude_max + grid_size_lat, grid_size_lat)

#     # Handle longitude edges based on whether we have longitude constraints
#     if longitude_min is not None and longitude_max is not None:
#         if longitude_min < longitude_max:
#             lon_edges = np.arange(longitude_min, longitude_max + grid_size_lon, grid_size_lon)
#         else:
#             # Handle wrap-around case
#             # Create bins from longitude_min to 180, then from -180 to longitude_max
#             lon_edges_1 = np.arange(longitude_min, 180 + grid_size_lon, grid_size_lon)
#             lon_edges_2 = np.arange(-180, longitude_max + grid_size_lon, grid_size_lon)
#             lon_edges = np.concatenate([lon_edges_1, lon_edges_2[1:]])  # Avoid duplicate at boundary
#     else:
#         lon_edges = np.arange(-180, 180 + grid_size_lon, grid_size_lon)

#     # Compute bin centers
#     lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
#     lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2

#     # Create weighting functions for each bin and track valid bins
#     weighting_functions = []
#     bin_centers_list = []
#     bin_center_indices = np.full(
#         (len(lat_centers), len(lon_centers)), np.nan
#     )

#     lats, lons = np.meshgrid(
#         finger_print.lats(), finger_print.lons(), indexing="ij"
#     )

#     for i, lat_c in enumerate(lat_centers):
#         for j, lon_c in enumerate(lon_centers):
#             # Define the bin extent
#             lat_min_bin = lat_edges[i]
#             lat_max_bin = lat_edges[i + 1]
#             lon_min_bin = lon_edges[j]
#             lon_max_bin = lon_edges[j + 1]

#             # Create a mask for this bin using the projection method
#             bin_mask = finger_print.lat_long_altimetry_projection(
#                 latitude_min=lat_min_bin,
#                 latitude_max=lat_max_bin,
#                 longitude_min=lon_min_bin,
#                 longitude_max=lon_max_bin,
#                 value=0,
#             )

#             # Combine with the overall ocean mask
#             bin_ocean_mask = bin_mask.data * (ocean_mask.data > 0)

#             # Only include bins that have ocean coverage
#             if np.any(bin_ocean_mask):
#                 # Normalize the weighting function by the total ocean area in the bin
#                 # This makes the operator compute the average rather than the integral
#                 bin_area = np.sum(bin_ocean_mask)
#                 weighting_function = SHGrid.from_array(
#                     bin_ocean_mask / bin_area, grid=ocean_mask.grid
#                 )

#                 weighting_functions.append(weighting_function)
#                 bin_centers_list.append((lat_c, lon_c))
#                 bin_center_indices[i, j] = len(bin_centers_list) - 1

#     # Create the averaging operator using the existing function
#     averaging_op = averaging_operator(load_space, weighting_functions)

#     # Create a grid showing where the bin centers are
#     bin_centers_grid = SHGrid.from_array(
#         bin_center_indices, grid=ocean_mask.grid
#     )

#     return averaging_op, bin_centers_grid, bin_centers_list
