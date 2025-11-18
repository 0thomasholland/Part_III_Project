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
from pyshtools import SHGrid
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
