from typing import Optional, Tuple

import numpy as np
import pyslfp as sl
from pyshtools import SHGrid


class SeaSurfaceFingerPrint(sl.FingerPrint):
    """
    Extends FingerPrint to return sea surface height instead of sea level.
    """

    def __call__(
        self,
        /,
        *,
        direct_load: Optional[SHGrid] = None,
        displacement_load: Optional[SHGrid] = None,
        gravitational_potential_load: Optional[SHGrid] = None,
        angular_momentum_change: Optional[np.ndarray] = None,
        rotational_feedbacks: bool = True,
        rtol: float = 1.0e-6,
        verbose: bool = False,
    ) -> Tuple[SHGrid, SHGrid, SHGrid, np.ndarray]:
        """
        Solves the sea level equation and returns sea surface height.

        Returns:
            A tuple containing:
                - `sea_surface_height_change` (SHGrid): Sea surface height change.
                - `displacement` (SHGrid): The vertical surface displacement.
                - `gravity_potential_change` (SHGrid): Change in gravity potential.
                - `angular_velocity_change` (np.ndarray): Change in angular velocity.
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