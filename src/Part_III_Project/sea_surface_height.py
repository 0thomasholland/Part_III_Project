from typing import Optional, Tuple

import numpy as np
import pyslfp as sl
from pyshtools import SHGrid


def dSL_to_dSSH(delta_sea_level, delta_gravity_potential):
    delta_sea_surface = delta_sea_level + (delta_gravity_potential / 9.81)
    return delta_sea_surface

class SeaSurfaceFingerPrint(sl.FingerPrint):
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
        Solves the generalized sea level equation for a given load.

        Args:
            direct_load: The direct surface mass load (e.g., from ice melt).
            displacement_load: An externally imposed displacement load.
            gravitational_potential_load: An externally imposed gravitational potential load.
            angular_momentum_change: An externally imposed change in angular momentum.
            rotational_feedbacks: If True, include the effects of polar wander.
            rtol: The relative tolerance for the iterative solver to determine convergence.
            verbose: If True, print the relative error at each iteration.

        Returns:
            A tuple containing:
                - `sea_surface_height_change` (SHGrid): The self-consistent sea surface height change.
                - `displacement` (SHGrid): The vertical surface displacement.
                - `gravity_potential_change` (SHGrid): Change in gravity potential.
                - `angular_velocity_change` (np.ndarray): Change in angular velocity `[ω_x, ω_y]`.
        """
        loads_present = False
        non_zero_rhs = False

        if direct_load is not None:
            loads_present = True
            assert self.check_field(direct_load)
            mean_sea_level_change = -self.integrate(direct_load) / (
                self.water_density * self.ocean_area
            )
            non_zero_rhs = non_zero_rhs or np.max(np.abs(direct_load.data)) > 0

        else:
            direct_load = self.zero_grid()
            mean_sea_level_change = 0

        if displacement_load is not None:
            loads_present = True
            assert self.check_field(displacement_load)
            displacement_load_lm = self.expand_field(displacement_load)
            non_zero_rhs = non_zero_rhs or np.max(np.abs(displacement_load.data)) > 0

        if gravitational_potential_load is not None:
            loads_present = True
            assert self.check_field(gravitational_potential_load)
            gravitational_potential_load_lm = self.expand_field(
                gravitational_potential_load
            )
            non_zero_rhs = (
                non_zero_rhs or np.max(np.abs(gravitational_potential_load.data)) > 0
            )

        if angular_momentum_change is not None:
            loads_present = True
            non_zero_rhs = non_zero_rhs or np.max(np.abs(angular_momentum_change)) > 0

        if loads_present is False or not non_zero_rhs:
            return self.zero_grid(), self.zero_grid(), self.zero_grid(), np.zeros(2)

        self._solver_counter += 1

        load = (
            direct_load
            + self.water_density * self.ocean_function * mean_sea_level_change
        )

        angular_velocity_change = np.zeros(2)

        g = self.gravitational_acceleration
        r = self._rotation_factor
        i = self._inertia_factor
        m = 1 / (self.polar_moment_of_inertia - self.equatorial_moment_of_inertia)
        ht = self._ht[2]
        kt = self._kt[2]

        err = 1
        count = 0
        count_print = 0
        while err > rtol:
            displacement_lm = self.expand_field(load)
            gravity_potential_change_lm = displacement_lm.copy()

            for l in range(self.lmax + 1):
                displacement_lm.coeffs[:, l, :] *= self._h[l]
                gravity_potential_change_lm.coeffs[:, l, :] *= self._k[l]

                if displacement_load is not None:
                    displacement_lm.coeffs[:, l, :] += (
                        self._h_u[l] * displacement_load_lm.coeffs[:, l, :]
                    )

                    gravity_potential_change_lm.coeffs[:, l, :] += (
                        self._k_u[l] * displacement_load_lm.coeffs[:, l, :]
                    )

                if gravitational_potential_load is not None:
                    displacement_lm.coeffs[:, l, :] += (
                        self._h_phi[l] * gravitational_potential_load_lm.coeffs[:, l, :]
                    )

                    gravity_potential_change_lm.coeffs[:, l, :] += (
                        self._k_phi[l] * gravitational_potential_load_lm.coeffs[:, l, :]
                    )

            if rotational_feedbacks:
                centrifugal_coeffs = r * angular_velocity_change

                displacement_lm.coeffs[:, 2, 1] += ht * centrifugal_coeffs
                gravity_potential_change_lm.coeffs[:, 2, 1] += kt * centrifugal_coeffs

                angular_velocity_change = (
                    i * gravity_potential_change_lm.coeffs[:, 2, 1]
                )

                if angular_momentum_change is not None:
                    angular_velocity_change -= m * angular_momentum_change

                gravity_potential_change_lm.coeffs[:, 2, 1] += (
                    r * angular_velocity_change
                )

            displacement = self.expand_coefficient(displacement_lm)
            gravity_potential_change = self.expand_coefficient(
                gravity_potential_change_lm
            )

            sea_level_change = (-1 / g) * (g * displacement + gravity_potential_change)
            sea_level_change.data += mean_sea_level_change - self.ocean_average(
                sea_level_change
            )

            load_new = (
                direct_load
                + self.water_density * self.ocean_function * sea_level_change
            )
            if count > 1 or mean_sea_level_change != 0:
                err = np.max(np.abs((load_new - load).data)) / np.max(np.abs(load.data))
                if verbose:
                    count_print += 1
                    print(f"Iteration = {count_print}, relative error = {err:6.4e}")

            load = load_new
            count += 1

        sea_surface_height_change = dSL_to_dSSH(
            sea_level_change, gravity_potential_change
        )

        return (
            sea_surface_height_change,
            displacement,
            gravity_potential_change,
            angular_velocity_change,
        )