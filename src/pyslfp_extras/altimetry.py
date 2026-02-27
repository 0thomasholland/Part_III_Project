from __future__ import annotations

import numpy as np
from pygeoinf import HilbertSpace, LinearOperator
from pyslfp import FingerPrint


class GridPoints:
    """
    Represents a set of grid points on the Earth's surface filtered by
    a projection mask.

    Parameters
    ----------
    projection : pyshtools.SHGrid
        A projection grid (e.g. from ``fp.ocean_projection(value=0)``).
    degree_spacing : float, optional
        Spacing between evaluation points in degrees, by default 5.0.
    threshold : float, optional
        Minimum value from the SH expansion to classify a point as valid,
        by default 0.5. The projection is a binary mask (0/1), but
        a threshold of 0.5 cleanly separates the two classes incase of any
        artifacting.
    """

    def __init__(
        self,
        projection,
        degree_spacing: float = 5.0,
        threshold: float = 0.5,
    ):
        self._degree_spacing = degree_spacing
        self._threshold = threshold
        self._lats, self._lons = self._compute_points(
            projection
        )

    def _compute_points(
        self, projection
    ) -> tuple[list[float], list[float]]:
        # Build the regular grid of candidate points.
        target_lats = np.arange(
            90,
            -90 - self._degree_spacing,
            -self._degree_spacing,
        )
        target_lons = np.arange(
            0, 360, self._degree_spacing
        )
        lon_grid, lat_grid = np.meshgrid(
            target_lons, target_lats
        )
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()

        # Evaluate the projection mask at every candidate point via SH expansion.
        coeffs = projection.expand()
        values = coeffs.expand(lat=flat_lats, lon=flat_lons)

        # Filter: projection values near 1 indicate valid points.
        valid = values >= self._threshold

        return flat_lats[valid].tolist(), flat_lons[
            valid
        ].tolist()

    @property
    def lats(self) -> list[float]:
        """Latitudes of the filtered grid points."""
        return self._lats

    @property
    def lons(self) -> list[float]:
        """Longitudes of the filtered grid points."""
        return self._lons

    @property
    def coords(self) -> list[tuple[float, float]]:
        """List of (latitude, longitude) tuples for the filtered grid points."""
        return list(zip(self._lats, self._lons))

    def point_evaluation_operator(
        self, measurement_space: HilbertSpace
    ) -> LinearOperator:
        """
        Constructs a linear operator that evaluates the surface height at
        the filtered grid points.
        """
        return measurement_space.point_evaluation_operator(
            self.coords
        )

    def __len__(self) -> int:
        return len(self._lats)

    def __repr__(self) -> str:
        return f"GridPoints(n_points={len(self)}, degree_spacing={self._degree_spacing})"

    # ------------------------------------------------------------------ #
    #  Factory methods for common projections                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def ocean(
        cls,
        finger_print: FingerPrint,
        degree_spacing: float = 5.0,
    ) -> GridPoints:
        """Grid points over the ocean."""
        projection = finger_print.ocean_projection(value=0)
        return cls(
            projection, degree_spacing=degree_spacing
        )

    @classmethod
    def ocean_altimetry(
        cls,
        finger_print: FingerPrint,
        degree_spacing: float = 5.0,
        latitude_range: float = 66.0,
    ) -> GridPoints:
        """Grid points within the altimetry coverage band."""
        projection = finger_print.altimetry_projection(
            latitude_max=latitude_range,
            latitude_min=-latitude_range,
            value=0,
        )
        return cls(
            projection, degree_spacing=degree_spacing
        )

    @classmethod
    def ice(
        cls,
        finger_print: FingerPrint,
        degree_spacing: float = 5.0,
    ) -> GridPoints:
        """Grid points over ice regions."""
        projection = finger_print.ice_projection(value=0)
        return cls(
            projection, degree_spacing=degree_spacing
        )


def altimetry_error_gaussian_measure(
    measurement_space: Sobolev,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    finger_print: FingerPrint | None = None,
    altimetry_operator: LinearOperator | None = None,
) -> GaussianMeasure:
    """Create a Gaussian measure for altimetry observation errors on the SSH measurement space.

    Args:
        measurement_space: The SSH space (e.g. from sea_surface_height_operator.codomain).
        order: Sobolev kernel order. Default 1.5.
        length_scale: Correlation length scale. Default 0.005 * fp.mean_sea_floor_radius.
        amplitude: Pointwise standard deviation. Default 0.0001 / fp.length_scale.
        finger_print: FingerPrint instance used for default parameter scaling.
            Required if length_scale or amplitude are not provided.
        altimetry_operator: Optional operator for spatial masking (e.g. latitude bounds).

    Returns:
        A GaussianMeasure for altimetry observation errors.
    """
    if length_scale is None or amplitude is None:
        if finger_print is None:
            raise ValueError(
                "finger_print is required when length_scale or amplitude are not provided."
            )
        if length_scale is None:
            length_scale = (
                0.005 * finger_print.mean_sea_floor_radius
            )
        if amplitude is None:
            amplitude = 0.0001 / finger_print.length_scale

    base_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, length_scale, amplitude
    )

    if altimetry_operator is not None:
        return base_measure.affine_mapping(
            operator=altimetry_operator
        )

    return base_measure
