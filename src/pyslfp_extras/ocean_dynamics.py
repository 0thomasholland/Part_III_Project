from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from importlib import resources
from typing import Optional

import numpy as np
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ocean_projection_operator,
    remove_ocean_average_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from pyslfp_extras.gmsl import (
    GMSLOperatorBase,
)


class OceanDynamics(GMSLOperatorBase):
    """Gaussian measures for non-ice ocean dynamic topography (ODT) SSH variability.

    Models the spatial structure of SSH variability not attributable to ice-sheet
    driven sea level change or instrument error — i.e. ocean dynamics, steric
    effects, and circulation changes.

    The measure is constructed as a rotationally invariant Sobolev kernel Gaussian
    measure, scaled to a target pointwise std, multiplied by a normalised [0,1]
    spatial weight field from the chosen pattern. All patterns expose the same
    contract: `spatial_field()` returns weights in [0,1] over ocean points.

    The internal chain is:

        base measure → height (ocean, weighted, zero-avg) → load
                                                          → SSH (direct + fingerprint)

    Height and load/SSH branch from the same point, ensuring consistent samples.

    Parameters
    ----------
    finger_print:
        The FingerPrint instance.
    finger_print_operator:
        Linear operator mapping loads to the response space.
    std:
        Target pointwise standard deviation for ODT SSH variability (metres).
        Default 0.003.
    length_scale:
        Correlation length scale in km. Default 5000.
    pattern:
        An OceanDynamics.VariabilityPattern instance providing normalised [0,1]
        spatial weights. If None, UniformPattern() is used.
    altimetry_latitude_range:
        Latitude cutoff for altimetry coverage used in SSH/GMSL estimates.
        Default 66.0.
    point_degree_spacing:
        Degree spacing for point-altimetry sampling. Default 5.0.
    parallel_workers:
        Optional thread count for point sampling.

    Examples
    --------
    >>> od = OceanDynamics(fp, op, std=0.003)
    >>> od_data = OceanDynamics(fp, op, std=0.005, pattern=OceanDynamics.DataPattern())
    >>> od_synth = OceanDynamics(fp, op, pattern=OceanDynamics.SyntheticPattern())
    >>> load_sample, ssh_sample = od.sample()
    """

    # -----------------------------------------------------------------------
    # Nested pattern classes
    # -----------------------------------------------------------------------

    class VariabilityPattern(ABC):
        """Abstract base for spatial variability patterns.

        All subclasses must return a normalised [0,1] SHGrid from
        `spatial_field()`, defined over ocean points. Amplitude is
        controlled entirely by OceanDynamics.std.
        """

        @abstractmethod
        def spatial_field(
            self, finger_print: FingerPrint
        ) -> SHGrid:
            """Return a dimensionless SHGrid of spatial weights in [0,1].

            Normalisation should be over ocean points only, so the maximum
            ocean value maps to 1.
            """
            ...

    class UniformPattern(VariabilityPattern):
        """Rotationally invariant — uniform weight of 1 over the ocean."""

        def spatial_field(
            self, finger_print: FingerPrint
        ) -> SHGrid:
            return finger_print.ocean_projection(value=0)

    class SyntheticPattern(VariabilityPattern):
        """Synthetic spatial weights based on observed oceanographic patterns.

        Gaussian blobs are placed at major current systems (Gulf Stream,
        Kuroshio, ACC etc.), summed, masked to ocean, then normalised to
        [0,1] over ocean points.

        Parameters
        ----------
        point_multiplier:
            Relative peak amplitude in major current systems before
            normalisation. Higher values increase contrast between
            high/low variability regions. Default 20.
        """

        def __init__(self, point_multiplier: float = 20):
            self.point_multiplier = point_multiplier

        @staticmethod
        def _gaussian_blob(
            lat_grid,
            lon_grid,
            lat0,
            lon0,
            lat_width,
            lon_width,
        ) -> np.ndarray:
            dlat = lat_grid - lat0
            dlon = (
                np.mod((lon_grid - lon0) + 180, 360) - 180
            )
            return np.exp(
                -0.5
                * (
                    (dlat / lat_width) ** 2
                    + (dlon / lon_width) ** 2
                )
            )

        def spatial_field(
            self, finger_print: FingerPrint
        ) -> SHGrid:
            lats = finger_print.lats()
            lons = finger_print.lons()
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            b = self._gaussian_blob
            pm = self.point_multiplier
            field = np.zeros_like(lat_grid)

            # Gulf Stream
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, 35, 305, 20, 20)
            )
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, 45, 320, 20, 30)
            )
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, 35, 305, 8, 8)
            )
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, 45, 320, 8, 8)
            )
            # South America / Brazil Current
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, -45, 305, 20, 30)
            )
            field += (
                pm
                * 0.3
                * b(lat_grid, lon_grid, -45, 295, 6, 6)
            )
            # Kuroshio
            field += (
                pm
                * 0.6
                * b(lat_grid, lon_grid, 35, 155, 10, 30)
            )
            # Agulhas
            field += (
                pm
                * 0.6
                * b(lat_grid, lon_grid, -35, 35, 10, 15)
            )
            # East Australian Current
            field += (
                pm
                * 0.8
                * b(lat_grid, lon_grid, -30, 150, 10, 15)
            )
            # Antarctic Circumpolar Current
            field += (
                pm
                * 0.6
                * b(lat_grid, lon_grid, -55, 180, 10, 60)
            )
            # Equatorial Pacific
            field += (
                pm
                * 0.7
                * b(lat_grid, lon_grid, 10, 240, 5, 30)
            )
            # Indonesian Throughflow
            field += (
                pm
                * 0.6
                * b(lat_grid, lon_grid, -5, 120, 5, 10)
            )
            # Arabian Sea / Afar
            field += (
                pm
                * 0.7
                * b(lat_grid, lon_grid, 25, 40, 15, 15)
            )

            # Mask to ocean and normalise to [0,1] over ocean points
            ocean_mask = finger_print.ocean_projection(
                value=0
            ).to_array()
            field = field * ocean_mask
            ocean_vals = field[ocean_mask > 0]
            if ocean_vals.max() > 0:
                field = field / ocean_vals.max()

            grid = finger_print.zero_grid()
            grid.data[:, :] = field
            return grid

    class DataPattern(VariabilityPattern):
        """Spatial weights loaded from a pre-computed altimetry field.

        Loads an NPZ field from the package's data/altimetry directory,
        masks to ocean, then normalises to [0,1] over ocean points. When
        no filename is provided, the lmax is inferred from the fingerprint
        and the file `sla_diff_std_lmax{lmax}.npz` is selected.

        Parameters
        ----------
        filename:
            Optional file name within package data/altimetry. If None,
            uses the inferred lmax to load `sla_diff_std_lmax{lmax}.npz`.
        """

        def __init__(self, filename: Optional[str] = None):
            self.filename = filename

        def spatial_field(
            self, finger_print: FingerPrint
        ) -> SHGrid:
            if self.filename is not None:
                data_path = resources.files(
                    __package__
                ).joinpath(
                    f"data/altimetry/{self.filename}"
                )
            else:
                lmax = finger_print.lmax
                data_path = resources.files(
                    __package__
                ).joinpath(
                    f"data/altimetry/sla_diff_std_lmax{lmax}.npz"
                )

            with np.load(str(data_path)) as data:
                if "sla_diff_std" not in data:
                    raise KeyError(
                        "Expected key 'sla_diff_std' in altimetry NPZ."
                    )
                raw = data["sla_diff_std"]

            ocean_mask = finger_print.ocean_projection(
                value=0
            ).to_array()

            if raw.shape != ocean_mask.shape:
                raise ValueError(
                    "Altimetry field shape does not match fingerprint grid."
                )

            field = raw * ocean_mask

            # Normalise to [0,1] over ocean points only
            ocean_vals = field[ocean_mask > 0]
            v_min, v_max = (
                ocean_vals.min(),
                ocean_vals.max(),
            )
            if v_max > v_min:
                field = (field - v_min) / (v_max - v_min)
                field = (
                    field * ocean_mask
                )  # re-apply mask after shift

            grid = finger_print.zero_grid()
            grid.data[:, :] = field
            return grid

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        std: float = 0.003,
        length_scale: Optional[float] = 5000,
        pattern: Optional[VariabilityPattern] = None,
        altimetry_latitude_range: float = 66.0,
        point_degree_spacing: float = 5.0,
        parallel_workers: None | int = None,
    ):
        super().__init__(
            finger_print,
            finger_print_operator,
            altimetry_latitude_range=altimetry_latitude_range,
            point_degree_spacing=point_degree_spacing,
        )
        self._fp = finger_print
        self._op = finger_print_operator
        self._std = std
        self._length_scale = length_scale
        self._pattern = (
            pattern
            if pattern is not None
            else OceanDynamics.UniformPattern()
        )

    # -----------------------------------------------------------------------
    # Operators
    # -----------------------------------------------------------------------

    @cached_property
    def _load_space(self) -> HilbertSpace:
        return self._op.domain

    @cached_property
    def _base_measure(self) -> GaussianMeasure:
        """Rotationally invariant heat kernel measure scaled to target std."""
        return self._load_space.point_value_scaled_heat_kernel_gaussian_measure(
            self._length_scale, self._std
        )

    @cached_property
    def _height_op(self) -> LinearOperator:
        """Base measure space → ocean-projected, spatially weighted, zero-average height."""
        spatial_op = spatial_mutliplication_operator(
            self._pattern.spatial_field(self._fp),
            self._load_space,
        )
        ocean_proj_op = ocean_projection_operator(
            self._fp, self._load_space
        )
        remove_avg_op = remove_ocean_average_operator(
            self._fp, self._load_space
        )
        return remove_avg_op @ spatial_op @ ocean_proj_op

    @cached_property
    def _height_to_load_op(self) -> LinearOperator:
        """Physical height change → equivalent mass load."""
        return sea_level_change_to_load_operator(
            self._fp, self._load_space
        )

    @cached_property
    def _height_to_ssh_op(self) -> LinearOperator:
        """Physical height change → total SSH (direct change + fingerprint response).

        The fingerprint response accounts for the gravitational, rotational,
        and deformational effects of the redistributed ocean mass on the
        sea surface. The total SSH is the sum of the direct height change
        and this indirect response.
        """
        ssh_op = sea_surface_height_operator(
            self._fp, self._op.codomain
        )
        fingerprint_response = (
            ssh_op @ self._op @ self._height_to_load_op
        )
        return (
            fingerprint_response
            + self._load_space.identity_operator()
        )

    @cached_property
    def _load_to_ssh_operator(self) -> LinearOperator:
        """Height (= load) space → total SSH. Satisfies GMSLOperatorBase contract.

        For ODT, the 'load space' is the same Hilbert space as the height space,
        so this delegates directly to _height_to_ssh_op.
        """
        return self._height_to_ssh_op

    @cached_property
    def _height_to_slc_op(self) -> LinearOperator:
        """Physical height change → sea level change (fingerprint response).

        Converts height to an equivalent mass load, then applies the fingerprint
        operator to obtain the sea level change field. Analogous to
        IceSheetChange.load_to_slc_operator.
        """
        return self._op @ self._height_to_load_op

    @cached_property
    def _gmsl_operator(self) -> LinearOperator:
        """Maps from the base measure space to scalar GMSL."""
        ocean_projection = self._fp.ocean_projection(
            value=0
        )
        weighting = ocean_projection / self._fp.integrate(
            ocean_projection
        )
        return (
            averaging_operator(
                self._height_to_ssh_op.codomain, [weighting]
            )
            @ self._height_to_ssh_op
            @ self._height_op
        )

    # -----------------------------------------------------------------------
    # Measures
    # -----------------------------------------------------------------------

    @cached_property
    def height_measure(self) -> GaussianMeasure:
        """ODT variability as physical height change.

        Ocean-projected, spatially weighted by the chosen pattern,
        with zero ocean average, scaled to the target pointwise std.
        """
        return self._base_measure.affine_mapping(
            operator=self._height_op
        )

    @cached_property
    def load_measure(self) -> GaussianMeasure:
        """ODT variability as equivalent mass load."""
        return self.height_measure.affine_mapping(
            operator=self._height_to_load_op
        )

    @cached_property
    def ssh_measure(self) -> GaussianMeasure:
        """Total SSH from ODT (direct height change + fingerprint response)."""
        return self.height_measure.affine_mapping(
            operator=self._height_to_ssh_op
        )

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(self) -> tuple[SHGrid, SHGrid]:
        """Draw consistent load and SSH samples from a single height realisation."""
        height_sample = self.height_measure.sample()
        load_sample = self._height_to_load_op(height_sample)
        ssh_sample = self._height_to_ssh_op(height_sample)
        return (load_sample, ssh_sample)
