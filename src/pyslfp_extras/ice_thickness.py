from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

import numpy as np
from pygeoinf import GaussianMeasure, LinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from pygeoinf_extras import standard_dev
from pyslfp_extras.gmsl import (
    GMSLOperatorBase,
    gmsl_from_ice_thickness_operator,
)

# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class IceSheetChangeSample:
    ice_thickness: SHGrid
    firn_thickness: SHGrid
    total_thickness: SHGrid

    ice_load: SHGrid
    firn_load: SHGrid
    total_load: SHGrid

    ice_slc: SHGrid
    firn_slc: SHGrid
    total_slc: SHGrid

    ice_ssh: SHGrid
    firn_ssh: SHGrid
    total_ssh: SHGrid


# ---------------------------------------------------------------------------
# Operator mixin
# ---------------------------------------------------------------------------


class IceThicknessGMSLOperators(GMSLOperatorBase):
    """Operators mapping thickness/load to SLC, SSH, and GMSL."""

    # -----------------------------------------------------------------------
    # Abstract implementations required by GMSLOperatorBase
    # -----------------------------------------------------------------------
    @cached_property
    def _thickness_to_load_op(self) -> LinearOperator:
        return ice_thickness_change_to_load_operator(
            self._fp, self._op.domain
        )

    @cached_property
    def _load_to_ssh_operator(self) -> LinearOperator:
        ssh_op = sea_surface_height_operator(
            self._fp, self._op.codomain
        )
        return (
            ssh_op @ self._op @ self._thickness_to_load_op
        )

    @cached_property
    def _gmsl_operator(self) -> LinearOperator:
        return gmsl_from_ice_thickness_operator(
            self._fp, self._op
        )

    # -----------------------------------------------------------------------
    # New Public / Density-Aware Operators for Sampling
    # -----------------------------------------------------------------------

    # -- Ice Operators --
    @cached_property
    def ice_thickness_to_load_operator(
        self,
    ) -> LinearOperator:
        scale = self.ice_density / self._fp.ice_density
        return (
            scale * self._thickness_to_load_op
            if scale != 1.0
            else self._thickness_to_load_op
        )

    @cached_property
    def ice_thickness_to_gmsl_operator(
        self,
    ) -> LinearOperator:
        scale = self.ice_density / self._fp.ice_density
        return (
            scale * self._gmsl_operator
            if scale != 1.0
            else self._gmsl_operator
        )

    # -- Firn Operators --
    @cached_property
    def firn_thickness_to_load_operator(
        self,
    ) -> LinearOperator:
        scale = self.firn_density / self._fp.ice_density
        return (
            scale * self._thickness_to_load_op
            if scale != 1.0
            else self._thickness_to_load_op
        )

    @cached_property
    def firn_thickness_to_gmsl_operator(
        self,
    ) -> LinearOperator:
        scale = self.firn_density / self._fp.ice_density
        return (
            scale * self._gmsl_operator
            if scale != 1.0
            else self._gmsl_operator
        )

    # -- Load to SLC/SSH (Density agnostic) --
    @cached_property
    def load_to_slc_operator(self) -> LinearOperator:
        """Maps surface load to Relative Sea Level Change (Fingerprint)."""
        return self._op

    @cached_property
    def load_to_ssh_operator(self) -> LinearOperator:
        """Maps surface load to Sea Surface Height."""
        ssh_op = sea_surface_height_operator(
            self._fp, self._op.codomain
        )
        return ssh_op @ self._op


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IceSheetChange(IceThicknessGMSLOperators):
    """Generates paired Gaussian measures and samples for ice and firn
    thickness, load, SLC, and SSH changes over a specified region.
    """

    # -----------------------------------------------------------------------
    # Nested pattern classes
    # -----------------------------------------------------------------------

    class MeltPattern(ABC):
        """Abstract base class for spatial melt patterns."""

        @abstractmethod
        def spatial_weights(
            self, finger_print: FingerPrint
        ):
            """Return an SHGrid of ice spatial weights over the ice extent."""
            ...

        def firn_weights(self, finger_print: FingerPrint):
            """Return an SHGrid of firn spatial weights over the ice extent.

            Defaults to the same as spatial_weights. Override in subclasses
            where ice and firn weights are complementary.
            """
            return self.spatial_weights(finger_print)

    class UniformPattern(MeltPattern):
        """Rotationally invariant — uniform weight over the ice projection."""

        def spatial_weights(
            self, finger_print: FingerPrint
        ):
            return finger_print.ice_projection(value=0)

    class ThicknessWeightedPattern(MeltPattern):
        """Weights derived from ice thickness via a generalised logistic activator.

        High weight near ice margins (thin ice), low weight in the ice sheet
        interior (thick ice). Firn weights are the complement (1 - ice weights),
        masked to the ice extent.
        """

        def __init__(
            self,
            lower_asymptote: float = 0.1,
            upper_asymptote: float = 1.0,
            steepness: float = 10.0,
            threshold: float = 0.45,
            asymmetry: float = 0.75,
        ):
            self.lower_asymptote = lower_asymptote
            self.upper_asymptote = upper_asymptote
            self.steepness = steepness
            self.threshold = threshold
            self.asymmetry = asymmetry

        def _activator(self, x: np.ndarray) -> np.ndarray:
            """Generalised logistic function over standardised thickness."""
            a = self.lower_asymptote
            k = self.upper_asymptote
            b = self.steepness
            m = self.threshold
            nu = self.asymmetry
            return a + (k - a) / (
                1 + np.exp(b * (x - m))
            ) ** (1 / nu)

        def _standardise(
            self, data: np.ndarray
        ) -> np.ndarray:
            return (data - data.min()) / (
                data.max() - data.min()
            )

        def spatial_weights(
            self, finger_print: FingerPrint
        ):
            grid = finger_print.ice_thickness.copy()
            grid.data = self._activator(
                self._standardise(grid.data)
            )
            return grid * finger_print.ice_projection(
                value=0
            )

        def firn_weights(self, finger_print: FingerPrint):
            grid = finger_print.ice_thickness.copy()
            grid.data = 1.0 - self._activator(
                self._standardise(grid.data)
            )
            return grid * finger_print.ice_projection(
                value=0
            )

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        length_scale: float,
        region_projection: Callable,
        pattern: MeltPattern,
        include_firn: bool = False,
        ice_gmsl_std: float = 0.001,
        firn_gmsl_std: Optional[float] = None,
        gmsl_target_mean: float = 0.0,
        altimetry_latitude_range: float = 66.0,
        point_degree_spacing: float = 5.0,
        ice_density: Optional[float] = None,
        firn_density: Optional[float] = None,
    ):
        super().__init__(
            finger_print,
            finger_print_operator,
            altimetry_latitude_range=altimetry_latitude_range,
            point_degree_spacing=point_degree_spacing,
        )
        self._fp = finger_print
        self._op = finger_print_operator
        self._length_scale = length_scale
        self._region_projection = region_projection
        self._pattern = pattern
        self._include_firn = include_firn
        self._ice_gmsl_std = ice_gmsl_std
        self._firn_gmsl_std = (
            firn_gmsl_std
            if firn_gmsl_std is not None
            else self._ice_gmsl_std
        )
        self._gmsl_target_mean = gmsl_target_mean

        # Set densities, falling back to fingerprint default if not provided
        self.ice_density = (
            ice_density
            if ice_density is not None
            else self._fp.ice_density
        )
        self.firn_density = (
            firn_density
            if firn_density is not None
            else self.ice_density * 0.5
        )

    # -----------------------------------------------------------------------
    # Shared internals
    # -----------------------------------------------------------------------

    @cached_property
    def _load_space(self) -> Sobolev:
        return self._op.domain

    @cached_property
    def _region_projection_grid(self):
        return self._region_projection(value=0)

    @cached_property
    def _ice_weights(self):
        return (
            self._pattern.spatial_weights(self._fp)
            * self._region_projection_grid
        )

    @cached_property
    def _firn_weights(self):
        return (
            self._pattern.firn_weights(self._fp)
            * self._region_projection_grid
        )

    def _build_measure(
        self,
        weights,
        gmsl_std: float,
        gmsl_mean: float,
        density: float,
        gmsl_operator: LinearOperator,
    ) -> GaussianMeasure:
        _base = (
            self._load_space.heat_kernel_gaussian_measure(
                self._length_scale
            )
        )
        _weight_op = spatial_mutliplication_operator(
            weights, self._load_space
        )
        _ice_proj_op = ice_projection_operator(
            self._fp, self._load_space
        )

        _gmsl_std_current = standard_dev(
            _base.affine_mapping(operator=gmsl_operator)
        )
        _std_scale = gmsl_std / _gmsl_std_current

        if gmsl_mean != 0.0:
            _gmsl_per_unit = self._fp.integrate(
                -density
                * self._fp.one_minus_ocean_function
                * weights
                * self._fp.length_scale
                / (
                    self._fp.water_density
                    * self._fp.ocean_area
                )
            )
            _shift = (
                gmsl_mean / _gmsl_per_unit
                if _gmsl_per_unit != 0
                else 0.0
            )
            _shift_vector = (
                self._load_space.project_function(
                    lambda _: _shift
                )
            )
        else:
            _shift_vector = None

        return _base.affine_mapping(
            operator=_std_scale * _weight_op,
            translation=_shift_vector,
        ).affine_mapping(operator=_ice_proj_op)

    # -----------------------------------------------------------------------
    # Thickness Measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_thickness(self) -> GaussianMeasure:
        return self._build_measure(
            self._ice_weights,
            self._ice_gmsl_std,
            self._gmsl_target_mean,
            density=self.ice_density,
            gmsl_operator=self.ice_thickness_to_gmsl_operator,
        )

    @cached_property
    def firn_thickness(self) -> Optional[GaussianMeasure]:
        if not self._include_firn:
            return None
        return self._build_measure(
            self._firn_weights,
            self._firn_gmsl_std,
            gmsl_mean=0.0,
            density=self.firn_density,
            gmsl_operator=self.firn_thickness_to_gmsl_operator,
        )

    @cached_property
    def total_thickness(self) -> GaussianMeasure:
        if self._include_firn:
            return self.ice_thickness + self.firn_thickness
        return self.ice_thickness

    # -----------------------------------------------------------------------
    # Load Measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_load(self) -> GaussianMeasure:
        return self.ice_thickness.affine_mapping(
            operator=self.ice_thickness_to_load_operator
        )

    @cached_property
    def firn_load(self) -> Optional[GaussianMeasure]:
        if not self._include_firn:
            return None
        return self.firn_thickness.affine_mapping(
            operator=self.firn_thickness_to_load_operator
        )

    @cached_property
    def total_load(self) -> GaussianMeasure:
        if self._include_firn:
            return self.ice_load + self.firn_load
        return self.ice_load

    # -----------------------------------------------------------------------
    # SLC (Fingerprint) Measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_slc(self) -> GaussianMeasure:
        _projection = self.load_to_slc_operator.codomain.subspace_projection(
            0
        )
        return self.ice_load.affine_mapping(
            operator=_projection @ self.load_to_slc_operator
        )

    @cached_property
    def firn_slc(self) -> Optional[GaussianMeasure]:
        _projection = self.load_to_slc_operator.codomain.subspace_projection(
            0
        )
        if not self._include_firn:
            return None
        return self.firn_load.affine_mapping(
            operator=_projection @ self.load_to_slc_operator
        )

    @cached_property
    def total_slc(self) -> GaussianMeasure:
        if self._include_firn:
            return self.ice_slc + self.firn_slc
        return self.ice_slc

    # -----------------------------------------------------------------------
    # SSH Measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_ssh(self) -> GaussianMeasure:
        return self.ice_load.affine_mapping(
            operator=self.load_to_ssh_operator
        )

    @cached_property
    def firn_ssh(self) -> Optional[GaussianMeasure]:
        if not self._include_firn:
            return None
        return self.firn_load.affine_mapping(
            operator=self.load_to_ssh_operator
        )

    @cached_property
    def total_ssh(self) -> GaussianMeasure:
        if self._include_firn:
            return self.ice_ssh + self.firn_ssh
        return self.ice_ssh

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(self) -> IceSheetChangeSample:
        """Draw an independent sample from the ice and firn priors and
        project them through load, SLC, and SSH operators.
        """
        ice_h = self.ice_thickness.sample()
        ice_l = self.ice_thickness_to_load_operator(ice_h)
        slc_proj = self.load_to_slc_operator.codomain.subspace_projection(
            0
        )
        slc_proj = (
            ocean_projection_operator(
                self._fp, slc_proj.codomain
            )
            @ slc_proj
        )
        ice_slc = (slc_proj @ self.load_to_slc_operator)(
            ice_l
        )
        ice_ssh = self.load_to_ssh_operator(ice_l)

        if self._include_firn:
            firn_h = self.firn_thickness.sample()
            firn_l = self.firn_thickness_to_load_operator(
                firn_h
            )
            firn_slc = (
                slc_proj @ self.load_to_slc_operator
            )(firn_l)
            firn_ssh = self.load_to_ssh_operator(firn_l)

            total_h = ice_h + firn_h
            total_l = ice_l + firn_l
            total_slc = ice_slc + firn_slc
            total_ssh = ice_ssh + firn_ssh
        else:
            firn_h = firn_l = firn_slc = firn_ssh = None
            total_h = ice_h
            total_l = ice_l
            total_slc = ice_slc
            total_ssh = ice_ssh

        return IceSheetChangeSample(
            ice_thickness=ice_h,
            firn_thickness=firn_h,
            total_thickness=total_h,
            ice_load=ice_l,
            firn_load=firn_l,
            total_load=total_l,
            ice_slc=ice_slc,
            firn_slc=firn_slc,
            total_slc=total_slc,
            ice_ssh=ice_ssh,
            firn_ssh=firn_ssh,
            total_ssh=total_ssh,
        )

    # -----------------------------------------------------------------------
    # Named region constructors
    # -----------------------------------------------------------------------

    @classmethod
    def greenland(
        cls,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        length_scale: float,
        pattern: MeltPattern,
        **kwargs,
    ) -> IceSheetChange:
        return cls(
            finger_print,
            finger_print_operator,
            length_scale,
            finger_print.greenland_projection,
            pattern,
            **kwargs,
        )

    @classmethod
    def west_antarctic(
        cls,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        length_scale: float,
        pattern: MeltPattern,
        **kwargs,
    ) -> IceSheetChange:
        return cls(
            finger_print,
            finger_print_operator,
            length_scale,
            finger_print.west_antarctic_projection,
            pattern,
            **kwargs,
        )

    @classmethod
    def east_antarctic(
        cls,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        length_scale: float,
        pattern: MeltPattern,
        **kwargs,
    ) -> IceSheetChange:
        return cls(
            finger_print,
            finger_print_operator,
            length_scale,
            finger_print.east_antarctic_projection,
            pattern,
            **kwargs,
        )

    @classmethod
    def global_ice(
        cls,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        length_scale: float,
        pattern: MeltPattern,
        **kwargs,
    ) -> IceSheetChange:
        return cls(
            finger_print,
            finger_print_operator,
            length_scale,
            finger_print.ice_projection,
            pattern,
            **kwargs,
        )
