from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional

import numpy as np
from pygeoinf import GaussianMeasure, LinearOperator
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    spatial_mutliplication_operator,
)

from pygeoinf_extras import standard_dev
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)

# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------


@dataclass
class IceSheetChangeSample:
    ice_thickness: object  # SHGrid
    firn_thickness: object  # SHGrid | None
    total_thickness: object  # SHGrid
    ice_load: object  # SHGrid
    firn_load: object  # SHGrid | None
    total_load: object  # SHGrid


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class IceSheetChange:
    """Generates paired Gaussian measures and samples for ice and firn
    thickness/load changes over a specified region.

    Parameters
    ----------
    finger_print:
        The FingerPrint instance.
    finger_print_operator:
        Linear operator mapping ice load to the observation space.
    length_scale:
        Length scale for the heat kernel Gaussian measure.
    region_projection:
        Callable returning an SHGrid projection mask for the region of
        interest (e.g. ``finger_print.greenland_projection``).
    pattern:
        An IceSheetChange.MeltPattern instance controlling spatial weighting.
    include_firn:
        Whether to model firn thickness/load changes separately.
    ice_gmsl_std:
        Prior standard deviation for ice-driven GMSL change (metres).
    firn_gmsl_std:
        Prior standard deviation for firn-driven GMSL change (metres).
        Defaults to 20% of ice_gmsl_std if None.
    gmsl_target_mean:
        Prior mean for GMSL change. Defaults to 0.

    Examples
    --------
    >>> spatial_pattern = IceSheetChange.ThicknessWeightedPattern(threshold=0.5)
    >>> greenland_change = IceSheetChange.greenland(fp, op, ls, pattern=spatial_pattern,
    ...                                include_firn=True, ice_gmsl_std=0.001)
    >>> sample = greenland_change.sample()
    >>> pyslfp.plot(sample.total_thickness)
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

        Parameters
        ----------
        lower_asymptote:
            Minimum melt probability (thick ice limit). Default 0.1.
        upper_asymptote:
            Maximum melt probability (thin ice limit). Default 1.0.
        steepness:
            Controls how rapidly the probability drops with thickness. Default 10.0.
        threshold:
            Standardised thickness (0–1) at which the drop-off occurs. Default 0.45.
        asymmetry:
            Controls the sharpness of the curve's turn. Default 0.75.
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
    ):
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
            else 0.2 * ice_gmsl_std
        )
        self._gmsl_target_mean = gmsl_target_mean

    # -----------------------------------------------------------------------
    # Shared internals
    # -----------------------------------------------------------------------

    @cached_property
    def _load_space(self) -> Sobolev:
        return self._op.domain

    @cached_property
    def _gmsl_op(self) -> LinearOperator:
        return gmsl_from_ice_thickness_operator(
            self._fp, self._op
        )

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

    @cached_property
    def _thickness_to_load_op(self) -> LinearOperator:
        """Linear operator converting thickness change to mass load change."""
        return ice_thickness_change_to_load_operator(
            self._fp, self._load_space
        )

    def _build_measure(
        self,
        weights,
        gmsl_std: float,
        gmsl_mean: float,
    ) -> GaussianMeasure:
        """Build a normalised, optionally shifted GaussianMeasure for a given
        weight grid."""
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

        # Scale to target GMSL std
        _gmsl_std_current = standard_dev(
            _base.affine_mapping(operator=self._gmsl_op)
        )
        _std_scale = gmsl_std / _gmsl_std_current

        # Shift to target GMSL mean
        if gmsl_mean != 0.0:
            _gmsl_per_unit = self._fp.integrate(
                -self._fp.ice_density
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
    # Thickness measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_thickness_measure(self) -> GaussianMeasure:
        return self._build_measure(
            self._ice_weights,
            self._ice_gmsl_std,
            self._gmsl_target_mean,
        )

    @cached_property
    def firn_thickness_measure(
        self,
    ) -> Optional[GaussianMeasure]:
        if not self._include_firn:
            return None
        return self._build_measure(
            self._firn_weights,
            self._firn_gmsl_std,
            gmsl_mean=0.0,
        )

    @cached_property
    def total_thickness_measure(self) -> GaussianMeasure:
        if self._include_firn:
            return (
                self.ice_thickness_measure
                + self.firn_thickness_measure
            )
        return self.ice_thickness_measure

    # -----------------------------------------------------------------------
    # Load measures
    # -----------------------------------------------------------------------

    @cached_property
    def ice_load_measure(self) -> GaussianMeasure:
        return self.ice_thickness_measure.affine_mapping(
            operator=self._thickness_to_load_op
        )

    @cached_property
    def firn_load_measure(
        self,
    ) -> Optional[GaussianMeasure]:
        if not self._include_firn:
            return None
        return self.firn_thickness_measure.affine_mapping(
            operator=self._thickness_to_load_op
        )

    @cached_property
    def total_load_measure(self) -> GaussianMeasure:
        if self._include_firn:
            return (
                self.ice_load_measure
                + self.firn_load_measure
            )
        return self.ice_load_measure

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample(self) -> IceSheetChangeSample:
        """Draw an independent sample from the ice and firn priors.

        Ice and firn are sampled independently, consistent with the independent
        prior. The inversion posterior will introduce correlation between them
        via the likelihood.
        """
        ice_h = self.ice_thickness_measure.sample()
        ice_l = self._thickness_to_load_op(ice_h)

        if self._include_firn:
            firn_h = self.firn_thickness_measure.sample()
            firn_l = self._thickness_to_load_op(firn_h)
            total_h = ice_h + firn_h
            total_l = ice_l + firn_l
        else:
            firn_h = firn_l = None
            total_h = ice_h
            total_l = ice_l

        return IceSheetChangeSample(
            ice_thickness=ice_h,
            firn_thickness=firn_h,
            total_thickness=total_h,
            ice_load=ice_l,
            firn_load=firn_l,
            total_load=total_l,
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
