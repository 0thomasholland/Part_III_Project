from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Literal, Optional

from numpy import ndarray
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import (
    Lebesgue,
    Sobolev,
)
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from pygeoinf_extras.operators import (
    point_averaging_operator,
)
from pyslfp_extras.altimetry import (
    ocean_point_evaluation_operator,
)


class GMSLOperatorBase(ABC):
    """Shared operators for mapping load -> SSH -> GMSL and GMSL estimations.

    Subclasses provide:
      - _load_to_ssh_operator: load -> SSH (including fingerprint effects)
      - _gmsl_operator: load -> true GMSL
    """

    def __init__(
        self,
        finger_print: FingerPrint,
        finger_print_operator: LinearOperator,
        altimetry_latitude_range: float = 66.0,
        point_degree_spacing: float = 5.0,
        parallel_workers: None | int = None,
    ) -> None:
        self._fp = finger_print
        self._op = finger_print_operator
        self._altimetry_latitude_range = (
            altimetry_latitude_range
        )
        self._point_degree_spacing = point_degree_spacing
        self._parallel_workers = parallel_workers

    @property
    def finger_print(self) -> FingerPrint:
        return self._fp

    @property
    def finger_print_operator(self) -> LinearOperator:
        return self._op

    @property
    def altimetry_latitude_range(self) -> float:
        return self._altimetry_latitude_range

    @property
    def point_degree_spacing(self) -> float:
        return self._point_degree_spacing

    @property
    def parallel_workers(self) -> None | int:
        return self._parallel_workers

    @cached_property
    def _altimetry_projection(self) -> SHGrid:
        return self._fp.altimetry_projection(
            latitude_min=-self._altimetry_latitude_range,
            latitude_max=self._altimetry_latitude_range,
            value=0,
        )

    @cached_property
    def _altimetry_weighting_grid(self) -> SHGrid:
        projection_integral = self._fp.integrate(
            self._altimetry_projection
        )
        return (
            self._altimetry_projection / projection_integral
        )

    @property
    @abstractmethod
    def _load_to_ssh_operator(self) -> LinearOperator:
        """Maps load space -> SSH space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _gmsl_operator(self) -> LinearOperator:
        """Maps load space -> true GMSL."""
        raise NotImplementedError

    @cached_property
    def _ssh_space(self) -> HilbertSpace:
        return self._load_to_ssh_operator.codomain

    @cached_property
    def _altimetry_mask_operator(self) -> LinearOperator:
        return spatial_mutliplication_operator(
            self._altimetry_projection, self._ssh_space
        )

    @cached_property
    def load_to_ssh_operator(self) -> LinearOperator:
        """Load -> SSH (full ocean)."""
        return self._load_to_ssh_operator

    @cached_property
    def load_to_altimetry_ssh_operator(
        self,
    ) -> LinearOperator:
        """Load -> SSH restricted to altimetry coverage."""
        return (
            self._altimetry_mask_operator
            @ self._load_to_ssh_operator
        )

    @cached_property
    def load_to_estimated_gmsl_operator(
        self,
    ) -> LinearOperator:
        """Load -> estimated GMSL by surface-averaged SSH (altimetry mask)."""
        return (
            averaging_operator(
                self._ssh_space,
                [self._altimetry_weighting_grid],
            )
            @ self.load_to_altimetry_ssh_operator
        )

    @cached_property
    def load_to_ssh_point_estimations_operator(
        self,
    ) -> LinearOperator:
        """Load -> SSH evaluated at altimetry sampling points."""
        point_op = ocean_point_evaluation_operator(
            self._fp,
            self._ssh_space,
            point_degree_spacing=self._point_degree_spacing,
            altimetry_latitude_range=self._altimetry_latitude_range,
            parallel_workers=self._parallel_workers,
        )
        return (
            point_op @ self.load_to_altimetry_ssh_operator
        )

    @cached_property
    def load_to_point_estimated_gmsl_operator(
        self,
    ) -> LinearOperator:
        """Load -> estimated GMSL via point altimetry averaging."""
        point_avg_op = point_averaging_operator(
            self.load_to_ssh_point_estimations_operator.codomain
        )
        return (
            point_avg_op
            @ self.load_to_ssh_point_estimations_operator
        )

    @cached_property
    def gmsl_estimation_error_operator(
        self,
    ) -> LinearOperator:
        """True GMSL minus surface-averaged (altimetry) estimate."""
        return (
            self._gmsl_operator
            - self.load_to_estimated_gmsl_operator
        )

    @cached_property
    def gmsl_point_estimation_error_operator(
        self,
    ) -> LinearOperator:
        """True GMSL minus point-altimetry estimate."""
        return (
            self._gmsl_operator
            - self.load_to_point_estimated_gmsl_operator
        )


def gmsl_from_ice_thickness_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
) -> LinearOperator:
    _op: LinearOperator = averaging_operator(
        finger_print_operator.domain,
        [
            -finger_print.ice_density
            * finger_print.one_minus_ocean_function
            * finger_print.ice_projection(value=0)
            * finger_print.length_scale
            / (
                finger_print.water_density
                * finger_print.ocean_area
            ),
        ],
    )
    return _op


def gmsl_from_ice_load_operator(
    load_space: Lebesgue | Sobolev | HilbertSpace,
    fp: FingerPrint,
) -> LinearOperator:
    """Convenience wrapper when the load space equals the thickness space."""
    identity_op = load_space.identity_operator()
    return gmsl_from_ice_thickness_operator(fp, identity_op)


def altimetry_gmsl(
    ssh: SHGrid,
    fp: FingerPrint,
    latitude: float = 66.0,
) -> float:
    _alt_projection = fp.altimetry_projection(
        latitude_max=latitude,
        latitude_min=-latitude,
        value=0,
    )
    _alt_projection_integral = fp.integrate(_alt_projection)
    _alt_weighting_func = (
        _alt_projection / _alt_projection_integral
    )
    _estimated_gmsl: float = fp.integrate(
        _alt_weighting_func * ssh
    )
    return _estimated_gmsl


def gmsl_error(
    true_gmsl: float | ndarray | GaussianMeasure,
    estimated_gmsl: float | ndarray | GaussianMeasure,
    error_type: Literal["numeric", "relative"] = "relative",
) -> float | ndarray | GaussianMeasure:
    if isinstance(true_gmsl, float) and isinstance(
        estimated_gmsl, float
    ):
        error: float = estimated_gmsl - true_gmsl
        return (
            error / true_gmsl
            if error_type == "relative"
            else error
        )

    if isinstance(true_gmsl, ndarray) and isinstance(
        estimated_gmsl, ndarray
    ):
        error: ndarray = estimated_gmsl - true_gmsl
        return (
            error / true_gmsl
            if error_type == "relative"
            else error
        )

    if isinstance(
        true_gmsl, GaussianMeasure
    ) and isinstance(estimated_gmsl, GaussianMeasure):
        error_measure: GaussianMeasure = (
            estimated_gmsl - true_gmsl
        )
        if error_type == "relative":
            raise NotImplementedError(
                "Relative error not implemented for GaussianMeasure."
            )
        return error_measure

    raise TypeError(
        f"Incompatible types: true_gmsl ({type(true_gmsl).__name__}) and "
        f"estimated_gmsl ({type(estimated_gmsl).__name__}) must be the same type."
    )
