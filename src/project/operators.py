from pygeoinf import LinearOperator
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)


def _ice_thickness_to_finger_print_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
) -> LinearOperator:
    return (
        finger_print_operator
        @ ice_thickness_change_to_load_operator(
            finger_print, finger_print_operator.domain
        )
    )


def ice_thickness_to_slc_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
) -> LinearOperator:
    return (
        ocean_projection_operator(
            finger_print, finger_print_operator.domain
        )
        @ finger_print_operator.codomain.subspace_projection(
            0
        )
        @ _ice_thickness_to_finger_print_operator(
            finger_print, finger_print_operator
        )
    )


def ice_thickness_to_ssh_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
) -> LinearOperator:
    """
    Create an operator that maps ice thickness changes to sea surface height changes
    based on altimetry coverage.
    """
    _sea_surface_height_op = sea_surface_height_operator(
        finger_print, finger_print_operator.codomain
    )
    return (
        spatial_mutliplication_operator(
            finger_print.altimetry_projection(
                latitude_min=-altimetry_latitude_range,
                latitude_max=altimetry_latitude_range,
                value=0,
            ),
            _sea_surface_height_op.codomain,
        )
        @ _sea_surface_height_op
        @ _ice_thickness_to_finger_print_operator(
            finger_print, finger_print_operator
        )
    )


def ice_thickness_to_estimated_gmsl_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
) -> LinearOperator:
    """
    Create an operator that maps ice thickness changes to estimated GMSL changes
    based on altimetry coverage.
    """
    _ssh_operator = ice_thickness_to_ssh_operator(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range,
    )
    _altimetry_projection = (
        finger_print.altimetry_projection(
            latitude_min=-altimetry_latitude_range,
            latitude_max=altimetry_latitude_range,
            value=0,
        )
    )
    return (
        averaging_operator(
            _ssh_operator.codomain,
            [
                _altimetry_projection
                / finger_print.integrate(
                    _altimetry_projection
                )
            ],
        )
        @ _ssh_operator
    )


def ice_thickness_to_gmsl_estimation_error_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
) -> LinearOperator:
    _gmsl = gmsl_from_ice_thickness_operator(
        finger_print_operator.domain, finger_print
    )
    _estimated_gmsl = (
        ice_thickness_to_estimated_gmsl_operator(
            finger_print,
            finger_print_operator,
            altimetry_latitude_range,
        )
    )
    return _gmsl - _estimated_gmsl
