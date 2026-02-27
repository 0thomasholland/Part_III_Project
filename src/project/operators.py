from pygeoinf import LinearOperator
from pyslfp import (
    FingerPrint,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
)

from pyslfp_extras.ice_thickness import (
    IceThicknessGMSLOperators,
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
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
    ).load_to_altimetry_ssh_operator


def ice_thickness_to_estimated_gmsl_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
) -> LinearOperator:
    """
    Create an operator that maps ice thickness changes to estimated GMSL changes
    based on altimetry coverage.
    """
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
    ).load_to_estimated_gmsl_operator


def ice_thickness_to_ssh_point_estimations_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
    point_degree_spacing: float = 5.0,
) -> LinearOperator:
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
        point_degree_spacing=point_degree_spacing,
    ).load_to_ssh_point_estimations_operator


def ice_thickness_to_point_estimated_gmsl_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
    point_degree_spacing: float = 5.0,
) -> LinearOperator:
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
        point_degree_spacing=point_degree_spacing,
    ).load_to_point_estimated_gmsl_operator


def ice_thickness_to_gmsl_estimation_error_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
) -> LinearOperator:
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
    ).gmsl_estimation_error_operator


def ice_thickness_to_gmsl_point_estimation_error_operator(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    altimetry_latitude_range: float = 66.0,
    point_degree_spacing: float = 5.0,
) -> LinearOperator:
    return IceThicknessGMSLOperators(
        finger_print,
        finger_print_operator,
        altimetry_latitude_range=altimetry_latitude_range,
        point_degree_spacing=point_degree_spacing,
    ).gmsl_point_estimation_error_operator
