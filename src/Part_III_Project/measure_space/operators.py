from pygeoinf import GaussianMeasure, LinearOperator, RowLinearOperator
from pyslfp import FingerPrint, sea_level_change_to_load_operator, ice_thickness_change_to_load_operator


def fp_operator(
    fingerprint: FingerPrint, scale: float
) -> tuple[LinearOperator, LinearOperator.domain, LinearOperator.codomain]:
    """
    -> operator(load->SLC), load_space, response_space

    Create a linear operator that takes a direct load and returns the sea level fingerprint. Also returns the load space and response space.
    """
    finger_print_operator = fingerprint.as_sobolev_linear_operator(order=2, scale=scale)
    return (
        finger_print_operator,
        finger_print_operator.domain,
        finger_print_operator.codomain,
    )


def slc_to_ssh_operator():
    pass


def height_to_direct_load_operator(
    fingerprint: FingerPrint, load_space: LinearOperator.domain
) -> LinearOperator:
    """
    -> RowLinearOperator [ice_thickness_change, sea_level_change] -> load
    """
    _ice_thickness_to_load_operator = ice_thickness_change_to_load_operator(finger_print= fingerprint, load_space=load_space, load_space=load_space)
    _sea_level_change_to_load_operator = sea_level_change_to_load_operator(finger_print= fingerprint, load_space=load_space, load_space=load_space)
    direct_load = RowLinearOperator(
        [_ice_thickness_to_load_operator, _sea_level_change_to_load_operator]
    )
    # collapses the two inputs into a single load output 


def ocean_projection_operator():
    pass


def gmsl_operator(
    space: Any, finger_print: GaussianMeasure
) -> tuple[GaussianMeasure, float, float]:
    pass

def slc_response_operator(fp_op: LinearOperator, direct_load_op: LinearOperator) -> LinearOperator:
    pass


def ssh_response_operator(fp_op: LinearOperator, direct_load_op: LinearOperator):
    pass

def ssh_response_with_odt(fp_op: LinearOperator, direct_load_op: LinearOperator, load_space: LinearOperator.domain) -> LinearOperator:
    pass

def gmsl_measure_condenser(measure: GaussianMeasure) -> GaussianMeasure:
    pass
