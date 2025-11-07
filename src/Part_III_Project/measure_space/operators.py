from pygeoinf import (
    GaussianMeasure,
    LinearOperator,
)
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
)


def gmsl_measure_condenser(
    measure: GaussianMeasure,
    fingerprint: FingerPrint,
) -> LinearOperator:
    weighting_function = (
        -fingerprint.ice_density
        * fingerprint.one_minus_ocean_function
        * fingerprint.ice_projection(value=0)
        * fingerprint.length_scale
        / (fingerprint.water_density * fingerprint.ocean_area)
    )
    gmsl_operator = averaging_operator(
        measure.domain,
        [weighting_function],
    )
    return gmsl_operator


"""
def remove_ice_average_operator(
    finger_print: FingerPrint,
    fingerprint_operator: LinearOperator,
):
    l2_load_space = underlying_space(fingerprint_operator.domain)

    ice_function = finger_print.ice_function
    ice_area = finger_print.ice_area

    def mapping(load):
        ocean_average = (
            finger_print.integrate(ocean_function * load) / ocean_area
        )
        new_load = load.copy()
        new_load.data -= ocean_average
        return new_load

    def adjoint_mapping(load):
        average = finger_print.integrate(load)
        return load - average * ocean_function / ocean_area

    l2_operator = LinearOperator(
        l2_load_space,
        l2_load_space,
        mapping,
        adjoint_mapping=adjoint_mapping,
    )

    return LinearOperator.from_formal_adjoint(
        load_space,
        load_space,
        l2_operator,
    )
"""
