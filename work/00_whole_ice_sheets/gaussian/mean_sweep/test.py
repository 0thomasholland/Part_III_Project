# %%

from pyslfp import LinearSeaLevelEquation
from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.linear_operators.physics import (
    centrifugal_potential_operator,
)
from pyslfp.state import EarthState
from pyslfp_extras.ice_thickness import IceSheetChange

fp = EarthState.from_defaults(lmax=256)
fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.model.parameters.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.02,
    gmsl_target_mean=0.02,
)
# %%

plot(ice_change.ice_load.expectation)

# %%

sl, d, _, v = LinearSeaLevelEquation(fp).solve_sea_level_equation(ice_change.ice_load.expectation
)
sl *= fp.ocean_projection(value=0)
print(fp.model.integrate(sl) / fp.ocean_area)
ssh = (sl + d + centrifugal_potential_operator(fp.model)(v) / fp.model.parameters.gravitational_acceleration)
ssh *= fp.altimetry_projection(value=0)
print(
    fp.model.integrate(ssh)
    / fp.model.integrate(fp.altimetry_projection(value=0))
)

fp.model.integrate(fp.altimetry_projection(value=0))

print(
    ice_change.load_to_estimated_gmsl_operator(
        ice_change.ice_load.expectation
    )[0]
)
