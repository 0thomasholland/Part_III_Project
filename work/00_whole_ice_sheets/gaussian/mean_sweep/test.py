# %%
from pyslfp import FingerPrint, IceModel, plot

from pyslfp_extras.ice_thickness import IceSheetChange

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)


ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.02,
    gmsl_target_mean=0.02,
)
# %%

plot(ice_change.ice_load.expectation)

# %%

sl, d, _, v = fp(
    direct_load=ice_change.ice_load.expectation
)
sl *= fp.ocean_projection(value=0)
print(fp.integrate(sl) / fp.ocean_area)
ssh = fp.sea_surface_height_change(sl, d, v)
ssh *= fp.altimetry_projection(value=0)
print(
    fp.integrate(ssh)
    / fp.integrate(fp.altimetry_projection(value=0))
)

fp.integrate(fp.altimetry_projection(value=0))


print(
    ice_change.load_to_estimated_gmsl_operator(
        ice_change.ice_load.expectation
    )[0]
)
