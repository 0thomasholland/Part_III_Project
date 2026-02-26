# %%
import colorcet as cc
from matplotlib import pyplot as plt
from pyshtools import SHGrid
from pyslfp import FingerPrint, IceModel, plot

from pygeoinf_extras import expectation, standard_dev
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)
# %%

ice_change_spatial = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.01 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.01,
    firn_gmsl_std=0.01,
    include_firn=True,
    ice_density=fp.ice_density,
    firn_density=fp.ice_density * 0.5,
)
print("done generation")
samples = ice_change_spatial.sample()
print("done sampling")

print(type(samples.ice_slc))
print(type(samples.firn_slc))
# %%
plot(
    samples.firn_thickness,
    symmetric=True,
    colorbar_label="Firn thickness change (m)",
)
plot(
    samples.firn_load,
    symmetric=True,
    colorbar_label="Firn load change (kg/m²)",
)
plot(
    samples.ice_thickness,
    symmetric=True,
    colorbar_label="Ice thickness change (m)",
)
plot(
    samples.ice_load,
    symmetric=True,
    colorbar_label="Ice load change (kg/m²)",
)
plot(
    samples.total_thickness,
    symmetric=True,
    colorbar_label="Total thickness change (m)",
)
plot(
    samples.total_load,
    symmetric=True,
    colorbar_label="Total load change (kg/m²)",
)


# %%
plot(
    samples.firn_slc,
    symmetric=True,
    colorbar_label="Firn SLC (m)",
)
plot(
    samples.ice_slc,
    symmetric=True,
    colorbar_label="Ice SLC (m)",
)
plot(
    samples.total_slc,
    symmetric=True,
    colorbar_label="Total SLC (m)",
)

plot(
    samples.firn_ssh,
    symmetric=True,
    colorbar_label="Firn SSH change (m)",
)
plot(
    samples.ice_ssh,
    symmetric=True,
    colorbar_label="Ice SSH change (m)",
)
plot(
    samples.total_ssh,
    symmetric=True,
    colorbar_label="Total SSH change (m)",
)

# %%
# use L12 cmap
plot(
    ice_change_spatial.ice_load.sample_pointwise_variance(
        30
    )
    * fp.ice_projection(),
    cmap=cc.cm.bmy,
)

plot(
    ice_change_spatial.firn_load.sample_pointwise_variance(
        30
    )
    * fp.ice_projection(),
    cmap=cc.cm.bmy,
)

plot(
    ice_change_spatial.total_slc.sample_pointwise_variance(
        90
    )
    * fp.ocean_projection(),
    cmap=cc.cm.bmy,
)
