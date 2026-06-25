# %%

from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
import colorcet as cc
import numpy as np

from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)
from pyslfp_extras.plotting import plot

fp = EarthState.from_defaults(lmax=256)
fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))
# %%

ice_change_spatial = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.01 * fp.model.parameters.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.02,
    firn_gmsl_std=0.01,
    include_firn=True,
    ice_density=fp.model.parameters.ice_density,
    firn_density=fp.model.parameters.ice_density * 0.4,
)
print("done generation")
samples = ice_change_spatial.sample()
print("done sampling")

print(type(samples.ice_slc))
print(type(samples.firn_slc))
# %%
thickness_max = np.max(
    [
        np.abs(samples.firn_thickness).max(),
        np.abs(samples.ice_thickness).max(),
        np.abs(samples.total_thickness).max(),
    ]
)
load_max = np.max(
    [
        np.abs(samples.firn_load).max(),
        np.abs(samples.ice_load).max(),
        np.abs(samples.total_load).max(),
    ]
)
plot(
    samples.firn_thickness,
    vmax=thickness_max,
    vmin=-thickness_max,
    colorbar_label="Firn thickness change (m)",
    tight_layout=True,
)[0].savefig("figs/firn_thickness.pdf", dpi=600)
plot(
    samples.firn_load,
    vmax=load_max,
    vmin=-load_max,
    tight_layout=True,
    colorbar_label="Firn load change (kg/m²)",
)[0].savefig("figs/firn_load.pdf", dpi=600)
plot(
    samples.ice_thickness,
    vmax=thickness_max,
    vmin=-thickness_max,
    tight_layout=True,
    colorbar_label="Ice thickness change (m)",
)[0].savefig("figs/ice_thickness.pdf", dpi=600)
plot(
    samples.ice_load,
    vmax=load_max,
    vmin=-load_max,
    tight_layout=True,
    colorbar_label="Ice load change (kg/m²)",
)[0].savefig("figs/ice_load.pdf", dpi=600)
plot(
    samples.total_thickness,
    vmax=thickness_max,
    vmin=-thickness_max,
    tight_layout=True,
    colorbar_label="Total thickness change (m)",
)[0].savefig("figs/total_thickness.pdf", dpi=600)
plot(
    samples.total_load,
    vmax=load_max,
    vmin=-load_max,
    tight_layout=True,
    colorbar_label="Total load change (kg/m²)",
)[0].savefig("figs/total_load.pdf", dpi=600)

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
        100
    )
    * fp.ice_projection(),
    cmap=cc.cm.bmy,
    colorbar_label="Ice load variance from 100 samples (kg²/m⁴)",
)[0].savefig("figs/ice_load_std.pdf", dpi=600)

plot(
    ice_change_spatial.firn_load.sample_pointwise_variance(
        100
    )
    * fp.ice_projection(),
    cmap=cc.cm.bmy,
    colorbar_label="Firn load variance from 100 samples (kg²/m⁴)",
)[0].savefig("figs/firn_load_variance.pdf", dpi=600)
