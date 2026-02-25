# %%
import colorcet as cc
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
ice_change_uniform = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.01 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.01,
)
ice_thickness_measure = ice_change_uniform.ice_thickness_measure

ice_change_spatial = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.01 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.01,
)
ice_thickness_measure_spatial = (
    ice_change_spatial.ice_thickness_measure
)

# %%
(
    fig1,
    ax1,
    im1,
) = plot(
    ice_thickness_measure.sample() * fp.ice_projection(),
    symmetric=True,
    colorbar_label="Sample (uniform field) (m)",
)
(
    fig2,
    ax2,
    im2,
) = plot(
    ice_thickness_measure_spatial.sample()
    * fp.ice_projection(),
    symmetric=True,
    colorbar_label="Sample (spatial field) (m)",
)

# %%
fig3, ax3, im3 = plot(
    ice_thickness_measure.expectation * fp.ice_projection(),
    symmetric=True,
    colorbar_label="Expectation (uniform field) (m)",
)
fig4, ax4, im4 = plot(
    ice_thickness_measure_spatial.expectation
    * fp.ice_projection(),
    symmetric=True,
    colorbar_label="Expectation (spatial field) (m)",
)

# %%
fig5, ax5, im5 = plot(
    ice_thickness_measure.sample_pointwise_std(
        1000, parallel=True
    )
    * fp.ice_projection(),
    cmap=cc.cm.blues,
    colorbar_label="Pointwise standard deviation (uniform field) (m)",
)
fig6, ax6, im6 = plot(
    ice_thickness_measure_spatial.sample_pointwise_std(
        1000, parallel=True
    )
    * fp.ice_projection(),
    cmap=cc.cm.blues,
    colorbar_label="Pointwise standard deviation (spatial field) (m)",
)

# %%

gmsl_from_ice_thickness_op = (
    gmsl_from_ice_thickness_operator(
        finger_print=fp, finger_print_operator=fp_op
    )
)

gmsl_uniform = ice_thickness_measure.affine_mapping(
    operator=gmsl_from_ice_thickness_op
)

gmsl_spatial = ice_thickness_measure_spatial.affine_mapping(
    operator=gmsl_from_ice_thickness_op
)

print("Desired GMSL expetation = 0.08 m, std = 0.01 m")
print(
    f"GMSL (uniform field): expectation={expectation(gmsl_uniform):.4f}, std={standard_dev(gmsl_uniform):.4f}"
)
print(
    f"GMSL (spatial field): expectation={expectation(gmsl_spatial):.4f}, std={standard_dev(gmsl_spatial):.4f}"
)

# %%
fig1.savefig("sample_uniform.png", dpi=600)
fig2.savefig("sample_spatial.png", dpi=600)
fig3.savefig("expectation_uniform.png", dpi=600)
fig4.savefig("expectation_spatial.png", dpi=600)
fig5.savefig("std_uniform.png", dpi=600)
fig6.savefig("std_spatial.png", dpi=600)
