# %%
import numpy as np
from matplotlib import pyplot as plt
from pygeoinf import GaussianMeasure, LinearOperator
from pyslfp import (
    FingerPrint,
    IceModel,
    plot,
    sea_surface_height_operator,
)

from pyslfp_extras.measures import (
    odt_fingerprint_ssh_measure,
    odt_gaussian_measure,
    odt_total_ssh_measure,
    odt_variability_field,
)

# %%

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%

odt_uniform: GaussianMeasure = odt_gaussian_measure(
    finger_print=fp,
    finger_print_operator=fp_op,
)

odt_uniform_sample = odt_uniform.sample()

# fig, ax, im = plot(
#     odt_uniform_sample
#     * fp.ocean_projection()
#     * 1000
#     * fp.length_scale,
#     coasts=True,
#     cmap="seismic",
#     symmetric=True,
#     colorbar_label="ODT (mm)",
# )
# ax.set_title("Uniform ODT Sample")

# %%


odt_variable: GaussianMeasure = odt_gaussian_measure(
    finger_print=fp,
    finger_print_operator=fp_op,
    use_spatial_variability=True,
    amplitude=0.0002,
    point_multiplier=30,
)

odt_variable_sample = odt_variable.sample()

fig, ax, im = plot(
    odt_variable_sample
    * fp.ocean_projection()
    * 1000
    * fp.length_scale,
    coasts=True,
    cmap="seismic",
    symmetric=True,
    colorbar_label="ODT (mm)",
)
ax.set_title("Spatially Variable ODT Sample")

# %%

variability = odt_variability_field(fp, point_multiplier=30)

fig, ax, im = plot(
    variability,
    coasts=True,
    cmap="YlOrRd",
    colorbar_label="ODT Variability (multiplier x)",
)
ax.set_title("Synthetic ODT Variability Field")
plt.show()
# %%

print(
    "Comparing samples from the uniform and variable ODT measures..."
)

for i in range(3):
    s_uniform = (
        odt_uniform.sample()
        * fp.ocean_projection()
        * 1000
        * fp.length_scale
    )
    s_variable = (
        odt_variable.sample()
        * fp.ocean_projection()
        * 1000
        * fp.length_scale
    )

    vmax = max(
        np.nanmax(np.abs(s_uniform.to_array())),
        np.nanmax(np.abs(s_variable.to_array())),
    )

    fig_u, ax_u, _ = plot(
        s_uniform,
        coasts=True,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        colorbar_label="ODT (mm)",
    )
    ax_u.set_title(f"Uniform ODT Sample {i + 1}")

    fig_v, ax_v, _ = plot(
        s_variable,
        coasts=True,
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        colorbar_label="ODT (mm)",
    )
    ax_v.set_title(f"Variable ODT Sample {i + 1}")

plt.show()
