# %%
from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["font.size"] = 24

# %%
# Setup
order = 2.0
lmax = 256
fp = EarthState.from_defaults(lmax=lmax)
scale = 0.05 * fp.model.parameters.mean_sea_floor_radius

# %%
# Build the operators

# A: load -> [slc, disp, gpc, avc]
A = FingerPrintOperator(fp, load_parameters=(order, scale), response_parameters=(order + 1, scale))

# T: ice thickness change -> load
T = ice_thickness_change_to_load_operator(fp, A.domain)

# P: [slc, disp, gpc, avc] -> slc (projection onto component 0)
P = A.codomain.subspace_projection(0)

# C: [slc, disp, gpc, avc] -> ssh
C = sea_surface_height_operator(fp, A.codomain)

# %%
# Compose the full operators (ice thickness -> observable)

# Sea level change: ice thickness -> load -> response -> slc
B_slc = P @ A @ T

# Sea surface height: ice thickness -> load -> response -> ssh
B_ssh = C @ A @ T

# %%
# Define observation location
lat, lon = 33, -80
# %%
# Compute sensitivity kernels using the adjoint method

# For sea level change
v_slc = B_slc.codomain.dirac_representation(
    (lat, lon),
)
w_slc = B_slc.adjoint(v_slc)

# For sea surface height
v_ssh = B_ssh.codomain.dirac_representation(
    (lat, lon),
)
w_ssh = B_ssh.adjoint(v_ssh)

# %%
# Compute the difference kernel (SSH - SLC = solid Earth displacement contribution)
w_diff = w_ssh - w_slc

# %%

vmax = np.max([np.abs(w_slc).max(), np.abs(w_ssh).max()])
print()

# %%
# Plot Sea Level Change kernel
fig_slc_kern, ax_slc_kern, im_slc_kern = plot(
    w_slc * 1000,
    coasts=True,
    cmap="seismic",
    symmetric=True,
    gridlines=False,
)
ax_slc_kern.set_title(
    "Sea Level Change Sensitivity\n(per unit ice thickness)",
)
ax_slc_kern.plot(
    lon,
    lat,
    "m*",
    markersize=10,
    transform=ccrs.PlateCarree(),
)
fig_slc_kern.colorbar(
    im_slc_kern,
    ax=ax_slc_kern,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
    label="mm / m ice",
)
plt.tight_layout()

# %%
# Plot Sea Surface Height kernel
fig_ssh_kern, ax_ssh_kern, im_ssh_kern = plot(
    -1 * w_ssh * 10000,
    coasts=True,
    cmap="seismic",
    symmetric=True,
    gridlines=False,
)
ax_ssh_kern.set_title(
    "Sea Surface Height Sensitivity\n(per unit ice thickness)",
)
ax_ssh_kern.plot(
    lon,
    lat,
    "m*",
    markersize=10,
    transform=ccrs.PlateCarree(),
)
fig_ssh_kern.colorbar(
    im_ssh_kern,
    ax=ax_ssh_kern,
    orientation="horizontal",
    pad=0.05,
    shrink=0.8,
    label="mm / m ice",
)
plt.tight_layout()

# %%
plt.show()
# %%
