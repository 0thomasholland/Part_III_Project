# %%
from pyslfp import LinearSeaLevelEquation
from pyslfp.linear_operators.physics import (
    centrifugal_potential_operator,
)
from pyslfp.state import EarthState
import numpy as np
from numpy.typing import NDArray

from pyslfp_extras.gmsl import altimetry_gmsl, gmsl_error

alimetry_resolution = (
    1440  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(1, 90, alimetry_resolution)

fp = EarthState.from_defaults(lmax=256)

# %% load setting
load = fp.direct_load_from_ice_thickness_change(
    fp.ice_projection(value=0)
)
# normalise the load to 1mm gmsl change
plot(load * fp.ice_projection())
print(-fp.model.integrate(load) / (fp.model.parameters.water_density * fp.ocean_area))

load /= -fp.model.integrate(load) / (fp.model.parameters.water_density * fp.ocean_area)
plot(load * fp.ice_projection())
print(gmsl := -fp.model.integrate(load) / (fp.model.parameters.water_density * fp.ocean_area))

# %%

slc, dis, _, avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(load)
ssh = (slc + dis + centrifugal_potential_operator(fp.model)(avc) / fp.model.parameters.gravitational_acceleration)
ssh_altimetry = ssh * fp.altimetry_projection(
    latitude_max=66, latitude_min=-66
)

plotting_value = max(np.abs(ssh).max(), np.abs(slc).max())

fig1, ax1, im1 = plot(
    slc * fp.ocean_projection(),
    vmin=-plotting_value,
    vmax=plotting_value,
)
ax1.set_title("Sea Level Change")
fig2, ax2, im2 = plot(
    ssh * fp.ocean_projection(),
    vmin=-plotting_value,
    vmax=plotting_value,
)
ax2.set_title("Sea Surface Height Change")
fig3, ax3, im3 = plot(
    ssh_altimetry,
    vmin=-plotting_value,
    vmax=plotting_value,
)
ax3.set_title(
    "Sea Surface Height Change (Altimetry Projection at 66˚)"
)

# %%

estimated_gmsl = altimetry_gmsl(ssh, fp, latitude=66)

numeric_error = gmsl_error(
    true_gmsl=gmsl,
    estimated_gmsl=estimated_gmsl,
    error_type="numeric",
)

relative_error = gmsl_error(
    true_gmsl=gmsl,
    estimated_gmsl=estimated_gmsl,
    error_type="relative",
)

print(f"True GMSL: {gmsl:.4f} m")
print(f"Estimated GMSL: {estimated_gmsl:.4f} m")
print(f"Numeric Error: {numeric_error:.4f} m")
print(f"Relative Error: {relative_error:.4f} ")
print(f"Relative Error: {relative_error * 100:.4f} %")

# %%

estimated_gmsl_latitudes = np.zeros_like(latitudes)

for i, latitude in enumerate(latitudes):
    estimated_gmsl_latitudes[i] = altimetry_gmsl(
        ssh,
        fp,
        latitude=latitude,
    )
    if i % 100 == 0:
        print(f"Completed latitude {latitude:.2f}˚")

# %%

numeric_errors_latitudes: NDArray = gmsl_error(
    true_gmsl=gmsl * np.ones_like(latitudes),
    estimated_gmsl=estimated_gmsl_latitudes,
    error_type="numeric",
)

relative_errors_latitudes: NDArray = gmsl_error(
    true_gmsl=gmsl * np.ones_like(latitudes),
    estimated_gmsl=estimated_gmsl_latitudes,
    error_type="relative",
)

# %%

np.savez(
    "all_ice_sheets_altimetry_errors.npz",
    latitudes=latitudes,
    numeric_errors=numeric_errors_latitudes,
    relative_errors=relative_errors_latitudes,
)
