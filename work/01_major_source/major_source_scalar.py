# Major sources - scalar field
# plots the major sources (GIS, WAIS, EAIS) at a fixed amount of melt across the
# ice sheet and then calculates the error of using ∆SSH ≈ ∆SL for different
# satalite ranges.
#

# %%
from pyslfp import LinearSeaLevelEquation
from pyslfp.linear_operators.physics import (
    centrifugal_potential_operator,
)
from pyslfp.state import EarthState
import numpy as np

from pyslfp_extras.gmsl import altimetry_gmsl, gmsl_error

# %%
# variable setting

alimetry_resolution = (
    1440  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(1, 90, alimetry_resolution)

fp = EarthState.from_defaults(lmax=256)

gis_load = fp.greenland_load()
eais_load = fp.east_antarctic_load()
wais_load = fp.west_antarctic_load()

# %%
# fingerprint response to major ice sheets

gis_slc, gis_dis, _, gis_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(gis_load)
gis_ssh = (gis_slc + gis_dis + centrifugal_potential_operator(fp.model)(gis_avc
) / fp.model.parameters.gravitational_acceleration)

eais_slc, eais_dis, _, eais_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(eais_load)
eais_ssh = (eais_slc + eais_dis + centrifugal_potential_operator(fp.model)(eais_avc
) / fp.model.parameters.gravitational_acceleration)

wais_slc, wais_dis, _, wais_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(wais_load)
wais_ssh = (wais_slc + wais_dis + centrifugal_potential_operator(fp.model)(wais_avc
) / fp.model.parameters.gravitational_acceleration)
# %%
# calculate true gmsl from ice load

gis_gmsl: float = -fp.model.integrate(gis_load
) / (fp.model.parameters.water_density * fp.ocean_area)
gis_estimated_gmsl = np.zeros_like(latitudes)

eais_gmsl: float = -fp.model.integrate(eais_load
) / (fp.model.parameters.water_density * fp.ocean_area)
eais_estimated_gmsl = np.zeros_like(latitudes)

wais_gmsl: float = -fp.model.integrate(wais_load
) / (fp.model.parameters.water_density * fp.ocean_area)
wais_estimated_gmsl = np.zeros_like(latitudes)

for i, lat in enumerate(latitudes):
    gis_estimated_gmsl[i] = altimetry_gmsl(
        gis_ssh,
        fp,
        latitude=lat,
    )
    eais_estimated_gmsl[i] = altimetry_gmsl(
        eais_ssh,
        fp,
        latitude=lat,
    )
    wais_estimated_gmsl[i] = altimetry_gmsl(
        wais_ssh,
        fp,
        latitude=lat,
    )

    if i % 100 == 0:
        print(f"Completed latitude {lat:.2f}˚")

# %%
print(gis_estimated_gmsl)

print(gis_gmsl)

# %%
gis_errors = gmsl_error(
    true_gmsl=gis_gmsl * np.ones_like(latitudes),
    estimated_gmsl=gis_estimated_gmsl,
    error_type="relative",
)

eais_errors = gmsl_error(
    true_gmsl=eais_gmsl * np.ones_like(latitudes),
    estimated_gmsl=eais_estimated_gmsl,
    error_type="relative",
)

wais_errors = gmsl_error(
    true_gmsl=wais_gmsl * np.ones_like(latitudes),
    estimated_gmsl=wais_estimated_gmsl,
    error_type="relative",
)

print(gis_errors)

# %% save data

np.savez(
    "major_source_altimetry_errors_scalar.npz",
    latitudes=latitudes,
    gis_errors=gis_errors,
    eais_errors=eais_errors,
    wais_errors=wais_errors,
)
