# Major sources - scalar field
# plots the major sources (GIS, WAIS, EAIS) at a fixed amount of melt across the
# ice sheet and then calculates the error of using ∆SSH ≈ ∆SL for different
# satalite ranges.
#

# %%
import numpy as np
from numpy.typing import NDArray
from pyslfp import FingerPrint, IceModel

from pyslfp_extras.gmsl import altimetry_gmsl, gmsl_error

# %%
# variable setting


alimetry_resolution = (
    1440  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(1, 90, alimetry_resolution)

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

gis_load = fp.greenland_load()
eais_load = fp.east_antarctic_load()
wais_load = fp.west_antarctic_load()

# %%
# fingerprint response to major ice sheets

gis_slc, gis_dis, _, gis_avc = fp(direct_load=gis_load)
gis_ssh = fp.sea_surface_height_change(
    gis_slc, gis_dis, gis_avc
)

eais_slc, eais_dis, _, eais_avc = fp(direct_load=eais_load)
eais_ssh = fp.sea_surface_height_change(
    eais_slc, eais_dis, eais_avc
)

wais_slc, wais_dis, _, wais_avc = fp(direct_load=wais_load)
wais_ssh = fp.sea_surface_height_change(
    wais_slc, wais_dis, wais_avc
)
# %%
# calculate true gmsl from ice load

gis_gmsl: float = fp.mean_sea_level_change(
    direct_load=gis_load
)
gis_estimated_gmsl = np.zeros_like(latitudes)

eais_gmsl: float = fp.mean_sea_level_change(
    direct_load=eais_load
)
eais_estimated_gmsl = np.zeros_like(latitudes)

wais_gmsl: float = fp.mean_sea_level_change(
    direct_load=wais_load
)
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
