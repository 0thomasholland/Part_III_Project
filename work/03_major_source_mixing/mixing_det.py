# Major sources - scalar field
# plots the major sources (GIS, WAIS, EAIS) at a fixed amount of melt across the
# ice sheet and then calculates the error of using ∆SSH ≈ ∆SL for different
# satalite ranges.
#

# %%

import numpy as np
from pyslfp import FingerPrint, IceModel, plot

from pyslfp_extras.gmsl import (
    altimetry_gmsl,
    gmsl_error,
    gmsl_from_ice_load_operator,
)

# %%
# variable setting

latitudes = np.array([50, 60, 65, 70, 75, 80, 90])
gis_contribution = np.linspace(0, 1, 20)
eais_contribution = np.linspace(0, 1, 20)

print(gis_contribution)

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

gis_load = fp.greenland_load()
print(fp.mean_sea_level_change(direct_load=gis_load))
gis_load /= fp.mean_sea_level_change(direct_load=gis_load)
print(fp.mean_sea_level_change(direct_load=gis_load))

eais_load = fp.east_antarctic_load()
print(fp.mean_sea_level_change(direct_load=eais_load))
eais_load /= fp.mean_sea_level_change(direct_load=eais_load)
print(fp.mean_sea_level_change(direct_load=eais_load))

wais_load = fp.west_antarctic_load()
print(fp.mean_sea_level_change(direct_load=wais_load))
wais_load /= fp.mean_sea_level_change(direct_load=wais_load)
print(fp.mean_sea_level_change(direct_load=wais_load))


# %%
# for each contribution pre-make ssh SHGrid
sshes = []

for gis_contrib in gis_contribution:
    for eais_contrib in eais_contribution:
        wais_contrib = 1.0 - gis_contrib - eais_contrib
        if wais_contrib < 0:
            continue
        total_load = (
            gis_contrib * gis_load
            + eais_contrib * eais_load
            + wais_contrib * wais_load
        )

        slc, dis, _, avc = fp(direct_load=total_load)
        ssh = fp.sea_surface_height_change(slc, dis, avc)
        if (
            abs(
                1
                - fp.mean_sea_level_change(
                    direct_load=total_load
                )
            )
            > 1e-5
        ):
            raise ValueError("Load normalization error")
        else:
            sshes.append(
                {
                    "gis": gis_contrib,
                    "eais": eais_contrib,
                    "wais": wais_contrib,
                    "ssh": ssh,
                }
            )

# %%
# calculate gmsl estimates and errors

results = []

for ssh_dict in sshes:
    ssh = ssh_dict["ssh"]
    gis_contrib = ssh_dict["gis"]
    eais_contrib = ssh_dict["eais"]
    wais_contrib = ssh_dict["wais"]

    for i, latitude in enumerate(latitudes):
        estimated_gmsl = altimetry_gmsl(
            ssh,
            fp,
            latitude=latitude,
        )
        error = gmsl_error(
            np.ones_like(estimated_gmsl),
            estimated_gmsl,
            error_type="relative",
        )

        results.append(
            {
                "gis": gis_contrib,
                "eais": eais_contrib,
                "wais": wais_contrib,
                "latitude": latitude,
                "true_gmsl": 1,
                "estimated_gmsl": estimated_gmsl,
                "relative_error": error,
            }
        )

# %%
# use numpy savez

np.savez(
    "mixing_det_results.npz",
    results=results,
)

print(results)
