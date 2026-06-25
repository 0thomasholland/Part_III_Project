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

from pyslfp_extras.gmsl import (
    altimetry_gmsl,
    gmsl_error,
)

# %%
# variable setting

latitudes = np.array([50, 60, 65, 70, 75, 80, 90])
gis_contribution = np.linspace(0, 1, 20)
eais_contribution = np.linspace(0, 1, 20)

print(gis_contribution)

fp = EarthState.from_defaults(lmax=128)

gis_load = fp.greenland_load()
print(-fp.model.integrate(gis_load) / (fp.model.parameters.water_density * fp.ocean_area))
gis_load /= -fp.model.integrate(gis_load) / (fp.model.parameters.water_density * fp.ocean_area)
print(-fp.model.integrate(gis_load) / (fp.model.parameters.water_density * fp.ocean_area))

eais_load = fp.east_antarctic_load()
print(-fp.model.integrate(eais_load) / (fp.model.parameters.water_density * fp.ocean_area))
eais_load /= -fp.model.integrate(eais_load) / (fp.model.parameters.water_density * fp.ocean_area)
print(-fp.model.integrate(eais_load) / (fp.model.parameters.water_density * fp.ocean_area))

wais_load = fp.west_antarctic_load()
print(-fp.model.integrate(wais_load) / (fp.model.parameters.water_density * fp.ocean_area))
wais_load /= -fp.model.integrate(wais_load) / (fp.model.parameters.water_density * fp.ocean_area)
print(-fp.model.integrate(wais_load) / (fp.model.parameters.water_density * fp.ocean_area))

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

        slc, dis, _, avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(total_load)
        ssh = (slc + dis + centrifugal_potential_operator(fp.model)(avc) / fp.model.parameters.gravitational_acceleration)
        if (
            abs(
                1
                - -fp.model.integrate(total_load
                ) / (fp.model.parameters.water_density * fp.ocean_area)
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
