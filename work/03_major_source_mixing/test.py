# %%
from pyslfp import LinearSeaLevelEquation
from pyslfp.linear_operators.physics import (
    centrifugal_potential_operator,
)
from pyslfp.state import EarthState

fp = EarthState.from_defaults(lmax=256)

gis_load = fp.greenland_load()
eais_load = fp.east_antarctic_load()
total_load = gis_load + eais_load

gis_slc, gis_dis, _, gis_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(gis_load)
eais_slc, eais_dis, _, eais_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(eais_load)
total_slc, total_dis, _, total_avc = LinearSeaLevelEquation(fp).solve_sea_level_equation(total_load
)

gis_ssh = (gis_slc + gis_dis + centrifugal_potential_operator(fp.model)(gis_avc
) / fp.model.parameters.gravitational_acceleration)
eais_ssh = (eais_slc + eais_dis + centrifugal_potential_operator(fp.model)(eais_avc
) / fp.model.parameters.gravitational_acceleration)
total_ssh = (total_slc + total_dis + centrifugal_potential_operator(fp.model)(total_avc
) / fp.model.parameters.gravitational_acceleration)

combined_ssh = gis_ssh + eais_ssh
difference_ssh = total_ssh - combined_ssh

plot(total_ssh)
plot(combined_ssh)

plot(difference_ssh)
# %%

rel_ssh_diff = 100 * difference_ssh / total_ssh
plot(rel_ssh_diff, symmetric=True)
