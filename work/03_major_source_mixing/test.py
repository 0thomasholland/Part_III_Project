# %%
import numpy as np
from pyslfp import FingerPrint, IceModel, plot

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


gis_load = fp.greenland_load()
eais_load = fp.east_antarctic_load()
total_load = gis_load + eais_load

gis_slc, gis_dis, _, gis_avc = fp(direct_load=gis_load)
eais_slc, eais_dis, _, eais_avc = fp(direct_load=eais_load)
total_slc, total_dis, _, total_avc = fp(
    direct_load=total_load
)

gis_ssh = fp.sea_surface_height_change(
    gis_slc, gis_dis, gis_avc
)
eais_ssh = fp.sea_surface_height_change(
    eais_slc, eais_dis, eais_avc
)
total_ssh = fp.sea_surface_height_change(
    total_slc, total_dis, total_avc
)

combined_ssh = gis_ssh + eais_ssh
difference_ssh = total_ssh - combined_ssh


plot(total_ssh)
plot(combined_ssh)

plot(difference_ssh)
# %%

rel_ssh_diff = 100 * difference_ssh / total_ssh
plot(rel_ssh_diff, symmetric=True)
