# %%
import numpy as np
from pyslfp import FingerPrint, IceModel

from pyslfp_extras.gmsl import altimetry_gmsl, gmsl_error

# %% variable setting

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0)

altimetry_resolution = 90  # number of points from 0 to 90˚ that are sampled by altimetry
load_latitude_resolution = 90  # number of points from 0 to 90˚ that define ice load latitude bands
load_thickness = 5  # degrees

# %%
