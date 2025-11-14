import random

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
from matplotlib import colors
from scipy import stats

from Part_III_Project import (
    ocean_dynamic_topography_measures,
)

lmax = 256
fp = sl.FingerPrint(
    lmax=lmax,
)
fp.set_state_from_ice_ng()

# %%
# model space

scale = 0.1 * fp.mean_sea_floor_radius

model_space = inf.symmetric_space.sphere.Sobolev(
    fp.lmax,
    2,
    scale,
    radius=fp.mean_sea_floor_radius,
)

# %% [markdown]
# ## Forward Model
# object that represents the complete chain of physical processes and measurements linking our model (ice thickness change) to our data (sea surface height change).
#
# 1. ice_projection_operator
# 2. ice_thickness_change_to_load_operator
# 3. fingerprint_operator
# 4. sea_surface_height_to_measurement_operator

# %%
op1 = sl.ice_projection_operator(fp, model_space)
op2 = sl.ice_thickness_change_to_load_operator(fp, model_space)
op3 = fp.as_sobolev_linear_operator(2, scale)
op4 = sl.sea_surface_height_operator(fp, op3.codomain)

forward_model = op4 @ op3 @ op2 @ op1
data_space = forward_model.codomain

sea_level_field = op3.codomain.subspace_projection(0)
sea_height_field = op4.codomain
# mapping from model space to the sea level field
A_sl = sea_level_field @ op3 @ op2 @ op1


# %% [markdown]
# ## Setting up "error"
# The error in this calculation comes from added ocean dynmaic topography (ODT) noise to the sea surface height measurements.
# We will model this as a Gaussian process with a Sobolev covariance operator.


# %%

odt_length_scale = 0.01 * fp.mean_sea_floor_radius
odt_amplitude_95_range = 1 / fp.length_scale  # in units of

odt_measure,odt_load_measure =ocean_dynamic_topography_measures(
    fingerprint=fp,
    op3,
    length_scale=odt_length_scale,
    amplitude_95_range=odt_amplitude_95_range,
)
