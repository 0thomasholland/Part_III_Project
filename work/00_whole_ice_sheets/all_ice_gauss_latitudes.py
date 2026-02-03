# %%
import numpy as np
from pygeoinf import GaussianMeasure
from pyslfp import FingerPrint, IceModel

from pyslfp_extras.measures import (
    ice_thickness_gaussian_measure,
)

# %%


alimetry_resolution = (
    1440  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(1, 90, alimetry_resolution)
gmsl_target_mean = np.array([0, 0.001, 0.01])
gmsl_target_std = np.array([0.001, 0.005])

fp = FingerPrint(lmax=64)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%

ice_thickness_measures = {}

for mean in gmsl_target_mean:
    for std in gmsl_target_std:
        _ice_thickness_measure: GaussianMeasure = (
            ice_thickness_gaussian_measure(
                finger_print=fp,
                finger_print_operator=fp_op,
                length_scale=0.2 * fp.mean_sea_floor_radius,
                gmsl_target_std=std,
                gmsl_target_mean=mean,
            )
        )
        ice_thickness_measures[(mean, std)] = (
            _ice_thickness_measure
        )

# %%

error_measures = {}

for latitude in latitudes:
    for mean in gmsl_target_mean:
        for std in gmsl_target_std:
            measure = ice_thickness_measures[(mean, std)]
            error_measures[(latitude, mean, std)] = (
                measure.restricted_to_altimetry(
                    latitude_range=latitude
                ).covariance.trace()
            )
