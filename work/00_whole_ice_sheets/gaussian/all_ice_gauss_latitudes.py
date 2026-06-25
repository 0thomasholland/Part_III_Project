# %%
from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
import numpy as np
from joblib import Parallel, delayed
from pygeoinf import GaussianMeasure

from project.operators import (
    ice_thickness_to_estimated_gmsl_operator,
    ice_thickness_to_gmsl_estimation_error_operator,
)
from pygeoinf_extras import expectation, variance
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

# %%

alimetry_resolution = (
    300  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(10, 90, alimetry_resolution)
gmsl_target_mean = np.array([-0.01, -0.001, 0, 0.001, 0.01])
gmsl_target_std = np.array([0.001, 0.005, 0.01])

fp = EarthState.from_defaults(lmax=128)

fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

# %%

ice_pattern = IceSheetChange.UniformPattern()

ice_thickness_measures = {}

for mean in gmsl_target_mean:
    for std in gmsl_target_std:
        ice_change = IceSheetChange.global_ice(
            finger_print=fp,
            finger_print_operator=fp_op,
            length_scale=0.2 * fp.model.parameters.mean_sea_floor_radius,
            pattern=ice_pattern,
            ice_gmsl_std=std,
            gmsl_target_mean=mean,
        )
        _ice_thickness_measure: GaussianMeasure = (
            ice_change.ice_thickness_measure
        )
        ice_thickness_measures[(mean, std)] = (
            _ice_thickness_measure
        )

# %%

error_measures = {}
estimate_measures = {}

# serial version

# for latitude in latitudes:
#     for mean in gmsl_target_mean:
#         for std in gmsl_target_std:
#             _ice_measure = ice_thickness_measures[
#                 (mean, std)
#             ]
#             _error_measure = _ice_measure.affine_mapping(
#                 operator=ice_thickness_to_gmsl_estimation_error_operator(
#                     finger_print=fp,
#                     finger_print_operator=fp_op,
#                     altimetry_latitude_range=latitude,
#                 )
#             )
#             _estimate_measure = _ice_measure.affine_mapping(
#                 operator=ice_thickness_to_estimated_gmsl_operator(
#                     finger_print=fp,
#                     finger_print_operator=fp_op,
#                     altimetry_latitude_range=latitude,
#                 )
#             )
#             error_measures[(latitude, mean, std)] = (
#                 _error_measure
#             )
#             estimate_measures[(latitude, mean, std)] = (
#                 _estimate_measure
#             )

# parallel version

def compute_measures(latitude, mean, std):
    _ice_measure = ice_thickness_measures[(mean, std)]
    _error_measure = _ice_measure.affine_mapping(
        operator=ice_thickness_to_gmsl_estimation_error_operator(
            finger_print=fp,
            finger_print_operator=fp_op,
            altimetry_latitude_range=latitude,
        )
    )
    _estimate_measure = _ice_measure.affine_mapping(
        operator=ice_thickness_to_estimated_gmsl_operator(
            finger_print=fp,
            finger_print_operator=fp_op,
            altimetry_latitude_range=latitude,
        )
    )
    return (
        (latitude, mean, std),
        _error_measure,
        _estimate_measure,
    )

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_measures)(latitude, mean, std)
    for latitude in latitudes
    for mean in gmsl_target_mean
    for std in gmsl_target_std
)
for key, measure, estimate_measure in results:
    error_measures[key] = measure
    estimate_measures[key] = estimate_measure

# %%
error_stats = {}
estimate_stats = {}

# serial version

# for key, measure in error_measures.items():
#     expectation_value = expectation(measure)
#     std_value = variance(measure)
#     stats[key] = (expectation_value, std_value)
# for key, measure in gmsl_measures.items():
#    expectation_value = expectation(measure)
#    std_value = variance(measure)
#    gmsl_stats[key] = (expectation_value, std_value)
# for key, measure in estimate_measures.items():
#    expectation_value = expectation(measure)
#    std_value = variance(measure)
#    estimate_stats[key] = (expectation_value, std_value)
#

# parallel version

def compute_stats(key, measure):
    expectation_value = expectation(measure)
    std_value = np.sqrt(variance(measure))
    return key, (expectation_value, std_value)

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_stats)(key, measure)
    for key, measure in error_measures.items()
)
for key, stat in results:
    error_stats[key] = stat

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(compute_stats)(key, measure)
    for key, measure in estimate_measures.items()
)
for key, stat in results:
    estimate_stats[key] = stat

# %%
# save the data

np.savez(
    "all_ice_sheets_gauss_latitudes.npz",
    latitudes=np.array(
        [
            latitude
            for _ in gmsl_target_mean
            for _ in gmsl_target_std
            for latitude in latitudes
        ]
    ),
    gmsl_means=np.array(
        [
            gmsl_mean
            for gmsl_mean in gmsl_target_mean
            for _ in gmsl_target_std
            for _ in latitudes
        ]
    ),
    gmsl_stds=np.array(
        [
            gmsl_std
            for _ in gmsl_target_mean
            for gmsl_std in gmsl_target_std
            for _ in latitudes
        ]
    ),
    error_means=[
        error_stats[(latitude, gmsl_mean, gmsl_std)][0]
        for gmsl_mean in gmsl_target_mean
        for gmsl_std in gmsl_target_std
        for latitude in latitudes
    ],
    error_stds=[
        error_stats[(latitude, gmsl_mean, gmsl_std)][1]
        for gmsl_mean in gmsl_target_mean
        for gmsl_std in gmsl_target_std
        for latitude in latitudes
    ],
    estimate_means=[
        estimate_stats[(latitude, gmsl_mean, gmsl_std)][0]
        for gmsl_mean in gmsl_target_mean
        for gmsl_std in gmsl_target_std
        for latitude in latitudes
    ],
    estimate_stds=[
        estimate_stats[(latitude, gmsl_mean, gmsl_std)][1]
        for gmsl_mean in gmsl_target_mean
        for gmsl_std in gmsl_target_std
        for latitude in latitudes
    ],
)
