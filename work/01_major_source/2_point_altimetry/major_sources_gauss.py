# Major sources - Gaussian framework (Point Altimetry)
# Computes the GMSL estimation error for major sources (GIS, WAIS, EAIS)
# using a Gaussian measure framework with point-based altimetry estimation
# instead of surface averaging.
#

# %%
import numpy as np
from joblib import Parallel, delayed
from pygeoinf import GaussianMeasure
from pyslfp import FingerPrint, IceModel

from project.operators import (
    ice_thickness_to_gmsl_point_estimation_error_operator,
    ice_thickness_to_point_estimated_gmsl_operator,
)
from pygeoinf_extras import expectation, variance
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.measures import (
    east_antarctic_ice_thickness_gaussian_measure,
    greenland_ice_thickness_gaussian_measure,
    west_antarctic_ice_thickness_gaussian_measure,
)

# %%
# variable setting

# Latitudes for altimetry sampling (can be extended for multiple latitudes)
altimetry_latitudes = np.linspace(10, 90, 30)

# Set gmsl_target_std and gmsl_target_mean to 1 so they cancel out of the
# equation for the error
gmsl_target_stds = np.array([1.0])
gmsl_target_means = np.array([0.0])

fp = FingerPrint(lmax=64)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

length_scale = 0.2 * fp.mean_sea_floor_radius

# %%
# Create Gaussian measures for each major ice sheet source
# keyed by (source, mean, std)

ice_thickness_measures = {}

for mean in gmsl_target_means:
    for std in gmsl_target_stds:
        ice_thickness_measures[("gis", mean, std)] = (
            greenland_ice_thickness_gaussian_measure(
                finger_print=fp,
                finger_print_operator=fp_op,
                length_scale=length_scale,
                gmsl_target_std=std,
                gmsl_target_mean=mean,
            )
        )
        ice_thickness_measures[("wais", mean, std)] = (
            west_antarctic_ice_thickness_gaussian_measure(
                finger_print=fp,
                finger_print_operator=fp_op,
                length_scale=length_scale,
                gmsl_target_std=std,
                gmsl_target_mean=mean,
            )
        )
        ice_thickness_measures[("eais", mean, std)] = (
            east_antarctic_ice_thickness_gaussian_measure(
                finger_print=fp,
                finger_print_operator=fp_op,
                length_scale=length_scale,
                gmsl_target_std=std,
                gmsl_target_mean=mean,
            )
        )

# %%
# Compute error and estimate measures for each source at each latitude

error_measures = {}
estimate_measures = {}
true_gmsl_measures = {}

# parallel version


def compute_measures(
    source: str, latitude: float, mean: float, std: float
) -> tuple[
    tuple[str, float, float, float],
    GaussianMeasure,
    GaussianMeasure,
    GaussianMeasure,
]:
    _ice_measure = ice_thickness_measures[
        (source, mean, std)
    ]

    gmsl_op = gmsl_from_ice_thickness_operator(
        finger_print=fp, finger_print_operator=fp_op
    )
    estimated_gmsl_op = (
        ice_thickness_to_point_estimated_gmsl_operator(
            finger_print=fp,
            finger_print_operator=fp_op,
            altimetry_latitude_range=latitude,
        )
    )
    error_op = (
        ice_thickness_to_gmsl_point_estimation_error_operator(
            finger_print=fp,
            finger_print_operator=fp_op,
            altimetry_latitude_range=latitude,
        )
    )

    _true_gmsl_measure = _ice_measure.affine_mapping(
        operator=gmsl_op
    )
    _estimate_measure = _ice_measure.affine_mapping(
        operator=estimated_gmsl_op
    )
    _error_measure = _ice_measure.affine_mapping(
        operator=error_op
    )

    return (
        (source, latitude, mean, std),
        _true_gmsl_measure,
        _estimate_measure,
        _error_measure,
    )


results = Parallel(n_jobs=4, verbose=5)(
    delayed(compute_measures)(source, latitude, mean, std)
    for source in ["gis", "wais", "eais"]
    for latitude in altimetry_latitudes
    for mean in gmsl_target_means
    for std in gmsl_target_stds
)

for (
    key,
    true_measure,
    estimate_measure,
    error_measure,
) in results:
    true_gmsl_measures[key] = true_measure
    estimate_measures[key] = estimate_measure
    error_measures[key] = error_measure

# %%
# Compute statistics for each measure

error_stats = {}
estimate_stats = {}
true_gmsl_stats = {}


def compute_stats(
    key: tuple[str, float, float, float],
    measure: GaussianMeasure,
) -> tuple[
    tuple[str, float, float, float], tuple[float, float]
]:
    expectation_value = expectation(measure)
    std_value = np.sqrt(variance(measure))
    return key, (expectation_value, std_value)


results = Parallel(n_jobs=4, verbose=5)(
    delayed(compute_stats)(key, measure)
    for key, measure in error_measures.items()
)
for key, stat in results:
    error_stats[key] = stat

results = Parallel(n_jobs=4, verbose=5)(
    delayed(compute_stats)(key, measure)
    for key, measure in estimate_measures.items()
)
for key, stat in results:
    estimate_stats[key] = stat

results = Parallel(n_jobs=4, verbose=5)(
    delayed(compute_stats)(key, measure)
    for key, measure in true_gmsl_measures.items()
)
for key, stat in results:
    true_gmsl_stats[key] = stat

# %%
# Save the data

sources = ["gis", "wais", "eais"]

np.savez(
    "major_source_altimetry_errors_gauss.npz",
    altimetry_latitudes=altimetry_latitudes,
    gmsl_target_stds=gmsl_target_stds,
    gmsl_target_means=gmsl_target_means,
    sources=np.array(sources),
    # Error statistics
    error_means=np.array(
        [
            error_stats[(source, lat, mean, std)][0]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
    error_stds=np.array(
        [
            error_stats[(source, lat, mean, std)][1]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
    # Estimate statistics
    estimate_means=np.array(
        [
            estimate_stats[(source, lat, mean, std)][0]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
    estimate_stds=np.array(
        [
            estimate_stats[(source, lat, mean, std)][1]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
    # True GMSL statistics
    true_gmsl_means=np.array(
        [
            true_gmsl_stats[(source, lat, mean, std)][0]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
    true_gmsl_stds=np.array(
        [
            true_gmsl_stats[(source, lat, mean, std)][1]
            for source in sources
            for lat in altimetry_latitudes
            for mean in gmsl_target_means
            for std in gmsl_target_stds
        ]
    ),
)
