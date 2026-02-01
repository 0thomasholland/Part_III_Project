import numpy as np
import pandas as pd
import pyslfp as sl
from joblib import Parallel, delayed
from pygeoinf import GaussianMeasure, LinearOperator
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
    sea_surface_height_operator,
)


def ice_thickness_change_measures(
    fingerprint: FingerPrint,
    fingerprint_operator: LinearOperator,
    length_scale: float,
    ice_gmsl_target_std: float,
    net_thickness_change: float,
) -> tuple[GaussianMeasure, GaussianMeasure]:
    """-> ice_thickness_measure, ice_thickness_load_measure
    Takes a length scale for the ice thickness changes and either a target GMSL std or a 95% range for the ice thickness changes to set the amplitude of the ocean dynamic topography measure.
    All parameters should be passed in non-dimensionalized form (already divided by fingerprint.length_scale).
    """
    initial_ice_thickness_measure = (
        fingerprint_operator.domain.heat_kernel_gaussian_measure(
            length_scale,
        )
    )
    ice_projection = sl.ice_projection_operator(
        fingerprint,
        fingerprint_operator.domain,
    )
    ice_thickness_measure = initial_ice_thickness_measure.affine_mapping(
        operator=ice_projection,
    )
    GMSL_weighting_function = (
        -fingerprint.ice_density
        * fingerprint.one_minus_ocean_function
        * fingerprint.ice_projection(value=0)
        * fingerprint.length_scale
        / (fingerprint.water_density * fingerprint.ocean_area)
    )
    GMSL_operator = sl.averaging_operator(
        fingerprint_operator.domain,
        [GMSL_weighting_function],
    )
    GMSL_measure = ice_thickness_measure.affine_mapping(
        operator=GMSL_operator,
    )
    GMSL_variance = GMSL_measure.covariance.matrix(dense=True)[0, 0]
    GMSL_std = np.sqrt(GMSL_variance)

    # Normalise the ice load thickness measure and then recompute the load measure
    ice_thickness_measure *= ice_gmsl_target_std / GMSL_std
    if net_thickness_change != 0.0:
        shift_vector = ice_thickness_measure.domain.project_function(
            lambda point: net_thickness_change,
        )
        ice_thickness_measure = ice_thickness_measure.affine_mapping(
            translation=shift_vector,
        )
    ice_thickness_measure = ice_thickness_measure.affine_mapping(
        operator=ice_projection_operator(
            fingerprint,
            fingerprint_operator.domain,
        ),
    )
    ice_load_measure = ice_thickness_measure.affine_mapping(
        operator=ice_thickness_change_to_load_operator(
            finger_print=fingerprint,
            load_space=fingerprint_operator.domain,
        ),
    )
    return ice_thickness_measure, ice_load_measure


def get_sea_level_change_measure(
    fingerprint_operator: LinearOperator,
    fingerprint: FingerPrint,
    load_measure: GaussianMeasure,
) -> GaussianMeasure:
    """Returns a single sea level change measure"""
    if load_measure.domain != fingerprint_operator.domain:
        raise ValueError(
            "load_measure and fingerprint_operator must be defined on the same domain",
        )
    response_measure = load_measure.affine_mapping(
        operator=fingerprint_operator,
    )

    response_space = response_measure.domain
    projection_operator = response_space.subspace_projection(
        0,
    )

    slc_measure = response_measure.affine_mapping(
        operator=projection_operator,
    )

    slc_measure = slc_measure.affine_mapping(
        operator=ocean_projection_operator(
            fingerprint,
            fingerprint_operator.domain,
        ),
    )

    return slc_measure


def sea_surface_height_measure(
    fingerprint: FingerPrint,
    fingerprint_operator: LinearOperator,
    load_measure: GaussianMeasure,
    odt_measure: GaussianMeasure
    | tuple[GaussianMeasure, GaussianMeasure]
    | None = None,
    noise_measure: GaussianMeasure | None = None,
) -> GaussianMeasure:
    ssh_measure = load_measure.affine_mapping(
        operator=sea_surface_height_operator(
            fingerprint,
            fingerprint_operator.codomain,
        )
        @ fingerprint_operator,
    )
    return ssh_measure


def get_gmsl_measure(
    measure: GaussianMeasure,
    fingerprint: FingerPrint,
) -> GaussianMeasure:
    weighting_function = (
        -fingerprint.ice_density
        * fingerprint.one_minus_ocean_function
        * fingerprint.ice_projection(value=0)
        * fingerprint.length_scale
        / (fingerprint.water_density * fingerprint.ocean_area)
    )
    gmsl_operator = averaging_operator(
        measure.domain,
        [weighting_function],
    )
    gmsl_measure = measure.affine_mapping(
        operator=gmsl_operator,
    )
    return gmsl_measure


def stats(
    lmax,
    ice_length_scale,
    ice_gmsl_target_std,
    net_ice_thickness_change,
):
    fp = sl.FingerPrint(
        lmax=lmax,
        earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
    )

    fp.set_state_from_ice_ng()

    fingerprint_operator = fp.as_sobolev_linear_operator(
        2,
        0.1 * fp.mean_sea_floor_radius,
    )

    _, ice_load_measure = ice_thickness_change_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=ice_length_scale,
        ice_gmsl_target_std=ice_gmsl_target_std,
        net_thickness_change=net_ice_thickness_change,
    )

    sea_level_change_measure = get_sea_level_change_measure(
        fingerprint_operator=fingerprint_operator,
        fingerprint=fp,
        load_measure=ice_load_measure,
    )
    ssh_measure = sea_surface_height_measure(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        load_measure=ice_load_measure,
    )

    _slc_gmsl = get_gmsl_measure(sea_level_change_measure, fp)
    _ssh_gmsl = get_gmsl_measure(ssh_measure, fp)

    _slc_mean = _slc_gmsl.expectation[0]
    _slc_std = np.sqrt(_slc_gmsl.covariance.matrix(dense=True)[0, 0])

    _ssh_mean = _ssh_gmsl.expectation[0]
    _ssh_std = np.sqrt(_ssh_gmsl.covariance.matrix(dense=True)[0, 0])

    return (
        lmax,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
        _slc_mean,
        _slc_std,
        _ssh_mean,
        _ssh_std,
    )


# lmaxes = [64]
# ice_length_scales = [0.2 * 6368000.0]
# ice_gmsl_target_stds = [0.01]
# net_ice_thickness_changes = [0]

lmaxes = np.trunc(2 ** np.linspace(6, 9.6, 10)).astype(int)
ice_length_scales = np.linspace(0.05, 0.2, 4) * 6368000.0
ice_gmsl_target_stds = np.linspace(0.002, 1, 5)  # in meters
net_ice_thickness_changes = np.linspace(-200, 200, 4)

print(
    f"Number of iterations: {len(lmaxes) * len(ice_length_scales) * len(ice_gmsl_target_stds) * len(net_ice_thickness_changes)}",
)

results = Parallel(n_jobs=-1, verbose=3)(
    delayed(stats)(
        lmax,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
    )
    for lmax in lmaxes
    for ice_length_scale in ice_length_scales
    for ice_gmsl_target_std in ice_gmsl_target_stds
    for net_ice_thickness_change in net_ice_thickness_changes
)

df = pd.DataFrame(
    results,
    columns=[
        "lmax",
        "ice_length_scale",
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "slc_mean",
        "slc_std",
        "ssh_mean",
        "ssh_std",
    ],
)

df["variance_ratio"] = (df["ssh_std"] ** 2) / (df["slc_std"] ** 2)
df["mean_errors"] = df["ssh_mean"] - df["slc_mean"]
df["standardized_errors"] = (df["mean_errors"]) / np.sqrt(
    df["slc_std"] ** 2 + df["ssh_std"] ** 2,
)

# Wasserstein distance
df["w2_distances"] = np.sqrt(
    (np.array(df["slc_mean"]) - np.array(df["ssh_mean"])) ** 2
    + (np.array(df["slc_std"]) - np.array(df["ssh_std"])) ** 2,
)

# kullback-leibler divergence, slc as true distribution
df["kl_divergences"] = (
    (
        np.log(df["ssh_std"] / df["slc_std"])
        + (df["slc_std"] ** 2 + (df["slc_mean"] - df["ssh_mean"]) ** 2)
        / (2 * df["ssh_std"] ** 2)
        - 0.5
    )
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

df.to_csv(
    "work/6-lmax_issues/outputs/explore_lmax/large_dataset.csv",
    index=False,
)
