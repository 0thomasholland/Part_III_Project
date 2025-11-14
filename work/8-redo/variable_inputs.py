# %%
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyslfp as sl
from joblib import Parallel, delayed, dump, load

from Part_III_Project import (
    get_gmsl_measure,
    get_stats_from_measure,
    ice_thickness_change_measures,
    load_measure,
    ocean_dynamic_topography_measures,
    sea_level_change_measure,
    sea_surface_height_measure,
)

directory = path.dirname(path.abspath(__file__))
verbosity = 4
# %%
print("Starting variable input GMSL estimation script")


# --- Set up a fingerprint instance ---
lmax = 256
fp = sl.FingerPrint(
    lmax=lmax,
    # earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)


odt_length_scales = (
    # np.linspace(0.01, 0.5, 10)
    # * fp.mean_sea_floor_radius
    np.linspace(0.001, 0.5, 10) * fp.mean_sea_floor_radius
)
odt_amplitude_95_ranges = (
    np.array(
        [
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.05,
        ],
        # [0.01, 0.1, 1.0],
    )
    / fp.length_scale
)  # in units of sea level, non-dimensionalized

ice_length_scales = (
    np.linspace(0.1, 0.3, 6) * fp.mean_sea_floor_radius
)
ice_gmsl_target_stds = (
    np.linspace(0.001, 0.1, 6) / fp.length_scale
)  # in meters, non-dimensionalized
# in meters, non-dimensionalized
ice_shifts = (
    np.linspace(-200, 200, 6) / fp.length_scale
)  # in meters, non-dimensionalized

output_data = pd.DataFrame({})

number_of_jobs = (
    len(odt_length_scales)
    * len(odt_amplitude_95_ranges)
    * len(
        ice_length_scales,
    )
    * len(ice_gmsl_target_stds)
    * len(ice_shifts)
)
print(f"Total number of jobs to run: {number_of_jobs}")


def main(
    odt_length_scale,
    odt_amplitude_95_range,
    ice_length_scale,
    ice_gmsl_target_std,
    net_ice_thickness_change,
    fingerprint_operator,
    fp,
):  # -> tuple[Any, Any, Any, Any, Any, Any]:
    _ocean_dynamic_measure, _ocean_dynamic_load_measure = (
        ocean_dynamic_topography_measures(
            fingerprint=fp,
            fingerprint_operator=fingerprint_operator,
            length_scale=odt_length_scale,
            amplitude_95_range=odt_amplitude_95_range,
        )
    )
    _ice_thickness_measure, _ice_load_measure = (
        ice_thickness_change_measures(
            fingerprint=fp,
            fingerprint_operator=fingerprint_operator,
            length_scale=ice_length_scale,
            ice_gmsl_target_std=ice_gmsl_target_std,
            net_thickness_change=net_ice_thickness_change,
        )
    )
    _direct_load_measure = load_measure(
        ice_thickness_load_measure=_ice_load_measure,
        odt_load_measure=_ocean_dynamic_load_measure,
    )

    _slc = sea_level_change_measure(
        fingerprint_operator=fingerprint_operator,
        fingerprint=fp,
        load_measure=_direct_load_measure,
    )
    _ssh, _ssh_odc, _ = sea_surface_height_measure(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        load_measure=_direct_load_measure,
        odt_measure=_ocean_dynamic_measure,
    )
    _slc_gmsl = get_gmsl_measure(_slc, fp)
    _ssh_gmsl = get_gmsl_measure(_ssh, fp)
    _ssh_odt_gmsl = get_gmsl_measure(_ssh_odc, fp)
    _slc_gmsl_expectation = _slc_gmsl.expectation[0] * fp.length_scale
    _slc_gmsl_std = (
        np.sqrt(
            _slc_gmsl.covariance.matrix(dense=True)[0, 0],
        )
        * fp.length_scale
    )
    _ssh_gmsl_expectation = _ssh_gmsl.expectation[0] * fp.length_scale
    _ssh_gmsl_std = (
        np.sqrt(
            _ssh_gmsl.covariance.matrix(dense=True)[0, 0],
        )
        * fp.length_scale
    )
    _ssh_odt_gmsl_expectation = (
        _ssh_odt_gmsl.expectation[0] * fp.length_scale
    )
    _ssh_odt_gmsl_std = (
        np.sqrt(
            _ssh_odt_gmsl.covariance.matrix(dense=True)[0, 0],
        )
        * fp.length_scale
    )
    return (
        odt_length_scale,
        odt_amplitude_95_range,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
        _slc_gmsl_expectation,
        _slc_gmsl_std,
        _ssh_gmsl_expectation,
        _ssh_gmsl_std,
        _ssh_odt_gmsl_expectation,
        _ssh_odt_gmsl_std,
    )


results = Parallel(n_jobs=-1, verbose=verbosity)(
    delayed(main)(
        odt_length_scale,
        odt_amplitude_95_range,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
        fingerprint_operator,
        fp,
    )
    for odt_length_scale in odt_length_scales
    for odt_amplitude_95_range in odt_amplitude_95_ranges
    for ice_length_scale in ice_length_scales
    for ice_gmsl_target_std in ice_gmsl_target_stds
    for net_ice_thickness_change in ice_shifts
)

for res in results:
    (
        odt_length_scale,
        odt_amplitude_95_range,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
        slc_gmsl_expectation,
        slc_gmsl_std,
        ssh_gmsl_expectation,
        ssh_gmsl_std,
        ssh_odt_gmsl_expectation,
        ssh_odt_gmsl_std,
    ) = res
    output_data = pd.concat(
        [
            output_data,
            pd.DataFrame(
                {
                    "odt_length_scale": [odt_length_scale],
                    "odt_amplitude_95_range": [
                        odt_amplitude_95_range,
                    ],
                    "ice_length_scale": [ice_length_scale],
                    "ice_gmsl_target_std": [ice_gmsl_target_std],
                    "net_ice_thickness_change": [
                        net_ice_thickness_change,
                    ],
                    "slc_gmsl_expectation": [slc_gmsl_expectation],
                    "slc_gmsl_std": [slc_gmsl_std],
                    "ssh_gmsl_expectation": [ssh_gmsl_expectation],
                    "ssh_gmsl_std": [ssh_gmsl_std],
                    "ssh_odt_gmsl_expectation": [
                        ssh_odt_gmsl_expectation,
                    ],
                    "ssh_odt_gmsl_std": [ssh_odt_gmsl_std],
                },
            ),
        ],
        ignore_index=True,
    )
file_name = "variable_input_data_big.pkl"
# load existing data if present, and append new data
if path.exists(
    path.join(directory, "output", file_name),
):
    existing_data = load(
        path.join(
            directory,
            "output",
            file_name,
        ),
    )
    output_data = pd.concat(
        [existing_data, output_data],
        ignore_index=True,
    )

dump(
    output_data,
    path.join(directory, "output", file_name),
)
