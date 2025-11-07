import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyslfp as sl
from joblib import Parallel, delayed, dump, load

from Part_III_Project import (
    get_stats_from_measure,
    gmsl_measure,
    ice_thickness_change_measures,
    load_measure,
    ocean_dynamic_topography_measures,
    sea_level_change_measure,
    sea_surface_height_measure,
)

# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=128)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)


odt_length_scales = (
    # np.linspace(0.01, 0.5, 10) * fp.mean_sea_floor_radius
    np.linspace(0.01, 0.5, 3) * fp.mean_sea_floor_radius
)
odt_amplitude_95_ranges = np.array(
    # [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 10],
    [0.001, 0.01, 0.1, 1.0, 10],
)  # in units of sea level

ice_length_scales = (
    # np.linspace(0.1, 0.5, 10) * fp.mean_sea_floor_radius
    np.linspace(0.1, 0.5, 3) * fp.mean_sea_floor_radius
)
ice_thickness_95_ranges = np.array(
    # [1, 10, 25, 50, 100, 200, 300, 400, 500],
    [25, 100, 250, 500],
)  # in meters
# net_ice_thickness_changes = np.linspace(-200, 200, 20)  # in meters
net_ice_thickness_changes = np.linspace(-200, 200, 4)  # in meters


load_data = pd.DataFrame(
    columns=[
        "odt_length_scale",
        "odt_amplitude_95_range",
        "ice_length_scale",
        "ice_thickness_95_range",
        "net_ice_thickness_change",
        "ice_load_measure",
        "ocean_dynamic_load_measure",
        "ocean_dynamic_measure",
    ],
)


# make joblib parallel for each combination of inputs
def generate_measures(
    odt_length_scale: float,
    odt_amplitude_95_range: float,
    ice_length_scale: float,
    ice_thickness_95_range: float,
    net_ice_thickness_change: float,
):
    _, ice_load_measure = ice_thickness_change_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=ice_length_scale,
        thickness_95_range=ice_thickness_95_range,
        net_thickness_change=net_ice_thickness_change,
    )
    (
        ocean_dynamic_measure,
        ocean_dynamic_load_measure,
    ) = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=odt_length_scale,
        amplitude_95_range=odt_amplitude_95_range,
    )
    load_data_row = {
        "odt_length_scale": odt_length_scale,
        "odt_amplitude_95_range": odt_amplitude_95_range,
        "ice_length_scale": ice_length_scale,
        "ice_thickness_95_range": ice_thickness_95_range,
        "net_ice_thickness_change": net_ice_thickness_change,
        "ice_load_measure": ice_load_measure,
        "ocean_dynamic_load_measure": ocean_dynamic_load_measure,
        "ocean_dynamic_measure": ocean_dynamic_measure,
    }
    return load_data_row


print(
    "number of combinations:",
    len(odt_length_scales)
    * len(odt_amplitude_95_ranges)
    * len(ice_length_scales)
    * len(ice_thickness_95_ranges)
    * len(net_ice_thickness_changes),
)

results = Parallel(n_jobs=-1, verbose=4)(
    delayed(generate_measures)(
        odt_length_scale,
        odt_amplitude_95_range,
        ice_length_scale,
        ice_thickness_95_range,
        net_ice_thickness_change,
    )
    for odt_length_scale in odt_length_scales
    for odt_amplitude_95_range in odt_amplitude_95_ranges
    for ice_length_scale in ice_length_scales
    for ice_thickness_95_range in ice_thickness_95_ranges
    for net_ice_thickness_change in net_ice_thickness_changes
)

data = pd.DataFrame(results)


## generate the direct loads for each row and add to the dataframe using joblib parallel
def compute_direct_loads(row):
    direct_load = load_measure(
        ice_thickness_load_measure=row["ice_load_measure"],
        odt_load_measure=row["ocean_dynamic_load_measure"],
    )
    return direct_load


data["direct_load_measure"] = Parallel(n_jobs=-1, verbose=4)(
    delayed(compute_direct_loads)(row) for _, row in data.iterrows()
)


# multiple job libs for gmsl true and ssh estimates that all run at once
def gmsl_via_slc(row):
    gmsl_true = gmsl_measure(
        measure=sea_level_change_measure(
            fingerprint_operator=fingerprint_operator,
            fingerprint=fp,
            load_measure=row["direct_load_measure"],
        ),
        fingerprint=fp,
    )
    expectation, variance = get_stats_from_measure(gmsl_true)
    return expectation, np.sqrt(variance)


def gmsl_via_ssh(row):
    ssh, ssh_odt, _ = sea_surface_height_measure(
        fingerprint_operator=fingerprint_operator,
        fingerprint=fp,
        load_measure=row["direct_load_measure"],
        ocean_dynamic_measure=row["ocean_dynamic_measure"],
    )
    gmsl_ssh_estimate = gmsl_measure(
        measure=ssh,
        fingerprint=fp,
    )
    gmsl_ssh_odt_estimate = gmsl_measure(
        measure=ssh_odt,
        fingerprint=fp,
    )
    ssh_expectation, ssh_variance = get_stats_from_measure(
        gmsl_ssh_estimate,
    )
    ssh_odt_expectation, ssh_odt_variance = get_stats_from_measure(
        gmsl_ssh_odt_estimate,
    )
    return (
        ssh_expectation,
        np.sqrt(ssh_variance),
        ssh_odt_expectation,
        np.sqrt(ssh_odt_variance),
    )


# select the input variables into new dataframe
gmsl_data = data[
    [
        "odt_length_scale",
        "odt_amplitude_95_range",
        "ice_length_scale",
        "ice_thickness_95_range",
        "net_ice_thickness_change",
    ]
].copy()

# calc stats
gmsl_data[["true_mean", "true_std"]] = Parallel(n_jobs=-1, verbose=4)(
    delayed(gmsl_via_slc)(row) for _, row in data.iterrows()
)
gmsl_data[["ssh_mean", "ssh_std", "ssh_odt_mean", "ssh_odt_std"]] = (
    Parallel(n_jobs=-1, verbose=4)(
        delayed(gmsl_via_ssh)(row) for _, row in data.iterrows()
    )
)

# dump data to file as csv
gmsl_data.to_csv("gmsl_data.csv", index=False)
