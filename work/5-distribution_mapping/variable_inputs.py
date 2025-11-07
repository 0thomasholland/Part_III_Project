# %%
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

verbosity = 1
# %%
print("Starting variable input GMSL estimation script")

# --- Set up a fingerprint instance ---
lmax = 128
fp = sl.FingerPrint(
    lmax=lmax,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)
fp.set_state_from_ice_ng()

fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)


odt_length_scales = (
    # np.linspace(0.01, 0.5, 10) * fp.mean_sea_floor_radius / fp.length_scale
    np.linspace(0.01, 0.5, 2)
    * fp.mean_sea_floor_radius
    / fp.length_scale
)
odt_amplitude_95_ranges = (
    np.array(
        # [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 10],
        [0.01, 0.1, 10],
    )
    / fp.length_scale
)  # in units of sea level, non-dimensionalized

ice_length_scales = (
    # np.linspace(0.1, 0.5, 10) * fp.mean_sea_floor_radius / fp.length_scale
    np.linspace(0.1, 0.5, 2)
    * fp.mean_sea_floor_radius
    / fp.length_scale
)
ice_thickness_95_ranges = (
    np.array(
        # [1, 10, 25, 50, 100, 200, 300, 400, 500],
        [100, 500],
    )
    / fp.length_scale
)  # in meters, non-dimensionalized
# net_ice_thickness_changes = np.linspace(-200, 200, 20) / fp.length_scale  # in meters, non-dimensionalized
net_ice_thickness_changes = (
    np.linspace(-200, 200, 2) / fp.length_scale
)  # in meters, non-dimensionalized


odt_length_scale = []
odt_amplitude_95_range = []
ice_length_scale = []
ice_thickness_95_range = []
net_ice_thickness_change = []
ice_load_measure = []
ocean_dynamic_load_measure = []
ocean_dynamic_measure = []
# %%
print("Generating measures for variable inputs...")


# make joblib parallel for each combination of inputs
def return_odt_measure(
    fingerprint,
    fingerprint_operator,
    length_scale,
    amplitude_95_range,
):
    odt_measure, odt_load_measure = ocean_dynamic_topography_measures(
        fingerprint=fingerprint,
        fingerprint_operator=fingerprint_operator,
        length_scale=length_scale,
        amplitude_95_range=amplitude_95_range,
    )
    return (
        odt_measure,
        odt_load_measure,
        length_scale,
        amplitude_95_range,
    )


def return_ice_measure(
    fingerprint,
    fingerprint_operator,
    length_scale,
    thickness_95_range,
    net_thickness_change,
):
    _, ice_load_measure = ice_thickness_change_measures(
        fingerprint=fingerprint,
        fingerprint_operator=fingerprint_operator,
        length_scale=length_scale,
        thickness_95_range=thickness_95_range,
        net_thickness_change=net_thickness_change,
    )
    return (
        length_scale,
        thickness_95_range,
        net_thickness_change,
        ice_load_measure,
    )


print(
    "number of combinations:",
    len(odt_length_scales) * len(odt_amplitude_95_ranges)
    + len(ice_length_scales)
    * len(ice_thickness_95_ranges)
    * len(net_ice_thickness_changes),
)
# %%
ice_results = Parallel(n_jobs=-1, verbose=verbosity)(
    delayed(return_ice_measure)(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=ice_length_scale,
        thickness_95_range=ice_thickness_95_range,
        net_thickness_change=net_ice_thickness_change,
    )
    for ice_length_scale in ice_length_scales
    for ice_thickness_95_range in ice_thickness_95_ranges
    for net_ice_thickness_change in net_ice_thickness_changes
)
odt_results = Parallel(n_jobs=-1, verbose=verbosity)(
    delayed(return_odt_measure)(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=odt_length_scale,
        amplitude_95_range=odt_amplitude_95_range,
    )
    for odt_length_scale in odt_length_scales
    for odt_amplitude_95_range in odt_amplitude_95_ranges
)
for res in ice_results:
    (
        ice_length_scale_out,
        ice_thickness_95_range_out,
        ice_net_thickness_change_out,
        ice_load_measure_out,
    ) = res
    for odt_res in odt_results:
        (
            odt_measure_out,
            odt_load_measure_out,
            odt_length_scale_out,
            odt_amplitude_95_range_out,
        ) = odt_res
        odt_length_scale.append(odt_length_scale_out)
        odt_amplitude_95_range.append(odt_amplitude_95_range_out)
        ice_length_scale.append(ice_length_scale_out)
        ice_thickness_95_range.append(ice_thickness_95_range_out)
        net_ice_thickness_change.append(ice_net_thickness_change_out)
        ice_load_measure.append(ice_load_measure_out)
        ocean_dynamic_load_measure.append(odt_load_measure_out)
        ocean_dynamic_measure.append(odt_measure_out)

# %%
## generate the direct loads for each combination of inputs and add to list

direct_loads = []

print("Computing direct loads for each combination of inputs...")
print("Total combinations:", len(ice_load_measure))


def compute_direct_loads(
    ice_load_measure,
    ocean_dynamic_load_measure,
):
    direct_load = load_measure(
        ice_thickness_load_measure=ice_load_measure,
        odt_load_measure=ocean_dynamic_load_measure,
    )
    return direct_load


direct_loads = Parallel(n_jobs=-1, verbose=verbosity)(
    delayed(compute_direct_loads)(
        ice_load_measure[i],
        ocean_dynamic_load_measure[i],
    )
    for i in range(len(ice_load_measure))
)
# %%
print("Computed all direct loads, now computing responses metrics...")


def gmsl_via_slc(fingerprint, fingerprint_operator, direct_load):
    slc = sea_level_change_measure(
        fingerprint_operator=fingerprint_operator,
        fingerprint=fingerprint,
        load_measure=direct_load,
    )
    gmsl_slc_estimate = gmsl_measure(
        measure=slc,
        fingerprint=fingerprint,
    )
    slc_expectation, slc_variance = get_stats_from_measure(
        gmsl_slc_estimate,
        fingerprint=fingerprint,
    )
    return slc_expectation, np.sqrt(slc_variance)


def gmsl_via_ssh(
    fingerprint,
    fingerprint_operator,
    direct_load,
    odt_measure,
):
    ssh, ssh_odt, _ = sea_surface_height_measure(
        fingerprint_operator=fingerprint_operator,
        fingerprint=fingerprint,
        load_measure=direct_load,
        odt_measure=odt_measure,
    )
    gmsl_ssh_estimate = gmsl_measure(
        measure=ssh,
        fingerprint=fingerprint,
    )
    gmsl_ssh_odt_estimate = gmsl_measure(
        measure=ssh_odt,
        fingerprint=fingerprint,
    )
    ssh_expectation, ssh_variance = get_stats_from_measure(
        gmsl_ssh_estimate,
        fingerprint=fingerprint,
    )
    ssh_odt_expectation, ssh_odt_variance = get_stats_from_measure(
        gmsl_ssh_odt_estimate,
        fingerprint=fingerprint,
    )
    return (
        ssh_expectation,
        np.sqrt(ssh_variance),
        ssh_odt_expectation,
        np.sqrt(ssh_odt_variance),
    )


gmsl_slc_expectation = []
gmsl_slc_std = []
gmsl_ssh_expectation = []
gmsl_ssh_std = []
gmsl_ssh_odt_expectation = []
gmsl_ssh_odt_std = []

# use joblib to parallelize the computation of gmsl estimates adding ssh and slc to same joblist
print("Number of jobs: ", len(direct_loads))

results_gmsl = Parallel(
    n_jobs=-1,
    verbose=verbosity,
    idle_worker_timeout=300,
)(
    delayed(
        lambda i: (
            *gmsl_via_slc(
                fingerprint=fp,
                fingerprint_operator=fingerprint_operator,
                direct_load=direct_loads[i],
            ),
            *gmsl_via_ssh(
                fingerprint=fp,
                fingerprint_operator=fingerprint_operator,
                direct_load=direct_loads[i],
                odt_measure=ocean_dynamic_measure[i],
            ),
        ),
    )(i)
    for i in range(len(direct_loads))
)
for res in results_gmsl:
    gmsl_slc_expectation.append(res[0])
    gmsl_slc_std.append(res[1])
    gmsl_ssh_expectation.append(res[2])
    gmsl_ssh_std.append(res[3])
    gmsl_ssh_odt_expectation.append(res[4])
    gmsl_ssh_odt_std.append(res[5])
# %%
print("Computed all GMSL estimates, now saving results...")

# --- Save results to a dataframe ---
df = pd.DataFrame(
    {
        "odt_length_scale /": odt_length_scale * fp.length_scale,
        "odt_amplitude_95_range /": odt_amplitude_95_range
        * fp.length_scale,
        "ice_length_scale /": ice_length_scale * fp.length_scale,
        "ice_thickness_95_range /": ice_thickness_95_range
        * fp.length_scale,
        "net_ice_thickness_change /": net_ice_thickness_change
        * fp.length_scale,
        "gmsl_slc_expectation /": gmsl_slc_expectation
        * fp.length_scale,
        "gmsl_slc_std /": gmsl_slc_std * fp.length_scale,
        "gmsl_ssh_expectation /": gmsl_ssh_expectation
        * fp.length_scale,
        "gmsl_ssh_std /": gmsl_ssh_std * fp.length_scale,
        "gmsl_ssh_odt_expectation /": gmsl_ssh_odt_expectation
        * fp.length_scale,
        "gmsl_ssh_odt_std /": gmsl_ssh_odt_std * fp.length_scale,
        "lmax /": lmax * np.ones(len(gmsl_slc_expectation)),
    },
)

df.to_csv(
    "work/5-distribution_mapping/variable_input_gmsl_estimates_low_res.csv",
    index=False,
)

print("Saved results to csv")
