# %%
import matplotlib as mpl
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

mpl.rcParams["figure.dpi"] = 600

# %%
# Setup
lmax = 128
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()
fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)

load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain
sea_surface_height_op = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space = sea_surface_height_op.codomain
# %%
# Parameters
ice_length_scale = np.array([0.05, 0.2, 0.5, 0.7]) * fp.mean_sea_floor_radius
ice_gmsl_target_std = (
    np.array(
        [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
    )
    / fp.length_scale
)
net_ice_thickness_change = (
    np.array(
        [
            0.0,
            150.0,
            100.0,
            75.0,
            50.0,
            25.0,
            10.0,
            5.0,
        ],
    )
    / fp.length_scale
)

odt_length_scale = np.array([0.01, 0.001, 0.1]) * fp.mean_sea_floor_radius
odt_standard_deviation = np.array([0.08, 0.016, 0.008, 0.0008]) / fp.length_scale

altimetry_range = np.array(
    [90, 85, 80, 75, 70, 66, 60, 55, 50],
)  # in degrees
altimetry_error_length_scale = (
    np.array(
        [
            0.05,
            0.005,
            0.0005,
        ],
    )
    * fp.mean_sea_floor_radius
)
altimetry_error_amplitude = np.array([0.03, 0.003, 0.0003]) / fp.length_scale
# %%


def get_data(
    ice_length_scale=0.1 * fp.mean_sea_floor_radius,
    ice_gmsl_target_std=0.005 / fp.length_scale,
    net_ice_thickness_change=0,
    odt_length_scale=0.01 * fp.mean_sea_floor_radius,
    odt_standard_deviation=0.08 / fp.length_scale,
    altimetry_error_length_scale=0.005 * fp.mean_sea_floor_radius,
    altimetry_error_amplitude=0.003 / fp.length_scale,
    altimetry_range=66,
):
    # Measures
    ice_thickness_change, _ = ice_thickness_change_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=ice_length_scale,
        ice_gmsl_target_std=ice_gmsl_target_std,
        net_thickness_change=net_ice_thickness_change,
    )

    odt_change, _ = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_operator,
        length_scale=odt_length_scale,
        standard_deviation=odt_standard_deviation,
    )

    measurement_error = (
        measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
            1.5,
            altimetry_error_length_scale,
            altimetry_error_amplitude,
        )
    )

    # Operators
    GMSL_from_ice_op = averaging_operator(
        load_space,
        [
            -fp.ice_density
            * fp.one_minus_ocean_function
            * fp.ice_projection(value=0)
            * fp.length_scale
            / (fp.water_density * fp.ocean_area),
        ],
    )

    altimetry_weight = fp.ocean_projection(
        value=0,
    ) * fp.altimetry_projection(
        latitude_min=-altimetry_range,
        latitude_max=altimetry_range,
        value=0,
    )
    Altimetry_op = averaging_operator(
        measurement_space,
        [altimetry_weight / fp.integrate(altimetry_weight)],
    )

    Load_w_op = sea_level_change_to_load_operator(fp, load_space)
    Load_i_op = ice_thickness_change_to_load_operator(fp, load_space)
    Fingerprint_ssh_op = sea_surface_height_op @ fingerprint_operator

    # Compute distributions using affine mappings
    true_gmsl = ice_thickness_change.affine_mapping(
        operator=GMSL_from_ice_op,
    )

    estimated_gmsl = (
        ice_thickness_change.affine_mapping(
            operator=Altimetry_op @ Fingerprint_ssh_op @ Load_i_op,
        )
        + odt_change.affine_mapping(
            operator=Altimetry_op @ Fingerprint_ssh_op @ Load_w_op,
        )
        + odt_change.affine_mapping(operator=Altimetry_op)
        + measurement_error.affine_mapping(operator=Altimetry_op)
    )

    error = estimated_gmsl - true_gmsl

    # Extract statistics (convert to meters)
    true_mean = true_gmsl.expectation[0] * fp.length_scale
    true_std = np.sqrt(true_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale

    est_mean = estimated_gmsl.expectation[0] * fp.length_scale
    est_std = (
        np.sqrt(estimated_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale
    )

    error_mean = error.expectation[0] * fp.length_scale
    error_std = np.sqrt(error.covariance.matrix(dense=True)[0, 0]) * fp.length_scale

    # return results and then input parameters
    return {
        "true_mean": true_mean,
        "true_std": true_std,
        "est_mean": est_mean,
        "est_std": est_std,
        "error_mean": error_mean,
        "error_std": error_std,
        "ice_length_scale": ice_length_scale,
        "ice_gmsl_target_std": ice_gmsl_target_std,
        "net_ice_thickness_change": net_ice_thickness_change,
        "odt_length_scale": odt_length_scale,
        "odt_standard_deviation": odt_standard_deviation,
        "altimetry_error_length_scale": altimetry_error_length_scale,
        "altimetry_error_amplitude": altimetry_error_amplitude,
        "altimetry_range": altimetry_range,
    }


# %%

print("Number of simulations to run:")
print(
    len(ice_gmsl_target_std)
    * len(ice_length_scale)
    * len(net_ice_thickness_change)
    * len(odt_length_scale)
    * len(odt_standard_deviation)
    * len(altimetry_error_length_scale)
    * len(altimetry_error_amplitude)
    * len(altimetry_range),
)

# %%

results = Parallel(n_jobs=-1, verbose=4)(
    delayed(get_data)(
        ice_length_scale=ils,
        ice_gmsl_target_std=igts,
        net_ice_thickness_change=nitc,
        odt_length_scale=olts,
        odt_standard_deviation=ods,
        altimetry_error_length_scale=aels,
        altimetry_error_amplitude=aea,
        altimetry_range=ar,
    )
    for ils in ice_length_scale
    for igts in ice_gmsl_target_std
    for nitc in net_ice_thickness_change
    for olts in odt_length_scale
    for ods in odt_standard_deviation
    for aels in altimetry_error_length_scale
    for aea in altimetry_error_amplitude
    for ar in altimetry_range
)

# %%
# reshape results into a dataframe and save as csv

# old_dataframe = pd.read_csv(
# "gmsl_error_with_measurement_noise_results_lmax64.csv",
# )

dataframe = pd.DataFrame(results)

# dataframe = pd.concat([old_dataframe, dataframe], ignore_index=True)


# save dataframe to csv
dataframe.to_csv(
    "gmsl_error_with_measurement_noise_results_lmax128.csv",
    index=False,
)


# %%
