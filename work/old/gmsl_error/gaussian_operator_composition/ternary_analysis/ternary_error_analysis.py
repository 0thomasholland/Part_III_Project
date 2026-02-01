# %%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pyslfp import (
    FingerPrint,
    averaging_operator,
    ice_thickness_change_to_load_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

# %%
greenland = np.linspace(0, 1, 12)
west_antarctica = np.linspace(0, 1, 12)

G, W = np.meshgrid(greenland, west_antarctica)

E = 1 - G - W

mask = E >= 0
G_valid = G[mask]
W_valid = W[mask]
E_valid = E[mask]

# Create load_tuples from valid ternary coordinates
load_tuples = list(zip(G_valid, W_valid, E_valid))

# %%
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
ice_length_scale = 0.2 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.01 / fp.length_scale
net_ice_thickness_change = -10.0

odt_length_scale = 0.1 * fp.mean_sea_floor_radius
odt_std = 0.05 / fp.length_scale

altimetry_range = 66

altimetry_error_length_scale = 0.05 * fp.mean_sea_floor_radius
altimetry_error_std = 0.002 / fp.length_scale

# %%

ice_thickness_change, _ = ice_thickness_change_measures(
    fp,
    fingerprint_operator,
    ice_length_scale,
    ice_gmsl_target_std,
    net_ice_thickness_change,
)

odt_change, _ = ocean_dynamic_topography_measures(
    fp,
    fingerprint_operator,
    odt_length_scale,
    odt_std,
)

measurement_error = (
    measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        1.5,
        altimetry_error_length_scale,
        altimetry_error_std,
    )
)

# %%

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


def compute_error(load_tuple: tuple):
    greenland_operator = spatial_mutliplication_operator(
        fp.greenland_projection(value=0),
        load_space,
    )
    west_antarctica_operator = spatial_mutliplication_operator(
        fp.west_antarctic_projection(value=0),
        load_space,
    )
    east_antarctica_operator = spatial_mutliplication_operator(
        fp.east_antarctic_projection(value=0),
        load_space,
    )
    _G, _W, _E = load_tuple

    G_ice_thickness_change, _ = ice_thickness_change_measures(
        fp,
        fingerprint_operator,
        ice_length_scale,
        ice_gmsl_target_std * _G,
        net_ice_thickness_change,
    )
    G_ice_thickness_change.affine_mapping(operator=greenland_operator)

    W_ice_thickness_change, _ = ice_thickness_change_measures(
        fp,
        fingerprint_operator,
        ice_length_scale,
        ice_gmsl_target_std * _W,
        net_ice_thickness_change,
    )
    W_ice_thickness_change.affine_mapping(
        operator=west_antarctica_operator,
    )

    E_ice_thickness_change, _ = ice_thickness_change_measures(
        fp,
        fingerprint_operator,
        ice_length_scale,
        ice_gmsl_target_std * _E,
        net_ice_thickness_change,
    )
    E_ice_thickness_change.affine_mapping(
        operator=east_antarctica_operator,
    )

    total_ice_thickness_change = (
        G_ice_thickness_change + W_ice_thickness_change + E_ice_thickness_change
    )

    true_gmsl = total_ice_thickness_change.affine_mapping(
        operator=GMSL_from_ice_op,
    )

    estimated_gmsl = (
        total_ice_thickness_change.affine_mapping(
            operator=Altimetry_op @ Fingerprint_ssh_op @ Load_i_op,
        )
        + odt_change.affine_mapping(
            operator=Altimetry_op @ Fingerprint_ssh_op @ Load_w_op,
        )
        + odt_change.affine_mapping(operator=Altimetry_op)
        + measurement_error.affine_mapping(operator=Altimetry_op)
    )
    error = estimated_gmsl - true_gmsl

    true_mean = true_gmsl.expectation[0] * fp.length_scale
    true_std = np.sqrt(true_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale
    est_mean = estimated_gmsl.expectation[0] * fp.length_scale
    est_std = (
        np.sqrt(estimated_gmsl.covariance.matrix(dense=True)[0, 0]) * fp.length_scale
    )
    error_mean = error.expectation[0] * fp.length_scale
    error_std = np.sqrt(error.covariance.matrix(dense=True)[0, 0]) * fp.length_scale
    error_mean = error.expectation[0] * fp.length_scale
    error_std = np.sqrt(error.covariance.matrix(dense=True)[0, 0]) * fp.length_scale
    return {
        "true_mean": true_mean,
        "true_std": true_std,
        "est_mean": est_mean,
        "est_std": est_std,
        "error_mean": error_mean,
        "error_std": error_std,
        "load_tuple": load_tuple,
    }


# %%
print("Number of simulations to run:")
print(len(load_tuples))
results = Parallel(n_jobs=-1, verbose=4)(
    delayed(compute_error)(load_tuple) for load_tuple in load_tuples
)

# %%

dataframe = pd.DataFrame(results)
dataframe[["G", "W", "E"]] = pd.DataFrame(
    dataframe["load_tuple"].tolist(),
    index=dataframe.index,
)
dataframe = dataframe.drop(columns=["load_tuple"])
dataframe.to_csv(
    "ternary_error_analysis_w_shift_hr_alt.csv",
    index=False,
)

# %%
