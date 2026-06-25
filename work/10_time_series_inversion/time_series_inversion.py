# %%
from pyslfp.linear_operators import (
    FingerPrintOperator,
    l2_products_operator,
)
from pyslfp.state import EarthState
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
)
from tqdm import tqdm

from project.operators import (
    ice_thickness_to_ssh_point_estimations_operator,
)
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

# %%

DUACS_PATH = Path("../../data/duacs/duacs_annual.nc")
YEAR_START = 1993
YEAR_END = 2020
ALTIMETRY_DEGREE_SPACING = 3.0

fp = EarthState.from_defaults(lmax=128)

fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

# %%
ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.model.parameters.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.015,  # 15 mm of uncertainty in GMSL change from ice melt over the year
    gmsl_target_mean=0.003,  # 3 mm of GMSL change from ice melt over the year
)
ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness_measure
)

ice_thickness_to_ssh_point_estimations_op: LinearOperator = ice_thickness_to_ssh_point_estimations_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
    altimetry_latitude_range=66.0,
    point_degree_spacing=ALTIMETRY_DEGREE_SPACING,
)

grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=ALTIMETRY_DEGREE_SPACING,
    latitude_range=66.0,
)

altimetry_error_std = 0.1

data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)

error_sampling_points = (
    GaussianMeasure.from_standard_deviation(
        data_space, altimetry_error_std
    )
)

#### SPATIAL ERROR (disabled)
# odt_error = OceanDynamics(
#     finger_print=fp,
#     finger_print_operator=fp_op,
#     std=0.0003,
#     pattern=OceanDynamics.SyntheticPattern(
#         point_multiplier=30.0,
#     ),
# )
# error_field_measure: GaussianMeasure = (
#     odt_error.load_measure
# )

# error_sampling_points += error_field_measure.affine_mapping(
#     operator=GridPoints.ocean_altimetry(
#         fp,
#         degree_spacing=ALTIMETRY_DEGREE_SPACING,
#         latitude_range=66.0,
#     ).point_evaluation_operator(error_field_measure.domain)
# )

# %%

GIS_weighting_function = (
    -fp.model.parameters.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * 1000
    * fp.model.parameters.length_scale
    / (fp.model.parameters.water_density * fp.ocean_area)
)
WAIS_weighting_function = (
    -fp.model.parameters.ice_density
    * fp.one_minus_ocean_function
    * fp.west_antarctic_projection(value=0)
    * 1000
    * fp.model.parameters.length_scale
    / (fp.model.parameters.water_density * fp.ocean_area)
)
EAIS_weighting_function = (
    -fp.model.parameters.ice_density
    * fp.one_minus_ocean_function
    * fp.east_antarctic_projection(value=0)
    * 1000
    * fp.model.parameters.length_scale
    / (fp.model.parameters.water_density * fp.ocean_area)
)
GLOBAL_weighting_function = (
    -fp.model.parameters.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.model.parameters.length_scale
    / (fp.model.parameters.water_density * fp.ocean_area)
)
model_space = ice_thickness_measure.domain
C = l2_products_operator(
    model_space,
    [
        GIS_weighting_function,
        WAIS_weighting_function,
        EAIS_weighting_function,
    ],
)

D = l2_products_operator(
    model_space,
    [
        GLOBAL_weighting_function,
    ],
)

def progress_callback(xk):
    residuals.append(np.linalg.norm(xk))
    pbar.set_postfix({"||x||": f"{residuals[-1]:.2e}"})
    pbar.update(1)

# %%

output_data = {}

for year in range(YEAR_START, YEAR_END + 1):
    ds = xr.open_dataset(DUACS_PATH)

    sla_start = ds["sla"].sel(
        time=f"{year}-01-01", method="nearest"
    )
    sla_end = ds["sla"].sel(
        time=f"{year + 1}-01-01", method="nearest"
    )
    sla_diff = sla_end - sla_start  # metres

    print(f"SLA difference: {year + 1} minus {year}")

    # Extract SLA difference at each ocean point
    n_points = len(grid_points)
    data_array = np.zeros(n_points)
    nan_count = 0

    for i in range(n_points):
        lat = grid_points.lats[i]
        lon = grid_points.lons[i]

        # Convert longitude from [0, 360] to [-180, 180] for DUACS grid
        lon_duacs = lon - 360.0 if lon > 180.0 else lon

        val = float(
            sla_diff.sel(
                latitude=lat,
                longitude=lon_duacs,
                method="nearest",
            )
        )

        if np.isnan(val):
            # Local fallback: ±1° box around the point
            box = sla_diff.sel(
                latitude=slice(lat - 1, lat + 1),
                longitude=slice(
                    lon_duacs - 1, lon_duacs + 1
                ),
            )
            box_vals = box.values[~np.isnan(box.values)]
            if len(box_vals) > 0:
                val = float(np.mean(box_vals))
            else:
                # Wider fallback: ±2° box
                box = sla_diff.sel(
                    latitude=slice(lat - 2, lat + 2),
                    longitude=slice(
                        lon_duacs - 2, lon_duacs + 2
                    ),
                )
                box_vals = box.values[~np.isnan(box.values)]
                val = (
                    float(np.mean(box_vals))
                    if len(box_vals) > 0
                    else 0.0
                )
            nan_count += 1

        data_array[i] = val

    ds.close()

    print(
        f"Extracted {n_points} data points ({nan_count} required NaN fallback)"
    )
    print(
        f"SLA difference range: {data_array.min() * 1000:.1f} to {data_array.max() * 1000:.1f} mm"
    )
    print(
        f"Mean SLA difference: {data_array.mean() * 1000:.1f} mm"
    )
    data_space = (
        ice_thickness_to_ssh_point_estimations_op.codomain
    )
    data = data_space.from_components(data_array)

    forward_problem = LinearForwardProblem(
        ice_thickness_to_ssh_point_estimations_op,
        data_error_measure=error_sampling_points,
    )

    bayesian_inversion = LinearBayesianInversion(
        forward_problem, ice_thickness_measure
    )

    print("Starting inversion...")
    residuals = []
    pbar = tqdm(desc="CG solve")

    model_posterior_measure = (
        bayesian_inversion.model_posterior_measure(
            data,
            CGMatrixSolver(
                callback=progress_callback,
                rtol=1e-3,
                maxiter=300,
            ),
        )
    )
    pbar.close()
    print("Inversion complete.")

    model_posterior_expectation = (
        model_posterior_measure.expectation
    )

    property_posterior_measure = (
        model_posterior_measure.affine_mapping(operator=C)
    )

    global_property_posterior_measure = (
        model_posterior_measure.affine_mapping(operator=D)
    )

    posterior_mean = property_posterior_measure.expectation
    posterior_covariance = (
        property_posterior_measure.covariance.matrix(
            dense=True
        )
    )

    global_posterior_mean = (
        global_property_posterior_measure.expectation[0]
    )
    global_posterior_covariance = (
        global_property_posterior_measure.covariance.matrix(
            dense=True
        )[0, 0]
    )
    _data = {
        "GIS_mean": posterior_mean[0],
        "WAIS_mean": posterior_mean[1],
        "EAIS_mean": posterior_mean[2],
        "GIS_marginal_cov": posterior_covariance[0, 0],
        "WAIS_marginal_cov": posterior_covariance[1, 1],
        "EAIS_marginal_cov": posterior_covariance[2, 2],
        "global_mean": global_posterior_mean,
        "global_cov": global_posterior_covariance,
    }
    output_data[f"{year}-{year + 1}"] = _data

# %%

print(output_data)

# %%

output_df = pd.DataFrame(output_data).T

print(output_df)
output_df.to_csv("time_series_inversion_results.csv")
