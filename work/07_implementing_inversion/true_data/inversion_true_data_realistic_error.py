# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    plot_1d_distributions,
    plot_corner_distributions,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
    plot,
)
from tqdm import tqdm

from project import (
    ice_thickness_to_slc_operator,
)
from project.operators import (
    ice_thickness_to_ssh_point_estimations_operator,
)
from pyslfp_extras.helpers import (
    get_ocean_point_coordinates,
)
from pyslfp_extras.measures import (
    ice_thickness_gaussian_measure,
    odt_gaussian_measure,
)
from pyslfp_extras.operators import (
    ocean_point_evaluation_operator,
)

# ---------------------------------------------------------------------------
# Configuration — adjust these as needed
# ---------------------------------------------------------------------------
DUACS_PATH = Path("../../../data/duacs/duacs_annual.nc")
YEAR_START = 1993
YEAR_END = 2019
ALTIMETRY_DEGREE_SPACING = 2.0

# %%
# ---------------------------------------------------------------------------
# FingerPrint and Sobolev operator
# ---------------------------------------------------------------------------
fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%
# ---------------------------------------------------------------------------
# Prior measure for ice thickness
# ---------------------------------------------------------------------------
ice_thickness_measure: GaussianMeasure = (
    ice_thickness_gaussian_measure(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=0.1 * fp.mean_sea_floor_radius,
        gmsl_target_std=0.01,
        gmsl_target_mean=0.07,
    )
)

# %%
# ---------------------------------------------------------------------------
# Forward operator and ocean point coordinates
# ---------------------------------------------------------------------------
ice_thickness_to_ssh_point_estimations_op: LinearOperator = ice_thickness_to_ssh_point_estimations_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
    altimetry_latitude_range=66.0,
    point_degree_spacing=ALTIMETRY_DEGREE_SPACING,
)

points: tuple[list[float], list[float]] = (
    get_ocean_point_coordinates(
        finger_print=fp,
        point_degree_spacing=ALTIMETRY_DEGREE_SPACING,
        altimetry_latitude_range=66.0,
    )
)

print(
    f"Number of ocean evaluation points: {len(points[0])}"
)

# %%
# ---------------------------------------------------------------------------
# Realistic error model (spatially varying ODT error)
# ---------------------------------------------------------------------------
error_field_measure: GaussianMeasure = odt_gaussian_measure(
    finger_print=fp,
    finger_print_operator=fp_op,
    use_spatial_variability=True,
    amplitude=0.003,
    point_multiplier=20,
)

plot(
    error_field_measure.sample() * 1000,
    symmetric=True,
)

# %%

data_error_measure = error_field_measure.affine_mapping(
    operator=ocean_point_evaluation_operator(
        finger_print=fp,
        measurement_space=error_field_measure.domain,
        point_degree_spacing=ALTIMETRY_DEGREE_SPACING,
        altimetry_latitude_range=66.0,
    )
)


# %%
# ---------------------------------------------------------------------------
# Load DUACS data and extract SLA difference at ocean points
# ---------------------------------------------------------------------------
ds = xr.open_dataset(DUACS_PATH)

sla_start = ds["sla"].sel(
    time=f"{YEAR_START}-01-01", method="nearest"
)
sla_end = ds["sla"].sel(
    time=f"{YEAR_END}-01-01", method="nearest"
)
sla_diff = sla_end - sla_start  # metres

print(f"SLA difference: {YEAR_END} minus {YEAR_START}")

# Extract SLA difference at each ocean point
n_points = len(points[0])
data_array = np.zeros(n_points)
nan_count = 0

for i in range(n_points):
    lat = points[0][i]
    lon = points[1][i]

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
            longitude=slice(lon_duacs - 1, lon_duacs + 1),
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

# Convert to data-space vector
data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)
data = data_space.from_components(data_array)

# %%
# ---------------------------------------------------------------------------
# Diagnostic plot: observed SLA difference at ocean points
# ---------------------------------------------------------------------------
fig_obs, ax_obs = plt.subplots(
    1,
    1,
    figsize=(12, 6),
    subplot_kw={"projection": ccrs.Robinson()},
)

plot_lons = [
    lon - 360.0 if lon > 180.0 else lon for lon in points[1]
]

sc = ax_obs.scatter(
    plot_lons,
    points[0],
    c=data_array * 1000,
    cmap="RdBu_r",
    s=10,
    vmin=-np.max(np.abs(data_array)) * 1000,
    vmax=np.max(np.abs(data_array)) * 1000,
    transform=ccrs.PlateCarree(),
)
ax_obs.coastlines()
ax_obs.set_global()
cb = plt.colorbar(
    sc,
    ax=ax_obs,
    orientation="horizontal",
    pad=0.05,
    shrink=0.7,
)
cb.set_label(f"SLA Difference {YEAR_END}–{YEAR_START} (mm)")
ax_obs.set_title(
    f"DUACS Observed SLA Change: {YEAR_END} minus {YEAR_START}"
)

# %%
# ---------------------------------------------------------------------------
# Forward problem and Bayesian inversion
# ---------------------------------------------------------------------------
forward_problem = LinearForwardProblem(
    ice_thickness_to_ssh_point_estimations_op,
    data_error_measure=data_error_measure,
)

bayesian_inversion = LinearBayesianInversion(
    forward_problem, ice_thickness_measure
)

print("Starting inversion...")
residuals = []
pbar = tqdm(desc="CG solve")


def progress_callback(xk):
    residuals.append(np.linalg.norm(xk))
    pbar.set_postfix({"||x||": f"{residuals[-1]:.2e}"})
    pbar.update(1)


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

# %%
# ---------------------------------------------------------------------------
# Plot: Posterior ice thickness change
# ---------------------------------------------------------------------------
max_abs_ice_change = (
    np.nanmax(
        np.abs(model_posterior_expectation.data.flatten())
    )
    * 1000
    * fp.length_scale
)

fig_ice, ax_ice, im_ice = plot(
    1000
    * model_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax_ice.set_title(
    f"Posterior Ice Thickness Change ({YEAR_END} minus {YEAR_START})"
)

# %%
# ---------------------------------------------------------------------------
# Plot: Posterior sea level change
# ---------------------------------------------------------------------------
ice_thickness_to_slc_op = ice_thickness_to_slc_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
)

sea_level_posterior = ice_thickness_to_slc_op(
    model_posterior_expectation
)

# convert to mm/yr
sea_level_posterior = sea_level_posterior / (
    YEAR_END - YEAR_START
)

ocean_mask = fp.ocean_projection()
max_abs_sl_change = (
    np.nanmax(
        np.abs(
            (
                sea_level_posterior * ocean_mask
            ).data.flatten()
        )
    )
    * 1000
    * fp.length_scale
)

fig_sl, ax_sl, im_sl = plot(
    1000
    * sea_level_posterior
    * ocean_mask
    * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm / yr)",
)
ax_sl.set_title(
    f"Posterior Sea-Level Fingerprint ({YEAR_END} minus {YEAR_START})"
)

# %%
# plot at 160W to 0W and between 55N to 90N
fig_zoom, ax_zoom, im_zoom = plot(
    1000
    * sea_level_posterior
    * ocean_mask
    * fp.length_scale,
    coasts=True,
    cmap="seismic",
    symmetric=True,
    map_extent=[-120, 0, 55, 90],
    colorbar_label="Sea Level Change (mm/yr)",
    projection=ccrs.NorthPolarStereo(),
)
ax_zoom.set_title(
    f"Posterior Sea-Level Fingerprint (Zoomed: Greenland Region)"
)

# %%
# ---------------------------------------------------------------------------
# GMSL posterior PDF
# ---------------------------------------------------------------------------
model_space = ice_thickness_measure.domain

GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

B = averaging_operator(
    model_space, [GMSL_weighting_function]
)

GMSL_prior_measure = ice_thickness_measure.affine_mapping(
    operator=B
)
GMSL_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=B)
)

fig_gmsl, ax_gmsl = plot_1d_distributions(
    GMSL_posterior_measure,
    prior_measures=GMSL_prior_measure,
    xlabel="GMSL Change (mm)",
    title=f"GMSL Inference from DUACS Altimetry ({YEAR_END} minus {YEAR_START})",
)

# %%
# ---------------------------------------------------------------------------
# Corner plot: ice sheet contributions
# ---------------------------------------------------------------------------
GLI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.greenland_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
WAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.west_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)
EAI_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.east_antarctic_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

C = averaging_operator(
    model_space,
    [
        GLI_weighting_function,
        WAI_weighting_function,
        EAI_weighting_function,
    ],
)

property_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=C)
)

plot_corner_distributions(
    property_posterior_measure,
    labels=[
        "Greenland Contribution (mm)",
        "West Antarctica Contribution (mm)",
        "East Antarctica Contribution (mm)",
    ],
    title=f"Joint Posterior: Ice Sheet GMSL Contributions ({YEAR_END} minus {YEAR_START})",
)
