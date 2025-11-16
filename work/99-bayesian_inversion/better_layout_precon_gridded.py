"""Gridded Altimetry Inversion with Spatial Averaging

This performs Bayesian inversion of ice thickness change from
satellite altimetry data, where the data is spatially averaged over
grid cells within the altimetry coverage region.
"""

import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    CGMatrixSolver,
    EigenSolver,
    GaussianMeasure,
    HilbertSpace,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    RowLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    EarthModelParameters,
    FingerPrint,
    averaging_operator,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
)

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)


def generate_grid_cell_weighting_functions(
    fp: FingerPrint,
    grid_spacing_deg: float = 1.0,
    latitude_max: float = 66.0,
    latitude_min: float = -66.0,
) -> list:
    """Generate weighting functions for spatial averaging over grid cells.

    Each weighting function is the altimetry projection restricted to a single
    grid cell, normalized by its area so that the averaging operator returns
    the mean value over that cell.
    """
    # Generate grid cell boundaries
    lat_edges = np.arange(
        latitude_min,
        latitude_max + grid_spacing_deg,
        grid_spacing_deg,
    )
    lon_edges = np.arange(
        -180.0,
        180.0 + grid_spacing_deg,
        grid_spacing_deg,
    )

    weighting_functions = []
    grid_centers = []

    print("Generating grid cell weighting functions...")
    print(f"  Latitude range: [{latitude_min}, {latitude_max}]")
    print(f"  Grid spacing: {grid_spacing_deg}°")

    n_lat_cells = len(lat_edges) - 1
    n_lon_cells = len(lon_edges) - 1
    print(
        f"  Potential cells: {n_lat_cells} x {n_lon_cells} = {n_lat_cells * n_lon_cells}",
    )

    # Get grid coordinates (2D arrays)
    lats_1d = fp.lats()
    lons_1d = fp.lons()
    lats_grid, lons_grid = np.meshgrid(
        lats_1d,
        lons_1d,
        indexing="ij",
    )

    for i in range(len(lat_edges) - 1):
        lat_min_cell = lat_edges[i]
        lat_max_cell = lat_edges[i + 1]
        lat_center = (lat_min_cell + lat_max_cell) / 2.0

        for j in range(len(lon_edges) - 1):
            lon_min_cell = lon_edges[j]
            lon_max_cell = lon_edges[j + 1]
            lon_center = (lon_min_cell + lon_max_cell) / 2.0

            # Create indicator function for this grid cell
            # altimetry_projection handles ocean masking and latitude bounds
            cell_lat_band = fp.altimetry_projection(
                latitude_max=lat_max_cell,
                latitude_min=lat_min_cell,
                value=0,  # Zero outside the latitude band and ocean
            )

            # Create longitude mask
            if lon_min_cell < lon_max_cell:
                lon_mask_data = np.where(
                    (lons_grid >= lon_min_cell)
                    & (lons_grid < lon_max_cell),
                    1.0,
                    0.0,
                )
            else:  # Wrapping around 180/-180
                lon_mask_data = np.where(
                    (lons_grid >= lon_min_cell)
                    | (lons_grid < lon_max_cell),
                    1.0,
                    0.0,
                )

            # Create SHGrid from longitude mask
            lon_mask_field = fp.zero_field()
            lon_mask_field.data = lon_mask_data

            # Combine: cell = lat_band (includes ocean) * lon_band
            cell_indicator = cell_lat_band * lon_mask_field

            # Check if cell has any ocean coverage
            cell_area = fp.integrate(cell_indicator)

            if cell_area > 1e-10:  # Cell has ocean coverage
                # Normalize by area to get averaging weight
                # The averaging operator computes: integral(f * w)
                # We want: integral(f * indicator) / integral(indicator)
                # So weight = indicator / integral(indicator)
                weighting_function = cell_indicator / cell_area
                weighting_functions.append(weighting_function)
                grid_centers.append((lat_center, lon_center))

    print(
        f"  Generated {len(weighting_functions)} valid grid cells (with ocean coverage)",
    )

    return weighting_functions, grid_centers


def setup_gridded_altimetry_inversion(
    lmax: int,
    grid_spacing_deg: float = 1.0,
    # Model space parameters
    model_sobolev_order: float = 2.0,
    model_length_scale_km: float = 250.0,
    # Ice prior parameters
    ice_pointwise_std_m: float = 0.1,
    # ODT prior parameters (two-scale model)
    meso_scale_lengthscale_m: float = 250e3,
    meso_scale_std_m: float = 0.001,
    sub_meso_scale_lengthscale_m: float = 2e3,
    sub_meso_scale_std_m: float = 0.001,
    # Altimetry parameters
    altimetry_latitude_max: float = 66.0,
    altimetry_latitude_min: float = -66.0,
    along_track_error_m: float = 0.005,
    passes: int = 3,
):
    """Set up gridded altimetry inversion with spatial averaging.

    This approach computes the spatial average of sea surface height over
    each grid cell within the altimetry coverage region.
    Returns dictionary containing all inversion components.

    """
    # Initialize fingerprint model
    fp = FingerPrint(
        lmax=lmax,
        earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # Convert to non-dimensional units
    model_length_scale = (
        model_length_scale_km * 1000 / fp.length_scale
    )
    ice_pointwise_std = ice_pointwise_std_m / fp.length_scale
    meso_scale_lengthscale = (
        meso_scale_lengthscale_m / fp.length_scale
    )
    sub_meso_scale_lengthscale = (
        sub_meso_scale_lengthscale_m / fp.length_scale
    )
    meso_scale_std = meso_scale_std_m / fp.length_scale
    sub_meso_scale_std = sub_meso_scale_std_m / fp.length_scale
    measurement_noise_std = (
        along_track_error_m / np.sqrt(passes)
    ) / fp.length_scale

    # Generate grid cell weighting functions
    weighting_functions, grid_centers = (
        generate_grid_cell_weighting_functions(
            fp,
            grid_spacing_deg=grid_spacing_deg,
            latitude_max=altimetry_latitude_max,
            latitude_min=altimetry_latitude_min,
        )
    )

    # Define model space
    model_space = Sobolev(
        fp.lmax,
        model_sobolev_order,
        model_length_scale,
        radius=fp.mean_sea_floor_radius,
    )

    # Build forward operator chain
    # 1. Ice projection: non-zero only over ice sheets
    op_ice_proj = ice_projection_operator(fp, model_space)

    # 2. Ice thickness to load conversion
    op_ice_to_load = ice_thickness_change_to_load_operator(
        fp,
        model_space,
    )

    # 3. Fingerprint operator: load -> full response
    op_fingerprint = fp.as_sobolev_linear_operator(
        model_sobolev_order,
        model_length_scale,
        rtol=1e-9,
    )
    response_space = op_fingerprint.codomain

    # 4. Extract sea surface height from full response
    op_ssh = sea_surface_height_operator(fp, response_space)
    ssh_space = op_ssh.codomain

    # 5. Spatial averaging over grid cells
    op_average = averaging_operator(ssh_space, weighting_functions)
    data_space = op_average.codomain

    # Compose full forward operator
    forward_op = (
        op_average
        @ op_ssh
        @ op_fingerprint
        @ op_ice_to_load
        @ op_ice_proj
    )

    # Also create operator for sea level field (for visualization)
    sea_level_field_op = (
        ssh_space.subspace_projection(0)
        @ op_ssh
        @ op_fingerprint
        @ op_ice_to_load
        @ op_ice_proj
    )

    # Build error model
    # ODT contributes through gravitational and direct effects
    meso_measure, _ = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=op_fingerprint,
        length_scale=meso_scale_lengthscale,
        standard_deviation=meso_scale_std,
    )
    sub_meso_measure, _ = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=op_fingerprint,
        length_scale=sub_meso_scale_lengthscale,
        standard_deviation=sub_meso_scale_std,
    )
    odt_measure = meso_measure + sub_meso_measure

    # Water load operator for ODT gravitational effect
    op_water_to_load = sea_level_change_to_load_operator(
        fp,
        op_fingerprint.domain,
    )

    # ODT error operator components (now with averaging)
    # Gravitational effect: ODT -> load -> fingerprint -> SSH -> average
    odt_grav_op = (
        op_average @ op_ssh @ op_fingerprint @ op_water_to_load
    )
    # Direct effect: ODT -> SSH -> average
    odt_direct_op = op_average @ ssh_space.identity_operator()

    # Measurement noise (independent at each grid cell)
    measurement_noise_measure = (
        GaussianMeasure.from_standard_deviation(
            data_space,
            measurement_noise_std,
        )
    )

    # Combined error model
    error_input_measure = GaussianMeasure.from_direct_sum(
        [
            odt_measure,  # ODT for gravitational effect
            odt_measure,  # ODT for direct effect
            measurement_noise_measure,  # Instrument noise
        ],
    )

    # Error operator maps error sources to data space
    error_op = RowLinearOperator(
        [
            odt_grav_op,
            odt_direct_op,
            data_space.identity_operator(),
        ],
    )

    data_error_measure = error_input_measure.affine_mapping(
        operator=error_op,
    )

    # Ice thickness change prior
    initial_prior = (
        model_space.point_value_scaled_heat_kernel_gaussian_measure(
            model_length_scale,
            ice_pointwise_std,
        )
    )
    ice_thickness_change_measure = initial_prior.affine_mapping(
        operator=op_ice_proj,
    )

    # Construct forward problem
    forward_problem = LinearForwardProblem(
        forward_op,
        data_error_measure=data_error_measure,
    )

    return {
        "fp": fp,
        "grid_centers": grid_centers,
        "weighting_functions": weighting_functions,
        "forward_problem": forward_problem,
        "ice_thickness_change_measure": ice_thickness_change_measure,
        "model_space": model_space,
        "data_space": data_space,
        "forward_op": forward_op,
        "sea_level_field_op": sea_level_field_op,
        "op_ice_proj": op_ice_proj,
        "op_average": op_average,
    }


def build_low_degree_preconditioner(
    lmax_precon: int,
    weighting_functions: list,
    model_sobolev_order: float,
    model_length_scale_km: float,
    ice_pointwise_std_m: float,
    measurement_noise_std_m: float,
):
    """Build preconditioner using lower truncation degree. Returns inverse of the approximate normal operator for preconditioning."""
    print(f"Building preconditioner at lmax={lmax_precon}...")

    # Set up reduced problem
    fp_precon = FingerPrint(
        lmax=lmax_precon,
        earth_model_parameters=EarthModelParameters.from_standard_non_dimensionalisation(),
    )
    fp_precon.set_state_from_ice_ng()

    # Convert units
    model_length_scale = (
        model_length_scale_km * 1000 / fp_precon.length_scale
    )
    ice_pointwise_std = ice_pointwise_std_m / fp_precon.length_scale
    measurement_noise_std = (
        measurement_noise_std_m / fp_precon.length_scale
    )

    # Build operators at reduced degree
    model_space_precon = Sobolev(
        fp_precon.lmax,
        model_sobolev_order,
        model_length_scale,
        radius=fp_precon.mean_sea_floor_radius,
    )

    op1 = ice_projection_operator(fp_precon, model_space_precon)
    op2 = ice_thickness_change_to_load_operator(
        fp_precon,
        model_space_precon,
    )
    op3 = fp_precon.as_sobolev_linear_operator(
        model_sobolev_order,
        model_length_scale,
        rtol=1e-9,
    )
    op4 = sea_surface_height_operator(fp_precon, op3.codomain)

    # Use same weighting functions for averaging
    # Note: weighting functions are defined on the high-res grid,
    # but averaging_operator should handle this
    op5 = averaging_operator(op4.codomain, weighting_functions)

    forward_op_precon = op5 @ op4 @ op3 @ op2 @ op1
    data_space = forward_op_precon.codomain

    # Simple error model for preconditioner (just measurement noise)
    data_error_measure_precon = (
        GaussianMeasure.from_standard_deviation(
            data_space,
            measurement_noise_std,
        )
    )

    # Prior at reduced degree
    initial_prior_precon = model_space_precon.point_value_scaled_heat_kernel_gaussian_measure(
        model_length_scale,
        ice_pointwise_std,
    )
    ice_measure_precon = initial_prior_precon.affine_mapping(
        operator=op1,
    )

    # Forward problem at reduced degree
    forward_problem_precon = LinearForwardProblem(
        forward_op_precon,
        data_error_measure=data_error_measure_precon,
    )

    # Bayesian inversion
    inversion_precon = LinearBayesianInversion(
        forward_problem_precon,
        ice_measure_precon,
    )

    # Get normal operator and invert
    normal_op_precon = inversion_precon.normal_operator

    print(
        "Computing eigen-decomposition of preconditioner normal operator...",
    )
    solver = EigenSolver(parallel=True)
    inverse_normal_op = solver(normal_op_precon)
    print("Preconditioner ready.")

    return inverse_normal_op


class ConvergenceMonitor:
    """Monitor convergence of iterative solver."""

    def __init__(self):
        self.iteration = 0
        self.x_norms = []
        self.x_changes = []
        self.prev_x = None

    def __call__(self, xk):
        self.iteration += 1
        x_norm = np.linalg.norm(xk)
        self.x_norms.append(x_norm)

        if self.prev_x is not None:
            change = np.linalg.norm(xk - self.prev_x)
            relative_change = (
                change / x_norm if x_norm > 0 else change
            )
            self.x_changes.append(relative_change)

            if self.iteration % 10 == 0:
                print(
                    f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}, "
                    f"relative change = {relative_change:.6e}",
                )
        elif self.iteration % 10 == 0:
            print(f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}")

        self.prev_x = xk.copy()


# %%
# Main inversion workflow

# Configuration
lmax = 128
lmax_precon = 32
grid_spacing_deg = 2.0  # Start with coarser grid for testing

# Model parameters
model_sobolev_order = 2.0
model_length_scale_km = 250.0
ice_pointwise_std_m = 0.1

# ODT parameters
meso_scale_lengthscale_m = 250e3
meso_scale_std_m = 0.001
sub_meso_scale_lengthscale_m = 2e3
sub_meso_scale_std_m = 0.001

# Altimetry parameters
altimetry_latitude_max = 66.0
altimetry_latitude_min = -66.0
along_track_error_m = 0.005
passes = 3
measurement_noise_std_m = along_track_error_m / np.sqrt(passes)

# Solver parameters
solver_rtol = 1e-6
solver_maxiter = 500

# Set up main problem
print(f"Setting up gridded altimetry inversion (lmax={lmax})...")
components = setup_gridded_altimetry_inversion(
    lmax=lmax,
    grid_spacing_deg=grid_spacing_deg,
    model_sobolev_order=model_sobolev_order,
    model_length_scale_km=model_length_scale_km,
    ice_pointwise_std_m=ice_pointwise_std_m,
    meso_scale_lengthscale_m=meso_scale_lengthscale_m,
    meso_scale_std_m=meso_scale_std_m,
    sub_meso_scale_lengthscale_m=sub_meso_scale_lengthscale_m,
    sub_meso_scale_std_m=sub_meso_scale_std_m,
    altimetry_latitude_max=altimetry_latitude_max,
    altimetry_latitude_min=altimetry_latitude_min,
    along_track_error_m=along_track_error_m,
    passes=passes,
)

fp = components["fp"]
grid_centers = components["grid_centers"]
weighting_functions = components["weighting_functions"]
forward_problem = components["forward_problem"]
ice_thickness_change_measure = components[
    "ice_thickness_change_measure"
]
sea_level_field_op = components["sea_level_field_op"]

# Build preconditioner
print(f"\nBuilding preconditioner (lmax={lmax_precon})...")
preconditioner = build_low_degree_preconditioner(
    lmax_precon=lmax_precon,
    weighting_functions=weighting_functions,
    model_sobolev_order=model_sobolev_order,
    model_length_scale_km=model_length_scale_km,
    ice_pointwise_std_m=ice_pointwise_std_m,
    measurement_noise_std_m=measurement_noise_std_m,
)

# Generate synthetic data
print("\nGenerating synthetic model and data...")
model_true, data = forward_problem.synthetic_model_and_data(
    ice_thickness_change_measure,
)

# Set up Bayesian inversion
print("\nSetting up Bayesian inversion...")
inversion = LinearBayesianInversion(
    forward_problem,
    ice_thickness_change_measure,
)

# Solve
print("\nSolving for posterior measure...")
monitor = ConvergenceMonitor()
solver = CGMatrixSolver(
    callback=monitor,
    rtol=solver_rtol,
    maxiter=solver_maxiter,
)

model_posterior_measure = inversion.model_posterior_measure(
    data,
    solver,
    preconditioner=preconditioner,
)

print(f"\nTotal CG iterations: {monitor.iteration}")
print(
    f"Number of fingerprint problem solutions: {fp.solver_counter}",
)

# Extract results
model_posterior_expectation = model_posterior_measure.expectation

# %%
# Visualization

# Calculate shared color scale for ice thickness
max_abs_ice = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    model_true.data.flatten(),
                    model_posterior_expectation.data.flatten(),
                ],
            ),
        ),
    )
    * 1000
    * fp.length_scale
)

# True ice thickness change
fig1, ax1, im1 = plot(
    1000 * model_true * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice,
    vmax=max_abs_ice,
)
ax1.set_title("True Ice Thickness Change")
fig1.colorbar(
    im1,
    ax=ax1,
    label="Ice Thickness Change (mm)",
    orientation="horizontal",
)

# Posterior expectation
fig2, ax2, im2 = plot(
    1000 * model_posterior_expectation * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice,
    vmax=max_abs_ice,
)
ax2.set_title("Posterior Expectation (Inferred)")
fig2.colorbar(
    im2,
    ax=ax2,
    label="Ice Thickness Change (mm)",
    orientation="horizontal",
)

# Difference
fig3, ax3, im3 = plot(
    1000
    * (model_posterior_expectation - model_true)
    * fp.length_scale,
    coasts=True,
    cmap="seismic",
    symmetric=True,
)
ax3.set_title("Difference (Posterior - True)")
fig3.colorbar(
    im3,
    ax=ax3,
    label="Difference (mm)",
    orientation="horizontal",
)

# Sea level fingerprints
sea_level_true = sea_level_field_op(model_true)
sea_level_posterior = sea_level_field_op(
    model_posterior_expectation,
)
ocean_mask = fp.ocean_projection()

max_abs_sl = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (sea_level_true * ocean_mask).data.flatten(),
                    (sea_level_posterior * ocean_mask).data.flatten(),
                ],
            ),
        ),
    )
    * 1000
    * fp.length_scale
)

fig4, ax4, im4 = plot(
    1000 * sea_level_true * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl,
    vmax=max_abs_sl,
)
ax4.set_title("True Sea Level Fingerprint")
fig4.colorbar(
    im4,
    ax=ax4,
    label="Sea Level Change (mm)",
    orientation="horizontal",
)

fig5, ax5, im5 = plot(
    1000 * sea_level_posterior * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl,
    vmax=max_abs_sl,
)
ax5.set_title("Posterior Sea Level Fingerprint")
fig5.colorbar(
    im5,
    ax=ax5,
    label="Sea Level Change (mm)",
    orientation="horizontal",
)

plt.show()
