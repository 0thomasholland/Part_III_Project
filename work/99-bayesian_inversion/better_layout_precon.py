import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    CGMatrixSolver,
    GaussianMeasure,
    LinearBayesianInversion,
    LinearForwardProblem,
    LinearOperator,
    RowLinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)


def setup_altimetry_inversion_components(
    lmax: int,
    # Model space parameters
    model_sobolev_order: float = 2.0,
    model_length_scale_factor: float = 0.1,
    # Ice prior parameters
    ice_length_scale_factor: float = 0.1,
    ice_gmsl_target_std_m: float = 0.01,
    ice_net_thickness_change_m: float = 0.0,
    # ODT prior parameters (two-scale model)
    meso_scale_lengthscale_m: float = 250e3,
    meso_scale_std_m: float = 0.001,
    sub_meso_scale_lengthscale_m: float = 2e3,
    sub_meso_scale_std_m: float = 0.001,
    # Altimetry parameters
    altimetry_latitude_max: float = 66.0,
    altimetry_latitude_min: float = -66.0,
    altimetry_noise_sobolev_order: float = 1.5,
    altimetry_noise_length_scale_factor: float = 0.001,
    along_track_error_m: float = 0.005,
    passes: int = 3,
):
    """Set up all components for Bayesian inversion of ice thickness change
    from satellite altimetry data at a given truncation degree.

    Parameters
    ----------
    lmax : int
        Maximum spherical harmonic degree.
    model_sobolev_order : float
        Sobolev order for model space.
    model_length_scale_factor : float
        Model length scale as fraction of mean sea floor radius.
    ice_length_scale_factor : float
        Ice prior length scale as fraction of mean sea floor radius.
    ice_gmsl_target_std_m : float
        Target standard deviation for ice GMSL contribution (metres).
    ice_net_thickness_change_m : float
        Net ice thickness change (metres).
    meso_scale_lengthscale_m : float
        Meso-scale ODT length scale (metres).
    meso_scale_std_m : float
        Meso-scale ODT standard deviation (metres).
    sub_meso_scale_lengthscale_m : float
        Sub-meso-scale ODT length scale (metres).
    sub_meso_scale_std_m : float
        Sub-meso-scale ODT standard deviation (metres).
    altimetry_latitude_max : float
        Maximum latitude for altimetry coverage (degrees).
    altimetry_latitude_min : float
        Minimum latitude for altimetry coverage (degrees).
    altimetry_noise_sobolev_order : float
        Sobolev order for altimetry noise model.
    altimetry_noise_length_scale_factor : float
        Noise length scale as fraction of mean sea floor radius.
    along_track_error_m : float
        Along-track measurement error (metres).
    passes : int
        Number of satellite passes (typically ~3 per 30 days).

    Returns
    -------
    dict
        Dictionary containing all necessary components for inversion.

    """
    # Initialize fingerprint model
    fp = FingerPrint(
        lmax=lmax,
        earth_model_parameters=FingerPrint.from_standard_non_dimensionalisation(),
    )
    fp.set_state_from_ice_ng()

    # Convert physical units to non-dimensional
    model_length_scale = (
        model_length_scale_factor * fp.mean_sea_floor_radius
    )
    ice_length_scale = (
        ice_length_scale_factor * fp.mean_sea_floor_radius
    )
    ice_gmsl_target_std = ice_gmsl_target_std_m / fp.length_scale
    ice_net_thickness_change = (
        ice_net_thickness_change_m / fp.length_scale
    )
    meso_scale_lengthscale = (
        meso_scale_lengthscale_m / fp.length_scale
    )
    sub_meso_scale_lengthscale = (
        sub_meso_scale_lengthscale_m / fp.length_scale
    )
    meso_scale_std = meso_scale_std_m / fp.length_scale
    sub_meso_scale_std = sub_meso_scale_std_m / fp.length_scale
    altimetry_noise_length_scale = (
        altimetry_noise_length_scale_factor * fp.mean_sea_floor_radius
    )
    altimetry_noise_std = (
        along_track_error_m / np.sqrt(passes)
    ) / fp.length_scale

    # Define model space
    model_space = Sobolev(
        fp.lmax,
        model_sobolev_order,
        model_length_scale,
        radius=fp.mean_sea_floor_radius,
    )

    # Core operators
    fingerprint_op = fp.as_sobolev_linear_operator(
        model_sobolev_order,
        model_length_scale,
    )
    load_space = fingerprint_op.domain
    response_space = fingerprint_op.codomain

    sea_surface_height_op = sea_surface_height_operator(
        fp,
        response_space,
    )
    measurement_space = sea_surface_height_op.codomain

    # Load conversion operators
    water_to_load_op = sea_level_change_to_load_operator(
        fp,
        load_space,
    )
    ice_to_load_op = ice_thickness_change_to_load_operator(
        fp,
        load_space,
    )

    # Altimetry spatial mask
    altimetry_mask_op = spatial_mutliplication_operator(
        fp.altimetry_projection(
            latitude_max=altimetry_latitude_max,
            latitude_min=altimetry_latitude_min,
            value=0,
        )
        * fp.ocean_function,
        measurement_space,
    )

    # Error/noise model operator
    altimetry_noise_op = altimetry_mask_op @ RowLinearOperator(
        [
            sea_surface_height_op @ fingerprint_op @ water_to_load_op,
            measurement_space.identity_operator(),
            measurement_space.identity_operator(),
        ],
    )

    # Signal forward operator: ice thickness → altimetry observations
    ice_to_altimetry_op = (
        altimetry_mask_op
        @ sea_surface_height_op
        @ fingerprint_op
        @ ice_to_load_op
        @ ice_projection_operator(fp, load_space)
    )

    # Prior measures
    ice_thickness_change_measure, _ = ice_thickness_change_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_op,
        length_scale=ice_length_scale,
        ice_gmsl_target_std=ice_gmsl_target_std,
        net_thickness_change=ice_net_thickness_change,
    )

    # Two-scale ocean dynamic topography model
    meso_scale_measure, _ = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_op,
        length_scale=meso_scale_lengthscale,
        standard_deviation=meso_scale_std,
    )
    sub_meso_scale_measure, _ = ocean_dynamic_topography_measures(
        fingerprint=fp,
        fingerprint_operator=fingerprint_op,
        length_scale=sub_meso_scale_lengthscale,
        standard_deviation=sub_meso_scale_std,
    )
    odt_change_measure = meso_scale_measure + sub_meso_scale_measure

    measurement_noise_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        altimetry_noise_sobolev_order,
        altimetry_noise_length_scale,
        altimetry_noise_std,
    )

    # Combine error sources
    error_input_measure = GaussianMeasure.from_direct_sum(
        [
            odt_change_measure,
            odt_change_measure,
            measurement_noise_measure,
        ],
    )
    data_error_measure = error_input_measure.affine_mapping(
        operator=altimetry_noise_op,
    )

    # Construct forward problem
    forward_problem = LinearForwardProblem(
        ice_to_altimetry_op,
        data_error_measure=data_error_measure,
    )

    return {
        "fp": fp,
        "forward_problem": forward_problem,
        "ice_thickness_change_measure": ice_thickness_change_measure,
        "model_space": model_space,
        "measurement_space": measurement_space,
        "load_space": load_space,
        "response_space": response_space,
        "ice_to_altimetry_op": ice_to_altimetry_op,
        "sea_surface_height_op": sea_surface_height_op,
        "fingerprint_op": fingerprint_op,
        "ice_to_load_op": ice_to_load_op,
        "odt_change_measure": odt_change_measure,
    }


def build_truncated_spectral_preconditioner(
    components: dict,
    lmax_exact: int = 32,
    high_degree_scaling: float = 1.0,
):
    """Build preconditioner that's exact for low spherical harmonic degrees
    and uses simple scaling for high degrees.

    Parameters
    ----------
    components : dict
        Output from setup_altimetry_inversion_components.
    lmax_exact : int
        Maximum degree for exact treatment. Degrees 0 to lmax_exact
        will have the exact normal operator inverted.
    high_degree_scaling : float
        Scaling factor applied to high-degree coefficients.
        A value of 1.0 means identity (no preconditioning for high degrees).

    Returns
    -------
    LinearOperator
        Preconditioner operator on the data space.

    """
    inversion = LinearBayesianInversion(
        components["forward_problem"],
        components["ice_thickness_change_measure"],
    )

    normal_op = inversion.normal_operator
    data_space = normal_op.domain
    lmax_full = components["fp"].lmax

    if lmax_exact >= lmax_full:
        raise ValueError(
            f"lmax_exact ({lmax_exact}) must be less than full lmax ({lmax_full})",
        )

    # Number of coefficients up to degree l is (l+1)^2
    n_low = (lmax_exact + 1) ** 2
    n_full = (lmax_full + 1) ** 2

    # Build the low-degree block of the normal operator matrix
    print(
        f"Building {n_low}x{n_low} low-degree block of normal operator...",
    )
    print(
        f"This requires {n_low} applications of the normal operator.",
    )

    N_low_matrix = np.zeros((n_low, n_low))

    for i in range(n_low):
        if (i + 1) % 100 == 0 or i == n_low - 1:
            print(f"  Progress: {i + 1}/{n_low}")

        e_i = np.zeros(n_full)
        e_i[i] = 1.0
        vec = data_space.from_components(e_i)
        result = data_space.to_components(normal_op(vec))
        N_low_matrix[:, i] = result[:n_low]

    # Invert the low-degree block
    print("Inverting low-degree block...")
    N_low_inv = np.linalg.inv(N_low_matrix)
    print("Preconditioner ready.")

    def preconditioner_mapping(x):
        cx = data_space.to_components(x)
        result = np.zeros_like(cx)

        # Apply exact inverse for low degrees
        result[:n_low] = N_low_inv @ cx[:n_low]

        # Apply simple scaling for high degrees
        result[n_low:] = high_degree_scaling * cx[n_low:]

        return data_space.from_components(result)

    def preconditioner_adjoint_mapping(x):
        # For symmetric normal operator, adjoint is the same
        cx = data_space.to_components(x)
        result = np.zeros_like(cx)

        result[:n_low] = N_low_inv.T @ cx[:n_low]
        result[n_low:] = high_degree_scaling * cx[n_low:]

        return data_space.from_components(result)

    return LinearOperator(
        data_space,
        data_space,
        preconditioner_mapping,
        adjoint_mapping=preconditioner_adjoint_mapping,
    )


class ConvergenceMonitor:
    """Monitor convergence of iterative solver with detailed tracking."""

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

            if self.iteration % 5 == 0:
                print(
                    f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}, "
                    f"relative change = {relative_change:.6e}",
                )
        elif self.iteration % 5 == 0:
            print(f"Iteration {self.iteration}: ||x|| = {x_norm:.6e}")

        self.prev_x = xk.copy()

    def plot_convergence(self):
        """Plot convergence history."""
        fig = plt.figure(figsize=(10, 4))

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(
            range(1, self.iteration + 1),
            self.x_norms,
            marker="o",
            markersize=3,
        )
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Norm of solution ||x||")
        ax1.set_title("Convergence of Solution Norm")
        ax1.grid(True, alpha=0.3)

        if self.x_changes:
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.semilogy(
                range(2, self.iteration + 1),
                self.x_changes,
                marker="o",
                markersize=3,
            )
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Relative change")
            ax2.set_title("Convergence Rate")
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


# %%
# Main inversion workflow

# Common parameters
inversion_params = {
    "model_sobolev_order": 2.0,
    "model_length_scale_factor": 0.1,
    "ice_length_scale_factor": 0.1,
    "ice_gmsl_target_std_m": 0.01,
    "ice_net_thickness_change_m": 0.0,
    "meso_scale_lengthscale_m": 250e3,
    "meso_scale_std_m": 0.001,
    "sub_meso_scale_lengthscale_m": 2e3,
    "sub_meso_scale_std_m": 0.001,
    "altimetry_latitude_max": 66.0,
    "altimetry_latitude_min": -66.0,
    "altimetry_noise_sobolev_order": 1.5,
    "altimetry_noise_length_scale_factor": 0.001,
    "along_track_error_m": 0.005,
    "passes": 3,
}

# Solver parameters
solver_rtol = 1e-6
solver_maxiter = 1000

# Set up components
print("Setting up inversion components (lmax=128)...")
components = setup_altimetry_inversion_components(
    lmax=128,
    **inversion_params,
)

fp = components["fp"]
forward_problem = components["forward_problem"]
ice_thickness_change_measure = components[
    "ice_thickness_change_measure"
]

# Build truncated spectral preconditioner
print("\nBuilding truncated spectral preconditioner...")
preconditioner = build_truncated_spectral_preconditioner(
    components,
    lmax_exact=32,  # Exact for degrees 0-32, that's 1089 basis vectors
    high_degree_scaling=1.0,  # Identity for high degrees
)

# Generate synthetic data
print("\nGenerating synthetic model and data...")
model_prior, data = forward_problem.synthetic_model_and_data(
    ice_thickness_change_measure,
)

# Set up Bayesian inversion
inversion = LinearBayesianInversion(
    forward_problem,
    ice_thickness_change_measure,
)

# Solve with convergence monitoring
monitor = ConvergenceMonitor()
solver = CGMatrixSolver(
    callback=monitor,
    rtol=solver_rtol,
    maxiter=solver_maxiter,
)

print("\nSolving for posterior measure...")
model_posterior_measure = inversion.model_posterior_measure(
    data,
    solver,
    preconditioner=preconditioner,
)

print(f"\nTotal CG iterations: {monitor.iteration}")
print(
    f"Number of fingerprint problem solutions: {fp.solver_counter}",
)

# Plot convergence
monitor.plot_convergence()

# Extract posterior expectation
model_posterior_expectation = model_posterior_measure.expectation

# Visualize results
fig, ax, im = plot(
    model_posterior_expectation
    * fp.length_scale
    * fp.ice_projection(),
    symmetric=True,
)
fig.colorbar(
    im,
    ax=ax,
    label="Posterior expectation (m)",
    orientation="horizontal",
)
plt.title("Ice Thickness Change - Posterior Expectation")
plt.show()
