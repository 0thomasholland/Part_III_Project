# %%
import cartopy.crs as ccrs
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
)
from tqdm import tqdm

from project import (
    error_plot,
    ice_thickness_to_slc_operator,
)
from project.operators import (
    ice_thickness_to_point_estimated_gmsl_operator,
    ice_thickness_to_ssh_point_estimations_operator,
)
from pygeoinf_extras import standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.gmsl import (
    altimetry_gmsl,
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)
from pyslfp_extras.plotting import plot

sns.set_theme("paper")
sns.set_style("ticks")
palette = sns.color_palette("deep")
blue = palette[0]
orange = palette[1]

fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# %%

# generate prior dataset

altimetry_degree_density = 5.0

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.1 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.ThicknessWeightedPattern(),
    ice_gmsl_std=0.01,  # gmsl std = 5mm
    point_degree_spacing=altimetry_degree_density,
)
ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness
)


ice_thickness_to_ssh_point_estimations_op: LinearOperator = (
    ice_change.load_to_ssh_point_estimations_operator
    @ ice_change.ice_thickness_to_load_operator
)

grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=altimetry_degree_density,
    latitude_range=66.0,
)


plot(ice_thickness_measure.sample(), symmetric=True)

# %%

data_space = (
    ice_thickness_to_ssh_point_estimations_op.codomain
)

# %%

altimetry_std_dev = 0.001
data_error_measure = (
    GaussianMeasure.from_standard_deviation(
        data_space, altimetry_std_dev
    )
)

forward_problem = LinearForwardProblem(
    ice_thickness_to_ssh_point_estimations_op,
    data_error_measure=data_error_measure,
)

model_true, data = forward_problem.synthetic_model_and_data(
    ice_thickness_measure
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
        data, CGMatrixSolver(callback=progress_callback)
    )
)
pbar.close()
print("")
print("Inversion complete.")
model_posterior_expectation = (
    model_posterior_measure.expectation
)

# %%

max_abs_ice_change = (
    np.nanmax(
        np.abs(
            np.concatenate(
                [
                    (model_true).data.flatten(),
                    (
                        model_posterior_expectation
                    ).data.flatten(),
                ]
            )
        )
    )
    * 1000
    * fp.length_scale
)

# --- Plot 1: The "Ground Trutsh" Model ---
fig1, ax1, im1 = plot(
    1000
    * model_true
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_ice_change,
    vmax=max_abs_ice_change,
    colorbar_label="Ice Thickness Change (mm)",
)
ax1.set_title("a) True Ice Thickness Change")

# --- Plot 2: The Posterior Expectation (Our Best Estimate) ---
fig2, ax2, im2 = plot(
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
ax2.set_title(
    "b) Posterior Expectation (Inferred from Data)"
)

fig2_1, ax2_1, im2_1 = plot(
    1000
    * np.abs(model_true - model_posterior_expectation)
    * fp.length_scale
    * fp.ice_projection(),
    coasts=True,
    cmap=cc.cm.CET_L17,
    colorbar_label="error (mm)",
)
ax2_1.set_title("Difference")


# %%

ice_thickness_to_slc_op = (
    ice_change.load_to_slc_operator
    @ ice_change.ice_thickness_to_load_operator
)

sea_level_posterior = ice_thickness_to_slc_op(
    model_posterior_expectation
)

sea_level_true = ice_thickness_to_slc_op(model_true)

ocean_mask = fp.ocean_projection()
# max_abs_sl_change = (
#     np.nanmax(
#         np.abs(
#             np.concatenate(
#                 [
#                     (
#                         sea_level_true * ocean_mask
#                     ).data.flatten(),
#                     (
#                         sea_level_posterior * ocean_mask
#                     ).data.flatten(),
#                 ]
#             )
#         )
#     )
#     * 1000
# )


# --- Plot 3: The "True" Sea-Level Field ---
fig3, ax3, im13 = plot(
    1000 * sea_level_true * ocean_mask * fp.length_scale,
    coasts=True,
    cmap="seismic",
    # vmin=-max_abs_sl_change,
    # vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax3.set_title("a) True Sea-Level Fingerprint")
# ax3.plot(
#     points[1],  # longitudes
#     points[0],  # latitudes
#     "kx",
#     label="Altimetry Point Estimations",
#     transform=ccrs.PlateCarree(),
# )


# --- Plot 4: The Sea-Level Field Predicted by the Inversion ---
fig4, ax4, im4 = plot(
    1000 * sea_level_posterior * fp.ocean_projection(),
    coasts=True,
    cmap="seismic",
    vmin=-max_abs_sl_change,
    vmax=max_abs_sl_change,
    colorbar_label="Sea Level Change (mm)",
)
ax4.set_title("b) Predicted Sea-Level Fingerprint")
# ax4.plot(
#     points[1],  # longitudes
#     points[0],  # latitudes
#     "kx",
#     label="Altimetry Point Estimations",
#     transform=ccrs.PlateCarree(),
# )

fig4_a, ax4_a, im4_a = plot(
    1000
    * np.abs(sea_level_posterior - sea_level_true)
    * fp.ocean_projection()
    * fp.length_scale,
    coasts=True,
    cmap=cc.cm.CET_L17,
    colorbar_label="Sea Level Change Error (mm)",
)
# %%


model_space = ice_thickness_measure.domain

# Set the weighting function for GMSL estimates  - Note that length scale factor to dimensionalise the result into mm
GMSL_weighting_function = (
    -fp.ice_density
    * fp.one_minus_ocean_function
    * fp.ice_projection(value=0)
    * 1000
    * fp.length_scale
    / (fp.water_density * fp.ocean_area)
)

# Form the mapping to GSML.
B = averaging_operator(
    model_space, [GMSL_weighting_function]
)

# Get the true GMSL
GMSL_true = B(model_true)

# Push forward the posterior to the GMSL space.
GMSL_prior_measure = ice_thickness_measure.affine_mapping(
    operator=B
)
GMSL_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=B)
)

ssh_esimation = (
    ice_change.load_to_estimated_gmsl_operator(
        ice_change.ice_thickness_to_load_operator(
            model_true
        )
    )[0]
    * 1000
)

ssh_esimation_alt = (
    ice_change.load_to_point_estimated_gmsl_operator(
        ice_change.ice_thickness_to_load_operator(
            model_true
        )
    )[0]
    * 1000
)

# make two plots next to each other, one the includes the prior and posterior distributions, with vertical lines for true values and ssh estimation (with the two pdfs on different y scales)
# the other is the same but without the prior distribution
#
prior_expectation = GMSL_prior_measure.expectation[0]
posterior_expectation = GMSL_posterior_measure.expectation[
    0
]

prior_std_dev = standard_dev(GMSL_prior_measure)
posterior_std_dev = standard_dev(GMSL_posterior_measure)

x1_range = np.linspace(
    prior_expectation - 5 * prior_std_dev,
    prior_expectation + 5 * prior_std_dev,
    1000,
)
x2_range = np.linspace(
    posterior_expectation - 6 * posterior_std_dev,
    posterior_expectation + 6 * posterior_std_dev,
    1000,
)


def gaussian(x, mean, std_dev):
    return (
        1
        / (std_dev * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )


prior_func = gaussian(
    x1_range, prior_expectation, prior_std_dev
)
posterior_func = gaussian(
    x2_range, posterior_expectation, posterior_std_dev
)

fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
# two y scale, one for prior and one for posterior

ax1.plot(
    x2_range,
    posterior_func,
    label=f"Posterior Distribution\n(mean={posterior_expectation:.2f} mm, std={posterior_std_dev:.2f} mm)",
    color="blue",
)
ax1.axvline(
    GMSL_true,
    color="black",
    linestyle="--",
    label=f"True GMSL ({GMSL_true[0]:.2f} mm)",
)

ax1.get_yaxis().set_visible(False)
ax1.axvline(
    ssh_esimation_alt,
    color="red",
    label=f"SSH Estimation ({ssh_esimation_alt:.2f} mm)",
)

ax1.set_title(
    "Prior and Posterior Distributions of GMSL Contribution"
)
ax1.legend()

plt.tight_layout()
plt.savefig("gmsl_contribution_distributions.png", dpi=600)
plt.show()

# %%

plot(model_true)
# %%

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

property_true = C(model_true)
property_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=C)
)


property_true = C(model_true)
property_posterior_measure = (
    model_posterior_measure.affine_mapping(operator=C)
)

# Visualise the distribution using a corner plot
plot_corner_distributions(
    property_posterior_measure,
    true_values=property_true,
    labels=[
        "Greenland Contribution (mm)",
        "West Antarctica Contribution (mm)",
        "East Antarctica Contribution (mm)",
    ],
    title="Joint Posterior Distributions of GMSL Contributions from Major Ice Sheets",
)
