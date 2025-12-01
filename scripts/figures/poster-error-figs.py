# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyslfp import (
    FingerPrint,
    averaging_operator,
    plot,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

mpl.rcParams["font.size"] = 24
mpl.rcParams["figure.dpi"] = 600
output_dir = "../../outputs/poster/AutomatedFigures/Distributions"

# %%
# Setup
lmax = 256
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
ice_length_scale = 0.5 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.005
net_ice_thickness_change = -5.0

odt_length_scale = 0.1 * fp.mean_sea_floor_radius
odt_standard_deviation_factor = 0.001 / fp.length_scale

altimetry_error_length_scale = 0.01 * fp.mean_sea_floor_radius
altimetry_error_standard_deviation = 0.0005 / fp.length_scale

altimetry_range = 66


# %%

ice_thickness_change_measure, ice_direct_load_change_measure = (
    ice_thickness_change_measures(
        fp,
        fingerprint_operator,
        ice_length_scale,
        ice_gmsl_target_std,
        net_ice_thickness_change,
    )
)

ice_thickness_change_measure_sample = (
    ice_thickness_change_measure.sample()
)

ice_load_measure_sample = fp.direct_load_from_ice_thickness_change(
    ice_thickness_change_measure_sample,
)
ice_slc_sample_response = fp(direct_load=ice_load_measure_sample)

ice_ssh_sample = fp.sea_surface_height_change(
    ice_slc_sample_response[0],
    ice_slc_sample_response[1],
    ice_slc_sample_response[3],
)

ice_slc_sample = ice_slc_sample_response[0]

ice_slc_measure = ice_direct_load_change_measure.affine_mapping(
    operator=response_space.subspace_projection(0)
    @ fingerprint_operator,
)

ice_ssh_measure = ice_direct_load_change_measure.affine_mapping(
    operator=sea_surface_height_op @ fingerprint_operator,
)

# %%

odt_measure, odt_load_measure = ocean_dynamic_topography_measures(
    fp,
    fingerprint_operator,
    odt_length_scale,
    odt_standard_deviation_factor,
)

### SAMPLES

odt_measure_sample = odt_measure.sample()
odt_load_sample = fp.direct_load_from_sea_level_change(
    odt_measure_sample,
)
odt_fingerprint_response = fp(direct_load=odt_load_sample)
odt_fingerprint_sample = fp.sea_surface_height_change(
    odt_fingerprint_response[0],
    odt_fingerprint_response[1],
    odt_fingerprint_response[3],
)
odt_error_sample = odt_fingerprint_sample + odt_measure_sample


### OPERATORS


sea_level_to_load_op = sea_level_change_to_load_operator(
    fp,
    load_space,
)
odt_combined_op = (
    sea_surface_height_op
    @ fingerprint_operator
    @ sea_level_to_load_op
    + odt_measure.domain.identity_operator()
)

odt_error_measure = odt_measure.affine_mapping(
    operator=odt_combined_op,
)


# %%
altimetry_error_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
    1.5,
    altimetry_error_length_scale,
    altimetry_error_standard_deviation,
)
altimetry_error_sample = altimetry_error_measure.sample()


# %%

combined_error_measure = odt_error_measure + altimetry_error_measure
combined_error_sample = odt_error_sample + altimetry_error_sample

total_observerable_measure = ice_ssh_measure + combined_error_measure
total_observable_sample = ice_ssh_sample + combined_error_sample

# %%
altimetry_projection = fp.altimetry_projection(
    latitude_min=-altimetry_range,
    latitude_max=altimetry_range,
    value=0,
) * fp.ocean_projection(value=0)

altimetry_projection_for_plotting = fp.altimetry_projection(
    latitude_min=-altimetry_range,
    latitude_max=altimetry_range,
    value=np.nan,
) * fp.ocean_projection(value=np.nan)

altimetry_operator = spatial_mutliplication_operator(
    altimetry_projection,
    measurement_space,
)

total_observed_measure = total_observerable_measure.affine_mapping(
    operator=altimetry_operator,
)
total_observed_sample = total_observable_sample * altimetry_projection

# %% map plots

plots = [
    (
        ice_thickness_change_measure_sample * fp.ice_projection(),
        "Ice Thickness Change",
        "metres",
    ),
    (
        ice_slc_sample * fp.ocean_projection() * 1000,
        "Sea Level Change from Ice Load",
        "mm",
    ),
    (
        ice_ssh_sample * fp.ocean_projection() * 1000,
        "Sea Surface Height Change from Ice Load",
        "mm",
    ),
    (
        odt_measure_sample * fp.ocean_projection() * 1000,
        "Ocean Dynamic Topography Change",
        "mm",
    ),
    (
        odt_error_sample * fp.ocean_projection() * 1000,
        "ODT Error",
        "mm",
    ),
    (
        altimetry_error_sample * fp.ocean_projection() * 1000,
        "Altimetry Error",
        "mm",
    ),
    (
        combined_error_sample * fp.ocean_projection() * 1000,
        "Combined Ocean Error",
        "mm",
    ),
    (
        total_observed_sample
        * altimetry_projection_for_plotting
        * 1000,
        "Total Observed Sea Surface Height Change",
        "mm",
    ),
]


for x in plots:
    fig, ax, im = plot(
        x[0],
        symmetric=True,
        gridlines=False,
    )
    # set colorbar below plot with label
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.8,
    )
    cbar.set_label(x[2])
    ax.set_title(x[1])
    # if ice driven sea level change plot with tab:blue title color, if total observed plot with tab:orange title color else black
    if x[1] == "Sea Level Change from Ice Load":
        ax.title.set_color("tab:blue")
    elif x[1] == "Total Observed Sea Surface Height Change":
        ax.title.set_color("tab:orange")
    else:
        ax.title.set_color("black")
    try:
        plt.savefig(
            f"{output_dir}/{x[1].lower().replace(' ', '_')}.png",
            dpi=600,
            bbox_inches="tight",
        )
    except:
        print("Could not save figure")
    plt.close()

# %% measure plots
plots = [
    (
        "Ice Thickness Change Measure",
        ice_thickness_change_measure,
        fp.ice_projection(value=0),
        "m",
        1.0,
    ),
    (
        "Ice Change Driven Sea Level Change",
        ice_slc_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "Ice Change Driven Sea Surface Height Change",
        ice_ssh_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "Ocean Dynamic Topography Change Measure",
        odt_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "ODT Error Measure",
        odt_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "Altimetry Error Measure",
        altimetry_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "Combined Ocean Error Measure",
        combined_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
    ),
    (
        "Total Observed Sea Surface Height Change Measure",
        total_observed_measure,
        altimetry_projection,
        "mm",
        1000.0,
    ),
]


# %%
ice_gmsl_spatial_average = averaging_operator(
    ice_slc_measure.domain,
    [
        fp.ocean_projection(value=0)
        / fp.integrate(fp.ocean_projection(value=0)),
    ],
)
ice_gmsl_averaged_measure = ice_slc_measure.affine_mapping(
    operator=ice_gmsl_spatial_average,
)
ice_gmsl_std = (
    ice_gmsl_averaged_measure.covariance.matrix(dense=True)[0, 0]
    ** 0.5
)
ice_gmsl_std_scaled = (
    ice_gmsl_std * 1000.0 * fp.length_scale
)  # Convert to mm and plot units

ice_gmsl_expectation_scaled = (
    ice_gmsl_averaged_measure.expectation[0]
    * 1000.0
    * fp.length_scale
)  # Convert to mm and plot units

for data in plots:
    spatial_average = averaging_operator(
        data[1].domain,
        [data[2] / fp.integrate(data[2])],
    )
    averaged_measure = data[1].affine_mapping(
        operator=spatial_average,
    )
    expectation = averaged_measure.expectation[0]
    std = averaged_measure.covariance.matrix(dense=True)[0, 0] ** 0.5
    expectation *= data[4]
    std *= data[4]
    print(
        f"{data[0]}: Expectation = {expectation:.4e} {data[3]}, Standard Deviation = {std:.4e} {data[3]}",
    )
    x_space = np.linspace(
        expectation - 4 * std,
        expectation + 4 * std,
        100,
    )
    pdf = norm.pdf(x_space, loc=expectation, scale=std)

    # Get current axes
    ax = plt.gca()
    if data[0] == "Ice Change Driven Sea Level Change":
        color = "tab:blue"
    elif (
        data[0] == "Total Observed Sea Surface Height Change Measure"
    ):
        color = "tab:orange"
    else:
        color = "black"
    ax.plot(
        x_space * fp.length_scale,
        pdf * fp.length_scale,
        color=color,
        linewidth=3,
    )
    # ax.set_title(f"{data[0]} Distribution")
    ax.set_xlabel(f"{data[3]}")
    ax.set_ylabel("Probability Density")

    # Add secondary x axis - need to account for length_scale in the conversion
    if data[3] == "mm":
        ax2 = ax.secondary_xaxis(
            -0.25,
            functions=(
                lambda x,
                e=expectation * fp.length_scale,
                s=ice_gmsl_std_scaled: (x - e) / s,
                lambda x,
                e=expectation * fp.length_scale,
                s=ice_gmsl_std_scaled: x * s + e,
            ),
        )
        ax2.set_xlabel("Relative to Ice GMSL (σ)")

    try:
        plt.savefig(
            f"{output_dir}/{data[0].lower().replace(' ', '_')}_distribution.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.savefig(
            f"{output_dir}/{data[0].lower().replace(' ', '_')}_distribution.svg",
            dpi=600,
            bbox_inches="tight",
        )
    except:
        print("Could not save figure")
    plt.close()

# %%

gmsl_true_expectation = (
    ice_gmsl_averaged_measure.expectation[0]
    * 1000.0
    * fp.length_scale
)
gmsl_true_std = (
    (
        ice_gmsl_averaged_measure.covariance.matrix(dense=True)[0, 0]
        ** 0.5
    )
    * 1000.0
    * fp.length_scale
)

alt_projection = fp.altimetry_projection(
    latitude_min=-altimetry_range,
    latitude_max=altimetry_range,
    value=0,
) * fp.ocean_projection(value=0)

alt_projection_avg_op = averaging_operator(
    total_observed_measure.domain,
    [alt_projection / fp.integrate(alt_projection)],
)

gmsl_estimated = total_observed_measure.affine_mapping(
    operator=alt_projection_avg_op,
)

gmsl_estimated_expectation = (
    gmsl_estimated.expectation[0] * 1000.0 * fp.length_scale
)
gmsl_estimated_std = (
    (gmsl_estimated.covariance.matrix(dense=True)[0, 0] ** 0.5)
    * 1000.0
    * fp.length_scale
)

error_mean = gmsl_estimated_expectation - gmsl_true_expectation
error_std = (gmsl_estimated_std**2 + gmsl_true_std**2) ** 0.5

print(f"True GMSL Expectation: {gmsl_true_expectation:.4e} mm")
print(
    f"True GMSL Std: {gmsl_true_std * fp.length_scale:.4e} mm",
)
print(
    f"Estimated GMSL Expectation: {gmsl_estimated_expectation:.4e} mm",
)
print(f"Estimated GMSL Std: {gmsl_estimated_std:.4e} mm")
print(f"GMSL Error Mean: {error_mean:.4e} mm")
print(f"GMSL Error Std: {error_std:.4e} mm")


# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# GMSL distributions
x_range = 4
x_min = min(
    gmsl_estimated_expectation - x_range * gmsl_estimated_std,
    gmsl_true_expectation - x_range * gmsl_true_std,
)
x_max = max(
    gmsl_estimated_expectation + x_range * gmsl_estimated_std,
    gmsl_true_expectation + x_range * gmsl_true_std,
)
x = np.linspace(x_min, x_max, 1000)

ax1.plot(
    x,
    norm.pdf(x, gmsl_true_expectation, gmsl_true_std),
    "tab:blue",
    label="True GMSL",
    linewidth=3,
)
ax1.plot(
    x,
    norm.pdf(x, gmsl_estimated_expectation, gmsl_estimated_std),
    "tab:orange",
    label="Estimated GMSL",
    linewidth=3,
)
ax1.set_xlabel("GMSL (mm)")
ax1.set_ylabel("Probability Density")
ax1.set_title("GMSL Distributions")
ax1.legend()
ax1.grid(alpha=0.3)

# Secondary axis for ax1: centred on true GMSL expectation
ax1_sec = ax1.secondary_xaxis(
    -0.25,
    functions=(
        lambda x, e=gmsl_true_expectation, s=gmsl_true_std: (x - e)
        / s,
        lambda x, e=gmsl_true_expectation, s=gmsl_true_std: x * s + e,
    ),
)
ax1_sec.set_xlabel("Relative to True GMSL (σ)")
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.7), ncol=2)


# Error distribution
error_x = np.linspace(
    error_mean - 4 * error_std,
    error_mean + 4 * error_std,
    1000,
)
ax2.plot(
    error_x,
    norm.pdf(error_x, error_mean, error_std),
    "tab:red",
    linewidth=3,
)
ax2.axvline(
    0,
    color="k",
    linestyle="--",
    alpha=0.3,
)
ax2.set_xlabel("Error (mm)")
ax2.set_title("Error Distribution")
ax2.grid(alpha=0.3)

# Secondary axis for ax2: centred on error expectation
ax2_sec = ax2.secondary_xaxis(
    -0.25,
    functions=(
        lambda x, e=error_mean, s=gmsl_true_std: (x - e) / s,
        lambda x, e=error_mean, s=gmsl_true_std: x * s + e,
    ),
)
ax2_sec.set_xlabel("Relative to Error Mean (σ)")

try:
    plt.savefig(
        f"{output_dir}/gmsl_and_error_distributions.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.savefig(
        f"{output_dir}/gmsl_and_error_distributions.svg",
        dpi=600,
        bbox_inches="tight",
    )
except:
    print("Could not save figure")
# %%
plt.close()
