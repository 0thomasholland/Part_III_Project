# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from pyslfp import (
    FingerPrint,
    averaging_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

mpl.rcParams["font.size"] = 16
mpl.rcParams["figure.dpi"] = 600
output_dir = "../../outputs/poster/AutomatedFigures/Distributions_Flow/combined"

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

altimetry_range = 77

# %%
# Generate data
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

odt_measure, odt_load_measure = ocean_dynamic_topography_measures(
    fingerprint=fp,
    fingerprint_operator=fingerprint_operator,
    length_scale=odt_length_scale,
    standard_deviation=odt_standard_deviation_factor,
)

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

altimetry_error_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
    1.5,
    altimetry_error_length_scale,
    altimetry_error_standard_deviation,
)
altimetry_error_sample = altimetry_error_measure.sample()

combined_error_measure = odt_error_measure + altimetry_error_measure
combined_error_sample = odt_error_sample + altimetry_error_sample

total_observerable_measure = ice_ssh_measure + combined_error_measure
total_observable_sample = ice_ssh_sample + combined_error_sample

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

# %%
# Calculate Ice GMSL standard deviation for secondary axis
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
ice_gmsl_std_scaled = ice_gmsl_std * 1000.0 * fp.length_scale


# %%
def create_composite_figure(
    shgrid_data,
    map_title,
    map_units,
    dist_expectation,
    dist_std,
    ice_gmsl_std_scaled,
    title_color="black",
    bg_color=None,
    dist_color="black",
    projection=ccrs.Robinson(),
    cmap="RdBu",
    symmetric=False,
    coasts=True,
    show_secondary_axis=True,
    projection_for_avg=None,
):
    """Create a square composite figure with map and inset distribution."""
    # Create square figure
    fig_size = 10
    fig = plt.figure(figsize=(fig_size, fig_size))

    # Set background color
    if bg_color is not None:
        fig.set_facecolor(mcolors.to_rgba(bg_color, alpha=0.2))
    else:
        fig.set_facecolor((1, 1, 1, 0.0))  # fully transparent

    # Create main map axes - nearly full figure
    ax_map = fig.add_axes(
        [0.02, 0.0, 0.96, 0.88],
        projection=projection,
    )

    # Get lon/lat from SHGrid
    lons = shgrid_data.lons()
    lats = shgrid_data.lats()
    data = shgrid_data.data

    # Set up color scaling
    plot_kwargs = {"cmap": cmap}
    if symmetric:
        data_max = 1.2 * np.nanmax(np.abs(data))
        plot_kwargs["vmin"] = -data_max
        plot_kwargs["vmax"] = data_max

    # Plot the data
    im = ax_map.pcolormesh(
        lons,
        lats,
        data,
        transform=ccrs.PlateCarree(),
        **plot_kwargs,
    )

    # Add coastlines
    if coasts:
        ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)

    # Set global extent
    ax_map.set_global()

    # Add title
    ax_map.set_title(map_title, fontsize=36, pad=20)

    # Background for inset (slightly larger than inset)
    ax_inset_bg = fig.add_axes(
        # [left, bottom, width, height]
        [-0.02, 0.05, 0.38, 0.35],  # previous 0.37
    )
    ax_inset_bg.set_facecolor((1, 1, 1, 0.6))
    ax_inset_bg.set_xticks([])
    ax_inset_bg.set_yticks([])
    for spine in ax_inset_bg.spines.values():
        spine.set_visible(False)
    ax_inset_bg.set_zorder(ax_map.get_zorder() + 1)

    # Then create the actual inset on top
    ax_inset = fig.add_axes([0.06, 0.15, 0.28, 0.22])
    ax_inset.set_facecolor((1, 1, 1, 0.5))  # 50% white for plot area
    ax_inset.set_zorder(ax_map.get_zorder() + 2)

    clean_data = shgrid_data.copy()
    clean_data.data = np.nan_to_num(shgrid_data.data, nan=0.0)

    map_mean_value = fp.integrate(
        clean_data * projection_for_avg,
    ) / fp.integrate(projection_for_avg)

    # Add horizontal colorbar next to inset plot
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.45, 0.15, 0.45, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    if map_mean_value != 0.0:
        cbar.set_label(
            f"Example Sample [{map_units}]\nAverage Value: {map_mean_value:.2f} {map_units}",
            fontsize=12,
        )
    else:
        cbar.set_label(
            f"Example Sample [{map_units}]",
            fontsize=12,
        )
    # Plot distribution
    x_space = np.linspace(
        dist_expectation - 4 * dist_std,
        dist_expectation + 4 * dist_std,
        100,
    )
    pdf = norm.pdf(x_space, loc=dist_expectation, scale=dist_std)

    ax_inset.plot(x_space, pdf, color=dist_color, linewidth=2)
    ax_inset.set_xlabel(f"Global Average [{map_units}]", fontsize=12)
    ax_inset.set_ylabel("Probability Density", fontsize=10)
    ax_inset.set_title("Distribution of Global Average", fontsize=12)
    # inset figure background 20% white

    # Add secondary x-axis if requested and units are mm
    if show_secondary_axis and map_units == "mm":
        ax_inset_sec = ax_inset.secondary_xaxis(
            -0.25,
            functions=(
                lambda x, e=dist_expectation, s=ice_gmsl_std_scaled: (
                    x - e
                )
                / s,
                lambda x, e=dist_expectation, s=ice_gmsl_std_scaled: x
                * s
                + e,
            ),
        )
        ax_inset_sec.set_xlabel(
            "Relative to True GMSL [σ]",
            fontsize=10,
        )
        ax_inset_sec.tick_params(labelsize=10)

    # Style the inset
    ax_inset.tick_params(labelsize=10)

    # Add border to inset
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(1)

    return fig, ax_map, ax_inset


def get_measure_stats(measure, projection, scale_factor, fp):
    """Calculate expectation and standard deviation for a measure."""
    spatial_average = averaging_operator(
        measure.domain,
        [projection / fp.integrate(projection)],
    )
    averaged_measure = measure.affine_mapping(
        operator=spatial_average,
    )
    expectation = (
        averaged_measure.expectation[0]
        * scale_factor
        * fp.length_scale
    )
    std = (
        averaged_measure.covariance.matrix(dense=True)[0, 0] ** 0.5
        * scale_factor
        * fp.length_scale
    )
    var = averaged_measure.covariance.matrix(dense=True)[0, 0]
    print(f"Variance: {var}")  # Likely something like -1e-30
    return expectation, std


# %%
# Define all plot configurations
# Each entry: (
#     title,
#     map_sample,
#     measure,
#     projection_for_avg,
#     units,
#     scale_factor,
#     title_color,
#     bg_color,
# )

composite_plots = [
    (
        "Ice Thickness Change",
        ice_thickness_change_measure_sample * fp.ice_projection(),
        ice_thickness_change_measure,
        fp.ice_projection(value=0),
        "m",
        1.0,
        "black",
        None,
    ),
    (
        "Sea Level Change from Ice Load",
        ice_slc_sample * fp.ocean_projection() * 1000,
        ice_slc_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "tab:blue",
        "tab:blue",
    ),
    (
        "Sea Surface Height Change from Ice Load",
        ice_ssh_sample * fp.ocean_projection() * 1000,
        ice_ssh_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "black",
        None,
    ),
    (
        "Ocean Dynamic Topography Change",
        odt_measure_sample * fp.ocean_projection() * 1000,
        odt_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "black",
        None,
    ),
    (
        "Ocean Dynamic Topography induced Error",
        odt_error_sample * fp.ocean_projection() * 1000,
        odt_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "black",
        None,
    ),
    (
        "Altimetry Sensor Error",
        altimetry_error_sample * fp.ocean_projection() * 1000,
        altimetry_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "black",
        None,
    ),
    (
        "Combined Error",
        combined_error_sample * fp.ocean_projection() * 1000,
        combined_error_measure,
        fp.ocean_projection(value=0),
        "mm",
        1000.0,
        "black",
        None,
    ),
    (
        "Total Observed Sea Surface Height Change",
        total_observed_sample
        * altimetry_projection_for_plotting
        * 1000,
        total_observed_measure,
        altimetry_projection,
        "mm",
        1000.0,
        "tab:orange",
        "tab:orange",
    ),
]

# %%
# Generate all composite figures
for plot_config in composite_plots:
    (
        title,
        map_sample,
        measure,
        projection_for_avg,
        units,
        scale_factor,
        title_color,
        bg_color,
    ) = plot_config

    # Calculate distribution statistics
    expectation, std = get_measure_stats(
        measure,
        projection_for_avg,
        scale_factor,
        fp,
    )

    print(
        f"{title}: Expectation = {expectation:.4e} {units}, Std = {std:.4e} {units}",
    )

    # Determine distribution color (same as title color, or black if None)
    dist_color = title_color if title_color != "black" else "black"

    # Create composite figure
    fig, ax_map, ax_inset = create_composite_figure(
        shgrid_data=map_sample,
        map_title=title,
        map_units=units,
        dist_expectation=expectation,
        dist_std=std,
        ice_gmsl_std_scaled=ice_gmsl_std_scaled,
        title_color=title_color,
        bg_color=bg_color,
        dist_color=dist_color,
        projection_for_avg=projection_for_avg,
    )

    # Save figure
    filename = title.lower().replace(" ", "_")
    try:
        plt.savefig(
            f"{output_dir}/{filename}_composite.png",
            dpi=600,
            bbox_inches="tight",
        )
    except Exception as e:
        print(f"  Could not save figure: {e}")

    plt.close()

print("\nAll composite figures generated!")


# %%
# GMSL Comparison and Error Distribution Plot
# (adapted from code A)

# Calculate true GMSL statistics
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

# Calculate estimated GMSL statistics using altimetry projection
alt_projection_avg_op = averaging_operator(
    total_observed_measure.domain,
    [altimetry_projection / fp.integrate(altimetry_projection)],
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

# Calculate error statistics
error_mean = gmsl_estimated_expectation - gmsl_true_expectation
error_std = (gmsl_estimated_std**2 + gmsl_true_std**2) ** 0.5

print(f"True GMSL Expectation: {gmsl_true_expectation:.4e} mm")
print(f"True GMSL Std: {gmsl_true_std:.4e} mm")
print(
    f"Estimated GMSL Expectation: {gmsl_estimated_expectation:.4e} mm",
)
print(f"Estimated GMSL Std: {gmsl_estimated_std:.4e} mm")
print(f"GMSL Error Mean: {error_mean:.4e} mm")
print(f"GMSL Error Std: {error_std:.4e} mm")

# %%
# Create the dual-panel figure
error_fig, (error_ax1, error_ax2) = plt.subplots(
    1,
    2,
    figsize=(12, 5),
    sharey=True,
)
error_fig.patch.set_alpha(0.0)

# GMSL distributions
gmsl_x_range = 4
gmsl_x_min = min(
    gmsl_estimated_expectation - gmsl_x_range * gmsl_estimated_std,
    gmsl_true_expectation - gmsl_x_range * gmsl_true_std,
)
gmsl_x_max = max(
    gmsl_estimated_expectation + gmsl_x_range * gmsl_estimated_std,
    gmsl_true_expectation + gmsl_x_range * gmsl_true_std,
)
gmsl_x = np.linspace(gmsl_x_min, gmsl_x_max, 1000)

error_ax1.plot(
    gmsl_x,
    norm.pdf(gmsl_x, gmsl_true_expectation, gmsl_true_std),
    "tab:blue",
    label="True GMSL",
    linewidth=3,
)
error_ax1.plot(
    gmsl_x,
    norm.pdf(gmsl_x, gmsl_estimated_expectation, gmsl_estimated_std),
    "tab:orange",
    label="Estimated GMSL",
    linewidth=3,
)
error_ax1.set_xlabel("GMSL (mm)")
error_ax1.set_ylabel("Probability Density")
error_ax1.set_title("GMSL Distributions")
error_ax1.legend()
error_ax1.grid(alpha=0.3)

# Secondary axis for error_ax1
error_ax1_sec = error_ax1.secondary_xaxis(
    -0.25,
    functions=(
        lambda x, e=gmsl_true_expectation, s=gmsl_true_std: (x - e)
        / s,
        lambda x, e=gmsl_true_expectation, s=gmsl_true_std: x * s + e,
    ),
)
error_ax1_sec.set_xlabel("Relative to True GMSL (σ)")
error_ax1.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.6),  # (x, y)
    ncol=2,
)

# Error distribution
error_dist_x = np.linspace(
    error_mean - 4 * error_std,
    error_mean + 4 * error_std,
    1000,
)
error_ax2.plot(
    error_dist_x,
    norm.pdf(error_dist_x, error_mean, error_std),
    "tab:red",
    linewidth=3,
)
error_ax2.axvline(
    0,
    color="k",
    linestyle="--",
    alpha=0.3,
)
error_ax2.set_xlabel("Error (mm)")
error_ax2.set_title("Error Distribution")
error_ax2.grid(alpha=0.3)

# Secondary axis for error_ax2
error_ax2_sec = error_ax2.secondary_xaxis(
    -0.25,
    functions=(
        lambda x, e=error_mean, s=gmsl_true_std: (x - e) / s,
        lambda x, e=error_mean, s=gmsl_true_std: x * s + e,
    ),
)
error_ax2_sec.set_xlabel("Relative to True GMSL (σ)")

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
except Exception as e:
    print(f"Could not save figure: {e}")

plt.close()


# %%
# ODT change measure plot, but with the distribution input being dotted and a delta function at zero
fig_size = 10
fig = plt.figure(figsize=(fig_size, fig_size))

# Set background color
fig.set_facecolor((1, 1, 1, 0.0))  # fully transparent

# Create main map axes - nearly full figure
ax_map = fig.add_axes(
    [0.02, 0.0, 0.96, 0.88],
    projection=ccrs.Robinson(),
)

grid_data = odt_measure_sample * fp.ocean_projection() * 1000

# Get lon/lat from SHGrid
lons = grid_data.lons()
lats = grid_data.lats()
data = grid_data.data

# Set up color scaling
plot_kwargs = {"cmap": "RdBu"}

data_max = 1.2 * np.nanmax(np.abs(data))
plot_kwargs["vmin"] = -data_max
plot_kwargs["vmax"] = data_max

# Plot the data
im = ax_map.pcolormesh(
    lons,
    lats,
    data,
    transform=ccrs.PlateCarree(),
    **plot_kwargs,
)

# Add coastlines

ax_map.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Set global extent
ax_map.set_global()

# Add title
ax_map.set_title(
    "Ocean Dynamic Topography Change",
    fontsize=36,
    pad=20,
)

# Background for inset (slightly larger than inset)
ax_inset_bg = fig.add_axes(
    # [left, bottom, width, height]
    [-0.02, 0.05, 0.38, 0.35],  # previous 0.37
)
ax_inset_bg.set_facecolor((1, 1, 1, 0.6))
ax_inset_bg.set_xticks([])
ax_inset_bg.set_yticks([])
for spine in ax_inset_bg.spines.values():
    spine.set_visible(False)
ax_inset_bg.set_zorder(ax_map.get_zorder() + 1)

# Then create the actual inset on top
ax_inset = fig.add_axes([0.06, 0.15, 0.28, 0.22])
ax_inset.set_facecolor((1, 1, 1, 0.5))  # 50% white for plot area
ax_inset.set_zorder(ax_map.get_zorder() + 2)

clean_data = grid_data.copy()
clean_data.data = np.nan_to_num(grid_data.data, nan=0.0)

map_mean_value = fp.integrate(
    clean_data * projection_for_avg,
) / fp.integrate(projection_for_avg)

# Add horizontal colorbar next to inset plot
# [left, bottom, width, height]
cbar_ax = fig.add_axes([0.45, 0.15, 0.45, 0.025])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
if map_mean_value != 0.0:
    cbar.set_label(
        f"Example Sample [mm]\nAverage Value: {map_mean_value:.2f} mm",
        fontsize=12,
    )
else:
    cbar.set_label(
        "Example Sample [mm]",
        fontsize=12,
    )
# Plot distribution

x_space = np.linspace(
    -4 * odt_standard_deviation_factor * 1000.0,
    4 * odt_standard_deviation_factor * 1000.0,
    1000,
)
# Create dotted normal distribution for the input parameters of ODT defined at top and delta function at zero
pdf_dotted = norm.pdf(
    x_space,
    loc=0.0,
    scale=odt_standard_deviation_factor * 1000.0,
)
pdf_delta = np.zeros_like(x_space)
closest_index = np.argmin(np.abs(x_space - 0.0))
pdf_delta[closest_index] = 1.0 / (x_space[1] - x_space[0])

ax_inset.plot(
    x_space,
    pdf_dotted,
    color="k",
    linewidth=2,
    linestyle=":",
    label="Point Location",
)
ax_inset.plot(
    x_space,
    pdf_delta,
    color="k",
    linewidth=2,
    label="Global Average",
)
ax_inset.set_xlabel(
    "Global Average and Point Location [mm]",
    fontsize=12,
)
ax_inset.legend(fontsize=10)

# limit the y axis to ~ 3 times the max of the dotted distribution
ax_inset.set_ylim(0, 3 * np.max(pdf_dotted))
ax_inset.set_ylabel("Probability Density", fontsize=10)
ax_inset.set_title("Distribution", fontsize=12)
# inset figure background 20% white

# Add secondary x-axis if requested and units are mm

ax_inset_sec = ax_inset.secondary_xaxis(
    -0.25,
    functions=(
        lambda x, e=0, s=ice_gmsl_std_scaled: (x - e) / s,
        lambda x, e=0, s=ice_gmsl_std_scaled: x * s + e,
    ),
)
ax_inset_sec.set_xlabel(
    "Relative to True GMSL [σ]",
    fontsize=10,
)
ax_inset_sec.tick_params(labelsize=10)

# Style the inset
ax_inset.tick_params(labelsize=10)

# Add border to inset
for spine in ax_inset.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1)

try:
    plt.savefig(
        f"{output_dir}/odt_change_with_distribution_composite.png",
        dpi=600,
        bbox_inches="tight",
    )
except Exception as e:
    print(f"  Could not save figure: {e}")

# %%
