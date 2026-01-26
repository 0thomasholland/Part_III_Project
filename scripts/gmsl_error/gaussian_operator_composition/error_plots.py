# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.stats import norm

data = pd.read_csv(
    "gmsl_error_with_measurement_noise_results_lmax128.csv",
)

# %%

# do a scatterplot matrix where y values are the error and standard deviations and the x values are the rest

# x values: error_mean, error_std
# y values: ice_length_scale" "ice_gmsl_target_std", "net_ice_thickness_change",    "odt_length_scale", "odt_standard_deviation", "altimetry_error_length_scale", "altimetry_error_amplitude", "altimetry_range",

x_vars = [
    "ice_gmsl_target_std",
    "ice_length_scale",
    "net_ice_thickness_change",
    "odt_length_scale",
    "odt_standard_deviation",
    "altimetry_error_length_scale",
    "altimetry_error_amplitude",
    "altimetry_range",
]
y_vars = ["error_mean", "error_std"]

fig, axes = plt.subplots(
    len(y_vars), len(x_vars), figsize=(len(x_vars) * 4, len(y_vars) * 4)
)
for i, y_var in enumerate(y_vars):
    for j, x_var in enumerate(x_vars):
        axes[i, j].scatter(data[x_var], data[y_var])
        axes[i, j].set_xlabel(x_var)
        axes[i, j].set_ylabel(y_var)
plt.tight_layout()

# %%
# for each of the input parameters, average the error_mean and error_std over the other parameters and plot each of the distributions using scipy norm

input_parameters = [
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_standard_deviation",
    "altimetry_error_amplitude",
    "altimetry_range",
]

for param in input_parameters:
    grouped = data.groupby(param).agg(
        {"error_mean": "mean", "error_std": "mean"},
    )
    # find the max and min y values of the distributions via 4* standard deviation
    xmax = (
        grouped["error_mean"].max() + 4 * grouped["error_std"].max()
    )
    xmin = (
        grouped["error_mean"].min() - 4 * grouped["error_std"].max()
    )
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(8, 6))
    for k, (i, row) in enumerate(grouped.iterrows()):
        mean = row["error_mean"]
        std = row["error_std"]

        # Use the sequential index 'k' for color mapping
        color_value = k / len(grouped)

        plt.plot(
            x,
            norm.pdf(x, mean, std),
            label=f"{param}={i:.4f}",
            # Use the sequential index for the color map
            color=plt.cm.magma(color_value),
        )
    plt.title(f"GMSL Error Distribution varying {param}")
    plt.xlabel("GMSL Error (m)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()


# %%

# plot the net_ice_thickness_change vs altimetry_range with size being error_mean
# use the cet D12

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    data["net_ice_thickness_change"],
    data["altimetry_range"],
    # s=data["error_mean"] * 1000,`
    c=data["error_mean"],
    cmap="coolwarm",
    norm=mpl.colors.TwoSlopeNorm(
        vcenter=0,
        vmin=-data["error_mean"].max(),
        vmax=data["error_mean"].max(),
    ),
)
plt.colorbar(scatter, label="GMSL Error Mean (m)")
plt.xlabel("Net Ice Thickness Change (m)")
plt.ylabel("Altimetry Range (deg)")
plt.title(
    "Net Ice Thickness Change vs Altimetry Range with GMSL Error Mean",
)

# do same plot but with but with continous colourisation across the entire data range using interpolation using pcolormesh
plt.figure(figsize=(8, 6))
# create a grid of net_ice_thickness_change and altimetry_range
xi = np.linspace(
    data["net_ice_thickness_change"].min(),
    data["net_ice_thickness_change"].max(),
    100,
)
yi = np.linspace(
    data["altimetry_range"].min(),
    data["altimetry_range"].max(),
    100,
)
xi, yi = np.meshgrid(xi, yi)
# interpolate the error_mean onto the grid

zi = griddata(
    (data["net_ice_thickness_change"], data["altimetry_range"]),
    data["error_mean"],
    (xi, yi),
    method="cubic",
)
# plot the interpolated data using pcolormesh
plt.pcolormesh(
    xi,
    yi,
    zi,
    shading="auto",
    cmap="coolwarm",
    # set 0 as center of the colormap
    norm=mpl.colors.TwoSlopeNorm(
        vcenter=0,
        vmin=-data["error_mean"].max(),
        vmax=data["error_mean"].max(),
    ),
)
plt.colorbar(label="GMSL Error Mean (m)")
plt.xlabel("Net Ice Thickness Change (m)")
plt.ylabel("Altimetry Range (deg)")
plt.title(
    "Interpolated GMSL Error over Net Ice Thickness Change and Altimetry Range",
)
plt.savefig(
    "gmsl_error_over_ice_change_and_altimetry_range_contour.pdf",
    dpi=600,
)
plt.savefig(
    "gmsl_error_over_ice_change_and_altimetry_range_contour.png",
    dpi=600,
)
plt.show()

# %%
# generate a 3D plot of net_ice_thickness_change vs altimetry_range vs error_mean with height being error_mean
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    data["net_ice_thickness_change"],
    data["altimetry_range"],
    data["error_mean"],
    c=data["error_mean"],
    cmap="viridis",
    alpha=0.7,
)
ax.set_xlabel("Net Ice Thickness Change (m)")
ax.set_ylabel("Altimetry Range (deg)")
ax.set_zlabel("GMSL Error Mean (m)")
ax.set_title("3D Scatter Plot of GMSL Error Mean")

fig.colorbar(
    ax.scatter(
        data["net_ice_thickness_change"],
        data["altimetry_range"],
        data["error_mean"],
        c=data["error_mean"],
        cmap="viridis",
        alpha=0.7,
    ),
    ax=ax,
    label="GMSL Error Mean (m)",
)


# %%

# plot the cdfs for varying net_ice_thickness_change average over other parameters

grouped = data.groupby("net_ice_thickness_change").agg(
    {"error_mean": "mean", "error_std": "mean"},
)
xmax = grouped["error_mean"].max() + 4 * grouped["error_std"].max()
xmin = grouped["error_mean"].min() - 4 * grouped["error_std"].max()
x = np.linspace(xmin, xmax, 1000)
plt.figure(figsize=(8, 6))
for i, row in grouped.iterrows():
    mean = row["error_mean"]
    std = row["error_std"]
    plt.plot(
        x,
        norm.cdf(x, mean, std),
        label=f"net_ice_thickness_change={i:.4f}",
    )
plt.title("GMSL Error CDFs varying net_ice_thickness_change")
plt.xlabel("GMSL Error (m)")
plt.ylabel("Cumulative Probability")
plt.legend()


# %%

# linear model for error_mean as a function of the input parameters using scipy
