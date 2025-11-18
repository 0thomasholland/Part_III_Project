# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
import colorcet as cc
from scipy.interpolate import griddata

data = pd.read_csv(
    "gmsl_error_with_measurement_noise_results_lmax64.csv",
)

# %%

# do a scatterplot matrix where y values are the error and standard deviations and the x values are the rest

# x values: error_mean, error_std
# y values: ice_length_scale" "ice_gmsl_target_std", "net_ice_thickness_change",    "odt_length_scale", "odt_standard_deviation", "altimetry_error_length_scale", "altimetry_error_amplitude", "altimetry_range",

sns.pairplot(
    data,
    x_vars=[
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "odt_standard_deviation",
        "altimetry_error_amplitude",
        "altimetry_range",
    ],
    y_vars=["error_mean", "error_std"],
    height=4,
    aspect=1,
    kind="scatter",
)

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
    xmax = grouped["error_mean"].max() + 4 * grouped["error_std"].max()
    xmin = grouped["error_mean"].min() - 4 * grouped["error_std"].max()
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(8, 6))
    for i, row in grouped.iterrows():
        mean = row["error_mean"]
        std = row["error_std"]
        plt.plot(
            x,
            norm.pdf(x, mean, std),
            label=f"{param}={i:.4f}",
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
    cmap=cc.cm.CET_D13,
    alpha=0.7,
)
plt.colorbar(scatter, label="Mean GMSL Error (m)")
plt.xlabel("Net Ice Thickness Change (m/year)")
plt.ylabel("Altimetry Range (m)")
plt.title("Net Ice Thickness Change vs Altimetry Range with GMSL Error")

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
    cmap=cc.cm.CET_D13,
)
plt.colorbar(label="Mean GMSL Error (m)")
plt.xlabel("Net Ice Thickness Change (m/year)")
plt.ylabel("Altimetry Range (m)")
plt.title("Interpolated GMSL Error over Net Ice Thickness Change and Altimetry Range")

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
    np.abs(data["error_mean"]),
    (xi, yi),
    method="cubic",
)
# plot the interpolated data using pcolormesh
plt.pcolormesh(
    xi,
    yi,
    zi,
    shading="auto",
    cmap=cc.cm.CET_L17,
)
plt.colorbar(label="Absolute Mean GMSL Error (m)")
plt.xlabel("Net Ice Thickness Change (m)")
plt.ylabel("Altimetry Range (+/- deg)")
plt.title("GMSL Error over Net Ice Thickness Change and Altimetry Range")
plt.savefig("gmsl_error_over_ice_change_and_altimetry_range.pdf", dpi=600)

# %%
# for 66 deg altimetry range, plot net_ice_thickness_change vs error_mean with error bars of error_std

subset = data[data["altimetry_range"] == 66]
plt.figure(figsize=(8, 6))
plt.errorbar(
    subset["net_ice_thickness_change"],
    subset["error_mean"],
    yerr=subset["error_std"],
    fmt="o",
    ecolor="r",
    capsize=5,
)
plt.xlabel("Net Ice Thickness Change (m)")
plt.ylabel("Mean GMSL Error (m)")
plt.title("GMSL Error vs Net Ice Thickness Change at 66 deg Altimetry Range")


# %%
# generate a 3D plot of net_ice_thickness_change vs altimetry_range vs error_mean with height being error_mean
%matplotlib qt
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
ax.set_xlabel("Net Ice Thickness Change (m/year)")
ax.set_ylabel("Altimetry Range (m)")
ax.set_zlabel("Mean GMSL Error (m)")
ax.set_title("3D Scatter Plot of GMSL Error")
plt.show()



# %%

%matplotlib inline

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
