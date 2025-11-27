# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.interpolate import griddata
from scipy.stats import norm
from statsmodels.formula.api import ols

mpl.rcParams["figure.dpi"] = 600

data: pd.DataFrame = pd.read_csv(
    "data_128_scaled.csv",
)

# %%
# print data column names
print(data.columns)

# %%
# print unique net_ice_thickness_change values
print(data["net_ice_thickness_change"].unique())
print(data["altimetry_range"].unique())

# %%

# linear model of error_mean ~net_ice_thickness_change + net_ice_thickness_change:altimetry_range

formula = "error_mean ~ net_ice_thickness_change + net_ice_thickness_change:altimetry_range"
model = ols(formula, data=data).fit()
print("Linear regression results for interaction model:")
print(model.summary())

# %%

data["error_mean_mm"] = data["error_mean"] * 1000

# %% error mean plot

# plot a field of error_mean as a smooth color grid over net_ice_thickness_change and altimetry_range, and show a cross section for altimetry range = 66 and net_ice_thickness_change = 10m, showing dotted lines on the grid for the cross section values, and plots of error vs the other variable on the side panels using subplot mosaic

x = data["net_ice_thickness_change"]
y = data["altimetry_range"]
z = data["error_mean_mm"]
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method="cubic")

fig, axs = plt.subplot_mosaic(
    [["net_ice", "grid"], [".", "altimetry"]],
    figsize=(10, 9),
    width_ratios=(1, 4),
    height_ratios=(4, 1),
    layout="constrained",
)

axs["grid"].sharex(axs["altimetry"])

# 2. Main Grid and Left Cross-Section share the Y-axis
axs["grid"].sharey(axs["net_ice"])

c = axs["grid"].pcolormesh(
    xi,
    yi,
    zi,
    shading="gouraud",
    cmap="seismic",
    norm=mpl.colors.TwoSlopeNorm(
        vcenter=0,
        vmin=-data["error_mean_mm"].max(),
        vmax=data["error_mean_mm"].max(),
    ),
)
# colourbar plotted to the right of the entire mosaic
fig.colorbar(
    c,
    ax=axs["grid"],
    label="GMSL Error (mm)",
    pad=0.01,
)
axs["grid"].set_xlabel("Net Ice Thickness Change (m)")
axs["grid"].set_ylabel("Altimetry Range (deg)")
axs["grid"].set_title(
    "GMSL Error over Net Ice Thickness Change and Altimetry Range",
)
# cross section lines
altimetry_cs_1 = 75
altimetry_cs_0 = 66
net_ice_cs_0 = -10
net_ice_cs_1 = 10
axs["grid"].axhline(
    altimetry_cs_0,
    color="purple",
    linestyle="--",
    linewidth=1,
)
axs["grid"].axhline(
    altimetry_cs_1,
    color="purple",
    linestyle="-.",
    linewidth=1,
)
axs["grid"].axvline(
    net_ice_cs_0,
    color="g",
    linestyle="--",
    linewidth=1,
)
axs["grid"].axvline(
    net_ice_cs_1,
    color="g",
    linestyle="-.",
    linewidth=1,
)
# altimetry cross section, there is data at 66 deg, i want to have a smooth line through the data, i want the x axis to be flipped so negative is to the right
altimetry_data_0 = data[
    data["altimetry_range"] == altimetry_cs_0
].sort_values(by="net_ice_thickness_change")
axs["altimetry"].plot(
    altimetry_data_0["net_ice_thickness_change"],
    altimetry_data_0["error_mean_mm"],
    "--",
    color="purple",
)
altimetry_data_1 = data[
    data["altimetry_range"] == altimetry_cs_1
].sort_values(by="net_ice_thickness_change")
axs["altimetry"].plot(
    altimetry_data_1["net_ice_thickness_change"],
    altimetry_data_1["error_mean_mm"],
    "-.",
    color="purple",
)
# axs["altimetry"].set_xlabel("Net Ice Thickness Change (m)")
axs["altimetry"].set_ylabel("GMSL Error (mm)")

# net ice cross section
net_ice_data_0 = data[
    data["net_ice_thickness_change"] == net_ice_cs_0
].sort_values(by="altimetry_range")
axs["net_ice"].plot(
    net_ice_data_0["error_mean_mm"],
    net_ice_data_0["altimetry_range"],
    "--",
    color="g",
)
net_ice_data_1 = data[
    data["net_ice_thickness_change"] == net_ice_cs_1
].sort_values(by="altimetry_range")
axs["net_ice"].plot(
    net_ice_data_1["error_mean_mm"],
    net_ice_data_1["altimetry_range"],
    "-.",
    color="g",
)
axs["net_ice"].invert_xaxis()
# axs["net_ice"].set_ylabel("Altimetry Range (deg)")
axs["net_ice"].set_xlabel("GMSL Error (mm)")
# location of suptitle at bottom center
plt.suptitle(
    "\nGMSL Error Mean Cross Sections at Altimetry Range = 66° and Net Ice Thickness Change = 10 m\nIncludes contributions from Ice Change, Ocean Dynamic Topography, and Altimetry Error",
    y=0.0,
)
plt.savefig(
    "gmsl_error_over_ice_change_and_altimetry_range_mosaic.png",
    dpi=600,
    bbox_inches="tight",
)

# %%
