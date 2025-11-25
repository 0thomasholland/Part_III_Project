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

# %%

# linear model of error_mean ~net_ice_thickness_change + net_ice_thickness_change:altimetry_range

formula = "error_mean ~ net_ice_thickness_change + net_ice_thickness_change:altimetry_range"
model = ols(formula, data=data).fit()
print("Linear regression results for interaction model:")
print(model.summary())

# %%

data["error_mean_mm"] = data["error_mean"] * 1000

# %%

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
    figsize=(10, 8),
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
    cmap="bwr",
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
altimetry_cs = 66
net_ice_cs = 10
axs["grid"].axhline(
    altimetry_cs,
    color="k",
    linestyle="--",
    linewidth=1,
)
axs["grid"].axvline(
    net_ice_cs,
    color="k",
    linestyle="--",
    linewidth=1,
)
# altimetry cross section, there is data at 66 deg, i want to have a smooth line through the data, i want the x axis to be flipped so negative is to the right
altimetry_data = data[
    data["altimetry_range"] == altimetry_cs
].sort_values(by="net_ice_thickness_change")
axs["altimetry"].plot(
    altimetry_data["net_ice_thickness_change"],
    altimetry_data["error_mean_mm"],
    "-o",
    color="k",
)
# axs["altimetry"].set_xlabel("Net Ice Thickness Change (m)")
axs["altimetry"].set_ylabel("GMSL Error (mm)")
# net ice cross section
net_ice_data = data[
    data["net_ice_thickness_change"] == net_ice_cs
].sort_values(by="altimetry_range")
axs["net_ice"].plot(
    net_ice_data["error_mean_mm"],
    net_ice_data["altimetry_range"],
    "-o",
    color="k",
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
