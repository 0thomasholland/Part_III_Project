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

# plot a field of error_mean as a smooth color grid over net_ice_thickness_change and altimetry_range, and show a cross section for altimetry range = 66 and net_ice_thickness_change = 10m, showing dotted lines on the grid for the cross section values, and plots of error vs the other variable on the side panels using subplot mosaic

x = data["net_ice_thickness_change"]
y = data["altimetry_range"]
z = data["error_mean"]
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x, y), z, (xi, yi), method="cubic")

fig, axs = plt.subplot_mosaic(
    [["net_ice", "grid"], [".", "altimetry"]],
    figsize=(7, 6),
    width_ratios=(1, 4),
    height_ratios=(4, 1),
    layout="constrained",
)

c = axs["grid"].pcolormesh(
    xi,
    yi,
    zi,
    shading="gouraud",
    cmap="bwr",
    norm=mpl.colors.TwoSlopeNorm(
        vcenter=0,
        vmin=-data["error_mean"].max(),
        vmax=data["error_mean"].max(),
    ),
)
# colourbar plotted to the right of the entire mosaic
fig.colorbar(c, ax=axs["grid"], label="GMSL Error Mean (m)", pad=0.01)
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
    altimetry_data["error_mean"],
    "-o",
)
axs["altimetry"].set_xlabel("Net Ice Thickness Change (m)")
axs["altimetry"].set_ylabel("GMSL Error (m)")

# net ice cross section
net_ice_data = data[
    data["net_ice_thickness_change"] == net_ice_cs
].sort_values(by="altimetry_range")
axs["net_ice"].plot(
    net_ice_data["error_mean"],
    net_ice_data["altimetry_range"],
    "-o",
)
axs["net_ice"].invert_xaxis()
axs["net_ice"].set_ylabel("Altimetry Range (deg)")
axs["net_ice"].set_xlabel("GMSL Error (m)")
