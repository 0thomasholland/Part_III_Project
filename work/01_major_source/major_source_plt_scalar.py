# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("ticks")
sns.color_palette("colorblind")
sns.set_context("paper")


# %% import data

data = np.load("major_source_altimetry_errors_scalar.npz")

latitudes = data["latitudes"]

gis_errors = data["gis_errors"] * 100
eais_errors = data["eais_errors"] * 100
wais_errors = data["wais_errors"] * 100

gis_errors_abs = np.abs(gis_errors)
eais_errors_abs = np.abs(eais_errors)
wais_errors_abs = np.abs(wais_errors)

print(data)

# %% plot data

plt.figure(figsize=(10, 6))
plt.plot(
    latitudes,
    gis_errors,
    label="GIS Altimetry GMSL % Error",
    color="tab:blue",
)
plt.plot(
    latitudes,
    eais_errors,
    label="EAIS Altimetry GMSL % Error",
    color="tab:orange",
)
plt.plot(
    latitudes,
    wais_errors,
    label="WAIS Altimetry GMSL % Error",
    color="tab:green",
)
plt.axhline(0, color="black", linestyle="-", linewidth=1)
plt.axvline(66, color="red", linestyle="--", linewidth=1)
plt.xlabel("Latitude (degrees)")

plt.ylabel("Relative Error (%) [true - estimated / true]")

plt.title(
    "Altimetry GMSL Estimation Errors from Major Ice Sheet Sources"
)
plt.legend()
plt.grid()
plt.savefig(
    "major_source_altimetry_errors_scalar.png", dpi=600
)
plt.show()

# %%
# same as above but with seaborn, using pandas dataframe

df = pd.DataFrame(
    {
        "Latitude": np.concatenate(
            [latitudes, latitudes, latitudes]
        ),
        "Relative Error (%)": np.concatenate(
            [gis_errors, eais_errors, wais_errors]
        ),
        "Source": np.concatenate(
            [
                ["GrIS"] * len(latitudes),
                ["EAIS"] * len(latitudes),
                ["WAIS"] * len(latitudes),
            ]
        ),
    }
)

plt.figure(figsize=(6.5, 4))
sns.lineplot(
    data=df,
    x="Latitude",
    y="Relative Error (%)",
    hue="Source",
    palette="colorblind",
)
plt.axhline(0, color="black", linestyle="-", linewidth=1)
plt.legend(title="Source")
plt.axvline(
    66,
    color="red",
    linestyle="--",
    linewidth=1,
    label="Typical altimetry range",
)
# fill between x values 60 to 75 with red 10% transparent
plt.fill_between(
    x=[60, 75],
    y1=-10,
    y2=10,
    color="red",
    alpha=0.1,
    # label="Typical altimetry range",
)
plt.ylim(-10, 4)
plt.legend()
plt.xlabel("Latitude (degrees)")
plt.ylabel("Relative Error (%) [(true - estimated) / true]")
plt.title(
    "Altimetry GMSL Estimation Errors from Major Ice Sheet Sources"
)
plt.grid()
plt.savefig(
    "figures/major_source_altimetry_errors_scalar.png",
    dpi=600,
)
plt.show()
