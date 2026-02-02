# %%
import matplotlib.pyplot as plt
import numpy as np

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
    label="GIS Altimetry GMSL Error",
    color="blue",
)
plt.plot(
    latitudes,
    eais_errors,
    label="EAIS Altimetry GMSL Error",
    color="orange",
)
plt.plot(
    latitudes,
    wais_errors,
    label="WAIS Altimetry GMSL Error",
    color="green",
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


# %% plot abs data

plt.figure(figsize=(10, 6))
plt.plot(
    latitudes,
    gis_errors_abs,
    label="GIS Altimetry GMSL Error",
    color="blue",
)
plt.plot(
    latitudes,
    eais_errors_abs,
    label="EAIS Altimetry GMSL Error",
    color="orange",
)
plt.plot(
    latitudes,
    wais_errors_abs,
    label="WAIS Altimetry GMSL Error",
    color="green",
)
plt.xlabel("Latitude (degrees)")

plt.ylabel("Relative Error (%) [true - estimated / true]")

plt.title(
    "Altimetry GMSL Estimation Errors from Major Ice Sheet Sources"
)
plt.legend()
plt.grid()
plt.savefig(
    "major_source_altimetry_errors_scalar_abs.png", dpi=600
)
plt.show()
