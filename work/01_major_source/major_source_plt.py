# %%
import matplotlib.pyplot as plt
import numpy as np

# %% import data

data = np.load("major_source_altimetry_errors_scalar.npz")

latitudes = data["latitudes"]

gis_errors = data["gis_errors"] * 100
eais_errors = data["eais_errors"] * 100
wias_errors = data["wias_errors"] * 100

gis_errors_abs = np.abs(gis_errors)
eais_errors_abs = np.abs(eais_errors)
wias_errors_abs = np.abs(wias_errors)

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
    wias_errors,
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
    wias_errors_abs,
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
plt.show()
