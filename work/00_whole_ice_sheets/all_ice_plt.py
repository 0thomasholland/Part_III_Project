# %%
import matplotlib.pyplot as plt
import numpy as np

data = np.load("all_ice_sheets_altimetry_errors.npz")

# %%

latitudes = data["latitudes"]
numeric_errors = data["numeric_errors"]
relative_errors = data["relative_errors"]

# %%

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    latitudes,
    relative_errors,
    label="Relative Error",
    color="tab:blue",
)
ax1.set_xlabel("Latitude (˚)")
ax1.set_ylabel("Relative Error")
ax1.set_title(
    "Altimetry GMSL Estimation Errors for All Ice Sheets"
)

# %%

fig.savefig("all_ice_sheets_altimetry_errors.png", dpi=600)
