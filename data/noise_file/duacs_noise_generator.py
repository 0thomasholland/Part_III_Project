import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

ds = xr.open_dataset("../../data/duacs/duacs_annual.nc")
sla = ds["sla"]

valid_count = sla.count("time")
sla_masked = sla.where(valid_count >= 20)

# --- Compute inter-annual RMS ---
rms_annual = (
    np.sqrt((sla_masked.diff("time") ** 2).mean("time"))
    * 1000
)  # mm

# --- Blur to smooth coastal artefacts and small-scale noise ---
rms_raw = rms_annual.values.copy()
nan_mask = np.isnan(rms_raw)
rms_filled = rms_raw.copy()
rms_filled[nan_mask] = 0.0

rms_blurred = gaussian_filter(rms_filled, sigma=20)
rms_blurred[nan_mask] = np.nan

# --- Normalise so mean = 1 (used as spatial multiplier) ---
non_ice_ssh_variability = xr.DataArray(
    rms_blurred / np.nanmean(rms_blurred),
    coords=rms_annual.coords,
    dims=rms_annual.dims,
    attrs={
        "long_name": "Non-ice SSH variability spatial structure",
        "description": "Empirically derived from DUACS inter-annual RMS, normalised to mean=1",
        "units": "dimensionless",
    },
)

# --- Plot ---
fig = plt.figure(figsize=(14, 5))

for i, (data, title) in enumerate(
    [
        (rms_annual, "Raw inter-annual RMS (mm)"),
        (
            non_ice_ssh_variability,
            "non_ice_ssh_variability (normalised, mean=1)",
        ),
    ]
):
    vals = data.values.flatten()
    vals = vals[~np.isnan(vals)]

    ax = fig.add_subplot(
        1, 2, i + 1, projection=ccrs.Robinson()
    )
    ax.add_feature(cfeature.LAND, color="lightgray")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
    im = ax.pcolormesh(
        sla_masked.longitude,
        sla_masked.latitude,
        data,
        cmap="YlOrRd",
        vmin=0,
        vmax=np.percentile(vals, 95),
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        label="mm" if i == 0 else "relative to mean",
    )
    ax.set_title(title)

plt.tight_layout()
plt.show()

# --- Save for use in ODT model ---
non_ice_ssh_variability.to_netcdf(
    "non_ice_ssh_variability.nc"
)
print(
    f"Range: {float(non_ice_ssh_variability.min()):.2f} to {float(non_ice_ssh_variability.max()):.2f}"
)
print(f"Mean: {float(non_ice_ssh_variability.mean()):.2f}")
