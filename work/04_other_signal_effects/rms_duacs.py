import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# ds = xr.open_dataset("../../data/duacs/duacs_annual.nc")
ds = xr.open_dataset("data/duacs/duacs_annual.nc")
sla = ds["sla"]  # shape: (time, lat, lon)

# Mask to well-sampled points
valid_count = sla.count("time")
sla_masked = sla.where(valid_count >= 20)

# --- Annual differences ---
sla_diff = sla_masked.diff("time")
rms_annual = (
    np.sqrt((sla_diff**2).mean("time")) * 1000
)  # mm
rms_annual_vals = rms_annual.values.flatten()
rms_annual_vals = rms_annual_vals[
    ~np.isnan(rms_annual_vals)
]

# --- 1993 to 2019 difference ---
sla_long = (
    sla_masked.sel(time="2019-01-01", method="nearest")
    - sla_masked.sel(time="1993-01-01", method="nearest")
) * 1000  # mm
rms_long_vals = sla_long.values.flatten()
rms_long_vals = rms_long_vals[~np.isnan(rms_long_vals)]

# --- Plot ---
fig = plt.figure(figsize=(14, 10))

# ── Annual ──────────────────────────────────────────────
ax1_map = fig.add_subplot(
    2, 2, 1, projection=ccrs.Robinson()
)
ax1_map.add_feature(cfeature.LAND, color="lightgray")
ax1_map.add_feature(cfeature.COASTLINE, linewidth=0.4)
im1 = ax1_map.pcolormesh(
    sla_masked.longitude,
    sla_masked.latitude,
    rms_annual,
    cmap="YlOrRd",
    vmin=0,
    vmax=np.percentile(rms_annual_vals, 95),
    transform=ccrs.PlateCarree(),
)
plt.colorbar(
    im1,
    ax=ax1_map,
    orientation="horizontal",
    pad=0.05,
    label="RMS (mm)",
)
ax1_map.set_title("Inter-annual SLA variability (RMS)")

ax1_hist = fig.add_subplot(2, 2, 3)
ax1_hist.hist(
    rms_annual_vals,
    bins=60,
    edgecolor="k",
    linewidth=0.3,
    color="steelblue",
)
ax1_hist.axvline(
    np.mean(rms_annual_vals),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(rms_annual_vals):.1f} mm",
)
ax1_hist.axvline(
    np.median(rms_annual_vals),
    color="orange",
    linestyle="--",
    label=f"Median: {np.median(rms_annual_vals):.1f} mm",
)
ax1_hist.set_xlabel("RMS of year-to-year SLA change (mm)")
ax1_hist.set_ylabel("Number of grid points")
ax1_hist.set_title(
    "Inter-annual SLA variability distribution"
)
ax1_hist.legend()

# ── Long-term ────────────────────────────────────────────
vmax_long = np.percentile(np.abs(rms_long_vals), 95)

ax2_map = fig.add_subplot(
    2, 2, 2, projection=ccrs.Robinson()
)
ax2_map.add_feature(cfeature.LAND, color="lightgray")
ax2_map.add_feature(cfeature.COASTLINE, linewidth=0.4)
im2 = ax2_map.pcolormesh(
    sla_masked.longitude,
    sla_masked.latitude,
    sla_long,
    cmap="RdBu_r",
    vmin=-vmax_long,
    vmax=vmax_long,
    transform=ccrs.PlateCarree(),
)
plt.colorbar(
    im2,
    ax=ax2_map,
    orientation="horizontal",
    pad=0.05,
    label="SLA change (mm)",
)
ax2_map.set_title("Long-term SLA change 1993–2019")

ax2_hist = fig.add_subplot(2, 2, 4)
ax2_hist.hist(
    rms_long_vals,
    bins=60,
    edgecolor="k",
    linewidth=0.3,
    color="seagreen",
)
ax2_hist.axvline(
    np.mean(rms_long_vals),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(rms_long_vals):.1f} mm",
)
ax2_hist.axvline(
    np.median(rms_long_vals),
    color="orange",
    linestyle="--",
    label=f"Median: {np.median(rms_long_vals):.1f} mm",
)
ax2_hist.set_xlabel("SLA difference 2019 minus 1993 (mm)")
ax2_hist.set_ylabel("Number of grid points")
ax2_hist.set_title("Long-term SLA change distribution")
ax2_hist.legend()

plt.suptitle("DUACS SLA Variability", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()
