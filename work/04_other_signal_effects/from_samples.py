# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dill
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pygeoinf import GaussianMeasure, HilbertSpace
from pygeoinf.symmetric_space.circle import Sobolev
from pyshtools import SHGrid
from pyslfp import FingerPrint, IceModel, plot

# %%
lmax = 128
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)
print("loading data")
ds = xr.open_dataset("../../data/duacs/duacs_monthly.nc")
# ds = xr.open_dataset("data/duacs/duacs_monthly.nc")
sla = ds["sla"]  # shape: (time, lat, lon)

# Mask to well-sampled points
valid_count = sla.count("time")
sla_masked = sla.where(valid_count >= 20)
print("data loaded and masked")

# %%
# --- monthly differences ---

sla_diff = sla_masked.diff("time")
# select random subset of 40 sla's for testing
# sla_diff = sla_diff.isel(time=slice(0, 40))

# Regrid to the FingerPrint grid so SHGrid metadata matches fp_op.domain
target_grid = fp.zero_grid()
target_lats = xr.DataArray(
    target_grid.lats(), dims=("latitude",)
)
target_lons = xr.DataArray(
    target_grid.lons(), dims=("longitude",)
)

# Make sure longitude is 0–360 if needed
lon = sla_diff["longitude"]
if float(lon.min()) < 0:
    sla_diff = sla_diff.assign_coords(
        longitude=((lon + 360.0) % 360.0)
    ).sortby("longitude")

sla_diff = sla_diff.interp(
    latitude=target_lats, longitude=target_lons
)


# %%

# for each month generate a SHGrid object with with the lats and longs and SLA data

shgrids = []
template = fp.zero_grid()

for i in range(sla_diff.sizes["time"]):
    shgrid = template.copy()
    shgrid.data[:, :] = sla_diff.isel(time=i).values
    shgrids.append(shgrid)

print(len(shgrids))

measure = GaussianMeasure.from_samples(
    fp_op.domain, shgrids
)
# save measure
# %%

with open(f"odt{lmax}.pkl", "wb") as f:
    dill.dump(measure, f)

# %%

plot(
    measure.sample() * 1000,
)
