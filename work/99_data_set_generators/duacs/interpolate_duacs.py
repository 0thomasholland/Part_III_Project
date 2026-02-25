# %%
import os

import dill
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from pyslfp import FingerPrint, IceModel

# %%

lmax = 512
output_dir = f"shgrids_lmax{lmax}"
dataset = "duacs_monthly.nc"
os.makedirs(output_dir, exist_ok=True)

# %%

# Precompute target grid coords once
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)
domain = fp_op.domain
target_grid = domain.zero()
target_lats = target_grid.lats()
target_lons = target_grid.lons()

# Precompute valid mask
ds = xr.open_dataset(dataset)
sla = ds["sla"]
valid_mask = (sla.count("time") >= 20).compute()
ds.close()

# %%


def process_timestep(
    i,
    lmax,
    target_lats,
    target_lons,
    valid_mask,
    output_dir,
):
    out_path = os.path.join(
        output_dir, f"shgrid_{i:04d}.pkl"
    )
    if os.path.exists(out_path):
        return f"Skipped {i} (already exists)"

    # Each worker opens the dataset independently
    ds = xr.open_dataset(dataset)
    sla = ds["sla"]

    sla_pair = sla.isel(time=slice(i, i + 2)).load()
    ds.close()

    sla_pair = sla_pair.where(valid_mask)
    sla_diff_i = sla_pair.diff("time").isel(time=0)

    lon = sla_diff_i["longitude"]
    if float(lon.min()) < 0:
        sla_diff_i = sla_diff_i.assign_coords(
            longitude=((lon + 360.0) % 360.0)
        ).sortby("longitude")

    target_lats_da = xr.DataArray(
        target_lats, dims=("latitude",)
    )
    target_lons_da = xr.DataArray(
        target_lons, dims=("longitude",)
    )
    sla_interp = sla_diff_i.interp(
        latitude=target_lats_da, longitude=target_lons_da
    )

    # Rebuild FingerPrint in worker (not picklable, so can't pass from parent)
    from pyslfp import FingerPrint, IceModel

    fp = FingerPrint(lmax=lmax)
    fp.set_state_from_ice_ng(
        version=IceModel.ICE7G, date=0.0
    )
    shgrid = fp.zero_grid()

    data = np.nan_to_num(sla_interp.values, nan=0.0)
    shgrid.data[:, :] = data

    with open(out_path, "wb") as f:
        dill.dump(shgrid, f)

    return f"Saved {i}"


# %%
#
time_indices = list(range(0, sla.sizes["time"] - 1, 3))

results = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_timestep)(
        i,
        lmax,
        target_lats,
        target_lons,
        valid_mask,
        output_dir,
    )
    for i in time_indices
)
# %%
for r in results:
    print(r)
