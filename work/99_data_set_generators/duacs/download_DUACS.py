"""
Run `copernicusmarine login` once before executing this script,
"""

from pathlib import Path

import copernicusmarine
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Output directory — change this to suit your project layout
OUTPUT_DIR = Path("./")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Downloaded monthly file
MONTHLY_FILE = OUTPUT_DIR / "duacs_monthly.nc"

# Derived yearly-averaged file
ANNUAL_FILE = OUTPUT_DIR / "duacs_annual.nc"

# Dataset — reprocessed multi-year monthly means at 0.125°
# Product:  SEALEVEL_GLO_PHY_L4_MY_008_047
DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1M-m"

# Spatial extent: match pyslfp altimetry_projection bounds (±66° latitude)
# Longitude in 0–360° convention (consistent with pyshtools)
LAT_MIN = -66.0
LAT_MAX = 66.0
LON_MIN = 0.0
LON_MAX = 360.0

# Full reprocessed record
TIME_START = "1993-01-01"
TIME_END = "2024-12-31"

# Preferred variables (in order of preference).
# The DT2024 reprocessing renamed some variables, so we try candidate names
# and only request those that actually exist in the dataset.
#   sla / msla  — sea level anomaly relative to 1993–2012 mean
#   adt / madt  — absolute dynamic topography (sla + mean dynamic topography)
SLA_CANDIDATES = ["sla", "msla"]
ADT_CANDIDATES = ["adt", "madt"]


def resolve_variables(dataset_id: str) -> list[str]:
    """
    Inspect the dataset catalogue entry and return whichever SLA and ADT
    variable names are actually present, falling back gracefully.
    """
    print("Inspecting dataset variables ...")
    try:
        ds_remote = copernicusmarine.open_dataset(
            dataset_id=dataset_id
        )
        available = set(ds_remote.data_vars)
        print(f"  Available variables: {sorted(available)}")

        variables = []
        for candidates in (SLA_CANDIDATES, ADT_CANDIDATES):
            for name in candidates:
                if name in available:
                    variables.append(name)
                    break
            else:
                print(
                    f"  Warning: none of {candidates} found — skipping."
                )

        print(f"  Requesting: {variables}")
        return variables

    except Exception as exc:
        print(
            f"  Could not inspect dataset ({exc}); will download all variables."
        )
        return []  # empty list → copernicusmarine downloads everything


# ---------------------------------------------------------------------------
# Step 1: Download monthly data (skip if already present)
# ---------------------------------------------------------------------------

if MONTHLY_FILE.exists():
    print(
        f"Monthly file already exists: {MONTHLY_FILE} — skipping download."
    )
else:
    variables = resolve_variables(DATASET_ID)

    print(
        "Downloading DUACS monthly SSH data from CMEMS ..."
    )
    copernicusmarine.subset(
        dataset_id=DATASET_ID,
        variables=variables if variables else None,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        start_datetime=TIME_START,
        end_datetime=TIME_END,
        output_filename=MONTHLY_FILE.name,
        output_directory=str(MONTHLY_FILE.parent),
        skip_existing=True,
    )
    print(f"Download complete: {MONTHLY_FILE}")

# ---------------------------------------------------------------------------
# Step 2: Compute yearly averages and save
# ---------------------------------------------------------------------------

if ANNUAL_FILE.exists():
    print(
        f"Annual file already exists: {ANNUAL_FILE} — skipping computation."
    )
else:
    print("Computing yearly averages ...")
    ds = xr.open_dataset(MONTHLY_FILE)
    print(
        f"  Monthly dataset variables: {list(ds.data_vars)}"
    )

    # Resample to annual means (YE = year-end label)
    ds_annual = ds.resample(time="YE").mean()

    # Relabel time coordinate to Jan 1 of each year (cleaner for downstream use)
    import pandas as pd

    ds_annual["time"] = pd.to_datetime(
        [
            f"{y}-01-01"
            for y in ds_annual.time.dt.year.values
        ]
    )

    # Propagate global attributes and add a processing note
    ds_annual.attrs = ds.attrs
    ds_annual.attrs["temporal_resolution"] = (
        "annual mean (computed from monthly)"
    )
    ds_annual.attrs["source_dataset"] = DATASET_ID

    # Write to disk
    ds_annual.to_netcdf(ANNUAL_FILE)
    print(f"Annual file written: {ANNUAL_FILE}")
    print(
        f"  Years covered: {int(ds_annual.time.dt.year[0])} – "
        f"{int(ds_annual.time.dt.year[-1])}"
    )
    print(f"  Variables: {list(ds_annual.data_vars)}")
    print(f"  Grid: {ds_annual.dims}")

# ---------------------------------------------------------------------------
# Step 3: Quick sanity check
# ---------------------------------------------------------------------------

print("\nSanity check on annual file:")
ds_check = xr.open_dataset(ANNUAL_FILE)
print(ds_check)
