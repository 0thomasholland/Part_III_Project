"""
Run `copernicusmarine login` once before executing this script.

Downloads DUACS monthly SSH data, subsamples onto DH grids at lmax = 512, 256, 128, 64,
and saves them into DATA_DIR as compressed NPZ files (for inclusion in the pyslfp package).
Also computes the pointwise standard deviation of month-to-month differences for each resolution.
"""

from pathlib import Path

import copernicusmarine
import numpy as np
import pandas as pd
import pyshtools as sh
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Temporary cache for raw downloaded files — not committed to the package
OUTPUT_DIR = Path("./")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Package data directory — these files are committed and shipped with pyslfp
DATA_DIR = Path("../../src/pyslfp_extras/data/altimetry")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Downloaded monthly file (raw cache)
MONTHLY_FILE = OUTPUT_DIR / "duacs_monthly.nc"

# Derived yearly-averaged file (raw cache)
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

# Spherical harmonic truncation degrees to produce
LMAX_VALUES = [512, 256, 128, 64, 32]

# Preferred variables (in order of preference).
# The DT2024 reprocessing renamed some variables, so we try candidate names.
#   sla / msla  — sea level anomaly relative to 1993–2012 mean
#   adt / madt  — absolute dynamic topography (sla + mean dynamic topography)
SLA_CANDIDATES = ["sla", "msla"]
ADT_CANDIDATES = ["adt", "madt"]


# ---------------------------------------------------------------------------
# Helper: resolve variable names present in the remote dataset
# ---------------------------------------------------------------------------


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
        return []


# ---------------------------------------------------------------------------
# Helper: detect which SSH variable is present in an open dataset
# ---------------------------------------------------------------------------


def detect_ssh_variable(ds: xr.Dataset) -> str:
    """Return the name of the SLA (or ADT) variable in *ds*."""
    for name in SLA_CANDIDATES + ADT_CANDIDATES:
        if name in ds.data_vars:
            return name
    raise RuntimeError(
        f"None of the expected SSH variables found in dataset. "
        f"Available: {list(ds.data_vars)}"
    )


# ---------------------------------------------------------------------------
# Step 1: Download monthly data (skip if already present)
# ---------------------------------------------------------------------------


def download_monthly() -> None:
    """Download the DUACS monthly file if it does not already exist."""
    if MONTHLY_FILE.exists():
        print(
            f"Monthly file already exists: {MONTHLY_FILE} — skipping download."
        )
        return

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
# Step 2: Compute yearly averages and save to raw cache
# ---------------------------------------------------------------------------


def compute_annual() -> None:
    """Resample the monthly file to annual means and save to raw cache."""
    if ANNUAL_FILE.exists():
        print(
            f"Annual file already exists: {ANNUAL_FILE} — skipping computation."
        )
        return

    print("Computing yearly averages ...")
    ds = xr.open_dataset(MONTHLY_FILE)
    print(
        f"  Monthly dataset variables: {list(ds.data_vars)}"
    )

    ds_annual = ds.resample(time="YE").mean()

    # Relabel time coordinate to Jan 1 of each year
    ds_annual["time"] = pd.to_datetime(
        [
            f"{y}-01-01"
            for y in ds_annual.time.dt.year.values
        ]
    )

    ds_annual.attrs = ds.attrs
    ds_annual.attrs["temporal_resolution"] = (
        "annual mean (computed from monthly)"
    )
    ds_annual.attrs["source_dataset"] = DATASET_ID

    ds_annual.to_netcdf(ANNUAL_FILE)
    print(f"Annual file written: {ANNUAL_FILE}")
    print(
        f"  Years covered: {int(ds_annual.time.dt.year[0])} – "
        f"{int(ds_annual.time.dt.year[-1])}"
    )


# ---------------------------------------------------------------------------
# Helper: interpolate one 2-D SSH snapshot onto a pyshtools DH grid
# ---------------------------------------------------------------------------


def _ssh_to_shgrid(
    ssh_values: np.ndarray,
    src_lats: np.ndarray,
    src_lons: np.ndarray,
    lmax: int,
) -> sh.SHGrid:
    """
    Interpolate a 2-D SSH array (lat × lon) onto a pyshtools DH grid at *lmax*,
    expand to spherical harmonics, and return the re-expanded SHGrid.

    The DH sampling (sampling=1) is used to match FingerPrint grids: it gives a
    grid of shape (2*(lmax+1)-1, 2*(lmax+1)-1) with extend=True.

    Parameters
    ----------
    ssh_values : np.ndarray, shape (n_lat, n_lon)
        SSH data on the source regular grid.  NaN = masked ocean boundary.
    src_lats : np.ndarray
        Latitude axis of *ssh_values* (degrees, ascending or descending).
    src_lons : np.ndarray
        Longitude axis of *ssh_values* (degrees, 0 – 360).
    lmax : int
        Target spherical harmonic truncation degree.

    Returns
    -------
    sh.SHGrid
        Band-limited SSH field on a DH2 grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    # pyshtools lats run 90 → −90; lons run 0 → 360 (with extend=True → 0…360)
    target = sh.SHGrid.from_zeros(
        lmax, grid="DH", sampling=1, extend=True
    )
    tgt_lats = target.lats()  # descending: +90 … −90
    tgt_lons = target.lons()  # ascending:   0  …  360

    # Build interpolator — handle NaNs by filling with 0 (open ocean only)
    ssh_filled = np.where(
        np.isnan(ssh_values), 0.0, ssh_values
    )

    # Wrap longitudes to 0–360 and ensure ascending order
    src_lons_wrapped = np.mod(src_lons, 360.0)
    lon_order = np.argsort(src_lons_wrapped)
    src_lons_wrapped = src_lons_wrapped[lon_order]
    ssh_filled = ssh_filled[:, lon_order]

    # RegularGridInterpolator requires ascending lat axis
    if src_lats[0] > src_lats[-1]:
        src_lats = src_lats[::-1]
        ssh_filled = ssh_filled[::-1, :]

    interp = RegularGridInterpolator(
        (src_lats, src_lons_wrapped),
        ssh_filled,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Evaluate on the target grid
    tgt_lon_grid, tgt_lat_grid = np.meshgrid(
        tgt_lons, tgt_lats
    )
    points = np.column_stack(
        [tgt_lat_grid.ravel(), tgt_lon_grid.ravel()]
    )
    interp_vals = interp(points).reshape(tgt_lat_grid.shape)

    target.data = interp_vals

    # Round-trip through SH to enforce band-limit
    coeffs = target.expand(normalization="ortho", csphase=1)
    grid_out = coeffs.expand(grid="DH", extend=True)

    return grid_out


# ---------------------------------------------------------------------------
# Step 3: Subsample onto multi-resolution grids and compute monthly-diff std
# ---------------------------------------------------------------------------


def process_resolutions() -> None:
    """
    For each lmax in LMAX_VALUES:
      1. Subsample every monthly SSH snapshot onto a DH grid (sampling=1).
      2. Save the full monthly time-series as a NetCDF file in DATA_DIR.
      3. Compute month-to-month differences and their pointwise std.
      4. Save the std field to DATA_DIR.

    Output files (per lmax L):
      DATA_DIR/sla_monthly_lmax{L}.npz   — monthly SLA on DH grid
      DATA_DIR/sla_diff_std_lmax{L}.npz  — pointwise std of monthly differences
    """
    print("\nLoading monthly dataset ...")
    ds = xr.open_dataset(MONTHLY_FILE)
    ssh_var = detect_ssh_variable(ds)
    print(f"  SSH variable: '{ssh_var}'")

    # Source coordinate axes
    src_lats = ds["latitude"].values  # (n_lat,)
    src_lons = ds["longitude"].values  # (n_lon,)
    times = ds["time"].values  # (n_time,)
    n_time = len(times)
    print(f"  Time steps: {n_time}")

    for lmax in LMAX_VALUES:
        monthly_file = (
            DATA_DIR / f"sla_monthly_lmax{lmax}.npz"
        )
        std_file = DATA_DIR / f"sla_diff_std_lmax{lmax}.npz"

        if monthly_file.exists() and std_file.exists():
            print(
                f"\nlmax={lmax}: both output files exist — skipping."
            )
            continue

        print(f"\nProcessing lmax={lmax} ...")

        # Target grid shape for a DH grid: (2*(lmax+1)-1, 2*(lmax+1)-1) with extend
        target_template = sh.SHGrid.from_zeros(
            lmax, grid="DH", sampling=1, extend=True
        )
        tgt_lats = target_template.lats()
        tgt_lons = target_template.lons()
        n_lat_tgt = len(tgt_lats)
        n_lon_tgt = len(tgt_lons)

        # Pre-allocate storage: (n_time, n_lat, n_lon)
        data_cube = np.zeros(
            (n_time, n_lat_tgt, n_lon_tgt), dtype=np.float32
        )

        for t_idx in range(n_time):
            ssh_slice = (
                ds[ssh_var].isel(time=t_idx).values
            )  # (n_lat, n_lon)
            grid = _ssh_to_shgrid(
                ssh_slice, src_lats, src_lons, lmax
            )
            data_cube[t_idx] = grid.data.astype(np.float32)

            if (t_idx + 1) % 12 == 0:
                print(
                    f"  Processed {t_idx + 1}/{n_time} months ..."
                )

        # ------------------------------------------------------------------
        # Save monthly time-series
        # ------------------------------------------------------------------
        if not monthly_file.exists():
            np.savez_compressed(
                monthly_file,
                sla=data_cube,
                time=times,
                lat=tgt_lats,
                lon=tgt_lons,
                source_dataset=DATASET_ID,
                lmax=lmax,
                grid_type="DH (pyshtools sampling=1)",
                description=(
                    "DUACS monthly SLA subsampled onto a pyshtools "
                    f"Driscoll-Healy grid with lmax={lmax}."
                ),
            )
            print(f"  Saved monthly file: {monthly_file}")
        else:
            print(
                f"  Monthly file exists, loading for diff computation ..."
            )
            with np.load(monthly_file) as npz:
                data_cube = npz["sla"]

        # ------------------------------------------------------------------
        # Monthly differences and their pointwise standard deviation
        # ------------------------------------------------------------------
        if not std_file.exists():
            print(
                "  Computing monthly differences and pointwise std ..."
            )

            # Shape: (n_time - 1, n_lat, n_lon)
            diffs = np.diff(data_cube, axis=0)

            # Pointwise std across all monthly differences: (n_lat, n_lon)
            diff_std = diffs.std(
                axis=0, dtype=np.float64
            ).astype(np.float32)

            np.savez_compressed(
                std_file,
                sla_diff_std=diff_std,
                lat=tgt_lats,
                lon=tgt_lons,
                source_dataset=DATASET_ID,
                lmax=lmax,
                grid_type="DH (pyshtools sampling=1)",
                n_differences=n_time - 1,
                description=(
                    "Pointwise std of month-to-month SLA differences on a "
                    f"pyshtools DH grid with lmax={lmax}. "
                    f"Computed from {n_time - 1} consecutive monthly pairs."
                ),
            )
            print(f"  Saved std file: {std_file}")

    print("\nAll resolutions processed.")


# ---------------------------------------------------------------------------
# Step 4: Quick sanity check
# ---------------------------------------------------------------------------


def sanity_check() -> None:
    """Print a brief summary of all output files in DATA_DIR."""
    print("\n" + "=" * 60)
    print("Sanity check — DATA_DIR contents:")
    for npz_file in sorted(DATA_DIR.glob("*.npz")):
        size_mb = npz_file.stat().st_size / 1e6
        print(f"\n  {npz_file.name}  ({size_mb:.1f} MB)")
        with np.load(npz_file) as npz:
            keys = list(npz.keys())
            print(f"    Keys: {keys}")
            for key in keys:
                val = npz[key]
                if hasattr(val, "shape"):
                    print(f"      {key}: shape={val.shape}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_monthly()
    compute_annual()
    process_resolutions()
    sanity_check()
