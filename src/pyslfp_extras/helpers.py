import numpy as np
from joblib import Parallel, delayed
from pyslfp import FingerPrint


def get_ocean_point_coordinates(
    finger_print: FingerPrint,
    point_degree_spacing: float = 5.0,
    altimetry_latitude_range: float = 66.0,
    parallel_workers: None | int = None,
) -> tuple[list[float], list[float]]:
    """
    Returns the latitude and longitude coordinates of ocean points on the Earth's
    surface, as determined by the provided `FingerPrint`. Points are selected
    based on the ocean and altimetry projections at the specified degree spacing.

    Parameters
    ----------
    finger_print : FingerPrint
        The fingerprint object containing ocean and altimetry projections
    point_degree_spacing : float, optional
        Spacing between evaluation points in degrees, by default 5.0

    Returns
    -------
    tuple[list[float], list[float]]
        A tuple of (latitudes, longitudes) for ocean points
    """
    mask = (
        finger_print.ocean_projection(value=0)
        * finger_print.altimetry_projection(
            latitude_max=altimetry_latitude_range,
            latitude_min=-altimetry_latitude_range,
            value=0,
        )
    ).to_array()
    nlat, nlon = mask.shape

    mask_lats = np.linspace(90, -90, nlat)
    mask_lons = (
        np.linspace(0, 360, nlon, endpoint=True)
        if nlon > nlat
        else np.linspace(0, 360, nlon, endpoint=False)
    )

    target_lats = np.arange(
        90,
        -90 - point_degree_spacing,
        -point_degree_spacing,
    )
    target_lons = np.arange(0, 360, point_degree_spacing)

    def is_ocean_point(lat, lon):
        mask_lat_idx = np.argmin(np.abs(mask_lats - lat))
        mask_lon_idx = np.argmin(np.abs(mask_lons - lon))

        if mask[mask_lat_idx, mask_lon_idx] == 1:
            return (lat, lon)
        return None

    if parallel_workers is not None:
        # Use threads to avoid copying large arrays across processes.
        results = Parallel(n_jobs=parallel_workers, prefer="threads")(
            delayed(is_ocean_point)(lat, lon)
            for lat in target_lats
            for lon in target_lons
        )
        ocean_coords = [coord for coord in results if coord is not None]
    else:
        ocean_coords = []
        for lat in target_lats:
            for lon in target_lons:
                coord = is_ocean_point(lat, lon)
                if coord is not None:
                    ocean_coords.append(coord)

    if not ocean_coords:
        return [], []

    ocean_lats, ocean_lons = zip(*ocean_coords)
    return list(ocean_lats), list(ocean_lons)
