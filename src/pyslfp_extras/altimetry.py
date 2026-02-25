from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
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
        results = Parallel(
            n_jobs=parallel_workers, prefer="threads"
        )(
            delayed(is_ocean_point)(lat, lon)
            for lat in target_lats
            for lon in target_lons
        )
        ocean_coords = [
            coord for coord in results if coord is not None
        ]
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


def ocean_point_evaluation_operator(
    finger_print: FingerPrint,
    measurement_space: HilbertSpace,
    point_degree_spacing: float = 5.0,
    altimetry_latitude_range: float = 66.0,
    parallel_workers: None | int = None,
) -> LinearOperator:
    """
    Constructs a linear operator that evaluates the ocean surface height at
    specific points on the Earth's surface, as determined by the provided
    `FingerPrint`. The operator is designed to only evaluate points that are
    classified as ocean according to the `FingerPrint`'s ocean and altimetry
    projections.
    """
    ocean_coords = get_ocean_point_coordinates(
        finger_print,
        point_degree_spacing=point_degree_spacing,
        altimetry_latitude_range=altimetry_latitude_range,
        parallel_workers=parallel_workers,
    )

    ocean_coords = list(zip(*ocean_coords))

    _op = measurement_space.point_evaluation_operator(
        ocean_coords
    )
    return _op


def altimetry_error_gaussian_measure(
    measurement_space: Sobolev,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    finger_print: FingerPrint | None = None,
    altimetry_operator: LinearOperator | None = None,
) -> GaussianMeasure:
    """Create a Gaussian measure for altimetry observation errors on the SSH measurement space.

    Args:
        measurement_space: The SSH space (e.g. from sea_surface_height_operator.codomain).
        order: Sobolev kernel order. Default 1.5.
        length_scale: Correlation length scale. Default 0.005 * fp.mean_sea_floor_radius.
        amplitude: Pointwise standard deviation. Default 0.0001 / fp.length_scale.
        finger_print: FingerPrint instance used for default parameter scaling.
            Required if length_scale or amplitude are not provided.
        altimetry_operator: Optional operator for spatial masking (e.g. latitude bounds).

    Returns:
        A GaussianMeasure for altimetry observation errors.
    """
    if length_scale is None or amplitude is None:
        if finger_print is None:
            raise ValueError(
                "finger_print is required when length_scale or amplitude are not provided."
            )
        if length_scale is None:
            length_scale = (
                0.005 * finger_print.mean_sea_floor_radius
            )
        if amplitude is None:
            amplitude = 0.0001 / finger_print.length_scale

    base_measure = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, length_scale, amplitude
    )

    if altimetry_operator is not None:
        return base_measure.affine_mapping(
            operator=altimetry_operator
        )

    return base_measure
