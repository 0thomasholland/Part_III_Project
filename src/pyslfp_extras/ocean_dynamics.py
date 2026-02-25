import numpy as np
from pygeoinf import (
    GaussianMeasure,
    HilbertSpace,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyshtools import SHGrid
from pyslfp import (
    FingerPrint,
    ocean_projection_operator,
    remove_ocean_average_operator,
    sea_level_change_to_load_operator,
    sea_surface_height_operator,
    spatial_mutliplication_operator,
)


def non_ice_ssh_variability_field(
    finger_print: FingerPrint,
    use_dataset: bool = False,
    variability_path: str = "data/non_ice_ssh_variability.nc",
    use_synthetic: bool = False,
    base_multiplier: float = 1,
    point_multiplier: float = 20,
):
    return True


def dataset_non_ice_ssh_variability_field(
    finger_print: FingerPrint,
    variability_path: str = "data/non_ice_ssh_variability.nc",
) -> SHGrid:
    """Load empirically-derived non-ice SSH variability field from DUACS inter-annual RMS.

    This field captures spatial structure of SSH variability not attributable to
    ice-sheet driven sea level change or instrument error — i.e. ocean dynamics,
    steric effects, circulation changes etc. Derived from DUACS annual SLA data,
    blurred and normalised to mean=1 so it acts as a spatial multiplier.

    Args:
        finger_print: The FingerPrint instance (used for grid creation).
        variability_path: Path to the NetCDF file containing the normalised
            variability field (produced by compute_non_ice_ssh_variability.py).

    Returns:
        An SHGrid with normalised variability values (dimensionless, mean=1).
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    # Load the DUACS-derived variability field
    ds = xr.open_dataset(variability_path)
    # Find the data variable (should be the only non-coordinate variable)
    var_name = [v for v in ds.data_vars][0]
    da = ds[var_name]

    # Get source coordinates
    src_lats = da.latitude.values
    src_lons = da.longitude.values  # assumed [-180, 180]
    src_data = da.values  # (lat, lon)

    # Fill NaNs with 0 (land/ice — will be masked by ocean projection)
    src_data = np.where(np.isnan(src_data), 0.0, src_data)

    # Build interpolator
    interp = RegularGridInterpolator(
        (src_lats, src_lons),
        src_data,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    # Interpolate onto fingerprint grid
    lats = finger_print.lats()
    lons = finger_print.lons()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Convert fingerprint lons from [0, 360] to [-180, 180]
    lon_grid_converted = np.where(
        lon_grid > 180, lon_grid - 360, lon_grid
    )

    query_points = np.stack(
        [lat_grid.ravel(), lon_grid_converted.ravel()],
        axis=-1,
    )
    field = interp(query_points).reshape(lat_grid.shape)

    # Mask to ocean
    ocean_mask = finger_print.ocean_projection(
        value=0
    ).to_array()
    field = field * ocean_mask

    # Re-normalise after masking so ocean mean = 1
    ocean_mean = field[ocean_mask > 0].mean()
    if ocean_mean > 0:
        field = field / ocean_mean

    grid = finger_print.zero_grid()
    grid.data[:, :] = field
    return grid


def _old_non_ice_ssh_variability_variability_field(
    finger_print: FingerPrint,
    base_multiplier: float = 1,
    point_multiplier: float = 20,
) -> SHGrid:
    """Generate a synthetic spatial variability field for ODT based on observed patterns.

    Creates a field with variability values (in meters) reflecting where ODT variance
    is expected to be high (western boundary currents, ACC) vs low (deep basins).

    Args:
        finger_print: The FingerPrint instance (used for grid creation).
        base_multiplier: Uniform base amplitude across the ocean. Default 1.
        point_multiplier: Peak amplitude in major current systems. Default 20.

    Returns:
        An SHGrid with variability values in meters.
    """

    def _gaussian_blob(
        lat_grid, lon_grid, lat0, lon0, lat_width, lon_width
    ):
        """Smooth Gaussian-like spatial weight centered at (lat0, lon0)."""
        dlat = lat_grid - lat0
        dlon = lon_grid - lon0
        # Wrap longitude differences to [-180, 180]
        dlon = np.mod(dlon + 180, 360) - 180
        return np.exp(
            -0.5
            * (
                (dlat / lat_width) ** 2
                + (dlon / lon_width) ** 2
            )
        )

    def _latitude_band(lat_grid, lat_center, lat_width):
        """Smooth zonal band centered at lat_center."""
        return np.exp(
            -0.5
            * ((lat_grid - lat_center) / lat_width) ** 2
        )

    grid = finger_print.zero_grid()
    lats = finger_print.lats()
    lons = finger_print.lons()
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Start with base amplitude everywhere
    field = np.full_like(lat_grid, base_multiplier)

    # Gulf Stream
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(
            lat_grid, lon_grid, 35, 305, 20, 20
        )
    )
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(
            lat_grid, lon_grid, 45, 320, 20, 30
        )
    )
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(lat_grid, lon_grid, 35, 305, 8, 8)
    )
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(lat_grid, lon_grid, 45, 320, 8, 8)
    )

    # South America
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(
            lat_grid, lon_grid, -45, 305, 20, 30
        )
    )
    field += (
        point_multiplier
        * 0.3
        * _gaussian_blob(lat_grid, lon_grid, -45, 295, 6, 6)
    )

    # Kuroshio
    field += (
        point_multiplier
        * 0.6
        * _gaussian_blob(
            lat_grid, lon_grid, 35, 155, 10, 30
        )
    )

    # Agulhas
    field += (
        point_multiplier
        * 0.6
        * _gaussian_blob(
            lat_grid, lon_grid, -35, 35, 10, 15
        )
    )

    # East Australian Current
    field += (
        point_multiplier
        * 0.8
        * _gaussian_blob(
            lat_grid, lon_grid, -30, 150, 10, 15
        )
    )

    # Antarctic current blob
    field += (
        point_multiplier
        * 0.6
        * _gaussian_blob(
            lat_grid, lon_grid, -55, 180, 10, 60
        )
    )

    # Equatorial Pacific
    field += (
        point_multiplier
        * 0.7
        * _gaussian_blob(lat_grid, lon_grid, 10, 240, 5, 30)
    )

    # Indonesian Throughflow
    field += (
        point_multiplier
        * 0.6
        * _gaussian_blob(lat_grid, lon_grid, -5, 120, 5, 10)
    )

    # Afar
    field += (
        point_multiplier
        * 0.7
        * _gaussian_blob(lat_grid, lon_grid, 25, 40, 15, 15)
    )

    # Mask to ocean only
    ocean_mask = finger_print.ocean_projection(
        value=0
    ).to_array()
    field = field * ocean_mask

    grid.data[:, :] = field
    return grid


def non_ice_ssh_variability_gaussian_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    order: float = 1.5,
    length_scale: float | None = 5000,
    amplitude: float | None = 0.003,
    use_spatial_variability: bool = False,
    point_multiplier: float = 30,
    variability_path: str = "data/non_ice_ssh_variability.nc",
) -> GaussianMeasure:
    """Create a Gaussian measure for non-ice SSH variability as a height field on the load space.

    The measure is projected onto the ocean and constrained to have zero ocean average.
    Optional spatially-varying amplitude derived empirically from DUACS inter-annual RMS
    can be applied via non_ice_ssh_variability_field().

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        order: Sobolev kernel order. Default 1.5.
        length_scale: Correlation length scale in km. Default 5000.
        amplitude: Pointwise standard deviation. Default 0.003.
        use_spatial_variability: If True, apply empirically-derived spatial variability
            from non_ice_ssh_variability_field().
        point_multiplier: Scalar multiplier applied to the spatial variability field.
        variability_path: Path to the non_ice_ssh_variability NetCDF file.

    Returns:
        A GaussianMeasure for non-ice SSH variability on the load space.
    """
    load_space: HilbertSpace = finger_print_operator.domain

    base_measure = load_space.point_value_scaled_sobolev_kernel_gaussian_measure(
        order, length_scale, amplitude
    )

    ocean_proj = ocean_projection_operator(
        finger_print, load_space
    )
    remove_avg = remove_ocean_average_operator(
        finger_print, load_space
    )

    if use_spatial_variability:
        spatial_variability = non_ice_ssh_variability_field(
            finger_print,
            variability_path=variability_path,
        )
        spatial_op = spatial_mutliplication_operator(
            spatial_variability,
            load_space,
        )
        combined_op = remove_avg @ spatial_op @ ocean_proj
    else:
        combined_op = remove_avg @ ocean_proj

    return base_measure.affine_mapping(operator=combined_op)


def non_ice_ssh_variability_fingerprint_ssh_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    non_ice_ssh_variability_measure: GaussianMeasure
    | None = None,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    use_spatial_variability: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for the SSH contribution from ODT's fingerprint response.

    This represents only the gravitationally-induced sea surface height change caused
    by the ocean load redistribution from ODT, not the ODT height itself.

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        non_ice_ssh_variability_measure: Pre-built ODT measure. If None, one is created with the given parameters.
        order: Sobolev kernel order for ODT measure creation. Default 1.5.
        length_scale: Correlation length scale for ODT measure creation.
        amplitude: Pointwise standard deviation for ODT measure creation.
        use_spatial_variability: Whether to use spatial variability for ODT.

    Returns:
        A GaussianMeasure on the SSH space representing the fingerprint response to ODT.
    """
    if non_ice_ssh_variability_measure is None:
        non_ice_ssh_variability_measure = non_ice_ssh_variability_gaussian_measure(
            finger_print,
            finger_print_operator,
            order=order,
            length_scale=length_scale,
            amplitude=amplitude,
            use_spatial_variability=use_spatial_variability,
        )

    load_space: Sobolev = finger_print_operator.domain
    load_op = sea_level_change_to_load_operator(
        finger_print, load_space
    )
    ssh_op = sea_surface_height_operator(
        finger_print, finger_print_operator.codomain
    )

    fingerprint_ssh_op = (
        ssh_op @ finger_print_operator @ load_op
    )
    return non_ice_ssh_variability_measure.affine_mapping(
        operator=fingerprint_ssh_op
    )


def non_ice_ssh_variability_total_ssh_measure(
    finger_print: FingerPrint,
    finger_print_operator: LinearOperator,
    non_ice_ssh_variability_measure: GaussianMeasure
    | None = None,
    order: float = 1.5,
    length_scale: float | None = None,
    amplitude: float | None = None,
    use_spatial_variability: bool = False,
) -> GaussianMeasure:
    """Create a Gaussian measure for the total SSH contribution from ODT.

    This is the sum of the ODT height field itself and its gravitationally-induced
    fingerprint response in SSH (see sea_surface_height_operator docstring).

    Args:
        finger_print: The FingerPrint instance.
        finger_print_operator: Linear operator mapping loads to the response space.
        non_ice_ssh_variability_measure: Pre-built ODT measure. If None, one is created with the given parameters.
        order: Sobolev kernel order for ODT measure creation. Default 1.5.
        length_scale: Correlation length scale for ODT measure creation.
        amplitude: Pointwise standard deviation for ODT measure creation.
        use_spatial_variability: Whether to use spatial variability for ODT.

    Returns:
        A GaussianMeasure on the SSH space representing the total ODT contribution.
    """
    if non_ice_ssh_variability_measure is None:
        non_ice_ssh_variability_measure = non_ice_ssh_variability_gaussian_measure(
            finger_print,
            finger_print_operator,
            order=order,
            length_scale=length_scale,
            amplitude=amplitude,
            use_spatial_variability=use_spatial_variability,
        )

    load_space: Sobolev = finger_print_operator.domain
    load_op = sea_level_change_to_load_operator(
        finger_print, load_space
    )
    ssh_op = sea_surface_height_operator(
        finger_print, finger_print_operator.codomain
    )

    fingerprint_ssh_op = (
        ssh_op @ finger_print_operator @ load_op
    )
    identity_op = load_space.identity_operator()
    total_op = fingerprint_ssh_op + identity_op
    return non_ice_ssh_variability_measure.affine_mapping(
        operator=total_op
    )
