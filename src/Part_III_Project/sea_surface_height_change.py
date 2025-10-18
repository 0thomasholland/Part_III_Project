from pyshtools import SHGrid
from pyslfp.physical_parameters import GRAVITATIONAL_ACCELERATION


def sea_surface_height_change(
    finger_print, sea_level_change, displacement, angular_velocity_change
) -> SHGrid:
    """Calculate the sea surface height change from the sea level change,
    surface displacement, and angular velocity change.

    Args:
        sea_level_change (SHGrid): The change in sea level.
        displacement (SHGrid): The vertical surface displacement.
        angular_velocity_change (SHGrid): The change in angular velocity.

    Returns:
        SHGrid: The change in sea surface height.
    """
    return (
        sea_level_change
        + displacement
        + (
            finger_print.centrifugal_potential_change(
                angular_velocity_change=angular_velocity_change
            )
            / GRAVITATIONAL_ACCELERATION
        )
    )
