"""Plotting helpers built on pyslfp's modern matplotlib API."""

from typing import List, Optional, Tuple, Union

import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.collections import QuadMesh
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from pyshtools import SHGrid
from pyslfp import create_map_figure, plot as pyslfp_plot


def plot(
    f: SHGrid,
    /,
    *,
    projection: ccrs.Projection = ccrs.Robinson(),
    contour: bool = False,
    cmap: str = "RdBu",
    coasts: bool = True,
    rivers: bool = False,
    borders: bool = False,
    map_extent: Optional[List[float]] = None,
    gridlines: bool = False,
    symmetric: bool = False,
    colorbar: bool = True,
    colorbar_label: Optional[str] = None,
    colorbar_orientation: str = "horizontal",
    colorbar_pad: float = 0.05,
    colorbar_shrink: float = 0.7,
    tight_layout: bool = True,
    figsize=(6, 4),
    **kwargs,
) -> Tuple[
    Figure, GeoAxes, Union[QuadMesh, QuadContourSet]
]:
    """
    Plots a pyshtools SHGrid object on a map.

    This function provides a flexible interface to visualize spherical harmonic
    grid data by acting as a wrapper around the plotting facilities provided
    by the pygeoinf library.

    Args:
        f (SHGrid): The scalar field to be plotted.
        projection (ccrs.Projection): The cartopy projection to be used.
            Defaults to ccrs.Robinson().
        contour (bool): If True, a filled contour plot is created. If False,
            a pcolormesh plot is created. Defaults to False.
        cmap (str): The colormap for the plot. Defaults to 'RdBu'.
        coasts (bool): If True, coastlines are drawn. Defaults to True.
        rivers (bool): If True, major rivers are drawn. Defaults to False.
        borders (bool): If True, country borders are drawn. Defaults to False.
        map_extent (Optional[List[float]]): Sets the longitude and latitude
            range for the plot, given as [lon_min, lon_max, lat_min, lat_max].
            Defaults to None (global extent).
        gridlines (bool): If True, latitude and longitude gridlines are
            included. Defaults to True.
        symmetric (bool): If True, the color scale is set symmetrically
            around zero. This is overridden if 'vmin' or 'vmax' are provided
            in kwargs. Defaults to False.
        colorbar (bool): If True, a colorbar is added to the plot.
            Defaults to True.
        colorbar_label (Optional[str]): Label for the colorbar.
            Defaults to None (no label).
        colorbar_orientation (str): Orientation of the colorbar
            ('horizontal' or 'vertical'). Defaults to 'horizontal'.
        colorbar_pad (float): Padding between the axes and the colorbar.
            Defaults to 0.05.
        colorbar_shrink (float): Fraction by which to multiply the size
            of the colorbar. Defaults to 0.7.
        **kwargs: Additional keyword arguments are forwarded to the underlying
            matplotlib plotting function (ax.pcolormesh or ax.contourf).

    Returns:
        Tuple[Figure, GeoAxes, Union[QuadMesh, QuadContourSet]]:
            A tuple containing the matplotlib Figure, the cartopy GeoAxes,
            and the plot artist object (e.g., QuadMesh or QuadContourSet).
    """

    if not isinstance(f, SHGrid):
        raise ValueError("must be of SHGrid type.")

    plot_options = {
        "contour": contour,
        "cmap": cmap,
        "coasts": coasts,
        "rivers": rivers,
        "borders": borders,
        "map_extent": map_extent,
        "gridlines": gridlines,
        "symmetric": symmetric,
    }

    plot_options.update(kwargs)

    if projection is None:
        projection = ccrs.Robinson()

    fig: Figure
    if "ax" in kwargs and kwargs["ax"] is not None:
        ax = kwargs.pop("ax")
        plot_options.pop("ax", None)
        fig = ax.figure
    else:
        fig, ax = create_map_figure(
            figsize=figsize,
            projection=projection,
        )

    ax, im = pyslfp_plot(
        f,
        ax=ax,
        projection=projection,
        colorbar=False,
        **plot_options,
    )

    if tight_layout:
        fig.tight_layout()

    # Add colorbar if requested
    if colorbar:
        fig.colorbar(
            im,
            ax=ax,
            orientation=colorbar_orientation,
            pad=colorbar_pad,
            shrink=colorbar_shrink,
            label=colorbar_label,
        )

    return fig, ax, im
