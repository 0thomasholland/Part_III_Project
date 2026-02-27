# Altimetry sampling setup
#   Goal to generate methods for sampling points from the ocean field to
#   generate a list of points with lats and longs and altimetry values
#
#   uses pygeoinf's point_evaluation_operator under the hood that comes from
#   within a space's class
# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from pygeoinf import (
    EuclideanSpace,
    GaussianMeasure,
    LinearOperator,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    averaging_operator,
    ocean_projection_operator,
    plot,
    spatial_mutliplication_operator,
)

from project.operators import (
    ice_thickness_to_estimated_gmsl_operator,
    ice_thickness_to_ssh_operator,
)
from pygeoinf_extras import expectation, standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

# %%
fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.001,
    gmsl_target_mean=0.01,
)

samples = ice_change.sample()

# %%
fig, ax, im = plot(
    samples.total_ssh * fp.altimetry_projection()
)
grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=30.0,
    latitude_range=66.0,
)

ax.plot(
    grid_points.lons,
    grid_points.lats,
    "w^",
    transform=ccrs.PlateCarree(),
)

# %%

measurement_space = ssh_operator.codomain

point_op = grid_points.point_evaluation_operator(
    measurement_space
)

point_measure = ssh.affine_mapping(operator=point_op)

print(point_measure.sample())
print(point_measure.domain.dim)

# %%
fig, ax, im = plot(
    samples.total_thickness * fp.ice_projection()
)
ice_grid_points = GridPoints.ice(fp, 10.0)

ax.plot(
    ice_grid_points.lons,
    ice_grid_points.lats,
    "w.",
    transform=ccrs.PlateCarree(),
)
