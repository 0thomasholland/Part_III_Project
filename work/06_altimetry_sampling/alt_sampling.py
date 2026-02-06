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
from pyslfp_extras.helpers import (
    get_ocean_point_coordinates,
)
from pyslfp_extras.measures import (
    ice_thickness_gaussian_measure,
)
from pyslfp_extras.operators import (
    ocean_point_evaluation_operator,
)

# %%
fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

ice_thickness_measure: GaussianMeasure = (
    ice_thickness_gaussian_measure(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=0.2 * fp.mean_sea_floor_radius,
        gmsl_target_std=0.001,
        gmsl_target_mean=0.01,
    )
)

ssh_operator = ice_thickness_to_ssh_operator(
    finger_print=fp,
    finger_print_operator=fp_op,
)

ssh: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ssh_operator,
)

# %%
fig, ax, im = plot(ssh.sample()*fp.altimetry_projection())
points = get_ocean_point_coordinates(
    fp,
    point_degree_spacing=30.0,
    altimetry_latitude_range=66.0,
)

ax.plot(
    points[1],
    points[0],
    "w^",
    transform=ccrs.PlateCarree(),
)

# %%

measurement_space = ssh_operator.codomain

point_op = ocean_point_evaluation_operator(
    fp,
    measurement_space,
    point_degree_spacing=30.0,
    altimetry_latitude_range=66.0,
)

point_measure = ssh.affine_mapping(operator=point_op)

print(point_measure.sample())
print(point_measure.domain.dim)
