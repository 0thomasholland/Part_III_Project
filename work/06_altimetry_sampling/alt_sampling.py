# Altimetry sampling setup
#   Goal to generate methods for sampling points from the ocean field to
#   generate a list of points with lats and longs and altimetry values
#
#   uses pygeoinf's point_evaluation_operator under the hood that comes from
#   within a space's class
# %%
import cartopy.crs as ccrs
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
plot(ssh.sample())

# %%

measurement_space = ssh_operator.codomain

point_eval_op = ocean_point_evaluation_operator(
    finger_print=fp,
    measurement_space=measurement_space,
    point_degree_spacing=5.0,
)

# %%
# SSH sampling: push forward the SSH measure through point evaluation
values_measure: GaussianMeasure = ssh.affine_mapping(
    operator=point_eval_op,
)
sample_values = values_measure.sample()
print(sample_values)
print(point_eval_op.codomain.dim)

# %%
points = [
    (lat, lon, val)
    for (lat, lon), val in zip(coords, sample_values)
]

# %%
longs_all = [lon for lat, lon in ocean_coords]
lats_all = [lat for lat, lon in ocean_coords]
fig, ax, im = plot(ssh.sample())

ax.plot(
    longs_all,
    lats_all,
    "ro",
    markersize=1,
    transform=ccrs.PlateCarree(),
)


# %%

n_points = len(coords)

averaging_op = LinearOperator.from_matrix(
    EuclideanSpace(n_points),
    EuclideanSpace(1),
    np.array([[1.0 / n_points] * n_points]),
)
average_measure = ssh.affine_mapping(
    operator=averaging_op @ point_eval_op
)

print(expectation(average_measure))
print(standard_dev(average_measure))

_ssh_est = ice_thickness_measure.affine_mapping(
    operator=ice_thickness_to_estimated_gmsl_operator(
        finger_print=fp,
        finger_print_operator=fp_op,
        altimetry_latitude_range=66,
    )
)

print(expectation(_ssh_est))
print(standard_dev(_ssh_est))
