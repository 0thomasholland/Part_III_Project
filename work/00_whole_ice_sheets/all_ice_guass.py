# %%

import numpy as np
from pygeoinf import (
    GaussianMeasure,
    LinearOperator,
)
from pygeoinf.symmetric_space.sphere import Sobolev
from pyslfp import (
    FingerPrint,
    IceModel,
    ice_projection_operator,
    ice_thickness_change_to_load_operator,
    ocean_projection_operator,
    plot,
    sea_surface_height_operator, averaging_operator,
)
from stack_data.core import Line

from pygeoinf_extras.stats import (
    standard_dev,
)
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)

# %%

alimetry_resolution = (
    1440  # number of points from 0 to 90˚ that are sampled
)

latitudes = np.linspace(1, 90, alimetry_resolution)

fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

load_space: Sobolev = fp_op.domain
response_space: Sobolev = fp_op.codomain

# %%

ice_thickness_measure: GaussianMeasure = (
    load_space.heat_kernel_gaussian_measure(
        0.2 * fp.mean_sea_floor_radius
    )
)

# %%

ice_projection_op: LinearOperator = ice_projection_operator(
    fp, load_space
)

ice_thickness_measure: GaussianMeasure = (
    ice_thickness_measure.affine_mapping(
        operator=ice_projection_op
    )
)

gmsl_from_ice_thickness_operator_op: LinearOperator = (
    gmsl_from_ice_thickness_operator(
        load_space=load_space, fp=fp
    )
)

gmsl_target_std = 0.01


ice_thickness_measure *= gmsl_target_std / standard_dev(
    ice_thickness_measure.affine_mapping(
        operator=gmsl_from_ice_thickness_operator_op
    )
)

print(
    standard_dev(
        ice_thickness_measure.affine_mapping(
            operator=gmsl_from_ice_thickness_operator_op
        )
    )
)
plot(ice_thickness_measure.sample(), symmetric=True)

# %%

ice_thickness_to_load_op: LinearOperator = (
    ice_thickness_change_to_load_operator(fp, load_space)
)
sea_surface_height_op: LinearOperator = (
    sea_surface_height_operator(fp, response_space)
)
sl_subspace_op: LinearOperator = (
    response_space.subspace_projection(0)
)

ocean_projection_op: LinearOperator = (
    ocean_projection_operator(fp, load_space)
)

# %%

slc: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ocean_projection_op
    @ sl_subspace_op
    @ fp_op
    @ ice_thickness_to_load_op
)

ssh: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ocean_projection_op
    @ sea_surface_height_op
    @ fp_op
    @ ice_thickness_to_load_op
)

plot(slc.sample(), symmetric=True)
plot(ssh.sample(), symmetric=True)

# %%

gmsl_from_sea_level_op = averaging_operator()

true_gmsl_measure =

estimated_gmsl_measure = ssh
