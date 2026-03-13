# %%
import numpy as np
from matplotlib import pyplot as plt
from pygeoinf import (
    GaussianMeasure,
    LinearOperator,
)
from pyslfp import (
    FingerPrint,
    IceModel,
    plot,
)

from project import ice_thickness_to_slc_operator
from project.operators import (
    ice_thickness_to_estimated_gmsl_operator,
    ice_thickness_to_gmsl_estimation_error_operator,
    ice_thickness_to_ssh_operator,
)
from project.plots import error_plot
from pygeoinf_extras.stats import expectation, standard_dev
from pyslfp_extras.gmsl import (
    gmsl_from_ice_thickness_operator,
)
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

# %%


fp = FingerPrint(lmax=128)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)


# %%

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.001,
    gmsl_target_mean=0.01,
)
ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness_measure
)

# %%

gmsl_from_ice_thickness_operator_op: LinearOperator = (
    gmsl_from_ice_thickness_operator(
        finger_print=fp, finger_print_operator=fp_op
    )
)


print(
    standard_dev(
        ice_thickness_measure.affine_mapping(
            operator=gmsl_from_ice_thickness_operator_op
        )
    )
)

print(
    expectation(
        ice_thickness_measure.affine_mapping(
            operator=gmsl_from_ice_thickness_operator_op
        )
    )
)

# plot(ice_thickness_measure.sample(), symmetric=True)

# %%

slc: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ice_thickness_to_slc_operator(
        finger_print=fp,
        finger_print_operator=fp_op,
    )
)

ssh: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ice_thickness_to_ssh_operator(
        finger_print=fp,
        finger_print_operator=fp_op,
    )
)

slc_s = slc.sample()
ssh_s = ssh.sample()

# get the max value from the absolute values of slc and ssh samples
value = max(np.abs(slc_s).max(), np.abs(ssh_s).max())

# plot(slc_s, vmax=value, vmin=-value, symmetric=True)
# plot(ssh_s, vmax=value, vmin=-value, symmetric=True)

# %%

true_gmsl_measure: GaussianMeasure = (
    ice_thickness_measure.affine_mapping(
        operator=gmsl_from_ice_thickness_operator_op
    )
)

estimated_gmsl_measure: GaussianMeasure = (
    ice_thickness_measure.affine_mapping(
        operator=(
            ice_thickness_to_estimated_gmsl_operator(
                finger_print=fp,
                finger_print_operator=fp_op,
                altimetry_latitude_range=66.0,
            )
        )
    )
)

print(
    f"GMSL_true: expectation = {(gmsl_exp := expectation(true_gmsl_measure))}, std = {(gmsl_std := standard_dev(true_gmsl_measure))}"
)
print(
    f"GMSL_est: expectation = {(est_exp := expectation(estimated_gmsl_measure))}, std = {(est_std := standard_dev(estimated_gmsl_measure))}"
)
# %%
# error = estimate - true

error_measure: GaussianMeasure = ice_thickness_measure.affine_mapping(
    operator=ice_thickness_to_gmsl_estimation_error_operator(
        finger_print=fp,
        finger_print_operator=fp_op,
        altimetry_latitude_range=66.0,
    )
)

print(error_measure.covariance.matrix(dense=True))

print(
    f"Error: expectation = {(err_exp := expectation(error_measure))}, std = {(err_std := standard_dev(error_measure))}"
)

# %%

# on two line graphs next to each other plot on left: gmsl distribution and
# estimation, and on right the error distribution

fig, (ax1, ax2) = error_plot(
    true_measure=true_gmsl_measure,
    estimation_measure=estimated_gmsl_measure,
    true_color="blue",
    est_color="orange",
    true_label=f"True GMSL (mean: {gmsl_exp:.2e} m, std: {gmsl_std:.2e} m)",
    est_label=f"Estimated GMSL (mean: {est_exp:.2e} m, std: {est_std:.2e} m)",
    error_label=f"GMSL Estimation Error (mean: {err_exp:.4e} m, std: {err_std:.4e} m)",
    ax1_title="GMSL Distribution",
    ax1_xlabel="GMSL (m)",
    ax2_title="GMSL Error Distribution",
    ax2_xlabel="GMSL Error (m)",
    suptitle="GMSL Estimation from Altimetry",
    show_bias=True,
)
plt.tight_layout()
plt.savefig("gmsl_estimation_error.pdf", dpi=600)
