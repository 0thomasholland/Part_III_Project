# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pyslfp import (
    FingerPrint,
    averaging_operator,
    plot,
    sea_surface_height_operator,
)
from scipy.stats import norm

from Part_III_Project import (
    ice_thickness_change_measures,
    ocean_dynamic_topography_measures,
)

mpl.rcParams["figure.dpi"] = 600
output_dir = "../../outputs/poster/AutomatedFigures/Distributions"

# %%
# Setup
lmax = 128
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()
fingerprint_operator = fp.as_sobolev_linear_operator(
    2,
    0.1 * fp.mean_sea_floor_radius,
)
load_space = fingerprint_operator.domain
response_space = fingerprint_operator.codomain
sea_surface_height_op = sea_surface_height_operator(
    fp,
    response_space,
)
measurement_space = sea_surface_height_op.codomain


# %%
# Parameters
ice_length_scale = 0.5 * fp.mean_sea_floor_radius
ice_gmsl_target_std = 0.005
net_ice_thickness_change = 0.0

odt_length_scale = 0.1 * fp.mean_sea_floor_radius
odt_standard_deviation_factor = 0.002

altimetry_error_length_scale = 0.01 * fp.mean_sea_floor_radius
altimetry_error_standard_deviation = 0.001 / fp.length_scale

altimetry_range = 66


# %%

ice_thickness_change_measure, _ = ice_thickness_change_measures(
    fp,
    fingerprint_operator,
    ice_length_scale,
    ice_gmsl_target_std,
    net_ice_thickness_change,
)

ice_thickness_change_measure_sample = (
    ice_thickness_change_measure.sample()
)

ice_load_measure = fp.direct_load_from_ice_thickness_change(
    ice_thickness_change_measure_sample,
)
ice_slc = fp(direct_load=ice_load_measure)

ice_ssh = fp.sea_surface_height_change(
    ice_slc[0],
    ice_slc[1],
    ice_slc[3],
)

ice_slc_measure = ice_thickness_change_measure.affine_mapping(
    operator=response_space.subspace_projection(0)
    @ fingerprint_operator,
)

ice_ssh_measure = ice_thickness_change_measure.affine_mapping(
    operator=sea_surface_height_op @ fingerprint_operator,
)

odt_measure, _ = ocean_dynamic_topography_measures(
    fp,
    fingerprint_operator,
    odt_length_scale,
    odt_standard_deviation_factor,
)
odt_measure_sample = odt_measure.sample()
odt_load_measure = fp.direct_load_from_sea_level_change(
    odt_measure_sample,
)
odt_fingerprint = fp(direct_load=odt_load_measure)

altimetry_error = measurement_space.point_value_scaled_sobolev_kernel_gaussian_measure(
    1.5,
    altimetry_error_length_scale,
    altimetry_error_standard_deviation,
)
altimetry_error = altimetry_error.sample()


# %% map plots

plots = [
    (
        ice_thickness_change_measure_sample * fp.ice_projection(),
        "Ice Thickness Change",
        "metres",
    ),
    (
        ice_slc[0] * fp.ocean_projection() * 1000,
        "Sea Level Change from Ice Load",
        "mm",
    ),
    (
        ice_ssh * fp.ocean_projection() * 1000,
        "Sea Surface Height Change from Ice Load",
        "mm",
    ),
]


for x in plots:
    fig, ax, im = plot(
        x[0],
    )
    # set colorbar below plot with label
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        shrink=0.7,
    )
    cbar.set_label(x[2])
    ax.set_title(x[1])
    plt.savefig(
        f"{output_dir}/{x[1].lower().replace(' ', '_')}.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()

# %% measure plots
plots = [
    (
        "Ice Thickness Change Measure",
        ice_thickness_change_measure,
        fp.ice_projection(value=0),
    ),
    (
        "Ice Change Driven Sea Level Change",
        ice_slc_measure,
        fp.ocean_projection(value=0),
    ),
    (
        "Ice Change Driven Sea Surface Height Change",
        ice_ssh_measure,
        fp.ocean_projection(value=0),
    ),
]


# %%
def relative_to_std(x, expectation, std):
    return (x - expectation) / std


def inverse_relative_to_std(x, expectation, std):
    return x * std + expectation


for data in plots:
    spatial_average = averaging_operator(
        data[1].domain,
        [data[2] / fp.integrate(data[2])],
    )
    averaged_measure = data[1].affine_mapping(
        operator=spatial_average,
    )
    expectation = averaged_measure.expectation[0]
    std = averaged_measure.covariance.matrix(dense=True)[0, 0] ** 0.5
    print(
        f"{data[0]}: Expectation = {expectation:.4e}, Standard Deviation = {std:.4e}",
    )
    x_space = np.linspace(
        expectation - 4 * std,
        expectation + 4 * std,
        100,
    )
    pdf = norm.pdf(x_space, loc=expectation, scale=std)

    # Get current axes
    ax = plt.gca()
    ax.plot(x_space * fp.length_scale, pdf * fp.length_scale)
    ax.set_title(f"{data[0]} Distribution")
    ax.set_xlabel("Meters")
    ax.set_ylabel("Probability Density")

    # Add secondary x axis - need to account for length_scale in the conversion
    ax2 = ax.secondary_xaxis(
        -0.15,
        functions=(
            lambda x: relative_to_std(
                x,
                expectation,
                std,
            ),
            lambda x: inverse_relative_to_std(x, expectation, std),
        ),
    )
    ax2.set_xlabel("Relative to Ice GMSL (stds)")
    plt.savefig(
        f"{output_dir}/{data[0].lower().replace(' ', '_')}_distribution.png",
        dpi=600,
        bbox_inches="tight",
    )
    plt.show()

# %%
