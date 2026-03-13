# Auto-generated from notebook code cells.
# Source: notebooks/01 - Sea Surface Height.ipynb

# ---- Notebook code cell 1 ----
from pathlib import Path

import matplotlib.pyplot as plt
from pyslfp import (
    FingerPrint,
    IceModel,
    plot,
    averaging_operator,
)
from pyslfp_extras.ice_thickness import IceSheetChange
from pygeoinf_extras import standard_dev, expectation
from project import error_plot
import numpy as np

np.random.seed(120101)

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.show = lambda *args, **kwargs: None
print = lambda *args, **kwargs: None


def _save_all_figures(prefix):
    for index, figure_number in enumerate(
        plt.get_fignums(), start=1
    ):
        fig = plt.figure(figure_number)
        fig.savefig(
            FIGURES_DIR / f"{prefix}_{index:02d}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close("all")


# ---- Notebook code cell 2 ----
fp = FingerPrint(lmax=512)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

load = fp.direct_load_from_ice_thickness_change(
    fp.ice_projection(value=0)
)
load /= fp.mean_sea_level_change(
    direct_load=load
)  # normalise to 1mm GMSL change

plot(
    load * fp.ice_projection(),
    colorbar_label="Load (kg/m²)",
)

# ---- Notebook code cell 3 ----
slc, dis, _, avc = fp(direct_load=load)

plot(
    slc * fp.ocean_projection(),
    colorbar_label="Sea Level Change (mm)",
    vmin=-1.25,
    vmax=1.25,
)

sshc = fp.sea_surface_height_change(slc, dis, avc)

plot(
    sshc * fp.ocean_projection(),
    colorbar_label="Sea Surface Height Change (mm)",
    vmin=-1.25,
    vmax=1.25,
)

# ---- Notebook code cell 4 ----
observed_sshc = sshc * fp.altimetry_projection(
    latitude_min=-66.0, latitude_max=66.0
)

plot(
    observed_sshc,
    colorbar_label="Observed Sea Surface Height Change (mm)",
    vmin=-1.25,
    vmax=1.25,
)

# ---- Notebook code cell 5 ----
true_gmsl = fp.ocean_average(slc)
ssh_gmsl = fp.ocean_average(sshc)
estimated_gmsl = fp.integrate(
    sshc * fp.altimetry_projection(value=0)
) / fp.integrate(fp.altimetry_projection(value=0))

percentage_error = (
    100 * abs(true_gmsl - estimated_gmsl) / abs(true_gmsl)
)

print(f"True GMSL change (using SLC): {true_gmsl:.3f} mm")
print(f"GMSL change using SSHC: {ssh_gmsl:.3f} mm")
print(
    f"Estimated GMSL change from observations: {estimated_gmsl:.3f} mm"
)
print(f"Percentage error: {percentage_error:.2f}%")

# ---- Notebook code cell 6 ----
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

# ---- Notebook code cell 7 ----
ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.001,
    gmsl_target_mean=0.01,
)
ice_thickness_measure = ice_change.ice_thickness

plot(ice_thickness_measure.expectation, symmetric=True)

# ---- Notebook code cell 8 ----
true_gmsl = ice_thickness_measure.affine_mapping(
    operator=ice_change.ice_thickness_to_gmsl_operator
)

print(
    f"Expectation of true GMSL: {expectation(true_gmsl * 1000):.3f} mm"
)
print(
    f"Standard deviation of true GMSL: {standard_dev(true_gmsl * 1000):.3f} mm"
)

# ---- Notebook code cell 9 ----
slc = ice_change.ice_slc
sshc = ice_change.ice_ssh

plot(
    slc.expectation * fp.ocean_projection() * 1000,
    colorbar_label="Sea Level Change (mm)",
)
plot(
    sshc.expectation * fp.ocean_projection() * 1000,
    colorbar_label="Sea Surface Height Change (mm)",
)

# ---- Notebook code cell 10 ----
samples = ice_change.sample()  # draw linked samples from the ice thickness, and derive their SLC and SSH

plot(
    samples.ice_slc * 1000,
    symmetric=True,
    colorbar_label="Sea Level Change (mm)",
)
plot(
    samples.ice_ssh * 1000,
    symmetric=True,
    colorbar_label="Sea Surface Height Change (mm)",
)

# ---- Notebook code cell 11 ----
altimetry_operator = averaging_operator(
    sshc.domain,
    [
        fp.altimetry_projection(value=0)
        / fp.integrate(fp.altimetry_projection(value=0))
    ],
)

estimated_gmsl_from_sshc = sshc.affine_mapping(
    operator=altimetry_operator
)


print(
    f"Expectation of GMSL change from SSHC: {expectation(estimated_gmsl_from_sshc * 1000):.3f} mm"
)
print(
    f"Standard deviation of GMSL change from SSHC: {standard_dev(estimated_gmsl_from_sshc * 1000):.3f} mm"
)

# and as a reminder:

print(
    f"Expectation of true GMSL: {expectation(true_gmsl * 1000):.3f} mm"
)
print(
    f"Standard deviation of true GMSL: {standard_dev(true_gmsl * 1000):.3f} mm"
)

# ---- Notebook code cell 12 ----
fig, (ax1, ax2) = error_plot(
    true_measure=true_gmsl * 1000,
    estimation_measure=estimated_gmsl_from_sshc * 1000,
    figsize=(12, 5),
    ax1_xlabel="GMSL Change (mm)",
    ax2_xlabel="Estimation Error (mm)",
)

# ---- Notebook code cell 13 ----
region = "NEU"
# region = "CAR"

val = np.max(
    [
        np.abs(
            slc.expectation
            * fp.regionmask_projection(region)
            * fp.ocean_function
            * 1000
        ).max(),
        np.abs(
            sshc.expectation
            * fp.regionmask_projection(region)
            * fp.ocean_function
            * 1000
        ).max(),
    ]
)
plot(
    slc.expectation
    * 1000
    * fp.regionmask_projection(region)
    * fp.ocean_function,
    vmax=val,
    vmin=-val,
    # map_extent=(-100, -30, -10, 50), # Caribbean region
    map_extent=(-20, 40, 30, 80),  # NEU region
)
plot(
    sshc.expectation
    * 1000
    * fp.regionmask_projection(region)
    * fp.ocean_function,
    vmax=val,
    vmin=-val,
    # map_extent=(-100, -30, -10, 50), # Caribbean region
    map_extent=(-20, 40, 30, 80),  # NEU region
)

# ---- Notebook code cell 14 ----
car_sshc_averaging_op = averaging_operator(
    sshc.domain,
    [
        (
            fp.regionmask_projection(region, value=0.0)
            * fp.ocean_projection(value=0)
        )
        / fp.integrate(
            (
                fp.regionmask_projection(region, value=0)
                * fp.ocean_projection(value=0)
            )
        )
    ],
)
car_slc_averaging_op = averaging_operator(
    slc.domain,
    [
        (
            fp.regionmask_projection(region, value=0.0)
            * fp.ocean_projection(value=0)
        )
        / fp.integrate(
            (
                fp.regionmask_projection(region, value=0)
                * fp.ocean_projection(value=0)
            )
        )
    ],
)

estimated_regional_gmsl = sshc.affine_mapping(
    operator=car_sshc_averaging_op
)
regional_gmsl = slc.affine_mapping(
    operator=car_slc_averaging_op
)

# ---- Notebook code cell 15 ----
fig, (ax1, ax2) = error_plot(
    true_measure=regional_gmsl * 1000,
    estimation_measure=estimated_regional_gmsl * 1000,
    ax1_xlabel="Regional GMSL Change (mm)",
    ax2_xlabel="Estimation Error (mm)",
)

# ---- Notebook code cell 16 ----
error = ice_change.ice_slc - ice_change.ice_ssh

plot(
    error.expectation * 1000 * fp.ocean_projection(),
    colorbar_label="Error Expectation: SLC - SSH (mm)",
)

plot(
    error.sample_pointwise_std(20)
    * 1000
    * fp.ocean_projection(),
    colorbar_label="Error Sample Pointwise Std: SLC - SSH (mm)",
    cmap="Reds",
)

_save_all_figures("01_sea_surface_height")
