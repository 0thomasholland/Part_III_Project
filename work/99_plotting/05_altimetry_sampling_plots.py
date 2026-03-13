# Auto-generated from notebook code cells.
# Source: notebooks/05 - Altimetry Sampling.ipynb

# ---- Notebook code cell 1 ----
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(120105)
import seaborn as sns
from scipy.stats import norm

from pygeoinf import GaussianMeasure
from pyslfp import FingerPrint, IceModel, averaging_operator

from project import colors, error_plot
from pygeoinf_extras.operators import (
    point_averaging_area_weighted_operator,
    point_averaging_operator,
)
from pygeoinf_extras.stats import expectation, standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange
from pyslfp_extras.plotting import plot

# ---- Notebook code cell 2 ----
lmax = 128
ALTIMETRY_LATITUDE_RANGE = 66.0

fp = FingerPrint(lmax=lmax)
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

print(f"Truncation degree: {lmax}")
print(
    f"Altimetry latitude range: ±{ALTIMETRY_LATITUDE_RANGE}°"
)

# ---- Notebook code cell 3 ----
fig, ax, im = plot(
    samples.total_ssh * fp.altimetry_projection() * 1000,
    colorbar_label="Sea Surface Height Change (mm)",
)

grid_coarse = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=30.0,
    latitude_range=ALTIMETRY_LATITUDE_RANGE,
)

ax.plot(
    grid_coarse.lons,
    grid_coarse.lats,
    "w^",
    markersize=4,
    transform=ccrs.PlateCarree(),
)
ax.set_title(
    f"SSH sample with ocean-altimetry grid (30° spacing, {len(grid_coarse)} points)"
)
plt.show()

# ---- Notebook code cell 4 ----
fig, ax, im = plot(
    samples.total_ssh * fp.altimetry_projection() * 1000,
    colorbar_label="Sea Surface Height Change (mm)",
)

grid_fine = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=5.0,
    latitude_range=ALTIMETRY_LATITUDE_RANGE,
)

ax.plot(
    grid_fine.lons,
    grid_fine.lats,
    "w.",
    markersize=1,
    transform=ccrs.PlateCarree(),
)
ax.set_title(
    f"SSH sample with ocean-altimetry grid (5° spacing, {len(grid_fine)} points)"
)
plt.show()

# ---- Notebook code cell 5 ----
fig, ax, im = plot(
    samples.total_thickness * fp.ice_projection() * 1000,
    colorbar_label="Ice Thickness Change (mm)",
    symmetric=True,
)

ice_grid = GridPoints.ice(fp, degree_spacing=10.0)

ax.plot(
    ice_grid.lons,
    ice_grid.lats,
    "w.",
    markersize=2,
    transform=ccrs.PlateCarree(),
)
ax.set_title(
    f"Ice thickness sample with ice grid (10° spacing, {len(ice_grid)} points)"
)
plt.show()

# ---- Notebook code cell 6 ----
ssh_operator = ice_change.load_to_altimetry_ssh_operator
ssh_space = ssh_operator.codomain

ice_thickness_measure = ice_change.ice_thickness
load_measure = ice_change.total_load

# Point evaluation at the fine grid
point_eval_op = grid_fine.point_evaluation_operator(
    ssh_space
)
point_ssh_op = point_eval_op @ ssh_operator

# Push the load measure through to get the point SSH distribution
point_ssh_measure = load_measure.affine_mapping(
    operator=point_ssh_op
)

print(f"SSH space dimension: {ssh_space.dim}")
print(
    f"Number of sampling points: {point_ssh_op.codomain.dim}"
)
print(
    f"Point SSH expectation range: [{point_ssh_measure.expectation.min() * 1e3:.3f}, "
    f"{point_ssh_measure.expectation.max() * 1e3:.3f}] mm"
)

# ---- Notebook code cell 7 ----
# True GMSL (no altimetry restriction)
true_gmsl = ice_thickness_measure.affine_mapping(
    operator=ice_change.ice_thickness_to_gmsl_operator
)

# Continuous surface-integral GMSL (altimetry band only)
continuous_gmsl = load_measure.affine_mapping(
    operator=ice_change.load_to_estimated_gmsl_operator
)

# Point-sampled GMSL (arithmetic mean of point evaluations)
point_avg_op = point_averaging_operator(
    point_ssh_op.codomain
)
point_gmsl = load_measure.affine_mapping(
    operator=point_avg_op @ point_ssh_op
)

print(
    f"True GMSL:       mean = {expectation(true_gmsl) * 1e3:.4f} mm,  "
    f"std = {standard_dev(true_gmsl) * 1e3:.4f} mm"
)
print(
    f"Continuous:      mean = {expectation(continuous_gmsl) * 1e3:.4f} mm,  "
    f"std = {standard_dev(continuous_gmsl) * 1e3:.4f} mm"
)
print(
    f"Point-sampled:   mean = {expectation(point_gmsl) * 1e3:.4f} mm,  "
    f"std = {standard_dev(point_gmsl) * 1e3:.4f} mm"
)

# ---- Notebook code cell 8 ----
fig, (ax1, ax2) = error_plot(
    true_measure=true_gmsl * 1000,
    estimation_measure=point_gmsl * 1000,
    ax1_xlabel="GMSL Change (mm)",
    ax2_xlabel="Estimation Error (mm)",
)

# ---- Notebook code cell 9 ----
altimetry_spacings = [20, 15, 10, 5, 2.5]

results = {}

for spacing in altimetry_spacings:
    grid = GridPoints.ocean_altimetry(
        fp,
        degree_spacing=spacing,
        latitude_range=ALTIMETRY_LATITUDE_RANGE,
    )
    point_op = grid.point_evaluation_operator(ssh_space)
    gmsl_op = (
        point_averaging_operator(point_op.codomain)
        @ point_op
        @ ssh_operator
    )
    gmsl_est = load_measure.affine_mapping(operator=gmsl_op)
    results[spacing] = {
        "n_points": len(grid),
        "mean_mm": expectation(gmsl_est) * 1e3,
        "std_mm": standard_dev(gmsl_est) * 1e3,
    }
    print(
        f"Spacing {spacing:5.1f}°: {len(grid):5d} points, "
        f"mean = {results[spacing]['mean_mm']:.4f} mm, "
        f"std = {results[spacing]['std_mm']:.4f} mm"
    )

# Reference values
mu_true = expectation(true_gmsl) * 1e3
sd_true = standard_dev(true_gmsl) * 1e3
mu_cont = expectation(continuous_gmsl) * 1e3
sd_cont = standard_dev(continuous_gmsl) * 1e3
print(
    f"\nTrue GMSL:       mean = {mu_true:.4f} mm, std = {sd_true:.4f} mm"
)
print(
    f"Continuous:      mean = {mu_cont:.4f} mm, std = {sd_cont:.4f} mm"
)

# ---- Notebook code cell 10 ----
spacings = sorted(results.keys())
means = [results[s]["mean_mm"] for s in spacings]
stds = [results[s]["std_mm"] for s in spacings]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Mean convergence
ax = axes[0]
ax.plot(
    spacings,
    means,
    "o-",
    color="tab:blue",
    label="Point-sampled",
)
ax.axhline(
    mu_cont,
    color="tab:green",
    linestyle="--",
    label="Continuous",
)
ax.axhline(
    mu_true,
    color="tab:red",
    linestyle=":",
    label="True GMSL",
)
ax.set_xlabel("Grid spacing (°)")
ax.set_ylabel("GMSL mean (mm)")
ax.set_title("Convergence of Mean GMSL Estimate")
ax.legend()
ax.grid(alpha=0.3)
ax.invert_xaxis()

# Std convergence
ax = axes[1]
ax.plot(
    spacings,
    stds,
    "o-",
    color="tab:blue",
    label="Point-sampled",
)
ax.axhline(
    sd_cont,
    color="tab:green",
    linestyle="--",
    label="Continuous",
)
ax.axhline(
    sd_true,
    color="tab:red",
    linestyle=":",
    label="True GMSL",
)
ax.set_xlabel("Grid spacing (°)")
ax.set_ylabel("GMSL std (mm)")
ax.set_title("Convergence of GMSL Standard Deviation")
ax.legend()
ax.grid(alpha=0.3)
ax.invert_xaxis()

sns.despine()
plt.tight_layout()
plt.show()

# ---- Notebook code cell 11 ----
DEGREE_SPACING = 5.0

grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=DEGREE_SPACING,
    latitude_range=ALTIMETRY_LATITUDE_RANGE,
)
latitudes = np.array(grid_points.lats)
print(
    f"Grid points: {len(grid_points)} at {DEGREE_SPACING}° spacing"
)

# Continuous area-weighted integral
gmsl_op_continuous = (
    ice_change.load_to_estimated_gmsl_operator
)

# Point SSH chain
point_eval_op_avg = grid_points.point_evaluation_operator(
    ssh_space
)
load_to_point_ssh = (
    point_eval_op_avg
    @ ice_change.load_to_altimetry_ssh_operator
)
point_ssh_space = load_to_point_ssh.codomain

# Unweighted (1/N) point mean
avg_unweighted = point_averaging_operator(point_ssh_space)
gmsl_op_unweighted = avg_unweighted @ load_to_point_ssh

# Area-weighted cos(lat) point mean
avg_area_weighted = point_averaging_area_weighted_operator(
    point_ssh_space, latitudes
)
gmsl_op_area_weighted = (
    avg_area_weighted @ load_to_point_ssh
)

# ---- Notebook code cell 12 ----
gmsl_continuous = load_measure.affine_mapping(
    operator=gmsl_op_continuous
)
gmsl_unweighted = load_measure.affine_mapping(
    operator=gmsl_op_unweighted
)
gmsl_area_weighted = load_measure.affine_mapping(
    operator=gmsl_op_area_weighted
)

mu_t = expectation(true_gmsl) * 1e3
mu_c = expectation(gmsl_continuous) * 1e3
mu_u = expectation(gmsl_unweighted) * 1e3
mu_aw = expectation(gmsl_area_weighted) * 1e3

sd_t = standard_dev(true_gmsl) * 1e3
sd_c = standard_dev(gmsl_continuous) * 1e3
sd_u = standard_dev(gmsl_unweighted) * 1e3
sd_aw = standard_dev(gmsl_area_weighted) * 1e3

print("=" * 52)
print(
    f"{'Method':<22}  {'Mean (mm)':>10}  {'Std (mm)':>10}"
)
print("=" * 52)
print(f"{'True GMSL':.<22}  {mu_t:>10.4f}  {sd_t:>10.4f}")
print(f"{'Continuous':.<22}  {mu_c:>10.4f}  {sd_c:>10.4f}")
print(f"{'Unweighted':.<22}  {mu_u:>10.4f}  {sd_u:>10.4f}")
print(
    f"{'Area-weighted':.<22}  {mu_aw:>10.4f}  {sd_aw:>10.4f}"
)
print("=" * 52)

# ---- Notebook code cell 13 ----
all_mus = [mu_c, mu_u, mu_aw]
all_sds = [sd_c, sd_u, sd_aw]

x_lo = min(m - 4 * s for m, s in zip(all_mus, all_sds))
x_hi = max(m + 4 * s for m, s in zip(all_mus, all_sds))
xs = np.linspace(x_lo, x_hi, 1000)

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7, 8),
    sharex=True,
    height_ratios=[3, 1],
)

styles = [
    ("Continuous (area integral)", "tab:green", "-"),
    ("Unweighted point mean", "tab:orange", "--"),
    ("Area-weighted point mean", "tab:blue", "--"),
]

for (label, color, ls), mu, sd in zip(
    styles, all_mus, all_sds
):
    ax1.plot(
        xs,
        norm.pdf(xs, loc=mu, scale=sd),
        color=color,
        linestyle=ls,
        linewidth=2,
        label=f"{label}  \u03bc={mu:.4f} mm, \u03c3={sd:.4f} mm",
    )

ax1.set_ylabel("Probability density (mm\u207b\u00b9)")
ax1.set_title(
    f"GMSL Averaging Comparison ({DEGREE_SPACING}\u00b0 grid, "
    f"\u00b1{ALTIMETRY_LATITUDE_RANGE}\u00b0 band)"
)
ax1.legend()
ax1.grid(alpha=0.3)

# Residuals
ax2.plot(
    xs,
    norm.pdf(xs, loc=mu_c, scale=sd_c)
    - norm.pdf(xs, loc=mu_u, scale=sd_u),
    color="tab:orange",
    linestyle="--",
    linewidth=1.5,
    label="Continuous \u2212 Unweighted",
)
ax2.plot(
    xs,
    norm.pdf(xs, loc=mu_c, scale=sd_c)
    - norm.pdf(xs, loc=mu_aw, scale=sd_aw),
    color="tab:blue",
    linestyle="--",
    linewidth=1.5,
    label="Continuous \u2212 Area-weighted",
)
ax2.axhline(0, color="black", linestyle=":", alpha=0.5)
ax2.set_xlabel("GMSL estimate (mm)")
ax2.set_ylabel("PDF difference (mm\u207b\u00b9)")
ax2.legend()
ax2.grid(alpha=0.3)

sns.despine()
plt.tight_layout()
plt.show()

# ---- Notebook code cell 14 ----
error = ice_change.ice_slc - ice_change.ice_ssh

fig, ax, im = plot(
    error.expectation * 1000 * fp.ocean_projection(),
    colorbar_label="Error Expectation: SLC \u2212 SSH (mm)",
)
ax.set_title("Mean Estimation Error (SLC \u2212 SSH)")
plt.show()

fig, ax, im = plot(
    error.sample_pointwise_std(20)
    * 1000
    * fp.ocean_projection(),
    colorbar_label="Error Sample Pointwise Std: SLC \u2212 SSH (mm)",
    cmap="Reds",
)
ax.set_title(
    "Pointwise Standard Deviation of Estimation Error"
)
plt.show()
