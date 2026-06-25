# %%
# =============================================================================
# 12 - GMSL Averaging Comparison
# =============================================================================
#
# Compares three scalar Gaussian distributions obtained by pushing the
# ice-load GaussianMeasure through three different GMSL operators:
#
#   1. Continuous  — area-weighted surface integral over the altimetry
#                    band via fp.model.integrate()  (the reference)
#   2. Unweighted  — arithmetic mean (1/N) at a regular lat/lon point grid
#   3. Area-weighted — cos(lat)-weighted mean at the same point grid
#
# Each operator is an affine map from the load space to EuclideanSpace(1),
# so the pushed-forward measure is an exact scalar Gaussian.  No Monte Carlo
# is needed — we just read off the mean and variance analytically.
#
# The three Gaussians are overlaid on a single axes so the shift in mean
# and change in spread between the methods is immediately visible.
# =============================================================================

# %%
from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from pygeoinf_extras.operators import (
    point_averaging_area_weighted_operator,
)
from pygeoinf_extras.stats import expectation, standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import IceSheetChange

# %%
# -----------------------------------------------------------------------------
# 1.  Fingerprint model
# -----------------------------------------------------------------------------
print("Setting up fingerprint model...")

fp = EarthState.from_defaults(lmax=256)

fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

# %%
# -----------------------------------------------------------------------------
# 2.  Ice-sheet change prior  (no firn — keeps the chain lean)
# -----------------------------------------------------------------------------
print("Building ice-sheet change prior...")

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.15 * fp.model.parameters.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.005,
    gmsl_target_mean=0.02,
    include_firn=False,
)

# The load GaussianMeasure — the single source we push forward.
load_measure = ice_change.total_load

# %%
# -----------------------------------------------------------------------------
# 3.  Altimetry point grid  (shared by both point estimators)
# -----------------------------------------------------------------------------
DEGREE_SPACING = 2.5
LATITUDE_RANGE = 66.0

print(
    f"Building altimetry grid "
    f"(spacing={DEGREE_SPACING}°, band=±{LATITUDE_RANGE}°)..."
)

grid_points = GridPoints.ocean_altimetry(
    fp,
    degree_spacing=DEGREE_SPACING,
    latitude_range=LATITUDE_RANGE,
)
latitudes = np.array(grid_points.lats)
print(f"  Points in grid: {len(grid_points)}")

# %%
# -----------------------------------------------------------------------------
# 4.  Build the three load -> scalar GMSL operators
# -----------------------------------------------------------------------------

# -- Operator 1: continuous area-weighted integral --
# GMSLOperatorBase.load_to_estimated_gmsl_operator integrates SSH against
# the normalised altimetry projection weight via fp.model.integrate(), which
# uses the exact spherical area element (cos-lat weighting implicit in SH).
gmsl_op_continuous = (
    ice_change.load_to_estimated_gmsl_operator
)

# -- Operators 2 & 3: build the point SSH chain from the same GridPoints
# instance so that the latitudes array and the operator's point dimension
# are guaranteed to match.  Composing manually avoids the mismatch that
# arises when ice_change.load_to_ssh_point_estimations_operator builds its
# own internal GridPoints (which can resolve to a different number of ocean
# points depending on the SH lmax used for the mask evaluation).
ssh_space = (
    ice_change.load_to_altimetry_ssh_operator.codomain
)
point_eval_op = grid_points.point_evaluation_operator(
    ssh_space
)
load_to_point_ssh = (
    point_eval_op
    @ ice_change.load_to_altimetry_ssh_operator
)
point_ssh_space = load_to_point_ssh.codomain

avg_unweighted = point_l2_products_operator(point_ssh_space)
avg_area_weighted = point_averaging_area_weighted_operator(
    point_ssh_space,
    latitudes,
)

# -- Operator 2: unweighted (1/N) point mean --
gmsl_op_unweighted = avg_unweighted @ load_to_point_ssh

# -- Operator 3: area-weighted cos(lat) point mean --
gmsl_op_area_weighted = (
    avg_area_weighted @ load_to_point_ssh
)

# %%
# -----------------------------------------------------------------------------
# 5.  Push the load measure through each operator analytically
# -----------------------------------------------------------------------------
print("Pushing load measure through GMSL operators...")

# True GMSL: push the ice_thickness measure (not load measure) through
# ice_thickness_to_gmsl_operator, which is the physically exact scalar
# GMSL implied by the prior — no altimetry band restriction, no sampling.
gmsl_true = ice_change.ice_thickness.affine_mapping(
    operator=ice_change.ice_thickness_to_gmsl_operator
)

gmsl_continuous = load_measure.affine_mapping(
    operator=gmsl_op_continuous
)
gmsl_unweighted = load_measure.affine_mapping(
    operator=gmsl_op_unweighted
)
gmsl_area_weighted = load_measure.affine_mapping(
    operator=gmsl_op_area_weighted
)

# %%
# -----------------------------------------------------------------------------
# 6.  Extract mean and std from each scalar Gaussian
# -----------------------------------------------------------------------------
mu_t = expectation(gmsl_true)
mu_c = expectation(gmsl_continuous)
mu_u = expectation(gmsl_unweighted)
mu_aw = expectation(gmsl_area_weighted)

sd_t = standard_dev(gmsl_true)
sd_c = standard_dev(gmsl_continuous)
sd_u = standard_dev(gmsl_unweighted)
sd_aw = standard_dev(gmsl_area_weighted)

print("\n" + "=" * 52)
print(
    f"{'Method':<22}  {'Mean (mm)':>10}  {'Std (mm)':>10}"
)
print("=" * 52)
print(
    f"{'True GMSL':.<22}  {mu_t * 1e3:>10.4f}  {sd_t * 1e3:>10.4f}"
)
print(
    f"{'Continuous':.<22}  {mu_c * 1e3:>10.4f}  {sd_c * 1e3:>10.4f}"
)
print(
    f"{'Unweighted':.<22}  {mu_u * 1e3:>10.4f}  {sd_u * 1e3:>10.4f}"
)
print(
    f"{'Area-weighted':.<22}  {mu_aw * 1e3:>10.4f}  {sd_aw * 1e3:>10.4f}"
)
print("=" * 52)

# %%
# -----------------------------------------------------------------------------
# 7.  Plot the three Gaussians on a shared axis
# # plot the residuals in subplot below
# -----------------------------------------------------------------------------
all_mus = [mu_c, mu_u, mu_aw]
all_sds = [sd_c, sd_u, sd_aw]

# x range: cover ±4 std of the widest distribution
x_lo = min(m - 4 * s for m, s in zip(all_mus, all_sds))
x_hi = max(m + 4 * s for m, s in zip(all_mus, all_sds))
xs = np.linspace(x_lo, x_hi, 1000)

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True, height_ratios=[3, 1]
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
        xs * 1e3,
        norm.pdf(xs, loc=mu, scale=sd) / 1e3,
        color=color,
        linestyle=ls,
        linewidth=2,
        label=f"{label}"
        f"  μ = {mu * 1e3:.4f} mm,  "
        f"σ = {sd * 1e3:.4f} mm",
    )

ax1.axvline(
    all_mus[0] * 1e3,
    color="tab:red",
    linestyle=":",
    alpha=0.7,
)
ax1.axvline(
    all_mus[1] * 1e3,
    color="tab:orange",
    linestyle=":",
    alpha=0.7,
)
ax1.axvline(
    all_mus[2] * 1e3,
    color="tab:blue",
    linestyle=":",
    alpha=0.7,
)

ax1.set_xlabel("GMSL estimate  [mm]")
ax1.set_ylabel("Probability density  [mm⁻¹]")
ax1.set_title(
    "Point Altimetry Sampling Errors\n"
    f"({DEGREE_SPACING}° grid, "
    f"±{LATITUDE_RANGE}° altimetry band)"
)

ax2.plot(
    xs * 1e3,
    norm.pdf(xs, loc=mu_c, scale=sd_c) / 1e3
    - norm.pdf(xs, loc=mu_u, scale=sd_u) / 1e3,
    color="tab:orange",
    linestyle="--",
    linewidth=1.5,
    label="Unweighted - Continuous",
)
ax2.plot(
    xs * 1e3,
    norm.pdf(xs, loc=mu_c, scale=sd_c) / 1e3
    - norm.pdf(xs, loc=mu_aw, scale=sd_aw) / 1e3,
    color="tab:blue",
    linestyle="--",
    linewidth=1.5,
    label="Area-weighted - Continuous",
)
ax2.axhline(0, color="black", linestyle=":", alpha=0.7)
ax2.set_xlabel("GMSL estimate  [mm]")
ax2.set_ylabel("PDF difference  [mm⁻¹]")
ax2.set_title("Difference in Probability Densities")

plt.legend(
    loc="upper left", bbox_to_anchor=(0, 1.02, 1.0, -1.15)
)
ax1.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
fig.savefig("gmsl_averaging_comparison.pdf", dpi=600)
plt.show()
#

# %%

# %%
# -----------------------------------------------------------------------------
# 8.  Ridge plot — one row per method, filled Gaussian curves
# -----------------------------------------------------------------------------
# Convert everything to mm upfront so all PDF arithmetic is consistent.
styles_ridge = [
    # ("True GMSL", mu_t * 1e3, sd_t * 1e3, "tab:red"),
    (
        "Continuous (area integral)",
        mu_c * 1e3,
        sd_c * 1e3,
        "tab:green",
    ),
    (
        "Unweighted point mean",
        mu_u * 1e3,
        sd_u * 1e3,
        "tab:orange",
    ),
    (
        "Area-weighted point mean",
        mu_aw * 1e3,
        sd_aw * 1e3,
        "tab:blue",
    ),
]

n_rows = len(styles_ridge)

# x range in mm: cover ±4 std of the widest distribution
all_mus_r = [r[1] for r in styles_ridge]
all_sds_r = [r[2] for r in styles_ridge]
x_lo_r = min(
    m - 4 * s for m, s in zip(all_mus_r, all_sds_r)
)
x_hi_r = max(
    m + 4 * s for m, s in zip(all_mus_r, all_sds_r)
)
xs_r = np.linspace(x_lo_r, x_hi_r, 1000)

# overlap: fraction of peak height between successive baselines
OVERLAP = 0.1

# peak PDF height (mm⁻¹) — used to set a uniform row spacing
max_pdf = max(
    norm.pdf(mu, loc=mu, scale=sd)
    for _, mu, sd, _ in styles_ridge
)
row_height = max_pdf * OVERLAP

fig, ax = plt.subplots(figsize=(7, 5))

for i, (label, mu, sd, color) in enumerate(styles_ridge):
    baseline = i * row_height
    pdf = norm.pdf(xs_r, loc=mu, scale=sd)  # mm⁻¹
    peak = norm.pdf(mu, loc=mu, scale=sd)  # mm⁻¹

    # filled area
    ax.fill_between(
        xs_r,
        baseline,
        baseline + pdf,
        color=color,
        alpha=0.35,
        zorder=i,
    )
    # outline
    ax.plot(
        xs_r,
        baseline + pdf,
        color=color,
        linewidth=1.8,
        zorder=i + n_rows,
    )
    # mean line from baseline to peak
    ax.vlines(
        mu,
        baseline,
        baseline + peak,
        color=color,
        linewidth=1.2,
        linestyle="--",
        zorder=i + n_rows,
    )
    # thin white baseline rule to visually separate rows
    ax.axhline(
        baseline, color="white", linewidth=0.8, zorder=i
    )

    # row label to the left of the plot
    ax.text(
        x_lo_r - 0.015 * (x_hi_r - x_lo_r),
        baseline + 0.03 * peak,
        f"{label}\nμ={mu:.3f} mm  σ={sd:.3f} mm",
        ha="right",
        va="center",
        fontsize=7.5,
        color=color,
        zorder=i + n_rows,
    )

ax.set_xlabel("GMSL estimate  [mm]")
ax.set_yticks([])
ax.set_title(
    "Ridge Plot — GMSL Averaging Comparison\n"
    f"({DEGREE_SPACING}° grid, ±{LATITUDE_RANGE}° altimetry band)"
)
ax.grid(axis="x", linestyle=":", alpha=0.4)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.tight_layout()
fig.savefig("gmsl_averaging_comparison_ridge.pdf", dpi=600)
plt.show()

# %%
# box plot of the three distributions side by side, with the mean as a point and the std as the box

fig, ax = plt.subplots(figsize=(6, 5))
box_width = 0.6
for i, (label, mu, sd, color) in enumerate(styles_ridge):
    ax.bar(
        i,
        0,
        width=box_width,
        color=color,
        alpha=0.35,
        edgecolor=color,
        linewidth=1.8,
    )
    ax.errorbar(
        i,
        mu,
        yerr=sd,
        color=color,
        fmt="o",
        capsize=5,
        label=f"{label}\nμ={mu * 1e3:.3f} mm  σ={sd * 1e3:.3f} mm",
    )
ax.set_ylim(x_lo_r, x_hi_r)
ax.set_xticks(range(len(styles)))
ax.set_xticklabels(
    [s[0] for s in styles], rotation=45, ha="right"
)
ax.set_ylabel("GMSL estimate  [mm]")
ax.set_title(
    "Box Plot — GMSL Averaging Comparison\n"
    f"({DEGREE_SPACING}° grid, ±{LATITUDE_RANGE}° altimetry band)"
)
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
