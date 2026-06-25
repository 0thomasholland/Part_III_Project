# %% [markdown]
# # Bayesian Inversion Visualization Suite
# This script processes the results of the parallel inversion runs.

# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from project import colors

def gaussian(x, mean, std_dev):
    return (
        1
        / (std_dev * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    )

# %% [markdown]
# ## 1. Load Data
# Loads the consolidated master file from your disk.

# %%
master_file = "master_results.csv"

if pathlib.Path(master_file).exists():
    df = pd.read_csv(master_file)
else:
    # Placeholder for testing if file doesn't exist yet
    print("Master file not found. Exiting")
    exit()
# filter the dataframe so that it keeps shift values of zero or nan
# df = df[df["shift"].isna() | (df["shift"] == 0)]

# %%

print(np.mean(df["posterior_std_dev"]))
altimetry_std_dev = 0.001 * 1000
number_points = 1402
old_method_error = altimetry_std_dev / np.sqrt(
    number_points
)
print(f"Old method error: {old_method_error} mm")
# %%
# sns contour plot

z_score_posterior = (
    df["gmsl_true"] - df["posterior_mean"]
) / df["posterior_std_dev"]
z_score_ssh = (
    df["gmsl_true"] - df["ssh_estimation"]
) / old_method_error

max_val = max(
    np.abs(z_score_ssh).max(),
    np.abs(z_score_posterior).max(),
)

fig, ax = plt.subplots(figsize=(3.25, 3.25))

# KDE Plot
# sns.kdeplot(
#     x=z_score_ssh,
#     y=z_score_posterior,
#     cmap=colors.error_cmap,
# )

# Scatter Plot
ax.scatter(
    x=z_score_ssh,
    y=z_score_posterior,
    alpha=0.2,
    color=colors.primary_error,
    label="Inversion run",
    marker=".",
)

# Axis Labels with Dynamic Coloring
ax.set_xlabel(
    "Old Method (SSH) z-score",
    # color=colors.old_method,
    fontweight="bold",
)
ax.set_ylabel(
    "New Method (Bayesian) z-score",
    # color=colors.new_method,
    fontweight="bold",
)

# Limits and Guides
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)
# ax.axhline(
#     0,
#     color=colors.new_method,
#     linestyle="--",
#     alpha=0.5,
#     label="Zero New Method Bias",
# )
# ax.axvline(
#     0,
#     color=colors.old_method,
#     linestyle="--",
#     alpha=0.5,
#     label="Zero Old Method Bias",
# )

# Title with mixed colors requires a bit of a trick if you want
# different words to have different colors.
# For a standard single-color title:
ax.set_title(
    f"Methods z-score: Old vs. New\n[no. inversion = {len(df)}]",
    fontweight="bold",
)

# ax.xaxis.label.set_color(colors.old_method)
# ax.tick_params(axis="x", colors=colors.old_method)

# ax.yaxis.label.set_color(colors.new_method)
# ax.tick_params(axis="y", colors=colors.new_method)
plt.tight_layout()
ax.legend()
fig.savefig("z_score_comparison_contour_b.pdf", dpi=600)

# %%
# do the residuals plot but with z scores instead of raw errors
fig, ax = plt.subplots(figsize=(3.25, 3.25))
sns.kdeplot(
    z_score_ssh,
    fill=True,
    label="Old Method (SSH)",
    color=colors.old_method,
)
sns.kdeplot(
    z_score_posterior,
    fill=True,
    label="New Method (Bayesian)",
    color=colors.new_method,
)
# plt.axvline(
#     0,
#     color=colors.true,
#     linestyle="--",
#     label="Zero z-score",
# )
plt.title(
    f"Distribution of z-scores\n[(Estimate - Truth) / Std Dev] [n={len(df)}]"
)
plt.xlabel("z-score")
plt.ylabel("Density")
plt.legend(loc="upper left")
plt.tight_layout()
fig.savefig("z_score_comparison_kde.pdf", dpi=600)

# %%
# calcualte the mean z score for both, and also the standard deviation of the z score
mean_z_score_bayesian = z_score_posterior.mean()
std_z_score_bayesian = z_score_posterior.std()

mean_z_score_ssh = z_score_ssh.mean()
std_z_score_ssh = z_score_ssh.std()

print(
    f"New method z-score: mean = {mean_z_score_bayesian:.2f}, std dev = {std_z_score_bayesian:.2f}"
)
print(
    f"Old method z-score: mean = {mean_z_score_ssh:.2f}, std dev = {std_z_score_ssh:.2f}"
)
