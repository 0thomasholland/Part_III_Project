# %% [markdown]
# # Bayesian Inversion Visualization Suite
# This script processes the results of the parallel inversion runs.

# %%
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

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
    print(
        "Master file not found. Generating synthetic plotting data..."
    )
    n_samples = 100
    true_vals = np.random.uniform(2.0, 5.0, n_samples)
    df = pd.DataFrame(
        {
            "gmsl_true": true_vals,
            "posterior_mean": true_vals
            + np.random.normal(0, 0.2, n_samples),
            "posterior_std_dev": np.random.uniform(
                0.15, 0.25, n_samples
            ),
            "ssh_estimation": true_vals
            + np.random.normal(0, 0.5, n_samples),
        }
    )
# filter the dataframe so that it keeps shift values of zero or nan
# df = df[df["shift"].isna() | (df["shift"] == 0)]

# %%
# sns contour plot
max_val = max(
    np.abs(df["gmsl_true"] - df["ssh_estimation"]).max(),
    np.abs(df["gmsl_true"] - df["posterior_mean"]).max(),
)

fig, ax = plt.subplots(figsize=(3.25, 3.25))

# KDE Plot
sns.kdeplot(
    x=df["gmsl_true"] - df["ssh_estimation"],
    y=df["gmsl_true"] - df["posterior_mean"],
    cmap=colors.error_cmap,
)

# Scatter Plot
ax.scatter(
    df["gmsl_true"] - df["ssh_estimation"],
    df["gmsl_true"] - df["posterior_mean"],
    alpha=0.2,
    color=colors.primary_error,
    label="Inversion run",
    marker=".",
)

# Axis Labels with Dynamic Coloring
ax.set_xlabel(
    "Old Method (SSH) Bias [True GMSL - SSH Estimation] (mm)",
    color=colors.old_method,
    fontweight="bold",
)
ax.set_ylabel(
    "New Method (Bayesian) Bias [True GMSL - Posterior Mean] (mm)",
    color=colors.new_method,
    fontweight="bold",
)

# Limits and Guides
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)
ax.axhline(
    0,
    color=colors.new_method,
    linestyle="--",
    alpha=0.5,
    label="Zero New Method Bias",
)
ax.axvline(
    0,
    color=colors.old_method,
    linestyle="--",
    alpha=0.5,
    label="Zero Old Method Bias",
)

# Title with mixed colors requires a bit of a trick if you want
# different words to have different colors.
# For a standard single-color title:
ax.set_title(
    f"Method Biases: Old vs. New (no. inversion = {len(df)})",
    color=colors.true,
    fontweight="bold",
)

ax.xaxis.label.set_color(colors.old_method)
ax.tick_params(axis="x", colors=colors.old_method)

ax.yaxis.label.set_color(colors.new_method)
ax.tick_params(axis="y", colors=colors.new_method)

ax.legend()
fig.savefig("bias_comparison_contour.pdf", dpi=600)


# %%
# sns contour plot
max_val = max(
    np.abs(df["gmsl_true"] - df["ssh_estimation"]).max(),
    np.abs(df["gmsl_true"] - df["posterior_mean"]).max(),
)

fig, ax = plt.subplots(figsize=(3.25, 3.25))

# sns scatter plot, where hue is one color if true value is postiive, and another color if true value is negative

sns.scatterplot(
    x=df["gmsl_true"] - df["ssh_estimation"],
    y=df["gmsl_true"] - df["posterior_mean"],
    hue=df["gmsl_true"] > 0,
    palette=[colors.true, colors.primary_error],
    alpha=1,
)

# kdes for both case

sns.kdeplot(
    x=df[df["gmsl_true"] > 0]["gmsl_true"]
    - df[df["gmsl_true"] > 0]["ssh_estimation"],
    y=df[df["gmsl_true"] > 0]["gmsl_true"]
    - df[df["gmsl_true"] > 0]["posterior_mean"],
    cmap=colors.error_cmap,
    label="True GMSL > 0",
)

sns.kdeplot(
    x=df[df["gmsl_true"] <= 0]["gmsl_true"]
    - df[df["gmsl_true"] <= 0]["ssh_estimation"],
    y=df[df["gmsl_true"] <= 0]["gmsl_true"]
    - df[df["gmsl_true"] <= 0]["posterior_mean"],
    cmap="Greys",
    label="True GMSL <= 0",
)

# Axis Labels with Dynamic Coloring
ax.set_xlabel(
    "Old Method (SSH) Bias [True GMSL - SSH Estimation] (mm)",
    color=colors.old_method,
    fontweight="bold",
)
ax.set_ylabel(
    "New Method (Bayesian) Bias [True GMSL - Posterior Mean] (mm)",
    color=colors.new_method,
    fontweight="bold",
)

# Limits and Guides
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)

# Title with mixed colors requires a bit of a trick if you want
# different words to have different colors.
# For a standard single-color title:
ax.set_title(
    f"Method Biases: Old vs. New (no. inversion = {len(df)})",
    color=colors.true,
    fontweight="bold",
)

ax.xaxis.label.set_color(colors.old_method)
ax.tick_params(axis="x", colors=colors.old_method)

ax.yaxis.label.set_color(colors.new_method)
ax.tick_params(axis="y", colors=colors.new_method)

ax.legend()

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
sns.kdeplot(
    x=z_score_ssh,
    y=z_score_posterior,
    cmap=colors.error_cmap,
)

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
    color=colors.old_method,
    fontweight="bold",
)
ax.set_ylabel(
    "New Method (Bayesian) z-score",
    color=colors.new_method,
    fontweight="bold",
)

# Limits and Guides
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)
ax.axhline(
    0,
    color=colors.new_method,
    linestyle="--",
    alpha=0.5,
    label="Zero New Method Bias",
)
ax.axvline(
    0,
    color=colors.old_method,
    linestyle="--",
    alpha=0.5,
    label="Zero Old Method Bias",
)

# Title with mixed colors requires a bit of a trick if you want
# different words to have different colors.
# For a standard single-color title:
ax.set_title(
    f"Methods z-score: Old vs. New (no. inversion = {len(df)})",
    color=colors.true,
    fontweight="bold",
)

ax.xaxis.label.set_color(colors.old_method)
ax.tick_params(axis="x", colors=colors.old_method)

ax.yaxis.label.set_color(colors.new_method)
ax.tick_params(axis="y", colors=colors.new_method)

ax.legend()
fig.savefig("z_score_comparison_contour.pdf", dpi=600)


# %%

# Pre-calculating required metrics
df["error_ssh"] = df["ssh_estimation"] - df["gmsl_true"]
df["error_bayesian"] = (
    df["posterior_mean"] - df["gmsl_true"]
)

df["abs_error_ssh"] = df["error_ssh"].abs()
df["abs_error_bayesian"] = df["error_bayesian"].abs()

df["pct_error_bayesian"] = (
    df["error_bayesian"] / df["gmsl_true"]
) * 100

# %% [markdown]
# ## 1. Residual Comparison (Method 1 vs Method 2)
# Calculating the difference of the methods from the true value.
# This shows which method is more "centered" on zero.

# %%
plt.figure(figsize=(3.25, 3.25))
sns.kdeplot(
    df["error_ssh"],
    fill=True,
    label="Old Method (SSH) Bias",
    color=colors.old_method,
)
sns.kdeplot(
    df["error_bayesian"],
    fill=True,
    label="New Method (Bayesian) Bias",
    color=colors.new_method,
)

# calculate the mean of the new method biases

mean = df["error_bayesian"].mean()
# plot a "hypothetical" gaussian with that mean and the std dev as the new method std dev

std_dev = df["posterior_std_dev"].mean()

x = np.linspace(mean - 4 * std_dev, mean + 4 * std_dev, 100)
y = gaussian(x, mean, std_dev)
plt.plot(
    x,
    y,
    label="Hypothetical Gaussian Centered in Bias",
    color=colors.new_method,
    linestyle="--",
)


plt.axvline(
    0, color=colors.true, linestyle="--", label="Zero Bias"
)
plt.title(
    f"Distribution of Residuals (Estimate - Truth) [n={len(df)}]"
)
plt.xlabel("Error (mm)")
plt.ylabel("Density")
plt.legend()


# Calculate the number of times that the new method is within one sigma of the true value, compared to the old method
within_one_sigma_bayesian = (
    df["abs_error_bayesian"] <= df["posterior_std_dev"]
).sum()
within_one_sigma_ssh = (
    df["abs_error_ssh"]
    <= altimetry_std_dev / np.sqrt(number_points)
).sum()

print(
    f"New method within 1 sigma: {within_one_sigma_bayesian} / {len(df)} ({within_one_sigma_bayesian / len(df) * 100:.1f}%)"
)
print(
    f"Old method within 1 sigma: {within_one_sigma_ssh} / {len(df)} ({within_one_sigma_ssh / len(df) * 100:.1f}%)"
)
# %%
# do the residuals plot but with z scores instead of raw errors
colors.apply_style()
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
plt.axvline(
    0,
    color=colors.true,
    linestyle="--",
    label="Zero z-score",
)
plt.title(
    f"Distribution of z-scores [(Estimate - Truth) / Std Dev] [n={len(df)}]"
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
