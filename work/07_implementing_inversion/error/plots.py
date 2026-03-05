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

# %% [markdown]
# ## 2. Calibration Plot: Truth vs. Estimate
# This checks for systematic bias. Points should lie on the diagonal.

# %%
fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    df["gmsl_true"],
    df["posterior_mean"],
    alpha=0.6,
    edgecolors="w",
    label="Inversion Samples",
)
ax.scatter(
    df["gmsl_true"],
    df["ssh_estimation"],
    alpha=0.6,
    edgecolors="w",
    label="SSH Estimation",
)

# Reference diagonal line
mn, mx = df["gmsl_true"].min(), df["gmsl_true"].max()
ax.plot(
    [mn, mx],
    [mn, mx],
    "k--",
    alpha=0.8,
    label="1:1 Ideal Line",
)

ax.set_xlabel("True GMSL (mm)")
ax.set_ylabel("Posterior Mean Estimate (mm)")
ax.set_title("Calibration: Is the Inversion Biased?")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.7)
plt.show()

# %%
# sns contour plot

max_val = max(
    np.abs(df["gmsl_true"] - df["ssh_estimation"]).max(),
    np.abs(df["gmsl_true"] - df["posterior_mean"]).max(),
)

fig, ax = plt.subplots(figsize=(7, 7))
sns.kdeplot(
    x=df["gmsl_true"] - df["ssh_estimation"],
    y=df["gmsl_true"] - df["posterior_mean"],
    cmap="Blues",
    alpha=0.5,
)

ax.scatter(
    df["gmsl_true"] - df["ssh_estimation"],
    df["gmsl_true"] - df["posterior_mean"],
    alpha=0.2,
    color="black",
    label="Data Points",
)

ax.set_xlabel(
    "Old Method (SSH) Bias [True GMSL - SSH Estimation] (mm)"
)
ax.set_ylabel(
    "New Method (Bayesian) Bias [True GMSL - Posterior Mean] (mm)"
)
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)
ax.axhline(0, color="k", linestyle="--", alpha=0.2)
ax.axvline(0, color="k", linestyle="--", alpha=0.2)
ax.legend()
ax.set_title(
    "Method Biases: Old Method (SSH) vs. New Method (Bayesian)"
)
plt.show()

# %%
# sns contour plot

max_val = max(
    np.abs(df["gmsl_true"] - df["ssh_estimation"]).max(),
    np.abs(df["gmsl_true"] - df["posterior_mean"]).max(),
)

fig, ax = plt.subplots(figsize=(7, 7))


ax.scatter(
    df["gmsl_true"] - df["ssh_estimation"],
    df["gmsl_true"] - df["posterior_mean"],
    alpha=0.2,
    color="black",
    label="Data Points",
)

ax.errorbar(
    df["gmsl_true"] - df["ssh_estimation"],
    df["gmsl_true"] - df["posterior_mean"],
    yerr=df["posterior_std_dev"],
    fmt="none",
    ecolor="red",
    alpha=0.3,
    label="Posterior Std Dev",
)

ax.set_xlabel(
    "Old Method (SSH) Bias [True GMSL - SSH Estimation] (mm)"
)
ax.set_ylabel(
    "New Method (Bayesian) Bias [True GMSL - Posterior Mean] (mm)"
)
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_ylim(-max_val * 1.1, max_val * 1.1)
ax.axhline(0, color="k", linestyle="--", alpha=0.2)
ax.axvline(0, color="k", linestyle="--", alpha=0.2)
ax.legend()
ax.set_title(
    "Method Biases: Old Method (SSH) vs. New Method (Bayesian)"
)
plt.show()


# %%

print(np.mean(df["posterior_std_dev"]))
altimetry_std_dev = 0.001 * 1000
number_points = 1402
old_method_error = altimetry_std_dev / np.sqrt(
    number_points
)
print(f"Old method error: {old_method_error:.4f} mm")
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
plt.figure(figsize=(10, 5))
sns.kdeplot(
    df["error_ssh"],
    fill=True,
    label="Old Method (SSH) Bias",
    color="gray",
)
sns.kdeplot(
    df["error_bayesian"],
    fill=True,
    label="New Method (Bayesian) Bias",
    color="blue",
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
    color="red",
    linestyle="--",
)

plt.axvline(0, color="black", linestyle="--")
plt.title(
    f"Distribution of Residuals (Estimate - Truth) [n={len(df)}]"
)
plt.xlabel("Error (mm)")
plt.ylabel("Density")
plt.legend()
plt.show()
