# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the cleaned data
df = pd.read_csv("ice_results.csv")

# print the unique input means and stds to verify

print("Unique input means (mm):", df["mean_in"].unique())
print("Unique input stds (mm):", df["std_in"].unique())

# %%

for mean_val in df["mean_in"].unique():
    subset = df[df["mean_in"] == mean_val]
    # two subplots, left is true and estimate, other is error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(
        subset["std_in"],
        subset["true_gmsl_exp"],
        label="True GMSL Expectation",
    )
    ax1.fill_between(
        subset["std_in"],
        subset["true_gmsl_exp"] - subset["true_gmsl_std"],
        subset["true_gmsl_exp"] + subset["true_gmsl_std"],
        alpha=0.2,
        label="True GMSL Std Dev",
    )
    ax1.plot(
        subset["std_in"],
        subset["est_gmsl_exp"],
        label="Estimated GMSL Expectation",
    )
    ax1.fill_between(
        subset["std_in"],
        subset["est_gmsl_exp"] - subset["est_gmsl_std"],
        subset["est_gmsl_exp"] + subset["est_gmsl_std"],
        alpha=0.2,
        label="Estimated GMSL Std Dev",
    )
    # ax1.set_xscale("log")
    ax1.set_xlabel("Input GMSL Std Dev (mm)")
    ax1.set_ylabel("GMSL (mm)")
    ax1.set_title(
        f"GMSL Expectation vs Input Std Dev (Mean={mean_val:.3f} mm)"
    )
    ax1.legend()

    ax2.plot(
        subset["std_in"],
        subset["error_exp"],
        label="Error Expectation",
    )
    ax2.fill_between(
        subset["std_in"],
        subset["error_exp"] - subset["error_std"],
        subset["error_exp"] + subset["error_std"],
        alpha=0.2,
        label="Error Std Dev",
    )
    # ax1.set_xscale("log")
    ax2.set_xlabel("Input GMSL Std Dev (mm)")
    ax2.set_ylabel("GMSL Estimation Error (mm)")
    ax2.set_title(
        f"GMSL Estimation Error vs Input Std Dev (Mean={mean_val:.3f} mm)"
    )
    ax2.legend()
    plt.show()

# %%

for std_val in df["std_in"].unique():
    subset = df[df["std_in"] == std_val]
    # two subplots, left is true and estimate, other is error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(
        subset["mean_in"],
        subset["true_gmsl_exp"],
        label="True GMSL Expectation",
    )
    ax1.fill_between(
        subset["mean_in"],
        subset["true_gmsl_exp"] - subset["true_gmsl_std"],
        subset["true_gmsl_exp"] + subset["true_gmsl_std"],
        alpha=0.2,
        label="True GMSL Std Dev",
    )
    ax1.plot(
        subset["mean_in"],
        subset["est_gmsl_exp"],
        label="Estimated GMSL Expectation",
    )
    ax1.fill_between(
        subset["mean_in"],
        subset["est_gmsl_exp"] - subset["est_gmsl_std"],
        subset["est_gmsl_exp"] + subset["est_gmsl_std"],
        alpha=0.2,
        label="Estimated GMSL Std Dev",
    )
    ax1.set_xlabel("Input GMSL Mean (mm)")
    ax1.set_ylabel("GMSL (mm)")
    ax1.set_title(
        f"GMSL Expectation vs Input Mean (Std={std_val:.2e} mm)"
    )
    ax1.legend()

    ax2.plot(
        subset["mean_in"],
        subset["error_exp"],
        label="Error Expectation",
    )
    ax2.fill_between(
        subset["mean_in"],
        subset["error_exp"] - subset["error_std"],
        subset["error_exp"] + subset["error_std"],
        alpha=0.2,
        label="Error Std Dev",
    )
    ax2.set_xlabel("Input GMSL Mean (mm)")
    ax2.set_ylabel("GMSL Estimation Error (mm)")
    ax2.set_title(
        f"GMSL Estimation Error vs Input Mean (Std={std_val:.2e} mm)"
    )
    ax2.legend()
    plt.show()
    print("Error Means:")
    print(subset[["mean_in", "error_exp"]])

### NON DIM'ed STD and MEANS

# %%

for mean_val in df["mean_in"].unique():
    subset = df[df["mean_in"] == mean_val]
    # two subplots, left is true and estimate, other is error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(
        subset["std_in"],
        subset["true_gmsl_exp"] / mean_val,
        label="True GMSL Expectation",
    )
    ax1.fill_between(
        subset["std_in"],
        (
            subset["true_gmsl_exp"]
            - (subset["true_gmsl_std"] / subset["std_in"])
        )
        / mean_val,
        (
            subset["true_gmsl_exp"]
            + (subset["true_gmsl_std"] / subset["std_in"])
        )
        / mean_val,
        alpha=0.2,
        label="True GMSL Std Dev",
    )
    ax1.plot(
        subset["std_in"],
        subset["est_gmsl_exp"] / mean_val,
        label="Estimated GMSL Expectation",
    )
    ax1.fill_between(
        subset["std_in"],
        (
            subset["est_gmsl_exp"]
            - (subset["est_gmsl_std"] / subset["std_in"])
        )
        / mean_val,
        (
            subset["est_gmsl_exp"]
            + (subset["est_gmsl_std"] / subset["std_in"])
        )
        / mean_val,
        alpha=0.2,
        label="Estimated GMSL Std Dev",
    )
    # ax1.set_xscale("log")
    ax1.set_xlabel("Input GMSL Std Dev (mm)")
    ax1.set_ylabel("GMSL (mm)")
    ax1.set_title(
        f"GMSL Expectation vs Input Std Dev (Mean={mean_val:.3f} mm)"
    )
    ax1.legend()

    ax2.plot(
        subset["std_in"],
        subset["error_exp"] / mean_val,
        label="Error Expectation",
    )
    ax2.fill_between(
        subset["std_in"],
        (
            subset["error_exp"]
            - (subset["error_std"] / subset["std_in"])
        )
        / mean_val,
        (
            subset["error_exp"]
            + (subset["error_std"] / subset["std_in"])
        )
        / mean_val,
        alpha=0.2,
        label="Error Std Dev",
    )
    # ax1.set_xscale("log")
    ax2.set_xlabel("Input GMSL Std Dev (mm)")
    ax2.set_ylabel("GMSL Estimation Error (mm)")
    ax2.set_title(
        f"GMSL Estimation Error vs Input Std Dev (Mean={mean_val:.3f} mm)"
    )
    ax2.legend()
    plt.show()

# %%

for std_val in df["std_in"].unique():
    subset = df[df["std_in"] == std_val]
    # two subplots, left is true and estimate, other is error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(
        subset["mean_in"],
        subset["true_gmsl_exp"] / subset["mean_in"],
        label="True GMSL Expectation",
    )
    ax1.fill_between(
        subset["mean_in"],
        (
            subset["true_gmsl_exp"] / subset["mean_in"]
            - subset["true_gmsl_std"] / std_val
        ),
        (
            subset["true_gmsl_exp"] / subset["mean_in"]
            + subset["true_gmsl_std"] / std_val
        ),
        alpha=0.2,
        label="True GMSL Std Dev",
    )
    ax1.plot(
        subset["mean_in"],
        subset["est_gmsl_exp"] / subset["mean_in"],
        label="Estimated GMSL Expectation",
    )
    ax1.fill_between(
        subset["mean_in"],
        (
            subset["est_gmsl_exp"] / subset["mean_in"]
            - subset["est_gmsl_std"] / std_val
        ),
        (
            subset["est_gmsl_exp"] / subset["mean_in"]
            + subset["est_gmsl_std"] / std_val
        ),
        alpha=0.2,
        label="Estimated GMSL Std Dev",
    )
    ax1.set_xlabel("Input GMSL Mean (mm)")
    # units are now dimensionless, so no need to label as mm, but should note that it's a ratio of GMSL to input mean
    ax1.set_ylabel("GMSL / Input Mean")
    ax1.set_title(
        f"GMSL Expectation vs Input Mean (Std={std_val:.2e} mm)"
    )
    ax1.legend()

    ax2.plot(
        subset["mean_in"],
        subset["error_exp"] / subset["mean_in"],
        label="Error Expectation",
    )
    ax2.fill_between(
        subset["mean_in"],
        (
            subset["error_exp"] / subset["mean_in"]
            - subset["error_std"] / std_val
        ),
        (
            subset["error_exp"] / subset["mean_in"]
            + subset["error_std"] / std_val
        ),
        alpha=0.2,
        label="Error Std Dev",
    )
    ax2.set_xlabel("Input GMSL Mean (mm)")
    ax2.set_ylabel("GMSL Estimation Error / Input Mean")
    ax2.set_title(
        f"GMSL Estimation Error vs Input Mean (Std={std_val:.2e} mm)"
    )
    ax2.legend()
    plt.show()

# %%
# %%

# grid plot of non-dimensionalised error expectation (error / mean_in)
pivot_table = df.pivot(
    index="std_in", columns="mean_in", values="error_exp"
)
# Divide each column by its corresponding mean_in value
pivot_nondim = pivot_table.div(
    pivot_table.columns, axis="columns"
)

plt.figure(figsize=(8, 6))
# Use a diverging colormap centered at 0
abs_max = pivot_nondim.abs().max().max()
plt.imshow(
    pivot_nondim,
    aspect="auto",
    origin="lower",
    cmap="coolwarm",
    vmin=-abs_max,
    vmax=abs_max,
)
plt.colorbar(
    label="GMSL Estimation Error / Input Mean (dimensionless)"
)
plt.xlabel("Input GMSL Mean (mm)")
plt.ylabel("Input GMSL Std Dev (mm)")
plt.title(
    "Non-Dimensionalised GMSL Estimation Error vs Input Mean and Std Dev"
)
plt.xticks(
    ticks=np.arange(len(pivot_nondim.columns)),
    labels=[f"{m:.3f}" for m in pivot_nondim.columns],
)
plt.yticks(
    ticks=np.arange(len(pivot_nondim.index)),
    labels=[f"{s:.3f}" for s in pivot_nondim.index],
)
plt.show()

# %%
# grid plot of non-dimensionalised std of error (error_std / mean_in)

pivot_table_std = df.pivot(
    index="std_in", columns="mean_in", values="error_std"
)
# Divide each column by its corresponding mean_in value
pivot_nondim_std = pivot_table_std.div(
    pivot_table.columns, axis="columns"
)

plt.figure(figsize=(8, 6))
plt.imshow(
    pivot_nondim_std,
    aspect="auto",
    origin="lower",
    cmap="viridis",
)
plt.colorbar(
    label="GMSL Estimation Error Std Dev / Input Mean (dimensionless)"
)
plt.xlabel("Input GMSL Mean (mm)")
plt.ylabel("Input GMSL Std Dev (mm)")
plt.title(
    "Non-Dimensionalised GMSL Estimation Error Std Dev vs Input Mean and Std Dev"
)
# ticks every 4 values along

plt.show()
