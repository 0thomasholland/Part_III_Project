import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# Set plotting style
sns.set_theme(style="whitegrid")

text = "slc_no_translation_no_gmsl_weighting"
file_path = f"work/5-distribution_mapping/outputs/lmax_problems/{text}.csv"
output_dir = "work/5-distribution_mapping/outputs/lmax_problems/"

# Ensure output directory exists (though R script assumes it does)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    data_full = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Plot 1: Mean and Std Dev vs lmax (Dual Axis, Linear Scale)
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot Mean on left y-axis
color1 = "blue"
ax1.set_xlabel("lmax")
ax1.set_ylabel("Mean", color=color1)
# Using scatter for points
sns.scatterplot(
    x="lmax", y="mean", data=data_full, ax=ax1, color=color1, label="Mean", s=50
)
ax1.tick_params(axis="y", labelcolor=color1)
# Set x-ticks similar to R script: seq(0, max, 50)
max_lmax = data_full["lmax"].max()
ax1.set_xticks(np.arange(0, max_lmax + 50, 50))

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
color2 = "red"
# In R script: sec.axis = sec_axis(~ . / 1e10, name = "Std Dev")
# This means the values plotted on the primary axis (scaled up) correspond to raw values on secondary axis.
# The R script plots `std * 1e10`.
# To mimic the dual axis logic in Python manually:
scaled_std = data_full["std"] * 1e10
ax2.set_ylabel("Std Dev (x1e3)", color=color2)  # Label from R code legends
sns.scatterplot(
    x=data_full["lmax"],
    y=scaled_std,
    ax=ax2,
    color=color2,
    label="Std Dev (x1e3)",
    s=50,
)
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Mean and Std Dev vs lmax")

# Combine legends (a bit tricky with twin axes, manually placing them)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
# Note: scatterplot might not return Line2D objects directly for legend handles easily combined this way
# depending on seaborn version, but let's try to just let them be or place custom legend.
# Simpler approach:
fig.legend(
    lines_1 + lines_2,
    labels_1 + labels_2,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.9),
)

plt.tight_layout()
plt.show()
fig.savefig(f"{output_dir}{text}_lmax_plot.png")


# Plot 2: Mean and Std Dev vs lmax (Log Scale with regression)
fig2, ax1 = plt.subplots(figsize=(8, 6))

# Log scales
ax1.set_xscale("log")
ax1.set_yscale("log")

color1 = "blue"
ax1.set_xlabel("lmax")
ax1.set_ylabel("Mean", color=color1)

# Points
sns.scatterplot(
    x="lmax", y="mean", data=data_full, ax=ax1, color=color1, label="Mean", s=50
)

# Regression line for Mean (Power law fit for log-log plot)
df_pos_mean = data_full[(data_full["lmax"] > 0) & (data_full["mean"] > 0)]
if not df_pos_mean.empty:
    log_x = np.log(df_pos_mean["lmax"])
    log_y = np.log(df_pos_mean["mean"])
    m, c = np.polyfit(log_x, log_y, 1)
    x_fit = np.geomspace(df_pos_mean["lmax"].min(), df_pos_mean["lmax"].max(), 100)
    y_fit = np.exp(c) * x_fit**m
    ax1.plot(x_fit, y_fit, color=color1)

ax1.tick_params(axis="y", labelcolor=color1)

# Second y-axis
ax2 = ax1.twinx()
ax2.set_yscale("log")
color2 = "red"
ax2.set_ylabel("Std Dev (x1e3)", color=color2)

# Points
sns.scatterplot(
    x=data_full["lmax"],
    y=scaled_std,
    ax=ax2,
    color=color2,
    label="Std Dev (x1e3)",
    s=50,
)

# Regression line for Std (Power law fit)
df_pos_std = pd.DataFrame({"lmax": data_full["lmax"], "std": scaled_std})
df_pos_std = df_pos_std[(df_pos_std["lmax"] > 0) & (df_pos_std["std"] > 0)]

if not df_pos_std.empty:
    log_x = np.log(df_pos_std["lmax"])
    log_y = np.log(df_pos_std["std"])
    m, c = np.polyfit(log_x, log_y, 1)
    x_fit = np.geomspace(df_pos_std["lmax"].min(), df_pos_std["lmax"].max(), 100)
    y_fit = np.exp(c) * x_fit**m
    ax2.plot(x_fit, y_fit, color=color2)

ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Mean and Std Dev vs lmax")
fig2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.9))

plt.tight_layout()
plt.show()
fig2.savefig(f"{output_dir}{text}_lmax_plot1.png")


# Plot 3: Gaussian Distributions
# R Function:
# generate_gaussian <- function(mean, std, lmax) {
#   x <- seq(0 - 4*std, 0 + 4*std, length.out = 200)
#   y <- dnorm(x, mean = 0, sd = std)
#   data.frame(x = x, y = y, lmax = lmax)
# }


def generate_gaussian_df(row):
    mean_val = row["mean"]
    std_val = row["std"]
    lmax_val = row["lmax"]

    # R script generates x around 0 +/- 4*std, ignoring the actual mean for the x-range centering
    x = np.linspace(0 - 4 * std_val, 0 + 4 * std_val, 200)
    # R script uses mean=0 for density calculation
    y = norm.pdf(x, loc=0, scale=std_val)

    return pd.DataFrame({"x": x, "y": y, "lmax": lmax_val})


distribution_data_list = []
for index, row in data_full.iterrows():
    distribution_data_list.append(generate_gaussian_df(row))

distribution_data = pd.concat(distribution_data_list, ignore_index=True)

plt.figure(figsize=(8, 6))
# Using lineplot with hue mapping to 'lmax'
# Use 'viridis' colormap to match R's scale_color_viridis_c()
p3 = sns.lineplot(
    data=distribution_data,
    x="x",
    y="y",
    hue="lmax",
    palette="viridis",
    legend="full",
    linewidth=1,
)

p3.set_title(
    "Gaussian Distributions (only by standard deviation, ignores mean) by lmax"
)
p3.set_xlabel("Value")
p3.set_ylabel("Density")

plt.tight_layout()
plt.show()
p3.figure.savefig(f"{output_dir}{text}_lmax_gaussian_plot.png")
