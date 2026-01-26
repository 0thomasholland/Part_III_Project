# %%

from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from scipy.stats import norm

# %%
# import variable_input_data_initial.pkl from ./output

output_data = pd.DataFrame(
    load(
        path.join(
            path.dirname(path.abspath(__file__)),
            "output",
            "metrics_big.pkl",
        ),
    ),
)


# %%

ice_mean = output_data["net_ice_thickness_change"]
ice_std = output_data["ice_gmsl_target_std"]

slc_mean = output_data["slc_gmsl_expectation"]
slc_std = output_data["slc_gmsl_std"]
ssh_mean = output_data["ssh_gmsl_expectation"]
ssh_std = output_data["ssh_gmsl_std"]
ssh_odt_mean = output_data["ssh_odt_gmsl_expectation"]
ssh_odt_std = output_data["ssh_odt_gmsl_std"]

slc_gausses = [
    norm(loc=mean, scale=std) for mean, std in zip(slc_mean, slc_std, strict=True)
]
ssh_gausses = [
    norm(loc=mean, scale=std) for mean, std in zip(ssh_mean, ssh_std, strict=True)
]
ssh_odt_gausses = [
    norm(loc=mean, scale=std)
    for mean, std in zip(ssh_odt_mean, ssh_odt_std, strict=True)
]

# %%
# subplots, each with the three line plots of the three types of gausses for the same index, title containing the ice mean and std values
num_plots = len(output_data)
cols = 3
rows = (num_plots + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(30, 10 * rows))
for i in range(num_plots):
    row = i // cols
    col = i % cols
    ax = axes[row, col] if rows > 1 else axes[col]

    x = np.linspace(
        min(
            slc_gausses[i].ppf(0.001),
            ssh_gausses[i].ppf(0.001),
            ssh_odt_gausses[i].ppf(0.001),
        ),
        max(
            slc_gausses[i].ppf(0.999),
            ssh_gausses[i].ppf(0.999),
            ssh_odt_gausses[i].ppf(0.999),
        ),
        1000,
    )

    ax.plot(
        x,
        slc_gausses[i].pdf(x),
        label=f"SLC GMSL (μ {slc_mean[i]:.2f}, σ {slc_std[i]:.2e})",
        color="blue",
    )
    ax.plot(
        x,
        ssh_gausses[i].pdf(x),
        label=f"SSH GMSL (μ {ssh_mean[i]:.2f}, σ {ssh_std[i]:.2e})",
        color="orange",
    )
    ax.plot(
        x,
        ssh_odt_gausses[i].pdf(x),
        label=f"SSH ODT GMSL (μ {ssh_odt_mean[i]:.2f}, σ {ssh_odt_std[i]:.2e})",
        color="green",
    )

    ax.set_title(
        f"Ice thickness change: {ice_mean[i]:.2f}m, Target GMSL std: {ice_std[i]:.2e}",
    )
    ax.set_xlabel("GMSL Change (mm)")
    ax.set_ylabel("Probability Density")
    ax.legend()
# plt.tight_layout()

# %%
# extract error metrics from output_data, and key inputs

ssh_kl = output_data["ssh_gmsl_kl"]
ssh_odt_kl = output_data["ssh_odt_gmsl_kl"]

ssh_mse = output_data["ssh_gmsl_mse"]
ssh_odt_mse = output_data["ssh_odt_gmsl_mse"]

ssh_cohens_d = output_data["ssh_gmsl_cohens_d"]
ssh_odt_cohens_d = output_data["ssh_odt_gmsl_cohens_d"]

ssh_w2 = output_data["ssh_gmsl_wasserstein_distance"]
ssh_odt_w2 = output_data["ssh_odt_gmsl_wasserstein_distance"]

ice_length_scale = output_data["ice_length_scale"]
ice_gmsl_target_std = output_data["ice_gmsl_target_std"]
net_ice_thickness_change = output_data["net_ice_thickness_change"]
odt_length_scale = output_data["odt_length_scale"]
odt_amplitude = output_data["odt_amplitude_95_range"]

# make a subplot grid, so that columns is the key inputs, and rows is the error metrics, with seaborn scatterplots and regression lines if applicable for each subplot, plotting ssh and ssh_odt in different colors on the same axes

# %%
fig, axes = plt.subplots(4, 5, figsize=(30, 20))
for i, (error_metric, error_label) in enumerate(
    [
        (ssh_kl, "KL Divergence"),
        (ssh_mse, "Mean Squared Error"),
        (ssh_cohens_d, "Cohen's d"),
        (ssh_w2, "Wasserstein Distance"),
    ],
):
    for j, (input_metric, input_label) in enumerate(
        [
            (ice_length_scale, "Ice Length Scale"),
            (ice_gmsl_target_std, "Ice GMSL Target Std"),
            (net_ice_thickness_change, "Net Ice Thickness Change"),
            (odt_length_scale, "ODT Length Scale"),
            (odt_amplitude, "ODT Amplitude"),
        ],
    ):
        ax = axes[i, j]

        ax.scatter(
            input_metric,
            error_metric,
            label="SSH GMSL",
            color="blue",
            alpha=0.5,
        )

        ax.scatter(
            input_metric,
            [
                ssh_odt_kl,
                ssh_odt_mse,
                ssh_odt_cohens_d,
                ssh_odt_w2,
            ][i],
            label="SSH ODT GMSL",
            color="orange",
            alpha=0.5,
        )

        ax.set_xlabel(input_label)
        ax.set_ylabel(error_label)
        ax.legend()
# save
plt.tight_layout()
plt.savefig("error_metrics_vs_inputs.pdf")

# %%


# %%

mean_diff = np.average(
    output_data["ssh_gmsl_expectation"] - output_data["ssh_odt_gmsl_expectation"],
)
std_diff = np.average(
    output_data["ssh_gmsl_std"] - output_data["ssh_odt_gmsl_std"],
)

print(mean_diff)
print(std_diff)
# %%

odt_range_to_ice_gmsl_target_std = np.abs(
    output_data["odt_amplitude_95_range"] / output_data["ice_gmsl_target_std"],
)

odt_range_to_ice_net_ice_thickness_change = np.abs(
    100
    * output_data["odt_amplitude_95_range"]
    / output_data["net_ice_thickness_change"].replace(0, np.nan),
)


# %%
# plot scatterplot of odt_range_to_ice_gmsl_target_std vs kl, cohens d, and w2

fig, axes = plt.subplots(3, 2, figsize=(20, 15))
for i, (error_metric, error_label) in enumerate(
    [
        (ssh_kl, "KL Divergence"),
        (ssh_cohens_d, "Cohen's d"),
        (ssh_w2, "Wasserstein Distance"),
    ],
):
    ax = axes[i, 0]

    ax.scatter(
        odt_range_to_ice_gmsl_target_std,
        error_metric,
        label="SSH GMSL",
        color="blue",
        alpha=0.5,
    )

    ax.scatter(
        odt_range_to_ice_gmsl_target_std,
        [
            ssh_odt_kl,
            ssh_odt_cohens_d,
            ssh_odt_w2,
        ][i],
        label="SSH ODT GMSL",
        color="orange",
        alpha=0.5,
    )

    ax.set_xlabel("ODT Amplitude to Ice GMSL Target Std Ratio")
    ax.set_ylabel(error_label)
    ax.legend()

    ax = axes[i, 1]

    ax.scatter(
        odt_range_to_ice_net_ice_thickness_change,
        error_metric,
        label="SSH GMSL",
        color="blue",
        alpha=0.5,
    )

    ax.scatter(
        odt_range_to_ice_net_ice_thickness_change,
        [
            ssh_odt_kl,
            ssh_odt_cohens_d,
            ssh_odt_w2,
        ][i],
        label="SSH ODT GMSL",
        color="orange",
        alpha=0.5,
    )

    ax.set_xlabel(
        "ODT Amplitude to Net Ice Thickness Change Ratio (x100)",
    )
    ax.set_ylabel(error_label)
    ax.legend()
plt.savefig("error_metrics_vs_odt_ratios.pdf")

# %%

plt.show()


# %%

# CORNER PLOT TIMEEEEE

x_vars = [
    "ice_length_scale",
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_length_scale",
    "odt_amplitude_95_range",
]
y_vars = [
    "ssh_gmsl_kl",
    "ssh_gmsl_cohens_d",
    "ssh_gmsl_wasserstein_distance",
    "slc_gmsl_expectation",
    "ssh_gmsl_expectation",
    "slc_gmsl_std",
    "ssh_gmsl_std",
]

fig, axes = plt.subplots(
    len(y_vars), len(x_vars), figsize=(len(x_vars) * 3, len(y_vars) * 3)
)
for i, y_var in enumerate(y_vars):
    for j, x_var in enumerate(x_vars):
        axes[i, j].scatter(output_data[x_var], output_data[y_var], alpha=0.5)
        if i == len(y_vars) - 1:
            axes[i, j].set_xlabel(x_var)
        if j == 0:
            axes[i, j].set_ylabel(y_var)
plt.tight_layout()
plt.savefig("corner_plot.pdf")


# %%
# reform the dataset so that slc, ssh, and ssh_odt are a column called type, and there are columns for the expectation and std, and the metrics (e.g. kl, cohens d, wasserstein distance) are columns as well, nans for slc where not applicable
melted_data = pd.melt(
    output_data,
    id_vars=[
        "ice_length_scale",
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "odt_length_scale",
        "odt_amplitude_95_range",
    ],
    value_vars=[
        "slc_gmsl_expectation",
        "ssh_gmsl_expectation",
        "ssh_odt_gmsl_expectation",
        "slc_gmsl_std",
        "ssh_gmsl_std",
        "ssh_odt_gmsl_std",
        "ssh_gmsl_kl",
        "ssh_odt_gmsl_kl",
        "ssh_gmsl_cohens_d",
        "ssh_odt_gmsl_cohens_d",
        "ssh_gmsl_wasserstein_distance",
        "ssh_odt_gmsl_wasserstein_distance",
    ],
    var_name="type_stat",
    value_name="value",
)
melted_data["type"] = melted_data["type_stat"].apply(
    lambda x: x.split("_gmsl_")[0],
)
melted_data["stat"] = melted_data["type_stat"].apply(
    lambda x: x.split("_gmsl_")[1],
)
melted_data = melted_data.pivot_table(
    index=[
        "ice_length_scale",
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "odt_length_scale",
        "odt_amplitude_95_range",
        "type",
    ],
    columns="stat",
    values="value",
).reset_index()


# %%
print(melted_data.head())

# %%
# redo pairplot with melted data, using hue=type to differentiate between slc, ssh, and ssh_odt

x_vars = [
    "ice_length_scale",
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_length_scale",
    "odt_amplitude_95_range",
]
y_vars = [
    "expectation",
    "std",
    "kl",
    "cohens_d",
    "wasserstein_distance",
]

fig, axes = plt.subplots(
    len(y_vars), len(x_vars), figsize=(len(x_vars) * 3, len(y_vars) * 3)
)
groups = melted_data.groupby("type")
# Get default color cycle
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, y_var in enumerate(y_vars):
    for j, x_var in enumerate(x_vars):
        for k, (name, group) in enumerate(groups):
            color = colors[k % len(colors)]
            axes[i, j].scatter(
                group[x_var], group[y_var], label=name, color=color, alpha=0.5
            )
        if i == len(y_vars) - 1:
            axes[i, j].set_xlabel(x_var)
        if j == 0:
            axes[i, j].set_ylabel(y_var)
        if i == 0 and j == len(x_vars) - 1:
            axes[i, j].legend()

plt.tight_layout()
plt.savefig("corner_plot_melted.pdf")

# %%

# pick a random line from the original data frame, and plot the corresponding gausses
random_index = np.random.randint(0, len(output_data))
slc_gauss = slc_gausses[random_index]
ssh_gauss = ssh_gausses[random_index]
ssh_odt_gauss = ssh_odt_gausses[random_index]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(
    min(
        slc_gauss.ppf(0.001),
        ssh_gauss.ppf(0.001),
        ssh_odt_gauss.ppf(0.001),
    ),
    max(
        slc_gauss.ppf(0.999),
        ssh_gauss.ppf(0.999),
        ssh_odt_gauss.ppf(0.999),
    ),
    1000,
)
ax.plot(
    x,
    slc_gauss.pdf(x),
    label=f"SLC GMSL (μ {slc_mean[random_index]:.2f}, σ {slc_std[random_index]:.2e})",
    color="blue",
)
ax.plot(
    x,
    ssh_gauss.pdf(x),
    label=f"SSH GMSL (μ {ssh_mean[random_index]:.2f}, σ {ssh_std[random_index]:.2e})",
    color="orange",
)
ax.plot(
    x,
    ssh_odt_gauss.pdf(x),
    label=f"SSH ODT GMSL (μ {ssh_odt_mean[random_index]:.2f}, σ {ssh_odt_std[random_index]:.2e})",
    color="green",
)
ax.set_title(
    f"Ice thickness change: {ice_mean[random_index]:.2f}m, Target GMSL std: {ice_std[random_index]:.2e}",
)
ax.set_xlabel("GMSL Change (mm)")
ax.set_ylabel("Probability Density")
ax.legend()
