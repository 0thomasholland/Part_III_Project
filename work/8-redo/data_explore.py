# %%

from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
from scipy.stats import norm

# %%
# import variable_input_data_initial.pkl from ./output

output_data = pd.DataFrame(
    load(
        path.join(
            path.dirname(path.abspath(__file__)),
            "output",
            "metrics_initial.pkl",
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
    norm(loc=mean, scale=std)
    for mean, std in zip(slc_mean, slc_std, strict=True)
]
ssh_gausses = [
    norm(loc=mean, scale=std)
    for mean, std in zip(ssh_mean, ssh_std, strict=True)
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

        sns.scatterplot(
            x=input_metric,
            y=error_metric,
            label="SSH GMSL",
            color="blue",
            ax=ax,
        )
        # sns.regplot(
        #     x=input_metric,
        #     y=error_metric,
        #     scatter=False,
        #     color="blue",
        #     ax=ax,
        # )

        sns.scatterplot(
            x=input_metric,
            y=[
                ssh_odt_kl,
                ssh_odt_mse,
                ssh_odt_cohens_d,
                ssh_odt_w2,
            ][i],
            label="SSH ODT GMSL",
            color="orange",
            ax=ax,
        )
        # sns.regplot(
        #     x=input_metric,
        #     y=[
        #         ssh_odt_kl,
        #         ssh_odt_mse,
        #         ssh_odt_cohens_d,
        #         ssh_odt_w2,
        #     ][i],
        #     scatter=False,
        #     color="orange",
        #     ax=ax,
        # )

        ax.set_xlabel(input_label)
        ax.set_ylabel(error_label)
        ax.legend()
# save
plt.tight_layout()
plt.savefig("error_metrics_vs_inputs.pdf")

# %%


# %%

mean_diff = np.average(
    output_data["ssh_gmsl_expectation"]
    - output_data["ssh_odt_gmsl_expectation"],
)
std_diff = np.average(
    output_data["ssh_gmsl_std"] - output_data["ssh_odt_gmsl_std"],
)

print(mean_diff)
print(std_diff)
# %%

plt.show()
