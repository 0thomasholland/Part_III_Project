# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

mpl.rcParams["figure.dpi"] = 600

data = pd.read_csv(
    "gmsl_error_with_measurement_noise_results.csv",
)

# %%

# do a scatterplot matrix where y values are the error and standard deviations and the x values are the rest

# x values: error_mean, error_std
# y values: ice_length_scale" "ice_gmsl_target_std", "net_ice_thickness_change",    "odt_length_scale", "odt_standard_deviation", "altimetry_error_length_scale", "altimetry_error_amplitude", "altimetry_range",

sns.pairplot(
    data,
    x_vars=[
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "odt_standard_deviation",
        "altimetry_error_amplitude",
        "altimetry_range",
    ],
    y_vars=["error_mean", "error_std"],
    height=4,
    aspect=1,
    kind="scatter",
)

# %%
# for each of the input parameters, average the error_mean and error_std over the other parameters and plot each of the distributions using scipy norm

input_parameters = [
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_standard_deviation",
    "altimetry_error_amplitude",
    "altimetry_range",
]

for param in input_parameters:
    grouped = data.groupby(param).agg(
        {"error_mean": "mean", "error_std": "mean"},
    )
    # find the max and min y values of the distributions via 4* standard deviation
    xmax = (
        grouped["error_mean"].max() + 4 * grouped["error_std"].max()
    )
    xmin = (
        grouped["error_mean"].min() - 4 * grouped["error_std"].max()
    )
    x = np.linspace(xmin, xmax, 1000)
    plt.figure(figsize=(8, 6))
    for i, row in grouped.iterrows():
        mean = row["error_mean"]
        std = row["error_std"]
        plt.plot(
            x,
            norm.pdf(x, mean, std),
            label=f"{param}={i:.4f}",
        )
    plt.title(f"GMSL Error Distribution varying {param}")
    plt.xlabel("GMSL Error (m)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()
