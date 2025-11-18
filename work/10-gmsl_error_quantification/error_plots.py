# %%
import matplotlib as mpl
import pandas as pd
import seaborn as sns

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
        "ice_length_scale",
        "ice_gmsl_target_std",
        "net_ice_thickness_change",
        "odt_length_scale",
        "odt_standard_deviation",
        "altimetry_error_length_scale",
        "altimetry_error_amplitude",
        "altimetry_range",
    ],
    y_vars=["error_mean", "error_std"],
    height=4,
    aspect=1,
    kind="scatter",
)

# %%
# Sensitivity analysis using SALib
