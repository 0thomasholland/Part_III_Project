# %%

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from statsmodels.formula.api import ols

mpl.rcParams["figure.dpi"] = 600
# %%

data = pd.read_csv(
    "data_128_scaled.csv",
)

print(data.head())
data["error_mean_mm"] = data["error_mean"] * 1000
data["error_std_mm"] = data["error_std"] * 1000


# %%

# Linear regression using statsmodels to quantify the relationship between input parameters and error_mean and error_std
input_parameters = [
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_standard_deviation_factor",
    "altimetry_error_amplitude_factor",
    "altimetry_range",
]

target = ["error_mean_mm", "error_std_mm"]

for t in target:
    formula = f"{t} ~ " + " + ".join(input_parameters)
    model = ols(formula, data=data).fit()
    print(f"Linear regression results for {t}:")
    print(model.summary())
    plt.figure(figsize=(10, 6))
    plt.scatter(data[t], model.fittedvalues)
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Observed vs Predicted Values for {t}")
    plt.plot(
        [data[t].min(), data[t].max()],
        [data[t].min(), data[t].max()],
        color="red",
        linestyle="--",
    )
    plt.show()

# %%
# Do interaction terms for each pair of input parameters improve the model?
for t in target:
    interaction_terms = " + ".join(
        [
            f"{param1}:{param2}"
            for i, param1 in enumerate(input_parameters)
            for param2 in input_parameters[i + 1 :]
        ],
    )
    formula = f"{t} ~ " + " + ".join(input_parameters) + " + " + interaction_terms
    model = ols(formula, data=data).fit()
    print(
        f"Linear regression results with interaction terms for {t}:",
    )
    print(model.summary())
    plt.figure(figsize=(10, 6))
    plt.scatter(data[t], model.fittedvalues)
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.title(
        f"Observed vs Predicted Values with Interaction Terms for {t}",
    )
    plt.plot(
        [data[t].min(), data[t].max()],
        [data[t].min(), data[t].max()],
        color="red",
        linestyle="--",
    )
    plt.show()
    # print the values where p-value of any interaction term is < 0.05
    significant_interactions = [
        term
        for term in model.pvalues.index
        if ":" in term and model.pvalues[term] < 0.05
    ]
    if significant_interactions:
        print(
            f"Significant interaction terms for {t} (p < 0.05): {significant_interactions}",
        )
    else:
        print(f"No significant interaction terms for {t}")


# %%
# for each of the factor variables, plot that along the x axis, the gmsl_target_std along the y axis, and color by error_mean and error_std
# use

factor_variables = [
    "odt_standard_deviation_factor",
    "altimetry_error_amplitude_factor",
]

target = [
    "error_mean_mm",
    "error_std_mm",
]

ndata = data.copy()
for factor_var in factor_variables:
    for error_var in target:
        x_i = np.logspace(
            np.log10(ndata[factor_var].min()),
            np.log10(ndata[factor_var].max()),
            100,
        )
        y_i = np.linspace(
            ndata["ice_gmsl_target_std"].min(),
            ndata["ice_gmsl_target_std"].max(),
            100,
        )
        X, Y = np.meshgrid(x_i, y_i)
        Z = griddata(
            (ndata[factor_var], ndata["ice_gmsl_target_std"]),
            ndata[error_var],
            (X, Y),
            method="cubic",
        )
        plt.figure(figsize=(8, 6))
        cp = plt.pcolormesh(
            X,
            Y,
            Z,
            cmap="viridis",
            shading="auto",  # or 'gouraud' for smoother interpolation
        )
        plt.colorbar(cp, label=error_var)
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(factor_var)
        plt.ylabel("ice_gmsl_target_std")
        plt.title(
            f"{error_var} vs {factor_var} and ice_gmsl_target_std",
        )
        plt.show()

# %%
# for each of the factor variables, plot that along the x axis, the gmsl_target_std along the y axis, and color by error_mean and error_std
# use

factor_variables = [
    "odt_standard_deviation_factor",
    "altimetry_error_amplitude_factor",
]

target = [
    "error_std_normalised",
]

ndata = data.copy()
ndata["error_std_normalised"] = ndata["error_std"] / ndata["ice_gmsl_target_std"]
for factor_var in factor_variables:
    for error_var in target:
        x_i = np.logspace(
            np.log10(ndata[factor_var].min()),
            np.log10(ndata[factor_var].max()),
            100,
        )
        y_i = np.linspace(
            ndata["ice_gmsl_target_std"].min(),
            ndata["ice_gmsl_target_std"].max(),
            100,
        )
        X, Y = np.meshgrid(x_i, y_i)
        Z = griddata(
            (ndata[factor_var], ndata["ice_gmsl_target_std"]),
            ndata[error_var],
            (X, Y),
            method="cubic",
        )
        plt.figure(figsize=(8, 6))
        cp = plt.pcolormesh(
            X,
            Y,
            Z,
            cmap="viridis",
            shading="auto",  # or 'gouraud' for smoother interpolation
            # log colorscale
            norm=mpl.colors.LogNorm(),
        )
        plt.colorbar(cp, label=error_var)
        plt.xscale("log")
        # plt.yscale("log")
        plt.xlabel(factor_var)
        plt.ylabel("ice_gmsl_target_std")
        plt.title(
            f"{error_var} vs {factor_var} and ice_gmsl_target_std",
        )
        plt.show()


# %%

# histograpm or error_mean

plt.figure(figsize=(8, 6))
plt.hist(
    data["error_mean"],
    bins=30,
    density=True,
    alpha=0.6,
    color="g",
)
