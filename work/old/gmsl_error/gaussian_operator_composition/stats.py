# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

data = pd.read_csv(
    "gmsl_error_with_measurement_noise_results_lmax128.csv",
)
print(data.head())

# %%
# linear regression using statsmodels to quantify the relationship between input parameters and error_mean and error_std

input_parameters = [
    "ice_gmsl_target_std",
    "net_ice_thickness_change",
    "odt_standard_deviation",
    "altimetry_error_amplitude",
    "altimetry_range",
    "ice_length_scale",
    "odt_length_scale",
    "altimetry_error_length_scale",
]

results = {}

for target in ["error_mean", "error_std"]:
    X = data[input_parameters]
    y = data[target]
    X = sm.add_constant(X)  # adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    print(f"Linear regression results for {target}:")
    print(model.summary())
    results[target] = model.summary()

# %% plot the models linear fits
for target in ["error_mean", "error_std"]:
    X = data[input_parameters]
    y = data[target]
    X = sm.add_constant(X)  # adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions)
    plt.xlabel("Observed Values")
    plt.ylabel("Predicted Values")
    plt.title(f"Observed vs Predicted Values for {target}")
    plt.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        color="red",
        lw=2,
    )
    # plt.show()
    plt.savefig(f"observed_vs_predicted_{target}.png", dpi=600)
    plt.close()


# %%
# plot each input parameter against error_mean and error_std with the linear regression line where p value < 0.05

for target in ["error_mean", "error_std"]:
    # Fit the full model to get p-values
    X = data[input_parameters]
    y = data[target]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Filter significant parameters
    significant_params = [
        param for param in input_parameters if model.pvalues[param] < 0.05
    ]

    if not significant_params:
        print(f"No significant parameters for {target}")
        continue

    # Create subplots based on number of significant parameters
    n_sig = len(significant_params)
    n_cols = min(4, n_sig)
    n_rows = (n_sig + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
    )
    if n_sig == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_sig > 1 else [axes]

    for i, param in enumerate(significant_params):
        ax = axes[i]

        # Scatter plot
        ax.scatter(data[param], y, alpha=0.5)

        # Fit simple linear regression for this parameter
        X_simple = sm.add_constant(data[param])
        model_simple = sm.OLS(y, X_simple).fit()

        # Generate prediction line
        x_range = np.linspace(
            data[param].min(),
            data[param].max(),
            100,
        )
        X_pred = sm.add_constant(x_range)
        y_pred = model_simple.predict(X_pred)

        p_value = model.pvalues[param]
        ax.plot(
            x_range,
            y_pred,
            "r-",
            linewidth=2,
            label=f"p={p_value:.3f}",
        )
        ax.legend()
        ax.set_xlabel(param)
        ax.set_ylabel(target)

    plt.suptitle(f"Significant Input Parameters vs {target}")
    plt.tight_layout()
    # plt.show()


# %%
# run linear model for error_std vs ice_gmsl_target_std only
X_a = data[["ice_gmsl_target_std"]]
y_a = data["error_std"]
X_a = sm.add_constant(X_a)  # adds a constant term to the predictor
model = sm.OLS(y_a, X_a).fit()
print(
    "Linear regression results for error_std vs ice_gmsl_target_std:",
)
print(model.summary())


# %%
# test for interaction effects

formula = """error_mean ~ net_ice_thickness_change + altimetry_range +
             net_ice_thickness_change:altimetry_range +
             ice_gmsl_target_std + odt_standard_deviation +
             altimetry_error_amplitude"""


model = smf.ols(formula, data=data).fit()
print(model.summary())

# %%

# plot the residuals
residuals = model.resid
fitted = model.fittedvalues
plt.figure(figsize=(8, 6))
plt.scatter(fitted, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
# plt.show()
# plot true vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(data["error_mean"], fitted)
plt.plot(
    [data["error_mean"].min(), data["error_mean"].max()],
    [data["error_mean"].min(), data["error_mean"].max()],
    color="red",
    lw=2,
)
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title("Observed vs Predicted Values")
# # plt.show()
plt.savefig(
    "observed_vs_predicted_error_mean_with_interactions.png",
    dpi=600,
)


# %%


formula = """error_mean ~ net_ice_thickness_change +
             net_ice_thickness_change:altimetry_range"""


model = smf.ols(formula, data=data).fit()
print(model.summary())

# %%

# plot the residuals
residuals = model.resid
fitted = model.fittedvalues
plt.figure(figsize=(8, 6))
plt.scatter(fitted, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
# plt.show()
# plot true vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(data["error_mean"], fitted)
plt.plot(
    [data["error_mean"].min(), data["error_mean"].max()],
    [data["error_mean"].min(), data["error_mean"].max()],
    color="red",
    lw=2,
)
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title(
    "Observed vs Predicted Values\nmodel: error_mean ~ net_ice_thickness_change + net_ice_thickness_change:altimetry_range",
)
# plt.show()
# %%

ice_vals = np.linspace(
    data["net_ice_thickness_change"].min(),
    data["net_ice_thickness_change"].max(),
    50,
)
alt_vals = np.linspace(
    data["altimetry_range"].min(),
    data["altimetry_range"].max(),
    50,
)
ICE, ALT = np.meshgrid(ice_vals, alt_vals)

ERROR = -0.0027 * ICE + 4.439e-5 * ICE * ALT

# set the colorscale to be symmetric around zero
plt.show()

plt.contourf(
    ICE,
    ALT,
    ERROR,
    levels=20,
    cmap="coolwarm",
    vmin=-np.max(np.abs(ERROR)),
    vmax=np.max(np.abs(ERROR)),
)
plt.colorbar(label="Predicted error_mean")
plt.xlabel("Net Ice Thickness Change")
plt.ylabel("Altimetry Range")
plt.title("Bias as Function of Signal × Data Quality")
# plt.contour(ICE, ALT, ERROR, levels=[0], colors="black", linewidths=2)

plt.savefig(
    "predicted_error_mean_vs_ice_thickness_change_and_altimetry_range.png",
    dpi=600,
)
# plt.show()
# %%
# plot each input parameter against error_mean and error_std with the linear regression line where p value < 0.05 including interaction terms

for target in ["error_mean", "error_std"]:
    # Fit the full model to get p-values
    X = data[input_parameters]
    y = data[target]
    X_with_const = sm.add_constant(X)
    model_formula = " + ".join(input_parameters)
    # add interaction terms between all input parameters
    for i in range(len(input_parameters)):
        for j in range(i + 1, len(input_parameters)):
            model_formula += f" + {input_parameters[i]}:{input_parameters[j]}"
    model = smf.ols(f"{target} ~ {model_formula}", data=data).fit()
    # Filter significant parameters
    significant_params = [
        param for param in input_parameters if model.pvalues[param] < 0.05
    ]

    if not significant_params:
        print(f"No significant parameters for {target}")
        continue

    # Create subplots based on number of significant parameters
    n_sig = len(significant_params)
    n_cols = min(4, n_sig)
    n_rows = (n_sig + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
    )
    if n_sig == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_sig > 1 else [axes]

    for i, param in enumerate(significant_params):
        ax = axes[i]

        # Scatter plot
        ax.scatter(data[param], y, alpha=0.5)

        # Fit simple linear regression for this parameter
        X_simple = sm.add_constant(data[param])
        model_simple = sm.OLS(y, X_simple).fit()

        # Generate prediction line
        x_range = np.linspace(
            data[param].min(),
            data[param].max(),
            100,
        )
        X_pred = sm.add_constant(x_range)
        y_pred = model_simple.predict(X_pred)

        p_value = model.pvalues[param]
        ax.plot(
            x_range,
            y_pred,
            "r-",
            linewidth=2,
            label=f"p={p_value:.3f}",
        )
        ax.legend()
        ax.set_xlabel(param)
        ax.set_ylabel(target)

    plt.suptitle(f"Significant Input Parameters vs {target}")
    plt.tight_layout()
    # plt.show()
