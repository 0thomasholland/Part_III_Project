# %%
import matplotlib.pyplot as plt
import numpy as np

# %%

data_scalar = np.load("all_ice_sheets_altimetry_errors.npz")

# %%

latitudes_scalar = data_scalar["latitudes"]
numeric_errors_scalar = data_scalar["numeric_errors"]
relative_errors_scalar = data_scalar["relative_errors"]

# %%

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    latitudes_scalar,
    relative_errors_scalar,
    label="Relative Error",
    color="tab:blue",
)
ax1.set_xlabel("Latitude (˚)")
ax1.set_ylabel("Relative Error")
ax1.set_title(
    "Altimetry GMSL Estimation Errors for All Ice Sheets"
)

# %%

fig1.savefig(
    "figures/all_ice_sheets_altimetry_errors.png", dpi=600
)
plt.close(fig1)

# %% ##### GAUSSIAN DATA SET #####

data_gauss = np.load("all_ice_sheets_gauss_latitudes.npz")

gmsl_means_gauss = data_gauss["gmsl_means"]
gmsl_stds_gauss = data_gauss["gmsl_stds"]
error_means_gauss = data_gauss["error_means"]
error_stds_gauss = data_gauss["error_stds"]
latitudes_gauss = data_gauss["latitudes"]
estimate_means_gauss = data_gauss["estimate_means"]
estimate_stds_gauss = data_gauss["estimate_stds"]


# %%

# for each unique gmsl values, plot the error means with fill for 2x standard deviations

unique_gmsl_means = np.unique(gmsl_means_gauss)
unique_gmsl_stds = np.unique(gmsl_stds_gauss)


for gmsl_mean in unique_gmsl_means:
    for gmsl_std in unique_gmsl_stds:
        mask = (gmsl_means_gauss == gmsl_mean) & (
            gmsl_stds_gauss == gmsl_std
        )
        latitudes_subset = latitudes_gauss[mask]
        error_means_subset = (
            error_means_gauss[mask] * 1e3
        )  # convert to mm
        error_stds_subset = (
            error_stds_gauss[mask] * 1e3
        )  # convert to mm

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            latitudes_subset,
            error_means_subset,
            label=f"Estimation error for GMSL Mean: {gmsl_mean}, Std: {gmsl_std}",
            color="tab:orange",
        )
        # ax.scatter(
        #     latitudes_subset,
        #     error_means_subset,
        #     color="tab:orange",)
        ax.fill_between(
            latitudes_subset,
            error_means_subset - 2 * error_stds_subset,
            error_means_subset + 2 * error_stds_subset,
            color="tab:orange",
            alpha=0.3,
            label="±2 Std Dev",
        )
        ax.set_xlabel("Latitude (˚)")
        ax.set_ylabel("GMSL Estimation Error (mm)")
        ax.set_title(
            f"GMSL Estimation Error vs Latitude\n(GMSL Mean: {gmsl_mean}, Std: {gmsl_std})"
        )
        ax.legend()

        # %%

        fig.savefig(
            f"figures/guassian_error_single/all_ice_sheets_gauss_errors_gmslmean_{gmsl_mean}_std_{gmsl_std}.png",
            dpi=600,
        )
        plt.close(fig)

# %%
# two sub plots, on the left the true and estimated gmsl each with 2x std fill, on the right the error with 2x std fill

for gmsl_mean in unique_gmsl_means:
    for gmsl_std in unique_gmsl_stds:
        mask = (gmsl_means_gauss == gmsl_mean) & (
            gmsl_stds_gauss == gmsl_std
        )
        latitudes_subset = latitudes_gauss[mask]
        estimate_means_subset = (
            estimate_means_gauss[mask] * 1e3
        )  # convert to mm
        estimate_stds_subset = (
            estimate_stds_gauss[mask] * 1e3
        )  # convert to mm
        error_means_subset = (
            error_means_gauss[mask] * 1e3
        )  # convert to mm
        error_stds_subset = (
            error_stds_gauss[mask] * 1e3
        )  # convert to mm

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 6)
        )

        # Left plot: True and Estimated GMSL
        ax1.plot(
            latitudes_subset,
            np.full_like(latitudes_subset, gmsl_mean * 1e3),
            label="True GMSL",
            color="tab:green",
        )
        ax1.fill_between(
            latitudes_subset,
            (gmsl_mean - 2 * gmsl_std) * 1e3,
            (gmsl_mean + 2 * gmsl_std) * 1e3,
            color="tab:green",
            alpha=0.3,
            label="True ±2 Std Dev",
        )
        ax1.plot(
            latitudes_subset,
            estimate_means_subset,
            label="Estimated GMSL",
            color="tab:blue",
        )
        ax1.fill_between(
            latitudes_subset,
            estimate_means_subset
            - 2 * estimate_stds_subset,
            estimate_means_subset
            + 2 * estimate_stds_subset,
            color="tab:blue",
            alpha=0.3,
            label="Estimated ±2 Std Dev",
        )
        ax1.set_xlabel("Latitude (˚)")
        ax1.set_ylabel("GMSL (mm)")
        ax1.set_title("True vs Estimated GMSL")
        ax1.legend()

        # Right plot: Estimation Error
        ax2.plot(
            latitudes_subset,
            error_means_subset,
            label="Estimation Error",
            color="tab:orange",
        )
        ax2.fill_between(
            latitudes_subset,
            error_means_subset - 2 * error_stds_subset,
            error_means_subset + 2 * error_stds_subset,
            color="tab:orange",
            alpha=0.3,
            label="±2 Std Dev",
        )
        ax2.set_xlabel("Latitude (˚)")
        ax2.set_ylabel("GMSL Estimation Error (mm)")
        ax2.set_title("GMSL Estimation Error vs Latitude")
        ax2.legend()

        fig.savefig(
            f"figures/guassian_error_comparison/all_ice_sheets_gauss_comparison_gmslmean_{gmsl_mean}_std_{gmsl_std}.png",
            dpi=600,
        )
