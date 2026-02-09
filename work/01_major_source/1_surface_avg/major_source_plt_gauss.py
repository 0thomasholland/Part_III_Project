# %%
import matplotlib.pyplot as plt
import numpy as np

from project.plots import (
    error_latitude_plot,
)

# %%

data_scalar = np.load(
    "major_source_altimetry_errors_scalar.npz"
)

# %%

latitudes_scalar = data_scalar["latitudes"]
gis_errors_scalar = data_scalar["gis_errors"] * 100
eais_errors_scalar = data_scalar["eais_errors"] * 100
wais_errors_scalar = data_scalar["wais_errors"] * 100

# %%

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(
    latitudes_scalar,
    gis_errors_scalar,
    label="GIS Relative Error",
    color="tab:blue",
)
ax1.plot(
    latitudes_scalar,
    eais_errors_scalar,
    label="EAIS Relative Error",
    color="tab:orange",
)
ax1.plot(
    latitudes_scalar,
    wais_errors_scalar,
    label="WAIS Relative Error",
    color="tab:green",
)
ax1.axhline(0, color="black", linestyle="-", linewidth=1)
ax1.axvline(66, color="red", linestyle="--", linewidth=1)
ax1.set_xlabel("Latitude (degrees)")
ax1.set_ylabel("Relative Error (%)")
ax1.set_title(
    "Altimetry GMSL Estimation Errors from Major Ice Sheet Sources (Scalar)"
)
ax1.legend()
ax1.grid()

# %%

fig1.savefig(
    "figures/major_source_altimetry_errors_scalar.png",
    dpi=600,
)
plt.close(fig1)

# %% ##### GAUSSIAN DATA SET #####

data_gauss = np.load(
    "major_source_altimetry_errors_gauss.npz"
)

altimetry_latitudes = data_gauss["altimetry_latitudes"]
gmsl_target_stds = data_gauss["gmsl_target_stds"]
gmsl_target_means = data_gauss["gmsl_target_means"]
sources = data_gauss["sources"]

error_means = data_gauss["error_means"]
error_stds = data_gauss["error_stds"]
estimate_means = data_gauss["estimate_means"]
estimate_stds = data_gauss["estimate_stds"]
true_gmsl_means = data_gauss["true_gmsl_means"]
true_gmsl_stds = data_gauss["true_gmsl_stds"]

n_sources = len(sources)
n_latitudes = len(altimetry_latitudes)
n_means = len(gmsl_target_means)
n_stds = len(gmsl_target_stds)

# Reshape from flat arrays to (n_sources, n_latitudes, n_means, n_stds)
error_means = error_means.reshape(
    n_sources, n_latitudes, n_means, n_stds
)
error_stds = error_stds.reshape(
    n_sources, n_latitudes, n_means, n_stds
)
estimate_means = estimate_means.reshape(
    n_sources, n_latitudes, n_means, n_stds
)
estimate_stds = estimate_stds.reshape(
    n_sources, n_latitudes, n_means, n_stds
)
true_gmsl_means = true_gmsl_means.reshape(
    n_sources, n_latitudes, n_means, n_stds
)
true_gmsl_stds = true_gmsl_stds.reshape(
    n_sources, n_latitudes, n_means, n_stds
)

source_labels = {"gis": "GIS", "wais": "WAIS", "eais": "EAIS"}
source_colors = {
    "gis": "tab:blue",
    "wais": "tab:green",
    "eais": "tab:orange",
}

# %%
# Plot 1: Error std vs latitude for each source (combined on one plot)
# Since gmsl_target_std=1 and gmsl_target_mean=0, the error std is the
# fractional uncertainty

for mi, gmsl_mean in enumerate(gmsl_target_means):
    for si, gmsl_std in enumerate(gmsl_target_stds):
        fig, ax = plt.subplots(figsize=(10, 6))
        for src_i, source in enumerate(sources):
            err_stds_subset = (
                error_stds[src_i, :, mi, si] * 1e3
            )  # convert to mm
            err_means_subset = (
                error_means[src_i, :, mi, si] * 1e3
            )
            ax.plot(
                altimetry_latitudes,
                err_means_subset,
                label=f"{source_labels[source]} Error Mean",
                color=source_colors[source],
            )
            ax.fill_between(
                altimetry_latitudes,
                err_means_subset - 2 * err_stds_subset,
                err_means_subset + 2 * err_stds_subset,
                color=source_colors[source],
                alpha=0.2,
                label=f"{source_labels[source]} +/-2 Std Dev",
            )
        # make array like latitudes that is full of gmsl mean and std
        gmsl_means = np.full_like(altimetry_latitudes, gmsl_mean * 1e3)
        gmsl_stds = np.full_like(altimetry_latitudes, gmsl_std)
        ax.plot(
            altimetry_latitudes,
            gmsl_means,
            label="True GMSL Mean",
            color="grey"
        )
        ax.fill_between(
            x=altimetry_latitudes,
            y1=gmsl_means - 2 * gmsl_stds,
            y2=gmsl_means + 2 * gmsl_stds,
            color="grey",
            alpha=0.2,
            label="True GMSL +/-2 Std Dev",
        )

        ax.set_xlabel("Latitude (degrees)")
        ax.set_ylabel("GMSL Estimation Error (mm)")
        ax.set_title(
            f"GMSL Estimation Error vs Latitude by Source\n(GMSL Mean: {gmsl_mean}, Std: {gmsl_std})"
        )
        ax.legend()
        ax.grid(alpha=0.3)

        fig.savefig(
            f"figures/major_source_gauss_errors_combined_mean_{gmsl_mean}_std_{gmsl_std}.png",
            dpi=600,
        )
        plt.close(fig)

# %%
# Plot 2: Per-source error latitude plots (true vs estimated + error)

for mi, gmsl_mean in enumerate(gmsl_target_means):
    for si, gmsl_std in enumerate(gmsl_target_stds):
        for src_i, source in enumerate(sources):
            err_means_subset = (
                error_means[src_i, :, mi, si] * 1e3
            )
            err_stds_subset = (
                error_stds[src_i, :, mi, si] * 1e3
            )
            est_means_subset = (
                estimate_means[src_i, :, mi, si] * 1e3
            )
            est_stds_subset = (
                estimate_stds[src_i, :, mi, si] * 1e3
            )
            true_means_subset = (
                true_gmsl_means[src_i, :, mi, si] * 1e3
            )
            true_stds_subset = (
                true_gmsl_stds[src_i, :, mi, si] * 1e3
            )

            fig, (ax1, ax2) = error_latitude_plot(
                latitude=altimetry_latitudes,
                true_mean=true_means_subset,
                true_std=true_stds_subset,
                true_label="True GMSL",
                estimate_mean=est_means_subset,
                estimate_std=est_stds_subset,
                estimate_label="Estimated GMSL",
                ax1_title=f"{source_labels[source]}: True vs Estimated GMSL",
                ax1_ylabel="GMSL (mm)",
                error_mean=err_means_subset,
                error_std=err_stds_subset,
                error_label="Estimation Error",
                ax2_title=f"{source_labels[source]}: GMSL Estimation Error",
                ax2_ylabel="Estimation Error (mm)",
                suptitle=f"{source_labels[source]} GMSL Estimation and Error vs Latitude\n(GMSL Mean: {gmsl_mean}, Std: {gmsl_std})",
                error_100_value=gmsl_std * 1000,
                error_100_value_name=f"1 GMSL Std Dev ({gmsl_std * 1000:.1f} mm)",
            )
            fig.savefig(
                f"figures/major_source_gauss_{source}_mean_{gmsl_mean}_std_{gmsl_std}.png",
                dpi=600,
            )
            plt.close(fig)

# %%
# Plot 3: Error std as fraction of signal std (since mean=0, std=1)
# This shows the fractional error for each source

fig, ax = plt.subplots(figsize=(10, 6))
for src_i, source in enumerate(sources):
    # With gmsl_target_std=1, error_stds directly gives the fractional error
    ax.plot(
        altimetry_latitudes,
        error_stds[src_i, :, 0, 0] * 100,
        label=f"{source_labels[source]}",
        color=source_colors[source],
    )
ax.axvline(
    66,
    color="red",
    linestyle="--",
    linewidth=1,
    label="66 degrees",
)
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("Error Std Dev (% of GMSL Std Dev)")
ax.set_title(
    "Fractional GMSL Estimation Error by Source"
)
ax.legend()
ax.grid(alpha=0.3)

fig.savefig(
    "figures/major_source_gauss_fractional_error.png",
    dpi=600,
)
plt.close(fig)
