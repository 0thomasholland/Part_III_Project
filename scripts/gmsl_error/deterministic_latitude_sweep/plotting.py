import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

resolution = 75  # number of points from -90 to 90 degrees
load_radius = [0.2, 1, 5, 10]  # degrees
load_thickness_change = -100  # meters

satellite_range = np.round(np.linspace(1, 90, resolution), 2)
latitude = np.round(np.linspace(-90, 90, resolution), 2)

error_output = pd.read_csv(
    "load_lat_error_output.csv",
)
print(error_output)
# %% [markdown]
# ## Plotting results
#
# Plotting as a heatmap of error vs load latitude and satellite coverage

# %%
# Get unique load radius values
unique_load_radii = error_output["load_radius"].unique()
n_subplots = len(unique_load_radii)

# Calculate subplot layout (rows x cols)
n_cols = min(2, n_subplots)  # Max 3 columns
n_rows = int(np.ceil(n_subplots / n_cols))

# First pass: determine global vmin and vmax across all load radii
vmin = float("inf")
vmax = float("-inf")
for load_rad in unique_load_radii:
    data_subset = error_output[
        error_output["load_radius"] == load_rad
    ]
    pivot_table = data_subset.pivot(
        index="latitude",
        columns="satellite_range",
        values="error",
    )
    vmin = min(vmin, pivot_table.min().min())
    vmax = max(vmax, pivot_table.max().max())

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(11, 8),
)

# Flatten axes array for easier iteration
if n_subplots == 1:
    axes = np.array([axes])
else:
    axes = axes.flatten()

# Create a plot for each load radius
for idx, load_rad in enumerate(sorted(unique_load_radii)):
    ax = axes[idx]

    # Filter data for this load radius
    data_subset = error_output[
        error_output["load_radius"] == load_rad
    ]

    pivot_table = data_subset.pivot(
        index="latitude",
        columns="satellite_range",
        values="error",
    )

    im = ax.imshow(
        pivot_table,
        aspect="auto",
        origin="lower",
        cmap="plasma",
        vmin=vmin,  # Set consistent color scale
        vmax=vmax,  # Set consistent color scale
    )

    plt.colorbar(im, ax=ax, label="Relative Error (%)")
    ax.set_xlabel("Satellite Coverage Latitude (±degrees)")
    ax.set_ylabel("Load Latitude (degrees)")
    ax.set_title(f"Relative Error for {load_rad}° load band")

    # Show all ticks and label them with the respective list entries
    max_ticks = 7

    # X-axis ticks (satellite range)
    if len(satellite_range) <= max_ticks + 1:
        x_tick_indices = list(range(len(satellite_range)))
    else:
        step = len(satellite_range) // max_ticks
        x_tick_indices = list(range(0, len(satellite_range), step))
        if x_tick_indices[-1] != len(satellite_range) - 1:
            x_tick_indices.append(len(satellite_range) - 1)

    # Y-axis ticks (latitude)
    if len(latitude) <= max_ticks:
        y_tick_indices = list(range(len(latitude)))
    else:
        middle_idx = np.argmin(np.abs(latitude))
        step = len(latitude) // (max_ticks - 2)
        y_tick_indices = list(range(0, len(latitude), step))
        if middle_idx not in y_tick_indices:
            y_tick_indices.append(middle_idx)
        if y_tick_indices[-1] != len(latitude) - 1:
            y_tick_indices.append(len(latitude) - 1)
        y_tick_indices = sorted(set(y_tick_indices))

    ax.set_xticks(x_tick_indices)
    ax.set_xticklabels(
        [satellite_range[i] for i in x_tick_indices],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(y_tick_indices)
    ax.set_yticklabels([latitude[i] for i in y_tick_indices])

# Hide any unused subplots
for idx in range(n_subplots, len(axes)):
    axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig("ice_load_latitude_coverage_sweep_errors.pdf", dpi=600)
plt.show()

# %%
# plot of error for 1 degree error minus 10 degrees error

difference_data = error_output[
    error_output["load_radius"].isin([1, 10])
]
difference_pivot = difference_data.pivot_table(
    index="latitude",
    columns=["satellite_range", "load_radius"],
    values="error",
)

# Extract error grids for load_radius 1 and 10 across all satellite ranges
err_r1 = difference_pivot.xs(1, level="load_radius", axis=1)
err_r10 = difference_pivot.xs(10, level="load_radius", axis=1)
difference = err_r1 - err_r10

fig, ax = plt.subplots(figsize=(11, 8))
vmax = np.nanmax(np.abs(difference.values))
im = ax.imshow(
    difference,
    aspect="auto",
    origin="lower",
    cmap="bwr",
    vmin=-vmax,
    vmax=vmax,
)
plt.colorbar(
    im,
    ax=ax,
    label="Relative Error Difference [(R1-R10) as %]",
)
ax.set_xlabel("Satellite Coverage Latitude (±degrees)")
ax.set_ylabel("Load Latitude (degrees)")
ax.set_title(
    "Relative Error Difference: 1° Load Band - 10° Load Band",
)

# Show all ticks and label them with the respective list entries
max_ticks = 7

# X-axis ticks (satellite range)
if len(satellite_range) <= max_ticks + 1:
    x_tick_indices = list(range(len(satellite_range)))
else:
    step = len(satellite_range) // max_ticks
    x_tick_indices = list(range(0, len(satellite_range), step))
    if x_tick_indices[-1] != len(satellite_range) - 1:
        x_tick_indices.append(len(satellite_range) - 1)

# Y-axis ticks (latitude)
if len(latitude) <= max_ticks:
    y_tick_indices = list(range(len(latitude)))
else:
    middle_idx = np.argmin(np.abs(latitude))
    step = len(latitude) // (max_ticks - 2)
    y_tick_indices = list(range(0, len(latitude), step))
    if middle_idx not in y_tick_indices:
        y_tick_indices.append(middle_idx)
    if y_tick_indices[-1] != len(latitude) - 1:
        y_tick_indices.append(len(latitude) - 1)
    y_tick_indices = sorted(set(y_tick_indices))

ax.set_xticks(x_tick_indices)
ax.set_xticklabels(
    [satellite_range[i] for i in x_tick_indices],
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)
ax.set_yticks(y_tick_indices)
ax.set_yticklabels([latitude[i] for i in y_tick_indices])

plt.tight_layout()
plt.savefig(
    "ice_load_latitude_coverage_sweep_difference_errors.pdf",
    dpi=600,
)
plt.show()
