# %%

import os

import matplotlib.pyplot as plt
import mpltern  # noqa: F401
import numpy as np
import pandas as pd

print(os.getcwd())

# %%

data = pd.read_csv("ternary_error_analysis_w_shift_hr.csv")

# remove any rows with negative fractions (these are invalid)
data = data[
    (data["G"] >= 0.0) & (data["W"] >= 0.0) & (data["E"] >= 0.0)
]

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="ternary")

top = data["G"].to_numpy()
left = data["W"].to_numpy()
right = data["E"].to_numpy()
value = data["error_std"].to_numpy()
vmin = np.min(value)
vmax = np.max(value)
cs = ax.tripcolor(
    top,
    left,
    right,
    value,
    shading="gouraud",
)
ax.set_tlabel("Greenland")
ax.set_llabel("West Antarctica")
ax.set_rlabel("East Antarctica")
ax.taxis.set_label_rotation_mode("horizontal")
ax.laxis.set_label_rotation_mode("horizontal")
ax.raxis.set_label_rotation_mode("horizontal")
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")

ax.set_title(
    "GMSL Error Standard Deviation (m) vs Ice Sheet Contribution Fractions",
)
colorbar = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.1)

fig.savefig(
    "gmsl_error_std_vs_ice_sheet_fractions_hr.png", dpi=600
)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="ternary")

top = data["G"].to_numpy()
left = data["W"].to_numpy()
right = data["E"].to_numpy()
value = data["true_std"].to_numpy()
vmin = np.min(value)
vmax = np.max(value)
cs = ax.tripcolor(
    top,
    left,
    right,
    value,
    shading="gouraud",
)
ax.set_tlabel("Greenland")
ax.set_llabel("West Antarctica")
ax.set_rlabel("East Antarctica")
ax.taxis.set_label_rotation_mode("horizontal")
ax.laxis.set_label_rotation_mode("horizontal")
ax.raxis.set_label_rotation_mode("horizontal")
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")

ax.set_title(
    "GMSL True Standard Deviation (m) vs Ice Sheet Contribution Fractions",
)
colorbar = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.1)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="ternary")

top = data["G"].to_numpy()
left = data["W"].to_numpy()
right = data["E"].to_numpy()
value = data["est_std"].to_numpy()
vmin = np.min(value)
vmax = np.max(value)
cs = ax.tripcolor(
    top,
    left,
    right,
    value,
    shading="gouraud",
)
ax.set_tlabel("Greenland")
ax.set_llabel("West Antarctica")
ax.set_rlabel("East Antarctica")
ax.taxis.set_label_rotation_mode("horizontal")
ax.laxis.set_label_rotation_mode("horizontal")
ax.raxis.set_label_rotation_mode("horizontal")
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")

ax.set_title(
    "GMSL Estimated Standard Deviation (m) vs Ice Sheet Contribution Fractions",
)
colorbar = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.1)

# %%


fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="ternary")

top = data["G"].to_numpy()
left = data["W"].to_numpy()
right = data["E"].to_numpy()
value = data["error_mean"].to_numpy()
vmin = np.min(value)
vmax = np.max(value)
cs = ax.tripcolor(
    top,
    left,
    right,
    value,
    shading="gouraud",
)
ax.set_tlabel("Greenland")
ax.set_llabel("West Antarctica")
ax.set_rlabel("East Antarctica")
ax.taxis.set_label_rotation_mode("horizontal")
ax.laxis.set_label_rotation_mode("horizontal")
ax.raxis.set_label_rotation_mode("horizontal")
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")

ax.set_title(
    "GMSL Error Mean (m) vs Ice Sheet Contribution Fractions",
)
colorbar = fig.colorbar(cs, ax=ax, orientation="horizontal", pad=0.1)
