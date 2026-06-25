# Auto-generated from notebook code cells.
# Source: notebooks/03 - Ocean Dynamics.ipynb

from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.show = lambda *args, **kwargs: None
print = lambda *args, **kwargs: None

def _save_all_figures(prefix):
    for index, figure_number in enumerate(
        plt.get_fignums(), start=1
    ):
        fig = plt.figure(figure_number)
        fig.savefig(
            FIGURES_DIR / f"{prefix}_{index:02d}.pdf",
            dpi=600,
            bbox_inches="tight",
        )
    plt.close("all")

# ---- Notebook code cell 1 ----
import colorcet as cc
import numpy as np

from pyslfp_extras.ocean_dynamics import OceanDynamics
from pyslfp_extras.plotting import plot

np.random.seed(423991)

lmax = 256
fp = EarthState.from_defaults(lmax=lmax)
fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

_save_all_figures("03_Ocean_Dynamics_cell_1")

# ---- Notebook code cell 2 ----
od_uniform = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.03,  # Target std of SSH variability in m
    length_scale=0.01 * fp.model.parameters.mean_sea_floor_radius,
    pattern=OceanDynamics.UniformPattern(),
)

uniform_ssh_measure = od_uniform.ssh_measure
uniform_sample = uniform_ssh_measure.sample()

fig, ax, _ = plot(
    uniform_sample * fp.ocean_projection(),
    symmetric=True,
    colorbar_label="Uniform-prior Ocean Dynamics SSH (m)",
    figsize=(3.25, 2.5),
)
ax.set_title("Sample from Uniform Ocean Dynamics Prior")
plt.show()

_save_all_figures("03_Ocean_Dynamics_cell_2")

# ---- Notebook code cell 3 ----
data_pattern = OceanDynamics.DataPattern()
field = data_pattern.spatial_field(fp)

fig, ax, _ = plot(
    field * fp.ocean_projection(),
    colorbar_label="Variability multiplier (from SLA std)",
    cmap=cc.cm.bmw_r,
    figsize=(3.25, 2.5),
)
ax.set_title("Data-derived SSH Variability Pattern")
plt.show()

_save_all_figures("03_Ocean_Dynamics_cell_3")

# ---- Notebook code cell 4 ----
od_data = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.03,  # Target std of SSH variability in m
    length_scale=0.01 * fp.model.parameters.mean_sea_floor_radius,
    pattern=data_pattern,
)

data_ssh_measure = od_data.ssh_measure
data_sample = data_ssh_measure.sample()

max_val = np.max(np.abs(data_sample.data))

fig, ax, _ = plot(
    data_sample * fp.ocean_projection(),
    symmetric=True,
    vmax=max_val,
    vmin=-max_val,
    colorbar_label="Data-weighted Ocean Dynamics SSH (m)",
    figsize=(3.25, 2.5),
)
ax.set_title("Sample from Data-Weighted Prior")
plt.show()

_save_all_figures("03_Ocean_Dynamics_cell_4")
