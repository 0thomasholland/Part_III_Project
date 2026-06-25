# %%
"""
Cell-based OceanDynamics showcase.

Each pattern (Uniform, Synthetic, Data) shows:
- ODT spatial weights
- ODT load sample
- ODT SSH sample

Run with a cell-aware runner or as a script:
    python work/04_other_signal_effects/other_signals.py
"""

# %%
from __future__ import annotations

from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState

import matplotlib.pyplot as plt
import numpy as np
from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState

from pyslfp_extras.ocean_dynamics import OceanDynamics

# %%
# --- Setup: FingerPrint and operator ---
lmax = 128
fp = EarthState.from_defaults(lmax=lmax)

fp_op = FingerPrintOperator(fp, load_parameters=(2, 0.1 * fp.model.parameters.mean_sea_floor_radius,
), response_parameters=(2 + 1, 0.1 * fp.model.parameters.mean_sea_floor_radius,
))

# %%
# --- Pattern: Uniform ---
uniform_pattern = OceanDynamics.UniformPattern()
od_uniform = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.003,
    pattern=uniform_pattern,
)

# Spatial weights (ODT pattern)
plot(uniform_pattern.spatial_field(fp), cmap="Blues")

# Sample ODT load and SSH
np.random.seed(1)
uniform_load = od_uniform.load_measure.sample()
uniform_ssh = od_uniform.ssh_measure.sample()

plot(uniform_load)
plot(uniform_ssh)

# %%
# --- Pattern: Synthetic ---
synthetic_pattern = OceanDynamics.SyntheticPattern(
    point_multiplier=20.0
)
od_synthetic = OceanDynamics(
    finger_print=fp,
    finger_print_operator=fp_op,
    std=0.003,
    pattern=synthetic_pattern,
)

# Spatial weights (ODT pattern)
plot(
    synthetic_pattern.spatial_field(fp),
    cmap="YlOrRd",
)

# Sample ODT load and SSH
np.random.seed(2)
synthetic_load = (
    od_synthetic.load_measure.sample_pointwise_std(100)
)
synthetic_ssh = (
    od_synthetic.ssh_measure.sample_pointwise_std(100)
)

plot(synthetic_load)
plot(synthetic_ssh)

# %%
# --- Pattern: Data (altimetry dataset) ---
data_pattern = None
try:
    data_pattern = OceanDynamics.DataPattern()
except Exception as exc:
    print("DataPattern unavailable:", exc)

if data_pattern is not None:
    od_data = OceanDynamics(
        finger_print=fp,
        finger_print_operator=fp_op,
        std=0.003,
        length_scale=10000.0,
        pattern=data_pattern,
    )

    # Spatial weights (ODT pattern)
    plot(
        data_pattern.spatial_field(fp),
        cmap="YlGnBu",
    )

    # Sample ODT load and SSH
    np.random.seed(3)
    data_load = od_data.load_measure.sample_pointwise_std(
        100
    )
    data_ssh = od_data.ssh_measure.sample_pointwise_std(100)

    plot(data_load)
    plot(data_ssh)
    plot((data_load - data_ssh))

# %%
plt.show()
