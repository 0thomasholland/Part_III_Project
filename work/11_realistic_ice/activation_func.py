# %%
import matplotlib.pyplot as plt
import numpy as np
from pyslfp.state import EarthState

lmax = 256
fp = EarthState.from_defaults(
    lmax=lmax, version="ICE7G", date=0.0
)
# %%

data = fp.ice_thickness.data.flatten()

plt.hist(data[data > 0], bins=50)

# %%
def activator_logistic(x, x_min, x_max):
    _x = (x - x_min) / (x_max - x_min)
    _x = 1 / (1 + np.exp(-10 * (0.45 - _x)))
    return _x

def activator_cloglog(x, x_min, x_max):
    _x = (x - x_min) / (x_max - x_min)
    _x = np.exp(-np.exp(-10 * (-_x + 0.45)))
    return _x

def activator_richards(x, x_min, x_max):
    # Standardize input: 0 at min thickness, 1 at max thickness
    _x = (x - x_min) / (x_max - x_min)

    # Parameters for a clean 0-to-1 probability curve
    a = 0.1  # Lower asymptote (Thick ice = 0 probability)
    k = 0.9  # Upper asymptote (Thin ice = 1 probability)
    b = 10.0  # Steepness
    m = 0.45  # Threshold (where the drop-off happens)
    nu = 0.75  # Asymmetry (adjusts how 'sharp' the turn is)

    # Note: We use (_x - M) to make probability drop as thickness increases
    _x = a + (k - a) / (1 + np.exp(b * (_x - m))) ** (
        1 / nu
    )
    return _x

input = np.linspace(data.min(), data.max(), 100)
fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(
#     input,
#     activator_logistic(input, data.min(), data.max()),
#     label="Logistic",
# )
# ax.plot(
#     input,
#     activator_cloglog(input, data.min(), data.max()),
#     label="Cloglog",
# )
ax.plot(
    input,
    activator_richards(input, data.min(), data.max()),
    label="Ice Melt Function",
    color="black",
)

ax.plot(
    input,
    1 - activator_richards(input, data.min(), data.max()),
    label="Firn Melt Function",
    color="black",
    linestyle="dashed",
)
ax.legend()
ax.set_xlabel("Input (ice thickness in m)")
ax.set_ylim(-0.0, 1.0)
ax.set_ylabel(
    "Output (melt field standard deviation multiplier)"
)
fig.savefig("figs/activator_func.pdf", dpi=600)

# %%

plt.hist(
    activator(x := data[data > 0], x.min(), x.max()),
    bins=50,
)
