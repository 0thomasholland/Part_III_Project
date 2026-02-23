# %%
import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl
from patsy.builtins import Q

lmax = 128
fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(
    version=sl.IceModel.ICE7G, date=0.0
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
    k = 1.0  # Upper asymptote (Thin ice = 1 probability)
    b = 10.0  # Steepness
    m = 0.45  # Threshold (where the drop-off happens)
    nu = 0.75  # Asymmetry (adjusts how 'sharp' the turn is)

    # Note: We use (_x - M) to make probability drop as thickness increases
    _x = a + (k - a) / (1 + np.exp(b * (_x - m))) ** (
        1 / nu
    )
    return _x


input = np.linspace(data.min(), data.max(), 100)
fig, ax = plt.subplots()
ax.plot(
    input,
    activator_logistic(input, data.min(), data.max()),
    label="Logistic",
)
ax.plot(
    input,
    activator_cloglog(input, data.min(), data.max()),
    label="Cloglog",
)
ax.plot(
    input,
    activator_richards(input, data.min(), data.max()),
    label="Richard's Curve",
)
ax.legend()
ax.set_xlabel("Input (ice thickness in m)")
ax.set_ylabel("Output (melt probability)")

# %%

plt.hist(
    activator(x := data[data > 0], x.min(), x.max()),
    bins=50,
)
