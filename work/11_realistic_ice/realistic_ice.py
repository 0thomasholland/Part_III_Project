# %%
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl
from dask.dataframe import melt
from pyshtools import SHGrid

lmax = 128
fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(
    version=sl.IceModel.ICE7G, date=0.0
)
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)


# %%

# plot with colorcet's blues

sl.plot(
    fp.ice_thickness,
    cmap=cc.cm.blues,
)

# %%


def activator(x, x_min, x_max):
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


activator = np.vectorize(activator)

melt_likelihood: SHGrid = fp.ice_thickness.copy()
# normalise to between 0 and 1
melt_likelihood: SHGrid = (
    fp.ice_thickness.max() - melt_likelihood
) / (fp.ice_thickness.max() - fp.ice_thickness.min())


sl.plot(
    melt_likelihood * fp.ice_projection(), cmap=cc.cm.blues
)

plt.hist(melt_likelihood.data.flatten(), bins=50)


melt_likelihood: SHGrid = fp.ice_thickness.copy()

melt_likelihood.data: SHGrid = activator(
    d := melt_likelihood.data,
    d.min(),
    d.max(),
)

sl.plot(
    melt_likelihood * fp.ice_projection(), cmap=cc.cm.blues
)

# %%
