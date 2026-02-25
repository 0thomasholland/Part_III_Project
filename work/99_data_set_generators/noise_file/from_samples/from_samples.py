# %%
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dill
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pygeoinf import GaussianMeasure, HilbertSpace
from pygeoinf.symmetric_space.circle import Sobolev
from pyshtools import SHGrid
from pyslfp import FingerPrint, IceModel, plot

# %%
lmax = 512
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

# This operator defines the Hilbert Space domain
fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)
domain = fp_op.domain

shgrid_dir = "../../duacs/shgrids_lmax512/"
samples = []

# 2. Load and transform to the correct Basis
for file in sorted(os.listdir(shgrid_dir)):
    if file.endswith(".pkl"):
        with open(
            os.path.join(shgrid_dir, file), "rb"
        ) as f:
            shgrid = dill.load(f)
            coeffs = shgrid.expand()
            # Convert SHCoeffs to a domain element
            # The domain likely expects the coefficient array directly
            vec = domain.from_components(
                coeffs.coeffs.ravel()
            )  # or similar
            samples.append(vec)

print(f"Number of valid samples: {len(samples)}")

# 3. Create the measure
measure = GaussianMeasure.from_samples(domain, samples)

# %% Save
with open(f"odt{lmax}.pkl", "wb") as f:
    dill.dump(measure, f)

# %% Plotting
# The sample from the measure will be in the coefficient domain
sample_coeffs = measure.sample()

# Transform back from coefficients to spatial grid for plotting
sample_grid = sample_coeffs.expand()

plot(sample_grid * 1000)
