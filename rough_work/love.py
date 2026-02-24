# %%

import matplotlib.pyplot as plt
import numpy as np
import pyslfp as sl

lmax = 4096

fp = sl.FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng()


# Get the loading Love numbers
h = fp.h
k = fp.k
deg = np.arange(len(h))

# %%
# Make the plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(deg[2:], h[2:])
ax1.set_title("Displacement loading Love number (m^3/kg)")

ax2.plot(deg[2:], k[2:])
ax2.set_title(
    "Potential loading Love number (m^4/(kg s^2))"
)

plt.show()
# %%

# plot the sum up to the degreen on x axis

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(deg[2:], np.cumsum(h[2:]))
ax1.set_title(
    "Cumulative sum of displacement loading Love number (m^3/kg)"
)

ax2.plot(deg[2:], np.cumsum(k[2:]))
ax2.set_title(
    "Cumulative sum of potential loading Love number (m^4/(kg s^2))"
)

plt.show()
