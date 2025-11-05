import matplotlib.pyplot as plt
import numpy as np
import pygeoinf as inf
import pyslfp as sl
from scipy import stats

# --- Set up a fingerprint instance ---
fp = sl.FingerPrint(lmax=64)
fp.set_state_from_ice_ng()
