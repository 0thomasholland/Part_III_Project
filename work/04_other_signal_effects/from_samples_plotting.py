# %%
import dill
from pyslfp import FingerPrint, IceModel, plot

lmax = 128
fp = FingerPrint(lmax=lmax)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)
# %%

with open("odt.pkl", "rb") as f:
    measure = dill.load(f)
