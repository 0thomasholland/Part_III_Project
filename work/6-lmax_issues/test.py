# %%
import pyslfp as sl

fp = sl.FingerPrint(
    lmax=128,
    earth_model_parameters=sl.EarthModelParameters.from_standard_non_dimensionalisation(),
)

print(fp.mean_sea_floor_radius)
print(fp.length_scale)

fp = sl.FingerPrint(
    lmax=128,
)

print(fp.mean_sea_floor_radius)
print(fp.length_scale)
# %%
