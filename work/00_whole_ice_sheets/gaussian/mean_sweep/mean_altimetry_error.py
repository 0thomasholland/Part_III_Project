from pyslfp.linear_operators import (
    FingerPrintOperator,
)
from pyslfp.state import EarthState
from time import perf_counter

from joblib import Parallel, delayed
from numpy import linspace
from pandas import DataFrame

from pygeoinf_extras import expectation, standard_dev
from pyslfp_extras.ice_thickness import IceSheetChange

t0 = perf_counter()
fp = EarthState.from_defaults(lmax=256)

fp_op = FingerPrintOperator(fp, load_parameters=(2, fp.model.parameters.mean_sea_floor_radius * 0.1
), response_parameters=(2 + 1, fp.model.parameters.mean_sea_floor_radius * 0.1
))

means = linspace(
    -0.1, 0.1, 20
)  # GMSL mean values from -0.01 to 0.01 m

stds = linspace(
    0.0001, 0.1, 20
)  # GMSL std values from 0.001 to 0.01 m

results = {}

t1 = perf_counter()

def compute_posterior(mean, std):
    """Modified to return data instead of writing to a global dict"""
    ice_change = IceSheetChange.global_ice(
        finger_print=fp,
        finger_print_operator=fp_op,
        length_scale=0.2 * fp.model.parameters.mean_sea_floor_radius,
        pattern=IceSheetChange.UniformPattern(),
        ice_gmsl_std=std,
        gmsl_target_mean=mean,
    )
    print(
        f"                                      Computing for mean={mean:.3f} mm, std={std:.3f} mm"
    )

    true_gmsl = ice_change.ice_thickness.affine_mapping(
        operator=ice_change.ice_thickness_to_gmsl_operator
    )
    estimated_gmsl = ice_change.ice_load.affine_mapping(
        operator=ice_change.load_to_estimated_gmsl_operator
    )
    error = true_gmsl - estimated_gmsl
    print(
        f"                                      Computed for mean={mean:.3f} mm, std={std:.3f} mm"
    )
    # Return a structured dictionary for this specific iteration
    return {
        "mean_in": float(mean),
        "std_in": float(std),
        "true_gmsl_exp": float(expectation(true_gmsl)),
        "true_gmsl_std": float(standard_dev(true_gmsl)),
        "est_gmsl_exp": float(expectation(estimated_gmsl)),
        "est_gmsl_std": float(standard_dev(estimated_gmsl)),
        "error_exp": float(expectation(error)),
        "error_std": float(standard_dev(error)),
    }

t2 = perf_counter()

# Run parallel jobs
list_of_dicts = Parallel(n_jobs=-1, verbose=10)(
    delayed(compute_posterior)(m, s)
    for m in means
    for s in stds
)

t3 = perf_counter()
print("Data saving........")

# Convert to DataFrame and save
df = DataFrame(list_of_dicts)
df.to_csv("ice_results.csv", index=False)

print("Data saved successfully.")
t4 = perf_counter()

print(f"Initialization time: {t1 - t0:.2f} seconds")
print(f"Setup time: {t2 - t1:.2f} seconds")
print(f"Computation time: {t3 - t2:.2f} seconds")
print(f"Data saving time: {t4 - t3:.2f} seconds")
