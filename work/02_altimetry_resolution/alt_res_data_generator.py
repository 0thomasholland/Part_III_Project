# %%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pygeoinf import (
    GaussianMeasure,
)
from pyslfp import (
    FingerPrint,
    IceModel,
)

from pygeoinf_extras.stats import expectation, standard_dev
from pyslfp_extras.altimetry import GridPoints
from pyslfp_extras.ice_thickness import (
    IceSheetChange,
)

# %%


fp = FingerPrint(lmax=256)
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)

fp_op = fp.as_sobolev_linear_operator(
    2, fp.mean_sea_floor_radius * 0.1
)

altimetry_spacing = np.array(
    [20, 15, 10, 5, 2.5, 1, 0.5, 0.25]
)

# %%

ice_change = IceSheetChange.global_ice(
    finger_print=fp,
    finger_print_operator=fp_op,
    length_scale=0.2 * fp.mean_sea_floor_radius,
    pattern=IceSheetChange.UniformPattern(),
    ice_gmsl_std=0.001,
    gmsl_target_mean=0.01,
)
ice_thickness_measure: GaussianMeasure = (
    ice_change.ice_thickness
)

# %%

ssh_ideal_estimate = ice_thickness_measure.affine_mapping(
    operator=ice_change.load_to_estimated_gmsl_operator
)

print(
    f"Ideal spacing, Expectation: {expectation(ssh_ideal_estimate)}, Std: {standard_dev(ssh_ideal_estimate)}"
)

true_gmsl = ice_thickness_measure.affine_mapping(
    operator=ice_change.ice_thickness_to_gmsl_operator
)

print(
    f"True GMSL, Expectation: {expectation(true_gmsl)}, Std: {standard_dev(true_gmsl)}"
)
# %%

# make a dictionary of expectations and std for each altimetry spacing

results: dict[str, tuple[float, float]] = {
    "true": (
        expectation(true_gmsl),
        standard_dev(true_gmsl),
    ),
    "ideal": (
        expectation(ssh_ideal_estimate),
        standard_dev(ssh_ideal_estimate),
    ),
}

# %%


for spacing in altimetry_spacing:
    print(f"Processing spacing: {spacing}")
    grid = GridPoints.ocean_altimetry(fp, spacing, 66.0)
    print(f"Grid points: {len(grid.coords)}")
    thickness_to_ssh_op = (
        ice_change.load_to_ssh_operator
        @ ice_change.ice_thickness_to_load_operator
    )
    gmsl_operator = grid.point_evaluation_operator(
        thickness_to_ssh_op.codomain
    )
    gmsl_estimate = ice_thickness_measure.affine_mapping(
        operator=gmsl_operator @ thickness_to_ssh_op
    )
    print("Done with estimate, calculating stats...")
    results[f"{spacing}"] = (
        expectation(gmsl_estimate),
        standard_dev(gmsl_estimate),
    )
    print(
        f"Spacing: {spacing}, Expectation: {results[f'{spacing}'][0]}, Std: {results[f'{spacing}'][1]}"
    )


# %%

# pandas save as csv

df = pd.DataFrame(
    {
        "spacing": list(results.keys()),
        "expectation": [v[0] for v in results.values()],
        "std": [v[1] for v in results.values()],
    }
)

df.to_csv("altimetry_resolution_results.csv", index=False)
