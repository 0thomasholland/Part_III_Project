import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pyslfp import FingerPrint, IceModel

from Part_III_Project import sea_surface_height_change

fp = FingerPrint()
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


latitude_max = np.arange(1, 90, 0.1)
latitude_min = -latitude_max

green_error = np.zeros_like(latitude_min)
west_error = np.zeros_like(latitude_min)
east_error = np.zeros_like(latitude_min)

error_output = pd.DataFrame(
    {
        "latitude_min": latitude_min,
        "latitude_max": latitude_max,
        "greenland_error": green_error,
        "west_antarctic_error": west_error,
        "east_antarctic_error": east_error,
    }
)

for ice_sheet in ["greenland", "west_antarctic", "east_antarctic"]:
    if ice_sheet == "greenland":
        direct_load = fp.greenland_load()
    elif ice_sheet == "west_antarctic":
        direct_load = fp.west_antarctic_load()
    else:
        direct_load = fp.east_antarctic_load()

    (
        sea_level_change,
        displacement,
        gravitational_potential_change,
        angular_velocity_change,
    ) = fp(direct_load=direct_load)

    sea_surface_height_change_result = sea_surface_height_change(
        fp, sea_level_change, displacement, angular_velocity_change
    )

    for i in range(len(latitude_min)):
        mean_sea_level_change = fp.mean_sea_level_change(direct_load)

        altimetry_projection = fp.altimetry_projection(
            latitude_min=latitude_min[i], latitude_max=latitude_max[i], value=0
        )

        altimetry_projection_integral = fp.integrate(altimetry_projection)

        altimetry_weighting_function = altimetry_projection / altimetry_projection_integral

        mean_sea_level_change_estimate = fp.integrate(
            altimetry_weighting_function * sea_surface_height_change_result
        )

        # print(f"True mean sea level change = {mean_sea_level_change}m")
        # print(f"Estimated mean sea level change = {mean_sea_level_change_estimate}m")
        # print(
        #     f"Relative error in estimate {100 * np.abs(mean_sea_level_change_estimate - mean_sea_level_change) / np.abs(mean_sea_level_change)}%"
        # )
        error = 100 * np.abs(
            mean_sea_level_change_estimate - mean_sea_level_change
        ) / np.abs(mean_sea_level_change)
        if ice_sheet == "greenland":
            error_output.loc[i, "greenland_error"] = error
        elif ice_sheet == "west_antarctic":
            error_output.loc[i, "west_antarctic_error"] = error
        else:
            error_output.loc[i, "east_antarctic_error"] = error
        
error_output.to_csv("./work/traditional_method_errors.csv", index=False)

plt.figure()
plt.plot(
    error_output["latitude_max"],  # or use np.abs(error_output["latitude_min"])
    error_output["greenland_error"],
    label="Greenland",
)
plt.plot(
    error_output["latitude_max"],
    error_output["west_antarctic_error"],
    label="West Antarctic",
)
plt.plot(
    error_output["latitude_max"],
    error_output["east_antarctic_error"],
    label="East Antarctic",
)
plt.xlabel("Maximum Absolute Latitude of Band (degrees)")
plt.ylabel("Relative Error (%)")
plt.title("Relative Error vs Latitude Band Coverage")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()