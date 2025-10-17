import numpy as np
import pandas as pd
import ternary
from matplotlib import pyplot as plt
from pyslfp import FingerPrint, IceModel

from Part_III_Project import sea_surface_height_change

fp = FingerPrint()
fp.set_state_from_ice_ng(version=IceModel.ICE7G, date=0.0)


sat_data_range = np.arange(1, 90, 5) # in degrees

error_output = pd.DataFrame()

for segment in sat_data_range:
    for west_contribution in np.arange(0,1,0.01):
        for east_contribution in np.arange(0,1-west_contribution,0.01):
            green_contribution = 1 - west_contribution - east_contribution
            print(f"Greenland: {green_contribution}, West Antarctica: {west_contribution}, East Antarctica: {east_contribution}")

            direct_load = (
                green_contribution * fp.greenland_load()
                + west_contribution * fp.west_antarctic_load()
                + east_contribution * fp.east_antarctic_load()
            )

            (
                sea_level_change,
                displacement,
                gravitational_potential_change,
                angular_velocity_change,
            ) = fp(direct_load=direct_load)

            sea_surface_height_change_result = sea_surface_height_change(
                fp, sea_level_change, displacement, angular_velocity_change
         )
            mean_sea_level_change = fp.mean_sea_level_change(direct_load)
            altimetry_projection = fp.altimetry_projection(
                latitude_min=-segment, latitude_max=segment, value=0
            )
            altimetry_projection_integral = fp.integrate(altimetry_projection)
            altimetry_weighting_function = altimetry_projection / altimetry_projection_integral

            mean_sea_level_change_estimate = fp.integrate(
                altimetry_weighting_function * sea_surface_height_change_result
            )

            error = 100 * np.abs(
                mean_sea_level_change_estimate - mean_sea_level_change
            ) / np.abs(mean_sea_level_change)

            error_output = pd.concat(
                [
                    error_output,
                    pd.DataFrame(
                        {
                            "segment": [segment],
                            "greenland_contribution": [green_contribution],
                            "west_antarctic_contribution": [west_contribution],
                            "east_antarctic_contribution": [east_contribution],
                            "error": [error],
                        }
                    ),
                ],
                ignore_index=True,
            )

error_output.to_csv("work/traditional_methods/ternary_sea_surface_height_error.csv", index=False)