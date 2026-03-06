from matplotlib.colors import LinearSegmentedColormap

gis = "#5CB85C"
eais = "#437C90"
wais = "#F18F01"

primary_error = "#B71C1C"
secondary_error = "#EF5350"
error_cmap = LinearSegmentedColormap.from_list(
    "error_cmap", ["#FFFFFF", primary_error]
)
error_cmap_r = LinearSegmentedColormap.from_list(
    "error_cmap_r", ["#FFFFFF", primary_error]
)

old_method = "#232ED1"
new_method = "#5CA4A9"

true = "#545454"


model_params = "#9C27B0"
firn = "#FBC02D"
ice = "#2196F3"
ocean_dynamics = "#4CAF50"

data_params = "#FF5722"
ice_altimetry = "#3F51B5"
ocean_altimetry = "#FF9800"
