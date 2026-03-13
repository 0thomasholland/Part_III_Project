import matplotlib.pyplot as plt
import seaborn as sns
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


def apply_style() -> None:
    """Apply the canonical project plot style.

    Sets seaborn theme/style and matplotlib rcParams for fonts.
    Called automatically when this module is imported.
    """
    sns.set_style("ticks")
    sns.set_context("paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["XCharter","Bitstream Charter", "Charter"],
            "text.usetex": False,
            "figure.figsize": (6, 4),
        }
    )


apply_style()
