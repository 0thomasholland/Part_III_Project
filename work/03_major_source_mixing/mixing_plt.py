# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
data = np.load("mixing_det_results.npz", allow_pickle=True)

results = data["results"]
gis = np.array([res["gis"] for res in results])
eais = np.array([res["eais"] for res in results])
wais = np.array([res["wais"] for res in results])
errors = np.array(
    [res["relative_error"] for res in results]
)
latitudes = np.array([res["latitude"] for res in results])
true_gmsl = np.array([res["true_gmsl"] for res in results])
estimated_gmsl = np.array(
    [res["estimated_gmsl"] for res in results]
)

# %%
# plot with interpolated pmesh

# set color range for max and min error across whole dataset

value = 100 * float(
    np.maximum(
        np.abs(np.min(errors)), np.abs(np.max(errors))
    )
)

for latitude in np.unique(latitudes):
    mask = latitudes == latitude
    fig = plt.figure(figsize=(7, 5))

    ax = fig.add_subplot(1, 1, 1, projection="ternary")
    cs = ax.tripcolor(
        gis[mask],
        eais[mask],
        wais[mask],
        errors[mask] * 100,
        vmin=-value,
        vmax=value,
        shading="gouraud",
        rasterized=True,
        # use symetric red blue color map at 0
        cmap="RdBu_r",
    )
    ax.set_title(
        f"Source Mixing - Relative Error at Altimetry Range Latitude {latitude:.0f}˚"
    )
    ax.set_tlabel("GIS")
    ax.set_llabel("EAIS")
    ax.set_rlabel("WAIS")
    # color bar using min and max values
    cbar = fig.colorbar(cs, ax=ax, orientation="vertical")
    cbar.set_label("Relative Error (%)")

    plt.tight_layout()
    plt.savefig(
        f"figures/mixing_det_altimetry_error_latitude_{latitude:.0f}.png",
        dpi=600,
    )
    plt.close()
