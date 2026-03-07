# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from pyslfp import read_gloss_tide_gauge_data

lats, lons = read_gloss_tide_gauge_data()

print(len(lats), len(lons))
print(lats[:5], lons[:5])

# %%
# plot on map

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.scatter(
    lons,
    lats,
    color="red",
    marker="x",
    label="Tide Gauge Locations",
)
ax.set_title("Tide Gauge Locations from GLOSS Dataset")
ax.legend()
plt.show()

# %%

# filter so that there are only tide guages more than 5 degrees apart, maximising the number of tide gauges retained
filtered_lats = lats.copy()
filtered_lons = lons.copy()

for i in range(len(lats)):
    for j in range(i + 1, len(lats)):
        if (
            abs(lats[i] - lats[j]) < 8.0
            and abs(lons[i] - lons[j]) < 8.0
        ):
            # Remove the second point (j) if it's too close to the first point (i)
            filtered_lats[j] = None
            filtered_lons[j] = None

filtered_lats = [
    lat for lat in filtered_lats if lat is not None
]
filtered_lons = [
    lon for lon in filtered_lons if lon is not None
]

print(len(filtered_lats), len(filtered_lons))

# plot on map

plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.scatter(
    filtered_lons,
    filtered_lats,
    color="blue",
    marker="x",
    label="Filtered Tide Gauge Locations",
)
ax.scatter(
    lons,
    lats,
    color="blue",
    marker="x",
    label="Original Tide Gauge Locations",
    alpha=0.3,
)
ax.set_title(
    "Filtered Tide Gauge Locations from GLOSS Dataset"
)
ax.legend()
plt.show()
