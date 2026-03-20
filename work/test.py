# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Define the extent: [lon_min, lon_max, lat_min, lat_max]
EXTENT_GREENLAND = [-65, -10, 58, 85]

fig = plt.figure(figsize=(6.5, 5))

ax = plt.axes(projection=ccrs.PlateCarree())

# Set the map extent
ax.set_extent(EXTENT_GREENLAND, crs=ccrs.PlateCarree())

# Add map features
ax.coastlines()
plt.show()
