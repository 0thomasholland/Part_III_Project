import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from src.project.projections import PROJ_ANTARCTICA, PROJ_GREENLAND

fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1, projection=PROJ_ANTARCTICA)
ax1.set_extent([-180, 180, -90, -55], crs=ccrs.PlateCarree())
print("Antarctica limits:", ax1.get_xlim(), ax1.get_ylim())

ax2 = fig.add_subplot(1, 2, 2, projection=PROJ_GREENLAND)
ax2.set_extent([-65, -10, 58, 84], crs=ccrs.PlateCarree())
print("Greenland limits:", ax2.get_xlim(), ax2.get_ylim())
