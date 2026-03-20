import cartopy.crs as ccrs

# Antarctica (EPSG:3031 equivalent)
PROJ_ANTARCTICA = ccrs.Stereographic(
    central_latitude=-90,
    true_scale_latitude=-71,
    central_longitude=0,
)

# [lon_min, lon_max, lat_min, lat_max] in PlateCarree
EXTENT_ANTARCTICA = [-180, 180, -90, -60]

# Greenland (EPSG:3413)
PROJ_GREENLAND = ccrs.Stereographic(
    central_latitude=90,
    true_scale_latitude=70,
    central_longitude=-45,
)

# [lon_min, lon_max, lat_min, lat_max] in PlateCarree
EXTENT_GREENLAND = [-65, -10, 58, 85]
