from datetime import datetime, timedelta

import numpy as np
import torch
from earth2studio.data import GEFS_FX, GEFS_FX_721x1440

GEFS_SELECT_VARIABLES = ["u10m","v10m","t2m","r2m","sp","msl","tcwv"]
GEFS_VARIABLES = ["u1000","u925","u850","u700","u500","u250","v1000","v925","v850",\
    "v700","v500","v250","z1000","z925","z850","z700","z500","z200","t1000","t925",\
    "t850","t700","t500","t100","r1000","r925","r850","r700","r500","r100"]

ds_gefs = GEFS_FX(cache=True)
ds_gefs_select = GEFS_FX_721x1440(cache=True, member="gec00")

# print(ds_gefs_select)

def fetch_input_gefs(
    time: datetime, lead_time: timedelta, content_dtype: str = "float32"
):
    dtype = np.dtype(getattr(np, content_dtype))
    
    # Texas bounding box: lat [28, 33], lon [-100.8, -95.4]
    texas_lat_min, texas_lat_max = 28.0, 33.0
    texas_lon_min, texas_lon_max = -100.8, -95.4
    
    # Fetch high-res select GEFS input data
    select_data = ds_gefs_select(time, lead_time, GEFS_SELECT_VARIABLES)
    lon = select_data.lon.values
    lat = select_data.lat.values
    
    print(f"Original lat range: {lat.min():.2f} to {lat.max():.2f}")
    print(f"Original lon range: {lon.min():.2f} to {lon.max():.2f}")
    
    # Convert longitude to 0-360 range if needed
    if lon.max() > 180:
        # Already in 0-360 range
        texas_lon_min_360 = texas_lon_min + 360
        texas_lon_max_360 = texas_lon_max + 360
    else:
        # In -180 to 180 range
        texas_lon_min_360 = texas_lon_min
        texas_lon_max_360 = texas_lon_max
    
    # Find longitude indices
    ilon_min = np.searchsorted(lon, texas_lon_min_360)
    ilon_max = np.searchsorted(lon, texas_lon_max_360)
    
    # Find latitude indices - handle descending latitude
    if lat[0] > lat[-1]:  # Descending latitude (90 to -90)
        # For descending lat, find closest indices
        ilat_min = np.argmin(np.abs(lat - texas_lat_max))  # Higher lat value = lower index
        ilat_max = np.argmin(np.abs(lat - texas_lat_min))  # Lower lat value = higher index
        # Ensure we have a range, not just a point
        if ilat_min == ilat_max:
            ilat_max = ilat_min + 1
        # Ensure proper order for slicing (min < max)
        if ilat_min > ilat_max:
            ilat_min, ilat_max = ilat_max, ilat_min
    else:  # Ascending latitude
        ilat_min = np.searchsorted(lat, texas_lat_min)
        ilat_max = np.searchsorted(lat, texas_lat_max)
    
    print(f"Latitude indices: {ilat_min} to {ilat_max} (size: {ilat_max - ilat_min})")
    print(f"Longitude indices: {ilon_min} to {ilon_max} (size: {ilon_max - ilon_min})")
    print(f"Actual lat range: {lat[ilat_min]:.2f} to {lat[ilat_max-1]:.2f}")
    print(f"Actual lon range: {lon[ilon_min]:.2f} to {lon[ilon_max-1]:.2f}")
    
    # Crop select_data to Texas region
    select_data_cropped = select_data.isel(
        lon=slice(ilon_min, ilon_max),
        lat=slice(ilat_min, ilat_max)
    )
    print(f"Select data shape after cropping: {select_data_cropped.shape}")
    
    # Convert to numpy and remove ensemble dimension
    select_data = select_data_cropped.values[:, 0, :, :, :].astype(dtype)
    print(f"Select data final shape: {select_data.shape}")

    # Fetch GEFS pressure level data
    pressure_data = ds_gefs(time, lead_time, GEFS_VARIABLES)
    # Interpolate to 0.25 degree grid (721x1440)
    pressure_data = torch.nn.functional.interpolate(
        torch.Tensor(pressure_data.values),
        (len(GEFS_VARIABLES), 721, 1440),
        mode="nearest",
    )
    pressure_data = pressure_data.numpy()
    
    # Calculate indices for the interpolated 721x1440 grid
    # Standard 0.25 degree grid: lat from 90 to -90, lon from 0 to 359.75
    pressure_lat = np.linspace(90, -90, 721)  # 721 points, 0.25 degree spacing
    pressure_lon = np.linspace(0, 359.75, 1440)  # 1440 points, 0.25 degree spacing
    
    print(f"Pressure grid lat range: {pressure_lat.min():.2f} to {pressure_lat.max():.2f}")
    print(f"Pressure grid lon range: {pressure_lon.min():.2f} to {pressure_lon.max():.2f}")
    
    # Find indices for Texas region in the interpolated grid
    # Convert longitude to 0-360 range
    texas_lon_min_360 = texas_lon_min + 360
    texas_lon_max_360 = texas_lon_max + 360
    
    # Find longitude indices for pressure grid
    pressure_ilon_min = np.searchsorted(pressure_lon, texas_lon_min_360)
    pressure_ilon_max = np.searchsorted(pressure_lon, texas_lon_max_360)
    
    # Find latitude indices for pressure grid (descending from 90 to -90)
    pressure_ilat_min = np.argmin(np.abs(pressure_lat - texas_lat_max))  # Higher lat value = lower index
    pressure_ilat_max = np.argmin(np.abs(pressure_lat - texas_lat_min))  # Lower lat value = higher index
    # Ensure we have a range, not just a point
    if pressure_ilat_min == pressure_ilat_max:
        pressure_ilat_max = pressure_ilat_min + 1
    # Ensure proper order for slicing (min < max)
    if pressure_ilat_min > pressure_ilat_max:
        pressure_ilat_min, pressure_ilat_max = pressure_ilat_max, pressure_ilat_min
    
    print(f"Pressure lat indices: {pressure_ilat_min} to {pressure_ilat_max} (size: {pressure_ilat_max - pressure_ilat_min})")
    print(f"Pressure lon indices: {pressure_ilon_min} to {pressure_ilon_max} (size: {pressure_ilon_max - pressure_ilon_min})")
    print(f"Pressure actual lat range: {pressure_lat[pressure_ilat_min]:.2f} to {pressure_lat[pressure_ilat_max-1]:.2f}")
    print(f"Pressure actual lon range: {pressure_lon[pressure_ilon_min]:.2f} to {pressure_lon[pressure_ilon_max-1]:.2f}")
    
    # Crop pressure_data to the Texas region using correct indices
    pressure_data = pressure_data[:, 0, :, pressure_ilat_min:pressure_ilat_max, pressure_ilon_min:pressure_ilon_max].astype(dtype)
    print(f"Pressure data shape after cropping: {pressure_data.shape}")

    # Get the actual dimensions for lead time field from pressure data
    # (since we need all arrays to have consistent spatial dimensions)
    pressure_nlat = pressure_ilat_max - pressure_ilat_min
    pressure_nlon = pressure_ilon_max - pressure_ilon_min
    
    # Resize select_data to match pressure_data dimensions if needed
    if select_data.shape[-2:] != (pressure_nlat, pressure_nlon):
        print(f"Resizing select_data from {select_data.shape[-2:]} to {(pressure_nlat, pressure_nlon)}")
        select_data_tensor = torch.tensor(select_data)
        select_data_resized = torch.nn.functional.interpolate(
            select_data_tensor,
            size=(pressure_nlat, pressure_nlon),
            mode="bilinear",
            align_corners=True
        )
        select_data = select_data_resized.numpy().astype(dtype)
        print(f"Select data shape after resizing: {select_data.shape}")
    
    # Create lead time field with pressure data dimensions
    lead_hour_value = int(lead_time.total_seconds() // (3 * 60 * 60))
    lead_hour = lead_hour_value * np.ones((1, 1, pressure_nlat, pressure_nlon)).astype(dtype)
    print(f"Lead hour field shape: {lead_hour.shape}")

    # Concatenate all data
    input_data = np.concatenate([select_data, pressure_data, lead_hour], axis=1)[None]
    print(f"Final input data shape: {input_data.shape}")
    
    return input_data 


# input_array = fetch_input_gefs(datetime(2025, 7, 3, 18, 0, 0), timedelta(hours=18))
for lead_hour in range(0, 25, 6):
    input_array = fetch_input_gefs(datetime(2025, 7, 3, 18, 0, 0), timedelta(hours=lead_hour))
    np.save(f"corrdiff_inputs_f{lead_hour}.npy", input_array)
