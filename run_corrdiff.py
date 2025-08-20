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
    # Fetch high-res select GEFS input data
    select_data = ds_gefs_select(time, lead_time, GEFS_SELECT_VARIABLES)
    lon = select_data.lon.values
    lat = select_data.lat.values
    # Convert bounding box to indices
    ilon_min = np.searchsorted(lon, -100.8 + 360)
    ilon_max = np.searchsorted(lon, -95.4 + 360)
    # Latitude is descending, so reverse for searchsorted
    ilat_max = np.searchsorted(lat, 28)
    ilat_min = np.searchsorted(lat, 33)
    # Indices for isel must be in ascending order
    ilat_start = min(ilat_min, ilat_max)
    ilat_end = max(ilat_min, ilat_max)
    select_data = select_data.isel(
        lon=slice(ilon_min, ilon_max),
        lat=slice(ilat_min, ilat_max)
    ).values
    print(select_data.shape)
    # Crop to bounding box Texas
    select_data= select_data[:,0,:,:,:].astype(dtype)
    # select_data = select_data[:, 0, :, 148:277, 900:1201].astype(dtype)
    # assert select_data.shape == (1, len(GEFS_SELECT_VARIABLES), 129, 301)

    # Fetch GEFS input data
    pressure_data = ds_gefs(time, lead_time, GEFS_VARIABLES)
    # Interpolate to 0.25 grid
    pressure_data = torch.nn.functional.interpolate(
        torch.Tensor(pressure_data.values),
        (len(GEFS_VARIABLES), 721, 1440),
        mode="nearest",
    )
    pressure_data = pressure_data.numpy()
    # Crop to bounding box [225, 21, 300, 53]
    pressure_data = pressure_data[:, 0, :, ilat_min:ilat_max, ilon_min:ilon_max].astype(dtype)
    


    # Create lead time field
    lead_hour = int(lead_time.total_seconds() // (3 * 60 * 60)) * np.ones(
        (1, 1, , 301)
    ).astype(dtype)

    input_data = np.concatenate([select_data, pressure_data, lead_hour], axis=1)[None]
    return input_data 


# input_array = fetch_input_gefs(datetime(2025, 7, 3, 18, 0, 0), timedelta(hours=18))
for lead_hour in range(0, 25, 6):
    input_array = fetch_input_gefs(datetime(2025, 7, 3, 18, 0, 0), timedelta(hours=lead_hour))
    np.save(f"corrdiff_inputs_f{lead_hour}.npy", input_array)

