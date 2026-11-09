import glob
import os
from pathlib import Path
# from datetime import datetime
# from datetime import timedelta
from matplotlib import pyplot
import matplotlib.pyplot as plt
# from matplotlib import colors
# matplotlib.use('Agg')
import time
import pandas as pd
import numpy as np
import h5py
import pyarrow.dataset as ds
# from pathlib import Path
import gc
import sys

from scintkit.data import identify_station
sys.path.append("/Users/jxg200016/Documents/scintpi/analysis")
import SP3
import scintpilib

def sp3_merge_lvl3(
    parquet_file,
    sp3_file,
    output_dir=None,
):
    
    # Read Level-3 parquet
    df_final = pd.read_parquet(parquet_file)

    # Read SP3 file
    file_path = Path(sp3_file)

    rows = []
    current_time = None
    # print('***** Opening SP3 ************')
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("*"):
                current_time = line.lstrip("*").strip()
            elif line.startswith("P"):
                parts = line.split()
                sensor = parts[0]
                try:
                    x, y, z = map(float, parts[1:4])
                    rows.append({
                        "timestampstr": current_time,
                        "satellite": sensor,
                        "X": x,
                        "Y": y,
                        "Z": z
                    })
                except ValueError:
                    print(f"Skipping invalid sensor line: {line}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    # Convert timestamp column
    df['timestamp'] = pd.to_datetime(df['timestampstr'],format="%Y %m %d %H %M %S.%f")
    df = df.drop(columns='timestampstr')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index(['timestamp'])
    df_resampled = df.groupby(['satellite']).apply(lambda group: group.resample('1min').interpolate(method='quadratic',axis=0),include_groups=False )
    # 4. Optional: remove multiindex if you prefer flat structure
    df_resampled = df_resampled.reset_index()
    #Convert to meters
    # Multiply 'price' and 'quantity' by 10
    cols_to_multiply = ['X', 'Y', 'Z']
    df_resampled[cols_to_multiply] = df_resampled[cols_to_multiply]*1000
        
    '''rx_lat = 32.99183368953561 #COORDINATES ARE IN THE MIDLE OF THE Receivers
    rx_long = -96.75730930926228
    rx_hei = 146.914 #mters'''

    '''rx_lat = -7.21245 #COORDINATES ARE IN THE MIDLE OF THE Receivers
    rx_long = -35.9066
    rx_hei = 552.50323 #mters'''

    #Change added by Priya

    station = identify_station(filename=parquet_file)

    if station is None:
        raise ValueError(f"Could not identify station for {parquet_file}")

    rx_lat = float(station["Latitude"])
    rx_long = float(station["Longitude"])

    height = station.get("Height")
    rx_hei = 146.914 if height is None or pd.isna(height) else float(height) #end
    
    XR,YR,ZR = SP3.wgs2xyz(rx_long,rx_lat,rx_hei)
    geoh = 350e3 #meters
    breaknumber = 0
    df_resampled = df_resampled.reset_index()
    for satellite, group in df_resampled.groupby('satellite'):
        # print(satellite)
        azim,elev,hlen,rang,NEU = SP3.azelle(group['X'], group['Y'], group['Z'],XR,YR,ZR)
        azim = np.where(azim<0,azim+2*np.pi,azim)
        zen_ang = np.ones(len(elev))*(np.pi/2.0) - elev; #THIS IS THE REAL ELEVATION, WE NEED TO CLEAN THE CODE AND RENAMEIT
        # Assign new values to the original dataframe
        ipplat_350,ipplong_350= scintpilib.get_IPP(rx_lat,rx_long,zen_ang,azim,geoh)
        # ipplat_110,ipplong_110= scintpilib.get_IPP(rx_lat,rx_long,zen_ang,azim,110e3)
        # SAFELY assign result back only to the rows in df_resampled for this satellite
        # print(group.index)
        df_resampled.loc[group.index, 'elev'] = zen_ang # make sure the index is not timestamp
        df_resampled.loc[group.index, 'azim'] = azim # elev and azim are in radians
        df_resampled.loc[group.index, 'ilat350'] = ipplat_350
        df_resampled.loc[group.index, 'ilon350'] = ipplong_350
    
    
    df_resampled['elev'] = df_resampled['elev']*180/np.pi
    df_resampled['azim'] = df_resampled['azim']*180/np.pi
    
    elevmin = 10
    elevmax = 90
    
    azimmin = 0
    azimmax = 360
    
    mask = ((df_resampled['elev'] >= elevmin) & (df_resampled['elev'] <= elevmax) & (df_resampled['azim'] >= azimmin) & (df_resampled['azim'] <= azimmax))
    df_resampled = df_resampled[mask]
    # Assuming your dataframe is called df and 'timestamp' is a datetime column:
    # 1. Ensure timestamp is in datetime format
    df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])
    
    # 2. Set timestamp as the index
    df_resampled = df_resampled.set_index('timestamp')
    df_resampled = df_resampled.reset_index()

# Ensure timestamps are datetime
    df_final["timestamp"] = pd.to_datetime(df_final["timestamp"])
    df_resampled["timestamp"] = pd.to_datetime(df_resampled["timestamp"])

# Sort (required for merge_asof)
    df_final = df_final.sort_values(["timestamp", "satellite"]).reset_index(drop=True)
    df_resampled = df_resampled.sort_values(["timestamp", "satellite"]).reset_index(drop=True)

    #Added by Priya

    df_final["minbin"] = df_final["timestamp"].dt.floor("1min")

    df_resampled["minbin"] = df_resampled["timestamp"].dt.floor("1min")

    print(
    df_resampled.groupby(["satellite", "minbin"]).size().max())
 
# --------------------------------------------------
# Merge using nearest timestamp for the same satellite
# --------------------------------------------------
    '''unified_df = pd.merge_asof(
      df_final,
      df_resampled,
      on="timestamp",
      by="satellite",
      direction="nearest",
      tolerance=pd.Timedelta("1min")   # or "1min" if you prefer
    )'''


    unified_df = df_final.merge(
    df_resampled[
        [
            "minbin",
            "satellite",
            "elev",
            "azim",
            "ilat350",
            "ilon350",
    
        ]
    ],
    on=["minbin", "satellite"],
    how="left",)




    # Convert to datetime (UTC recommended)
    unified_df["timestamp"] = pd.to_datetime(unified_df["timestamp"], utc=True)
    gps_epoch = pd.Timestamp("1980-01-06T00:00:00Z")
    
    # Time difference
    delta = unified_df["timestamp"] - gps_epoch

    
    # GPS week
    unified_df["gps_week"] = (delta.dt.days // 7).astype(int)
    unified_df["gps_sow"] = (
        delta.dt.total_seconds() % (7 * 24 * 3600)
    ).astype(int)
    
    parquet_file = Path(parquet_file)
    if output_dir is None:
        output_dir = parquet_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    outfile = output_dir / (
    parquet_file.stem.replace("_noelev", "_welev")
    + ".parquet")

    unified_df.to_parquet(outfile, index=True)

    print(f"Saved: {outfile}")

    return unified_df, outfile 