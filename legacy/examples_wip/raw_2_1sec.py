#%%


import sys

from pathlib import Path

import numpy as np
import pandas as pd

import scintkit
import glob

data_dir = Path("data")
raw_files=glob.glob("/Users/isaac/Documents/m_day/scintpi3_*lvl0.pq")
#These can also be .bin or .bin.zip files

lvl3_files = scintkit.pipelines.auto.process(raw_files, verbose=True)

#%%
import pandas as pd
import glob as glob


flist=glob.glob("/Users/isaac/Documents/m_day/scintpi3_*lvl3.pq")

total=[]
for f in flist:
    print(f)
    df=pd.read_parquet(f)
    total.append(df)

total=pd.concat(total)


#%%
import matplotlib.pyplot as plt


#%%
files=glob.glob('/Users/isaac/Documents/MothersDay_SigPhi/Data/*.pq')

datalist=[]
#%%
import matplotlib.patheffects as pe
import matplotlib.dates as mdates
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import pyIGRF
# Dictionary with coordinates and corresponding state names
sc3_dict = {
    (32.99, -96.757): 'Texas',
    (41.406, -75.658): 'Pennsylvania',
  #  (18.467, -66.914): 'Puerto Rico',
  #  (18.33, -65.307): 'Puerto Rico',
   # (18.347, -66.754): 'Puerto Rico',
    #(40.115, -88.228): 'Illinois',
   # (41.742, -111.81): 'Utah',
    (41.934, -111.421): 'Utah',

    (43.27, -120.36): 'Oregon',
    #(9.855, -83.91): 'Costa Rica',
    (34.35, -106.88): 'New Mexico',

    (38.381, -103.156): 'Colorado',
  #  (14.094, -87.16): 'Honduras',
    #(63.77,-20.54 ): 'Iceland',
    (43.70,-72.29 ): 'New Hampshire',
 #   (42.81,-73.91 ): 'New York',
    (38.9188,-92.127 ): 'Missouri',

   
}

abbr_dict = {
    'UTD': 'Texas',
    'UOS': 'Pennsylvania',
  #  (18.467, -66.914): 'Puerto Rico',
    'PRA': 'Puerto Rico',   
   # (18.347, -66.754): 'Puerto Rico',
    'IUC': 'Illinois',
   # (41.742, -111.81): 'Utah',
    'UTB': 'Utah',

    'ORE': 'Oregon',
    'CRT': 'Costa Rica',
    'NMX': 'New Mexico',

    'COL': 'Colorado',
    'HND': 'Honduras',
    'ICE': 'Iceland',
    'NHA': 'New Hampshire',
    #'NYO': 'New York',
    'MOS': 'Missouri'
}


abbr_dict =dict(map(reversed, abbr_dict.items()))

def calculate_dip_latitude(lat, lon, year=2024):

    _, dip_latitude, _, _, _, _, _ = pyIGRF.igrf_value(lat, lon, 0, year=year)  # 0 km altitude
    dip_lat = np.degrees(np.arctan(0.5 * np.tan(np.radians(dip_latitude))))
    return dip_lat



def IPP(lat_0,lon_0,elv,azim,height=350):
    r=6371#km radius
    h=height#km height of irregularities
    
    e=np.radians(elv)
    a=np.radians(azim)
    lat_0=np.radians(lat_0)
    lon_0=np.radians(lon_0)
    
    psi=np.pi/2-e-np.arcsin(r/(r+h)*np.cos(e))
    
    lat=np.arcsin( np.sin(lat_0) * np.cos(psi) + np.cos(lat_0)*np.sin(psi)*np.cos(a))
    lon=lon_0+np.arcsin(np.sin(psi)*np.sin(a)/np.cos(lat))
    return np.degrees(lat),np.degrees(lon)


def circle(lon,lat):
    elmask=30
    azs= np.linspace(0,360,360)
    
    elvs=np.repeat(elmask,len(azs))
    
    lats_circ,lons_circl=IPP(lat,lon,elvs,azs)
    return lats_circ,lons_circl
#%%


import glob, pandas as pd, numpy as np, time, zipfile, os, re, struct
from pathlib import Path
import matplotlib.pyplot as plt
import math


files=glob.glob('/Users/isaac/Documents/MothersDay_SigPhi/Data/*.pq')






sc3_dict = {
    (32.99, -96.757): 'Texas',
    (41.406, -75.658): 'Pennsylvania',
  #  (18.467, -66.914): 'Puerto Rico',
  #  (18.33, -65.307): 'Puerto Rico',
   # (18.347, -66.754): 'Puerto Rico',
    #(40.115, -88.228): 'Illinois',
   # (41.742, -111.81): 'Utah',
    (41.934, -111.421): 'Utah',

    (43.27, -120.36): 'Oregon',
 
    #(9.855, -83.91): 'Costa Rica',
    (34.35, -106.88): 'New Mexico',

    (38.381, -103.156): 'Colorado',
   # (14.094, -87.16): 'Honduras',
  #  (63.77,-20.54 ): 'Iceland',
    (43.70,-72.29 ): 'New Hampshire',
    (42.81,-73.91 ): 'New York',
    (38.9188,-92.127 ): 'Missouri',
}

dip_dict = {
    'UTD': 42.35,
    'PEN': 49.24,
    'PRA': 23.68,
    'UTB': 49.24,
    'ORE': 48.32,
    'CRT': 19.87,
    'NMX': 42.30,
    'COL': 47.19,
    'HND': 23.86,
    'ICE': 62.07,
    'NHA': 50.73,
   # 'NYO': 50.25,
    'MOS': 48.85
}

abbr_dict = {
    'UTD': 'Texas',
    'PEN': 'Pennsylvania',
  #  (18.467, -66.914): 'Puerto Rico',
   # 'PRA': 'Puerto Rico',   
   # (18.347, -66.754): 'Puerto Rico',
 #   'IUC': 'Illinois',
   # (41.742, -111.81): 'Utah',
    'UTB': 'Utah',

    'ORE': 'Oregon',
  #  'CRT': 'Costa Rica',
    'NMX': 'New Mexico',

    #'COL': 'Colorado',
    'HND': 'Honduras',
  #  'ICE': 'Iceland',
    'NHA': 'New Hampshire',
    #'NYO': 'New York',
    'MOS': 'Missouri'
}


def fname2latlon(fname: str):
    # parse
    m = re.search(r'_([0-9]+(?:\.[0-9]+)?)([EW])_([0-9]+(?:\.[0-9]+)?)([NS])', fname)
    if not m:
        raise ValueError("no lat/lon in filename")

    lon = float(m.group(1))
    lat = float(m.group(3))
    lon_h = m.group(2)
    lat_h = m.group(4)

    # scale if needed
    if lon > 180.0:
        lon /= 1e4
    if lat > 90.0:
        lat /= 1e4

    # apply sign
    if lon_h == 'W':
        lon = -lon
    if lat_h == 'S':
        lat = -lat

    return lat, lon


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

# region->code map
def _invert_abbr(abbr_dict):
    return {region: code for code, region in abbr_dict.items()}

def latlon2station(lat,lon, tol_km=1):
    # get coords
    # find nearest
    best = None
    best_d = float("inf")
    for (rlat, rlon), region in sc3_dict.items():
        d = _haversine(lat, lon, rlat, rlon)
        if d < best_d:
            best_d = d
            best = region
    # resolve code
    region2code = _invert_abbr(abbr_dict)
    code = region2code.get(best)
    if code is None or best_d > tol_km:
        return None
    return code
for file in files:
    
    lat,lon=fname2latlon(file)
    code=latlon2station(lat, lon)
    if code:
        dip=dip_dict[code]
        df=pd.read_parquet(file)
        df['code']=code
        df['lat0']=lat
        df['lon0']=lon
        df['dip0']=dip

        datalist.append(df)
 
total_df=pd.concat(datalist)
total_df.dropna(inplace=True)
# %%
pendf=total_df[total_df['code']=='PEN']

# %%
plt.figure(figsize=(12,6))


num='1'

sat='R1'

#r04 ro3 r02 all noiser
mask=(
    (total[f'quality_{num}']==0)
&(total[f'n_{num}']>1190 ) &(total['elev']>20)
&(total[f'cph{num}']!=0)
&(total['prn'].str.contains(sat))
)
t=total[mask]
plt.ylim(0,1)

mask=(
    pendf.PRN.str.contains(sat)  
       &(pendf['count']>1190) 
      &(pendf['elev']>20))
pen_masked=pendf[mask].reset_index()

plt.xlim(t.minbin.min(), t.minbin.max())

plt.plot(pen_masked['Datetime'],pen_masked[f'Sig_Phi_L1'],'x',label='old code')
plt.plot(t['minbin'],t[f'sigma_phi_{num}'],'x',label='new code')

plt.xlim(total['minbin'].min(), total['minbin'].max())
plt.legend()

#%%

from scintkit.services.compute import add_products


file='/Users/isaac/Documents/m_day/scintpi3_20240511_0800_756582.5000W_414055.0312N_v325_lvl0.pq'

df_raw=pd.read_parquet(file)

df=add_products(df_raw, verbose=True)
#%%
low=pd.Timestamp("2024-05-11 10:01:00")
period=pd.Timedelta(seconds=10)

df_sub=df[(df['datetime']>=low) & (df['datetime']<low+period)
          &(df['elev']>0)
          &(df['cph2']!=0)
          &df['prn'].str.contains('R')]


plt.figure(figsize=(12,6))

df_sub['clock2'] = (
    df_sub.groupby('datetime')['detrended_cph1']
          .transform('median') / 1610
)

clock_f2=df_sub['clock_term']*1227
clock_f1=df_sub['clock_term']*1574

plt.plot(df_sub.datetime, clock_f1, label='clock term', color='black', zorder=3)




df_sub['detrended_cph1'].median()
for prn, group in df_sub.groupby('prn'):


    color = 'red' if 'R' in str(prn) else 'black'
    zorder=1 if 'R' in str(prn) else 2
   
    plt.plot(group.datetime, group['detrended_cph1'], label=prn, color=None, zorder=zorder)
    #plt.plot(group.datetime, group['detrended_cph2'], label=prn, color=None, zorder=zorder)
   


    sigma_phi_1 = np.std(group['detrended_cph1']-clock_f1)

    sigma_phi_2 = np.std(group['detrended_cph2']-clock_f2)
    freq1=group['freq_1'].median()
    freq2=group['freq_2'].median()


    if 'R' in str(prn):
        print(f"{prn} sigma_phi_1: {sigma_phi_1:.3f} rad")
        print(f"{prn} sigma_phi_2: {sigma_phi_2:.3f} rad")
        print(f"{prn} freq1: {freq1:.2f} MHz")
        print(f"{prn} freq2: {freq2:.2f} MHz")
plt.legend()
plt.show()


print('reference signal at low time:')
print(pen_masked[pen_masked.Datetime==low].Sig_Phi_L1)



#%%


plt.scatter()
# %%
