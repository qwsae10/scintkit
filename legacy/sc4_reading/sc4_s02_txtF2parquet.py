import glob
import numpy as np
import time
from datetime import datetime
import os
import sys
#import scipy.interpolate
import pandas as pd
from pathlib import Path

print("checking libs")

def reading__measurements_file(files,redo=False): #converts text to data frame
    gps_epoch = pd.to_datetime('1980-01-06') #gps start time
    s4=pd.DataFrame()
    if type(files)==str:
        files=[files]

    txt_file = Path(files[0])

    name = txt_file.name
    name = name.replace("__measurements.txt", "__mearem.parquet")

    df_output = txt_file.with_name(name)

    if os.path.exists(df_output) and not(redo):
        print(df_output+' already exists, using it')
        return pd.read_parquet(df_output), df_output
    print(f"{df_output} doesn't exist, creating new file")
   # cols=[1,2,3,4,7,8,9] #columns
    cols=[0,1,2,3,5,6,8] #columns \nt
    #293400.000,2306,G10,GPS_L2C,Main,20928167.858000,85697380.178385,-1114.756836,45.437500,60
    names=['TOW','WN','SVID','SIG','PR','Phase','SNR']#dataframe namers
    rows=2 #in case of header skip some vals
    chunksize = 10 ** 7
    df=pd.DataFrame()
    buffer=pd.DataFrame()
    for file in files:
        with pd.read_csv(file,header=None,skiprows=rows, usecols=cols, names=names, chunksize=chunksize) as reader:
            for chunk in reader:
                chunk['datetime'] = gps_epoch + pd.to_timedelta(chunk['WN'] * 7, unit='D') + pd.to_timedelta(chunk['TOW'], unit='s')
                chunk.set_index('datetime', inplace=True)
                chunk.sort_values(by='datetime')

                df = pd.concat([df, chunk])

    df.reset_index(inplace=True)
    df.set_index('datetime',inplace=True)
    df.to_parquet(df_output)
    return df, df_output

for daystring in np.arange(53,55+1):
    for letter in range(ord('a'), ord('z') + 1):
        lvl0_files = []
        datafolder = "/home/jxg200016/scratch/measurements"
        #daystring = 52
#cr01052f30.26__measurements.txt
        lvl0_files = glob.glob("%s/*%03d%s*.txt"%(datafolder,daystring,chr(letter)))
        print(letter,lvl0_files)
#UTD01 = reading__measurements_file(lvl0_files)
        try:
            UTD01 = pd.concat(map(reading__measurements_file, lvl0_files))
        except:
            continue

