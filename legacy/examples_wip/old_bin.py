#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 09:47:13 2024

@author: josemaria
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 20:50:56 2024

@author: josemaria
"""
#%%

import matplotlib.dates as mdates
import zipfile,re 
import subprocess,struct
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import pandas as pd
import glob
import os









def PRN(SAT):
    if SAT>0 and SAT<=37:
        svid = str(SAT).zfill(2)
        return 'G'+svid
    elif SAT>=38 and SAT<=61:
        svid = str(SAT-37).zfill(2)
        return 'R'+svid
    elif SAT>=63 and SAT<=68:
        svid = str(SAT-38).zfill(2)
        return 'R'+svid
    elif SAT>=71 and SAT<=106:
        svid = str(SAT-70).zfill(2)
        return 'E'+svid
    elif SAT>=120 and SAT<=140:
        SAT=SAT-100
        svid = str(SAT).zfill(2)
        return 'S'+svid
    elif SAT>=141 and SAT<=180:
        svid = str(SAT-140).zfill(2)
        return 'C'+svid
    elif SAT>=181 and SAT<=187:
        svid = str(SAT-180).zfill(2)
        return 'J'+svid
    elif SAT>=198 and SAT<=215:
        svid = str(SAT-157).zfill(2)
        return 'J'+svid
    elif SAT>=223 and SAT<=245:
        svid = str(SAT-180).zfill(2)
        return 'J'+svid
    else:
        return '99',999   
    


from scipy import signal
outdir='/Users/isaac/Documents/m_day/'

def ismr_reader(files):
    if isinstance(files, str):
        files = [files]
    tot = pd.DataFrame([])

    for f in files:
        # Use a similar name but with a .parquet extension
        if f.lower().endswith('.ismr'):
            parquet_file = f[:-4] + '.parquet'
        else:
            parquet_file = f + '.parquet'
        
        # If the parquet exists, load it; otherwise process the CSV and save it.
        if (os.path.exists(parquet_file))and f!=files[0]:
            df = pd.read_parquet(parquet_file)
        else:
            df = pd.read_csv(f, names=[
                'WN', 'TOW', 'SVID', 'RxState', 'Azimuth', 'Elevation', 'AvgCNo1', 'TotalS4_1',
                'CorrectionS4_1', 'Phi01_1', 'Phi03_1', 'Phi10_1', 'Phi30_1', 'Phi60_1',
                'AvgCCD_1', 'SigmaCCD_1', 'TEC_TOW_45s', 'dTEC_60s_45s', 'TEC_TOW_30s',
                'dTEC_45s_30s', 'TEC_TOW_15s', 'dTEC_30s_15s', 'TEC_TOW', 'dTEC_15s_TOW',
                'LockTime_Sig1', 'sbf2ismrVersion', 'LockTime_Sig2_TEC', 'AvgCNo2_TEC',
                'SI_Index_1', 'SI_Index_Numerator_1', 'p_1', 'AvgCNo2', 'TotalS4_2',
                'CorrectionS4_2', 'Phi01_2', 'Phi03_2', 'Phi10_2', 'Phi30_2', 'Phi60_2',
                'AvgCCD_2', 'SigmaCCD_2', 'LockTime_Sig2', 'SI_Index_2', 'SI_Index_Numerator_2',
                'p_2', 'AvgCNo3', 'TotalS4_3', 'CorrectionS4_3', 'Phi01_3', 'Phi03_3',
                'Phi10_3', 'Phi30_3', 'Phi60_3', 'AvgCCD_3', 'SigmaCCD_3', 'LockTime_Sig3',
                'SI_Index_3', 'SI_Index_Numerator_3', 'p_3', 'T_Sig1', 'T_Sig2', 'T_Sig3'
            ])
            
            # Create datetime index from GPS time
            gps_epoch = pd.to_datetime('1980-01-06')
            df['Datetime'] = gps_epoch + pd.to_timedelta(df['WN'] * 7, unit='D') \
                             + pd.to_timedelta(df['TOW'], unit='s') - pd.to_timedelta(60, unit='s')
            df.set_index('Datetime', inplace=True)
            df.sort_index(inplace=True)  # sort by the datetime index
            
            # Process data (assumes PRN is defined elsewhere)
            df['PRN'] = df['SVID'].apply(PRN)
            
            # Compute S4 values
            df['S4L1'] = np.sqrt(np.abs(df['TotalS4_1'].astype(float)**2 - df['CorrectionS4_1'].astype(float)**2))
            df['S4L2'] = np.sqrt(np.abs(df['TotalS4_2'].astype(float)**2 - df['CorrectionS4_2'].astype(float)**2))
            df['S4L3'] = np.sqrt(np.abs(df['TotalS4_3'].astype(float)**2 - df['CorrectionS4_3'].astype(float)**2))
            
            df.loc[(df['TotalS4_1'].astype(float)**2 - df['CorrectionS4_1'].astype(float)**2) < 0, 'S4L1'] = 0
            df.loc[(df['TotalS4_2'].astype(float)**2 - df['CorrectionS4_2'].astype(float)**2) < 0, 'S4L2'] = 0
            # Assuming the intended column is S4L3 for the third set:
            df.loc[(df['TotalS4_3'].astype(float)**2 - df['CorrectionS4_3'].astype(float)**2) < 0, 'S4L3'] = 0
            
            # Create a subset of columns
            df = df[['PRN', 'Elevation', 'Azimuth','Phi60_1', 'Phi60_2', 'S4L1','TotalS4_1']]
            
            # Save processed DataFrame as parquet for faster future reads
            df.to_parquet(parquet_file)
        
        tot = pd.concat([tot, df])

    return tot


def getphase(files,redo=False): #converts text to data frame
    gps_epoch = pd.to_datetime('1980-01-06') #gps start time

    s4=pd.DataFrame()
    
    if type(files)==str:
        files=[files]
        
    
    letters=[ f[-13:-10] for f in files]
    
    namestr=''.join(letters)
    df_output=files[0][:-15]+namestr+'.parquet'
    

    
    if os.path.exists(df_output) and not(redo):
        print(df_output+' already exists, using it')

        return pd.read_parquet(df_output)
       
    print(df_output+' doesnt exist, creating new file')

   
   # cols=[1,2,3,4,7,8,9] #columns of file we want
    cols=[0,1,2,3,5,6,8] #columns of file we want

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
    return df




def repair_discontinuities(vec,threshold=10):
    
    

    
    first_order=pd.DataFrame({'delt':vec.diff()}) #seconds should be changed to 0.05s because of change in frequency [DONE]
    
    
    #first_order_full = first_order.reindex(full_index)['delt']
    first_order_full = first_order['delt']

    points=first_order.resample('10s').median().shift(freq='5s')
   
    if len(points)==1:
        print('too small!')
        defa=(vec)
        return defa,~(defa==defa)
    
    final_t=points.index[-1]+(points.index[-1]-points.index[-2])
    final_val=points.iloc[-1]+(points.iloc[-1]-points.iloc[-2])
    
    first_t=points.index[0]-(points.index[1]-points.index[0])
    first_val=points.iloc[0]-(points.iloc[1]-points.iloc[0])
    
    
    points.loc[final_t]=final_val
    points.loc[first_t]=first_val
    points.sort_index(inplace=True)
    #huber loss function into 3rd or 4th order polynomial??
    
    
    doppler = points.delt.reindex_like(first_order_full).interpolate('time')

    
    
    cycleslips=np.abs(doppler - first_order_full) <= threshold
    
    first_order_full=first_order_full.where(cycleslips,np.nan)
    first_order_full.iloc[0]=first_order_full.iloc[1]
    first_order_full=first_order_full.bfill()
    
    result=np.cumsum(first_order_full)
    
    return result,~(cycleslips)



def interp(series, freq='50ms'): #frequency should be changed [DONE]
 
    # Ensure the index is a DatetimeIndex
    series.index = pd.to_datetime(series.index)
    
    # Create a complete datetime range from the min to max timestamp
    full_index = pd.date_range(start=series.index.min(), end=series.index.max(), freq=freq)
    
    # Reindex the series to include all timestamps; missing ones become NaN.
    series_full = series.reindex(full_index)
    
    
    # Interpolate missing values using time-based linear interpolation.
    series_interp = series_full.interpolate(method='time')
    
    
    
    
    
    return series_interp




import time

def make_prn(df):
    constellation_map = {
        'GPS': 'G',
        'BDS': 'C',
        'GAL': 'E',
        'GLO': 'R',
        'QZSS': 'J',
        'IRNSS': 'I',
        'SBAS': 'S',
    }
    return df['SIG'].map(constellation_map) + df['SVID'].astype(str).str.zfill(2)





def filter_signal_cascaded(x, f_N=0.1, fs=10): #fs should be 20 (Hz) [DONE]

    # Calculate the angular cutoff frequency.
    omega_N = 2 * np.pi * f_N
    l=3
    # Define the coefficients for the three second-order sections.
    a1 = np.sqrt(2 + np.sqrt(l))
    a2 = np.sqrt(2)
    a3 = np.sqrt(2 - np.sqrt(l))

    # The numerator for each stage is s^2, represented as [1, 0, 0].
    num = [1, 0, 0]
    
    # Define the denominators for each stage.
    den1 = [1, a1 * omega_N, omega_N**2]
    den2 = [1, a2 * omega_N, omega_N**2]
    den3 = [1, a3 * omega_N, omega_N**2]
    
    # Convert each analog stage to digital using the bilinear transform.
    bz1, az1 = signal.bilinear(num, den1, fs)
    bz2, az2 = signal.bilinear(num, den2, fs)
    bz3, az3 = signal.bilinear(num, den3, fs)
    
    # Cascade the filtering stages:filtfilt
    # Stage 1: Filter the input signal.
    y_stage1 = signal.lfilter(bz1, az1, x)
    
    
    #y_stage1=remove_cycleslips(y_stage1,thres=0.3)

    # Stage 2: Filter the output of stage 1.
    y_stage2 = signal.lfilter(bz2, az2, y_stage1)
    
   

    # Stage 3: Filter the output of stage 2.
    y_stage3 = signal.lfilter(bz3, az3, y_stage2)
    
    return y_stage3



    


def detrend_phase(sig,tr=10): #should threshold (tr) be changed??
    
    #print(sig)
    


    sig=sig.dropna()
    
    
    snr=10**(sig['SNR']/10)
    
    el=sig['Elevation']
    az=sig['Azimuth']
    
    #snr=snr/lowpass_filter_signal_cascaded(snr)
    
    
    sig=sig['Phase']
    
    sig_repaired,cycleslips=repair_discontinuities((sig*np.pi*2),threshold=tr)
    
    sig_filtered=filter_signal_cascaded(sig_repaired)
    
    


    
    signal=pd.DataFrame({'dphi':sig_filtered,'cycleslips':cycleslips.astype(int),'SNR':snr,'az':az,'el':el},index=sig_repaired.index)


    
    
    return signal
def fft_cross_correlation(x, y):
        n = len(x)
        fsize = 2 * n - 1
        X = np.fft.fft(x, fsize)
        Y = np.fft.fft(y, fsize)
        cc = np.fft.ifft(X * np.conjugate(Y)).real
        cc = np.fft.fftshift(cc)
        lags = np.arange(-n + 1, n)
        return cc, lags
    
    
def norm_fft_cross_correlation(x, y):
     # Normalize each signal before computing cross-correlation.
     x_norm = (x - np.mean(x)) / np.linalg.norm(x - np.mean(x))
     y_norm = (y - np.mean(y)) / np.linalg.norm(y - np.mean(y))
     cc, lags = fft_cross_correlation(x_norm, y_norm)
     return cc, lags.astype(int)

def autocorrelation(x):
     return norm_fft_cross_correlation(x, x)

def decorrelation_time(snr, thresh=1/np.e):
    
    snr=snr.dropna()
    if len(snr)<10:
        return np.nan
    acf,lags=autocorrelation(snr)
    
    if len(snr)>700:
        dt=20
    else:
        dt= 10
        
        
    norm_val = acf[lags == 0]
    if norm_val.size and norm_val[0] != 0:
        acf_norm = acf / norm_val[0]
    else:
        acf_norm = acf
    for lag, corr in zip(lags[lags >= 0], acf_norm[lags >= 0]):
        if corr < thresh:
            return lag * dt
        
    return (lags[lags >= 0])[-1] * dt




def compute_sigma(detrended_sig, reference_sig):
    dphi_true = (detrended_sig['dphi'] - reference_sig['dphi']).dropna()

    sigma_std = dphi_true.resample('1min').std()
    count = dphi_true.resample('1min').count()
    
    cycleslips = detrended_sig['cycleslips'].resample('1min').sum()

    el=detrended_sig['el'].resample('1min').median()
    az=detrended_sig['az'].resample('1min').median()

    linsnr=(detrended_sig['SNR'])
    
    
    s4=linsnr.resample('1min').std()/linsnr.resample('1min').mean()
    
    tau=linsnr.resample('1min').apply(decorrelation_time)
    
    s4_corr=np.sqrt((100/linsnr.resample('1min').mean())*(1+500/(19*linsnr.resample('1min').mean())))

    result = pd.DataFrame({
        'dphi': sigma_std,
        'count': count,
        'cycleslips': cycleslips,
        's4':s4,
        'tau':tau,
        'elev':el,
        'azim':az
        
    })
    return result




def getref(df,ismr):
    mind=df.index.min()
    maxd=df.index.max()

    mini_ismr=ismr[ismr['Elevation']>30].loc[mind:maxd]    
    


    maxentries=len(np.unique(mini_ismr.index))    
    
    
    phi60_max=mini_ismr.groupby('PRN')['Phi60_1'].max()[mini_ismr.groupby('PRN')['Phi60_1'].count()==maxentries]
    
    
    if phi60_max.astype(float).dropna().min()>0.08:
        print('No quiet channel! Min sigma = '+str(phi60_max.astype(float).dropna().min()))
    
    quietsat=phi60_max.astype(float).dropna().index[phi60_max.astype(float).dropna().argmin()]
   
    return quietsat

candidate_pairs = [
    ('GPS_L1CA', 'GPS_L2C'),
    ('GPS_L1CA', 'GPS_L2PY'),
    ('BDS_B1I', 'BDS_B2I'),
    ('BDS_B1I', 'BDS_B3I'),

    ('GLO_L1CA', 'GLO_L2CA'),
    ('GAL_L1BC', 'GAL_E5'),
    ('GAL_L1BC', 'GAL_E5a'),
    ('GAL_L1BC', 'GAL_E5b'),
    # ('BDS_B2I', 'BDS_B3I'), DO WE INCLUDE??
    # ('GPS_L2C', 'GPS_L5'),
    # ('GPS_L2PY', 'GPS_L2PY')
]

    
from sklearn.metrics import r2_score

def r2(df):
    if df['F1'].isnull().values.any() or df['F2'].isnull().values.any():
        return np.nan
    if len(df['F1'].dropna()>50*60*.95):
        return r2_score(df['F1'],df['F2'])
    else:
        return np.nan

def choose_frequencies_vectorized(df, candidate_pairs):
    # Create two new columns, F1 and F2, initially with NaN values.
    df['F1'] = np.nan
    df['F2'] = np.nan

    # Loop over candidate pairs and fill in F1 and F2 for rows where they haven't been set yet.
    for sig1, sig2 in candidate_pairs:
        # Only process if both columns exist in the DataFrame.
        if sig1 in df.columns and sig2 in df.columns:
            # Create a mask for rows that:
            # 1. Still have F1 as NaN (i.e. haven't been assigned by a higher priority candidate).
            # 2. Have non-missing values for both candidate signals.
            mask = df['F1'].isna() & df[sig1].notna() & df[sig2].notna()
            # Assign the candidate signals to F1 and F2 for those rows.
            df.loc[mask, 'F1'] = df.loc[mask, sig1]
            df.loc[mask, 'F2'] = df.loc[mask, sig2]
    return df
from sklearn.metrics import r2_score



def getref_gps(df):
    
    df_pivot = df.pivot_table(index=['level_2', 'SVID'], columns='SIG', values='dphi').reset_index().set_index('level_2')
    
    df_pivot = choose_frequencies_vectorized(df_pivot, candidate_pairs)
    df_pivot['minbin']=df_pivot.index.floor('min')    
    s=df_pivot.groupby(['SVID','minbin']).resample('1min').apply(r2)
    s=s.reset_index()
    result=s.loc[s.groupby('minbin')[0].idxmax()]
    return result.set_index('level_2')


from scipy.signal import welch,butter,filtfilt
def equilizer_filt(sig):
    fs = 20 #needs to be changed [DONE]
    cutoff = 3
    k=11
    nyquist = 0.5 * fs
    b, a = butter(1, cutoff/nyquist, btype='high')
    filtered_result=sig
    
    valid_mask = sig.notna()
    group_ids = (valid_mask != valid_mask.shift()).cumsum()
    
    for _, group in sig.groupby(group_ids):
        if group.isna().all():
            continue
        seg = group.to_numpy(dtype=float)
        if len(seg) <= 20*30: #changed from 50 to 20 again (Hz) control + F [DONE]
            # Skip filtering if the segment is too short
            filt_seg = seg
        else:
            try:
                # Try filtering with padlen=0
                filt_seg = filtfilt(b, a, seg, padlen=0)
            except ValueError:
                # As a fallback, if an error still occurs, skip filtering
                filt_seg = seg
        seg_filtered = seg + k * filt_seg
        filtered_result.loc[group.index] = seg_filtered

    return filtered_result


def continuity(v,thres,correct_diff=False):
    vec=v
    diff=np.diff(vec)
    disconts=np.where(abs(diff)>thres)[0]
    for i in disconts:
        vec[i+1:]=vec[i+1:]+vec[i]-vec[i+1]+correct_diff*diff[i+1]
    return vec


def continuitypd(v,thres=1e3,cutoff=5):
    vec=v.copy()
    dts=vec.index.diff().total_seconds().values
    dts[dts>cutoff]=cutoff
    slopes=abs(vec.diff()/dts)
    disconts=slopes[slopes>thres]
    for ind,val in disconts.items():
        i=vec.index.get_loc(ind)

        vec.iloc[i:]=vec.iloc[i:]+-(vec.iloc[i]-vec.iloc[i-1])+(vec.iloc[i-1]-vec.iloc[i-2])/(vec.index[i-1]-vec.index[i-2]).total_seconds()*dts[i]
    return vec



def CalculateTEC(pandas_arr,sat):
    

    cycl1=pandas_arr[(pandas_arr['SVID']==sat)&(pandas_arr['SIG'].str.contains('L1'))]['PR']
    cycl5=pandas_arr[(pandas_arr['SVID']==sat)&(pandas_arr['SIG'].str.contains('L5'))]['PR']

    #L5=0.254826 # l5 band 

  
    L1=0.19029 #l1 band wavelength m

    L5=0.2548231345 # l5 band 
    cteL1L5=7.7713726 #f1^2*f2^2/(f1^2-f2^2)/40.3

    l1l5=pd.merge(cycl1,cycl5,on='datetime')
    l1l5.dropna(inplace=True)
    phi=continuity((l1l5['PR_x'])*L1,thres=5e4)#%.1e8
    phi5=continuity((l1l5['PR_y'])*L5,thres=5e4)#%.1e8


    TECL1L5=(phi-phi5)*cteL1L5
    TECL1L5=continuity(TECL1L5, .3)
    return TECL1L5


def getwavelengths(g):
	if g == 'G':
		return 0.19029367,0.24421021
	elif g == 'S': #sbas
		return 0.19029367,0.2548231345  #This values work for SBAS L1,L2
	elif g == 'E':
		return 0.19029367,0.24834937 #This values work for GALILEO L1,L2 #0.25482805
	elif g == 'C': #bds
		return 0.19203949,0.24834937 #0.19203949 , 0.24834937
	elif g == 'R':
		return 0.18713637,0.2406039
    
def getcte(g):
	if g == 'G':
		return 9.529101453519065
	elif g == 'S': #sbas
		return 7.7713726 #This values work for SBAS L1,L29.529101453519065
	elif g == 'E':
		return 8.766203502852782 #This values work for GALILEO L1,L2 #0.25482805
	elif g == 'C': #bds
		return 9.002159172072213 #0.19203949 , 0.24834937
	elif g == 'R':
		return 9.76106837


def getsig(g):
	if g == 'G':
		return 'GPS_L1CA','GPS_L2C'
	elif g == 'S': #sbas
		return 'GEO_L1','GEO_L5'
	elif g == 'E':
		return 'GAL_L1BC','GAL_E5'
	elif g == 'C': #bds
		return 'BDS_B1I','BDS_B3I'
	elif g == 'R':
		return 'GLO_L1CA','GLO_L2CA'
    
def TEC(arr_groupby):
    
    global L1m,L2m,PR1,PR2
    sv=arr_groupby.name
    group=sv[0]
    
    #get constants and sig names for the group
    f1name,f2name=getsig(group)
    f1len,f2len=getwavelengths(group)
    const=getcte(group)
    
    #print(f1name,f2name,group)
    
    #get constants and sig names for the group
    f1=arr_groupby[arr_groupby['SIG']==f1name]
    f2=arr_groupby[arr_groupby['SIG']==f2name]
    
    #print(f1)
    L1=f1['Phase']
    PR1=f1['PR']

    L2=f2['Phase']
    PR2=f2['PR']

    L1m=L1*f1len
    L2m=L2*f2len

    #correct for slips and jumps
    L1m_corr=continuitypd(L1m,thres=5e4)#%.1e8
    L2m_corr=continuitypd(L2m,thres=5e4)#%.1e8
    
    #compute tECS
    phase_diff=const*(L2m_corr-L1m_corr)
    pTEC=continuitypd(phase_diff.dropna(), .3)
    cTEC=const*(PR1-PR2)
    return pd.DataFrame({'PTEC':pTEC,'CTEC':cTEC}, index=arr_groupby.index)

DATE='285'



gnssdic={0:'GPS',1:'SBS',2:'GAL',3:'BDS',6:'GLO'}

usecols = ['cons','svid','week','towe','elev','azim',
           'snr1','snr2','cph1','cph2','rng1','rng2']
elmask = 20
s4mask = 0.2

def readv325(path):
    dt = np.dtype([
        ('week', np.int32), ('towe', np.float32),
        ('leap', np.uint8), ('cons', np.uint8),
        ('sats', np.uint8), ('svid', np.uint8),
        ('elev', np.int8), ('azim', np.int32),
        ('snr1', np.uint8), ('snr2', np.uint8),
        ('snr3', np.uint8), ('pst1', np.uint8),
        ('pst2', np.uint8), ('pst3', np.uint8),
        ('rst1', np.uint8), ('rst2', np.uint8),
        ('rst3', np.uint8), ('cph1', np.float64),
        ('cph2', np.float64), ('cph3', np.float64),
        ('rng1', np.float64), ('rng2', np.float64),
        ('rng3', np.float64), ('lon', np.float32),
        ('lat', np.float32), ('hei', np.float32),
    ], align=True)
    arr = np.fromfile(path, dtype=dt)
    return pd.DataFrame.from_records(arr)

def readv326(path):
    rec_dt = np.dtype([
       ('towe', np.float32), ('cons', np.uint8),
       ('sats', np.uint8), ('svid', np.uint8),
       ('elev', np.int8), ('azim', np.int32),
       ('snr1', np.uint8), ('snr2', np.uint8),
       ('pst1', np.uint8), ('pst2', np.uint8),
       ('rst1', np.uint8), ('rst2', np.uint8),
       ('cph1', np.float64), ('cph2', np.float64),
       ('rng1', np.float64), ('rng2', np.float64),
       ('lck1', np.int32), ('lck2', np.int32),
    ], align=True)
    hdr_size = 60
    with open(path, 'rb') as f:
        buf = f.read(hdr_size)
    fmt = '@fBbiBBBBBBddddi'
    vals = struct.unpack(fmt, buf)
    week = vals[-1]
    rec = np.fromfile(path, dtype=rec_dt, offset=64*2)
    df = pd.DataFrame.from_records(rec)
    df['week'] = week
    return df

def unzip_func(filepath, outpath):
    os.makedirs(outpath, exist_ok=True)
    with zipfile.ZipFile(filepath, 'r') as z:
        z.extractall(outpath)
    return os.path.join(outpath, z.namelist()[0])

def getvers(path):
    m = re.search(r"_v(\d+)", os.path.basename(path))
    return m.group(1) if m else None

def process_file(file):

    
    start = time.time()
	#change this too!
    print(file)

    outfile = unzip_func(file,outpath=outdir)
    v = getvers(outfile)
    if v == '326':
        df = readv326(outfile)
    elif v == '325':
        df = readv325(outfile)
    else:
        return
    df = df[usecols]
    df['cons']=df['cons'].map(gnssdic)
    

    print(f"processed {file} in {time.time() - start:.2f} secs")
    return df



  



quiet_sats_to_check=[]
quiet_sats_method2=[]
total_df=pd.DataFrame()
sigma_list=[]

from multiprocessing import get_context

def worker(binzip):

    start_t=time.time()
    df_mosiac=process_file(binzip)  
    gps_epoch=pd.to_datetime('1980-01-06') #start time
    df_mosiac['datetime']=gps_epoch+pd.to_timedelta(df_mosiac['week']*7,unit='D')+pd.to_timedelta(df_mosiac['towe'],unit='s')
    df_mosiac.set_index('datetime',inplace=True)

    df_mosiac.sort_index(inplace=True)
    df_mosiac.dropna(inplace=True) # sort and clears out the created data frame 
    print('time taken to read:'+str(time.time()-start_t)) 

    mapping = {'elev':'Elevation',
            'azim':'Azimuth',
            'cph1':'Phase',
            'cph2':'Phase_2',
            'S401':'S4L1',
            'cons' : 'SIG', #this is needed for detrending, but they attempt to reinsert which does not work??
            'svid' : 'SVID',
            'snr1' : 'SNR'
                    }

    df_mosiac = df_mosiac.rename(columns = {k: v for k, v in mapping.items() if k in df_mosiac.columns})

    df_mosiac['SVID']=make_prn(df_mosiac)
    #detrend all signals
    start_t=time.time()


    df_mosiac=df_mosiac[df_mosiac.SVID!='R255']

    detrended_df=df_mosiac.groupby(['SVID']).apply(detrend_phase,tr=1).reset_index().set_index('datetime')
    
    detrended_df_l1=detrended_df
    
    print('time taken to detrend:'+str(time.time()-start_t))

    nots=detrended_df_l1[~detrended_df_l1['SVID'].str.contains('S')]
    ref_signal=pd.DataFrame({'dphi':nots.groupby(nots.index)['dphi'].median()})
    
    ref_signal.iloc[0:1000]=np.nan
    #calculate sigmas
    
    #sigma=detrended_df_l1.groupby('SVID').apply(compute_sigma,ref_signal).reset_index()
    sigma=detrended_df_l1
    sigma['Datetime']=sigma['datetime']
    sigma['PRN']=sigma['SVID']
    sigma['Sig_Phi_L1']=sigma['dphi']
    sigma.drop(['datetime','SVID','dphi'],axis=1,inplace=True)
    sigma=sigma.set_index('Datetime')
    #total_df=pd.concat([total_df,sigma])
    print('total time taken :'+str(time.time()-start_t))
    base = os.path.basename(binzip)
    newname = base.replace(".bin.zip", "lvl1_alt.pq")
    outpath = os.path.join(outdir, newname)
    sigma.to_parquet(outpath)
    return (binzip,outpath, None)


# %%


file='/Users/isaac/Documents/m_day/scintpi3_20240511_0800_756582.5000W_414055.0312N_v325.bin.zip'
worker(file)



# %%
binzip=file

start_t=time.time()
df_mosiac=process_file(binzip)  
gps_epoch=pd.to_datetime('1980-01-06') #start time
df_mosiac['datetime']=gps_epoch+pd.to_timedelta(df_mosiac['week']*7,unit='D')+pd.to_timedelta(df_mosiac['towe'],unit='s')
df_mosiac.set_index('datetime',inplace=True)

df_mosiac.sort_index(inplace=True)
df_mosiac.dropna(inplace=True) # sort and clears out the created data frame 
print('time taken to read:'+str(time.time()-start_t)) 

mapping = {'elev':'Elevation',
        'azim':'Azimuth',
        'cph1':'Phase',
        'cph2':'Phase_2',
        'S401':'S4L1',
        'cons' : 'SIG', #this is needed for detrending, but they attempt to reinsert which does not work??
        'svid' : 'SVID',
        'snr1' : 'SNR'
                }

df_mosiac = df_mosiac.rename(columns = {k: v for k, v in mapping.items() if k in df_mosiac.columns})

df_mosiac['SVID']=make_prn(df_mosiac)
#detrend all signals
start_t=time.time()


df_mosiac=df_mosiac[df_mosiac.SVID!='R255']

detrended_df=df_mosiac.groupby(['SVID']).apply(detrend_phase,tr=1).reset_index().set_index('datetime')

detrended_df_l1=detrended_df

print('time taken to detrend:'+str(time.time()-start_t))

nots=detrended_df_l1[~detrended_df_l1['SVID'].str.contains('S')]
ref_signal=pd.DataFrame({'dphi':nots.groupby(nots.index)['dphi'].median()})

ref_signal.iloc[0:1000]=np.nan
#calculate sigmas

#sigma=detrended_df_l1.groupby('SVID').apply(compute_sigma,ref_signal).reset_index()
sigma=detrended_df_l1.reset_index()
sigma['Datetime']=sigma['datetime']
sigma['PRN']=sigma['SVID']
sigma['Sig_Phi_L1']=sigma['dphi']
sigma.drop(['datetime','SVID','dphi'],axis=1,inplace=True)
sigma=sigma.set_index('Datetime')
#total_df=pd.concat([total_df,sigma])
print('total time taken :'+str(time.time()-start_t))
base = os.path.basename(binzip)
newname = base.replace(".bin.zip", "lvl1_alt.pq")
outpath = os.path.join(outdir, newname)
sigma.to_parquet(outpath)

# %%


old=pd.read_parquet('/Users/isaac/Documents/m_day/scintpi3_20240511_0800_756582.5000W_414055.0312N_v325lvl1_alt.pq')
new=pd.read_parquet('/Users/isaac/Documents/m_day/scintpi3_20240511_0800_756582.5000W_414055.0312N_v325lvl1_new.pq')



# %%

time=pd.Timestamp('2024-05-11 10:02:00')
period=pd.Timedelta(seconds=40)
sat='R03'


new_sub=new[(new.datetime>=time) & (new.datetime<time+period) & (new['prn']==sat)]

old_sub=old[(old.index>=time) & (old.index<time+period) & (old['PRN']==sat)]

ref_signal_sub=ref_signal[(ref_signal.index>=time) & (ref_signal.index<time+period)]


plt.plot(new_sub['datetime'],new_sub['detrended_cph1'],'.-',label='new')
plt.plot(old_sub.index,old_sub['Sig_Phi_L1'],'.-',label='old')
plt.plot(ref_signal_sub.index,ref_signal_sub['dphi'],'.-',label='ref')
# %%
