import glob
import os
# from datetime import datetime
# from datetime import timedelta
# from matplotlib import pyplot
# from matplotlib import colors
# matplotlib.use('Agg')
import time
import pandas as pd
import numpy as np
# import h5py
import pyarrow.dataset as ds
# from pathlib import Path
import gc

#datafolder = "/titan/frodrigues/scintpi_storage/sc000_pq_files"
datafolder = "/home/jxg200016/scratch/sc4_sc000_tmp"

initday = 180
endday = 190#shoudl go up to 189
def get_TEC_cte(conste,prn):
	if conste == '00':
		return 9.529101453519065
	elif conste == '01':
		return 7.771372607668402
	elif conste == '02':
		return 8.766203502852782
	elif conste == '03':
		if prn == 11 or prn == 12 or prn == 14:
			return 9.002159172072213
		else:
			return 11.765536230498425
	elif conste == '06':
		if prn == 14 or prn == 10 :   #k=-7
			return 9.71314456
		elif prn == 2 or prn == 6 :   #k=-4
			return 9.73366889
		elif prn == 18 or prn == 22 : #k=-3
			return 9.74051515
		elif prn == 13 or prn == 9 :  #k=-2
			return 9.74736382
		elif prn == 12 or prn == 16 : #k=-1
			return 9.75421489
		elif prn == 15 or prn == 11 : #k=0
			return 9.76106837
		elif prn == 5 or prn == 1 :   #k=+1
			return 9.76792425
		elif prn == 20 or prn == 24 : #k=+2
			return 9.77478255
		elif prn == 19 or prn == 23 : #k=+3
			return 9.78164325
		elif prn == 21 or prn == 17 : #k=+4
			return 9.78164325
		elif prn == 3 or prn == 7 :   #k=+5
			return 9.79537187
		elif prn == 4 or prn == 8 :   #k=+6
			return 9.80223979
		else:
			print ('Error with PRN',prn)
			return 0
	else :
		print ('Error with Constellation')
		return 0

def getwavelengths(conste,prn):
	#c = 299 792 458 m / s
	if conste == '00':
		return 0.19029367,0.24421021
	elif conste == '01':
		return 0.1902936728,0.2548280488 #This values work for SBAS L1,L5
	elif conste == '02':
		return 0.19029367,0.24834937 #This values work for GALILEO L1,L2 #0.25482805
	elif conste == '03':
		if prn == 11 or prn == 12 or prn == 14:   #k=+1
			return 0.19203949,0.24834937 #0.19203949 , 0.24834937
		else:
			return 0.19203949,0.23633246 #0.19203949 , 0.24834937
	elif conste == '06':
		return 0.18713637,0.2406039
	else:
		print ('Error with Constellation')
		return 0,0

def get_sat_config(sat, satnum):

    if sat[1] == 'G':
        GNSSid = '00'
        sig1 = 'GPS_L1CA'
        sig2 = 'GPS_L2PY' if satnum == '20' else 'GPS_L2C'

    elif sat[1] == 'E':
        GNSSid = '02'
        sig1 = 'GAL_E1BC'
        sig2 = 'GAL_E5b'

    elif sat[1] == 'C':
        GNSSid = '03'
        sig1 = 'BDS_B1I'
        sig2 = 'BDS_B2I' if satnum in ['11','12','14'] else 'BDS_B3I'

    elif sat[1] == 'R':
        GNSSid = '06'
        sig1 = 'GLO_L1CA'
        sig2 = 'GLO_L2CA'

    elif sat[1] == 'S':
        GNSSid = '00'
        sig1 = 'GEO_L1'
        sig2 = 'GEO_L5'

    else:
        GNSSid = '00'
        sig1 = 'GPS_L1CA'
        sig2 = 'GPS_L2C'

    return GNSSid, sig1, sig2

def compute_params(row):

    GNSSid, sig1, sig2 = get_sat_config(row['satellite'], row['satnum'])

    L1_wlen, L2_wlen = getwavelengths(GNSSid, int(row['satnum']))
    tec_cte = get_TEC_cte(GNSSid, int(row['satnum']))

    col1 = row.get(sig1, np.nan)
    col2 = row.get(sig2, np.nan)

    if np.isnan(col1) or np.isnan(col2):
        return np.nan

    return (-col2 * L2_wlen + col1 * L1_wlen) * tec_cte

for daystring in np.arange(initday,endday):
    try:
        dfs=[]
        for num in np.arange(1,5):
            start = time.perf_counter()
            files = glob.glob("%s/sept%03d%01d*_F.parquet"%(datafolder,daystring,num))
            print(files)
            dfsc4 = (
                ds.dataset(files, format="parquet")
                  .to_table()
                  .to_pandas()
            )
            dfsc4['linSNR1'] = 10**(dfsc4['SNR']/10.0)

            tbin = dfsc4.index.floor('1min')

            # Better than list(zip(...))
            keys = pd.MultiIndex.from_arrays(
                [dfsc4.SVID, dfsc4.SIG, tbin]
            )

            gid, gvals = pd.factorize(keys)

            x = dfsc4['linSNR1'].to_numpy()

            count = np.bincount(gid)
            sum_  = np.bincount(gid, weights=x)
            sum2  = np.bincount(gid, weights=x*x)

            mean = sum_ / count
            std  = np.sqrt(sum2 / count - mean**2)

            # ---- first Phase value per group ----
            first_idx = np.full(len(gvals), -1, dtype=np.int64)

            for i, g in enumerate(gid):
                if first_idx[g] == -1:
                    first_idx[g] = i

            phase_first = dfsc4['Phase'].to_numpy()[first_idx]

            # ---- output dataframe ----
            tmp = pd.DataFrame(
                {
                    'S4': std / mean,
                    'NOS': count,
                    'Phase': phase_first
                },
                index=pd.MultiIndex.from_tuples(
                    gvals,
                    names=['SVID', 'SIG', 'time']
                )
            )
            end = time.perf_counter()
            print(f"Time taken: {end - start:.6f} seconds")

            # tmp.to_parquet("%s/sept%03d%01d_lvl3.parquet"%(datafolder,daystring,num),index=True)

            # FREE MEMORY
            del dfsc4, gid, gvals, x
            del count, sum_, sum2, mean, std, tbin, files
            gc.collect()

            dfs.append(tmp)

        df1min = pd.concat(dfs)
        df1min.reset_index(inplace=True)
        df1min['satellite'] = 'P'+df1min['SVID']
        df1min['timestamp'] = df1min['time']
        df1min.drop(columns=['time','SVID'], inplace=True)
        df1min.set_index('timestamp',inplace=True)

        all_dfs = []
        unique_sats = df1min['satellite'].unique()

        for i, sat in enumerate(unique_sats):

            satnum = sat[2:4]

            if sat[1] == 'G':
                GNSSid = '00'
                sig1 = 'GPS_L1CA'
                sig2 = 'GPS_L2PY' if satnum == '20' else 'GPS_L2C'

            elif sat[1] == 'E':
                GNSSid = '02'
                sig1 = 'GAL_E1BC'
                sig2 = 'GAL_E5b'

            elif sat[1] == 'C':
                GNSSid = '03'
                sig1 = 'BDS_B1I'
                sig2 = 'BDS_B2I' if satnum in ['11','12','14'] else 'BDS_B3I'

            elif sat[1] == 'R':
                GNSSid = '06'
                sig1 = 'GLO_L1CA'
                sig2 = 'GLO_L2CA'

            elif sat[1] == 'S':
                GNSSid = '00'
                sig1 = 'GEO_L1'
                sig2 = 'GEO_L5'

            else:
                GNSSid = '00'
                sig1 = 'GPS_L1CA'
                sig2 = 'GPS_L2C'

            L1_wlen, L2_wlen = getwavelengths(GNSSid, int(satnum))

            df_sat = df1min[df1min['satellite'] == sat]

            if df_sat.empty:
                continue

            tec_cte = get_TEC_cte(GNSSid, int(satnum))

            df_pivot = (
                df_sat
                .pivot_table(
                    index='timestamp',
                    columns='SIG',
                    values=['Phase', 'S4', 'NOS'],
                    aggfunc='first'
                )
            )

            df_pivot.columns = [f'{v}_{k}' for v, k in df_pivot.columns]
            df_pivot = df_pivot.reset_index()

            # ensure required signals exist
            if f'Phase_{sig1}' not in df_pivot.columns or f'Phase_{sig2}' not in df_pivot.columns:
                continue

            df_pivot = df_pivot.dropna(subset=[f'Phase_{sig1}', f'Phase_{sig2}'])

            df_pivot['satellite'] = sat
            df_pivot['SVID'] = satnum
            df_pivot['GNSSid'] = GNSSid

            df_pivot['sig1'] = sig1
            df_pivot['sig2'] = sig2

            df_pivot['sTEC'] = (
                -df_pivot[f'Phase_{sig2}'] * L2_wlen +
                 df_pivot[f'Phase_{sig1}'] * L1_wlen
            ) * tec_cte

            df_pivot['sTEC'] -= df_pivot['sTEC'].min()

            # rename to L1/L2
            df_pivot = df_pivot.rename(columns={
                f'Phase_{sig1}': 'Phase_L1',
                f'Phase_{sig2}': 'Phase_L2',
                f'S4_{sig1}': 'S4_L1',
                f'S4_{sig2}': 'S4_L2',
                f'NOS_{sig1}': 'NOS_L1',
                f'NOS_{sig2}': 'NOS_L2',
            })
            df_pivot = df_pivot[
                                ['timestamp',
                                    'S4_L1',
                                    'S4_L2',
                                    'NOS_L1',
                                    'NOS_L2',
                                    'satellite',
                                    'sTEC',
                                    'sig1',
                                    'sig2'
                                ]
                            ]

            all_dfs.append(df_pivot)

        df_final = pd.concat(all_dfs, ignore_index=True)
        yearstr=np.unique(df_final['timestamp'].dt.year)[0]
        doystr =np.unique(df_final['timestamp'].dt.dayofyear)[0]
        df_final.to_parquet(
            "%s/sc4_lvl3_%03d_%04d_noelev_sc000.parquet"%(datafolder,doystr,yearstr),
            index=True
        )
    except Exception as e:
        print('error:',e)
        continue
