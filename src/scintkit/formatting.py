
import pandas as pd
import numpy as np


CONSTELLATION_PREFIX = {
    "GPS": "G",
    "BDS": "C",
    "GAL": "E",
    "GLO": "R",
    "QZSS": "J",
    "IRNSS": "I",
    "SBAS": "S",
    "SBS": "S",
}
NUMERIC_CONSTELLATION = {0: "GPS", 1: "SBS", 2: "GAL", 3: "BDS", 6: "GLO"}
SIGNALS_BY_CONSTELLATION = {
    "GPS": ("GPS_L1CA", "GPS_L2C", "GPS_L5"),
    "GLO": ("GLO_L1CA", "GLO_L2C", "GLO_L3"),
    "GAL": ("GAL_L1BC", "GAL_E5b", "GAL_E5b"),
    "BDS": ("BDS_B1I", "BDS_B2I", "BDS_B3I"),
    "QZSS": ("QZS_L1CA", "QZS_L2C", "QZS_L5"),
}
FREQUENCY_MHZ_BY_SIGNAL = {
    "GPS_L1CA": 1575.42,
    "GLO_L1CA": 1602,
    "GEO_L1": 1575.42,
    "QZS_L1CA": 1575.42,
    "GAL_L1BC": 1575.42,
    "GAL_E1": 1575.42,
    "BDS_B1I": 1561.098,
    "IRNSS_L5": 1176.45,
    "GPS_L2C": 1227.60,
    "GLO_L2C": 1246.00,
    "GLO_L2CA": 1246.60,
    "QZS_L2C": 1227.60,
    "GAL_E5a": 1176.45,
    "SBAS_L5": 1176.45,
    "BDS_B2I": 1207.14,
    "GEO_L5": 1176.45,
    "GPS_L5": 1176.45,
    "QZS_L5": 1176.45,
    "GAL_E5b": 1207.14,
    "BDS_B3I": 1268.52,
    "GPS_L2PY": 1227.60,
    "GPS_L1P": 1575.42,
    "GLO_L1P": 1602.00,
    "GLO_L2P": 1246.0,
    "GAL_E5": 1191.795,
    "GAL_E6BC": 1278.75,
    "GLO_L3": 1202.025,
}


def make_prn(dfin):
    # Mapping the small set of unique SVIDs avoids invoking Python once per
    # row on multi-million-row receiver files.
    svid_labels = {
        value: str(int(value)).zfill(2)
        for value in dfin["svid"].unique()
    }
    return dfin["cons"].map(CONSTELLATION_PREFIX) + dfin["svid"].map(svid_labels)


def zero_cph_snr_to_nan(df):
    cols = [
        col for col in df.columns
        if (col.startswith("cph") or col.startswith("snr")) and col[3:].isdigit()
    ]
    for column in cols:
        df[column] = df[column].mask(df[column].eq(0))
    return df    
                                                                                 
def temp_formating(df):
    #check if cons is numeric
    s = pd.to_numeric(df['cons'], errors='coerce')

    if s.notna().all():
        df['cons'] = s.map(NUMERIC_CONSTELLATION)
    invalid_glonass = (df['cons'] == 'GLO') & (df['svid'] == 255)
    if invalid_glonass.any():
        df = df.loc[~invalid_glonass].copy()

    if not isinstance(df.index, pd.RangeIndex) or not df.index.equals(
        pd.RangeIndex(len(df))
    ):
        df = df.reset_index(drop=True)
    df['minbin'] = df['datetime'].dt.floor('1min')
    df['prn']=make_prn(df)
    df=add_sigs(df)
    df=zero_cph_snr_to_nan(df)
    return df


def add_sigs(df):
    if 'sig_1' not in df.columns:
        #scintpi3 doesn't have sig columns, but we can infer them from cons and svid. 
        #hardcoded for now, but could be made more flexible if needed
        for signal_number in (1, 2, 3):
            signal_map = {
                constellation: signals[signal_number - 1]
                for constellation, signals in SIGNALS_BY_CONSTELLATION.items()
            }
            df[f'sig_{signal_number}'] = df['cons'].map(signal_map)

    for signal_number in (1, 2, 3):
        df[f'freq_{signal_number}'] = df[f'sig_{signal_number}'].map(
            FREQUENCY_MHZ_BY_SIGNAL
        )

    return df


def make_1sec(df):
    """
    Resample the dataframe to 1 second intervals, grouping by 'datetime' and 'prn'.
    Default method is 'first', but can be changed to any valid pandas aggregation method (e.g., 'mean', 'max', 'min').
    """

    df['secbin'] = df['datetime'].dt.floor('1s')
    df=df.groupby(['secbin', 'prn']).first().reset_index()
    return df

def make_1min(df,method='first'):

    """
    Resample the dataframe to 1 minute intervals, grouping by 'minbin' and 'prn'.
    Default method is 'first', but can be changed to any valid pandas aggregation method (e.g., 'mean', 'max', 'min').
    """

    df=df.groupby(['minbin', 'prn']).agg(method).reset_index()
    return df
