from pathlib import Path
import pandas as pd
import numpy as np
import time


def pq_reshaping_lvl0(parquet_file, output_dir=None):
    """
    Convert a Septentrio measurement parquet file to a Level-0 parquet.

    Parameters
    ----------
    parquet_file : str or Path
        Path to the input parquet file.

    output_dir : str or Path, optional
        Directory to save the output parquet. If None, the output is saved
        in the same directory as the input file.

    Returns
    -------
    lvl0 : pandas.DataFrame
        Processed Level-0 dataframe.

    outfile : Path
        Path to the saved Level-0 parquet.
    """

    parquet_file = Path(parquet_file)

    if output_dir is None:
        output_dir = parquet_file.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Read parquet
    # --------------------------------------------------

    df = pd.read_parquet(parquet_file)

    df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # --------------------------------------------------
    # Signal mapping
    # --------------------------------------------------

    mapping = {

        # Sig1
        'GPS_L1CA': (1575.42, 'Sig1'),
        'GLO_L1CA': (1602.00, 'Sig1'),
        'GEO_L1': (1575.42, 'Sig1'),
        'QZS_L1CA': (1575.42, 'Sig1'),
        'GAL_L1BC': (1575.42, 'Sig1'),
        'GAL_E1': (1575.42, 'Sig1'),
        'GAL_E1BC': (1575.42, 'Sig1'),
        'BDS_B1I': (1561.098, 'Sig1'),
        'IRNSS_L5': (1176.45, 'Sig1'),

        # Sig2
        'GPS_L2C': (1227.60, 'Sig2'),
        'GLO_L2C': (1246.00, 'Sig2'),
        'GLO_L2CA': (1246.60, 'Sig2'),
        'QZS_L2C': (1227.60, 'Sig2'),
        'GAL_E5a': (1176.45, 'Sig2'),
        'SBAS_L5': (1176.45, 'Sig2'),
        'BDS_B2I': (1207.14, 'Sig2'),
        'GEO_L5': (1176.45, 'Sig2'),

        # Sig3
        'GPS_L5': (1176.45, 'Sig3'),
        'QZS_L5': (1176.45, 'Sig3'),
        'GAL_E5b': (1207.14, 'Sig3'),
        'BDS_B3I': (1268.52, 'Sig3'),

        # Sig4
        'GPS_L2PY': (1227.60, 'Sig4'),
        'GPS_L1P': (1575.42, 'Sig4'),
        'GLO_L1P': (1602.00, 'Sig4'),
        'GLO_L2P': (1246.00, 'Sig4'),
        'GAL_E5': (1191.795, 'Sig4'),
        'GAL_E6BC': (1278.75, 'Sig4'),
        'GLO_L3': (1202.025, 'Sig4')
    }

    df["freq"] = df["SIG"].map(
        lambda x: mapping.get(x, (np.nan, None))[0]
    )

    df["sig_group"] = df["SIG"].map(
        lambda x: mapping.get(x, (np.nan, None))[1]
    )

    df = df[df["sig_group"].notna()].copy()

    # --------------------------------------------------
    # Pivot tables
    # --------------------------------------------------

    sig_pivot = (
        df.pivot_table(
            index=["datetime", "SVID"],
            columns="sig_group",
            values="SIG",
            aggfunc="first"
        )
        .rename(columns={
            "Sig1": "sig_1",
            "Sig2": "sig_2",
            "Sig3": "sig_3",
            "Sig4": "sig_4"
        })
    )

    freq_pivot = (
        df.pivot_table(
            index=["datetime", "SVID"],
            columns="sig_group",
            values="freq",
            aggfunc="first"
        )
        .rename(columns={
            "Sig1": "freq_1",
            "Sig2": "freq_2",
            "Sig3": "freq_3",
            "Sig4": "freq_4"
        })
    )

    snr_pivot = (
        df.pivot_table(
            index=["datetime", "SVID"],
            columns="sig_group",
            values="SNR",
            aggfunc="first"
        )
        .rename(columns={
            "Sig1": "snr1",
            "Sig2": "snr2",
            "Sig3": "snr3",
            "Sig4": "snr4"
        })
    )

    phase_pivot = (
        df.pivot_table(
            index=["datetime", "SVID"],
            columns="sig_group",
            values="Phase",
            aggfunc="first"
        )
        .rename(columns={
            "Sig1": "cph1",
            "Sig2": "cph2",
            "Sig3": "cph3",
            "Sig4": "cph4"
        })
    )

    pr_pivot = (
        df.pivot_table(
            index=["datetime", "SVID"],
            columns="sig_group",
            values="PR",
            aggfunc="first"
        )
        .rename(columns={
            "Sig1": "rng1",
            "Sig2": "rng2",
            "Sig3": "rng3",
            "Sig4": "rng4"
        })
    )

    lvl0 = pd.concat(
        [sig_pivot, freq_pivot, snr_pivot, phase_pivot, pr_pivot],
        axis=1
    ).reset_index()

    # --------------------------------------------------
    # Satellite information
    # --------------------------------------------------

    constellation_map = {
        'G': 'GPS',
        'R': 'GLO',
        'E': 'GAL',
        'C': 'BDS',
        'J': 'QZS',
        'I': 'IRNSS',
        'S': 'SBAS'
    }

    lvl0["svid_str"] = lvl0["SVID"]
    lvl0["prn"] = "P" + lvl0["svid_str"]
    lvl0["cons"] = lvl0["svid_str"].str[0].map(constellation_map)
    lvl0["svid"] = lvl0["svid_str"].str[1:].astype(int)

    lvl0["minbin"] = lvl0["datetime"]
    lvl0["timestamp"] = lvl0["datetime"]
    lvl0["satellite"] = lvl0["prn"]

    lvl0.insert(0, "index", np.arange(len(lvl0)))

    desired_cols = [
        'index','cons','svid','satellite','timestamp',
        'snr1','snr2',
        'cph1','cph2',
        'rng1','rng2',
        'datetime','minbin','prn',
        'sig_1','sig_2','sig_3',
        'freq_1','freq_2','freq_3'
    ]

    for col in desired_cols:
        if col not in lvl0.columns:
            lvl0[col] = np.nan

    lvl0 = lvl0[desired_cols]

    lvl0 = lvl0.sort_values(
        ["datetime", "prn"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    first_time = pd.to_datetime(lvl0["datetime"].iloc[0])

    outfile = (
        output_dir /
        f"{parquet_file.stem}_noelev_lvl0.parquet"
    )

    lvl0.to_parquet(outfile, index=False)

    #print(f"Saved: {outfile}")

    return lvl0, outfile


