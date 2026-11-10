"""Column schema of lvl2/lvl3 files and the Parquet metadata written with every output."""

import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq

from scintkit import __version__
from scintkit.constants import COMPRESSION
from scintkit.stations import identify_station

logger = logging.getLogger(__name__)

_KEYS = [
    ("datetime", "datetime64[ns]", "UTC", "start of the minute (lvl3) or second (lvl2)"),
    ("prn", "string", "", "RINEX satellite label, e.g. G01, R05, E11, C23, J02, S31"),
    ("cons", "string", "", "constellation: GPS, SBAS, GAL, BDS, QZSS or GLO"),
    ("svid", "uint8", "", "satellite id as written by the receiver"),
    ("sig_1", "string", "", "signal on receiver channel 1, e.g. GPS_L1CA"),
    ("sig_2", "string", "", "signal on receiver channel 2, e.g. GPS_L2C; empty if none"),
    ("sig_3", "string", "", "signal on receiver channel 3 (ScintPi 4 only), e.g. GPS_L5; empty if none"),
    ("freq_1", "float64", "MHz", "carrier frequency of channel 1"),
    ("freq_2", "float64", "MHz", "carrier frequency of channel 2"),
    ("freq_3", "float64", "MHz", "carrier frequency of channel 3 (ScintPi 4 only)"),
]
_FIRST_SAMPLE = [
    ("snr1", "float32", "dB-Hz", "C/N0 of channel 1, first sample of the interval"),
    ("snr2", "float32", "dB-Hz", "C/N0 of channel 2, first sample of the interval"),
    ("snr3", "float32", "dB-Hz", "C/N0 of channel 3, first sample of the interval"),
    ("cph1", "float64", "cycles", "carrier phase of channel 1, first sample of the interval"),
    ("cph2", "float64", "cycles", "carrier phase of channel 2, first sample of the interval"),
    ("cph3", "float64", "cycles", "carrier phase of channel 3, first sample of the interval"),
    ("rng1", "float64", "m", "pseudorange of channel 1, first sample of the interval"),
    ("rng2", "float64", "m", "pseudorange of channel 2, first sample of the interval"),
    ("rng3", "float64", "m", "pseudorange of channel 3, first sample of the interval"),
    ("elev", "float32", "deg", "elevation, first sample; NaN where the receiver had no orbit"),
    ("azim", "float32", "deg", "azimuth, first sample; NaN where the receiver had no orbit"),
    ("tec_cph12", "float64", "TECU", "carrier-phase slant TEC levelled to tec_rng12, first sample"),
    ("tec_rng12", "float64", "TECU", "pseudorange slant TEC (uncalibrated biases), first sample"),
    ("tec_cph13", "float64", "TECU", "as tec_cph12 for channels 1 and 3, first sample"),
    ("tec_rng13", "float64", "TECU", "as tec_rng12 for channels 1 and 3, first sample"),
]


def _channel_products(i):
    return [
        (f"s4_{i}", "float64", "", "S4 of linear C/N0 over the minute"),
        (f"s4_corrected_{i}", "float64", "", "S4 with the C/N0 noise term removed"),
        (f"s4_detrended_{i}", "float64", "", "S4 of C/N0 after removing a 10 s moving average"),
        (f"tau_{i}", "float64", "s", "lag where the intensity autocorrelation drops below 0.5"),
        (f"sigma_phi_{i}", "float64", "rad", "std of high-pass filtered, clock-corrected phase"),
        (f"n_s4_{i}", "int32", "", "valid C/N0 samples in the minute"),
        (f"n_sigphi_{i}", "int32", "", "valid detrended phase samples in the minute"),
        (f"n_cycleslip_{i}", "int32", "", "samples flagged as cycle slips in the minute"),
        (f"s4_quality_flag_{i}", "int8", "", "1 = S4 not trustworthy (see docs/data_levels.md)"),
        (f"sigma_phi_quality_flag_{i}", "int8", "", "1 = sigma-phi not trustworthy (see docs/data_levels.md)"),
    ]


# Signal classification (lvl3 only); probabilities are NaN where ml_class is Unclassified.
_CLASSIFICATION = [
    ("ml_class", "string", "", "cause of the C/N0 fluctuation: Multipath, RFI or Scintillation; "
     "Unclassified when S4 < 0.2, single frequency, too few samples or classifier not run"),
    ("ml_prob_multipath", "float32", "", "model probability of Multipath"),
    ("ml_prob_rfi", "float32", "", "model probability of RFI"),
    ("ml_prob_scintillation", "float32", "", "model probability of Scintillation"),
]

# (name, dtype, unit, description) of every lvl2 column, in file order; lvl3 adds the classification.
COLUMNS = _KEYS + _FIRST_SAMPLE + _channel_products(1) + _channel_products(2) + _channel_products(3)
LVL3_COLUMNS = COLUMNS + _CLASSIFICATION
# Channel-3 columns are written only when the data have them (ScintPi 4).
CHANNEL3_NAMES = {"sig_3", "freq_3", "snr3", "cph3", "rng3", "tec_cph13", "tec_rng13"} | {
    c[0] for c in _channel_products(3)}
# Column names of a two-channel (ScintPi 3) file.
COLUMN_NAMES = [c[0] for c in COLUMNS if c[0] not in CHANNEL3_NAMES]
LVL3_COLUMN_NAMES = [c[0] for c in LVL3_COLUMNS if c[0] not in CHANNEL3_NAMES]


def conform(df, columns=COLUMNS):
    """The ``columns`` (COLUMNS or LVL3_COLUMNS) in order with their dtypes; channel-3 ones only if in ``df``."""
    columns = [c for c in columns if c[0] not in CHANNEL3_NAMES or c[0] in df.columns]
    return df[[c[0] for c in columns]].astype({name: dtype for name, dtype, _, _ in columns})


def file_metadata(source, firmware, fs, leap_seconds):
    """Parquet key/value metadata: version, firmware, rate (Hz), time scale, source and station."""
    meta = {
        "scintkit_version": __version__,
        "firmware": firmware,
        "sampling_rate_hz": f"{fs:g}",
        "time_scale": "UTC",
        "leap_seconds": str(leap_seconds),
        "source_file": os.path.basename(os.fspath(source)),
    }
    try:
        station = identify_station(filename=meta["source_file"])
    except ValueError as error:
        logger.warning("no station for %s: %s", meta["source_file"], error)
        station = None
    if station is not None:
        meta.update(
            station_code=str(station["Code"]),
            station_name=str(station["Station Location"]),
            station_latitude=str(station["Latitude"]),
            station_longitude=str(station["Longitude"]),
        )
    return meta


def write(df, path, metadata, compression_level=6):
    """Write ``df`` to Parquet with ``metadata`` added to the file's key/value metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    merged = {**(table.schema.metadata or {}), **{k.encode(): v.encode() for k, v in metadata.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.part"
    pq.write_table(table.replace_schema_metadata(merged), tmp,
                   compression=COMPRESSION, compression_level=compression_level)
    os.replace(tmp, path)


def read_metadata(path):
    """The scintkit key/value metadata of a Parquet file (empty for files written before it)."""
    raw = pq.read_schema(path).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}
