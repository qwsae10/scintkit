"""Lookup tables and processing thresholds used across scintkit."""

from datetime import datetime

# --- Receiver encodings -----------------------------------------------------

INVALID_SVID = 255  # receiver placeholder for an untracked channel

# Constellation codes written by ScintPi 3 firmware.
NUMERIC_CONSTELLATION = {0: "GPS", 1: "SBS", 2: "GAL", 3: "BDS", 6: "GLO"}

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
GLONASS_PREFIX = CONSTELLATION_PREFIX["GLO"]

# Signal tracked on receiver channels 1, 2 and 3 (ScintPi 3 files carry no
# signal columns).
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
L1_FREQUENCY_MHZ = 1575.42

# --- lvl0 conversion --------------------------------------------------------

GPS_EPOCH = datetime(1980, 1, 6)
LVL0_DROPPED_COLUMNS = ("sats", "pst1", "pst2", "rst1", "rst2", "lck1", "lck2")
LVL0_COMPRESSION = "brotli"

# --- Sampling ---------------------------------------------------------------

# Samples per satellite per minute that identify a 10, 20, 40 or 50 Hz file.
SAMPLES_PER_MINUTE = (600, 1200, 2400, 3000)
SAMPLES_PER_MINUTE_TOLERANCE = 10

# --- Phase detrending -------------------------------------------------------

HIGHPASS_CUTOFF_HZ = 0.1
CYCLE_SLIP_THRESHOLD_RAD = 1.0  # at L1; scaled by frequency for other signals
REPAIR_WINDOW_S = 10  # rolling-median window for slip detection
MAX_SLIP_FRACTION = 0.2  # arcs with more flagged epochs are left unrepaired
EDGE_GAP_S = 2  # a time step longer than this is a data gap
EDGE_PAD_S = 180  # filter settling time masked around arc edges and gaps
CLOCK_ELEVATION_MASK_DEG = 10  # satellites used for the receiver clock estimate

# --- TEC --------------------------------------------------------------------

SPEED_OF_LIGHT = 299792458  # m/s
IONO_CONSTANT = 40.3  # m^3/s^2, first-order ionospheric refraction
TECU = 1e16  # electrons/m^2
TEC_SLIP_THRESHOLD_TECU = 1
TEC_MAX_GAP = "5min"  # a longer gap starts a new levelling segment

# --- Amplitude --------------------------------------------------------------

SNR_DETREND_WINDOW = "10s"  # centred moving average removed from SNR
DETRENDED_SNR_OFFSET_DB = 40  # added back so detrended SNR stays positive
MIN_TAU_SAMPLES = 1000
MIN_UNIQUE_SNR = 2

# --- Quality flags ----------------------------------------------------------

SIGMA_PHI_MAX_DROPPED_SAMPLES = 10  # per minute
S4_MIN_SAMPLE_FRACTION = 0.8  # of the expected samples per minute
