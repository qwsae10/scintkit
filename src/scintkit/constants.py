"""Lookup tables and processing thresholds used across scintkit."""

from datetime import datetime

# --- Receiver encodings -----------------------------------------------------

# Firmware tags (from the file name) that the readers understand.
SUPPORTED_FIRMWARE = ("v325", "v326c", "v326d", "v326e", "v326f")

INVALID_SVID = 255  # GLONASS satellite with unknown slot; rows are dropped

# Constellation codes (u-blox gnssId) and RINEX satellite prefixes.
CONSTELLATIONS = {0: "GPS", 1: "SBAS", 2: "GAL", 3: "BDS", 5: "QZSS", 6: "GLO"}
RINEX_PREFIX = {"GPS": "G", "SBAS": "S", "GAL": "E", "BDS": "C", "QZSS": "J", "GLO": "R"}
# Subtracted from svid to get the RINEX number (SBAS PRN 131 -> S31, QZSS 194 -> J02).
RINEX_SVID_OFFSET = {"SBAS": 100, "QZSS": 192}

# (constellation, first svid, last svid, channel-1 signal, MHz, channel-2 signal, MHz).
# Measured for v325 and v326c-f on 26 files, 2021-2026. No match = channel never valid.
# BeiDou-3 (svid >= 19) channel 2 is junk; SBAS has no channel 2; GLONASS MHz come from k.
SIGNALS = (
    ("GPS", 1, 32, "GPS_L1CA", 1575.42, "GPS_L2C", 1227.60),
    ("SBAS", 120, 158, "SBAS_L1CA", 1575.42, None, None),
    ("GAL", 1, 36, "GAL_E1", 1575.42, "GAL_E5b", 1207.14),
    ("BDS", 1, 18, "BDS_B1I", 1561.098, "BDS_B2I", 1207.14),
    ("BDS", 19, 63, "BDS_B1I", 1561.098, None, None),
    ("QZSS", 193, 202, "QZS_L1CA", 1575.42, "QZS_L2C", 1227.60),
    ("GLO", 1, 27, "GLO_L1OF", None, "GLO_L2OF", None),
)

# GLONASS frequency channel k per slot; slots 25-27 measured 2022-2026 only.
GLONASS_K = {
    1: 1, 2: -4, 3: 5, 4: 6, 5: 1, 6: -4, 7: 5, 8: 6, 9: -2, 10: -7, 11: 0, 12: -1,
    13: -2, 14: -7, 15: 0, 16: -1, 17: 4, 18: -3, 19: 3, 20: 2, 21: 4, 22: -3, 23: 3, 24: 2,
    25: -5, 26: -6, 27: -5,
}
GLONASS_L1_MHZ, GLONASS_L1_STEP_MHZ = 1602.0, 0.5625
GLONASS_L2_MHZ, GLONASS_L2_STEP_MHZ = 1246.0, 0.4375

L1_FREQUENCY_MHZ = 1575.42

PSEUDORANGE_MIN_M, PSEUDORANGE_MAX_M = 1.8e7, 4.5e7  # plausible GNSS ranges

# --- lvl0 conversion --------------------------------------------------------

GPS_EPOCH = datetime(1980, 1, 6)
DEFAULT_LEAP_SECONDS = 18  # GPS - UTC since 2017; used when a file does not say
V326_HEADER_FORMAT = "@fBbiBBBBBBddddi"  # towe, leap, ..., GPS week; padded to one record
LVL0_DROPPED_COLUMNS = (
    "sats", "pst1", "pst2", "pst3", "rst1", "rst2", "rst3", "lck1", "lck2",
    "snr3", "cph3", "rng3", "leap",
)
COMPRESSION = "brotli"

# --- Sampling ---------------------------------------------------------------

# The modal samples per satellite-second must cover this share of satellite-seconds.
MIN_MODAL_FRACTION = 0.5
# A reconstructed epoch further than this from its raw timestamp is not trusted.
MAX_CLOCK_CORRECTION_S = 1.0

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
MIN_TAU_SECONDS = 50  # of valid SNR per minute needed for tau
MIN_UNIQUE_SNR = 2
TAU_CORRELATION_LEVEL = 0.5  # tau is the lag where the autocorrelation drops below this

# --- Quality flags ----------------------------------------------------------

SIGMA_PHI_MAX_DROPPED_SAMPLES = 10  # per minute
S4_MIN_SAMPLE_FRACTION = 0.8  # of the expected samples per minute
