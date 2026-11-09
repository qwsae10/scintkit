# Data levels

scintkit writes three levels of Parquet files. This document defines the rows,
columns, metadata, quality flags, and signal classes of each level. The column
definitions are stored in `scintkit.schema` and are enforced when a file is
written.

## 1. Levels

| Level | File name | Rows |
|---|---|---|
| lvl0 | `<raw name>_lvl0.pq` | One row per satellite and receiver epoch |
| lvl2 | `<raw name>_lvl2.pq` | One row per satellite and second |
| lvl3 | `<raw name>_lvl3.pq` | One row per satellite and minute |

All times are Coordinated Universal Time (UTC). In lvl2 and lvl3, `datetime`
is the start of the second or minute.

The lvl0 file holds the receiver measurements without derived products. Its
columns are `datetime`, `cons`, `svid`, `elev`, `azim`, and the carrier-to-noise
density ratio (`snr`), carrier phase (`cph`), and pseudorange (`rng`) of each
receiver channel. ScintPi 4 lvl0 files also hold the signal name and carrier
frequency of each channel.

## 2. Columns of lvl2 and lvl3

The suffix `1`, `2`, or `3` identifies the receiver channel. The table lists
the columns of channel 1. Channel 2 has the same columns. Channel 3 columns,
`tec_cph13`, and `tec_rng13` are written for ScintPi 4 only, because the third
channel of ScintPi 3 contains no measurements. The four classification columns
are written in lvl3 only.

| Column | Type | Unit | Description |
|---|---|---|---|
| `datetime` | datetime64[ns] | UTC | Start of the minute (lvl3) or second (lvl2) |
| `prn` | string | | RINEX satellite label, for example G01, R05, E11, C23, J02, or S31 |
| `cons` | string | | Constellation, namely GPS, SBAS, GAL, BDS, QZSS, or GLO |
| `svid` | uint8 | | Satellite number written by the receiver |
| `sig_1` | string | | Signal on channel 1, for example GPS_L1CA |
| `freq_1` | float64 | MHz | Carrier frequency of channel 1 |
| `snr1` | float32 | dB-Hz | Carrier-to-noise density ratio, first sample of the interval |
| `cph1` | float64 | cycles | Carrier phase, first sample of the interval |
| `rng1` | float64 | m | Pseudorange, first sample of the interval |
| `elev` | float32 | deg | Elevation, first sample. Missing where the receiver had no orbit. |
| `azim` | float32 | deg | Azimuth, first sample. Missing where the receiver had no orbit. |
| `tec_cph12` | float64 | TECU | Slant TEC from the carrier phases of channels 1 and 2, leveled to `tec_rng12`, first sample |
| `tec_rng12` | float64 | TECU | Slant TEC from the pseudoranges of channels 1 and 2, first sample |
| `s4_1` | float64 | | Amplitude scintillation index S4 over the minute |
| `s4_corrected_1` | float64 | | S4 with the contribution of receiver noise removed |
| `s4_detrended_1` | float64 | | S4 after a 10 s moving average was removed from the carrier-to-noise density ratio |
| `tau_1` | float64 | s | Decorrelation time of the signal intensity |
| `sigma_phi_1` | float64 | rad | Phase scintillation index, the standard deviation of the detrended carrier phase over the minute |
| `n_s4_1` | int32 | | Number of valid amplitude samples in the minute |
| `n_sigphi_1` | int32 | | Number of valid detrended phase samples in the minute |
| `n_cycleslip_1` | int32 | | Number of samples marked as cycle slips in the minute |
| `s4_quality_flag_1` | int8 | | 1 if S4 should not be used (Section 4) |
| `sigma_phi_quality_flag_1` | int8 | | 1 if the phase scintillation index should not be used (Section 4) |
| `ml_class` | string | | Multipath, RFI, Scintillation, or Unclassified (Section 5) |
| `ml_prob_multipath` | float32 | | Classifier probability of multipath |
| `ml_prob_rfi` | float32 | | Classifier probability of radio frequency interference |
| `ml_prob_scintillation` | float32 | | Classifier probability of scintillation |

The first-sample columns of one row come from the same receiver epoch. A
channel without a measurement in that epoch is missing (NaN). The TEC columns
contain uncalibrated receiver and satellite biases and can be negative.

## 3. File metadata

Every file carries key-value metadata in the Parquet footer. It can be read
with `scintkit.schema.read_metadata(path)`.

| Key | Content |
|---|---|
| `scintkit_version` | Version of scintkit that wrote the file |
| `source_file` | Name of the raw file |
| `firmware` | Firmware version from the file name |
| `sampling_rate_hz` | Sampling rate detected from the data |
| `time_scale` | `UTC` |
| `leap_seconds` | Leap seconds subtracted from the receiver time |
| `station_code`, `station_name`, `station_latitude`, `station_longitude` | Station registry entry, when the file name matches a station |
| `elevation_available` | Whether the file contains any elevation |
| `ml_classifier_ran`, `ml_model`, `ml_model_sha256` | Whether the classifier ran, and the model file and its hash (lvl3) |

Parquet files written by earlier versions have no `time_scale` entry. They are
in GPS time. scintkit converts them to UTC when they are read.

## 4. Quality flags

A flag of 0 means that the product passed the automatic checks. A flag of 1
means that the product should not be used without review. Here `fs` is the
sampling rate in hertz.

`sigma_phi_quality_flag` is set to 1 when any of the following conditions is
true.

1. The minute is within 180 s of the start or end of a satellite pass, of a
   data gap longer than 2 s, or of a missing phase sample. The high-pass filter
   has not settled in this interval.
2. More than 10 of the `fs × 60` expected phase samples are missing.
3. The satellite is GLONASS.

`s4_quality_flag` is set to 1 when fewer than 80% of the `fs × 60` expected
amplitude samples are present.

The two flags are independent. A GLONASS satellite always fails the phase
check and can pass the amplitude check.

## 5. Signal class

A gradient boosting classifier assigns a cause to elevated amplitude
fluctuations. The classes are `Multipath`, `RFI` (radio frequency
interference), and `Scintillation`. The classifier uses 25 features of each
satellite and minute, namely the spectral power of the two carrier-to-noise
density ratios and of the TEC in five frequency bands, statistics of the
fluctuation that is common to all satellites, and the S4 of both channels.

A minute is labeled `Unclassified`, and its three probabilities are missing,
when any of the following conditions is true.

1. The S4 of channel 1 is below 0.2.
2. The satellite has no valid second frequency in that minute.
3. Either channel has 1190 or fewer of the 1200 samples of the minute.
4. The data are not sampled at 20 Hz. The classifier was trained on 20 Hz
   ScintPi 3 data, so ScintPi 4 files (50 Hz) are not classified.
5. scikit-learn is not installed, or classification was turned off.

The metadata entry `ml_classifier_ran` states whether the classifier ran on
the file.
