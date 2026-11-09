# Processing description

This document describes how scintkit converts raw receiver files into the
products defined in [data_levels.md](data_levels.md). The steps are listed in
the order in which they are applied. Every threshold named here is a constant
in `scintkit/constants.py`.

## 1. Reading raw files

ScintPi 3 files are binary records with one record per satellite and receiver
epoch. Each record holds the carrier-to-noise density ratio, carrier phase,
and pseudorange of the receiver channels, together with the elevation and
azimuth of the satellite. The record layout depends on the firmware version in
the file name. Firmware v325, v326c, v326d, v326e, and v326f are read. Other
versions raise an `UnsupportedFirmware` error and are skipped.

ScintPi 4 files are in Septentrio Binary Format. They are converted to text
with the Septentrio program `bin2asc` and then arranged in the same row layout.
ScintPi 4 records three channels at 50 Hz. Elevation and azimuth are taken from
the satellite visibility block of the file. Stations that do not log this
block have no elevation or azimuth.

## 2. Time

The receivers record GPS time. The time is converted to UTC by subtracting the
leap seconds stored in the file (18 s for all data since 2017).

ScintPi 3 stores the time of week as a 32-bit floating point number. The
resolution of this number decreases during the GPS week. After Saturday 01:38
it is 62.5 ms, which is coarser than the 50 ms sampling interval, and adjacent
epochs can receive the same time stamp. The records of one epoch are written
together, so the row order still separates the epochs. scintkit therefore
identifies receiver epochs from the row order and places them on the sampling
grid. When this reconstruction would move an epoch by more than 1 s, the
recorded time stamps are kept and a warning is logged.

The sampling rate is detected from the data as the most common number of
samples per satellite and second.

## 3. Signals and frequencies

For ScintPi 3, the signal on each channel was determined from the data. For a
tracked satellite, the ratio of the change in carrier phase to the change in
pseudorange equals the carrier frequency divided by the speed of light. This
ratio was measured in 26 files from 15 receivers recorded between 2021 and
2026. The measured frequencies agreed with the nominal frequencies to within
0.0034 MHz at the median. The assignment was the same for every firmware
version and station.

| Constellation | Channel 1 | Channel 2 |
|---|---|---|
| GPS | L1 C/A, 1575.42 MHz | L2C, 1227.60 MHz |
| Galileo | E1, 1575.42 MHz | E5b, 1207.14 MHz |
| BeiDou-2 | B1I, 1561.098 MHz | B2I, 1207.14 MHz |
| BeiDou-3 | B1I, 1561.098 MHz | None |
| GLONASS | L1, 1602 + 0.5625 k MHz | L2, 1246 + 0.4375 k MHz |
| QZSS | L1 C/A, 1575.42 MHz | L2C, 1227.60 MHz |
| SBAS | L1, 1575.42 MHz | None |

The GLONASS frequency channel number k of each satellite slot was measured in
the same files. The third channel of ScintPi 3 contains no measurements and is
dropped. GLONASS records without a slot number are also dropped.

A channel is treated as missing in a row when its carrier-to-noise density
ratio, carrier phase, or pseudorange is zero, or when the satellite does not
transmit a signal on that channel. Pseudoranges outside 18,000 to 45,000 km
are set to missing.

For ScintPi 4, the signal names are provided by the receiver. Channels 1, 2,
and 3 hold L1 C/A, L2C, and L5 for GPS, E1, E5a, and E5b for Galileo, B1I, B3I,
and B2a for BeiDou, and L1, L2, and L3 for GLONASS.

## 4. Phase scintillation index

The carrier phase of each satellite is processed in four steps.

1. Cycle slips are detected as phase steps that differ from the 10 s running
   median of the phase rate by more than 1 rad at the L1 frequency. The
   threshold is scaled with the carrier frequency. The detected steps are
   removed.
2. The phase is filtered with a sixth-order Butterworth high-pass filter with a
   cutoff of 0.1 Hz.
3. The receiver clock is estimated at each epoch as the median, over all
   satellites above 10° elevation, of the detrended phase divided by the
   carrier frequency. This term is removed from every satellite. When a file
   contains no elevation, all satellites are used.
4. The phase scintillation index is the standard deviation of the result over
   each minute.

## 5. Amplitude scintillation index

The carrier-to-noise density ratio is converted from dB-Hz to linear units.
The amplitude scintillation index S4 is the standard deviation of this
quantity over each minute divided by its mean.

Three related quantities are provided. `s4_corrected` removes the contribution
of receiver noise following Van Dierendonck et al. (1993). `s4_detrended` is
computed after a centered 10 s moving average was removed from the
carrier-to-noise density ratio. `tau` is the decorrelation time, defined as
the first lag at which the normalized autocorrelation of the linear intensity
falls below 0.5. It requires at least 50 s of valid samples in the minute.

## 6. Total electron content

Slant TEC is computed for each pair of channels from the difference of the
carrier phases and from the difference of the pseudoranges. Each satellite
pass is divided into segments at gaps longer than 5 minutes. Within a segment,
discontinuities larger than 1 TECU are removed. The carrier-phase TEC is then
shifted so that its median equals the median of the pseudorange TEC.

The TEC values include the differential code biases of the receiver and the
satellite. These biases are not removed, so the values are relative and can be
negative.

## 7. Signal classification

For 20 Hz files, the features described in Section 5 of
[data_levels.md](data_levels.md) are computed for each satellite and minute
in which both channels have more than 1190 samples and the S4 of channel 1
exceeds 0.15. This is the selection that was used to train the classifier. The
bundled classifier labels the selected minutes that have an S4 of at least 0.2
on channel 1 and a valid TEC. The model file is
`src/scintkit/data/histgb_full_25features.joblib`.

## 8. Known limitations

1. The high-pass filter is causal. It delays the phase and needs time to
   settle, which is why 180 s around every pass edge and gap are flagged.
2. GLONASS phase scintillation indices are always flagged.
3. TEC is not calibrated for code biases.
4. ScintPi 4 stations without the satellite visibility block have no elevation.
   Their receiver clock estimate includes low-elevation satellites.
5. Some ScintPi 3 files contain more receiver epochs than the nominal sampling
   rate allows. The recorded time stamps are kept for these files.

## References

Van Dierendonck, A. J., Klobuchar, J., and Hua, Q. (1993). Ionospheric
scintillation monitoring using commercial single frequency C/A code receivers.
Proceedings of ION GPS-93, Salt Lake City, UT, 1333–1342.
