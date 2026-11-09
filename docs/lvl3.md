# Level 3 products

Level 3 files contain one row per satellite and one-minute interval;
`datetime` is the UTC start of the minute. Product suffixes `_1` and `_2`
identify receiver channels 1 and 2 (`sig_1`/`sig_2`, `freq_1`/`freq_2`). The
column list, with dtypes and units, is `scintkit.schema.COLUMNS`.

`n_sigphi_1` and `n_sigphi_2` contain the number of valid detrended phase
samples used to calculate sigma-phi. `n_s4_1` and `n_s4_2` contain the number
of valid SNR samples used to calculate S4. `tau_1`/`tau_2` are the lag (s) at
which the normalised autocorrelation of the linear intensity first drops below
0.5.

## Quality flags

Quality flags are binary integers. `0` means the product passes the automatic
quality checks, and `1` means it should not be used without further review.

`sigma_phi_quality_flag_1` and `sigma_phi_quality_flag_2` are set to `1` when any of the following is true:

- the phase edge/gap mask is set anywhere in the one-minute interval;
- the corresponding `n_sigphi_1/2` count is less than `fs * 60 - 10`, where
  `fs` is the sampling rate in hertz (detected from the data and stored in the
  file metadata as `sampling_rate_hz`) (exactly 10 dropped samples is
  allowed); or
- the satellite is GLONASS (its PRN begins with `R`).

`s4_quality_flag_1` and `s4_quality_flag_2` are set to `1`
when the corresponding `n_s4_1/2` count is less than `fs * 60 * 0.8`. A
minute with exactly 80% of its expected samples remains good.

The sigma-phi and S4 flags are independent. For example, GLONASS automatically
fails the sigma-phi check but can still have a good S4 flag.
