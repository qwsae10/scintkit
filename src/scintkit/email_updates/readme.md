# email_updates

Scans the raw-data archive for the days on which each station delivered files
over the last few months, draws one availability bar per station, and emails
the PNG.

Station metadata comes from the bundled registry
(`src/scintkit/data/station_scintpi_codes.csv`). A day counts as "data present"
when at least one file for that day exists:

- ScintPi 2/3: `*.bin.zip` / `*.dat.zip` under `<archive>/*/`, matched to a
  station by the coordinates in the file name.
- ScintPi 4: `<prefix>*_` files under `<archive>/*/<YYYYMMDD>/`, matched by the
  station's SC4 prefix.
- Septentrio: `CSS_DDDN.YY_` and `.ismr` files in `--septentrio-dir`.

## Environment variables

- `SMTP_USER`, `SMTP_PASS` (required): SMTP login. For Gmail use an app password.
- `SMTP_SENDER`: From address (default `SMTP_USER`).
- `SMTP_HOST`, `SMTP_PORT`: default `smtp.gmail.com`, `587` (STARTTLS).
- `SCINTPI_STATUS_RECIPIENTS`: comma-separated recipients, used when `--to` is not given.

## Usage

```bash
python -m scintkit.email_updates.run_pipeline --archive /data/scintpi \
    --septentrio-dir /data/septentrio --to ops@example.org --months 3
```

Options: `--to`, `--archive` (required), `--septentrio-dir` (scan skipped if
omitted), `--months` (default 3), `--output` (default
`stations_availability_lvl3_YYYY_MM_DD.png`). Any failure exits non-zero.

Weekly crontab entry (Mondays 06:00):

```cron
0 6 * * 1  . $HOME/.scintpi_env && python -m scintkit.email_updates.run_pipeline --archive /data/scintpi
```
