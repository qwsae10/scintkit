from ftplib import FTP_TLS
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import shutil


def date_to_gps_week(date_obj):
    """Convert a datetime object to GPS week and day of week."""
    gps_epoch = datetime(1980, 1, 6)
    delta_days = (date_obj - gps_epoch).days
    gps_week = delta_days // 7
    gps_dow = delta_days % 7
    return gps_week, gps_dow


def download_sp3_files(
    start_date,
    end_date,
    output_dir,
    email="anonymous@example.com",
):
    """
    Download and extract IAC final SP3 files.

    Parameters
    ----------
    start_date : str or datetime
        Start date in YYYY-MM-DD format or datetime object.

    end_date : str or datetime
        End date in YYYY-MM-DD format or datetime object.

    output_dir : str or Path
        Directory where SP3 files will be stored.

    email : str
        Email used for anonymous CDDIS login.
    """

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    if end_date < start_date:
        raise ValueError("End date must not be earlier than start date.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ftps = FTP_TLS("gdc.cddis.eosdis.nasa.gov")
    ftps.login(user="anonymous", passwd=email)
    ftps.prot_p()

    current_date = start_date

    try:

        while current_date <= end_date:

            year = current_date.year
            doy = current_date.timetuple().tm_yday

            gps_week, _ = date_to_gps_week(current_date)

            filename = (
                f"IAC0MGXFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3"
            )

            remote_file = (
                f"gnss/products/{gps_week:04d}/{filename}.gz"
            )

            gz_file = output_dir / f"{filename}.gz"
            sp3_file = output_dir / filename

            print(f"\nDownloading {filename}")

            if sp3_file.exists():
                print("✓ Already exists. Skipping.")
                current_date += timedelta(days=1)
                continue

            try:

                with open(gz_file, "wb") as f:
                    ftps.retrbinary(
                        f"RETR {remote_file}",
                        f.write,
                    )

                print("✓ Downloaded")

                with gzip.open(gz_file, "rb") as fin:
                    with open(sp3_file, "wb") as fout:
                        shutil.copyfileobj(fin, fout)

                gz_file.unlink()

                print(f"✓ Extracted {sp3_file.name}")

            except Exception as e:
                print(f"✗ Failed: {e}")

            current_date += timedelta(days=1)

    finally:
        ftps.quit()

    print("\nFinished downloading SP3 files.")

    