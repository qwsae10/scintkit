import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scintkit.filenames import firmware_version, output_path, start_time
from scintkit.reading import (
    V325_RECORD,
    V326_RECORD,
    UnsupportedFirmware,
    normalise_geometry,
    read_lvl0,
    read_raw,
    to_lvl0,
)

FIXTURES = Path(__file__).parent / "fixtures"
V326F = FIXTURES / "scintpi3_20261101_0400_96.75784W_32.99184N_v326f.bin.zip"
V325 = FIXTURES / "scintpi3_20260615_0400_1114207.5000W_419337.7188N_v325.bin.zip"
OLD_LVL0 = FIXTURES / "scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"


def _raw_bytes(path):
    with zipfile.ZipFile(path) as zf:
        return zf.read(zf.namelist()[0])


def test_v326_reader_keeps_the_first_record_after_the_one_record_header():
    buf = _raw_bytes(V326F)
    n_records = (len(buf) - 64) // V326_RECORD.itemsize
    first = np.frombuffer(buf, V326_RECORD, count=1, offset=64)[0]
    header_towe = struct.unpack("@f", buf[:4])[0]

    df, leap = read_raw(V326F)

    assert leap == 18
    assert len(df) == n_records  # the fixture has no GLONASS svid 255 rows
    assert df["cph1"].iloc[0] == first["cph1"]
    assert 0 < first["towe"] - header_towe < 0.1


def test_v325_reader_drops_glonass_svid_255_and_channel_3():
    buf = _raw_bytes(V325)
    records = np.frombuffer(buf, V325_RECORD, count=len(buf) // V325_RECORD.itemsize)
    unknown_slot = (records["cons"] == 6) & (records["svid"] == 255)

    df, leap = read_raw(V325)

    assert leap == 18
    assert unknown_slot.any() and len(df) == len(records) - unknown_slot.sum()
    assert not {"snr3", "cph3", "rng3", "leap", "week", "towe"} & set(df.columns)


@pytest.mark.parametrize("path", [V326F, V325])
def test_raw_times_are_utc(path):
    df, _ = read_raw(path)
    latency = (df["datetime"].iloc[0] - pd.Timestamp(start_time(path))).total_seconds()
    assert 0 <= latency < 1  # not 18 s late (GPS time)
    assert df["datetime"].is_monotonic_increasing


@pytest.mark.parametrize("name", [
    "scintpi3_20240101_0000_96.7573W_32.9919N_v326.bin.zip",
    "scintpi3_20240101_0000_96.7573W_32.9919N_v326h.bin.zip",
    "scintpi3_20220101_0000_96.7573W_32.9919N_v324.dat.zip",
    "scintpi2_20220101_0000_96.7573W_32.9919N_v226.bin.zip",
])
def test_unsupported_firmware_is_one_clear_error(name):
    with pytest.raises(UnsupportedFirmware, match=firmware_version(name)):
        read_raw(name)


def test_unknown_geometry_becomes_nan_and_negative_elevation_stays():
    df = pd.DataFrame({"elev": np.array([-91, 99, 0, -23, 45, 90], dtype=np.int8),
                       "azim": [0, 999, 0, 120, 360, 10]})
    out = normalise_geometry(df)
    assert out["elev"].isna().tolist() == [True, True, False, False, False, False]
    assert out["elev"].iloc[3] == -23
    assert out["azim"].iloc[4] == 360


def test_old_lvl0_is_converted_to_utc_and_signed_elevation(tmp_path):
    old = pd.read_parquet(OLD_LVL0).head(4)
    old["elev"] = np.array([165, 216, 255, 45], dtype=np.uint8)
    path = tmp_path / OLD_LVL0.name
    old.to_parquet(path)

    df, meta = read_lvl0(path)

    assert (old["datetime"] - df["datetime"]).eq(pd.Timedelta(seconds=18)).all()
    assert np.isnan(df["elev"].iloc[0])
    assert df["elev"].iloc[1:].tolist() == [-40, -1, 45]
    assert meta["time_scale"] == "UTC" and meta["firmware"] == "v326f"


def test_new_lvl0_round_trip_is_not_shifted_again(tmp_path):
    raw, _ = read_raw(V326F)
    df, meta = read_lvl0(to_lvl0(V326F, tmp_path))
    assert df["datetime"].equals(raw["datetime"])
    assert meta["sampling_rate_hz"] == "20"


def test_output_names_come_from_the_basename():
    assert firmware_version("/a/sc000_v325/scintpi3_20241011_1200_x_v326f_lvl0.pq") == "v326f"
    assert output_path("/data/sc000_lvl0/scintpi3_x_v326f.bin.zip", "/out", "lvl3") == \
        "/out/scintpi3_x_v326f_lvl3.pq"
    assert output_path("/data/scintpi3_x_v326f_lvl0.pq", "/out", "lvl2") == "/out/scintpi3_x_v326f_lvl2.pq"
