import pytest

from scintkit.stations import identify_station


def test_identify_station_from_coordinates():
    station = identify_station(latitude=32.9919, longitude=-96.7573)

    assert station["Code"] == "US-TX1"
    assert station["Station Location"] == "Dallas Texas"


def test_identify_station_from_decimal_coordinate_filename():
    station = identify_station(
        filename="scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
    )

    assert station["Code"] == "US-TX1"
    assert station["Type"] == "SC3"


def test_identify_station_from_scaled_coordinate_filename():
    station = identify_station(
        filename="scintpi3_20240511_0400_1203575.6250W_432707.1250N_v325.pq"
    )

    assert station["Code"] == "US-OR1"


def test_identify_sc4_station_from_filename_prefix():
    station = identify_station(filename="mx02_receiver_status.bin")

    assert station["Code"] == "ME-MO2"
    assert station["SC4 Prefix"] == "mx02"


def test_identify_station_returns_none_when_coordinates_are_too_far_away():
    assert identify_station(latitude=0, longitude=0) is None


def test_identify_station_rejects_ambiguous_coordinates():
    with pytest.raises(ValueError, match="multiple station entries"):
        identify_station(latitude=-7.212, longitude=-35.906)


def test_identify_station_requires_one_lookup_method():
    with pytest.raises(ValueError, match="provide filename or both"):
        identify_station(latitude=32.9919)
