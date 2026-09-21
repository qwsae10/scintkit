import pandas as pd

from scintkit.data import load_station_codes
from scintkit.email_updates import core


def test_shared_station_registry_is_loadable():
    stations = load_station_codes()

    assert "Station Location" in stations.columns
    assert "Code" in stations.columns
    assert len(stations) > 0
    septentrio = stations.loc[stations["Code"] == "US-TX4"].iloc[0]
    assert septentrio["Station Location"] == "Dallas Texas"
    assert septentrio["Latitude"] == 32.992
    assert septentrio["Longitude"] == -96.757
    assert septentrio["Type"] == "SEPT"


def test_load_targets_reads_sc4_prefix(tmp_path):
    csv_path = tmp_path / "stations.csv"
    csv_path.write_text(
        "Station Location,Latitude,Longitude,Code,Type,SC4 Prefix\n"
        "Morelia Mexico,19.649,-101.222,ME-MO1,SC4,mx01\n"
        "Oregon,43.271,-120.358,US-OR1,SC3,\n"
    )

    targets = core.load_targets(csv_path)
    by_code = {target["code"]: target for target in targets}

    assert by_code["ME-MO1"]["sc4_prefix"] == "mx01"
    assert by_code["US-OR1"]["sc4_prefix"] is None


def test_scan_sc4_uses_prefixes_from_targets(monkeypatch):
    targets = [
        {
            "code": "ME-MO1",
            "sc4_prefix": "mx01",
            "valid_times": set(),
        }
    ]
    monkeypatch.setattr(
        core.glob,
        "glob",
        lambda pattern: ["/data/site/20260701/mx01_receiver_"],
    )

    core.scan_sc4_files(
        targets,
        pd.Timestamp("2026-06-01"),
        base_dir="/data",
    )

    assert targets[0]["valid_times"] == {pd.Timestamp("2026-07-01")}


def test_scan_septentrio_files_parses_doy_and_ignores_invalid_files(monkeypatch):
    targets = [{"code": "US-TX4", "valid_times": set()}]
    paths = [
        "/sept/CSS_1832.26_",
        "/sept/CSS_1832.26_.ismr",
        "/sept/CSS_1841.26_.ismr",
        "/sept/CSS_0011.25_",
        "/sept/CSS_3671.26_",
        "/sept/CSS_183a.26_",
        "/sept/CSS_183_F.ismr",
    ]
    monkeypatch.setattr(core.glob, "glob", lambda pattern: paths)

    core.scan_septentrio_files(
        targets,
        pd.Timestamp("2026-07-02"),
        base_dir="/sept",
    )

    assert targets[0]["valid_times"] == {
        pd.Timestamp("2026-07-02"),
        pd.Timestamp("2026-07-03"),
    }
