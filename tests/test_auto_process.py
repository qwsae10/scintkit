import shutil
from pathlib import Path

import pandas as pd

from scintkit.pipelines.auto import process

FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "scintpi3_20241011_1200_96.7573W_32.9919N_v326f_lvl0.pq"
)

PRODUCT_COLUMNS = [
    "tec_cph12",
    "tec_rng12",
    *(
        f"{name}_{i}"
        for i in ("1", "2")
        for name in (
            "sigma_phi",
            "n_sigphi",
            "n_cycleslip",
            "s4",
            "s4_detrended",
            "s4_corrected",
            "tau",
            "n_s4",
            "sigma_phi_quality_flag",
            "s4_quality_flag",
        )
    ),
]


def test_process_writes_lvl2_and_lvl3_products(tmp_path):
    lvl0 = tmp_path / FIXTURE.name
    shutil.copy(FIXTURE, lvl0)

    outputs = process(lvl0, mode="both", fs=None)

    lvl2 = tmp_path / FIXTURE.name.replace("_lvl0", "_lvl2")
    lvl3 = tmp_path / FIXTURE.name.replace("_lvl0", "_lvl3")
    assert sorted(outputs) == sorted([str(lvl2), str(lvl3)])

    per_second = pd.read_parquet(lvl2)
    per_minute = pd.read_parquet(lvl3)
    for frame in (per_second, per_minute):
        assert not frame.empty
        assert set(PRODUCT_COLUMNS) <= set(frame.columns)

    assert not per_minute.duplicated(["minbin", "prn"]).any()
    assert not per_second.duplicated(["secbin", "prn"]).any()
    assert per_minute["s4_1"].notna().any()
    assert per_minute["sigma_phi_1"].notna().any()
