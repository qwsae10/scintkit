import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "scintkit"
    / "sc4_reading"
    / "sc4_s02_txtF2parquet.py"
)
SPEC = importlib.util.spec_from_file_location("sc4_s02_txtF2parquet", MODULE_PATH)
sc4_s02_txtF2parquet = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sc4_s02_txtF2parquet
SPEC.loader.exec_module(sc4_s02_txtF2parquet)

discover_binary_files = sc4_s02_txtF2parquet.discover_binary_files


def test_discover_binary_files_ignores_non_binary_and_dirs(tmp_path):
    (tmp_path / "file_a.bin").write_text("binary")
    (tmp_path / "file_b.BIN").write_text("binary")
    (tmp_path / "notes.txt").write_text("text")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "file_c.bin").write_text("binary")

    discovered = discover_binary_files(tmp_path, recursive=False)

    assert [p.name for p in discovered] == ["file_a.bin", "file_b.BIN"]
