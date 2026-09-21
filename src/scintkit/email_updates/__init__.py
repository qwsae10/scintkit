"""ScintPi email reporting: availability scanning, plotting, and emailing."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_SUBMODULES = {"core", "mailer", "plotting"}
_EXPORTS = {
    "load_targets": ("core", "load_targets"),
    "scan_legacy_files": ("core", "scan_legacy_files"),
    "scan_sc4_files": ("core", "scan_sc4_files"),
    "checklvl3datamissing": ("core", "checklvl3datamissing"),
    "generate_availability_plot": ("plotting", "generate_availability_plot"),
    "send_status_email": ("mailer", "send_status_email"),
}

__all__ = [
    "checklvl3datamissing",
    "core",
    "generate_availability_plot",
    "load_targets",
    "mailer",
    "plotting",
    "scan_legacy_files",
    "scan_sc4_files",
    "send_status_email",
]


def __getattr__(name: str) -> Any:
    """Load core, plotting, or mail functionality only when requested."""

    if name in _SUBMODULES:
        value: ModuleType | Any = import_module(f"{__name__}.{name}")
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
