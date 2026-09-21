"""ScintKit package with lazily imported functional subpackages.

Keeping the top-level package lightweight prevents unrelated optional systems
such as plotting and email reporting from becoming requirements for data-only
workflows.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

_SUBMODULES = {
    "data",
    "email_updates",
    "machine_learning",
    "pipelines",
    "preprocessing",
    "reading",
    "services",
}
__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> ModuleType:
    """Import a public subpackage only when it is first requested."""

    if name not in _SUBMODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES})
