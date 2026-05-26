"""Email update utilities for ScintKit."""

from .emailer import (
    load_targets,
    scan_legacy_files,
    scan_sc4_files,
    generate_availability_plot,
    send_status_email,
)

__all__ = [
    "load_targets",
    "scan_legacy_files",
    "scan_sc4_files",
    "generate_availability_plot",
    "send_status_email",
]