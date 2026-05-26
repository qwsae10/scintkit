# src/scintkit/email/__init__.py
"""ScintPi email reporting: availability scanning, plotting, and emailing."""

from .core import load_targets, scan_legacy_files, scan_sc4_files, checklvl3datamissing
from .plotting import generate_availability_plot
from .mailer import send_status_email

__all__ = [
    "load_targets",
    "scan_legacy_files",
    "scan_sc4_files",
    "checklvl3datamissing",
    "generate_availability_plot",
    "send_status_email",
]
