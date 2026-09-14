"""Calibration-role metadata and compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


FIXED_BATCH_DECODE = "fixed_batch_decode"
ONLINE_COLOCATED = "online_continuous_batch_colocated"
ONLINE_PD = "online_continuous_batch_pd"


def read_calibration_role(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    role = metadata.get("calibration_role")
    return str(role) if role else None


def require_calibration_role(path: Optional[Path], expected: str) -> str:
    """Reject a known-incompatible calibration while accepting legacy files."""
    role = read_calibration_role(path)
    if role is not None and role != expected:
        raise ValueError(
            f"calibration role mismatch: expected {expected!r}, received "
            f"{role!r} from {path}")
    return role or "legacy_unspecified"
