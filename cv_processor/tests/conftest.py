"""Test configuration for cv_processor."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "cv_processor" / "src"
COMMON_SRC = PROJECT_ROOT / "sundew_common" / "src"

for path in (SRC_ROOT, COMMON_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
