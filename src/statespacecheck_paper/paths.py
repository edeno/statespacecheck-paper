"""Default paths and identifiers for the paper's real-data analysis.

These exist so the 5 figure / sanity scripts that consume the real-data
pickles don't each redeclare the same constants. Override
``DATA_PATH`` via the ``STATESPACECHECK_DATA_PATH`` environment
variable and ``ANIMAL_DATE_EPOCH`` via
``STATESPACECHECK_ANIMAL_DATE_EPOCH`` to run any of these scripts
against a different dataset without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH: Path = Path(os.environ.get("STATESPACECHECK_DATA_PATH", _REPO_ROOT / "data"))
ANIMAL_DATE_EPOCH: str = os.environ.get("STATESPACECHECK_ANIMAL_DATE_EPOCH", "j1620210710_02_r1")
