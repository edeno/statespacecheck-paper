"""Default paths and identifiers for the paper's real-data analysis.

These exist so the modules and scripts that read the Figure-4 input file
(``{animal_date_epoch}_figure04_inputs.npz``) don't each redeclare the same
constants. Override ``DATA_PATH`` via the ``STATESPACECHECK_DATA_PATH``
environment variable and ``ANIMAL_DATE_EPOCH`` via
``STATESPACECHECK_ANIMAL_DATE_EPOCH`` to run them against a different dataset
without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The published Figure-4 input: the one epoch whose input file is archived on
# Zenodo, as version 1.0 of the record this DOI names (the manuscript's data
# statement cites it), and that file's SHA-256, which the Figure-4 summary also
# records. Kept here, with no heavy imports, so the manuscript-macro emitter can
# read the DOI without loading the data stack.
FIGURE04_INPUTS_EPOCH = "j1620210710_02_r1"
FIGURE04_INPUTS_DOI = "10.5281/zenodo.23020757"
FIGURE04_INPUTS_SHA256 = "60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3"

DATA_PATH: Path = Path(os.environ.get("STATESPACECHECK_DATA_PATH", _REPO_ROOT / "data"))
ANIMAL_DATE_EPOCH: str = os.environ.get("STATESPACECHECK_ANIMAL_DATE_EPOCH", FIGURE04_INPUTS_EPOCH)
