"""Default paths and identifiers for the paper's analysis artifacts.

These exist so the package modules and scripts that read or write the Figure-4
input file (``{animal_date_epoch}_figure04_inputs.npz``), the figures and their
summaries, the manuscript's macro file, or the website data don't each
redeclare the same constants. Every repository path is anchored at
:data:`REPO_ROOT`, so the recipes read and write the same files from any
working directory. (``scripts/spyglass_pipeline_figure04.py`` anchors its own
summary path to the repository root, since it imports no package module at
startup.) Override ``DATA_PATH`` via the ``STATESPACECHECK_DATA_PATH``
environment variable and ``ANIMAL_DATE_EPOCH`` via
``STATESPACECHECK_ANIMAL_DATE_EPOCH`` to run them against a different dataset
without editing source.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Main-figure directory: the figure recipes save the figures and their
# summaries here, and the macro emitter and website export read the summaries.
FIGURE_DIR = REPO_ROOT / "manuscript" / "figures" / "main"
FIGURE03_SUMMARY_PATH = FIGURE_DIR / "figure03_summary.json"
FIGURE04_SUMMARY_PATH = FIGURE_DIR / "figure04_summary.json"

# The manuscript's generated reported-value macros (``reported_values``), and
# this repository's citation metadata, whose ``doi`` the macros cite.
MACRO_FILE_PATH = REPO_ROOT / "manuscript" / "reported_values.tex"
CITATION_PATH = REPO_ROOT / "CITATION.cff"

# The website's exported data (``site_export``) and the reference cases that
# check the JavaScript port of the diagnostics.
SITE_DATA_DIR = REPO_ROOT / "site" / "data"
PARITY_FIXTURE_PATH = REPO_ROOT / "site" / "tests" / "fixtures" / "metric_parity.json"

# The published Figure-4 input: the one epoch whose input file is archived on
# Zenodo, as version 1.0 of the record this DOI names (the manuscript's data
# statement cites it), and that file's SHA-256, which the Figure-4 summary also
# records. Kept here, with no heavy imports, so the manuscript-macro emitter can
# read the DOI without loading the data stack.
FIGURE04_INPUTS_EPOCH = "j1620210710_02_r1"
FIGURE04_INPUTS_DOI = "10.5281/zenodo.23020757"
FIGURE04_INPUTS_SHA256 = "60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3"

DATA_PATH: Path = Path(os.environ.get("STATESPACECHECK_DATA_PATH", REPO_ROOT / "data"))
ANIMAL_DATE_EPOCH: str = os.environ.get("STATESPACECHECK_ANIMAL_DATE_EPOCH", FIGURE04_INPUTS_EPOCH)
