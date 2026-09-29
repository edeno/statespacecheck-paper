"""Values the documentation states must match the code and artifacts that define them.

The guides quote settings readers need (the figure resolution, the flag
cutoffs, the Figure-4 decoder settings, the input file's identifiers, ...),
and each is defined once in code. Every row below names a document, a pattern
that captures the stated value, and the value the code gives; the test fails
when either side changes without the other.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from statespacecheck_paper.diagnostics import (
    BASELINE_HPD_OVERLAP_QUANTILE,
    BASELINE_KL_DIVERGENCE_QUANTILE,
    FIXED_PREDICTIVE_PVALUE_CUTOFF,
    HPD_COVERAGE,
)
from statespacecheck_paper.figure03_generation import FIGURE03_SUMMARY_SCHEMA_VERSION
from statespacecheck_paper.figure03_protocol import STEP_SECONDS, Figure3Config, PhaseBoundary
from statespacecheck_paper.figure03_summary import N_REALIZATIONS
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_generation import (
    FIGURE04_DETAIL_WINDOW,
    FIGURE04_DIAGNOSTIC_THRESHOLDS,
    FIGURE04_SUMMARY_SCHEMA_VERSION,
)
from statespacecheck_paper.figure04_input import FIGURE04_INPUTS_FILE
from statespacecheck_paper.number_format import significant
from statespacecheck_paper.paths import (
    CITATION_PATH,
    DATA_PATH,
    FIGURE03_SUMMARY_PATH,
    FIGURE04_INPUTS_DOI,
    FIGURE04_INPUTS_EPOCH,
    FIGURE04_INPUTS_SHA256,
    REPO_ROOT,
)
from statespacecheck_paper.reported_values import analysis_code_doi, cardinal_word, ordinal
from statespacecheck_paper.style import FIGURE_DPI

PIPELINE = "docs/figure-pipeline.md"
REPRODUCE = "docs/reproduce.md"
LINEAGE = "docs/data-lineage.md"

_FIGURE03 = Figure3Config()
_FIGURE03_SUMMARY = json.loads(FIGURE03_SUMMARY_PATH.read_text(encoding="utf-8"))
_DECODER = Figure4Config().decoder
_DETAIL_SECONDS = 2 * FIGURE04_DETAIL_WINDOW.half_width_samples / _DECODER.sampling_frequency_hz
_NODE_ENGINE = json.loads((REPO_ROOT / "site" / "package.json").read_text(encoding="utf-8"))[
    "engines"
]["node"]

# (document, pattern capturing the stated value, the value from code). Patterns
# allow a line break wherever the document wraps (``\s+``).
DOCUMENTED_VALUES: tuple[tuple[str, str, str], ...] = (
    # Figure resolution
    (PIPELINE, r"exported as PDF and (\d+)-DPI PNG", str(FIGURE_DPI)),
    ("docs/development.md", r"`style\.FIGURE_DPI` \((\d+)\)", str(FIGURE_DPI)),
    # HPD coverage
    (PIPELINE, r"coverage `diagnostics\.HPD_COVERAGE` \(([\d.]+),", repr(HPD_COVERAGE)),
    # Figure-3 flag rule
    (
        PIPELINE,
        r"at or below the (\w+) percentile",
        ordinal(round(100 * BASELINE_HPD_OVERLAP_QUANTILE)),
    ),
    (
        PIPELINE,
        r"at or above the (\w+) percentile",
        ordinal(round(100 * BASELINE_KL_DIVERGENCE_QUANTILE)),
    ),
    (
        PIPELINE,
        r"pooled over every spike event before step (\d+)",
        str(_FIGURE03.phase_boundaries[PhaseBoundary.REMAP_START]),
    ),
    (
        PIPELINE,
        r"threshold is exactly (\d+), so a spike",
        f"{_FIGURE03_SUMMARY['flag_rules']['hpd_overlap']['threshold']:g}",
    ),
    (
        PIPELINE,
        r"about ([\d.]+)% of\s+the Figure-3\s+baseline spikes",
        significant(
            100
            * _FIGURE03_SUMMARY["threshold_provenance"]["hpd_overlap"]["baseline_flagged_fraction"]
        ),
    ),
    (PIPELINE, r"flagged at or below a fixed ([\d.]+)", repr(FIXED_PREDICTIVE_PVALUE_CUTOFF)),
    (PIPELINE, r"\$\\Delta t=(\d+)\$ ms", f"{1000 * STEP_SECONDS:g}"),
    # Figure-3 realizations and seeds
    (PIPELINE, r"of all (\d+) realizations", str(N_REALIZATIONS)),
    (PIPELINE, r"`figure03_summary\.N_REALIZATIONS = (\d+)`", str(N_REALIZATIONS)),
    (PIPELINE, r"\(pools (\d+) realizations", str(N_REALIZATIONS)),
    (PIPELINE, r"n_realizations=(\d+)\)", str(N_REALIZATIONS)),
    (REPRODUCE, r"for its (\d+) realizations", str(N_REALIZATIONS)),
    (REPRODUCE, r"runs (\d+) realizations", str(N_REALIZATIONS)),
    (
        REPRODUCE,
        r"realizations \(seeds (\d+–\d+)\)",
        f"{_FIGURE03.random_seed}–{_FIGURE03.random_seed + N_REALIZATIONS - 1}",
    ),
    # Figure-4 fixed cutoffs
    (
        PIPELINE,
        r"Fixed cutoffs: (HPD overlap ≤ [\d.]+ and predictive \$p\$-value ≤ [\d.]+)",
        f"HPD overlap ≤ {FIGURE04_DIAGNOSTIC_THRESHOLDS['hpd_overlap']!r} and predictive "
        f"$p$-value ≤ {FIGURE04_DIAGNOSTIC_THRESHOLDS['predictive_pvalue']!r}",
    ),
    (
        PIPELINE,
        r"`FIGURE04_DIAGNOSTIC_THRESHOLDS = (\{[^}]*\})`",
        json.dumps(FIGURE04_DIAGNOSTIC_THRESHOLDS),
    ),
    # Figure-4 decoder settings
    (PIPELINE, r"\$\\sqrt\{([\d.]+)\}\\approx", f"{_DECODER.position_std**2:g}"),
    (PIPELINE, r"\\approx([\d.]+)\$\s+cm", significant(_DECODER.position_std, 3)),
    (PIPELINE, r"`position_bin_size_cm` \(([\d.]+) cm\)", f"{_DECODER.position_bin_size_cm:g}"),
    (PIPELINE, r"`sampling_frequency_hz` \((\d+) Hz", f"{_DECODER.sampling_frequency_hz:g}"),
    (PIPELINE, r"i\.e\. (\d+) ms bins", f"{1000 / _DECODER.sampling_frequency_hz:g}"),
    (
        PIPELINE,
        r"`movement_var`\s+\(([\d.]+) cm²\)",
        f"{Figure4Config().package_defaults.movement_var:g}",
    ),
    (LINEAGE, r"upsampled\s+linearly to (\d+) Hz", f"{_DECODER.sampling_frequency_hz:g}"),
    # Figure-4 detail window (a candidate replay event)
    (
        PIPELINE,
        r"Figure4DetailWindow\(center_index=(\d+),",
        str(FIGURE04_DETAIL_WINDOW.center_index),
    ),
    (
        PIPELINE,
        r"half_width_samples=(\d+)\)` in",
        str(FIGURE04_DETAIL_WINDOW.half_width_samples),
    ),
    (PIPELINE, r"spans about (\w+)\s+seconds total", cardinal_word(round(_DETAIL_SECONDS))),
    (PIPELINE, r"\(a, b\) a (\w+)-second window", cardinal_word(round(_DETAIL_SECONDS))),
    # Summary schema versions
    (
        PIPELINE,
        r"`figure03_summary\.json` uses schema version (\d+)",
        str(FIGURE03_SUMMARY_SCHEMA_VERSION),
    ),
    (
        PIPELINE,
        r"`figure04_summary\.json`\s+uses schema version (\d+)",
        str(FIGURE04_SUMMARY_SCHEMA_VERSION),
    ),
    # The Figure-4 input file and the code's archive
    (REPRODUCE, r"The canonical epoch is `(\w+)`", FIGURE04_INPUTS_EPOCH),
    (LINEAGE, r"one recording epoch, `(\w+)`", FIGURE04_INPUTS_EPOCH),
    (LINEAGE, r"SHA-256\s+`([0-9a-f]{64})` \(", FIGURE04_INPUTS_SHA256),
    ("README.md", r"The Figure 4 input is archived at\s+\[doi:([^\]]+)\]", FIGURE04_INPUTS_DOI),
    (REPRODUCE, r"The dataset is archived at\s+\[doi:([^\]]+)\]", FIGURE04_INPUTS_DOI),
    (PIPELINE, r"archived on Zenodo\s+\(\[([^\]]+)\]", FIGURE04_INPUTS_DOI),
    (LINEAGE, r"version 1\.0 of\s+record \[([^\]]+)\]", FIGURE04_INPUTS_DOI),
    (
        "README.md",
        r"code is archived on \[Zenodo\]\(https://doi\.org/([^)]+)\)",
        analysis_code_doi(CITATION_PATH),
    ),
    # The website's Node requirement
    ("README.md", r"requires Node (\d+)\+", _NODE_ENGINE.removeprefix(">=")),
    (REPRODUCE, r"Node (\d+)\+ for website tests", _NODE_ENGINE.removeprefix(">=")),
    ("site/README.md", r"Node (\d+)\+ \(`engines`", _NODE_ENGINE.removeprefix(">=")),
)


@pytest.mark.parametrize(
    ("document", "pattern", "expected"),
    DOCUMENTED_VALUES,
    ids=[f"{document}:{pattern[:40]}" for document, pattern, _ in DOCUMENTED_VALUES],
)
def test_documented_value_matches_the_code(document: str, pattern: str, expected: str) -> None:
    text = (REPO_ROOT / document).read_text(encoding="utf-8")
    # A stated value may wrap across lines; compare it with single spaces.
    stated = [" ".join(value.split()) for value in re.findall(pattern, text)]
    assert stated, f"{document} no longer states {pattern!r}; update this table"
    assert all(value == expected for value in stated), (document, stated, expected)


# The documents' stated sizes of the Figure-4 input file, in MB (10**6 bytes).
DOCUMENTED_INPUT_SIZES_MB: tuple[tuple[str, str], ...] = (
    ("README.md", r"verifies the ([\d.]+) MB Figure 4"),
    (REPRODUCE, r"\| ([\d.]+) MB download"),
    (LINEAGE, r"\(([\d.]+) MB; not in"),
)


@pytest.mark.parametrize(("document", "pattern"), DOCUMENTED_INPUT_SIZES_MB)
def test_documented_input_file_size(document: str, pattern: str) -> None:
    """The stated size, to the precision stated, is the downloaded file's."""
    stated = re.findall(pattern, (REPO_ROOT / document).read_text(encoding="utf-8"))
    assert stated, f"{document} no longer states {pattern!r}; update this table"
    path = DATA_PATH / FIGURE04_INPUTS_FILE
    if not path.exists():
        pytest.skip(
            "No code records the input file's size; checking it needs the file at "
            f"{path} (make download-data)"
        )
    size_mb = path.stat().st_size / 1e6
    for value in stated:
        decimals = len(value.partition(".")[2])
        assert math.isclose(round(size_mb, decimals), float(value)), (document, value, size_mb)


def test_documented_viewer_reset_width() -> None:
    viewer = pytest.importorskip("statespacecheck_paper.interactive.viewer")
    stated = re.findall(
        r"Reset to a (\d+) s context window",
        (REPO_ROOT / "docs" / "interactive.md").read_text(encoding="utf-8"),
    )
    assert stated == [f"{viewer.RESET_WINDOW_SECONDS:g}"]
