"""Contracts for the LaTeX macro file the manuscript inputs.

The committed ``manuscript/reported_values.tex`` is the last link in the chain
from code to prose: the figure summaries are pinned against the pipeline by
``test_reported_statistics_artifacts``, and this module pins the macro file
against those summaries. A regenerated summary that never reaches the macro
file fails here rather than silently leaving a stale number in the paper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from statespacecheck_paper.reported_values import (
    MACRO_FILE_PATH,
    _decimals_for_standard_error,
    _exact,
    _from_standard_error,
    _rescue_rate_standard_error,
    _significant,
    cardinal_word,
    ordinal,
    render_macro_file,
    write_macro_file,
)
from tests.test_reported_statistics_artifacts import _load

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MACRO_FILE = REPO_ROOT / MACRO_FILE_PATH


def _macro_values(text: str) -> dict[str, str]:
    """Parse ``\\newcommand`` definitions into a name -> value mapping."""
    return {name: value for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text)}


def test_committed_macro_file_matches_the_figure_summaries(tmp_path: Path) -> None:
    """The committed macro file is exactly what the summaries generate now."""
    regenerated = write_macro_file(
        tmp_path / "reported_values.tex",
        figure03_path=REPO_ROOT / "manuscript/figures/main/figure03_summary.json",
        figure04_path=REPO_ROOT / "manuscript/figures/main/figure04_summary.json",
    )
    assert regenerated.read_text(encoding="utf-8") == COMMITTED_MACRO_FILE.read_text(
        encoding="utf-8"
    ), (
        "manuscript/reported_values.tex is stale; regenerate it with "
        "`uv run python scripts/emit_reported_values.py`."
    )


def test_macro_values_round_trip_the_canonical_statistics() -> None:
    """Spot-check that the headline numbers carry the summaries' values."""
    values = _macro_values(COMMITTED_MACRO_FILE.read_text(encoding="utf-8"))
    figure03 = _load("figure03_summary.json")
    figure04 = _load("figure04_summary.json")

    remap = figure03["condition_order"].index("remap")
    remap_percentages = [row[remap] for row in figure03["median_flag_percentages"]]
    remap_errors = [row[remap] for row in figure03["median_flag_percentage_standard_errors"]]
    decimals = _decimals_for_standard_error(max(remap_errors))
    assert values["SimRemapFlagMin"] == f"{min(remap_percentages):.{decimals}f}"
    assert values["SimRemapFlagMax"] == f"{max(remap_percentages):.{decimals}f}"

    accuracy = figure03["median_decoding_accuracy"][0]
    accuracy_error = figure03["median_decoding_accuracy_standard_errors"][0]
    assert (
        values["SimRemapError"]
        == f"{accuracy[remap]:.{_decimals_for_standard_error(accuracy_error[remap])}f}"
    )
    assert values["SimNRealizations"] == str(figure03["realizations"]["count"])

    assert values["RecNUnits"] == str(figure04["dataset"]["n_units"])
    hpd = next(item for item in figure04["flag_confusions"] if item["metric"] == "hpd_overlap")
    assert values["RecHpdRescued"] == str(hpd["a_only"])
    assert values["RecHpdFlaggedContinuous"] == str(hpd["a_only"] + hpd["both"])
    hpd_decimals = _decimals_for_standard_error(_rescue_rate_standard_error(hpd))
    assert values["RecHpdRescuedPercent"] == f"{100 * hpd['rescue_rate']:.{hpd_decimals}f}"


def test_macro_names_are_unique() -> None:
    """A duplicated name would make ``\\newcommand`` abort the LaTeX build."""
    text = COMMITTED_MACRO_FILE.read_text(encoding="utf-8")
    names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
    assert len(names) == len(set(names))


def test_every_macro_is_used_by_the_manuscript() -> None:
    """An unused macro is a number nobody reports; drop it rather than ship it."""
    macro_text = COMMITTED_MACRO_FILE.read_text(encoding="utf-8")
    manuscript = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    unused = [
        name
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", macro_text)
        if not re.search(rf"\\{name}(?![A-Za-z])", manuscript)
    ]
    assert unused == []


def test_exact_rejects_precision_loss() -> None:
    """``_exact`` is the guard against a config change quietly changing a digit."""
    assert _exact(0.88, 2) == "0.88"
    # Floating-point dust from a derived quantity must still pass.
    assert _exact(12.500000000000002, 1) == "12.5"
    with pytest.raises(ValueError, match="not exact"):
        _exact(0.875, 2)


@pytest.mark.parametrize(
    ("value", "digits", "expected"),
    [
        # The manuscript's "approximately 200 Hz": 199 to zero decimals, but
        # 200 to two significant figures.
        (199.47114020071638, 2, "200"),
        (0.19947114020071638, 2, "0.20"),
        (3.5355339059327378, 3, "3.54"),
    ],
)
def test_significant_figures_follow_the_hedged_claims(
    value: float, digits: int, expected: str
) -> None:
    assert _significant(value, digits) == expected


@pytest.mark.parametrize(
    ("standard_error", "expected_decimals"),
    [(4.14, 0), (2.55, 0), (0.155, 1), (0.056, 2), (0.00662, 3)],
)
def test_precision_follows_the_standard_error(
    standard_error: float, expected_decimals: int
) -> None:
    """The digit count is the decimal place of the SE at one significant figure."""
    assert _decimals_for_standard_error(standard_error) == expected_decimals


def test_estimates_print_at_their_own_precision() -> None:
    """Two estimates with different errors must not print to the same width."""
    # Remap flag percentage: SE of ~4 points earns whole percents.
    assert _from_standard_error(40.799, 4.14) == "41"
    # History-dependence flag percentage: SE of ~0.06 earns two decimals.
    assert _from_standard_error(1.764, 0.056) == "1.76"


def test_zero_standard_error_is_rejected() -> None:
    """A zero error carries no precision information; the caller must decide."""
    with pytest.raises(ValueError, match="must be positive"):
        _decimals_for_standard_error(0.0)


def test_rescue_rate_standard_error_is_binomial() -> None:
    confusion = {"a_only": 17289, "both": 1501, "rescue_rate": 0.9201170835550825}
    assert _rescue_rate_standard_error(confusion) == pytest.approx(0.198, abs=5e-4)


def test_word_helpers() -> None:
    assert cardinal_word(11) == "eleven"
    assert ordinal(1) == "1st"
    assert ordinal(99) == "99th"
    with pytest.raises(ValueError, match="cardinal_word covers"):
        cardinal_word(21)


def test_render_is_deterministic() -> None:
    """Two renders of the same payloads agree byte for byte."""
    figure03 = _load("figure03_summary.json")
    figure04 = _load("figure04_summary.json")
    assert render_macro_file(figure03, figure04) == render_macro_file(figure03, figure04)
