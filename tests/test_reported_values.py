"""Contracts for the LaTeX macro file the manuscript inputs.

The committed ``manuscript/reported_values.tex`` is the last link in the chain
from code to prose: ``test_reported_statistics_artifacts`` checks the summaries'
reference values, configuration, and provenance, and this module pins the macro
file against those summaries. A regenerated summary that never reaches the macro
file fails here rather than silently leaving a stale number in the paper.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from statespacecheck_paper.number_format import SIGNIFICANT_FIGURES, significant, whole_percent
from statespacecheck_paper.reported_values import (
    MACRO_FILE_PATH,
    _exact,
    cardinal_word,
    ordinal,
    render_macro_file,
    write_macro_file,
)
from tests.test_reported_statistics_artifacts import _load, _load_supplementary

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_MACRO_FILE = REPO_ROOT / MACRO_FILE_PATH


def _render(figure03: dict, figure04: dict) -> str:
    """Render the macro file with the committed Figure-4 supplement summary."""
    return render_macro_file(
        figure03, figure04, _load_supplementary("figure04_supplement_summary.json")
    )


def _macro_values(text: str) -> dict[str, str]:
    """Parse ``\\newcommand`` definitions into a name -> value mapping."""
    return {name: value for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text)}


def test_committed_macro_file_matches_the_figure_summaries(tmp_path: Path) -> None:
    """The committed macro file is exactly what the summaries generate now."""
    regenerated = write_macro_file(
        tmp_path / "reported_values.tex",
        figure03_path=REPO_ROOT / "manuscript/figures/main/figure03_summary.json",
        figure04_path=REPO_ROOT / "manuscript/figures/main/figure04_summary.json",
        figure04_supplement_path=REPO_ROOT
        / "manuscript/figures/supplementary/figure04_supplement_summary.json",
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
    assert values["SimRemapFlagMin"] == f"{min(remap_percentages):.0f}"
    assert values["SimRemapFlagMax"] == f"{max(remap_percentages):.0f}"

    accuracy = figure03["median_decoding_accuracy"][0]
    assert values["SimRemapError"] == significant(accuracy[remap], 2)
    assert values["SimNRealizations"] == str(figure03["realizations"]["count"])

    assert values["RecNUnits"] == str(figure04["dataset"]["n_units"])
    hpd = next(item for item in figure04["flag_confusions"] if item["metric"] == "hpd_overlap")
    assert values["RecHpdRescued"] == str(hpd["a_only"])
    assert values["RecHpdFlaggedContinuous"] == str(hpd["a_only"] + hpd["both"])
    assert values["RecHpdRescuedPercent"] == f"{100 * hpd['rescue_rate']:.0f}"


def test_supplement_macros_carry_the_broadening_and_rate_statistics() -> None:
    """The supplement's quoted numbers match its summary and the fitted rescue rate."""
    values = _macro_values(COMMITTED_MACRO_FILE.read_text(encoding="utf-8"))
    supplement = _load_supplementary("figure04_supplement_summary.json")
    main = supplement["region_size_and_broadening"]["by_coverage"]["0.95"]
    fitted_rescue = next(
        item["rescue_rate"]
        for item in _load("figure04_summary.json")["flag_confusions"]
        if item["metric"] == "hpd_overlap"
    )

    rescued = main["continuous_fragmented"]["region_size_rescued_events"]
    assert values["RecFragRegionRescued"] == _exact(
        rescued["quantiles"][rescued["quantile_probabilities"].index(0.5)]
    )
    # "Match" is the smallest weight removing at least the fitted model's share
    # of flags, so the text's comparison holds by construction.
    mixtures = {
        record["uniform_weight"]: record["fraction_original_flags_removed"]
        for record in main["uniform_mixture"].values()
        if record["uniform_weight"] < 1.0
    }
    match = float(values["RecMixWeightMatch"])
    assert mixtures[match] >= fitted_rescue
    assert all(removed < fitted_rescue for weight, removed in mixtures.items() if weight < match)
    assert float(values["RecMixWeightMin"]) == min(mixtures)
    assert values["RecMixRemovedMatch"] == whole_percent(100.0 * mixtures[match])
    immobile = supplement["rate_and_behavior"]["behavior"]["immobile"]
    assert values["RecImmobileHpdCont"] == significant(
        100.0 * immobile["continuous_hpd_overlap_flag_fraction"], SIGNIFICANT_FIGURES
    )


def test_asymmetric_mode_parameters_are_reported_independently() -> None:
    """Each mode keeps its own initial and transition probability."""
    figure03 = _load("figure03_summary.json")
    figure04 = copy.deepcopy(_load("figure04_summary.json"))
    provenance = figure04["configuration"]["provenance"]
    provenance["contfrag_discrete_initial_conditions"] = [0.6, 0.4]
    provenance["contfrag_diagonal_values"] = [0.9, 0.8]

    values = _macro_values(_render(figure03, figure04))

    assert values["RecModeContinuousInitial"] == "0.6"
    assert values["RecModeFragmentedInitial"] == "0.4"
    assert values["RecModeContinuousStay"] == "0.90"
    assert values["RecModeContinuousToFragmented"] == "0.10"
    assert values["RecModeFragmentedToContinuous"] == "0.20"
    assert values["RecModeFragmentedStay"] == "0.80"


def test_non_integral_burst_factor_is_not_silently_rounded() -> None:
    """The spelled-out prose cannot faithfully represent a fractional factor."""
    figure03 = copy.deepcopy(_load("figure03_summary.json"))
    figure03["configuration"]["history_burst_factor"] = 3.4

    with pytest.raises(ValueError, match="not exact"):
        _render(figure03, _load("figure04_summary.json"))


@pytest.mark.parametrize("quantile", [0.005, 0.995])
def test_fractional_percentile_is_not_silently_rounded(quantile: float) -> None:
    """0.005 must not print as "0th": the ordinal prose has no form for it."""
    figure03 = copy.deepcopy(_load("figure03_summary.json"))
    figure03["threshold_provenance"]["hpd_overlap"]["quantile"] = quantile

    with pytest.raises(ValueError, match="not exact"):
        _render(figure03, _load("figure04_summary.json"))


def test_zero_decoding_error_renders() -> None:
    """A perfectly decoded phase must not abort the emit."""
    figure03 = copy.deepcopy(_load("figure03_summary.json"))
    well_specified = figure03["condition_order"].index("well_specified")
    figure03["median_decoding_accuracy"][0][well_specified] = 0.0

    values = _macro_values(_render(figure03, _load("figure04_summary.json")))

    assert values["SimWellSpecifiedError"] == "0"


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
        # Rounding across a power of ten must not add a significant figure.
        (0.999, 2, "1.0"),
        (9.99, 2, "10"),
        # Exactly zero is a valid decoding error and prints as "0".
        (0.0, 2, "0"),
    ],
)
def test_significant_figures_follow_the_hedged_claims(
    value: float, digits: int, expected: str
) -> None:
    assert significant(value, digits) == expected


def test_significant_agrees_across_python_and_numpy_scalars() -> None:
    """The emitter passes floats and the figure passes NumPy scalars.

    Python and NumPy round half-way binary values differently (2.45 gives 2.5
    and 2.4 respectively), which once let the prose and Figure 3b disagree.
    """
    assert significant(2.45, 2) == "2.5"
    assert significant(np.float64(2.45), 2) == "2.5"
    assert significant(np.float32(2.45), 2) == "2.5"
    assert whole_percent(np.float64(36.5)) == whole_percent(36.5)


def test_whole_percent_rounds_to_the_nearest_percent() -> None:
    """Flag and rescue percentages report to whole percents by policy."""
    assert whole_percent(36.77115625352582) == "37"
    assert whole_percent(0.6675931668463562) == "1"
    assert whole_percent(27.60499499322613) == "28"


def test_published_standard_errors_do_not_set_precision() -> None:
    """The Figure-3 SEs are data for the reader, not a formatting authority."""
    figure03 = copy.deepcopy(_load("figure03_summary.json"))
    figure04 = _load("figure04_summary.json")
    baseline = _macro_values(_render(figure03, figure04))
    # Shrink every published SE a thousandfold; no printed digit may change.
    for key in (
        "median_flag_percentage_standard_errors",
        "median_decoding_accuracy_standard_errors",
    ):
        figure03[key] = [[value / 1000.0 for value in row] for row in figure03[key]]
    assert _macro_values(_render(figure03, figure04)) == baseline


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
    assert _render(figure03, figure04) == _render(figure03, figure04)
