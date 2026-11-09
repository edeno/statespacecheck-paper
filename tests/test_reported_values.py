"""Contracts for the LaTeX macro file the manuscript inputs.

The committed ``manuscript/reported_values.tex`` is the last link in the chain
from code to prose: ``test_reported_statistics_artifacts`` checks the summaries'
reference values, configuration, and provenance, and this module pins the macro
file against those summaries. A regenerated summary that never reaches the macro
file fails here rather than silently leaving a stale number in the paper.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.figure03_protocol import Figure3Config, compute_replay_step_window
from statespacecheck_paper.number_format import significant, whole_percent
from statespacecheck_paper.paths import (
    CITATION_PATH,
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    MACRO_FILE_PATH,
    REPO_ROOT,
    SOFTWARE_DOIS_PATH,
)
from statespacecheck_paper.reported_values import (
    REFRESH_DOIS_COMMAND,
    _exact,
    _recording_statistics,
    _simulation_configuration,
    _simulation_statistics,
    analysis_code_doi,
    cardinal_word,
    doi_from_zenodo_search,
    load_summary,
    lookup_statespacecheck_doi,
    ordinal,
    recorded_statespacecheck_doi,
    refresh_statespacecheck_doi,
    render_macro_file,
    require_matching_source_provenance,
    statespacecheck_version,
    write_macro_file,
)
from tests._scripts import load_script


def _macro_values(text: str) -> dict[str, str]:
    """Parse ``\\newcommand`` definitions into a name -> value mapping."""
    return {name: value for name, value in re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", text)}


# The DOI the committed macro file cites. Emitting reads it from the committed
# manuscript/software_dois.json (test_committed_doi_file_gives_the_cited_doi),
# and the network test test_recorded_doi_is_zenodos_record checks that file
# against Zenodo
CITED_DOI = _macro_values(MACRO_FILE_PATH.read_text(encoding="utf-8"))["StatespacecheckDOI"]
CODE_DOI = analysis_code_doi(CITATION_PATH)


def test_committed_macro_file_matches_the_figure_summaries(tmp_path: Path) -> None:
    """The committed macro file is exactly what the summaries generate now."""
    regenerated = write_macro_file(tmp_path / "reported_values.tex")
    assert regenerated.read_text(encoding="utf-8") == MACRO_FILE_PATH.read_text(encoding="utf-8"), (
        "manuscript/reported_values.tex is stale; regenerate it with "
        "`uv run python scripts/emit_reported_values.py`."
    )


@pytest.mark.parametrize(
    ("recovery_two", "start_fraction", "end_fraction"),
    [
        (None, None, None),
        # 10 * 0.25 and 10 * 0.75 tie at .5, where rounding before or after adding
        # the odd phase start gives different steps.
        ((14001, 14011), 0.25, 0.75),
    ],
)
def test_replay_macros_name_the_simulated_replay_window(
    recovery_two: tuple[int, int] | None,
    start_fraction: float | None,
    end_fraction: float | None,
) -> None:
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    config = Figure3Config()
    if recovery_two is not None:
        assert start_fraction is not None and end_fraction is not None
        boundaries = list(config.phase_boundaries)
        boundaries[3:5] = recovery_two
        config = dataclasses.replace(
            config,
            phase_boundaries=tuple(boundaries),
            replay_start_fraction=start_fraction,
            replay_end_fraction=end_fraction,
        )
        figure03["configuration"]["phase_boundaries"] = boundaries
        figure03["configuration"]["replay_start_fraction"] = start_fraction
        figure03["configuration"]["replay_end_fraction"] = end_fraction
    macros = {macro.name: macro.value for macro in _simulation_configuration(figure03)}
    assert (int(macros["SimReplayStart"]), int(macros["SimReplayEnd"])) == (
        compute_replay_step_window(config)
    )


def test_macro_values_round_trip_the_canonical_statistics() -> None:
    """Spot-check that the headline numbers carry the summaries' values."""
    values = _macro_values(MACRO_FILE_PATH.read_text(encoding="utf-8"))
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)

    remap = figure03["condition_order"].index("remap")
    hpd = figure03["metric_order"].index("hpd_overlap")
    remap_hpd = [
        realization[hpd][remap] for realization in figure03["realization_flag_percentages"]
    ]
    assert values["SimFlagRemapHpd"] == f"{figure03['median_flag_percentages'][hpd][remap]:.0f}"
    assert values["SimFlagRemapHpdQuartileLow"] == f"{np.percentile(remap_hpd, 25):.0f}"
    assert values["SimFlagRemapHpdQuartileHigh"] == f"{np.percentile(remap_hpd, 75):.0f}"
    # Below 10% the flag percentages keep one decimal place.
    history = figure03["condition_order"].index("history_dependent")
    assert values["SimFlagHistoryHpd"] == f"{figure03['median_flag_percentages'][hpd][history]:.1f}"

    decoding_error = figure03["median_decoding_error"]
    assert values["SimRemapError"] == significant(decoding_error[remap], 2)
    assert values["SimNRealizations"] == str(figure03["realizations"]["count"])

    assert values["RecNUnits"] == str(figure04["dataset"]["n_units"])
    assert values["RecNUnitsWithSpikes"] == str(figure04["dataset"]["n_units_with_spikes"])
    hpd = next(item for item in figure04["flag_confusions"] if item["metric"] == "hpd_overlap")
    assert values["RecHpdRescued"] == str(hpd["rescued"])
    assert values["RecHpdFlaggedContinuous"] == str(hpd["rescued"] + hpd["both"])
    assert values["RecHpdRescuedPercent"] == f"{100 * hpd['rescued_fraction']:.0f}"


_SummaryEdit = Callable[[dict[str, Any], dict[str, Any]], None]


def _set_flags(condition: str, *metrics: str) -> _SummaryEdit:
    """Edit the Figure-3 flag percentage of ``condition`` for ``metrics``."""

    def edit(figure03: dict[str, Any], _: dict[str, Any]) -> None:
        column = figure03["condition_order"].index(condition)
        for metric in metrics:
            figure03["median_flag_percentages"][figure03["metric_order"].index(metric)][column] = (
                77.0
            )

    return edit


def _spread_realization_flags(condition: str, *metrics: str) -> _SummaryEdit:
    """Replace the per-realization flag percentages of ``condition`` with 0..99."""

    def edit(figure03: dict[str, Any], _: dict[str, Any]) -> None:
        column = figure03["condition_order"].index(condition)
        rows = [figure03["metric_order"].index(metric) for metric in metrics]
        for i, realization in enumerate(figure03["realization_flag_percentages"]):
            for row in rows:
                realization[row][column] = float(i % 100)

    return edit


def _spread_realization_errors(condition: str) -> _SummaryEdit:
    """Replace the per-realization decoding errors of ``condition`` with 100..397.

    The range lies above every window's real error, so both quartiles move.
    """

    def edit(figure03: dict[str, Any], _: dict[str, Any]) -> None:
        column = figure03["condition_order"].index(condition)
        for i, realization in enumerate(figure03["realization_decoding_error"]):
            realization[column] = 100.0 + 3.0 * (i % 100)

    return edit


def _set_error(condition: str) -> _SummaryEdit:
    """Edit the Figure-3 median decoding error of ``condition``."""

    def edit(figure03: dict[str, Any], _: dict[str, Any]) -> None:
        figure03["median_decoding_error"][figure03["condition_order"].index(condition)] = 0.37

    return edit


def _set_figure03(path: tuple[str, ...], value: float) -> _SummaryEdit:
    """Edit one nested Figure-3 summary field."""

    def edit(figure03: dict[str, Any], _: dict[str, Any]) -> None:
        target = figure03
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _set_figure04(path: tuple[str | int, ...], value: float) -> _SummaryEdit:
    """Edit one nested Figure-4 summary field."""

    def edit(_: dict[str, Any], figure04: dict[str, Any]) -> None:
        target: Any = figure04
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _set_confusion(metric: str, field: str, value: float) -> _SummaryEdit:
    """Edit one count of the Figure-4 flag confusion of ``metric``."""

    def edit(_: dict[str, Any], figure04: dict[str, Any]) -> None:
        confusion = next(c for c in figure04["flag_confusions"] if c["metric"] == metric)
        confusion[field] = value

    return edit


def _set_event_count(_: dict[str, Any], figure04: dict[str, Any]) -> None:
    """Edit the spike-event count, which every flag confusion must agree on."""
    for confusion in figure04["flag_confusions"]:
        confusion["n"] = 1_234_567


_ALL_METRICS = ("hpd_overlap", "predictive_pvalue", "kl_divergence")
_ALL_METRIC_NAMES = ("Hpd", "Pvalue", "Kl")
_QUARTILES = ("QuartileLow", "QuartileHigh")
_MEDIAN_AND_QUARTILES = ("", *_QUARTILES)
_WINDOWS = {
    "well_specified": "WellSpecified",
    "remap": "Remap",
    "history_dependent": "History",
    "replay": "Replay",
    "momentum": "Momentum",
    "sparse_population": "Sparse",
}


def _flag_macros(window: str, metrics: tuple[str, ...], parts: tuple[str, ...]) -> set[str]:
    """Name the per-metric flag macros of one window."""
    return {f"SimFlag{window}{metric}{part}" for metric in metrics for part in parts}


# The mapping the prose depends on, written out independently of the emitter:
# one summary field, and the exact statistic macros that field alone may move.
# An empty set is a field no statistic macro may read.
_SUMMARY_FIELD_MACROS: dict[str, tuple[_SummaryEdit, set[str]]] = {
    "realization_count": (_set_figure03(("realizations", "count"), 7), {"SimNRealizations"}),
    "well_specified_flags": (
        _set_flags("well_specified", *_ALL_METRICS),
        _flag_macros("WellSpecified", _ALL_METRIC_NAMES, _MEDIAN_AND_QUARTILES),
    ),
    "remap_flags": (
        _set_flags("remap", *_ALL_METRICS),
        _flag_macros("Remap", _ALL_METRIC_NAMES, ("",)),
    ),
    "history_flags": (
        _set_flags("history_dependent", *_ALL_METRICS),
        _flag_macros("History", _ALL_METRIC_NAMES, _MEDIAN_AND_QUARTILES),
    ),
    "replay_flags": (
        _set_flags("replay", *_ALL_METRICS),
        _flag_macros("Replay", _ALL_METRIC_NAMES, _MEDIAN_AND_QUARTILES),
    ),
    # HPD overlap and the p-value already print to whole percents here; KL
    # crosses 10%, so its quartiles change precision too.
    "momentum_flags": (
        _set_flags("momentum", *_ALL_METRICS),
        _flag_macros("Momentum", _ALL_METRIC_NAMES, ("",))
        | _flag_macros("Momentum", ("Kl",), _QUARTILES),
    ),
    "sparse_kl_flag": (
        _set_flags("sparse_population", "kl_divergence"),
        _flag_macros("Sparse", ("Kl",), ("",)),
    ),
    "sparse_hpd_flag": (
        _set_flags("sparse_population", "hpd_overlap"),
        _flag_macros("Sparse", ("Hpd",), _MEDIAN_AND_QUARTILES),
    ),
    "sparse_pvalue_flag": (
        _set_flags("sparse_population", "predictive_pvalue"),
        _flag_macros("Sparse", ("Pvalue",), _MEDIAN_AND_QUARTILES),
    ),
    **{
        f"{condition}_realization_flags": (
            _spread_realization_flags(condition, *_ALL_METRICS),
            _flag_macros(window, _ALL_METRIC_NAMES, _QUARTILES),
        )
        for condition, window in _WINDOWS.items()
    },
    **{
        f"{condition}_realization_errors": (
            _spread_realization_errors(condition),
            {f"Sim{window}ErrorQuartileLow", f"Sim{window}ErrorQuartileHigh"},
        )
        for condition, window in _WINDOWS.items()
    },
    "well_specified_error": (_set_error("well_specified"), {"SimWellSpecifiedError"}),
    "remap_error": (_set_error("remap"), {"SimRemapError"}),
    "history_error": (_set_error("history_dependent"), {"SimHistoryError"}),
    "replay_error": (_set_error("replay"), {"SimReplayError"}),
    "momentum_error": (_set_error("momentum"), {"SimMomentumError"}),
    "sparse_error": (_set_error("sparse_population"), {"SimSparseError"}),
    "hpd_quantile": (
        _set_figure03(("threshold_provenance", "hpd_overlap", "quantile"), 0.02),
        {"SimHpdPercentile"},
    ),
    "kl_quantile": (
        _set_figure03(("threshold_provenance", "kl_divergence", "quantile"), 0.98),
        {"SimKlPercentile"},
    ),
    "hpd_baseline_flagged": (
        _set_figure03(("threshold_provenance", "hpd_overlap", "baseline_flagged_fraction"), 0.37),
        {"SimHpdBaselineFlagPercent"},
    ),
    "predictive_cutoff": (
        _set_figure03(("threshold_provenance", "predictive_pvalue", "cutoff"), 0.1),
        {"SimPredictiveCutoff", "SimPredictiveCutoffNegLog"},
    ),
    "baseline_end": (
        _set_figure03(("threshold_provenance", "baseline_end_index"), 1234),
        {"SimBaselineEnd"},
    ),
    "n_units": (_set_figure04(("dataset", "n_units"), 7), {"RecNUnits"}),
    "n_units_with_spikes": (
        _set_figure04(("dataset", "n_units_with_spikes"), 7),
        {"RecNUnitsWithSpikes"},
    ),
    "n_events": (
        _set_event_count,
        {
            "RecNEvents",
            "RecHpdFlagRateContinuous",
            "RecHpdFlagRateFragmented",
            "RecPvalueFlagRateContinuous",
            "RecPvalueFlagRateFragmented",
        },
    ),
    "hpd_rescued": (
        _set_confusion("hpd_overlap", "rescued", 12_345),
        {"RecHpdRescued", "RecHpdFlaggedContinuous", "RecHpdFlagRateContinuous"},
    ),
    "hpd_both": (
        _set_confusion("hpd_overlap", "both", 12_345),
        {"RecHpdFlaggedContinuous", "RecHpdFlagRateContinuous", "RecHpdFlagRateFragmented"},
    ),
    "hpd_rescued_fraction": (
        _set_confusion("hpd_overlap", "rescued_fraction", 0.37),
        {"RecHpdRescuedPercent"},
    ),
    "hpd_newly_flagged": (
        _set_confusion("hpd_overlap", "newly_flagged", 12_345),
        {"RecHpdNewlyFlagged", "RecHpdFlagRateFragmented"},
    ),
    "pvalue_rescued": (
        _set_confusion("predictive_pvalue", "rescued", 12_345),
        {"RecPvalueRescued", "RecPvalueFlaggedContinuous", "RecPvalueFlagRateContinuous"},
    ),
    "pvalue_both": (
        _set_confusion("predictive_pvalue", "both", 12_345),
        {
            "RecPvalueFlaggedContinuous",
            "RecPvalueFlagRateContinuous",
            "RecPvalueFlagRateFragmented",
        },
    ),
    "pvalue_rescued_fraction": (
        _set_confusion("predictive_pvalue", "rescued_fraction", 0.37),
        {"RecPvalueRescuedPercent"},
    ),
    "pvalue_newly_flagged": (
        _set_confusion("predictive_pvalue", "newly_flagged", 12_345),
        {"RecPvalueNewlyFlagged", "RecPvalueFlagRateFragmented"},
    ),
    "immobile_speed": (
        _set_figure04(("immobility_comparison", "speed_threshold_cm_per_s"), 7.0),
        {"RecImmobileSpeed"},
    ),
    "immobile_hpd_rate": (
        _set_figure04(("immobility_comparison", "immobile_flag_rates", "hpd_overlap"), 0.37),
        {"RecHpdFlagRateImmobile"},
    ),
    "moving_hpd_rate": (
        _set_figure04(("immobility_comparison", "moving_flag_rates", "hpd_overlap"), 0.37),
        {"RecHpdFlagRateMoving"},
    ),
    "immobile_pvalue_rate": (
        _set_figure04(("immobility_comparison", "immobile_flag_rates", "predictive_pvalue"), 0.37),
        {"RecPvalueFlagRateImmobile"},
    ),
    "moving_pvalue_rate": (
        _set_figure04(("immobility_comparison", "moving_flag_rates", "predictive_pvalue"), 0.37),
        {"RecPvalueFlagRateMoving"},
    ),
    "immobile_kl_median": (
        _set_figure04(("immobility_comparison", "immobile_kl_median"), 0.37),
        {"RecKlMedianImmobile"},
    ),
    "moving_kl_median": (
        _set_figure04(("immobility_comparison", "moving_kl_median"), 0.37),
        {"RecKlMedianMoving"},
    ),
    "hpd_flagged_nonzero_overlap": (
        _set_figure04(("hpd_flagged_nonzero_overlap",), 1_234),
        {"RecHpdFlaggedNonzero"},
    ),
    "kl_lower_fraction": (
        _set_figure04(("kl_divergence_comparison", "lower_fraction"), 0.37),
        {"RecKlLowerPercent"},
    ),
    "kl_lower_median_continuous": (
        _set_figure04(("kl_divergence_comparison", "lower_continuous_median"), 0.37),
        {"RecKlLowerMedianContinuous"},
    ),
    "kl_lower_median_fragmented": (
        _set_figure04(("kl_divergence_comparison", "lower_continuous_fragmented_median"), 0.37),
        {"RecKlLowerMedianFragmented"},
    ),
    "kl_higher_fraction": (
        _set_figure04(("kl_divergence_comparison", "higher_fraction"), 0.37),
        {"RecKlHigherPercent"},
    ),
    "kl_higher_median": (
        _set_figure04(("kl_divergence_comparison", "higher_continuous_median"), 0.37),
        {"RecKlHigherContinuousMedian"},
    ),
    "kl_higher_median_fragmented": (
        _set_figure04(("kl_divergence_comparison", "higher_continuous_fragmented_median"), 0.37),
        {"RecKlHigherFragmentedMedian"},
    ),
    "kl_higher_max": (
        _set_figure04(("kl_divergence_comparison", "higher_continuous_fragmented_max"), 0.37),
        {"RecKlHigherFragmentedMax"},
    ),
    "detail_half_width": (
        _set_figure04(("detail_window", "half_width_samples"), 1_250),
        {"RecDetailDurationS"},
    ),
    "detail_n_events": (
        _set_figure04(("detail_window_comparison", "n_events"), 1_234),
        {"RecDetailNEvents"},
    ),
    "detail_rescued": (
        _set_figure04(("detail_window_comparison", "hpd_overlap", "rescued"), 1_234),
        {"RecDetailHpdFlaggedContinuous"},
    ),
    "detail_both": (
        _set_figure04(("detail_window_comparison", "hpd_overlap", "both"), 1_234),
        {"RecDetailHpdFlaggedContinuous", "RecDetailHpdFlaggedFragmented"},
    ),
    "detail_newly_flagged": (
        _set_figure04(("detail_window_comparison", "hpd_overlap", "newly_flagged"), 1_234),
        {"RecDetailHpdFlaggedFragmented"},
    ),
    "detail_pvalue_rescued": (
        _set_figure04(("detail_window_comparison", "predictive_pvalue", "rescued"), 1_234),
        {"RecDetailPvalueFlaggedContinuous"},
    ),
    "detail_pvalue_both": (
        _set_figure04(("detail_window_comparison", "predictive_pvalue", "both"), 1_234),
        {"RecDetailPvalueFlaggedContinuous", "RecDetailPvalueFlaggedFragmented"},
    ),
    "detail_pvalue_newly_flagged": (
        _set_figure04(("detail_window_comparison", "predictive_pvalue", "newly_flagged"), 1_234),
        {"RecDetailPvalueFlaggedFragmented"},
    ),
    "detail_pvalue_higher": (
        _set_figure04(("detail_window_comparison", "predictive_pvalue_higher"), 1_234),
        {"RecDetailPvalueHigher"},
    ),
    "detail_kl_median_continuous": (
        _set_figure04(("detail_window_comparison", "continuous_kl_median"), 0.37),
        {"RecDetailKlMedianContinuous"},
    ),
    "detail_kl_median_fragmented": (
        _set_figure04(("detail_window_comparison", "continuous_fragmented_kl_median"), 0.37),
        {"RecDetailKlMedianFragmented"},
    ),
    "detail_median_speed": (
        _set_figure04(("detail_window_comparison", "median_speed_cm_per_s"), 0.37),
        {"RecDetailMedianSpeed"},
    ),
    "detail_fragmented_share_median": (
        _set_figure04(("detail_window_comparison", "fragmented_share_median"), 0.37),
        {"RecDetailFragmentedShareMedian"},
    ),
    "detail_fragmented_share_peak": (
        _set_figure04(("detail_window_comparison", "fragmented_share_peak"), 0.37),
        {"RecDetailFragmentedSharePeak"},
    ),
    "detail_fragmented_threshold": (
        _set_figure04(("detail_window_comparison", "fragmented_probability_threshold"), 0.7),
        {"RecFragmentedThreshold"},
    ),
    "detail_fragmented_above_threshold": (
        _set_figure04(
            ("detail_window_comparison", "fragmented_share_above_threshold_samples"), 1_234
        ),
        {"RecDetailFragmentedAboveThresholdMs"},
    ),
}


def _statistic_macros(figure03: dict[str, Any], figure04: dict[str, Any]) -> dict[str, str]:
    """Return the macros computed from the two summaries' statistics."""
    return {
        macro.name: macro.value
        for macro in (*_simulation_statistics(figure03), *_recording_statistics(figure04))
    }


@pytest.fixture(scope="module")
def committed_summaries() -> tuple[dict[str, Any], dict[str, Any]]:
    return load_summary(FIGURE03_SUMMARY_PATH), load_summary(FIGURE04_SUMMARY_PATH)


@pytest.mark.parametrize("field", _SUMMARY_FIELD_MACROS)
def test_each_summary_field_moves_exactly_its_macros(
    field: str, committed_summaries: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    """Editing one statistic moves its macros and no others, so a macro reading
    the wrong condition, metric, or count fails here. Comparing the committed
    file with a fresh render cannot see that, because both use the same emitter."""
    edit, expected = _SUMMARY_FIELD_MACROS[field]
    figure03, figure04 = copy.deepcopy(committed_summaries)
    before = _statistic_macros(figure03, figure04)

    edit(figure03, figure04)
    after = _statistic_macros(figure03, figure04)

    assert {name for name, value in before.items() if after[name] != value} == expected


def test_every_statistic_macro_has_a_summary_field(
    committed_summaries: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """A new statistic macro must declare its field in ``_SUMMARY_FIELD_MACROS``."""
    mapped = set().union(*(macros for _, macros in _SUMMARY_FIELD_MACROS.values()))
    assert mapped == set(_statistic_macros(*committed_summaries))


def test_asymmetric_mode_parameters_are_reported_independently() -> None:
    """Each mode keeps its own initial and transition probability."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    package_defaults = figure04["configuration"]["package_defaults"]
    package_defaults["continuous_fragmented_discrete_initial_conditions"] = [0.6, 0.4]
    package_defaults["continuous_fragmented_diagonal_values"] = [0.9, 0.8]

    values = _macro_values(
        render_macro_file(
            figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
        )
    )

    assert values["RecModeContinuousInitial"] == "0.6"
    assert values["RecModeFragmentedInitial"] == "0.4"
    assert values["RecModeContinuousStay"] == "0.90"
    assert values["RecModeContinuousToFragmented"] == "0.10"
    assert values["RecModeFragmentedToContinuous"] == "0.20"
    assert values["RecModeFragmentedStay"] == "0.80"


@pytest.mark.parametrize(
    ("event_counts", "message"),
    [((870, 871), "disagree on the spike-event count"), ((), "no flag_confusions")],
)
def test_spike_event_count_requires_one_shared_confusion_total(
    event_counts: tuple[int, ...], message: str
) -> None:
    """``RecNEvents`` is the one spike-event total every flag confusion counts."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    template = figure04["flag_confusions"][0]
    figure04["flag_confusions"] = [{**template, "n": n} for n in event_counts]
    with pytest.raises(ValueError, match=message):
        render_macro_file(
            figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
        )


def test_every_figure04_flag_rule_is_reported_or_fails_the_emit() -> None:
    """The flag-count macros follow the summary's flag rules, not a fixed list."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    figure04["flag_rules"]["kl_divergence"] = {
        "comparison": "greater_than_or_equal",
        "threshold": 4.0,
    }
    with pytest.raises(KeyError, match="kl_divergence"):
        render_macro_file(
            figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
        )


def test_time_conversions_follow_the_recorded_step_length() -> None:
    """Durations and rates are converted with the summary's step, not a fixed 1 ms."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["configuration"]["step_seconds"] = 0.002
    figure03["configuration"]["sparse_cell_peak_rate_per_step"] = 0.002

    values = _macro_values(
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )
    )

    assert values["SimStepMs"] == "2"
    assert values["SimDurationSeconds"] == "64"
    assert values["SimRefractoryMs"] == "2"
    assert (values["SimBurstStartMs"], values["SimBurstEndMs"]) == ("4", "20")
    assert values["SimPeakRateHz"] == "100"
    assert values["SimSparseActiveRateHz"] == "1"


def test_negative_log_cutoffs_follow_the_recorded_p_value_cutoffs() -> None:
    """The -log(p) axis positions the prose quotes move with each figure's cutoff."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    figure03["threshold_provenance"]["predictive_pvalue"]["cutoff"] = 0.01
    figure04["flag_rules"]["predictive_pvalue"]["threshold"] = 0.2

    values = _macro_values(
        render_macro_file(
            figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
        )
    )

    assert values["SimPredictiveCutoffNegLog"] == "5"  # -log(0.01) = 4.6
    assert values["RecPredictiveCutoffNegLog"] == "2"  # -log(0.2) = 1.6


@pytest.mark.parametrize(
    ("approach_steps", "phrase"), [(1000, "second"), (2500, "2.5 seconds"), (9000, "4 seconds")]
)
def test_sparse_approach_duration_is_written_as_a_phrase(approach_steps: int, phrase: str) -> None:
    """The approach lasts its configured steps, capped at clean recovery 3 (4 s)."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["configuration"]["sparse_approach_duration_steps"] = approach_steps

    values = _macro_values(
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )
    )

    assert values["SimSparseApproachDuration"] == phrase


def test_non_unit_position_grid_fails_the_emit() -> None:
    """The Methods call the simulated position grid unit-spaced."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["configuration"]["position_bin_size"] = 2

    with pytest.raises(ValueError, match="unit-spaced"):
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


def test_non_integral_burst_factor_is_not_silently_rounded() -> None:
    """The spelled-out prose cannot faithfully represent a fractional factor."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["configuration"]["history_burst_factor"] = 3.4

    with pytest.raises(ValueError, match="not exact"):
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


def test_mismatched_hpd_coverages_are_rejected() -> None:
    """The Methods state one HPD coverage for the simulation and the recording."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["configuration"]["hpd_coverage"] = 0.9

    with pytest.raises(ValueError, match="different HPD coverages"):
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


@pytest.mark.parametrize(
    "path",
    [("source",), ("figure04_caches",)],
    ids=["figure04_source", "figure04_caches"],
)
def test_mismatched_statespacecheck_versions_are_rejected(path: tuple[str]) -> None:
    """The manuscript cites one statespacecheck version for both figures."""
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    figure04["provenance"][path[0]]["statespacecheck_version"] = "0.0.0"

    with pytest.raises(ValueError, match="different statespacecheck versions"):
        render_macro_file(
            load_summary(FIGURE03_SUMMARY_PATH),
            figure04,
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


@pytest.mark.parametrize(
    "keys",
    [("source_tree_sha256",), ("uv_lock_sha256",), ("source_tree_sha256", "uv_lock_sha256")],
    ids=["source_tree", "uv_lock", "both"],
)
def test_mismatched_source_provenance_is_rejected(keys: tuple[str, ...]) -> None:
    """The macro file states one source digest, so both summaries must share their source."""
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    for key in keys:
        figure04["provenance"]["source"][key] = "0" * 64

    with pytest.raises(ValueError, match=rf"provenance.source values for {', '.join(keys)};"):
        render_macro_file(
            load_summary(FIGURE03_SUMMARY_PATH),
            figure04,
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


def test_source_provenance_key_recorded_by_one_summary_is_rejected() -> None:
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["provenance"]["source"]["extra"] = "value"

    with pytest.raises(ValueError, match="provenance.source values for extra;"):
        require_matching_source_provenance(figure03, load_summary(FIGURE04_SUMMARY_PATH))


def test_matching_source_provenance_is_accepted() -> None:
    require_matching_source_provenance(
        load_summary(FIGURE03_SUMMARY_PATH), load_summary(FIGURE04_SUMMARY_PATH)
    )


@pytest.mark.parametrize("line", ["doi: 10.5281/zenodo.7", 'doi: "10.5281/zenodo.7"'])
def test_analysis_code_doi_is_read_from_the_citation_file(tmp_path: Path, line: str) -> None:
    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        f"cff-version: 1.2.0\n{line}\nidentifiers:\n  - type: doi\n", encoding="utf-8"
    )
    assert analysis_code_doi(citation) == "10.5281/zenodo.7"


def test_citation_file_without_a_doi_is_rejected(tmp_path: Path) -> None:
    """Only a top-level doi counts; the paper's preferred-citation has its own fields."""
    citation = tmp_path / "CITATION.cff"
    citation.write_text("cff-version: 1.2.0\npreferred-citation:\n  doi: 1/2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level doi"):
        analysis_code_doi(citation)


def _zenodo_record(version: str, doi: str) -> dict[str, object]:
    return {"doi": doi, "metadata": {"version": version}}


def test_doi_is_taken_from_the_record_of_that_version() -> None:
    search = {
        "hits": {
            "hits": [
                _zenodo_record("v0.3.0", "10.5281/zenodo.1"),
                _zenodo_record("v0.3.10", "10.5281/zenodo.2"),
            ]
        }
    }
    assert doi_from_zenodo_search(search, "0.3.0") == "10.5281/zenodo.1"


@pytest.mark.parametrize("n_records", [0, 2])
def test_version_without_exactly_one_zenodo_record_is_rejected(n_records: int) -> None:
    """An unarchived version (or an ambiguous search) fails rather than citing a guess."""
    search = {"hits": {"hits": [_zenodo_record("v0.3.1", "10.5281/zenodo.3")] * n_records}}
    with pytest.raises(ValueError, match=f"{n_records} archived records"):
        doi_from_zenodo_search(search, "0.3.1")


def _recorded_version() -> str:
    return statespacecheck_version(
        load_summary(FIGURE03_SUMMARY_PATH), load_summary(FIGURE04_SUMMARY_PATH)
    )


def test_committed_doi_file_gives_the_cited_doi() -> None:
    """Emitting offline reads the DOI the manuscript cites from the committed file."""
    assert recorded_statespacecheck_doi(_recorded_version()) == CITED_DOI


def test_emitting_needs_no_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The routine emit path never calls the Zenodo lookup."""

    def no_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("emitting the macros must not open a URL")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    regenerated = write_macro_file(tmp_path / "reported_values.tex")
    assert _macro_values(regenerated.read_text(encoding="utf-8"))["StatespacecheckDOI"] == (
        CITED_DOI
    )


def test_unrecorded_version_names_the_refresh_command(tmp_path: Path) -> None:
    dois = tmp_path / "software_dois.json"
    dois.write_text('{"statespacecheck": {"0.3.1": "10.5281/zenodo.1"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="records no Zenodo DOI for statespacecheck 9.9.9") as err:
        recorded_statespacecheck_doi("9.9.9", dois)
    assert REFRESH_DOIS_COMMAND in str(err.value)
    with pytest.raises(ValueError, match="records no Zenodo DOI"):
        recorded_statespacecheck_doi("0.3.1", tmp_path / "absent.json")


def test_refresh_records_a_new_version_and_keeps_recorded_ones(tmp_path: Path) -> None:
    dois = tmp_path / "software_dois.json"
    lookups = {"0.3.1": "10.5281/zenodo.1", "0.4.0": "10.5281/zenodo.2"}
    assert refresh_statespacecheck_doi("0.3.1", dois, lookup=lookups.__getitem__) == (
        "10.5281/zenodo.1"
    )
    first = dois.read_text(encoding="utf-8")
    assert refresh_statespacecheck_doi("0.3.1", dois, lookup=lookups.__getitem__) == (
        "10.5281/zenodo.1"
    )
    assert dois.read_text(encoding="utf-8") == first  # verified, not rewritten
    refresh_statespacecheck_doi("0.4.0", dois, lookup=lookups.__getitem__)
    assert recorded_statespacecheck_doi("0.3.1", dois) == "10.5281/zenodo.1"
    assert recorded_statespacecheck_doi("0.4.0", dois) == "10.5281/zenodo.2"


def test_refresh_rejects_a_disagreeing_recorded_doi(tmp_path: Path) -> None:
    dois = tmp_path / "software_dois.json"
    dois.write_text('{"statespacecheck": {"0.3.1": "10.5281/zenodo.1"}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="archives it as 10.5281/zenodo.9"):
        refresh_statespacecheck_doi("0.3.1", dois, lookup=lambda _version: "10.5281/zenodo.9")
    assert "zenodo.1" in dois.read_text(encoding="utf-8")


def test_committed_doi_file_is_canonical_json() -> None:
    """The committed file is in the layout refresh_statespacecheck_doi writes."""
    text = SOFTWARE_DOIS_PATH.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


@pytest.mark.parametrize("refresh", [False, True])
def test_emit_script_refreshes_dois_only_on_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, refresh: bool
) -> None:
    script = load_script("emit_reported_values")
    refreshed: list[str] = []
    monkeypatch.setattr(
        script, "refresh_statespacecheck_doi", lambda version: refreshed.append(version) or "doi"
    )
    monkeypatch.setattr(script, "write_macro_file", lambda: tmp_path / "reported_values.tex")
    script.main(["--refresh-dois"] if refresh else [])
    assert refreshed == ([_recorded_version()] if refresh else [])


@pytest.mark.network
def test_recorded_doi_is_zenodos_record() -> None:
    """The committed DOI of the recorded version is Zenodo's record of it."""
    version = _recorded_version()
    try:
        doi = lookup_statespacecheck_doi(version)
    except (urllib.error.URLError, TimeoutError) as err:  # offline, or Zenodo unavailable
        pytest.skip(f"Zenodo cannot be reached: {err}")
    assert doi == recorded_statespacecheck_doi(version) == CITED_DOI


@pytest.mark.parametrize("quantile", [0.005, 0.995])
def test_fractional_percentile_is_not_silently_rounded(quantile: float) -> None:
    """0.005 must not print as "0th": the ordinal prose has no form for it."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure03["threshold_provenance"]["hpd_overlap"]["quantile"] = quantile

    with pytest.raises(ValueError, match="not exact"):
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )


def test_zero_decoding_error_renders() -> None:
    """A perfectly decoded phase must not abort the emit."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    well_specified = figure03["condition_order"].index("well_specified")
    figure03["median_decoding_error"][well_specified] = 0.0

    values = _macro_values(
        render_macro_file(
            figure03,
            load_summary(FIGURE04_SUMMARY_PATH),
            statespacecheck_doi=CITED_DOI,
            analysis_code_doi=CODE_DOI,
        )
    )

    assert values["SimWellSpecifiedError"] == "0"


def test_macro_names_are_unique() -> None:
    """A duplicated name would make ``\\newcommand`` abort the LaTeX build."""
    text = MACRO_FILE_PATH.read_text(encoding="utf-8")
    names = re.findall(r"\\newcommand\{\\(\w+)\}", text)
    assert len(names) == len(set(names))


def test_every_macro_is_used_by_the_manuscript() -> None:
    """An unused macro is a number nobody reports; drop it rather than ship it."""
    macro_text = MACRO_FILE_PATH.read_text(encoding="utf-8")
    manuscript = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    unused = [
        name
        for name in re.findall(r"\\newcommand\{\\(\w+)\}", macro_text)
        if not re.search(rf"\\{name}(?![A-Za-z])", manuscript)
    ]
    assert unused == []


def test_data_statement_links_the_published_input_file() -> None:
    """The Figure-4 data DOI is cited as a link through its macro, which the
    committed-macro test ties to the DOI the download verifies."""
    manuscript = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
    assert "\\href{https://doi.org/\\RecordingInputsDOI}" in manuscript


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
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    baseline = _macro_values(
        render_macro_file(
            figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
        )
    )
    # Shrink every published SE a thousandfold; no printed digit may change.
    for key in (
        "median_flag_percentage_standard_errors",
        "median_decoding_error_standard_errors",
    ):
        figure03[key] = (np.asarray(figure03[key]) / 1000.0).tolist()
    assert (
        _macro_values(
            render_macro_file(
                figure03, figure04, statespacecheck_doi=CITED_DOI, analysis_code_doi=CODE_DOI
            )
        )
        == baseline
    )


def test_word_helpers() -> None:
    assert cardinal_word(11) == "eleven"
    assert ordinal(1) == "1st"
    assert ordinal(99) == "99th"
    with pytest.raises(ValueError, match="cardinal_word covers"):
        cardinal_word(21)


def test_render_is_deterministic() -> None:
    """Two renders of the same payloads agree byte for byte."""
    figure03 = load_summary(FIGURE03_SUMMARY_PATH)
    figure04 = load_summary(FIGURE04_SUMMARY_PATH)
    dois = {"statespacecheck_doi": CITED_DOI, "analysis_code_doi": CODE_DOI}
    assert render_macro_file(figure03, figure04, **dois) == render_macro_file(
        figure03, figure04, **dois
    )
