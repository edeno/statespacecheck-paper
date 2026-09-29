r"""Emit the manuscript's reported numbers as LaTeX macros.

The manuscript's reported analysis statistics and configuration live in the
two canonical figure summaries (``figure03_summary.json`` /
``figure04_summary.json``). This module turns those summaries into a file of
``\newcommand`` definitions that ``main.tex`` inputs, so the prose cannot drift
from the artifacts the way hand-typed numbers can.

Rounding and derivation happen here, in Python, rather than in the document:
the manuscript says "42 a.u." where the summary holds ``42.16926259332958``,
and "92%" where it holds ``0.9201170835550825``. Ranges such as the remap
flag percentages are emitted as separate ``\dots Min`` / ``\dots Max`` macros
so the en-dash stays in the prose.

**Prose reporting policy.** Decoding errors are reported to two significant figures,
and flag and rescue percentages to the nearest whole percent. Exact counts and
configured parameters are reported without approximation. Machine-readable
summaries retain full numerical precision.

Rounding is a communication choice, not a statement about uncertainty: each
rule preserves the distinctions the prose draws with the number. Decoding
errors are compared as ratios ("four times the well-specified value", "one
fifth of a place-field width"), which two significant figures support; flag
percentages are compared as substantial, low, or modest, which whole percents
support; rescued percentages are descriptive of this recording and appear next to
their exact counts. Constants the text hedges with "approximately" (199.47 Hz,
sqrt(12.5) cm) are exact functions of chosen parameters and are shown to two
significant figures. The p-value cutoff's position on the figures' -log(p)
axis (-log(0.05) = 2.996) is shown to the nearest whole number, because the text
reads it off that axis ("x = 3"). Where variability matters to a claim it belongs in the
text or a figure, not in the digit count; the Figure-3 summary publishes
approximate standard errors of the aggregated medians, conditional on the
simulation setup. These do not describe the spread of individual realizations
and play no part in formatting.

Exact counts and configured parameters go through :func:`_exact`, which prints
them in full and *raises* if the requested precision would lose information,
so a configuration change from ``0.88`` to ``0.875`` fails the emit rather
than quietly printing ``0.88``.

Macros are prefixed ``\Sim`` (simulation study, Figure 3) or ``\Rec``
(hippocampal recording, Figure 4); a setting both analyses share, such as the
HPD coverage, carries no prefix and must agree between the two summaries.
``\newcommand`` deliberately errors on a
name clash, so a collision with a package macro fails the build rather than
silently redefining anything.

The analysis numbers come only from the committed summary JSONs. The three
DOIs come from elsewhere: the archived DOI of the cited ``statespacecheck``
version, which :func:`write_macro_file` looks up on Zenodo (so emitting needs
internet access); this repository's DOI, read from ``CITATION.cff``; and the
Figure-4 input file's DOI, ``paths.FIGURE04_INPUTS_DOI``. Its sibling imports
are ``number_format``, the rounding shared with the Figure-3 summary panel, so
the figure and the prose cannot round the same number differently, and
``paths``, for the summary locations and that data DOI.
"""

from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statespacecheck_paper.number_format import SIGNIFICANT_FIGURES, significant, whole_percent
from statespacecheck_paper.paths import (
    FIGURE03_SUMMARY_PATH,
    FIGURE04_INPUTS_DOI,
    FIGURE04_SUMMARY_PATH,
)

MACRO_FILE_PATH = Path("manuscript/reported_values.tex")
# This repository's citation metadata; its ``doi`` is the analysis code's Zenodo DOI
CITATION_PATH = Path("CITATION.cff")

# Zenodo record that groups every archived statespacecheck release (its concept
# record); each release's own DOI is looked up under it by version
STATESPACECHECK_ZENODO_CONCEPT_RECORD = "22999988"

# Unit conversion for the durations the prose gives in milliseconds
_MS_PER_SECOND = 1000.0

# Spelled-out cardinals for the counts the manuscript writes as words
# ("eleven place cells", "Five additional cells"). Only small counts appear,
# so a short table beats a spell-out dependency.
_CARDINAL_WORDS: tuple[str, ...] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)


@dataclass(frozen=True)
class MacroDefinition:
    r"""One ``\newcommand`` line plus the comment that documents its origin.

    Parameters
    ----------
    name : str
        Macro name without the leading backslash.
    value : str
        Fully formatted replacement text.
    comment : str
        Short note naming the summary field or derivation behind the value.
    """

    name: str
    value: str
    comment: str


def cardinal_word(count: int) -> str:
    """Spell a small non-negative count as an English word.

    Parameters
    ----------
    count : int
        Count to spell; must be in ``[0, 20]``.

    Returns
    -------
    str
        The spelled-out cardinal, e.g. ``"eleven"``.

    Raises
    ------
    ValueError
        If ``count`` falls outside the table.

    Examples
    --------
    >>> cardinal_word(11)
    'eleven'
    """
    if not 0 <= count < len(_CARDINAL_WORDS):
        raise ValueError(f"cardinal_word covers 0-{len(_CARDINAL_WORDS) - 1}; got {count}")
    return _CARDINAL_WORDS[count]


def ordinal(value: int) -> str:
    """Render an integer as an English ordinal.

    Parameters
    ----------
    value : int
        Non-negative integer.

    Returns
    -------
    str
        The ordinal, e.g. ``"1st"`` or ``"99th"``.

    Examples
    --------
    >>> ordinal(1), ordinal(99)
    ('1st', '99th')
    """
    if value % 100 in (11, 12, 13):
        return f"{value}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


# Relative slack when checking that a value survives its printed precision.
# Derived quantities carry floating-point dust (``position_std ** 2`` is
# 12.500000000000002, not 12.5); a genuine precision loss is orders of
# magnitude larger than this.
_EXACTNESS_TOLERANCE = 1e-9


def _exact(value: float, decimals: int = 0) -> str:
    """Render a value the manuscript reports without approximation.

    Parameters
    ----------
    value : float
        Value to render.
    decimals : int, default 0
        Decimal places the manuscript prints.

    Returns
    -------
    str
        The formatted value.

    Raises
    ------
    ValueError
        If ``decimals`` would lose information -- the printed form would no
        longer faithfully render the artifact's value.

    Examples
    --------
    >>> _exact(0.88, 2)
    '0.88'
    """
    if abs(value - round(value, decimals)) > _EXACTNESS_TOLERANCE * max(1.0, abs(value)):
        raise ValueError(
            f"{value} is not exact to {decimals} decimal(s); use significant or "
            "whole_percent for a summary statistic, or print more digits."
        )
    return f"{value:.{decimals}f}"


def _negative_log_cutoff(cutoff: float) -> str:
    """Render a p-value cutoff's position on the -log(p) axis, to a whole number.

    Parameters
    ----------
    cutoff : float
        Predictive p-value cutoff in ``(0, 1)``.

    Returns
    -------
    str
        ``-log(cutoff)`` (natural log), rounded to the nearest whole number.

    Examples
    --------
    >>> _negative_log_cutoff(0.05)
    '3'
    """
    return f"{-math.log(cutoff):.0f}"


def _load(path: Path) -> dict[str, Any]:
    """Read one summary JSON."""
    with open(path, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _flag_percentage_range(payload: dict[str, Any], condition: str) -> tuple[float, float]:
    """Return the min and max median flag percentage across the three metrics.

    Parameters
    ----------
    payload : dict
        Figure-3 summary payload.
    condition : str
        Entry of ``condition_order`` naming the column to summarize.

    Returns
    -------
    tuple of float
        ``(minimum, maximum)`` over the three diagnostic rows.
    """
    column = payload["condition_order"].index(condition)
    values = [row[column] for row in payload["median_flag_percentages"]]
    return min(values), max(values)


def _flag_percentage(payload: dict[str, Any], metric: str, condition: str) -> float:
    """Return one metric's median flag percentage for one condition."""
    row = payload["metric_order"].index(metric)
    column = payload["condition_order"].index(condition)
    percentage: float = payload["median_flag_percentages"][row][column]
    return percentage


def _decoding_error(payload: dict[str, Any], condition: str) -> float:
    """Return the median absolute decoding error for one condition."""
    row = payload["accuracy_metric_order"].index("median_absolute_error")
    column = payload["condition_order"].index(condition)
    error: float = payload["median_decoding_accuracy"][row][column]
    return error


def _confusion(payload: dict[str, Any], metric: str) -> dict[str, Any]:
    """Return the Figure-4 flag-confusion entry for one metric."""
    for confusion in payload["flag_confusions"]:
        if confusion["metric"] == metric:
            return dict(confusion)
    raise KeyError(f"figure04_summary.json has no flag_confusions entry for {metric!r}")


def _simulation_statistics(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-3 macros computed from the simulated data."""
    macros: list[MacroDefinition] = [
        MacroDefinition(
            "SimNRealizations",
            _exact(payload["realizations"]["count"]),
            "realizations.count",
        )
    ]

    # Flag-percentage ranges across the three diagnostics, one condition per row.
    for name, condition in (
        ("SimRemapFlag", "remap"),
        ("SimHistoryFlag", "history_dependent"),
        ("SimReplayFlag", "replay"),
        ("SimDriftFlag", "drift"),
    ):
        low, high = _flag_percentage_range(payload, condition)
        macros.append(
            MacroDefinition(
                f"{name}Min",
                whole_percent(low),
                f"min over metrics of median_flag_percentages[:, {condition}]",
            )
        )
        macros.append(
            MacroDefinition(
                f"{name}Max",
                whole_percent(high),
                f"max over metrics of median_flag_percentages[:, {condition}]",
            )
        )

    macros.append(
        MacroDefinition(
            "SimSparseKlFlag",
            whole_percent(_flag_percentage(payload, "kl_divergence", "sparse_population")),
            "median_flag_percentages[kl_divergence, sparse_population]",
        )
    )

    for name, condition in (
        ("SimWellSpecifiedError", "well_specified"),
        ("SimRemapError", "remap"),
        ("SimHistoryError", "history_dependent"),
        ("SimReplayError", "replay"),
        ("SimDriftError", "drift"),
        ("SimSparseError", "sparse_population"),
    ):
        macros.append(
            MacroDefinition(
                name,
                significant(_decoding_error(payload, condition), SIGNIFICANT_FIGURES),
                f"median_decoding_accuracy[median_absolute_error, {condition}]",
            )
        )

    # Thresholds: the HPD and KL cutoff values are estimated from the pooled
    # baseline at fixed percentile levels. The prose writes those levels as
    # ordinals ("1st percentile"), which has no faithful
    # form for a fractional level, so a configured 0.005 must fail the emit
    # rather than round to "0th".
    provenance = payload["threshold_provenance"]
    hpd_percentile = int(_exact(100.0 * provenance["hpd_overlap"]["quantile"]))
    kl_percentile = int(_exact(100.0 * provenance["kl_divergence"]["quantile"]))
    macros.extend(
        [
            MacroDefinition(
                "SimHpdPercentile",
                ordinal(hpd_percentile),
                "threshold_provenance.hpd_overlap.quantile",
            ),
            MacroDefinition(
                "SimKlPercentile",
                ordinal(kl_percentile),
                "threshold_provenance.kl_divergence.quantile",
            ),
            MacroDefinition(
                "SimPredictiveCutoff",
                _exact(provenance["predictive_pvalue"]["cutoff"], 2),
                "threshold_provenance.predictive_pvalue.cutoff",
            ),
            MacroDefinition(
                "SimPredictiveCutoffNegLog",
                _negative_log_cutoff(provenance["predictive_pvalue"]["cutoff"]),
                "-log(threshold_provenance.predictive_pvalue.cutoff), nearest whole number",
            ),
            MacroDefinition(
                "SimBaselineEnd",
                _exact(provenance["baseline_end_index"]),
                "threshold_provenance.baseline_end_index",
            ),
        ]
    )
    return macros


def _simulation_configuration(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-3 macros recording the simulation's chosen inputs."""
    config = payload["configuration"]
    boundaries = config["phase_boundaries"]
    centers = config["place_field_centers"]
    field_std: float = config["place_field_std"]
    rate_scale: float = config["place_field_rate_scale"]
    # Simulation time is counted in steps of this length.
    step_seconds: float = config["step_seconds"]
    step_ms = step_seconds * _MS_PER_SECOND
    # Peak of the Gaussian place field, in expected spikes per step.
    peak_count_per_step = rate_scale / (field_std * math.sqrt(2.0 * math.pi))
    # Per-cell sparse rates, converted from spikes/step to Hz.
    sparse_active_hz = config["sparse_cell_peak_rate_per_step"] / step_seconds
    sparse_baseline_hz = sparse_active_hz * config["sparse_cell_baseline_rate_fraction"]
    remap_displacement = min(abs(dst - src) for src, dst in config["place_field_remapping"])
    # The replay sweep occupies a fractional sub-window of clean recovery 2.
    recovery_two_start, recovery_two_end = boundaries[3], boundaries[4]
    recovery_two_span = recovery_two_end - recovery_two_start
    # The prose spells this value as "threefold", so a fractional factor
    # cannot be represented faithfully. Reuse the exact-value guard rather
    # than silently rounding a valid simulation setting to a different value.
    burst_factor_word = cardinal_word(int(_exact(config["history_burst_factor"])))
    # The Methods describe the position grid as having "unit spacing".
    if config["position_bin_size"] != 1:
        raise ValueError(
            "The Methods describe the simulated position grid as unit-spaced; "
            f"position_bin_size is {config['position_bin_size']}"
        )
    # The approach to the sparse-population location takes the last
    # sparse_approach_duration_steps of clean recovery 3 (all of it if shorter).
    recovery_three_steps = boundaries[6] - boundaries[5]
    approach_seconds = (
        min(config["sparse_approach_duration_steps"], recovery_three_steps) * step_seconds
    )
    # The prose reads "during the last second"; any other duration is written out.
    approach_phrase = (
        "second" if math.isclose(approach_seconds, 1.0) else f"{approach_seconds:g} seconds"
    )

    def replay_bound(fraction: float) -> int:
        return int(round(recovery_two_start + fraction * recovery_two_span))

    return [
        MacroDefinition("SimTrackLength", _exact(config["position_max"]), "position_max"),
        MacroDefinition("SimPositionMin", _exact(config["position_min"]), "position_min"),
        MacroDefinition(
            "SimPositionSecondBin",
            _exact(config["position_min"] + config["position_bin_size"]),
            "position_min + position_bin_size",
        ),
        MacroDefinition(
            "SimNPlaceCellsWord",
            cardinal_word(len(centers)),
            "len(place_field_centers)",
        ),
        MacroDefinition(
            "SimNNeuronsWord",
            cardinal_word(len(centers) + config["sparse_cell_count"]),
            "len(place_field_centers) + sparse_cell_count",
        ),
        MacroDefinition(
            "SimPlaceFieldSpacing",
            _exact(centers[1] - centers[0]),
            "place_field_centers spacing",
        ),
        MacroDefinition("SimPlaceFieldStd", _exact(field_std), "place_field_std"),
        MacroDefinition("SimRateScale", _exact(rate_scale), "place_field_rate_scale"),
        MacroDefinition(
            "SimPeakCountPerStep",
            significant(peak_count_per_step, SIGNIFICANT_FIGURES),
            "place_field_rate_scale / (place_field_std * sqrt(2 pi))",
        ),
        MacroDefinition(
            "SimPeakRateHz",
            significant(peak_count_per_step / step_seconds, SIGNIFICANT_FIGURES),
            "peak expected count per step / step_seconds, in Hz",
        ),
        MacroDefinition(
            "SimPredictionStepStd",
            _exact(config["prediction_step_std"], 1),
            "prediction_step_std",
        ),
        MacroDefinition(
            "SimInitialPosition", _exact(config["initial_position"]), "initial_position"
        ),
        MacroDefinition(
            "SimNSparseCellsWord",
            cardinal_word(config["sparse_cell_count"]),
            "sparse_cell_count",
        ),
        MacroDefinition(
            "SimNSparseCellsWordCap",
            cardinal_word(config["sparse_cell_count"]).capitalize(),
            "sparse_cell_count, sentence-initial",
        ),
        MacroDefinition("SimSparsePosition", _exact(config["sparse_position"]), "sparse_position"),
        MacroDefinition(
            "SimSparseApproachDuration",
            approach_phrase,
            "sparse_approach_duration_steps * step_seconds, as a phrase",
        ),
        MacroDefinition(
            "SimSparseSpreadLow",
            _exact(config["sparse_position"] - config["sparse_place_field_spread"], 1),
            "sparse_position - sparse_place_field_spread",
        ),
        MacroDefinition(
            "SimSparseSpreadHigh",
            _exact(config["sparse_position"] + config["sparse_place_field_spread"], 1),
            "sparse_position + sparse_place_field_spread",
        ),
        MacroDefinition(
            "SimSparseFieldStd",
            _exact(config["sparse_place_field_std"]),
            "sparse_place_field_std",
        ),
        MacroDefinition(
            "SimSparseBaselineRateHz",
            _exact(sparse_baseline_hz, 2),
            "sparse_cell_peak_rate_per_step * sparse_cell_baseline_rate_fraction / step_seconds",
        ),
        MacroDefinition(
            "SimSparseActiveRateHz",
            _exact(sparse_active_hz),
            "sparse_cell_peak_rate_per_step / step_seconds, in Hz",
        ),
        MacroDefinition("SimStepMs", _exact(step_ms), "step_seconds, in ms"),
        MacroDefinition("SimDurationSteps", _exact(boundaries[-1]), "phase_boundaries[-1]"),
        MacroDefinition(
            "SimDurationSeconds",
            _exact(boundaries[-1] * step_seconds),
            "phase_boundaries[-1] * step_seconds",
        ),
        MacroDefinition("SimNPhasesWord", cardinal_word(len(boundaries)), "len(phase_boundaries)"),
        MacroDefinition("SimRemapStart", _exact(boundaries[0]), "phase_boundaries[0]"),
        MacroDefinition("SimRemapEnd", _exact(boundaries[1]), "phase_boundaries[1]"),
        MacroDefinition("SimRecoveryOneEnd", _exact(boundaries[2]), "phase_boundaries[2]"),
        MacroDefinition("SimHistoryEnd", _exact(boundaries[3]), "phase_boundaries[3]"),
        MacroDefinition("SimRecoveryTwoEnd", _exact(boundaries[4]), "phase_boundaries[4]"),
        MacroDefinition("SimDriftEnd", _exact(boundaries[5]), "phase_boundaries[5]"),
        MacroDefinition("SimRecoveryThreeEnd", _exact(boundaries[6]), "phase_boundaries[6]"),
        MacroDefinition("SimSparseEnd", _exact(boundaries[7]), "phase_boundaries[7]"),
        MacroDefinition(
            "SimReplayStart",
            _exact(replay_bound(config["replay_start_fraction"])),
            "replay_start_fraction within clean recovery 2",
        ),
        MacroDefinition(
            "SimReplayEnd",
            _exact(replay_bound(config["replay_end_fraction"])),
            "replay_end_fraction within clean recovery 2",
        ),
        MacroDefinition(
            "SimRemapDisplacementWord",
            cardinal_word(remap_displacement),
            "min displacement in place_field_remapping",
        ),
        MacroDefinition("SimDriftMomentum", _exact(config["drift_momentum"], 2), "drift_momentum"),
        MacroDefinition(
            "SimRefractoryMs",
            _exact(config["history_refractory_steps"] * step_ms),
            "history_refractory_steps * step_seconds, in ms",
        ),
        MacroDefinition(
            "SimBurstStartMs",
            _exact(config["history_burst_window"][0] * step_ms),
            "history_burst_window[0] * step_seconds, in ms",
        ),
        MacroDefinition(
            "SimBurstEndMs",
            _exact(config["history_burst_window"][1] * step_ms),
            "history_burst_window[1] * step_seconds, in ms",
        ),
        MacroDefinition(
            "SimBurstFactorWord",
            burst_factor_word,
            "history_burst_factor",
        ),
    ]


# Macro-name prefix of each Figure-4 metric that can carry a flag rule. Flag
# counts are reported for every metric in the summary's ``flag_rules``; one
# missing here fails the emit instead of going unreported.
_RECORDING_FLAG_MACRO_PREFIXES = {"hpd_overlap": "RecHpd", "predictive_pvalue": "RecPvalue"}


def _recording_statistics(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-4 macros computed from the hippocampal recording."""
    macros = [
        MacroDefinition("RecNUnits", _exact(payload["dataset"]["n_units"]), "dataset.n_units")
    ]
    for metric in payload["flag_rules"]:
        prefix = _RECORDING_FLAG_MACRO_PREFIXES[metric]
        confusion = _confusion(payload, metric)
        # Flagged by the Continuous model = rescued by Continuous-Fragmented + flagged by both.
        flagged_continuous = confusion["rescued"] + confusion["both"]
        macros.extend(
            [
                MacroDefinition(
                    f"{prefix}FlaggedContinuous",
                    _exact(flagged_continuous),
                    f"flag_confusions[{metric}]: rescued + both",
                ),
                MacroDefinition(
                    f"{prefix}Rescued",
                    _exact(confusion["rescued"]),
                    f"flag_confusions[{metric}].rescued",
                ),
                MacroDefinition(
                    f"{prefix}RescuedPercent",
                    whole_percent(100.0 * confusion["rescued_fraction"]),
                    f"flag_confusions[{metric}].rescued_fraction",
                ),
                MacroDefinition(
                    f"{prefix}NewlyFlagged",
                    _exact(confusion["newly_flagged"]),
                    f"flag_confusions[{metric}].newly_flagged",
                ),
            ]
        )
    return macros


def _recording_configuration(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-4 macros recording the decoder's chosen inputs."""
    decoder = payload["configuration"]["decoder"]
    package_defaults = payload["configuration"]["package_defaults"]
    flag_rules = payload["flag_rules"]
    position_std: float = decoder["position_std"]
    continuous_initial, fragmented_initial = package_defaults[
        "continuous_fragmented_discrete_initial_conditions"
    ]
    continuous_diagonal, fragmented_diagonal = package_defaults[
        "continuous_fragmented_diagonal_values"
    ]
    return [
        MacroDefinition(
            "RecPositionBinSizeCm",
            _exact(decoder["position_bin_size_cm"]),
            "configuration.decoder.position_bin_size_cm",
        ),
        MacroDefinition(
            "RecTimeBinMs",
            _exact(_MS_PER_SECOND / decoder["sampling_frequency_hz"]),
            "1 / configuration.decoder.sampling_frequency_hz",
        ),
        MacroDefinition(
            "RecPositionStdVariance",
            _exact(position_std**2, 1),
            "configuration.decoder.position_std squared",
        ),
        MacroDefinition(
            "RecPositionStdCm",
            significant(position_std, SIGNIFICANT_FIGURES),
            "configuration.decoder.position_std",
        ),
        MacroDefinition(
            "RecMovementVar",
            _exact(package_defaults["movement_var"], 1),
            "configuration.package_defaults.movement_var",
        ),
        MacroDefinition(
            "RecModeContinuousInitial",
            _exact(continuous_initial, 1),
            "configuration.package_defaults.continuous_fragmented_discrete_initial_conditions[0]",
        ),
        MacroDefinition(
            "RecModeFragmentedInitial",
            _exact(fragmented_initial, 1),
            "configuration.package_defaults.continuous_fragmented_discrete_initial_conditions[1]",
        ),
        MacroDefinition(
            "RecModeContinuousStay",
            _exact(continuous_diagonal, 2),
            "configuration.package_defaults.continuous_fragmented_diagonal_values[0]",
        ),
        MacroDefinition(
            "RecModeContinuousToFragmented",
            _exact(1.0 - continuous_diagonal, 2),
            "1 - configuration.package_defaults.continuous_fragmented_diagonal_values[0]",
        ),
        MacroDefinition(
            "RecModeFragmentedToContinuous",
            _exact(1.0 - fragmented_diagonal, 2),
            "1 - configuration.package_defaults.continuous_fragmented_diagonal_values[1]",
        ),
        MacroDefinition(
            "RecModeFragmentedStay",
            _exact(fragmented_diagonal, 2),
            "configuration.package_defaults.continuous_fragmented_diagonal_values[1]",
        ),
        MacroDefinition(
            "RecNldVersion",
            package_defaults["non_local_detector_version"],
            "configuration.package_defaults.non_local_detector_version",
        ),
        MacroDefinition(
            "RecHpdCutoff",
            _exact(flag_rules["hpd_overlap"]["threshold"], 2),
            "flag_rules.hpd_overlap.threshold",
        ),
        MacroDefinition(
            "RecPredictiveCutoff",
            _exact(flag_rules["predictive_pvalue"]["threshold"], 2),
            "flag_rules.predictive_pvalue.threshold",
        ),
        MacroDefinition(
            "RecPredictiveCutoffNegLog",
            _negative_log_cutoff(flag_rules["predictive_pvalue"]["threshold"]),
            "-log(flag_rules.predictive_pvalue.threshold), nearest whole number",
        ),
    ]


def statespacecheck_version(
    figure03_payload: dict[str, Any], figure04_payload: dict[str, Any]
) -> str:
    """Return the ``statespacecheck`` version both figure summaries record.

    Raises
    ------
    ValueError
        If the two summaries, or the Figure-4 diagnostics cache, record
        different ``statespacecheck`` versions, since the manuscript cites one.
    """
    recorded = {
        "figure03 provenance.source": figure03_payload["provenance"]["source"][
            "statespacecheck_version"
        ],
        "figure04 provenance.source": figure04_payload["provenance"]["source"][
            "statespacecheck_version"
        ],
        "figure04 provenance.figure04_caches": figure04_payload["provenance"]["figure04_caches"][
            "statespacecheck_version"
        ],
    }
    if len(set(recorded.values())) != 1:
        raise ValueError(
            "The figure summaries record different statespacecheck versions; "
            f"regenerate them in one environment: {recorded}"
        )
    version: str = recorded["figure03 provenance.source"]
    return version


def hpd_coverage_percent(figure03_payload: dict[str, Any], figure04_payload: dict[str, Any]) -> str:
    """Return the HPD coverage both figures' diagnostics used, in percent.

    Raises
    ------
    ValueError
        If the two summaries record different coverages, since the Methods
        state one for both analyses.
    """
    simulation: float = figure03_payload["configuration"]["hpd_coverage"]
    recording: float = figure04_payload["configuration"]["diagnostics"]["hpd_coverage"]
    if simulation != recording:
        raise ValueError(
            f"The figure summaries record different HPD coverages: {simulation} (Figure 3), "
            f"{recording} (Figure 4)"
        )
    return _exact(100.0 * simulation)


def lookup_statespacecheck_doi(version: str) -> str:
    """Look up the Zenodo DOI of a ``statespacecheck`` release (needs internet access).

    Parameters
    ----------
    version : str
        The release, e.g. ``"0.3.0"``.

    Returns
    -------
    str
        The DOI of that version's Zenodo record, e.g. ``"10.5281/zenodo.22999989"``.

    Raises
    ------
    ValueError
        If Zenodo has not archived that version (yet).
    urllib.error.URLError
        If Zenodo cannot be reached.
    TimeoutError
        If Zenodo does not reply within 30 seconds.
    """
    query = urllib.parse.urlencode(
        {
            "q": f"conceptrecid:{STATESPACECHECK_ZENODO_CONCEPT_RECORD} "
            f'AND metadata.version:"v{version}"',
            "all_versions": "true",
        }
    )
    with urllib.request.urlopen(f"https://zenodo.org/api/records?{query}", timeout=30) as reply:
        search = json.load(reply)
    return doi_from_zenodo_search(search, version)


def doi_from_zenodo_search(search: dict[str, Any], version: str) -> str:
    """Return the DOI of ``version`` from a Zenodo records search reply.

    Raises
    ------
    ValueError
        If the reply has no record, or more than one, whose version is ``v{version}``.
    """
    records = [
        record
        for record in search["hits"]["hits"]
        if record["metadata"].get("version") == f"v{version}"
    ]
    if len(records) != 1:
        raise ValueError(
            f"Zenodo has {len(records)} archived records of statespacecheck {version} "
            "(it archives a release some minutes after it is published)"
        )
    doi: str = records[0]["doi"]
    return doi


def analysis_code_doi(citation_path: Path = CITATION_PATH) -> str:
    """Return this repository's Zenodo DOI (all versions), the ``doi`` of its CITATION.cff.

    Raises
    ------
    ValueError
        If the file has no top-level ``doi`` field.
    """
    match = re.search(r'^doi:\s*"?([^"\s]+)"?\s*$', citation_path.read_text(encoding="utf-8"), re.M)
    if match is None:
        raise ValueError(f"{citation_path} has no top-level doi field")
    return match.group(1)


def macro_sections(
    figure03_payload: dict[str, Any],
    figure04_payload: dict[str, Any],
    *,
    statespacecheck_doi: str | None = None,
    analysis_code_doi: str | None = None,
) -> tuple[tuple[str, list[MacroDefinition]], ...]:
    """Return every reported-value macro, grouped into titled sections.

    The manuscript's macro file and the project website both read these
    definitions, so the two cannot report a number differently.

    Parameters
    ----------
    figure03_payload, figure04_payload : dict
        Parsed contents of the two canonical figure summaries.
    statespacecheck_doi : str, optional, keyword-only
        Zenodo DOI of the recorded ``statespacecheck`` version
        (:func:`lookup_statespacecheck_doi`); the manuscript cites it and the website
        does not use it. Without it, no DOI macro is emitted.
    analysis_code_doi : str, optional, keyword-only
        This repository's Zenodo DOI (:func:`analysis_code_doi`), cited by the
        manuscript; without it, no macro for it is emitted.

    Returns
    -------
    tuple of (str, list of MacroDefinition)
        Section title and its macros, in file order.
    """
    software = [
        MacroDefinition(
            "StatespacecheckVersion",
            statespacecheck_version(figure03_payload, figure04_payload),
            "provenance.source.statespacecheck_version (both summaries)",
        )
    ]
    if statespacecheck_doi is not None:
        software.append(
            MacroDefinition("StatespacecheckDOI", statespacecheck_doi, "Zenodo DOI of that version")
        )
    if analysis_code_doi is not None:
        software.append(
            MacroDefinition(
                "AnalysisCodeDOI",
                analysis_code_doi,
                "this repository's Zenodo DOI, all versions (CITATION.cff doi)",
            )
        )
    data = [
        MacroDefinition(
            "RecordingInputsDOI",
            FIGURE04_INPUTS_DOI,
            "Zenodo DOI of the Figure-4 input file (paths.FIGURE04_INPUTS_DOI)",
        )
    ]
    diagnostics = [
        MacroDefinition(
            "HpdCoveragePercent",
            hpd_coverage_percent(figure03_payload, figure04_payload),
            "100 * hpd_coverage (figure03 configuration; figure04 configuration.diagnostics)",
        )
    ]
    return (
        ("Software", software),
        ("Data", data),
        ("Diagnostics (both figures)", diagnostics),
        (
            "Simulation study (Figure 3) --- computed from the simulated data",
            _simulation_statistics(figure03_payload),
        ),
        (
            "Simulation study (Figure 3) --- recorded configuration",
            _simulation_configuration(figure03_payload),
        ),
        (
            "Hippocampal recording (Figure 4) --- computed from the recording",
            _recording_statistics(figure04_payload),
        ),
        (
            "Hippocampal recording (Figure 4) --- recorded configuration",
            _recording_configuration(figure04_payload),
        ),
    )


def render_macro_file(
    figure03_payload: dict[str, Any],
    figure04_payload: dict[str, Any],
    *,
    statespacecheck_doi: str,
    analysis_code_doi: str,
) -> str:
    """Render the full ``reported_values.tex`` contents.

    Parameters
    ----------
    figure03_payload, figure04_payload : dict
        Parsed contents of the two canonical figure summaries.
    statespacecheck_doi : str, keyword-only
        Zenodo DOI of the recorded ``statespacecheck`` version
        (:func:`lookup_statespacecheck_doi`).
    analysis_code_doi : str, keyword-only
        This repository's Zenodo DOI (:func:`analysis_code_doi`).

    Returns
    -------
    str
        File text, ending in a newline.
    """
    sections = macro_sections(
        figure03_payload,
        figure04_payload,
        statespacecheck_doi=statespacecheck_doi,
        analysis_code_doi=analysis_code_doi,
    )
    source_hash = figure03_payload["provenance"]["source"]["source_tree_sha256"]
    lines = [
        "% Generated by scripts/emit_reported_values.py --- do not edit by hand.",
        "%",
        "% Every value below is read from the canonical figure summaries",
        "% (except the DOIs, whose comments name their sources):",
        f"%   figures/main/figure03_summary.json (schema {figure03_payload['schema_version']})",
        f"%   figures/main/figure04_summary.json (schema {figure04_payload['schema_version']})",
        f"% source_tree_sha256: {source_hash}",
        "%",
        "% Regenerate with: uv run python scripts/emit_reported_values.py",
    ]
    for title, macros in sections:
        lines.extend(["", f"% --- {title}"])
        definitions = [f"\\newcommand{{\\{macro.name}}}{{{macro.value}}}" for macro in macros]
        width = max(len(definition) for definition in definitions)
        for definition, macro in zip(definitions, macros, strict=True):
            lines.append(f"{definition:<{width}}  % {macro.comment}")
    return "\n".join(lines) + "\n"


def write_macro_file(
    output_path: Path = MACRO_FILE_PATH,
    *,
    figure03_path: Path = FIGURE03_SUMMARY_PATH,
    figure04_path: Path = FIGURE04_SUMMARY_PATH,
    statespacecheck_doi: str | None = None,
    citation_path: Path = CITATION_PATH,
) -> Path:
    """Write ``reported_values.tex`` from the committed figure summaries.

    Parameters
    ----------
    output_path : Path, default ``MACRO_FILE_PATH``
        Destination of the generated macro file.
    figure03_path, figure04_path : Path
        Canonical summary JSONs to read.
    statespacecheck_doi : str, optional
        Zenodo DOI of the recorded ``statespacecheck`` version. By default it is
        looked up on Zenodo (:func:`lookup_statespacecheck_doi`), which needs internet
        access.
    citation_path : Path, default ``CITATION_PATH``
        This repository's CITATION.cff, whose ``doi`` the manuscript cites.

    Returns
    -------
    Path
        The path written.
    """
    figure03_payload, figure04_payload = _load(figure03_path), _load(figure04_path)
    if statespacecheck_doi is None:
        statespacecheck_doi = lookup_statespacecheck_doi(
            statespacecheck_version(figure03_payload, figure04_payload)
        )
    text = render_macro_file(
        figure03_payload,
        figure04_payload,
        statespacecheck_doi=statespacecheck_doi,
        analysis_code_doi=analysis_code_doi(citation_path),
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
