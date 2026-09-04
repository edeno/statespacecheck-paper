r"""Emit the manuscript's reported numbers as LaTeX macros.

Every number the manuscript quotes from the analysis lives in one of the two
canonical figure summaries (``figure03_summary.json`` /
``figure04_summary.json``). This module turns those summaries into a file of
``\newcommand`` definitions that ``main.tex`` inputs, so the prose cannot drift
from the artifacts the way hand-typed numbers can.

Rounding and derivation happen here, in Python, rather than in the document:
the manuscript says "42.2 a.u." where the summary holds ``42.16926259332958``,
and "92%" where it holds ``0.9201170835550825``. Ranges such as the remap
flag percentages are emitted as separate ``\dots Min`` / ``\dots Max`` macros
so the en-dash stays in the prose.

Printed precision follows how each quantity was obtained, because the three
kinds of number here have three different error structures. Consistency is in
the *rule*, not in the digit count --- quantities with different uncertainties
should not print to the same number of digits.

**Estimates** (the Figure-3 medians over realizations, the Figure-4 rescue
rates) are reported to the decimal place of their own standard error rounded
to one significant figure, by :func:`_from_standard_error`. Those errors span
a factor of ~500 across the reported quantities, so the digit counts differ:
a remap flag percentage with an SE of 3.4 points earns whole percents, while a
history-dependence one with an SE of 0.05 earns two decimals. The Figure-3
errors are published in the summary (see
``figure03_summary.median_standard_error``) rather than chosen here; the
Figure-4 ones follow from the stored counts.

**Exact counts and configured constants** (17,289 spikes, 203 units,
``movement_var = 6.0``, a 6000-step boundary) are not estimates and carry no
error at all, so significant-figure rounding does not apply. :func:`_exact`
prints them in full and *raises* if the requested precision would lose
information, so a configuration change from ``0.88`` to ``0.875`` fails the
emit rather than quietly printing ``0.88``.

**Derived constants the text hedges with "approximately"** (199.47 Hz,
sqrt(12.5) cm) are exact functions of chosen parameters, so their digits are a
presentation choice with no uncertainty behind them: :func:`_significant`
renders them to :data:`SIGNIFICANT_FIGURES`. Working in significant figures
rather than decimal places matters here --- 199.47 Hz is 199 to zero decimals
but 200 to two significant figures, which is what the manuscript prints.

The per-macro comments in the generated file record each value's standard
error, so the printed precision can be checked without rerunning anything.

Macros are prefixed ``\Sim`` (simulation study, Figure 3) or ``\Rec``
(hippocampal recording, Figure 4). ``\newcommand`` deliberately errors on a
name clash, so a collision with a package macro fails the build rather than
silently redefining anything.

This module reads the committed summary JSONs and imports no sibling paper
module, so it stays a leaf of the dependency graph.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MACRO_FILE_PATH = Path("manuscript/reported_values.tex")
FIGURE03_SUMMARY_PATH = Path("manuscript/figures/main/figure03_summary.json")
FIGURE04_SUMMARY_PATH = Path("manuscript/figures/main/figure04_summary.json")

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

# Significant figures for derived constants the manuscript hedges with
# "approximately". These have no uncertainty --- they are exact functions of
# chosen parameters --- so their precision is a readability choice, unlike the
# estimates, whose digits come from their standard errors.
SIGNIFICANT_FIGURES = 2


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
            f"{value} is not exact to {decimals} decimal(s); use _rounded if the "
            "manuscript intends an approximation, or print more digits."
        )
    return f"{value:.{decimals}f}"


def _decimals_for_standard_error(standard_error: float) -> int:
    """Return the decimal place an estimate should be reported to.

    The textbook rule: round the standard error to one significant figure and
    report the estimate to that decimal place. A median flagged percentage with
    an SE of 3.4 points earns whole percents; one with an SE of 0.05 earns two
    decimals. Digit counts therefore differ between quantities because their
    uncertainties differ, which is the point.

    Parameters
    ----------
    standard_error : float
        Standard error of the estimate; must be positive.

    Returns
    -------
    int
        Decimal places to print.

    Raises
    ------
    ValueError
        If ``standard_error`` is not positive. A zero standard error carries no
        precision information, so the caller must decide what to print.

    Examples
    --------
    >>> _decimals_for_standard_error(3.36), _decimals_for_standard_error(0.053)
    (0, 2)
    """
    if not standard_error > 0.0:
        raise ValueError(
            f"standard_error must be positive to set a precision; got {standard_error}"
        )
    return max(0, -math.floor(math.log10(standard_error)))


def _from_standard_error(value: float, standard_error: float) -> str:
    """Render an estimate at the precision its standard error supports.

    Parameters
    ----------
    value : float
        The estimate.
    standard_error : float
        Its standard error.

    Returns
    -------
    str
        The formatted estimate.

    Examples
    --------
    >>> _from_standard_error(40.799, 3.36)
    '41'
    >>> _from_standard_error(1.764, 0.054)
    '1.76'
    """
    return f"{value:.{_decimals_for_standard_error(standard_error)}f}"


def _significant(value: float, digits: int) -> str:
    """Render a value to a stated number of significant figures.

    Used where the manuscript hedges with "approximately" or a tilde: those
    claims are significant-figure statements, not fixed-decimal ones. Rounding
    199.47 to zero decimals gives 199, but the manuscript's "approximately
    200 Hz" is the correct two-significant-figure rendering -- so the choice of
    formatter changes the printed number, not just its width.

    Parameters
    ----------
    value : float
        Value to render; must be non-zero.
    digits : int
        Significant figures to keep.

    Returns
    -------
    str
        The value in plain decimal notation.

    Examples
    --------
    >>> _significant(199.47114020071638, 2)
    '200'
    >>> _significant(0.19947114020071638, 2)
    '0.20'
    """
    exponent = math.floor(math.log10(abs(value)))
    rounded = round(value, -(exponent - digits + 1))
    return f"{rounded:.{max(0, digits - 1 - exponent)}f}"


def _load(path: Path) -> dict[str, Any]:
    """Read one summary JSON."""
    with open(path, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _flag_percentage_range(
    payload: dict[str, Any],
    condition: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the min and max median flag percentage, each with its own error.

    Parameters
    ----------
    payload : dict
        Figure-3 summary payload.
    condition : str
        Entry of ``condition_order`` naming the column to summarize.

    Returns
    -------
    low, high : tuple of float
        ``(value, standard_error)`` for the smallest and largest of the three
        diagnostic rows. The endpoints come from different metrics and so carry
        different errors; the caller prints both at the coarser precision.
    """
    column = payload["condition_order"].index(condition)
    errors = payload["median_flag_percentage_standard_errors"]
    pairs = [
        (row[column], error_row[column])
        for row, error_row in zip(payload["median_flag_percentages"], errors, strict=True)
    ]
    return min(pairs), max(pairs)


def _flag_percentage(payload: dict[str, Any], metric: str, condition: str) -> tuple[float, float]:
    """Return one metric's median flag percentage and its standard error."""
    row = payload["metric_order"].index(metric)
    column = payload["condition_order"].index(condition)
    return (
        payload["median_flag_percentages"][row][column],
        payload["median_flag_percentage_standard_errors"][row][column],
    )


def _decoding_error(payload: dict[str, Any], condition: str) -> tuple[float, float]:
    """Return the median absolute decoding error and its standard error."""
    row = payload["accuracy_metric_order"].index("median_absolute_error")
    column = payload["condition_order"].index(condition)
    return (
        payload["median_decoding_accuracy"][row][column],
        payload["median_decoding_accuracy_standard_errors"][row][column],
    )


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
        (low, low_error), (high, high_error) = _flag_percentage_range(payload, condition)
        # A range prints at one precision: the coarser of its two endpoints.
        shared_error = max(low_error, high_error)
        macros.append(
            MacroDefinition(
                f"{name}Min",
                _from_standard_error(low, shared_error),
                f"min over metrics of median_flag_percentages[:, {condition}]"
                f" [SE {shared_error:.3g}]",
            )
        )
        macros.append(
            MacroDefinition(
                f"{name}Max",
                _from_standard_error(high, shared_error),
                f"max over metrics of median_flag_percentages[:, {condition}]"
                f" [SE {shared_error:.3g}]",
            )
        )

    sparse_kl, sparse_kl_error = _flag_percentage(payload, "kl_divergence", "sparse_population")
    macros.append(
        MacroDefinition(
            "SimSparseKlFlag",
            _from_standard_error(sparse_kl, sparse_kl_error),
            f"median_flag_percentages[kl_divergence, sparse_population] [SE {sparse_kl_error:.3g}]",
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
        error, error_se = _decoding_error(payload, condition)
        macros.append(
            MacroDefinition(
                name,
                _from_standard_error(error, error_se),
                f"median_decoding_accuracy[median_absolute_error, {condition}] [SE {error_se:.3g}]",
            )
        )

    # Thresholds: the HPD and KL cutoffs are estimated from the pooled baseline,
    # so their percentile levels are themselves data-dependent choices.
    provenance = payload["threshold_provenance"]
    macros.extend(
        [
            MacroDefinition(
                "SimHpdPercentile",
                ordinal(round(provenance["hpd_overlap"]["quantile"] * 100)),
                "threshold_provenance.hpd_overlap.quantile",
            ),
            MacroDefinition(
                "SimKlPercentile",
                ordinal(round(provenance["kl_divergence"]["quantile"] * 100)),
                "threshold_provenance.kl_divergence.quantile",
            ),
            MacroDefinition(
                "SimPredictiveCutoff",
                _exact(provenance["predictive_pvalue"]["cutoff"], 2),
                "threshold_provenance.predictive_pvalue.cutoff",
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
    # Peak of the Gaussian place field, in expected spikes per 1 ms step.
    peak_count_per_step = rate_scale / (field_std * math.sqrt(2.0 * math.pi))
    # Per-cell sparse rates, converted from spikes/step to Hz at 1 ms/step.
    sparse_active_hz = config["sparse_cell_peak_rate_per_step"] * 1000.0
    sparse_baseline_hz = sparse_active_hz * config["sparse_cell_baseline_rate_fraction"]
    remap_displacement = min(abs(dst - src) for src, dst in config["place_field_remapping"])
    # The replay sweep occupies a fractional sub-window of clean recovery 2.
    recovery_two_start, recovery_two_end = boundaries[3], boundaries[4]
    recovery_two_span = recovery_two_end - recovery_two_start

    def replay_bound(fraction: float) -> int:
        return int(round(recovery_two_start + fraction * recovery_two_span))

    return [
        MacroDefinition("SimTrackLength", _exact(config["position_max"]), "position_max"),
        MacroDefinition("SimPositionMin", _exact(config["position_min"]), "position_min"),
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
            _significant(peak_count_per_step, SIGNIFICANT_FIGURES),
            "place_field_rate_scale / (place_field_std * sqrt(2 pi)) [2 s.f.]",
        ),
        MacroDefinition(
            "SimPeakRateHz",
            _significant(peak_count_per_step * 1000.0, SIGNIFICANT_FIGURES),
            "peak expected count per step at 1 ms/step, in Hz [2 s.f.]",
        ),
        MacroDefinition(
            "SimPredictionStepStd",
            _exact(config["prediction_step_std"], 1),
            "prediction_step_std",
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
            "sparse_cell_peak_rate_per_step * sparse_cell_baseline_rate_fraction, in Hz",
        ),
        MacroDefinition(
            "SimSparseActiveRateHz",
            _exact(sparse_active_hz),
            "sparse_cell_peak_rate_per_step, in Hz",
        ),
        MacroDefinition("SimDurationSteps", _exact(boundaries[-1]), "phase_boundaries[-1]"),
        MacroDefinition(
            "SimDurationSeconds",
            _exact(boundaries[-1] / 1000.0),
            "phase_boundaries[-1] at 1 ms/step",
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
            _exact(config["history_refractory_steps"]),
            "history_refractory_steps at 1 ms/step",
        ),
        MacroDefinition(
            "SimBurstStartMs",
            _exact(config["history_burst_window"][0]),
            "history_burst_window[0] at 1 ms/step",
        ),
        MacroDefinition(
            "SimBurstEndMs",
            _exact(config["history_burst_window"][1]),
            "history_burst_window[1] at 1 ms/step",
        ),
        MacroDefinition(
            "SimBurstFactorWord",
            cardinal_word(round(config["history_burst_factor"])),
            "history_burst_factor",
        ),
    ]


def _rescue_rate_standard_error(confusion: dict[str, Any]) -> float:
    """Return the standard error, in percentage points, of a rescue rate.

    The rescue rate is a proportion of the spikes the Continuous model flagged,
    so its error follows the binomial form ``sqrt(p (1 - p) / n)``. Spike events
    within a unit and across neighbouring time bins are not independent, so this
    understates the true error; it is used only to set the printed precision,
    where understating the error can at worst print one digit too many.

    Parameters
    ----------
    confusion : dict
        One ``flag_confusions`` entry from the Figure-4 summary.

    Returns
    -------
    float
        Standard error in percentage points.

    Examples
    --------
    >>> round(_rescue_rate_standard_error(
    ...     {"a_only": 17289, "both": 1501, "rescue_rate": 0.9201170835550825}), 3)
    0.198
    """
    n_flagged = confusion["a_only"] + confusion["both"]
    proportion = confusion["rescue_rate"]
    return 100.0 * math.sqrt(proportion * (1.0 - proportion) / n_flagged)


def _recording_statistics(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-4 macros computed from the hippocampal recording."""
    macros = [
        MacroDefinition("RecNUnits", _exact(payload["dataset"]["n_units"]), "dataset.n_units")
    ]
    for prefix, metric in (("RecHpd", "hpd_overlap"), ("RecPvalue", "predictive_pvalue")):
        confusion = _confusion(payload, metric)
        # Flagged by the Continuous model = rescued by ContFrag + flagged by both.
        flagged_continuous = confusion["a_only"] + confusion["both"]
        macros.extend(
            [
                MacroDefinition(
                    f"{prefix}FlaggedContinuous",
                    _exact(flagged_continuous),
                    f"flag_confusions[{metric}]: a_only + both",
                ),
                MacroDefinition(
                    f"{prefix}Rescued",
                    _exact(confusion["a_only"]),
                    f"flag_confusions[{metric}].a_only",
                ),
                MacroDefinition(
                    f"{prefix}RescuedPercent",
                    _from_standard_error(
                        100.0 * confusion["rescue_rate"],
                        _rescue_rate_standard_error(confusion),
                    ),
                    f"flag_confusions[{metric}].rescue_rate, percent "
                    f"[SE {_rescue_rate_standard_error(confusion):.3g}]",
                ),
                MacroDefinition(
                    f"{prefix}NewlyFlagged",
                    _exact(confusion["b_only"]),
                    f"flag_confusions[{metric}].b_only",
                ),
            ]
        )
    return macros


def _recording_configuration(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-4 macros recording the decoder's chosen inputs."""
    decoder = payload["configuration"]["decoder"]
    provenance = payload["configuration"]["provenance"]
    flag_rules = payload["flag_rules"]
    position_std: float = decoder["position_std"]
    return [
        MacroDefinition(
            "RecPositionBinSizeCm",
            _exact(decoder["position_bin_size_cm"]),
            "configuration.decoder.position_bin_size_cm",
        ),
        MacroDefinition(
            "RecTimeBinMs",
            _exact(1000.0 / decoder["sampling_frequency_hz"]),
            "1 / configuration.decoder.sampling_frequency_hz",
        ),
        MacroDefinition(
            "RecPositionStdVariance",
            _exact(position_std**2, 1),
            "configuration.decoder.position_std squared",
        ),
        MacroDefinition(
            "RecPositionStdCm",
            _significant(position_std, SIGNIFICANT_FIGURES),
            "configuration.decoder.position_std [2 s.f.]",
        ),
        MacroDefinition(
            "RecMovementVar",
            _exact(provenance["movement_var"], 1),
            "configuration.provenance.movement_var",
        ),
        MacroDefinition(
            "RecModeInitial",
            _exact(provenance["contfrag_discrete_initial_conditions"][0], 1),
            "configuration.provenance.contfrag_discrete_initial_conditions",
        ),
        MacroDefinition(
            "RecModeDiagonal",
            _exact(provenance["contfrag_diagonal_values"][0], 2),
            "configuration.provenance.contfrag_diagonal_values",
        ),
        MacroDefinition(
            "RecModeOffDiagonal",
            _exact(1.0 - provenance["contfrag_diagonal_values"][0], 2),
            "1 - configuration.provenance.contfrag_diagonal_values",
        ),
        MacroDefinition(
            "RecNldVersion",
            provenance["non_local_detector_version"],
            "configuration.provenance.non_local_detector_version",
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
    ]


def render_macro_file(
    figure03_payload: dict[str, Any],
    figure04_payload: dict[str, Any],
) -> str:
    """Render the full ``reported_values.tex`` contents.

    Parameters
    ----------
    figure03_payload, figure04_payload : dict
        Parsed contents of the two canonical figure summaries.

    Returns
    -------
    str
        File text, ending in a newline.
    """
    sections = (
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
    source_hash = figure03_payload["provenance"]["source"]["source_tree_sha256"]
    lines = [
        "% Generated by scripts/emit_reported_values.py --- do not edit by hand.",
        "%",
        "% Every value below is read from the canonical figure summaries:",
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
) -> Path:
    """Write ``reported_values.tex`` from the committed figure summaries.

    Parameters
    ----------
    output_path : Path, default ``MACRO_FILE_PATH``
        Destination of the generated macro file.
    figure03_path, figure04_path : Path
        Canonical summary JSONs to read.

    Returns
    -------
    Path
        The path written.
    """
    text = render_macro_file(_load(figure03_path), _load(figure04_path))
    output_path.write_text(text, encoding="utf-8")
    return output_path
