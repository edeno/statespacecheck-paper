r"""Emit the manuscript's reported numbers as LaTeX macros.

The manuscript's reported analysis statistics and configuration live in the
canonical figure summaries (``figure03_summary.json``, ``figure04_summary.json``,
the Figure-4 supplement summary, and the Figure-3 sensitivity summary). This
module turns those summaries into a file of ``\newcommand`` definitions that
``main.tex`` inputs, so the prose cannot drift from the artifacts the way
hand-typed numbers can.

Rounding and derivation happen here, in Python, rather than in the document:
the manuscript says "42 a.u." where the summary holds ``42.16926259332958``,
and "92%" where it holds ``0.9201170835550825``. Ranges such as the remap
flag percentages are emitted as separate ``\dots Min`` / ``\dots Max`` macros
so the en-dash stays in the prose.

**Prose reporting policy.** Decoding errors are reported to two significant figures,
and flag and rescue percentages to the nearest whole percent. Realized null flag
rates, rank-tail percentages, and behavior-stratified flag percentages, which are
read against nominal levels such as 1% and 5%, are reported to two significant
figures so that a realized 1.6% is not printed as the nominal 2%. Correlation
coefficients print to two decimals. Exact counts and configured parameters are
reported without approximation. Machine-readable summaries retain full numerical
precision.

Rounding is a communication choice, not a statement about uncertainty: each
rule preserves the distinctions the prose draws with the number. Decoding
errors are compared as ratios ("four times the well-specified value", "one
fifth of a place-field width"), which two significant figures support; flag
percentages are compared as substantial, low, or modest, which whole percents
support; rescue rates are descriptive of this recording and appear next to
their exact counts. Constants the text hedges with "approximately" (199.47 Hz,
sqrt(12.5) cm) are exact functions of chosen parameters and are shown to two
significant figures. Where variability matters to a claim it belongs in the
text or a figure, not in the digit count; the Figure-3 summary publishes
approximate standard errors of the aggregated medians, conditional on the
simulation setup. These do not describe the spread of individual realizations
and play no part in formatting.

Exact counts and configured parameters go through :func:`_exact`, which prints
them in full and *raises* if the requested precision would lose information,
so a configuration change from ``0.88`` to ``0.875`` fails the emit rather
than quietly printing ``0.88``.

Macros are prefixed ``\Sim`` (simulation study, Figure 3) or ``\Rec``
(hippocampal recording, Figure 4). ``\newcommand`` deliberately errors on a
name clash, so a collision with a package macro fails the build rather than
silently redefining anything.

This module reads the committed summary JSONs. Its only sibling import is
``number_format``, the rounding shared with the Figure-3 summary panel, so
the figure and the prose cannot round the same number differently.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statespacecheck_paper.number_format import SIGNIFICANT_FIGURES, significant, whole_percent

MACRO_FILE_PATH = Path("manuscript/reported_values.tex")
FIGURE03_SUMMARY_PATH = Path("manuscript/figures/main/figure03_summary.json")
FIGURE04_SUMMARY_PATH = Path("manuscript/figures/main/figure04_summary.json")
FIGURE04_SUPPLEMENT_SUMMARY_PATH = Path(
    "manuscript/figures/supplementary/figure04_supplement_summary.json"
)
FIGURE03_SENSITIVITY_SUMMARY_PATH = Path(
    "manuscript/figures/supplementary/figure03_sensitivity_summary.json"
)

# Condition identifiers of the Figure-3 summary paired with the macro stems
# the prose uses for them.
_CONDITION_STEMS: tuple[tuple[str, str], ...] = (
    ("matched_null", "MatchedNull"),
    ("recovery", "Recovery"),
    ("remap", "Remap"),
    ("history_dependent", "History"),
    ("replay", "Replay"),
    ("drift", "Drift"),
    ("reflected_map", "Reflect"),
    ("sparse_population", "Sparse"),
)
_METRIC_STEMS: tuple[tuple[str, str], ...] = (
    ("hpd_overlap", "Hpd"),
    ("predictive_pvalue", "Pvalue"),
    ("kl_divergence", "Kl"),
)

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
    "twenty-one",
    "twenty-two",
    "twenty-three",
    "twenty-four",
    "twenty-five",
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


def _load(path: Path) -> dict[str, Any]:
    """Read one summary JSON."""
    with open(path, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _flag_percentage_range(
    payload: dict[str, Any], condition: str, key: str = "median_flag_percentages"
) -> tuple[float, float]:
    """Return the min and max flag percentage across the three metrics.

    Parameters
    ----------
    payload : dict
        Figure-3 summary payload.
    condition : str
        Entry of ``condition_order`` naming the column to summarize.
    key : str
        Which percentage matrix to read (median or pooled).

    Returns
    -------
    tuple of float
        ``(minimum, maximum)`` over the three diagnostic rows.
    """
    column = payload["condition_order"].index(condition)
    values = [row[column] for row in payload[key]]
    return min(values), max(values)


def _flag_percentage(
    payload: dict[str, Any], metric: str, condition: str, key: str = "median_flag_percentages"
) -> float:
    """Return one metric's flag percentage for one condition."""
    row = payload["metric_order"].index(metric)
    column = payload["condition_order"].index(condition)
    percentage: float = payload[key][row][column]
    return percentage


def _accuracy(payload: dict[str, Any], metric: str, condition: str) -> float:
    """Return one accuracy row's median value for one condition."""
    row = payload["accuracy_metric_order"].index(metric)
    column = payload["condition_order"].index(condition)
    value: float = payload["median_decoding_accuracy"][row][column]
    return value


def _confusion(payload: dict[str, Any], metric: str) -> dict[str, Any]:
    """Return the Figure-4 flag-confusion entry for one metric."""
    for confusion in payload["flag_confusions"]:
        if confusion["metric"] == metric:
            return dict(confusion)
    raise KeyError(f"figure04_summary.json has no flag_confusions entry for {metric!r}")


def _percent_list(values: list[float]) -> str:
    r"""Render percentages as ``a\%, b\%, and c\%`` at two significant figures."""
    rendered = [f"{significant(v, SIGNIFICANT_FIGURES)}\\%" for v in values]
    return ", ".join(rendered[:-1]) + ", and " + rendered[-1]


def _simulation_statistics(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the Figure-3 macros computed from the simulated data."""
    calibration = payload["calibration"]
    macros: list[MacroDefinition] = [
        MacroDefinition(
            "SimNRealizations", _exact(payload["realizations"]["count"]), "realizations.count"
        ),
        MacroDefinition(
            "SimFirstSeed", _exact(payload["realizations"]["first_seed"]), "realizations.first_seed"
        ),
        MacroDefinition(
            "SimLastSeed", _exact(payload["realizations"]["last_seed"]), "realizations.last_seed"
        ),
        MacroDefinition("SimNCalibration", _exact(calibration["count"]), "calibration.count"),
        MacroDefinition(
            "SimCalibrationSeedFirst", _exact(calibration["first_seed"]), "calibration.first_seed"
        ),
        MacroDefinition(
            "SimCalibrationSeedLast", _exact(calibration["last_seed"]), "calibration.last_seed"
        ),
        MacroDefinition(
            "SimCalibrationSteps",
            _exact(calibration["steps_per_session"]),
            "calibration.steps_per_session",
        ),
        MacroDefinition(
            "SimCalibrationEvents",
            _exact(calibration["n_pooled_events"]),
            "calibration.n_pooled_events",
        ),
        MacroDefinition(
            "SimHpdThreshold",
            _exact(payload["flag_rules"]["hpd_overlap"]["threshold"]),
            "flag_rules.hpd_overlap.threshold",
        ),
        MacroDefinition(
            "SimKlThreshold",
            significant(payload["flag_rules"]["kl_divergence"]["threshold"], SIGNIFICANT_FIGURES),
            "flag_rules.kl_divergence.threshold",
        ),
        MacroDefinition(
            "SimCalibrationTail",
            _percent_list(calibration["rank_pvalue_tail_percentages"]),
            "calibration.rank_pvalue_tail_percentages at alphas 0.01, 0.05, 0.10",
        ),
        MacroDefinition(
            "SimMatchedNullTail",
            _percent_list(payload["matched_null_rank_pvalue_tail_percentages"]),
            "matched_null_rank_pvalue_tail_percentages at alphas 0.01, 0.05, 0.10",
        ),
    ]
    for metric, stem in _METRIC_STEMS:
        macros.append(
            MacroDefinition(
                f"SimNull{stem}Flag",
                significant(
                    calibration["pooled_null_flag_percentages"][metric], SIGNIFICANT_FIGURES
                ),
                f"calibration.pooled_null_flag_percentages.{metric}",
            )
        )

    # Flag-percentage ranges across the three diagnostics, one condition per row,
    # plus per-metric values for the reference and sparse columns.
    for condition, stem in _CONDITION_STEMS:
        if condition == "sparse_population":
            continue  # quoted per metric below, not as a range
        low, high = _flag_percentage_range(payload, condition)
        macros.append(
            MacroDefinition(
                f"Sim{stem}FlagMin",
                whole_percent(low),
                f"min over metrics of median_flag_percentages[:, {condition}]",
            )
        )
        macros.append(
            MacroDefinition(
                f"Sim{stem}FlagMax",
                whole_percent(high),
                f"max over metrics of median_flag_percentages[:, {condition}]",
            )
        )
    macros.append(
        MacroDefinition(
            "SimRemapHpdFlag",
            whole_percent(_flag_percentage(payload, "hpd_overlap", "remap")),
            "median_flag_percentages[hpd_overlap, remap]",
        )
    )
    for condition, stem in (("matched_null", "MatchedNull"), ("sparse_population", "Sparse")):
        for metric, metric_stem in _METRIC_STEMS:
            macros.append(
                MacroDefinition(
                    f"Sim{stem}{metric_stem}Flag",
                    significant(_flag_percentage(payload, metric, condition), SIGNIFICANT_FIGURES),
                    f"median_flag_percentages[{metric}, {condition}]",
                )
            )
    for metric, metric_stem in (("hpd_overlap", "Hpd"), ("predictive_pvalue", "Pvalue")):
        macros.append(
            MacroDefinition(
                f"SimSparsePooled{metric_stem}Flag",
                significant(
                    _flag_percentage(
                        payload, metric, "sparse_population", "pooled_flag_percentages"
                    ),
                    SIGNIFICANT_FIGURES,
                ),
                f"pooled_flag_percentages[{metric}, sparse_population]",
            )
        )
    low, high = _flag_percentage_range(payload, "remap", "pooled_flag_percentages")
    macros.append(
        MacroDefinition(
            "SimRemapPooledMin",
            whole_percent(low),
            "min over metrics of pooled_flag_percentages[:, remap]",
        )
    )
    macros.append(
        MacroDefinition(
            "SimRemapPooledMax",
            whole_percent(high),
            "max over metrics of pooled_flag_percentages[:, remap]",
        )
    )
    remap = payload["condition_order"].index("remap")
    hpd_row = payload["metric_order"].index("hpd_overlap")
    remap_hpd = [
        realization[hpd_row][remap]
        for realization in payload["flag_percentages_by_realization"]
        if realization[hpd_row][remap] is not None
    ]
    remap_hpd.sort()

    def quartile(q: float) -> float:
        """Nearest-rank quantile of the sorted per-realization remap percentages."""
        value: float = remap_hpd[min(len(remap_hpd) - 1, max(0, round(q * (len(remap_hpd) - 1))))]
        return value

    macros.append(
        MacroDefinition(
            "SimRemapHpdQLow",
            whole_percent(quartile(0.25)),
            "25th percentile over realizations of "
            "flag_percentages_by_realization[hpd_overlap, remap]",
        )
    )
    macros.append(
        MacroDefinition(
            "SimRemapHpdQHigh",
            whole_percent(quartile(0.75)),
            "75th percentile over realizations of "
            "flag_percentages_by_realization[hpd_overlap, remap]",
        )
    )

    for condition, stem in _CONDITION_STEMS:
        macros.append(
            MacroDefinition(
                f"Sim{stem}Error",
                significant(
                    _accuracy(payload, "median_absolute_error", condition), SIGNIFICANT_FIGURES
                ),
                f"median_decoding_accuracy[median_absolute_error, {condition}]",
            )
        )
        macros.append(
            MacroDefinition(
                f"Sim{stem}Coverage",
                whole_percent(_accuracy(payload, "filtered_hpd_coverage_percent", condition)),
                f"median_decoding_accuracy[filtered_hpd_coverage_percent, {condition}]",
            )
        )
    # Region sizes the prose quotes: the remap's filtered region and the
    # predictive region in the matched null and the sparse regime.
    for condition, stem, metric, metric_stem in (
        ("remap", "Remap", "median_filtered_hpd_size", "FilteredSize"),
        ("matched_null", "MatchedNull", "median_predictive_hpd_size", "PredictiveSize"),
        ("sparse_population", "Sparse", "median_predictive_hpd_size", "PredictiveSize"),
    ):
        macros.append(
            MacroDefinition(
                f"Sim{stem}{metric_stem}",
                significant(_accuracy(payload, metric, condition), SIGNIFICANT_FIGURES),
                f"median_decoding_accuracy[{metric}, {condition}]",
            )
        )
    represented = payload["replay_represented_accuracy"]["median"]
    macros.append(
        MacroDefinition(
            "SimReplayRepresentedError",
            significant(represented[0], SIGNIFICANT_FIGURES),
            "replay_represented_accuracy.median[median_absolute_error]",
        )
    )
    macros.append(
        MacroDefinition(
            "SimReplayRepresentedCoverage",
            whole_percent(represented[1]),
            "replay_represented_accuracy.median[filtered_hpd_coverage_percent]",
        )
    )
    rates = payload["phase_ordinary_spike_rates_hz"]["median"]
    macros.append(
        MacroDefinition(
            "SimMatchedNullRateHz",
            significant(rates[payload["condition_order"].index("matched_null")], 3),
            "phase_ordinary_spike_rates_hz.median[matched_null]",
        )
    )
    macros.append(
        MacroDefinition(
            "SimHistoryRateHz",
            significant(rates[payload["condition_order"].index("history_dependent")], 3),
            "phase_ordinary_spike_rates_hz.median[history_dependent]",
        )
    )
    sparse_counts = payload["sparse_population"]["sparse_cell_event_counts_by_realization"]
    macros.append(
        MacroDefinition(
            "SimSparseEventsMedian",
            _exact(payload["sparse_population"]["median_sparse_cell_event_count"]),
            "sparse_population.median_sparse_cell_event_count",
        )
    )
    macros.append(
        MacroDefinition(
            "SimSparseEventsMin",
            _exact(min(sparse_counts)),
            "min of sparse_population.sparse_cell_event_counts_by_realization",
        )
    )
    macros.append(
        MacroDefinition(
            "SimSparseEventsMax",
            _exact(max(sparse_counts)),
            "max of sparse_population.sparse_cell_event_counts_by_realization",
        )
    )

    # Thresholds: the HPD and KL cutoff values are estimated from the pooled
    # calibration events at fixed percentile levels. The prose writes those
    # levels as ordinals ("1st percentile"), which has no faithful form for a
    # fractional level, so a configured 0.005 must fail the emit rather than
    # round to "0th".
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
    # Reference sparse peak rate, converted from spikes/step to Hz at 1 ms/step.
    sparse_active_hz = config["sparse_cell_peak_rate_per_step"] * 1000.0
    multipliers = config["sparse_cell_rate_multipliers"]
    sparse_centers = config["sparse_place_field_centers"]
    remap_displacement = min(abs(dst - src) for src, dst in config["place_field_remapping"])
    # The replay sweep occupies a fractional sub-window of clean recovery 2.
    recovery_two_start, recovery_two_end = boundaries[3], boundaries[4]
    recovery_two_span = recovery_two_end - recovery_two_start
    # The prose spells these values as words ("threefold", "fourfold"), so a
    # fractional factor cannot be represented faithfully. Reuse the exact-value
    # guard rather than silently rounding a valid simulation setting.
    burst_factor_word = cardinal_word(int(_exact(config["history_burst_factor"])))
    replay_gain_word = cardinal_word(
        int(_exact(config["replay_place_field_rate_scale"] / rate_scale))
    )
    # Stationary standard deviation of the AR(1) drift velocity.
    momentum = config["drift_momentum"]
    drift_velocity_std = config["prediction_step_std"] / math.sqrt(1.0 - momentum**2)

    def replay_bound(fraction: float) -> int:
        return int(round(recovery_two_start + fraction * recovery_two_span))

    replay_steps = replay_bound(config["replay_end_fraction"]) - replay_bound(
        config["replay_start_fraction"]
    )
    # The out-and-back sweep's largest possible per-step displacement: a full
    # track length covered on the shorter leg of the sweep.
    replay_max_speed = (config["position_max"] - config["position_min"]) / ((replay_steps - 1) // 2)

    return [
        MacroDefinition("SimTrackLength", _exact(config["position_max"]), "position_max"),
        MacroDefinition("SimPositionMin", _exact(config["position_min"]), "position_min"),
        MacroDefinition(
            "SimNPlaceCellsWord", cardinal_word(len(centers)), "len(place_field_centers)"
        ),
        MacroDefinition(
            "SimNNeuronsWord",
            cardinal_word(len(centers) + len(sparse_centers)),
            "len(place_field_centers) + len(sparse_place_field_centers)",
        ),
        MacroDefinition(
            "SimPlaceFieldSpacing", _exact(centers[1] - centers[0]), "place_field_centers spacing"
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
            significant(peak_count_per_step * 1000.0, SIGNIFICANT_FIGURES),
            "peak expected count per step at 1 ms/step, in Hz",
        ),
        MacroDefinition(
            "SimPredictionStepStd", _exact(config["prediction_step_std"], 1), "prediction_step_std"
        ),
        MacroDefinition(
            "SimNSparseCellsWord",
            cardinal_word(len(sparse_centers)),
            "len(sparse_place_field_centers)",
        ),
        MacroDefinition(
            "SimNSparseCellsWordCap",
            cardinal_word(len(sparse_centers)).capitalize(),
            "len(sparse_place_field_centers), sentence-initial",
        ),
        MacroDefinition(
            "SimSparseFieldStd", _exact(config["sparse_place_field_std"]), "sparse_place_field_std"
        ),
        MacroDefinition(
            "SimSparseActiveRateHz",
            _exact(sparse_active_hz),
            "sparse_cell_peak_rate_per_step, in Hz",
        ),
        MacroDefinition(
            "SimSparseMultiplierMin",
            _exact(min(multipliers), 1),
            "min(sparse_cell_rate_multipliers)",
        ),
        MacroDefinition(
            "SimSparseMultiplierMax", _exact(max(multipliers)), "max(sparse_cell_rate_multipliers)"
        ),
        MacroDefinition(
            "SimSparseBaselineFractionPercent",
            _exact(100.0 * config["sparse_cell_baseline_rate_fraction"]),
            "100 * sparse_cell_baseline_rate_fraction",
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
        MacroDefinition("SimReflectEnd", _exact(boundaries[7]), "phase_boundaries[7]"),
        MacroDefinition("SimRecoveryFourEnd", _exact(boundaries[8]), "phase_boundaries[8]"),
        MacroDefinition("SimSparseEnd", _exact(boundaries[9]), "phase_boundaries[9]"),
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
            "SimReplayRateScale",
            _exact(config["replay_place_field_rate_scale"]),
            "replay_place_field_rate_scale",
        ),
        MacroDefinition(
            "SimReplayGainWord",
            replay_gain_word,
            "replay_place_field_rate_scale / place_field_rate_scale",
        ),
        MacroDefinition(
            "SimReplaySpeedCap", _exact(config["replay_speed_per_step"], 1), "replay_speed_per_step"
        ),
        MacroDefinition(
            "SimReplayMaxSpeed",
            significant(replay_max_speed, SIGNIFICANT_FIGURES),
            "track length / transitions per sweep leg",
        ),
        MacroDefinition(
            "SimRemapDisplacementWord",
            cardinal_word(remap_displacement),
            "min displacement in place_field_remapping",
        ),
        MacroDefinition("SimDriftMomentum", _exact(momentum, 2), "drift_momentum"),
        MacroDefinition(
            "SimDriftVelocityStd",
            significant(drift_velocity_std, SIGNIFICANT_FIGURES),
            "prediction_step_std / sqrt(1 - drift_momentum^2)",
        ),
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
        MacroDefinition("SimBurstFactorWord", burst_factor_word, "history_burst_factor"),
        MacroDefinition(
            "SimHistoryGain",
            _exact(config["history_rate_matching_gain"], 3),
            "history_rate_matching_gain",
        ),
    ]


def _sensitivity_macros(
    payload: dict[str, Any], main_payload: dict[str, Any]
) -> list[MacroDefinition]:
    """Build the Figure-3 sensitivity macros and table rows.

    The coverage table's first row is the main setting, read from the
    canonical Figure-3 summary so the comparison rows share its source.
    """
    settings = payload["settings"]
    order = payload["condition_order"]
    main_setting = {
        "summary": {
            "median_flag_percentages": main_payload["median_flag_percentages"],
            "metric_order": main_payload["metric_order"],
            "calibration": main_payload["calibration"],
            "flag_rules": main_payload["flag_rules"],
        }
    }

    def row_value(setting: dict[str, Any], metric: str, condition: str) -> float:
        summary = setting["summary"]
        value: float = summary["median_flag_percentages"][summary["metric_order"].index(metric)][
            order.index(condition)
        ]
        return value

    def null_rate(setting: dict[str, Any], metric: str) -> float:
        value: float = setting["summary"]["calibration"]["pooled_null_flag_percentages"][metric]
        return value

    coverage_rows: list[str] = []
    for setting_id, label in (
        ("main", "95\\% coverage (main setting)"),
        ("hpd_coverage_0.5", "50\\% coverage"),
        ("hpd_coverage_0.8", "80\\% coverage"),
        ("continuous_reflected_trajectory", "Reflected continuous walk (95\\%)"),
    ):
        setting = main_setting if setting_id == "main" else settings[setting_id]
        threshold = setting["summary"]["flag_rules"]["hpd_overlap"]["threshold"]
        coverage_rows.append(
            " & ".join(
                [
                    label,
                    _exact(threshold, 2),
                    significant(null_rate(setting, "hpd_overlap"), SIGNIFICANT_FIGURES),
                    significant(null_rate(setting, "predictive_pvalue"), SIGNIFICANT_FIGURES),
                    significant(null_rate(setting, "kl_divergence"), SIGNIFICANT_FIGURES),
                    whole_percent(row_value(setting, "hpd_overlap", "remap")),
                    whole_percent(row_value(setting, "hpd_overlap", "drift")),
                    whole_percent(row_value(setting, "kl_divergence", "sparse_population")),
                ]
            )
            + " \\\\"
        )
    sparse_rows: list[str] = []
    scales: list[str] = []
    sparse_ids = sorted(
        (sid for sid in settings if sid.startswith("sparse_ordinary_scale_")),
        key=lambda sid: float(
            settings[sid]["changed_configuration"]["sparse_control_ordinary_rate_scale"]
        ),
    ) + ["sparse_lower_rate"]
    for setting_id in sparse_ids:
        setting = settings[setting_id]
        summary = setting["summary"]
        sparse_col = order.index("sparse_population")
        counts = [r[sparse_col] for r in summary["event_counts_by_realization"]]
        counts.sort()
        median_count = counts[len(counts) // 2]
        sparse_median = summary["sparse_population"]["median_sparse_cell_event_count"]
        sparse_p = summary["sparse_population"]["median_sparse_cell_flag_percentages"][
            summary["metric_order"].index("predictive_pvalue")
        ]
        if setting_id.startswith("sparse_ordinary_scale_"):
            scale = setting["changed_configuration"]["sparse_control_ordinary_rate_scale"]
            scales.append(f"{scale:g}")
            label = f"Ordinary ensemble at {scale:g} of its rate"
        else:
            label = "Sparse cells at half rate"
        sparse_rows.append(
            " & ".join(
                [
                    label,
                    _exact(median_count),
                    _exact(sparse_median),
                    whole_percent(row_value(setting, "predictive_pvalue", "sparse_population")),
                    "--" if sparse_p is None else whole_percent(sparse_p),
                    whole_percent(row_value(setting, "hpd_overlap", "sparse_population")),
                    whole_percent(row_value(setting, "kl_divergence", "sparse_population")),
                ]
            )
            + " \\\\"
        )
    return [
        MacroDefinition(
            "SimSensitivityCoverageRows",
            " ".join(coverage_rows),
            "figure03_sensitivity_summary settings",
        ),
        MacroDefinition(
            "SimSensitivitySparseRows",
            " ".join(sparse_rows),
            "figure03_sensitivity_summary sparse settings",
        ),
        MacroDefinition(
            "SimSparseOrdinaryScales",
            ", ".join(scales[:-1]) + ", and " + scales[-1],
            "sparse_control_ordinary_rate_scale sensitivity values",
        ),
        MacroDefinition(
            "SensCoverageFiftyNullHpd",
            significant(
                null_rate(settings["hpd_coverage_0.5"], "hpd_overlap"), SIGNIFICANT_FIGURES
            ),
            "settings[hpd_coverage_0.5].calibration.pooled_null_flag_percentages.hpd_overlap",
        ),
        MacroDefinition(
            "SensContinuousRemapHpd",
            whole_percent(
                row_value(settings["continuous_reflected_trajectory"], "hpd_overlap", "remap")
            ),
            "settings[continuous_reflected_trajectory].median_flag_percentages[hpd_overlap, remap]",
        ),
        MacroDefinition(
            "SensCoverageEightyNullHpd",
            significant(
                null_rate(settings["hpd_coverage_0.8"], "hpd_overlap"), SIGNIFICANT_FIGURES
            ),
            "settings[hpd_coverage_0.8].calibration.pooled_null_flag_percentages.hpd_overlap",
        ),
    ]


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
                    whole_percent(100.0 * confusion["rescue_rate"]),
                    f"flag_confusions[{metric}].rescue_rate",
                ),
                MacroDefinition(
                    f"{prefix}NewlyFlagged",
                    _exact(confusion["b_only"]),
                    f"flag_confusions[{metric}].b_only",
                ),
            ]
        )
    hpd = _confusion(payload, "hpd_overlap")
    macros.append(MacroDefinition("RecNEvents", _exact(hpd["n"]), "flag_confusions[hpd_overlap].n"))
    macros.append(
        MacroDefinition(
            "RecHpdFlaggedContinuousPercent",
            significant(100.0 * (hpd["a_only"] + hpd["both"]) / hpd["n"], SIGNIFICANT_FIGURES),
            "flag_confusions[hpd_overlap]: (a_only + both) / n",
        )
    )
    return macros


def _recording_supplement(payload: dict[str, Any]) -> list[MacroDefinition]:
    """Build the macros computed by the Figure-4 supplement analyses."""
    broadening = payload["region_size_and_broadening"]
    rate = payload["rate_and_behavior"]
    by_coverage = broadening["by_coverage"]
    main = by_coverage["0.95"]
    n_bins = main["n_position_bins"]

    def median(record: dict[str, Any]) -> float:
        """Return the stored median of a quantile record."""
        value: float = record["quantiles"][record["quantile_probabilities"].index(0.5)]
        return value

    likelihood_median = median(main["likelihood_region_size_all_events"])
    macros = [
        MacroDefinition(
            "RecDurationMinutes",
            significant(rate["duration_seconds"] / 60.0, SIGNIFICANT_FIGURES),
            "rate_and_behavior.duration_seconds / 60",
        ),
        MacroDefinition(
            "RecNActiveUnits", _exact(rate["n_active_units"]), "rate_and_behavior.n_active_units"
        ),
        MacroDefinition(
            "RecNZeroEventUnits",
            _exact(rate["n_zero_event_units"]),
            "rate_and_behavior.n_zero_event_units",
        ),
        MacroDefinition(
            "RecNPositionBins",
            _exact(n_bins),
            "region_size_and_broadening.by_coverage[0.95].n_position_bins",
        ),
        MacroDefinition(
            "RecHpdCutoffPercent",
            _exact(100.0 * broadening["hpd_flag_threshold"]),
            "100 * region_size_and_broadening.hpd_flag_threshold",
        ),
        MacroDefinition(
            "RecContRegionAll",
            _exact(median(main["continuous"]["region_size_all_events"])),
            "median 95% Continuous predictive region size, all events",
        ),
        MacroDefinition(
            "RecContRegionRescued",
            _exact(median(main["continuous"]["region_size_rescued_events"])),
            "median 95% Continuous predictive region size, rescued events",
        ),
        MacroDefinition(
            "RecFragRegionAll",
            _exact(median(main["continuous_fragmented"]["region_size_all_events"])),
            "median 95% Continuous-Fragmented predictive region size, all events",
        ),
        MacroDefinition(
            "RecFragRegionRescued",
            _exact(median(main["continuous_fragmented"]["region_size_rescued_events"])),
            "median 95% Continuous-Fragmented predictive region size, rescued events",
        ),
        MacroDefinition(
            "RecLikelihoodRegion",
            _exact(likelihood_median),
            "median 95% likelihood region size, all events",
        ),
        MacroDefinition(
            "RecLikelihoodOutsidePercent",
            whole_percent(100.0 * (1.0 - likelihood_median / n_bins)),
            "100 * (1 - median likelihood region / n_position_bins)",
        ),
    ]
    mixtures = main["uniform_mixture"]
    weights = [w for w in sorted(mixtures, key=float) if float(w) < 1.0]
    new_flags = []
    for letter, key in zip("ABCD", weights[:4], strict=True):
        record = mixtures[key]
        macros.append(
            MacroDefinition(
                f"RecMixWeight{letter}",
                _exact(record["uniform_weight"], 2),
                f"uniform_mixture[{key}].uniform_weight",
            )
        )
        macros.append(
            MacroDefinition(
                f"RecMixRemoved{letter}",
                whole_percent(100.0 * record["fraction_original_flags_removed"]),
                f"uniform_mixture[{key}].fraction_original_flags_removed",
            )
        )
        new_flags.append(record["n_new_flags"])
    macros.append(
        MacroDefinition(
            "RecMixNewMax", _exact(max(new_flags)), "max over mixture weights < 1 of n_new_flags"
        )
    )
    for key, stem in (("0.8", "Eighty"), ("0.5", "Fifty")):
        record = by_coverage[key]
        macros.append(
            MacroDefinition(
                f"RecCov{stem}ContFlagged",
                _exact(record["continuous"]["n_flagged"]),
                f"by_coverage[{key}].continuous.n_flagged",
            )
        )
        if key == "0.5":
            macros.append(
                MacroDefinition(
                    f"RecCov{stem}ContFlaggedPercent",
                    whole_percent(100.0 * record["continuous"]["n_flagged"] / record["n_events"]),
                    f"by_coverage[{key}].continuous.n_flagged / n_events",
                )
            )
        macros.append(
            MacroDefinition(
                f"RecCov{stem}RescuedPercent",
                whole_percent(100.0 * record["rescue_fraction"]),
                f"by_coverage[{key}].rescue_fraction",
            )
        )
    table_rows = []
    for key in ("0.95", "0.8", "0.5"):
        record = by_coverage[key]
        rescued_percent = whole_percent(100.0 * record["rescue_fraction"])
        table_rows.append(
            " & ".join(
                [
                    f"{int(round(100 * record['coverage']))}\\%",
                    f"\\num{{{record['continuous']['n_flagged']}}}",
                    f"\\num{{{record['continuous_fragmented']['n_flagged']}}}",
                    f"\\num{{{record['n_rescued']}}} ({rescued_percent}\\%)",
                    f"\\num{{{record['n_newly_flagged']}}}",
                    _exact(median(record["continuous"]["region_size_all_events"])),
                    _exact(median(record["continuous_fragmented"]["region_size_all_events"])),
                ]
            )
            + " \\\\"
        )
    macros.append(
        MacroDefinition("RecBroadeningTableRows", " ".join(table_rows), "by_coverage rows")
    )

    behavior = rate["behavior"]
    macros.append(
        MacroDefinition(
            "RecSpeedCutoff",
            _exact(behavior["speed_cutoff_cm_s"]),
            "rate_and_behavior.behavior.speed_cutoff_cm_s",
        )
    )
    macros.append(
        MacroDefinition(
            "RecImmobileEvents",
            _exact(behavior["immobile"]["n_events"]),
            "behavior.immobile.n_events",
        )
    )
    macros.append(
        MacroDefinition(
            "RecMovingEvents", _exact(behavior["moving"]["n_events"]), "behavior.moving.n_events"
        )
    )
    for stratum, stratum_stem in (("immobile", "Immobile"), ("moving", "Moving")):
        for model, model_stem in (("continuous", "Cont"), ("continuous_fragmented", "Frag")):
            for metric, metric_stem in (("hpd_overlap", "Hpd"), ("predictive_pvalue", "Pvalue")):
                macros.append(
                    MacroDefinition(
                        f"Rec{stratum_stem}{metric_stem}{model_stem}",
                        significant(
                            100.0 * behavior[stratum][f"{model}_{metric}_flag_fraction"],
                            SIGNIFICANT_FIGURES,
                        ),
                        f"behavior.{stratum}.{model}_{metric}_flag_fraction",
                    )
                )
    cont = rate["continuous"]
    frag = rate["continuous_fragmented"]
    macros.extend(
        [
            MacroDefinition(
                "RecRateSpearmanPvalueCont",
                f"{cont['predictive_pvalue']['spearman_rate_vs_unit_flag_fraction_active_units']:.2f}",
                "continuous.predictive_pvalue.spearman_rate_vs_unit_flag_fraction_active_units",
            ),
            MacroDefinition(
                "RecRateSpearmanPvalueFrag",
                f"{frag['predictive_pvalue']['spearman_rate_vs_unit_flag_fraction_active_units']:.2f}",
                "continuous_fragmented.predictive_pvalue.spearman_rate_vs_unit_flag_fraction_active_units",
            ),
            MacroDefinition(
                "RecRateSpearmanHpdCont",
                f"{cont['hpd_overlap']['spearman_rate_vs_unit_flag_fraction_active_units']:.2f}",
                "continuous.hpd_overlap.spearman_rate_vs_unit_flag_fraction_active_units",
            ),
            MacroDefinition(
                "RecAlwaysFlaggedPvalueCont",
                _exact(cont["predictive_pvalue"]["n_active_units_all_events_flagged"]),
                "continuous.predictive_pvalue.n_active_units_all_events_flagged",
            ),
            MacroDefinition(
                "RecLowRateThresholdHz",
                _exact(rate["low_rate_threshold_hz"], 1),
                "rate_and_behavior.low_rate_threshold_hz",
            ),
            MacroDefinition(
                "RecLowRateActiveUnits",
                _exact(rate["n_low_rate_active_units"]),
                "rate_and_behavior.n_low_rate_active_units",
            ),
            MacroDefinition(
                "RecLowRateEvents",
                _exact(rate["n_low_rate_events"]),
                "rate_and_behavior.n_low_rate_events",
            ),
            MacroDefinition(
                "RecLowRateFlagsPvalueCont",
                significant(
                    100.0 * cont["predictive_pvalue"]["fraction_of_flags_from_low_rate_units"],
                    SIGNIFICANT_FIGURES,
                ),
                "continuous.predictive_pvalue.fraction_of_flags_from_low_rate_units",
            ),
        ]
    )
    groups = {g["label_hz"]: g for g in rate["rate_groups"]}
    under_one = [g for label, g in groups.items() if label.startswith("<") or label == "0.1-1"]
    one_to_five = groups["1-5"]
    macros.extend(
        [
            MacroDefinition(
                "RecUnderOneHzFlagsPvalueCont",
                whole_percent(
                    100.0
                    * sum(g["continuous_predictive_pvalue_fraction_of_flags"] for g in under_one)
                ),
                "rate_groups (<1 Hz): continuous_predictive_pvalue_fraction_of_flags",
            ),
            MacroDefinition(
                "RecUnderOneHzEventsPercent",
                significant(
                    100.0 * sum(g["fraction_of_events"] for g in under_one), SIGNIFICANT_FIGURES
                ),
                "rate_groups (<1 Hz): fraction_of_events",
            ),
            MacroDefinition(
                "RecOneToFiveEventsPercent",
                whole_percent(100.0 * one_to_five["fraction_of_events"]),
                "rate_groups[1-5].fraction_of_events",
            ),
            MacroDefinition(
                "RecOneToFiveFlagsPvalueCont",
                whole_percent(
                    100.0 * one_to_five["continuous_predictive_pvalue_fraction_of_flags"]
                ),
                "rate_groups[1-5].continuous_predictive_pvalue_fraction_of_flags",
            ),
        ]
    )
    constant = rate["constant_frequency_baseline"]
    changes = rate["rank_flag_changes"]
    macros.extend(
        [
            MacroDefinition(
                "RecConstantFlagPercent",
                significant(100.0 * constant["flag_fraction"], SIGNIFICANT_FIGURES),
                "constant_frequency_baseline.flag_fraction",
            ),
            MacroDefinition(
                "RecConstantOverlapCont",
                whole_percent(
                    100.0 * constant["continuous_fraction_of_actual_flags_also_constant_flagged"]
                ),
                "constant_frequency_baseline.continuous_fraction_of_actual_flags_also_constant_flagged",
            ),
            MacroDefinition(
                "RecConstantOverlapFrag",
                whole_percent(
                    100.0
                    * constant[
                        "continuous_fragmented_fraction_of_actual_flags_also_constant_flagged"
                    ]
                ),
                "constant_frequency_baseline.continuous_fragmented_fraction_of_actual_flags_also_constant_flagged",
            ),
            MacroDefinition(
                "RecRankRescuedLowRate",
                _exact(changes["n_rescued_from_low_rate_units"]),
                "rank_flag_changes.n_rescued_from_low_rate_units",
            ),
            MacroDefinition(
                "RecRankNewLowRate",
                _exact(changes["n_newly_flagged_from_low_rate_units"]),
                "rank_flag_changes.n_newly_flagged_from_low_rate_units",
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
    continuous_initial, fragmented_initial = provenance["contfrag_discrete_initial_conditions"]
    continuous_diagonal, fragmented_diagonal = provenance["contfrag_diagonal_values"]
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
            significant(position_std, SIGNIFICANT_FIGURES),
            "configuration.decoder.position_std",
        ),
        MacroDefinition(
            "RecMovementVar",
            _exact(provenance["movement_var"], 1),
            "configuration.provenance.movement_var",
        ),
        MacroDefinition(
            "RecModeContinuousInitial",
            _exact(continuous_initial, 1),
            "configuration.provenance.contfrag_discrete_initial_conditions[0]",
        ),
        MacroDefinition(
            "RecModeFragmentedInitial",
            _exact(fragmented_initial, 1),
            "configuration.provenance.contfrag_discrete_initial_conditions[1]",
        ),
        MacroDefinition(
            "RecModeContinuousStay",
            _exact(continuous_diagonal, 2),
            "configuration.provenance.contfrag_diagonal_values[0]",
        ),
        MacroDefinition(
            "RecModeContinuousToFragmented",
            _exact(1.0 - continuous_diagonal, 2),
            "1 - configuration.provenance.contfrag_diagonal_values[0]",
        ),
        MacroDefinition(
            "RecModeFragmentedToContinuous",
            _exact(1.0 - fragmented_diagonal, 2),
            "1 - configuration.provenance.contfrag_diagonal_values[1]",
        ),
        MacroDefinition(
            "RecModeFragmentedStay",
            _exact(fragmented_diagonal, 2),
            "configuration.provenance.contfrag_diagonal_values[1]",
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
    figure04_supplement_payload: dict[str, Any],
    figure03_sensitivity_payload: dict[str, Any],
) -> str:
    """Render the full ``reported_values.tex`` contents.

    Parameters
    ----------
    figure03_payload, figure04_payload : dict
        Parsed contents of the two canonical main-figure summaries.
    figure04_supplement_payload, figure03_sensitivity_payload : dict
        Parsed contents of the Figure-4 supplement and Figure-3 sensitivity
        summaries.

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
            "Simulation study (Figure 3) --- sensitivity settings",
            _sensitivity_macros(figure03_sensitivity_payload, figure03_payload),
        ),
        (
            "Hippocampal recording (Figure 4) --- computed from the recording",
            _recording_statistics(figure04_payload),
        ),
        (
            "Hippocampal recording (Figure 4 supplement) --- computed from the recording",
            _recording_supplement(figure04_supplement_payload),
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
        "%   figures/supplementary/figure04_supplement_summary.json "
        f"(schema {figure04_supplement_payload['schema_version']})",
        "%   figures/supplementary/figure03_sensitivity_summary.json "
        f"(schema {figure03_sensitivity_payload['schema_version']})",
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
    figure04_supplement_path: Path = FIGURE04_SUPPLEMENT_SUMMARY_PATH,
    figure03_sensitivity_path: Path = FIGURE03_SENSITIVITY_SUMMARY_PATH,
) -> Path:
    """Write ``reported_values.tex`` from the committed figure summaries.

    Parameters
    ----------
    output_path : Path, default ``MACRO_FILE_PATH``
        Destination of the generated macro file.
    figure03_path, figure04_path, figure04_supplement_path, figure03_sensitivity_path : Path
        Canonical summary JSONs to read.

    Returns
    -------
    Path
        The path written.
    """
    text = render_macro_file(
        _load(figure03_path),
        _load(figure04_path),
        _load(figure04_supplement_path),
        _load(figure03_sensitivity_path),
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
