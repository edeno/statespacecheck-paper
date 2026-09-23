r"""Emit the manuscript's reported numbers as LaTeX macros.

The manuscript's reported analysis statistics and configuration live in the
canonical figure summaries (``figure03_summary.json``,
``figure04_summary.json``, and the Figure-4 supplement's
``figure04_supplement_summary.json``). This module turns those summaries into a file of
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
    # The prose spells this value as "threefold", so a fractional factor
    # cannot be represented faithfully. Reuse the exact-value guard rather
    # than silently rounding a valid simulation setting to a different value.
    burst_factor_word = cardinal_word(int(_exact(config["history_burst_factor"])))

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
            significant(peak_count_per_step, SIGNIFICANT_FIGURES),
            "place_field_rate_scale / (place_field_std * sqrt(2 pi))",
        ),
        MacroDefinition(
            "SimPeakRateHz",
            significant(peak_count_per_step * 1000.0, SIGNIFICANT_FIGURES),
            "peak expected count per step at 1 ms/step, in Hz",
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
            burst_factor_word,
            "history_burst_factor",
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
    return macros


def _recording_supplement(
    payload: dict[str, Any], figure04_payload: dict[str, Any]
) -> list[MacroDefinition]:
    """Build the macros computed by the Figure-4 supplement analyses.

    The fitted model's HPD rescue rate comes from the Figure-4 summary, so the
    uniform-mixture comparison is against the same number the Results quote.
    """
    broadening = payload["region_size_and_broadening"]
    by_coverage = broadening["by_coverage"]
    main = by_coverage["0.95"]

    def median(record: dict[str, Any]) -> float:
        """Return the stored median of a quantile record."""
        value: float = record["quantiles"][record["quantile_probabilities"].index(0.5)]
        return value

    macros = [
        MacroDefinition(
            "RecNPositionBins",
            _exact(main["n_position_bins"]),
            "region_size_and_broadening.by_coverage[0.95].n_position_bins",
        )
    ]
    for model, model_stem in (("continuous", "Cont"), ("continuous_fragmented", "Frag")):
        for events, events_stem in (("all_events", "All"), ("rescued_events", "Rescued")):
            macros.append(
                MacroDefinition(
                    f"Rec{model_stem}Region{events_stem}",
                    _exact(median(main[model][f"region_size_{events}"])),
                    f"median of by_coverage[0.95].{model}.region_size_{events}",
                )
            )
    macros.append(
        MacroDefinition(
            "RecLikelihoodRegion",
            _exact(median(main["likelihood_region_size_all_events"])),
            "median of by_coverage[0.95].likelihood_region_size_all_events",
        )
    )

    # The smallest mixture weight evaluated, and the smallest one that removes
    # at least the share of flags the fitted Continuous--Fragmented model rescues.
    mixtures = sorted(
        (record for record in main["uniform_mixture"].values() if record["uniform_weight"] < 1.0),
        key=lambda record: record["uniform_weight"],
    )
    fitted_rescue = _confusion(figure04_payload, "hpd_overlap")["rescue_rate"]
    matching = next(
        record for record in mixtures if record["fraction_original_flags_removed"] >= fitted_rescue
    )
    for stem, record, note in (
        ("Min", mixtures[0], "smallest uniform weight < 1"),
        ("Match", matching, "smallest weight removing >= the fitted HPD rescue rate"),
    ):
        macros.extend(
            [
                MacroDefinition(f"RecMixWeight{stem}", _exact(record["uniform_weight"], 2), note),
                MacroDefinition(
                    f"RecMixRemoved{stem}",
                    whole_percent(100.0 * record["fraction_original_flags_removed"]),
                    f"uniform_mixture[{record['uniform_weight']}].fraction_original_flags_removed",
                ),
            ]
        )
    for key, stem in (("0.8", "Eighty"), ("0.5", "Fifty")):
        macros.append(
            MacroDefinition(
                f"RecCov{stem}RescuedPercent",
                whole_percent(100.0 * by_coverage[key]["rescue_fraction"]),
                f"by_coverage[{key}].rescue_fraction",
            )
        )

    rate = payload["rate_and_behavior"]
    # rate_groups follow rate_group_edges_hz: group i lies below edge i.
    n_under_one_hz = payload["analysis_configuration"]["rate_group_edges_hz"].index(1.0) + 1
    under_one_hz = rate["rate_groups"][:n_under_one_hz]
    macros.extend(
        [
            MacroDefinition(
                "RecLowRateThresholdHz",
                _exact(rate["low_rate_threshold_hz"], 1),
                "rate_and_behavior.low_rate_threshold_hz",
            ),
            MacroDefinition(
                "RecUnderOneHzEventsPercent",
                significant(
                    100.0 * sum(group["fraction_of_events"] for group in under_one_hz),
                    SIGNIFICANT_FIGURES,
                ),
                "rate_groups below 1 Hz: fraction_of_events",
            ),
            MacroDefinition(
                "RecUnderOneHzFlagsPvalueCont",
                whole_percent(
                    100.0
                    * sum(
                        group["continuous_predictive_pvalue_fraction_of_flags"]
                        for group in under_one_hz
                    )
                ),
                "rate_groups below 1 Hz: continuous_predictive_pvalue_fraction_of_flags",
            ),
        ]
    )
    behavior = rate["behavior"]
    macros.append(
        MacroDefinition(
            "RecSpeedCutoff",
            _exact(behavior["speed_cutoff_cm_s"]),
            "rate_and_behavior.behavior.speed_cutoff_cm_s",
        )
    )
    for stratum, stratum_stem in (("immobile", "Immobile"), ("moving", "Moving")):
        for metric, metric_stem in (("hpd_overlap", "Hpd"), ("predictive_pvalue", "Pvalue")):
            macros.append(
                MacroDefinition(
                    f"Rec{stratum_stem}{metric_stem}Cont",
                    significant(
                        100.0 * behavior[stratum][f"continuous_{metric}_flag_fraction"],
                        SIGNIFICANT_FIGURES,
                    ),
                    f"behavior.{stratum}.continuous_{metric}_flag_fraction",
                )
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
) -> str:
    """Render the full ``reported_values.tex`` contents.

    Parameters
    ----------
    figure03_payload, figure04_payload : dict
        Parsed contents of the two canonical figure summaries.
    figure04_supplement_payload : dict
        Parsed contents of the Figure-4 supplement summary.

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
            "Hippocampal recording (Figure 4 supplement) --- computed from the recording",
            _recording_supplement(figure04_supplement_payload, figure04_payload),
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
) -> Path:
    """Write ``reported_values.tex`` from the committed figure summaries.

    Parameters
    ----------
    output_path : Path, default ``MACRO_FILE_PATH``
        Destination of the generated macro file.
    figure03_path, figure04_path, figure04_supplement_path : Path
        Canonical summary JSONs to read.

    Returns
    -------
    Path
        The path written.
    """
    text = render_macro_file(
        _load(figure03_path), _load(figure04_path), _load(figure04_supplement_path)
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path
