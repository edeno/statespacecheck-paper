"""Figure-4 follow-up analyses on the full recording: broadening, coverage, and rate.

These analyses characterize *why* the Continuous--Fragmented model removes
Continuous-model flags, using the same fitted models, the same full recording,
and the same observed spike events as the canonical Figure 4. They modify only
the diagnostics' inputs or thresholds; none of them refits a decoder.

- **Predictive HPD-region sizes**: how many valid position bins each model's
  predictive HPD region occupies, across all events and at the events the
  Continuous--Fragmented model rescues.
- **Uniform-mixture counterfactual**: replace the Continuous prediction by
  ``P' = (1 - w) P + w U`` (``U`` uniform over valid bins) and recompute HPD
  overlap for the same events. The fraction of original Continuous flags this
  removes measures the diagnostic's response to broadening alone; it is a
  diagnostic counterfactual, not an alternative decoder.
- **Coverage sensitivity**: flags and rescue at several HPD coverage levels,
  on identical events, so every change has an interpretable denominator.
- **Behavior stratification**: flag fractions during immobility versus
  movement at a stated speed cutoff (no independent replay labels exist for
  this recording).
- **Firing-rate association**: per-unit session rates versus per-unit flag
  fractions, event and flag contributions by rate group, the fraction of
  rescued / newly flagged rank events from low-rate units, and a descriptive
  constant mark-frequency baseline that ranks each event by its unit's session
  frequency alone.

All quantities are computed from the cached predictive distributions in
bounded event chunks; the whole ``(n_time, n_bins)`` arrays are never gathered
per event.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator

import numpy as np
import statespacecheck as ssc
import xarray as xr
from numpy.typing import NDArray
from scipy.stats import spearmanr

from statespacecheck_paper.diagnostics import (
    SpikeEventDiagnostics,
    compute_normalized_event_likelihood,
)
from statespacecheck_paper.figure04_place_fields import get_state_marginalized_posterior
from statespacecheck_paper.figure04_workflow import Figure4RenderData

_QUANTILES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


@dataclasses.dataclass(frozen=True)
class Figure4BroadeningConfig:
    """Settings for the Figure-4 broadening, coverage, behavior, and rate analyses.

    Attributes
    ----------
    uniform_weights : tuple of float
        Uniform-mixture weights ``w`` for the counterfactual prediction
        ``(1 - w) P + w U``. ``1.0`` is the fully uniform limiting case.
    hpd_coverages : tuple of float
        HPD coverage levels at which flags, rescue, and region sizes are
        evaluated. The manuscript's primary setting is ``0.95``.
    hpd_flag_threshold : float
        HPD-overlap cutoff at or below which an event is flagged.
    pvalue_flag_threshold : float
        Rank-based predictive p-value cutoff at or below which an event is flagged.
    low_rate_threshold_hz : float
        Units with session rate below this are the "low-rate" group.
    rate_group_edges_hz : tuple of float
        Upper edges of the firing-rate groups (the last group is open-ended).
    speed_cutoff_cm_s : float
        Head-speed cutoff separating immobile from moving events.
    event_chunk : int
        Number of events gathered per chunk when reading the predictions.
    """

    uniform_weights: tuple[float, ...] = (0.02, 0.05, 0.06, 0.10, 1.0)
    hpd_coverages: tuple[float, ...] = (0.5, 0.8, 0.95)
    hpd_flag_threshold: float = 0.05
    pvalue_flag_threshold: float = 0.05
    low_rate_threshold_hz: float = 0.1
    rate_group_edges_hz: tuple[float, ...] = (0.1, 1.0, 5.0)
    speed_cutoff_cm_s: float = 4.0
    event_chunk: int = 50_000

    def __post_init__(self) -> None:
        if not all(0.0 < w <= 1.0 for w in self.uniform_weights):
            raise ValueError("uniform_weights must lie in (0, 1]")
        if not all(0.0 < c < 1.0 for c in self.hpd_coverages):
            raise ValueError("hpd_coverages must lie in (0, 1)")
        if 0.95 not in self.hpd_coverages:
            raise ValueError("hpd_coverages must include the primary 0.95 setting")
        if self.event_chunk < 1:
            raise ValueError("event_chunk must be positive")
        if not np.all(np.diff(self.rate_group_edges_hz) > 0):
            raise ValueError("rate_group_edges_hz must be strictly increasing")


def _quantile_record(values: NDArray[np.floating] | NDArray[np.integer]) -> dict[str, object]:
    """Quantiles (at :data:`_QUANTILES`) plus the count of ``values``."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"n": 0, "quantile_probabilities": list(_QUANTILES), "quantiles": None}
    return {
        "n": int(values.size),
        "quantile_probabilities": list(_QUANTILES),
        "quantiles": [float(q) for q in np.quantile(values, _QUANTILES)],
        "mean": float(np.mean(values)),
    }


def _event_prediction_chunks(
    predictive: NDArray[np.float64],
    event_time_ind: NDArray[np.intp],
    chunk: int,
) -> Iterator[tuple[slice, NDArray[np.float64]]]:
    """Yield ``(event_slice, predictions)`` for bounded event chunks."""
    n_events = event_time_ind.shape[0]
    for start in range(0, n_events, chunk):
        stop = min(start + chunk, n_events)
        sl = slice(start, stop)
        yield sl, np.asarray(predictive[event_time_ind[sl]], dtype=np.float64)


def _hpd_sizes_and_overlaps(
    predictive: NDArray[np.float64],
    likelihoods: NDArray[np.float64],
    event_time_ind: NDArray[np.intp],
    event_cell_ind: NDArray[np.intp],
    coverage: float,
    chunk: int,
    *,
    uniform_weight: float = 0.0,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Per-event predictive HPD-region size and HPD overlap at ``coverage``.

    With ``uniform_weight = w > 0`` the prediction is replaced by
    ``(1 - w) P + w U`` before both quantities are computed.
    """
    n_events = event_time_ind.shape[0]
    n_bins = predictive.shape[1]
    sizes = np.empty(n_events, dtype=np.int64)
    overlaps = np.empty(n_events, dtype=np.float64)
    for sl, pred in _event_prediction_chunks(predictive, event_time_ind, chunk):
        if uniform_weight > 0.0:
            pred = (1.0 - uniform_weight) * pred + uniform_weight / n_bins
        lik = likelihoods[event_cell_ind[sl]]
        sizes[sl] = ssc.highest_density_region(pred, coverage=coverage).sum(axis=1)
        overlaps[sl] = ssc.hpd_overlap(pred, lik, coverage=coverage)
    return sizes, overlaps


def _marginal_predictive(results: xr.Dataset) -> NDArray[np.float64]:
    """Mode-marginalized predictive distribution over valid position bins."""
    return np.asarray(get_state_marginalized_posterior(results, "predictive"), dtype=np.float64)


def _check_event_alignment(a: SpikeEventDiagnostics, b: SpikeEventDiagnostics) -> None:
    if not (
        np.array_equal(a.event_time_ind, b.event_time_ind)
        and np.array_equal(a.event_cell_ind, b.event_cell_ind)
    ):
        raise ValueError("The two models' diagnostics must carry identical event indices.")


@dataclasses.dataclass(frozen=True)
class CoverageBroadeningArrays:
    """Per-event arrays behind one coverage level of the broadening analysis.

    Attributes
    ----------
    coverage : float
    size_continuous, size_continuous_fragmented : np.ndarray, shape (n_events,)
        Predictive HPD-region size (valid bins) at each event.
    likelihood_size : np.ndarray, shape (n_events,)
        Likelihood HPD-region size at each event.
    flag_continuous, flag_continuous_fragmented : np.ndarray, shape (n_events,)
        HPD-overlap flags at the configured cutoff.
    mixture_flags : dict[str, np.ndarray]
        Flags under each uniform-mixture weight (key ``f"{w:g}"``).
    """

    coverage: float
    size_continuous: NDArray[np.int64]
    size_continuous_fragmented: NDArray[np.int64]
    likelihood_size: NDArray[np.int64]
    flag_continuous: NDArray[np.bool_]
    flag_continuous_fragmented: NDArray[np.bool_]
    mixture_flags: dict[str, NDArray[np.bool_]]


@dataclasses.dataclass(frozen=True)
class BroadeningResults:
    """Arrays and JSON-ready summary of the region-size / broadening analysis."""

    arrays_by_coverage: dict[str, CoverageBroadeningArrays]
    summary: dict[str, object]


def compute_region_size_and_broadening(
    render_data: Figure4RenderData,
    config: Figure4BroadeningConfig,
) -> BroadeningResults:
    """HPD-region sizes, uniform-mixture rescue, and coverage sensitivity.

    The returned ``summary`` is a JSON-ready mapping keyed by coverage level.
    For each coverage the record holds the Continuous and
    Continuous--Fragmented flag counts, the rescue/newly-flagged counts
    between them, predictive-region-size quantiles for all events and for the
    rescued events, the likelihood region-size quantiles, and for each uniform
    weight the number of original Continuous flags removed and the number of
    flags the mixture introduces among previously unflagged events. The
    per-event arrays behind those summaries are returned alongside for
    plotting.

    Every quantity uses the same ``n_events`` observed spike events, so the
    counts are directly comparable across coverages and weights.
    """
    decode = render_data.decode_results
    cont = decode.continuous_diagnostics
    frag = decode.continuous_fragmented_diagnostics
    _check_event_alignment(cont, frag)
    event_time_ind = np.asarray(cont.event_time_ind, dtype=np.intp)
    event_cell_ind = np.asarray(cont.event_cell_ind, dtype=np.intp)
    n_events = int(event_time_ind.size)

    likelihoods = np.asarray(
        compute_normalized_event_likelihood(
            np.asarray(decode.diagnostic_place_fields, dtype=np.float64)
        ),
        dtype=np.float64,
    )
    n_bins = likelihoods.shape[1]
    pred_cont = _marginal_predictive(decode.continuous_results)
    pred_frag = _marginal_predictive(decode.continuous_fragmented_results)
    if pred_cont.shape[1] != n_bins or pred_frag.shape[1] != n_bins:
        raise ValueError("Predictive distributions and place fields disagree on position bins.")

    threshold = config.hpd_flag_threshold
    by_coverage: dict[str, object] = {}
    arrays_by_coverage: dict[str, CoverageBroadeningArrays] = {}
    for coverage in config.hpd_coverages:
        size_cont, overlap_cont = _hpd_sizes_and_overlaps(
            pred_cont, likelihoods, event_time_ind, event_cell_ind, coverage, config.event_chunk
        )
        size_frag, overlap_frag = _hpd_sizes_and_overlaps(
            pred_frag, likelihoods, event_time_ind, event_cell_ind, coverage, config.event_chunk
        )
        if coverage == 0.95:
            # The recomputed 95% overlaps must reproduce the cached diagnostics.
            if not np.allclose(overlap_cont, cont.event_hpd_overlap) or not np.allclose(
                overlap_frag, frag.event_hpd_overlap
            ):
                raise ValueError("Recomputed 95% HPD overlaps do not match the cached diagnostics.")
        likelihood_sizes = ssc.highest_density_region(likelihoods, coverage=coverage).sum(axis=1)
        flag_cont = overlap_cont <= threshold
        flag_frag = overlap_frag <= threshold
        rescued = flag_cont & ~flag_frag
        newly = ~flag_cont & flag_frag

        mixtures: dict[str, object] = {}
        mixture_flags: dict[str, NDArray[np.bool_]] = {}
        for weight in config.uniform_weights:
            _, overlap_mix = _hpd_sizes_and_overlaps(
                pred_cont,
                likelihoods,
                event_time_ind,
                event_cell_ind,
                coverage,
                config.event_chunk,
                uniform_weight=weight,
            )
            flag_mix = overlap_mix <= threshold
            mixture_flags[f"{weight:g}"] = flag_mix
            mixtures[f"{weight:g}"] = {
                "uniform_weight": float(weight),
                "n_continuous_flags": int(flag_cont.sum()),
                "n_original_flags_removed": int(np.sum(flag_cont & ~flag_mix)),
                "fraction_original_flags_removed": (
                    float(np.sum(flag_cont & ~flag_mix) / flag_cont.sum())
                    if flag_cont.sum()
                    else None
                ),
                "n_new_flags": int(np.sum(~flag_cont & flag_mix)),
                "n_flags_under_mixture": int(flag_mix.sum()),
                "lower_bound_region_fraction": float(max(0.0, (coverage - 1.0 + weight) / weight)),
            }

        by_coverage[f"{coverage:g}"] = {
            "coverage": float(coverage),
            "n_events": n_events,
            "n_position_bins": int(n_bins),
            "continuous": {
                "n_flagged": int(flag_cont.sum()),
                "region_size_all_events": _quantile_record(size_cont),
                "region_size_flagged_events": _quantile_record(size_cont[flag_cont]),
                "region_size_rescued_events": _quantile_record(size_cont[rescued]),
            },
            "continuous_fragmented": {
                "n_flagged": int(flag_frag.sum()),
                "region_size_all_events": _quantile_record(size_frag),
                "region_size_flagged_events": _quantile_record(size_frag[flag_frag]),
                "region_size_rescued_events": _quantile_record(size_frag[rescued]),
            },
            "likelihood_region_size_all_events": _quantile_record(likelihood_sizes[event_cell_ind]),
            "n_rescued": int(rescued.sum()),
            "n_newly_flagged": int(newly.sum()),
            "rescue_fraction": (
                float(rescued.sum() / flag_cont.sum()) if flag_cont.sum() else None
            ),
            "uniform_mixture": mixtures,
        }
        arrays_by_coverage[f"{coverage:g}"] = CoverageBroadeningArrays(
            coverage=float(coverage),
            size_continuous=size_cont,
            size_continuous_fragmented=size_frag,
            likelihood_size=likelihood_sizes[event_cell_ind].astype(np.int64),
            flag_continuous=flag_cont,
            flag_continuous_fragmented=flag_frag,
            mixture_flags=mixture_flags,
        )
    summary: dict[str, object] = {
        "hpd_flag_threshold": float(threshold),
        "region_size_unit": "valid_position_bins",
        "uniform_mixture_definition": (
            "Replace the Continuous prediction P at every event by (1 - w) P + w U, U uniform "
            "over the valid position bins, and recompute HPD overlap; a diagnostic "
            "counterfactual on the same events, not a refitted decoder."
        ),
        "by_coverage": by_coverage,
    }
    return BroadeningResults(arrays_by_coverage=arrays_by_coverage, summary=summary)


def _unit_session_rates_hz(
    event_cell_ind: NDArray[np.intp], n_units: int, duration_s: float
) -> NDArray[np.float64]:
    counts = np.bincount(event_cell_ind, minlength=n_units).astype(np.float64)
    return counts / duration_s


def _constant_frequency_rank_pvalues(
    event_cell_ind: NDArray[np.intp], n_units: int
) -> NDArray[np.float64]:
    """Inclusive rank p-values under a constant (session-frequency) mark law.

    Each unit's mark probability is its share of all analyzed events; an
    event's p-value is the total probability of units whose share is at most
    its own. This ignores the state entirely and serves only as a descriptive
    baseline for how much of the rank diagnostic's behavior tracks marginal
    unit frequency.
    """
    counts = np.bincount(event_cell_ind, minlength=n_units).astype(np.float64)
    q = counts / counts.sum()
    order = np.argsort(q, kind="stable")
    sorted_q = q[order]
    cumulative = np.cumsum(sorted_q)
    # Inclusive tail: every unit with share <= q[c] contributes, so each unit
    # takes the cumulative mass through the last position of its tied block.
    last_index_of_value = np.searchsorted(sorted_q, sorted_q, side="right") - 1
    p_by_unit = np.empty(n_units, dtype=np.float64)
    p_by_unit[order] = cumulative[last_index_of_value]
    return np.minimum(p_by_unit[event_cell_ind], 1.0)


def compute_rate_and_behavior_association(
    render_data: Figure4RenderData,
    config: Figure4BroadeningConfig,
) -> dict[str, object]:
    """Per-unit rate association, rate-group contributions, behavior stratification.

    Unit rate is the analyzed event count divided by the position-recording
    duration. Units with no analyzed events are reported separately from the
    active units used in the rank correlation. Rate groups partition the
    active units by :attr:`Figure4BroadeningConfig.rate_group_edges_hz`; for
    each group the record reports the fraction of units, of events, and of
    each model's flags contributed. Behavior strata use the head speed at each
    event's decoder bin.
    """
    decode = render_data.decode_results
    cont = decode.continuous_diagnostics
    frag = decode.continuous_fragmented_diagnostics
    _check_event_alignment(cont, frag)
    event_time_ind = np.asarray(cont.event_time_ind, dtype=np.intp)
    event_cell_ind = np.asarray(cont.event_cell_ind, dtype=np.intp)
    n_units = int(decode.spike_counts.shape[1])
    time = np.asarray(render_data.time, dtype=np.float64)
    duration_s = float(time[-1] - time[0])

    rates_hz = _unit_session_rates_hz(event_cell_ind, n_units, duration_s)
    active = np.bincount(event_cell_ind, minlength=n_units) > 0
    low_rate_unit = rates_hz < config.low_rate_threshold_hz
    low_rate_event = low_rate_unit[event_cell_ind]

    flags = {
        "continuous": {
            "hpd_overlap": cont.event_hpd_overlap <= config.hpd_flag_threshold,
            "predictive_pvalue": cont.event_predictive_pvalue <= config.pvalue_flag_threshold,
        },
        "continuous_fragmented": {
            "hpd_overlap": frag.event_hpd_overlap <= config.hpd_flag_threshold,
            "predictive_pvalue": frag.event_predictive_pvalue <= config.pvalue_flag_threshold,
        },
    }

    # Per-unit flag fractions (active units only).
    per_unit: dict[str, object] = {"rate_hz": rates_hz.tolist(), "active": active.tolist()}
    unit_flag_fraction: dict[str, dict[str, NDArray[np.float64]]] = {}
    for model_name, model_flags in flags.items():
        unit_flag_fraction[model_name] = {}
        for metric, flag in model_flags.items():
            flagged_counts = np.bincount(event_cell_ind, weights=flag, minlength=n_units)
            unit_counts = np.bincount(event_cell_ind, minlength=n_units).astype(np.float64)
            fraction = np.full(n_units, np.nan)
            fraction[active] = flagged_counts[active] / unit_counts[active]
            unit_flag_fraction[model_name][metric] = fraction
            per_unit[f"{model_name}_{metric}_flag_fraction"] = [
                None if np.isnan(v) else float(v) for v in fraction
            ]

    # Rate groups over active units.
    edges = np.asarray(config.rate_group_edges_hz, dtype=float)
    group_index = np.digitize(rates_hz, edges)  # 0..len(edges)
    group_labels = (
        [f"<{edges[0]:g}"]
        + [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
        + [f">={edges[-1]:g}"]
    )
    groups: list[dict[str, object]] = []
    event_group = group_index[event_cell_ind]
    for g, label in enumerate(group_labels):
        in_group_units = active & (group_index == g)
        in_group_events = event_group == g
        record: dict[str, object] = {
            "label_hz": label,
            "n_units": int(in_group_units.sum()),
            "fraction_of_active_units": float(in_group_units.sum() / active.sum()),
            "n_events": int(in_group_events.sum()),
            "fraction_of_events": float(in_group_events.sum() / event_cell_ind.size),
        }
        for model_name, model_flags in flags.items():
            for metric, flag in model_flags.items():
                total = int(flag.sum())
                record[f"{model_name}_{metric}_n_flags"] = int(np.sum(flag & in_group_events))
                record[f"{model_name}_{metric}_fraction_of_flags"] = (
                    float(np.sum(flag & in_group_events) / total) if total else None
                )
                record[f"{model_name}_{metric}_flag_fraction_within_group"] = (
                    float(np.sum(flag & in_group_events) / in_group_events.sum())
                    if in_group_events.sum()
                    else None
                )
        groups.append(record)

    # Constant mark-frequency baseline for the rank diagnostic.
    p_constant = _constant_frequency_rank_pvalues(event_cell_ind, n_units)
    flag_constant = p_constant <= config.pvalue_flag_threshold
    constant_baseline: dict[str, object] = {
        "definition": (
            "Inclusive rank p-value of each event's unit under the constant mark law "
            "q_c = n_c / N (session event shares), ignoring the state distribution."
        ),
        "n_flagged": int(flag_constant.sum()),
        "flag_fraction": float(flag_constant.mean()),
    }
    for model_name, model_flags in flags.items():
        actual = model_flags["predictive_pvalue"]
        constant_baseline[f"{model_name}_fraction_of_actual_flags_also_constant_flagged"] = (
            float(np.sum(actual & flag_constant) / actual.sum()) if actual.sum() else None
        )
        constant_baseline[f"{model_name}_fraction_of_constant_flags_also_actual_flagged"] = (
            float(np.sum(actual & flag_constant) / flag_constant.sum())
            if flag_constant.sum()
            else None
        )

    # Rank rescue / newly flagged from low-rate units.
    rank_cont = flags["continuous"]["predictive_pvalue"]
    rank_frag = flags["continuous_fragmented"]["predictive_pvalue"]
    rescued = rank_cont & ~rank_frag
    newly = ~rank_cont & rank_frag
    rank_changes = {
        "n_rescued": int(rescued.sum()),
        "n_rescued_from_low_rate_units": int(np.sum(rescued & low_rate_event)),
        "n_newly_flagged": int(newly.sum()),
        "n_newly_flagged_from_low_rate_units": int(np.sum(newly & low_rate_event)),
    }

    # Behavior strata.
    head_speed = np.asarray(
        render_data.recording.position_info["head_speed"].to_numpy(dtype=np.float64)
    )
    immobile = head_speed[event_time_ind] < config.speed_cutoff_cm_s
    behavior: dict[str, object] = {
        "speed_cutoff_cm_s": float(config.speed_cutoff_cm_s),
        "speed_definition": "head_speed at each event's decoder time bin",
        "independent_replay_labels": None,
    }
    for stratum, mask in (("immobile", immobile), ("moving", ~immobile)):
        record = {"n_events": int(mask.sum())}
        for model_name, model_flags in flags.items():
            for metric, flag in model_flags.items():
                record[f"{model_name}_{metric}_flag_fraction"] = (
                    float(np.sum(flag & mask) / mask.sum()) if mask.sum() else None
                )
        behavior[stratum] = record

    summary: dict[str, object] = {
        "duration_seconds": duration_s,
        "n_units": n_units,
        "n_active_units": int(active.sum()),
        "n_zero_event_units": int((~active).sum()),
        "n_events": int(event_cell_ind.size),
        "low_rate_threshold_hz": float(config.low_rate_threshold_hz),
        "n_low_rate_units": int(low_rate_unit.sum()),
        "n_low_rate_active_units": int(np.sum(low_rate_unit & active)),
        "n_low_rate_events": int(low_rate_event.sum()),
        "rate_definition": "analyzed event count / position-recording duration",
        "rate_group_edges_hz": [float(e) for e in edges],
        "rate_groups": groups,
        "constant_frequency_baseline": constant_baseline,
        "rank_flag_changes": rank_changes,
        "behavior": behavior,
        "per_unit": per_unit,
    }
    for model_name in flags:
        model_record: dict[str, object] = {}
        for metric in ("hpd_overlap", "predictive_pvalue"):
            fraction = unit_flag_fraction[model_name][metric]
            rho, _ = spearmanr(rates_hz[active], fraction[active])
            flag = flags[model_name][metric]
            model_record[metric] = {
                "n_flagged": int(flag.sum()),
                "spearman_rate_vs_unit_flag_fraction_active_units": float(rho),
                "n_active_units_all_events_flagged": int(np.sum(active & (fraction >= 1.0))),
                "fraction_of_flags_from_low_rate_units": (
                    float(np.sum(flag & low_rate_event) / flag.sum()) if flag.sum() else None
                ),
            }
        summary[model_name] = model_record
    return summary
