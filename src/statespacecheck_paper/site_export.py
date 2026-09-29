"""Export compact data files for the project website's interactive explainer.

The website (``site/``) is static HTML and JavaScript. It has four interactive
pieces, and each reads data produced here from the paper's own pipeline:

- **Filter explainer** — the reader steps in time through a short, scripted
  spike train on the Figure-3 track, decoded by
  :func:`statespacecheck_paper.decoding.decode_with_diagnostics`: the
  prediction, each spike's place field and likelihood, and the posterior.
- **Playground** — the reader moves a Gaussian predictive distribution over the
  Figure-3 track and picks which place cell fired; the browser recomputes the
  three per-spike diagnostics live with a JavaScript port of
  :func:`statespacecheck.event_diagnostics`. :func:`metric_parity_fixture` writes
  reference cases computed through the paper's
  :func:`~statespacecheck_paper.diagnostics.compute_spike_event_diagnostics_from_rates`
  wrapper of that function, and ``site/tests/metrics.test.mjs`` checks the port against them.
- **Condition player** — one display window per Figure-3 condition from the
  seed-``config.random_seed`` realization shown in Figure 3a.
- **Recording comparison** — the Figure-4 detail window under both decoders.

The players show precomputed diagnostic values, never recomputed ones. Numbers
quoted in the page text come from :func:`statespacecheck_paper.reported_values.macro_sections`,
the same definitions the manuscript's macros use.

Distributions are shipped for display only: each row is scaled to its maximum
and quantized to ``uint8`` (see :func:`encode_display_rows`). Flag decisions are
computed here at full precision and shipped as booleans, so rounding cannot move
a spike across a threshold.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib as mpl
import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray

from statespacecheck_paper.decoding import decode_with_diagnostics
from statespacecheck_paper.diagnostics import (
    HPD_COVERAGE,
    INCLUSIVE_FLAG_COMPARISONS,
    DecodingDiagnostics,
    FlagDirection,
    compute_spike_event_diagnostics_from_rates,
    flag_mask,
)
from statespacecheck_paper.figure03_protocol import STEP_SECONDS, Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    all_place_field_centers,
    build_figure03_rate_tables,
    run_figure03_simulation,
)
from statespacecheck_paper.figure03_summary import conditions_by_id
from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_diagnostics import mean_event_likelihood_by_time
from statespacecheck_paper.figure04_models import CONTINUOUS, CONTINUOUS_FRAGMENTED
from statespacecheck_paper.figure04_place_fields import marginal_position_distribution
from statespacecheck_paper.figure04_protocol import FIGURE04_DETAIL_WINDOW, Figure4DetailWindow
from statespacecheck_paper.figure04_workflow import Figure4RenderData, prepare_figure04_render_data
from statespacecheck_paper.number_format import SIGNIFICANT_FIGURES, significant, whole_percent
from statespacecheck_paper.paths import (
    ANIMAL_DATE_EPOCH,
    DATA_PATH,
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    PARITY_FIXTURE_PATH,
    SITE_DATA_DIR,
)
from statespacecheck_paper.reported_values import cardinal_word, macro_sections
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    place_field_rates,
    simulate_spikes_position_tuned,
)
from statespacecheck_paper.style import (
    CMAP_LIKELIHOOD,
    CMAP_PREDICTIVE,
    METRIC_NAMES,
    PREDICTIVE_VMAX_QUANTILE,
)

# Significant figures kept for per-event diagnostic values. Significant figures,
# not decimal places, so a small predictive p-value keeps its magnitude.
EVENT_VALUE_SIGNIFICANT_FIGURES = 4

# Entries in each exported colormap lookup table.
COLORMAP_LUT_SIZE = 64

# Color scale of the Figure-4 predictive heatmaps: the 2nd to 98th percentiles
# of the plotted window, restating the limits that xarray's ``robust=True``
# (``xarray.plot.utils.ROBUST_PERCENTILE``) gives the figure in
# ``figure04_plot_primitives.plot_distribution_heatmap``. xarray drops
# non-finite values before taking them, whereas the export zero-fills NaN
# first. The zero-fill changes nothing here, since the state marginal the
# export reads has already dropped every bin holding a NaN in the window, but
# it would lower the limits of a window with NaN. Figure 3's scale is
# ``style.PREDICTIVE_VMAX_QUANTILE``, shared with ``figure03_plotting``.
FIGURE04_PREDICTIVE_PERCENTILES = (2.0, 98.0)


@dataclass(frozen=True)
class ConditionWindow:
    """Display window for one Figure-3 condition in the condition player.

    Parameters
    ----------
    condition_id : str
        A Figure-3 summary condition's ``condition_id``.
    start, stop : int
        Half-open range of simulation steps shown.
    """

    condition_id: str
    start: int
    stop: int


# Display choices, like ``FIGURE04_DETAIL_WINDOW``; each overlaps its condition's
# scored step windows (checked by the test suite). Abrupt conditions (remap,
# history dependence, replay) start a few hundred clean steps before onset so
# the change is visible. Drift builds up gradually, so its window sits mid-phase,
# where the flag rates are close to the across-realization medians. The sparse
# population fires only a handful of spikes, so its window spans the whole phase.
CONDITION_WINDOWS: tuple[ConditionWindow, ...] = (
    ConditionWindow("well_specified", 11_500, 13_000),
    ConditionWindow("remap", 5_700, 7_200),
    ConditionWindow("history_dependent", 13_700, 15_200),
    ConditionWindow("replay", 18_700, 21_300),
    ConditionWindow("drift", 24_000, 25_500),
    ConditionWindow("sparse_population", 29_700, 32_000),
)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def encode_display_rows(values: NDArray[np.floating]) -> str:
    """Scale each row to its maximum and base64-encode it as ``uint8``.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows, n_bins)
        Nonnegative finite values, e.g. distributions over position.

    Returns
    -------
    str
        Base64 of the row-major ``uint8`` array; a row's maximum maps to 255 and
        an all-zero row stays zero.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"values must be 2-D (n_rows, n_bins); got shape {array.shape}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("values must be finite and nonnegative")
    row_max = array.max(axis=1, keepdims=True)
    scaled = np.divide(array, row_max, out=np.zeros_like(array), where=row_max > 0.0)
    quantized = np.rint(scaled * 255.0).astype(np.uint8)
    return base64.b64encode(quantized.tobytes()).decode("ascii")


def heatmap_payload(
    values: NDArray[np.floating], value_range: tuple[float, float]
) -> dict[str, Any]:
    """Rows for a heatmap drawn on one shared color scale.

    Rows are shipped scaled to their own maxima (so each row's shape survives
    quantization, for the detail plots) together with each row's maximum, from
    which the page recovers the values and applies ``value_range``.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows, n_bins)
    value_range : (float, float)
        Values mapped to the bottom and top of the colormap.

    Returns
    -------
    dict
        ``rows`` (see :func:`encode_display_rows`), ``row_max``, and ``range``.
    """
    array = np.asarray(values, dtype=np.float64)
    vmin, vmax = (float(v) for v in value_range)
    if not vmax > vmin:
        raise ValueError(f"value_range must be increasing; got {value_range}")
    return {
        "rows": encode_display_rows(array),
        "row_max": _rounded_significant(array.max(axis=1), 6),
        "range": [vmin, vmax],
    }


def _finite(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """Coerce to float64 for JSON, refusing NaN/inf (which JSON cannot hold)."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite to export as JSON")
    return array


def _rounded(values: NDArray[np.floating], decimals: int) -> list[float]:
    """Round finite values to ``decimals`` places for JSON."""
    rounded: list[float] = np.round(_finite(values), decimals).tolist()
    return rounded


def _rounded_significant(values: NDArray[np.floating], digits: int) -> list[float]:
    """Round finite values to ``digits`` significant figures for JSON."""
    return [float(f"{value:.{digits}g}") for value in _finite(values).tolist()]


_FLAG_DIRECTION_BY_COMPARISON: dict[str, FlagDirection] = {
    comparison: direction for direction, comparison in INCLUSIVE_FLAG_COMPARISONS.items()
}


def flag_threshold_text(figure03_summary: Mapping[str, Any]) -> dict[str, str]:
    """Each Figure-3 flag threshold as the page prints it.

    The page states a threshold beside each diagnostic's readout, rounded by
    the manuscript's reporting policy: a threshold estimated from the pooled
    baseline is a derived constant, printed to
    :data:`~statespacecheck_paper.number_format.SIGNIFICANT_FIGURES`
    significant figures (the KL divergence's 4.138... prints as 4.1), and a
    fixed cutoff is a configured parameter, printed in full (0.05).

    Parameters
    ----------
    figure03_summary : mapping
        Parsed ``figure03_summary.json``: ``flag_rules`` and, for each metric,
        the ``threshold_provenance`` rule that set its threshold.

    Returns
    -------
    dict
        Metric name -> threshold text.
    """
    provenance = figure03_summary["threshold_provenance"]
    texts: dict[str, str] = {}
    for metric, rule in figure03_summary["flag_rules"].items():
        threshold = float(rule["threshold"])
        kind = provenance[metric]["rule"]
        if kind == "fixed_cutoff":
            texts[metric] = repr(threshold)
        elif kind == "pooled_baseline_quantile":
            texts[metric] = significant(threshold, SIGNIFICANT_FIGURES)
        else:
            raise ValueError(f"Unknown threshold rule {kind!r} for {metric}")
    return texts


def flag_events(values: NDArray[np.floating], rule: Mapping[str, Any]) -> NDArray[np.bool_]:
    """Apply an inclusive flag rule from a figure summary's ``flag_rules``.

    Parameters
    ----------
    values : np.ndarray, shape (n_events,)
        Full-precision diagnostic values.
    rule : mapping
        ``{"comparison": "less_than_or_equal" | "greater_than_or_equal",
        "threshold": float}``.

    Returns
    -------
    np.ndarray of bool, shape (n_events,)
    """
    threshold = float(rule["threshold"])
    comparison = rule["comparison"]
    if comparison not in _FLAG_DIRECTION_BY_COMPARISON:
        raise ValueError(f"Unknown flag comparison {comparison!r}")
    return flag_mask(values, threshold, _FLAG_DIRECTION_BY_COMPARISON[comparison])


class EventDiagnostics(Protocol):
    """Per-event diagnostic arrays shared by the Figure-3 and Figure-4 containers.

    Both ``DecodingDiagnostics`` and ``SpikeEventDiagnostics`` satisfy it.
    """

    @property
    def event_hpd_overlap(self) -> NDArray[np.floating]:
        """Per-event HPD overlap, shape (n_events,)."""
        ...

    @property
    def event_kl_divergence(self) -> NDArray[np.floating]:
        """Per-event KL divergence, shape (n_events,)."""
        ...

    @property
    def event_predictive_pvalue(self) -> NDArray[np.floating]:
        """Per-event rank-based predictive p-value, shape (n_events,)."""
        ...


def _event_payload(
    diagnostics: EventDiagnostics,
    selection: NDArray[np.bool_],
    flag_rules: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Per-event metric values and flags for the selected events."""
    payload: dict[str, Any] = {}
    flags: dict[str, list[bool]] = {}
    for metric in METRIC_NAMES:
        values = np.asarray(getattr(diagnostics, f"event_{metric}"))[selection]
        payload[metric] = _rounded_significant(values, EVENT_VALUE_SIGNIFICANT_FIGURES)
        if metric in flag_rules:
            flags[metric] = flag_events(values, flag_rules[metric]).tolist()
    payload["flagged"] = flags
    return payload


def colormap_lut(name: str) -> list[str]:
    """Sample a matplotlib colormap as ``COLORMAP_LUT_SIZE`` hex colors, low to high."""
    cmap = mpl.colormaps[name].resampled(COLORMAP_LUT_SIZE)
    return [mpl.colors.to_hex(cmap(i)) for i in range(COLORMAP_LUT_SIZE)]


# ---------------------------------------------------------------------------
# Filter explainer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterExplainerConfig:
    """The scripted spike train that the filter explainer steps through.

    The explainer keeps the Figure-3 track and place fields. Its animal runs
    smoothly up the track and back, from ``run_low`` to ``run_high`` and back
    to ``run_low``, and its cells fire less often than in the simulation, so
    the prediction visibly spreads between spikes. The decoder's movement
    model is the simulation decoder's Gaussian random walk, which ignores the
    animal's momentum, as decoders of this kind usually do. At
    ``conflict_step`` the ordinary spikes are replaced by one spike from each
    of the two cells whose fields are centered nearest to ``conflict_offset``
    from the animal, on the side of the track with room.

    Parameters
    ----------
    n_steps : int
        Time steps in the sequence.
    run_low, run_high : float
        Ends of the animal's run, in position units.
    step_std : float
        Standard deviation of the decoder's random-walk step, in position
        units per time step.
    rate_scale : float
        ``place_field_rate_scale`` of the cells. It multiplies the unit-area
        Gaussian place field, so a cell's peak expected count is
        ``rate_scale / (place_field_std * sqrt(2 * pi))`` spikes per step.
    seed : int
        Seed of the spikes.
    conflict_step : int
        Time step of the inconsistent spikes.
    conflict_offset : float
        Distance from the animal (position units) of the inconsistent spikes'
        place fields.
    """

    n_steps: int = 360
    run_low: float = 25.0
    run_high: float = 75.0
    step_std: float = 2.0
    rate_scale: float = 1.25
    seed: int = 26
    conflict_step: int = 288
    conflict_offset: float = 45.0


# A display choice, like ``CONDITION_WINDOWS``: a seed whose sequence shows the
# prediction spreading over a long gap, tracks the animal with no inconsistent
# spike before the conflict, and recovers after it (checked by the test suite).
FILTER_EXPLAINER = FilterExplainerConfig()


@dataclass(frozen=True)
class FilterExplainerSequence:
    """A decoded explainer sequence.

    Attributes
    ----------
    position_bins : np.ndarray, shape (n_bins,)
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spikes per step of each cell: the table the decoder builds
        from the same place-field parameters, kept for the page's field plots.
    true_position : np.ndarray, shape (n_steps,)
    spike_counts : np.ndarray, shape (n_steps, n_cells)
    decoded : DecodingDiagnostics
        Output of :func:`decode_with_diagnostics` for ``spike_counts``.
    """

    position_bins: NDArray[np.float64]
    rates: NDArray[np.float64]
    true_position: NDArray[np.float64]
    spike_counts: NDArray[np.int_]
    decoded: DecodingDiagnostics


def filter_explainer_sequence(config: Figure3Config) -> FilterExplainerSequence:
    """Simulate and decode the explainer's spike train.

    The spikes come from the simulation's generator, given the smooth run. The
    sequence is edited in two places: the first step holds one spike, from the
    cell whose field is centered nearest the animal, so the demonstration opens
    on a spike whose prediction is the flat initial distribution; and
    ``conflict_step`` holds the inconsistent spikes described in
    :class:`FilterExplainerConfig`. :data:`FILTER_EXPLAINER` supplies the run,
    rates, movement model, and the scripted edits.

    Parameters
    ----------
    config : Figure3Config
        Supplies the track and the place-field centers and width.
    """
    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
    explainer = FILTER_EXPLAINER
    centers = np.asarray(config.place_field_centers, dtype=np.float64)
    position_bins = config.position_bins
    steps = np.arange(explainer.n_steps)
    middle = (explainer.run_low + explainer.run_high) / 2
    half_range = (explainer.run_high - explainer.run_low) / 2
    position = middle - half_range * np.cos(2 * np.pi * steps / explainer.n_steps)
    rng = np.random.default_rng(explainer.seed)
    counts = simulate_spikes_position_tuned(
        position, centers, config.place_field_std, explainer.rate_scale, rng
    )
    counts[0] = 0
    counts[0, int(np.argmin(np.abs(centers - position[0])))] = 1
    here = position[explainer.conflict_step]
    midpoint = (config.position_min + config.position_max) / 2
    target = (
        here + explainer.conflict_offset if here <= midpoint else here - explainer.conflict_offset
    )
    counts[explainer.conflict_step] = 0
    counts[explainer.conflict_step, np.argsort(np.abs(centers - target))[:2]] = 1

    rates = np.asarray(
        place_field_rates(position_bins, centers, config.place_field_std, explainer.rate_scale),
        dtype=np.float64,
    )
    decoded = decode_with_diagnostics(
        counts,
        position_bins,
        gaussian_transition_matrix(position_bins, explainer.step_std),
        centers,
        config.place_field_std,
        explainer.rate_scale,
    )
    return FilterExplainerSequence(
        position_bins=position_bins,
        rates=rates,
        true_position=np.asarray(position, dtype=np.float64),
        spike_counts=counts,
        decoded=decoded,
    )


def _moments(
    distributions: NDArray[np.floating], position_bins: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and standard deviation of each row, shape (n_rows,) each."""
    mean = np.asarray(distributions @ position_bins, dtype=np.float64)
    variance = np.einsum("tb,tb->t", distributions, (position_bins[None, :] - mean[:, None]) ** 2)
    return mean, np.sqrt(variance)


def filter_explainer_payload(config: Figure3Config) -> dict[str, Any]:
    """Build the data for the filter explainer's time stepper, :data:`FILTER_EXPLAINER`.

    Parameters
    ----------
    config : Figure3Config

    Returns
    -------
    dict
        ``position_bins``, ``cell_centers``, ``true_position``; ``events``
        (time step, cell, and HPD overlap of each spike); ``predictive`` and
        ``posterior`` (display rows on one shared scale, see
        :func:`heatmap_payload`); ``likelihood`` (each step's normalized
        likelihood, the product of the fired cells' fields and the exposure
        term, as display rows); ``place_fields`` (one row per cell with its
        peak rate and the cells' shared range, see :func:`heatmap_payload`, so
        the page draws the fields in expected spikes per step); ``exposure``
        (``exp(-Λ(x))``, scaled to its maximum, where ``Λ`` is the cells' total
        expected count); ``moments`` (mean and SD of the prediction and
        posterior at each step); ``conflict_step``; and ``coverage``, the
        probability mass of the HPD regions the page's captions name.
    """
    sequence = filter_explainer_sequence(config)
    decoded = sequence.decoded
    bins = sequence.position_bins
    predictive = np.asarray(decoded.predictive, dtype=np.float64)
    posterior = np.asarray(decoded.posterior, dtype=np.float64)
    shared_range = (0.0, float(max(predictive.max(), posterior.max())))
    # exp(-Λ(x)) relative to its maximum, where Λ is lowest.
    total_rate = sequence.rates.sum(axis=1)
    exposure = np.exp(-(total_rate - total_rate.min()))
    predictive_mean, predictive_sd = _moments(predictive, bins)
    posterior_mean, posterior_sd = _moments(posterior, bins)
    return {
        "position_bins": bins.tolist(),
        "cell_centers": np.asarray(config.place_field_centers, dtype=np.float64).tolist(),
        "true_position": _rounded(sequence.true_position, 2),
        "events": {
            "t": np.asarray(decoded.event_time_ind).tolist(),
            "cell": np.asarray(decoded.event_cell_ind).tolist(),
            "hpd_overlap": _rounded_significant(
                decoded.event_hpd_overlap, EVENT_VALUE_SIGNIFICANT_FIGURES
            ),
        },
        "predictive": heatmap_payload(predictive, shared_range),
        "posterior": heatmap_payload(posterior, shared_range),
        "likelihood": encode_display_rows(decoded.combined_likelihood),
        "place_fields": heatmap_payload(sequence.rates.T, (0.0, float(sequence.rates.max()))),
        "exposure": _rounded(exposure, 4),
        "moments": {
            "predictive_mean": _rounded(predictive_mean, 2),
            "predictive_sd": _rounded(predictive_sd, 2),
            "posterior_mean": _rounded(posterior_mean, 2),
            "posterior_sd": _rounded(posterior_sd, 2),
        },
        "conflict_step": FILTER_EXPLAINER.conflict_step,
        "coverage": HPD_COVERAGE,
    }


# ---------------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------------


def gaussian_predictive(
    position_bins: NDArray[np.floating], mean: float, std: float
) -> NDArray[np.float64]:
    """Discretized Gaussian predictive distribution over position bins.

    The site's JavaScript builds the same distribution, and the parity test
    checks it against this definition.

    Parameters
    ----------
    position_bins : np.ndarray, shape (n_bins,)
    mean, std : float
        Center and standard deviation in position units; ``std > 0``.

    Returns
    -------
    np.ndarray, shape (n_bins,)
        Nonnegative weights summing to 1.
    """
    if std <= 0.0:
        raise ValueError(f"std must be positive; got {std}")
    bins = np.asarray(position_bins, dtype=np.float64)
    weights = np.exp(-0.5 * ((bins - mean) / std) ** 2)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("Gaussian predictive underflowed to zero on the grid")
    normalized: NDArray[np.float64] = weights / total
    return normalized


@dataclass(frozen=True)
class PlaygroundEnsemble:
    """One decoder rate table the playground can evaluate spikes against.

    Parameters
    ----------
    ensemble_id : str
        ``"place_cells"`` (the well-specified Figure-3 decoder) or
        ``"sparse_epoch"`` (the sparse-population decoder: a quiet ordinary
        ensemble plus the narrow, sparsely firing cells).
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spikes per step for every decoded cell.
    selectable_cells : tuple of int
        Cells the reader may choose as the one that fired: the cells that are
        active under this table.
    """

    ensemble_id: str
    rates: NDArray[np.float64]
    selectable_cells: tuple[int, ...]


def playground_ensembles(
    config: Figure3Config, sparse_centers: NDArray[np.floating]
) -> tuple[PlaygroundEnsemble, ...]:
    """Build the two Figure-3 decoder rate tables on ``config.position_bins``.

    Parameters
    ----------
    config : Figure3Config
    sparse_centers : np.ndarray, shape (sparse_cell_count,)
        Sparse-population field centers, as returned on
        ``Figure3SimulationResult.sparse_place_field_centers``.

    Returns
    -------
    tuple of PlaygroundEnsemble
        Place cells, then the sparse epoch. Cells are ordered as in
        :func:`all_place_field_centers`.
    """
    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
    sparse = np.asarray(sparse_centers, dtype=np.float64)
    tables = build_figure03_rate_tables(
        config.position_bins, config.place_field_centers, sparse, config
    )
    n_place_cells = len(config.place_field_centers)
    n_cells = n_place_cells + sparse.size
    return (
        PlaygroundEnsemble(
            "place_cells",
            np.asarray(tables.baseline_firing_rates, dtype=np.float64),
            tuple(range(n_place_cells)),
        ),
        PlaygroundEnsemble(
            "sparse_epoch",
            np.asarray(tables.sparse_population_firing_rates, dtype=np.float64),
            tuple(range(n_place_cells, n_cells)),
        ),
    )


@dataclass(frozen=True)
class PlaygroundPreset:
    """An example the playground loads from one of its buttons.

    Parameters
    ----------
    preset_id : str
        The button's ``data-preset`` in ``site/index.html``.
    ensemble_id : str
        A :class:`PlaygroundEnsemble`'s ``ensemble_id``.
    mean, std : float
        Center and standard deviation of the Gaussian prediction, in position
        units (:func:`gaussian_predictive`).
    cell : int
        The cell that fired; one of the ensemble's ``selectable_cells``.
    """

    preset_id: str
    ensemble_id: str
    mean: float
    std: float
    cell: int


# The playground's examples. Each button's label claims which diagnostics flag
# its spike under the Figure-3 flag rules, checked by the test suite.
PLAYGROUND_PRESETS: tuple[PlaygroundPreset, ...] = (
    PlaygroundPreset("consistent", "place_cells", mean=50.0, std=5.0, cell=5),
    PlaygroundPreset("conflicting", "place_cells", mean=25.0, std=4.0, cell=7),
    PlaygroundPreset("nested", "place_cells", mean=47.0, std=1.5, cell=5),
    PlaygroundPreset("broad", "sparse_epoch", mean=30.0, std=15.0, cell=13),
    # The sparse cells' fields nearly coincide, so which of them fired says
    # little; the p-value misses a conflict that HPD overlap catches.
    PlaygroundPreset("pvalue_miss", "sparse_epoch", mean=50.0, std=3.0, cell=13),
)


def playground_payload(
    config: Figure3Config,
    sparse_centers: NDArray[np.floating],
    figure03_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the data for the two-distribution playground.

    Parameters
    ----------
    config : Figure3Config
        Supplies the track, place fields, and rates.
    sparse_centers : np.ndarray, shape (sparse_cell_count,)
        Sparse-population field centers from the Figure-3 simulation.
    figure03_summary : mapping
        Parsed ``figure03_summary.json``; supplies the simulation flag rules.

    Returns
    -------
    dict
        ``position_bins``, ``cell_centers``, ``ensembles`` (rates and
        selectable cells), ``presets`` (:data:`PLAYGROUND_PRESETS`, keyed by
        ``preset_id``), ``flag_rules`` with their ``flag_threshold_text``
        (:func:`flag_threshold_text`), and the HPD ``coverage``.
    """
    return {
        "position_bins": config.position_bins.tolist(),
        "cell_centers": all_place_field_centers(config, sparse_centers).tolist(),
        "ensembles": [
            {
                "ensemble_id": ensemble.ensemble_id,
                "rates": ensemble.rates.tolist(),
                "selectable_cells": list(ensemble.selectable_cells),
            }
            for ensemble in playground_ensembles(config, sparse_centers)
        ],
        "presets": {
            preset.preset_id: {
                "ensemble": preset.ensemble_id,
                "mean": preset.mean,
                "std": preset.std,
                "cell": preset.cell,
            }
            for preset in PLAYGROUND_PRESETS
        },
        "flag_rules": figure03_summary["flag_rules"],
        "flag_threshold_text": flag_threshold_text(figure03_summary),
        "coverage": HPD_COVERAGE,
    }


def _single_event_diagnostics(
    predictive: NDArray[np.floating], rates: NDArray[np.floating], cell: int
) -> dict[str, float]:
    """Run the paper's per-spike diagnostics on one event at time 0."""
    diagnostics = compute_spike_event_diagnostics_from_rates(
        predictive[np.newaxis, :],
        rates,
        np.array([0], dtype=np.intp),
        np.array([cell], dtype=np.intp),
        coverage=HPD_COVERAGE,
        include_dense_matrices=False,
    )
    return {metric: float(getattr(diagnostics, f"event_{metric}")[0]) for metric in METRIC_NAMES}


def metric_parity_fixture(
    config: Figure3Config, sparse_centers: NDArray[np.floating]
) -> dict[str, Any]:
    """Build reference cases for the JavaScript port of the per-spike diagnostics.

    Each playground ensemble is evaluated for every selectable cell under
    consistent, inconsistent, nested (broad prediction), and near-point
    Gaussian predictions, plus hand-built predictive distributions with ties
    and flat regions that exercise the HPD cutoff.

    Returns
    -------
    dict
        ``position_bins``; ``ensembles`` (rates per ensemble); ``gaussian_cases``
        (mean, std, and the resulting predictive, which the JavaScript must
        reproduce); ``predictives`` (every predictive tested); and ``cases``
        (predictive index, ensemble index, cell, expected diagnostics).
    """
    position_bins = config.position_bins
    ensembles = playground_ensembles(config, sparse_centers)
    n_bins = position_bins.size
    gaussian_cases = [
        {
            "mean": mean,
            "std": std,
            "predictive": gaussian_predictive(position_bins, mean, std).tolist(),
        }
        for mean in (0.0, 12.5, 30.0, 50.0, 71.0, 100.0)
        for std in (0.4, 3.0, 10.0, 30.0, 200.0)
    ]
    predictives = [np.asarray(case["predictive"]) for case in gaussian_cases]
    predictives.append(np.full(n_bins, 1.0 / n_bins))  # flat: every bin tied
    two_bumps = gaussian_predictive(position_bins, 20.0, 4.0) + gaussian_predictive(
        position_bins, 80.0, 4.0
    )
    predictives.append(two_bumps / two_bumps.sum())
    plateau = np.where((position_bins >= 40.0) & (position_bins <= 60.0), 1.0, 0.0)
    predictives.append(plateau / plateau.sum())  # exact zeros outside a tied plateau
    staircase = np.repeat(np.arange(1.0, 1.0 + np.ceil(n_bins / 10)), 10)[:n_bins]
    predictives.append(staircase / staircase.sum())  # blocks of ten tied values

    cases = [
        {
            "predictive": predictive_index,
            "ensemble": ensemble_index,
            "cell": cell,
            "expected": _single_event_diagnostics(predictive, ensemble.rates, cell),
        }
        for predictive_index, predictive in enumerate(predictives)
        for ensemble_index, ensemble in enumerate(ensembles)
        for cell in ensemble.selectable_cells
    ]
    return {
        "position_bins": position_bins.tolist(),
        "coverage": HPD_COVERAGE,
        "ensembles": [ensemble.rates.tolist() for ensemble in ensembles],
        "gaussian_cases": gaussian_cases,
        "predictives": [predictive.tolist() for predictive in predictives],
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Condition player (Figure 3)
# ---------------------------------------------------------------------------


def condition_payloads(
    sim: Figure3SimulationResult,
    figure03_summary: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per-condition display data from one Figure-3 realization.

    Each condition is shown over its window in :data:`CONDITION_WINDOWS`.

    Parameters
    ----------
    sim : Figure3SimulationResult
        The realization shown in Figure 3a.
    figure03_summary : mapping
        Parsed ``figure03_summary.json``: flag rules plus per-condition median
        flag percentages and decoding errors across realizations.

    Returns
    -------
    dict
        Condition id -> JSON-ready payload.
    """
    diagnostics = sim.diagnostics
    n_time = diagnostics.predictive.shape[0]
    position_bins = np.asarray(sim.position_bins, dtype=np.float64)
    cell_centers = _rounded(all_place_field_centers(sim.config, sim.sparse_place_field_centers), 2)
    conditions = conditions_by_id(sim.config)
    order = list(figure03_summary["condition_order"])
    metric_order = list(figure03_summary["metric_order"])
    flag_percentages = np.asarray(figure03_summary["median_flag_percentages"], dtype=np.float64)
    decoding_error = np.asarray(figure03_summary["median_decoding_error"], dtype=np.float64)
    flag_rules = figure03_summary["flag_rules"]
    error_row = list(figure03_summary["error_metric_order"]).index("median_absolute_error")

    event_time = np.asarray(diagnostics.event_time_ind)
    event_cell = np.asarray(diagnostics.event_cell_ind)
    event_likelihood = np.asarray(diagnostics.event_likelihood, dtype=np.float64)

    predictive_range = (
        0.0,
        float(np.nanquantile(diagnostics.predictive, PREDICTIVE_VMAX_QUANTILE)),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for window in CONDITION_WINDOWS:
        if not 0 <= window.start < window.stop <= n_time:
            raise ValueError(f"{window} lies outside the {n_time}-step timeline")
        condition = conditions[window.condition_id]
        column = order.index(window.condition_id)
        in_window = (event_time >= window.start) & (event_time < window.stop)
        likelihood_rows, likelihood_index = np.unique(
            event_likelihood[in_window], axis=0, return_inverse=True
        )
        payloads[window.condition_id] = {
            "condition_id": window.condition_id,
            "model_component": condition.model_component,
            "start": window.start,
            "stop": window.stop,
            "step_seconds": STEP_SECONDS,
            "scored_windows": [
                [max(t0, window.start), min(t1, window.stop)]
                for t0, t1 in condition.step_windows
                if t0 < window.stop and t1 > window.start
            ],
            "position_bins": position_bins.tolist(),
            "true_position": _rounded(sim.true_position[window.start : window.stop], 2),
            "predictive": heatmap_payload(
                diagnostics.predictive[window.start : window.stop], predictive_range
            ),
            "cell_centers": cell_centers,
            "events": {
                "t": (event_time[in_window] - window.start).tolist(),
                "cell": event_cell[in_window].tolist(),
                "likelihood_row": np.asarray(likelihood_index).reshape(-1).tolist(),
                **_event_payload(diagnostics, in_window, flag_rules),
            },
            "likelihood_rows": encode_display_rows(likelihood_rows),
            # Formatted with the manuscript's rounding policy.
            "summary": {
                "median_flag_percent_text": {
                    metric: whole_percent(flag_percentages[metric_order.index(metric), column])
                    for metric in METRIC_NAMES
                },
                "median_absolute_error_text": significant(decoding_error[error_row, column]),
            },
        }
    for condition_id in conditions:
        if condition_id not in payloads:
            raise ValueError(f"No display window for condition {condition_id!r}")
    return payloads


# ---------------------------------------------------------------------------
# Recording comparison (Figure 4)
# ---------------------------------------------------------------------------


def recording_payload(
    render_data: Figure4RenderData,
    figure04_summary: Mapping[str, Any],
    detail_window: Figure4DetailWindow = FIGURE04_DETAIL_WINDOW,
) -> dict[str, Any]:
    """Build the Figure-4 detail window under both decoders.

    Parameters
    ----------
    render_data : Figure4RenderData
        Recording plus cached decode, as used to render Figure 4.
    figure04_summary : mapping
        Parsed ``figure04_summary.json``: flag rules and decode-cache identity.
    detail_window : Figure4DetailWindow, default ``FIGURE04_DETAIL_WINDOW``
        Samples shown; defaults to the window in Figure 4a/b.

    Raises
    ------
    ValueError
        If the summary's decode or diagnostics fingerprint differs from the
        provenance of the decode being exported (``render_data.cache_provenance``),
        so the window would be labeled with another decode's identity.
    """
    caches = figure04_summary["provenance"]["figure04_caches"]
    exported = render_data.cache_provenance
    mismatched = [
        key
        for key, exported_value in (
            ("fingerprint_sha256", exported.fingerprint_sha256),
            ("diagnostics_fingerprint_sha256", exported.diagnostics_fingerprint_sha256),
        )
        if caches[key] != exported_value
    ]
    if mismatched:
        raise ValueError(
            "figure04_summary.json provenance.figure04_caches does not match the decode "
            f"being exported ({', '.join(mismatched)} differ); regenerate Figure 4 from "
            "the same decode and diagnostics caches before exporting the recording window."
        )
    window = detail_window.to_slice(render_data.time.size)
    decode = render_data.decode_results
    time = np.asarray(render_data.time, dtype=np.float64)
    # Decoder bins are left-closed; the window spans [time[start], time[stop]).
    t0 = float(time[window.start])
    t_end = (
        float(time[window.stop])
        if window.stop < time.size
        else float(time[-1] + (time[-1] - time[-2]))
    )
    place_fields = np.asarray(decode.diagnostic_place_fields, dtype=np.float64)
    mean_likelihood, has_spikes = mean_event_likelihood_by_time(
        decode.spike_counts[window], place_fields
    )
    flag_rules = figure04_summary["flag_rules"]

    models: dict[str, Any] = {}
    for model, results, diagnostics in (
        (CONTINUOUS, decode.continuous_results, decode.continuous_diagnostics),
        (
            CONTINUOUS_FRAGMENTED,
            decode.continuous_fragmented_results,
            decode.continuous_fragmented_diagnostics,
        ),
    ):
        predictive = marginal_position_distribution(results.isel(time=window), "predictive")
        if predictive.shape[1] != place_fields.shape[1]:
            raise ValueError(
                f"{model.id} predictive has {predictive.shape[1]} bins; "
                f"place fields have {place_fields.shape[1]}"
            )
        if diagnostics.event_time is None:
            raise ValueError(f"{model.id} diagnostics lack exact event times")
        predictive = np.nan_to_num(predictive)
        low, high = np.percentile(predictive, FIGURE04_PREDICTIVE_PERCENTILES)
        # Events belong to the decoder bin that counted them (event_time_ind),
        # as in Figure 4, not to the bin nearest their exact spike time.
        event_bin = np.asarray(diagnostics.event_time_ind)
        in_window = (event_bin >= window.start) & (event_bin < window.stop)
        event_time = np.asarray(diagnostics.event_time, dtype=np.float64)
        models[model.id] = {
            "label": model.label,
            "short_label": model.short_label,
            "predictive": heatmap_payload(predictive, (float(low), float(high))),
            "events": {
                "bin": (event_bin[in_window] - window.start).tolist(),
                "t": _rounded(event_time[in_window] - t0, 4),
                "cell": np.asarray(diagnostics.event_cell_ind)[in_window].tolist(),
                **_event_payload(diagnostics, in_window, flag_rules),
            },
        }

    spike_times = render_data.recording.spike_times
    cell_rank = np.argsort(np.argsort(decode.place_field_peaks))
    return {
        "time": _rounded(time[window] - t0, 4),
        "position_bins": _rounded(decode.diagnostic_position_bins, 2),
        "linear_position": _rounded(render_data.linear_position[window], 2),
        "likelihood": encode_display_rows(mean_likelihood),
        "has_spikes": has_spikes.tolist(),
        # Each cell's normalized single-event likelihood (one row per cell).
        "cell_likelihoods": encode_display_rows(ssc.event_likelihood(place_fields)),
        "cell_rank": cell_rank.tolist(),
        "spike_times": [
            _rounded(times[(times >= t0) & (times < t_end)] - t0, 4) for times in spike_times
        ],
        "models": models,
        "flag_rules": flag_rules,
        # Identify the decode and the diagnostics this window was exported from.
        "decode_cache_fingerprint": caches["fingerprint_sha256"],
        "diagnostics_fingerprint": caches["diagnostics_fingerprint_sha256"],
    }


# ---------------------------------------------------------------------------
# Manifest and export
# ---------------------------------------------------------------------------


def page_values(figure04_summary: Mapping[str, Any]) -> dict[str, str]:
    """Numbers the page text states that the manuscript does not.

    The page fills them into ``data-macro`` placeholders alongside the
    manuscript's macros.

    Parameters
    ----------
    figure04_summary : mapping
        Parsed ``figure04_summary.json``: the detail window and decoder bin rate.

    Returns
    -------
    dict
        ``RecordingWindowSecondsWord``: the Figure-4 detail window's length in
        seconds, spelled out ("this two-second window").

    Raises
    ------
    ValueError
        If the window is not a whole number of seconds.
    """
    half_width = figure04_summary["detail_window"]["half_width_samples"]
    sampling_frequency_hz = figure04_summary["configuration"]["decoder"]["sampling_frequency_hz"]
    seconds = 2 * half_width / sampling_frequency_hz
    if not float(seconds).is_integer():
        raise ValueError(
            f"The page spells the recording window in whole seconds; it is {seconds} s"
        )
    return {"RecordingWindowSecondsWord": cardinal_word(int(seconds))}


def manifest_payload(
    figure03_summary: dict[str, Any], figure04_summary: dict[str, Any]
) -> dict[str, Any]:
    """Page-wide data: reported values, flag rules, colormaps, and conditions.

    ``macros`` holds the manuscript's macros and ``page_values`` the numbers
    only the page states (:func:`page_values`). ``flag_threshold_text`` holds
    the simulation's thresholds as the page prints them
    (:func:`flag_threshold_text`). Each ``conditions`` entry names
    a condition's data file and its tab title
    (:attr:`~statespacecheck_paper.figure03_summary.Figure3SummaryCondition.title`).
    """
    conditions = conditions_by_id(Figure3Config())
    macros = {
        macro.name: macro.value
        for _, section in macro_sections(figure03_summary, figure04_summary)
        for macro in section
    }
    return {
        "macros": macros,
        "page_values": page_values(figure04_summary),
        "flag_rules": {"simulation": figure03_summary["flag_rules"]},
        "flag_threshold_text": {"simulation": flag_threshold_text(figure03_summary)},
        "colormaps": {
            "predictive": colormap_lut(CMAP_PREDICTIVE),
            "likelihood": colormap_lut(CMAP_LIKELIHOOD),
        },
        "conditions": [
            {
                "condition_id": window.condition_id,
                "title": conditions[window.condition_id].title,
                "file": f"condition_{window.condition_id}.json",
            }
            for window in CONDITION_WINDOWS
        ],
    }


def write_site_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write compact JSON (no whitespace) with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def export_site_data(*, include_recording: bool = True) -> list[Path]:
    """Regenerate every website data file from the committed summaries.

    Page data goes to ``SITE_DATA_DIR`` and the JavaScript parity fixture to
    ``PARITY_FIXTURE_PATH``.

    Parameters
    ----------
    include_recording : bool, default True
        Also export the Figure-4 recording window. This needs the Figure-4 input
        file in ``DATA_PATH``. It reuses the Figure-4 decode and diagnostics
        caches when their fingerprints match and otherwise rebuilds them as
        ``generate_figure04.py`` does: a stale or missing decode cache refits
        both models (several minutes) and writes the ~8 GB decode cache. Pass
        False on a machine without the input file, or when the recording
        outputs are unchanged, to leave the committed ``recording.json`` untouched.

    Returns
    -------
    list of Path
        Files written.
    """
    figure03_summary = json.loads(FIGURE03_SUMMARY_PATH.read_text(encoding="utf-8"))
    figure04_summary = json.loads(FIGURE04_SUMMARY_PATH.read_text(encoding="utf-8"))
    config = Figure3Config()
    simulation = run_figure03_simulation(config)
    sparse_centers = np.asarray(simulation.sparse_place_field_centers, dtype=np.float64)
    written = [
        write_site_json(
            SITE_DATA_DIR / "manifest.json", manifest_payload(figure03_summary, figure04_summary)
        ),
        write_site_json(
            SITE_DATA_DIR / "playground.json",
            playground_payload(config, sparse_centers, figure03_summary),
        ),
        write_site_json(PARITY_FIXTURE_PATH, metric_parity_fixture(config, sparse_centers)),
        write_site_json(SITE_DATA_DIR / "filter.json", filter_explainer_payload(config)),
    ]
    for condition_id, payload in condition_payloads(simulation, figure03_summary).items():
        written.append(write_site_json(SITE_DATA_DIR / f"condition_{condition_id}.json", payload))
    if include_recording:
        render_data = prepare_figure04_render_data(
            Figure4Config(),
            Figure4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH),
        )
        written.append(
            write_site_json(
                SITE_DATA_DIR / "recording.json", recording_payload(render_data, figure04_summary)
            )
        )
    return written
