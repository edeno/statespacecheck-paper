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
  :func:`statespacecheck_paper.diagnostics.compute_spike_event_diagnostics_from_rates`.
  :func:`metric_parity_fixture` writes reference cases computed by that Python
  function, and ``site/tests/metrics.test.mjs`` checks the port against them.
- **Scenario player** — one display window per Figure-3 condition from the
  seed-``config.random_seed`` realization shown in Figure 3a.
- **Replay comparison** — the Figure-4 detail window under both decoders.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import matplotlib as mpl
import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.decoding import decode_with_diagnostics
from statespacecheck_paper.diagnostics import (
    DecodingDiagnostics,
    compute_normalized_event_likelihood,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.figure03_generation import FIGURE03_CONDITION_IDS
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    build_figure03_rate_tables,
    run_figure03_simulation,
)
from statespacecheck_paper.figure03_summary import build_summary_conditions
from statespacecheck_paper.figure04_cache import Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_diagnostics import mean_per_spike_likelihood_by_time
from statespacecheck_paper.figure04_generation import FIGURE4_DETAIL_WINDOW
from statespacecheck_paper.figure04_layout import Figure4DetailWindow
from statespacecheck_paper.figure04_place_fields import get_state_marginalized_posterior
from statespacecheck_paper.figure04_workflow import Figure4RenderData, prepare_figure04_render_data
from statespacecheck_paper.number_format import significant, whole_percent
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH, DATA_PATH
from statespacecheck_paper.reported_values import (
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    macro_sections,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    place_field_rates,
    simulate_spikes_position_tuned,
)
from statespacecheck_paper.style import CMAP_LIKELIHOOD, CMAP_POSTERIOR, MetricName

SITE_DATA_DIR = Path("site/data")
PARITY_FIXTURE_PATH = Path("site/tests/fixtures/metric_parity.json")

# Per-event metric fields, in the order the site lists them.
METRIC_FIELDS: tuple[MetricName, ...] = ("hpd_overlap", "predictive_pvalue", "kl_divergence")

# Significant figures kept for per-event diagnostic values. Significant figures,
# not decimal places, so a small predictive p-value keeps its magnitude.
EVENT_VALUE_SIGNIFICANT_FIGURES = 4

# Entries in each exported colormap lookup table.
COLORMAP_LUT_SIZE = 64

# Probability mass of the HPD regions, as throughout the paper.
HPD_COVERAGE = 0.95

# Color scale of the predictive heatmaps, matching the paper's figures. Figure 3a
# (figure03_plotting) spans 0 to the 97.5th percentile of the whole session;
# Figure 4 (xarray's ``robust=True``) spans the 2nd to 98th percentiles of the
# plotted window.
FIGURE3_PREDICTIVE_VMAX_QUANTILE = 0.975
FIGURE4_PREDICTIVE_PERCENTILES = (2.0, 98.0)


@dataclass(frozen=True)
class ScenarioWindow:
    """Display window for one Figure-3 condition in the scenario player.

    Parameters
    ----------
    condition_id : str
        One of ``FIGURE03_CONDITION_IDS``.
    start, stop : int
        Half-open range of simulation steps shown.
    """

    condition_id: str
    start: int
    stop: int


# Display choices, like ``FIGURE4_DETAIL_WINDOW``; each overlaps its condition's
# scored step windows (checked by the test suite). Abrupt conditions (remap,
# history dependence, replay) start a few hundred clean steps before onset so
# the change is visible. Drift builds up gradually, so its window sits mid-phase,
# where the flag rates are close to the across-realization medians. The sparse
# population fires only a handful of spikes, so its window spans the whole phase.
SCENARIO_WINDOWS: tuple[ScenarioWindow, ...] = (
    ScenarioWindow("well_specified", 11_500, 13_000),
    ScenarioWindow("remap", 5_700, 7_200),
    ScenarioWindow("history_dependent", 13_700, 15_200),
    ScenarioWindow("replay", 18_700, 21_300),
    ScenarioWindow("drift", 24_000, 25_500),
    ScenarioWindow("sparse_population", 29_700, 32_000),
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


def decode_display_rows(encoded: str, n_bins: int) -> NDArray[np.uint8]:
    """Invert :func:`encode_display_rows` to a ``(n_rows, n_bins)`` array."""
    flat = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    if flat.size % n_bins:
        raise ValueError(f"{flat.size} encoded values do not divide into rows of {n_bins}")
    return flat.reshape(-1, n_bins)


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


def _rounded(values: NDArray[np.floating], decimals: int) -> list[float]:
    """Round finite values for JSON, refusing NaN/inf (which JSON cannot hold)."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite to export as JSON")
    rounded: list[float] = np.round(array, decimals).tolist()
    return rounded


def _rounded_significant(values: NDArray[np.floating], digits: int) -> list[float]:
    """Round finite values to ``digits`` significant figures for JSON."""
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite to export as JSON")
    return [float(f"{value:.{digits}g}") for value in array.tolist()]


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
    array = np.asarray(values, dtype=np.float64)
    if comparison == "less_than_or_equal":
        return array <= threshold
    if comparison == "greater_than_or_equal":
        return array >= threshold
    raise ValueError(f"Unknown flag comparison {comparison!r}")


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
    for metric in METRIC_FIELDS:
        values = np.asarray(getattr(diagnostics, f"event_{metric}"))[selection]
        payload[metric] = _rounded_significant(values, EVENT_VALUE_SIGNIFICANT_FIGURES)
        if metric in flag_rules:
            flags[metric] = flag_events(values, flag_rules[metric]).tolist()
    payload["flagged"] = flags
    return payload


def colormap_lut(name: str, n: int = COLORMAP_LUT_SIZE) -> list[str]:
    """Sample a matplotlib colormap as ``n`` hex colors, low to high."""
    cmap = mpl.colormaps[name].resampled(n)
    return [mpl.colors.to_hex(cmap(i)) for i in range(n)]


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
        ``place_field_rate_scale`` of the cells: expected spikes per step.
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


# A display choice, like ``SCENARIO_WINDOWS``: a seed whose sequence shows the
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
        Expected spikes per step of each cell, for both simulation and decoding.
    true_position : np.ndarray, shape (n_steps,)
    spike_counts : np.ndarray, shape (n_steps, n_cells)
    conflict_cells : tuple of int
        The cells that fire at ``conflict_step``.
    decoded : DecodingDiagnostics
        Output of :func:`decode_with_diagnostics` for ``spike_counts``.
    """

    position_bins: NDArray[np.float64]
    rates: NDArray[np.float64]
    true_position: NDArray[np.float64]
    spike_counts: NDArray[np.int_]
    conflict_cells: tuple[int, ...]
    decoded: DecodingDiagnostics


def filter_explainer_sequence(
    config: Figure3Config, explainer: FilterExplainerConfig = FILTER_EXPLAINER
) -> FilterExplainerSequence:
    """Simulate and decode the explainer's spike train.

    The spikes come from the simulation's generator, given the smooth run. The
    sequence is edited in two places: the first step holds one spike, from the
    cell whose field is centered nearest the animal, so the demonstration opens
    on a spike whose prediction is the flat initial distribution; and
    ``conflict_step`` holds the inconsistent spikes described in
    :class:`FilterExplainerConfig`.

    Parameters
    ----------
    config : Figure3Config
        Supplies the track and the place-field centers and width.
    explainer : FilterExplainerConfig
        Supplies the run, rates, movement model, and the scripted edits.
    """
    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
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
    conflict_cells = tuple(sorted(int(c) for c in np.argsort(np.abs(centers - target))[:2]))
    counts[explainer.conflict_step] = 0
    counts[explainer.conflict_step, list(conflict_cells)] = 1

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
        baseline_firing_rates=rates,
    )
    return FilterExplainerSequence(
        position_bins=position_bins,
        rates=rates,
        true_position=np.asarray(position, dtype=np.float64),
        spike_counts=counts,
        conflict_cells=conflict_cells,
        decoded=decoded,
    )


def _moments(
    distributions: NDArray[np.floating], position_bins: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and standard deviation of each row, shape (n_rows,) each."""
    mean = np.asarray(distributions @ position_bins, dtype=np.float64)
    variance = np.einsum("tb,tb->t", distributions, (position_bins[None, :] - mean[:, None]) ** 2)
    return mean, np.sqrt(variance)


def filter_explainer_payload(
    config: Figure3Config, explainer: FilterExplainerConfig = FILTER_EXPLAINER
) -> dict[str, Any]:
    """Build the data for the filter explainer's time stepper.

    Parameters
    ----------
    config : Figure3Config
    explainer : FilterExplainerConfig

    Returns
    -------
    dict
        ``position_bins``, ``cell_centers``, ``true_position``; ``events``
        (time step, cell, and HPD overlap of each spike); ``predictive`` and
        ``posterior`` (display rows on one shared scale, see
        :func:`heatmap_payload`); ``likelihood`` (each step's normalized
        likelihood, the product of the fired cells' fields and the exposure
        term, as display rows); ``place_fields`` (display rows, one per cell);
        ``exposure`` (``exp(-Λ(x))``, scaled to its maximum, where ``Λ`` is the
        cells' total expected count); ``moments`` (mean and SD of the
        prediction and posterior at each step); and the sequence's parameters.
    """
    sequence = filter_explainer_sequence(config, explainer)
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
        "likelihood": encode_display_rows(decoded.likelihood),
        "place_fields": encode_display_rows(sequence.rates.T),
        "exposure": _rounded(exposure, 4),
        "moments": {
            "predictive_mean": _rounded(predictive_mean, 2),
            "predictive_sd": _rounded(predictive_sd, 2),
            "posterior_mean": _rounded(posterior_mean, 2),
            "posterior_sd": _rounded(posterior_sd, 2),
        },
        "conflict_step": explainer.conflict_step,
        "step_std": explainer.step_std,
        "peak_rate": float(sequence.rates.max()),
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[PlaygroundEnsemble, ...]]:
    """Position grid, cell centers, and the two Figure-3 decoder rate tables.

    Parameters
    ----------
    config : Figure3Config
    sparse_centers : np.ndarray, shape (sparse_cell_count,)
        Sparse-population field centers, as returned on
        ``Figure3SimulationResult.sparse_place_field_centers``.

    Returns
    -------
    position_bins : np.ndarray, shape (n_bins,)
    cell_centers : np.ndarray, shape (n_cells,)
        Ordinary place-field centers followed by the sparse-population centers.
    ensembles : tuple of PlaygroundEnsemble
        Place cells, then the sparse epoch.
    """
    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
    position_bins = config.position_bins
    sparse = np.asarray(sparse_centers, dtype=np.float64)
    tables = build_figure03_rate_tables(position_bins, config.place_field_centers, sparse, config)
    n_place_cells = len(config.place_field_centers)
    n_cells = n_place_cells + sparse.size
    cell_centers = np.append(np.asarray(config.place_field_centers, dtype=np.float64), sparse)
    ensembles = (
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
    return position_bins, cell_centers, ensembles


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
    """
    position_bins, cell_centers, ensembles = playground_ensembles(config, sparse_centers)
    return {
        "position_bins": position_bins.tolist(),
        "cell_centers": cell_centers.tolist(),
        "ensembles": [
            {
                "ensemble_id": ensemble.ensemble_id,
                "rates": ensemble.rates.tolist(),
                "selectable_cells": list(ensemble.selectable_cells),
            }
            for ensemble in ensembles
        ],
        "flag_rules": figure03_summary["flag_rules"],
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
    return {metric: float(getattr(diagnostics, f"event_{metric}")[0]) for metric in METRIC_FIELDS}


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
    position_bins, _, ensembles = playground_ensembles(config, sparse_centers)
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
# Scenario player (Figure 3)
# ---------------------------------------------------------------------------


def _all_cell_centers(sim: Figure3SimulationResult) -> NDArray[np.float64]:
    """Field centers for every decoded cell (ordinary then sparse), for sorting."""
    config = sim.config
    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
    return np.append(
        np.asarray(config.place_field_centers, dtype=np.float64),
        np.asarray(sim.sparse_place_field_centers, dtype=np.float64),
    )


def scenario_payloads(
    sim: Figure3SimulationResult,
    figure03_summary: Mapping[str, Any],
    windows: Sequence[ScenarioWindow] = SCENARIO_WINDOWS,
) -> dict[str, dict[str, Any]]:
    """Per-condition display data from one Figure-3 realization.

    Parameters
    ----------
    sim : Figure3SimulationResult
        The realization shown in Figure 3a.
    figure03_summary : mapping
        Parsed ``figure03_summary.json``: flag rules plus per-condition median
        flag percentages and decoding errors across realizations.
    windows : sequence of ScenarioWindow
        One display window per condition.

    Returns
    -------
    dict
        Condition id -> JSON-ready payload.
    """
    diagnostics = sim.diagnostics
    n_time = diagnostics.predictive.shape[0]
    position_bins = np.asarray(sim.position_bins, dtype=np.float64)
    conditions = dict(
        zip(FIGURE03_CONDITION_IDS, build_summary_conditions(sim.config), strict=True)
    )
    order = list(figure03_summary["condition_order"])
    metric_order = list(figure03_summary["metric_order"])
    flag_percentages = np.asarray(figure03_summary["median_flag_percentages"], dtype=np.float64)
    decoding_error = np.asarray(figure03_summary["median_decoding_accuracy"], dtype=np.float64)
    flag_rules = figure03_summary["flag_rules"]
    error_row = list(figure03_summary["accuracy_metric_order"]).index("median_absolute_error")

    posterior = np.asarray(diagnostics.posterior, dtype=np.float64)
    posterior_mean = (posterior @ position_bins) / posterior.sum(axis=1)

    event_time = np.asarray(diagnostics.event_time_ind)
    event_cell = np.asarray(diagnostics.event_cell_ind)
    per_spike_likelihood = np.asarray(diagnostics.per_spike_likelihood, dtype=np.float64)

    predictive_range = (
        0.0,
        float(np.nanquantile(diagnostics.predictive, FIGURE3_PREDICTIVE_VMAX_QUANTILE)),
    )
    payloads: dict[str, dict[str, Any]] = {}
    for window in windows:
        if not 0 <= window.start < window.stop <= n_time:
            raise ValueError(f"{window} lies outside the {n_time}-step timeline")
        condition = conditions[window.condition_id]
        column = order.index(window.condition_id)
        in_window = (event_time >= window.start) & (event_time < window.stop)
        likelihood_rows, likelihood_index = np.unique(
            per_spike_likelihood[in_window], axis=0, return_inverse=True
        )
        payloads[window.condition_id] = {
            "condition_id": window.condition_id,
            "label": figure03_summary["condition_labels"][column],
            "model_component": condition.model_component,
            "start": window.start,
            "stop": window.stop,
            "scored_windows": [
                [max(t0, window.start), min(t1, window.stop)]
                for t0, t1 in condition.step_windows
                if t0 < window.stop and t1 > window.start
            ],
            "position_bins": position_bins.tolist(),
            "true_position": _rounded(sim.true_position[window.start : window.stop], 2),
            "posterior_mean": _rounded(posterior_mean[window.start : window.stop], 2),
            "predictive": heatmap_payload(
                diagnostics.predictive[window.start : window.stop], predictive_range
            ),
            "cell_centers": _rounded(_all_cell_centers(sim), 2),
            "events": {
                "t": (event_time[in_window] - window.start).tolist(),
                "cell": event_cell[in_window].tolist(),
                "likelihood_row": np.asarray(likelihood_index).reshape(-1).tolist(),
                **_event_payload(diagnostics, in_window, flag_rules),
            },
            "likelihood_rows": encode_display_rows(likelihood_rows),
            "summary": {
                "median_flag_percent": {
                    metric: float(flag_percentages[metric_order.index(metric), column])
                    for metric in METRIC_FIELDS
                },
                "median_absolute_error": float(decoding_error[error_row, column]),
                # Formatted with the manuscript's rounding policy.
                "median_flag_percent_text": {
                    metric: whole_percent(flag_percentages[metric_order.index(metric), column])
                    for metric in METRIC_FIELDS
                },
                "median_absolute_error_text": significant(decoding_error[error_row, column]),
            },
        }
    for condition_id in FIGURE03_CONDITION_IDS:
        if condition_id not in payloads:
            raise ValueError(f"No display window for condition {condition_id!r}")
    return payloads


# ---------------------------------------------------------------------------
# Replay comparison (Figure 4)
# ---------------------------------------------------------------------------


def replay_payload(
    render_data: Figure4RenderData,
    figure04_summary: Mapping[str, Any],
    detail_window: Figure4DetailWindow = FIGURE4_DETAIL_WINDOW,
) -> dict[str, Any]:
    """Build the Figure-4 detail window under both decoders.

    Parameters
    ----------
    render_data : Figure4RenderData
        Recording plus cached decode, as used to render Figure 4.
    figure04_summary : mapping
        Parsed ``figure04_summary.json``: flag rules and decode-cache identity.
    detail_window : Figure4DetailWindow, default ``FIGURE4_DETAIL_WINDOW``
        Samples shown; defaults to the window in Figure 4a/b.
    """
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
    mean_likelihood, has_spikes = mean_per_spike_likelihood_by_time(
        decode.spike_counts[window], place_fields
    )
    flag_rules = figure04_summary["flag_rules"]

    models: dict[str, Any] = {}
    for name, results, diagnostics in (
        ("continuous", decode.continuous_results, decode.continuous_diagnostics),
        (
            "continuous_fragmented",
            decode.continuous_fragmented_results,
            decode.continuous_fragmented_diagnostics,
        ),
    ):
        predictive = get_state_marginalized_posterior(results.isel(time=window), "predictive")
        if predictive.shape[1] != place_fields.shape[1]:
            raise ValueError(
                f"{name} predictive has {predictive.shape[1]} bins; "
                f"place fields have {place_fields.shape[1]}"
            )
        if diagnostics.event_time is None:
            raise ValueError(f"{name} diagnostics lack exact event times")
        predictive = np.nan_to_num(predictive)
        low, high = np.percentile(predictive, FIGURE4_PREDICTIVE_PERCENTILES)
        # Events belong to the decoder bin that counted them (event_time_ind),
        # as in Figure 4, not to the bin nearest their exact spike time.
        event_bin = np.asarray(diagnostics.event_time_ind)
        in_window = (event_bin >= window.start) & (event_bin < window.stop)
        event_time = np.asarray(diagnostics.event_time, dtype=np.float64)
        models[name] = {
            "predictive": heatmap_payload(predictive, (float(low), float(high))),
            "events": {
                "bin": (event_bin[in_window] - window.start).tolist(),
                "t": _rounded(event_time[in_window] - t0, 4),
                "cell": np.asarray(diagnostics.event_cell_ind)[in_window].tolist(),
                **_event_payload(diagnostics, in_window, flag_rules),
            },
        }

    spike_times = render_data.recording.spike_times
    decode_cache = figure04_summary["provenance"]["figure04_decode_cache"]
    unit_rank = np.argsort(np.argsort(decode.place_field_peaks))
    return {
        "time": _rounded(time[window] - t0, 4),
        "position_bins": _rounded(decode.diagnostic_position_bins, 2),
        "linear_position": _rounded(render_data.linear_position[window], 2),
        "likelihood": encode_display_rows(mean_likelihood),
        "has_spikes": has_spikes.tolist(),
        # Each unit's normalized single-event likelihood (one row per unit).
        "unit_likelihoods": encode_display_rows(compute_normalized_event_likelihood(place_fields)),
        "unit_rank": unit_rank.tolist(),
        "spike_times": [
            _rounded(times[(times >= t0) & (times < t_end)] - t0, 4) for times in spike_times
        ],
        "models": models,
        "flag_rules": flag_rules,
        # Identify the decode and the diagnostics this window was exported from.
        "decode_cache_fingerprint": decode_cache["fingerprint_sha256"],
        "diagnostics_fingerprint": decode_cache["diagnostics_fingerprint_sha256"],
    }


# ---------------------------------------------------------------------------
# Manifest and export
# ---------------------------------------------------------------------------


def manifest_payload(
    figure03_summary: dict[str, Any], figure04_summary: dict[str, Any]
) -> dict[str, Any]:
    """Page-wide data: reported-value macros, flag rules, colormaps, and conditions."""
    macros = {
        macro.name: macro.value
        for _, section in macro_sections(figure03_summary, figure04_summary)
        for macro in section
    }
    return {
        "macros": macros,
        "flag_rules": {
            "simulation": figure03_summary["flag_rules"],
            "recording": figure04_summary["flag_rules"],
        },
        "colormaps": {
            "predictive": colormap_lut(CMAP_POSTERIOR),
            "likelihood": colormap_lut(CMAP_LIKELIHOOD),
        },
        "scenarios": [
            {
                "condition_id": window.condition_id,
                "label": figure03_summary["condition_labels"][
                    figure03_summary["condition_order"].index(window.condition_id)
                ],
                "file": f"scenario_{window.condition_id}.json",
            }
            for window in SCENARIO_WINDOWS
        ],
    }


def write_site_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write compact JSON (no whitespace) with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return path


def export_site_data(
    out_dir: Path = SITE_DATA_DIR,
    *,
    fixture_path: Path = PARITY_FIXTURE_PATH,
    include_recording: bool = True,
) -> list[Path]:
    """Regenerate every website data file from the committed summaries.

    Parameters
    ----------
    out_dir : Path, default ``SITE_DATA_DIR``
        Destination for the page data (manifest, playground, players).
    fixture_path : Path, default ``PARITY_FIXTURE_PATH``
        Destination for the JavaScript parity fixture.
    include_recording : bool, default True
        Also export the Figure-4 replay window. This needs the derived
        recording exports and the Figure-4 decode cache; pass False on a
        machine without them to leave the committed ``replay.json`` untouched.

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
            out_dir / "manifest.json", manifest_payload(figure03_summary, figure04_summary)
        ),
        write_site_json(
            out_dir / "playground.json",
            playground_payload(config, sparse_centers, figure03_summary),
        ),
        write_site_json(fixture_path, metric_parity_fixture(config, sparse_centers)),
        write_site_json(out_dir / "filter.json", filter_explainer_payload(config)),
    ]
    for condition_id, payload in scenario_payloads(simulation, figure03_summary).items():
        written.append(write_site_json(out_dir / f"scenario_{condition_id}.json", payload))
    if include_recording:
        render_data = prepare_figure04_render_data(
            Figure4Config(),
            Figure4Paths(data_path=DATA_PATH, animal_date_epoch=ANIMAL_DATE_EPOCH),
        )
        written.append(
            write_site_json(out_dir / "replay.json", replay_payload(render_data, figure04_summary))
        )
    return written
