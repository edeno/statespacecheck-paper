"""Tests for Figure-3 rendering (``compose_figure03`` + panels)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds
from statespacecheck_paper.figure03_plotting import (
    FIGURE3_PANEL_LABEL_GID,
    FIGURE3_PHASE_LABEL_GID,
    FIGURE3_SUMMARY_CELL_LABEL_GID,
    FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID,
    FIGURE3_SUMMARY_TITLE_GID,
    FIGURE3_THRESHOLD_LABEL_GID,
    FIGURE3_THRESHOLD_LINE_GID,
    FIGURE3_TRUE_POSITION_LABEL_GID,
    FIGURE3_WORSE_FIT_LABEL_GID,
    compose_figure03,
)
from statespacecheck_paper.figure03_protocol import Figure3Config


def _per_cell_metrics(rng: np.random.Generator, n_time: int, n_cells: int) -> dict[str, np.ndarray]:
    """Per-cell metric matrices ``(n_time, n_cells)`` with values in
    plausible ranges for the diagnostic plotting code."""
    return {
        "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
        "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
        "predictive_pvalue": rng.uniform(0, 1, (n_time, n_cells)),
    }


@pytest.fixture
def small_metrics(rng: np.random.Generator) -> dict[str, Any]:
    """``plot_original`` / ``plot_transformed``-shaped inputs for a small grid."""
    n_time, n_bins, n_cells = 100, 50, 5
    return {
        "xs": np.linspace(0, 1, n_bins),
        "x_true": rng.uniform(0, n_bins - 1, n_time),
        "metrics": {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            **_per_cell_metrics(rng, n_time, n_cells),
        },
    }


@pytest.fixture
def thresholds_default() -> DiagnosticThresholds:
    return DiagnosticThresholds(hpd_overlap=0.8, kl_divergence=2.0, predictive_pvalue=0.05)


def _combined_metrics(
    rng: np.random.Generator, n_time: int, n_bins: int, n_cells: int
) -> dict[str, Any]:
    """Build the full ``DecodingDiagnostics`` accepted by ``compose_figure03``."""
    spikes = rng.poisson(0.5, (n_time, n_cells))
    spike_lik = np.full((n_time, n_bins), np.nan)
    has_spk = spikes.sum(axis=1) > 0
    spike_lik[has_spk] = rng.dirichlet(np.ones(n_bins), size=int(has_spk.sum()))

    spike_time_ind, spike_cell_ind = np.nonzero(spikes[1:])
    spike_time_ind = (spike_time_ind + 1).astype(np.intp)
    spike_cell_ind = spike_cell_ind.astype(np.intp)
    n_spikes = max(len(spike_time_ind), 1)
    per_spike_lik = rng.dirichlet(np.ones(n_bins), size=n_spikes)[: len(spike_time_ind)]

    per_cell = _per_cell_metrics(rng, n_time, n_cells)
    diagnostics = DecodingDiagnostics(
        posterior=rng.dirichlet(np.ones(n_bins), size=n_time),
        predictive=rng.dirichlet(np.ones(n_bins), size=n_time),
        likelihood=rng.dirichlet(np.ones(n_bins), size=n_time),
        spike_likelihood=spike_lik,
        hpd_overlap=per_cell["hpd_overlap"],
        kl_divergence=per_cell["kl_divergence"],
        predictive_pvalue=per_cell["predictive_pvalue"],
        event_time_ind=spike_time_ind,
        event_cell_ind=spike_cell_ind,
        event_hpd_overlap=rng.uniform(0, 1, len(spike_time_ind)),
        event_kl_divergence=rng.uniform(0, 5, len(spike_time_ind)),
        event_predictive_pvalue=rng.uniform(0, 1, len(spike_time_ind)),
        per_spike_likelihood=per_spike_lik,
    )
    return {"spikes": spikes, "metrics": diagnostics}


def _bidirectional_remap(n_cells: int) -> tuple[tuple[int, int], ...]:
    """Pairwise swaps across the cell index range."""
    half = n_cells // 2
    pairs: list[tuple[int, int]] = []
    for i in range(half):
        pairs.append((i, half + i))
        pairs.append((half + i, i))
    return tuple(pairs)


def _params_for_short_run(
    n_time: int, n_cells: int, prediction_step_std: float = 0.5
) -> Figure3Config:
    """Figure3Config with phase boundaries scaled to fit ``n_time``.

    Distributes the 8 phase boundaries (3 misfits and a sparse-population
    control, with recovery between) so every highlighted window has at least
    a few timesteps. ``n_time``
    needs to be large enough that ``phase_boundaries[REMAP_START] - 1000``
    is positive (some downstream helpers index a 1000-timestep baseline
    preamble).
    """
    return Figure3Config(
        phase_boundaries=(
            int(n_time * 0.5),
            int(n_time * 0.6),
            int(n_time * 0.66),
            int(n_time * 0.74),
            int(n_time * 0.8),
            int(n_time * 0.85),
            int(n_time * 0.9),
            int(n_time * 0.99),
        ),
        prediction_step_std=prediction_step_std,
        place_field_remapping=_bidirectional_remap(n_cells),
    )


@pytest.mark.parametrize(
    ("n_time", "n_bins", "n_cells"),
    [(6000, 50, 10), (3500, 30, 5)],
    ids=["large", "small"],
)
def test_compose_figure03_runs(
    n_time: int, n_bins: int, n_cells: int, thresholds_default: DiagnosticThresholds
) -> None:
    rng = np.random.default_rng(42)
    x_true = rng.uniform(0, n_bins - 1, n_time)
    bundle = _combined_metrics(rng, n_time, n_bins, n_cells)
    params = _params_for_short_run(n_time, n_cells)

    fig = compose_figure03(
        x_true,
        bundle["spikes"],
        bundle["metrics"],
        thresholds_default,
        params,
        np.linspace(0, 1, n_cells),
    )
    try:
        assert isinstance(fig, plt.Figure)
        assert fig.axes[5].get_xlabel() == "Time (ms)"
    finally:
        plt.close(fig)


def test_compose_figure03_renders_precomputed_summary(
    thresholds_default: DiagnosticThresholds,
) -> None:
    """When an across-realization median array is supplied, the panel-(b)
    heatmap renders those values (with the median title) instead of
    recomputing from the single displayed realization."""
    rng = np.random.default_rng(0)
    n_time, n_bins, n_cells = 3500, 30, 5
    x_true = rng.uniform(0, n_bins - 1, n_time)
    bundle = _combined_metrics(rng, n_time, n_bins, n_cells)
    params = _params_for_short_run(n_time, n_cells)

    # Columns: well-specified, remap, history, replay, drift, sparse population.
    median = np.array(
        [
            [1.0, 60.0, 1.0, 4.0, 10.0, 0.0],
            [1.0, 60.0, 1.0, 4.0, 8.0, 17.0],
            [3.0, 64.0, 2.0, 2.0, 14.0, 0.0],
        ]
    )

    fig = compose_figure03(
        x_true,
        bundle["spikes"],
        bundle["metrics"],
        thresholds_default,
        params,
        np.linspace(0, 1, n_cells),
        median_flag_percentages=median,
    )
    try:
        # The summary axis is the last one added; its title flags the median
        # mode and at least one cell shows the supplied median.
        summary_ax = fig.axes[-1]
        assert "median across realizations" in summary_ax.get_title()
        cell_texts = {t.get_text() for t in summary_ax.texts}
        assert "60%" in cell_texts  # supplied remap median
    finally:
        plt.close(fig)


def test_compose_figure03_tags_figure3_annotations(
    thresholds_default: DiagnosticThresholds,
) -> None:
    """Figure 3 annotations should be targetable by semantic artist ids."""
    rng = np.random.default_rng(1)
    n_time, n_bins, n_cells = 3500, 30, 5
    x_true = rng.uniform(0, n_bins - 1, n_time)
    bundle = _combined_metrics(rng, n_time, n_bins, n_cells)
    params = _params_for_short_run(n_time, n_cells)

    fig = compose_figure03(
        x_true,
        bundle["spikes"],
        bundle["metrics"],
        thresholds_default,
        params,
        np.linspace(0, 1, n_cells),
    )
    try:
        texts = [text for ax in fig.axes for text in ax.texts]
        lines = [line for ax in fig.axes for line in ax.lines]

        assert sum(text.get_gid() == FIGURE3_PANEL_LABEL_GID for text in texts) == 2
        phase_labels = [text for text in texts if text.get_gid() == FIGURE3_PHASE_LABEL_GID]
        assert len(phase_labels) == 5
        assert {text.get_position()[1] for text in phase_labels} == {
            phase_labels[0].get_position()[1]
        }
        assert sum(text.get_gid() == FIGURE3_THRESHOLD_LABEL_GID for text in texts) == 3
        assert sum(text.get_gid() == FIGURE3_WORSE_FIT_LABEL_GID for text in texts) == 3
        assert any(text.get_gid() == FIGURE3_TRUE_POSITION_LABEL_GID for text in texts)
        assert any(text.get_gid() == FIGURE3_SUMMARY_KNOWN_COMPONENT_LABEL_GID for text in texts)
        assert sum(text.get_gid() == FIGURE3_SUMMARY_CELL_LABEL_GID for text in texts) == 18
        assert any(ax.title.get_gid() == FIGURE3_SUMMARY_TITLE_GID for ax in fig.axes)
        assert sum(line.get_gid() == FIGURE3_THRESHOLD_LINE_GID for line in lines) == 3
    finally:
        plt.close(fig)


def test_compose_figure03_uses_event_diagnostics_for_scatter() -> None:
    """When duplicate spike events fall in one bin, scatter plots must
    show each event independently — not collapse to the matrix value."""
    rng = np.random.default_rng(42)
    n_time, n_bins, n_cells = 3500, 30, 2
    x_true = rng.uniform(0, n_bins - 1, n_time)
    spikes = np.zeros((n_time, n_cells), dtype=int)
    spikes[10, 0] = 2

    spike_lik = np.full((n_time, n_bins), np.nan)
    hpd = np.full((n_time, n_cells), np.nan)
    kl = np.full((n_time, n_cells), np.nan)
    sp = np.full((n_time, n_cells), np.nan)
    per_spike_lik = rng.dirichlet(np.ones(n_bins), size=2)
    spike_lik[10] = per_spike_lik[0]
    hpd[10, 0] = 0.5
    kl[10, 0] = 2.0
    sp[10, 0] = 0.05

    metrics = DecodingDiagnostics(
        posterior=rng.dirichlet(np.ones(n_bins), size=n_time),
        predictive=rng.dirichlet(np.ones(n_bins), size=n_time),
        likelihood=rng.dirichlet(np.ones(n_bins), size=n_time),
        spike_likelihood=spike_lik,
        hpd_overlap=hpd,
        kl_divergence=kl,
        predictive_pvalue=sp,
        event_time_ind=np.array([10, 10], dtype=np.intp),
        event_cell_ind=np.array([0, 0], dtype=np.intp),
        event_hpd_overlap=np.array([0.25, 0.75]),
        event_kl_divergence=np.array([1.0, 3.0]),
        event_predictive_pvalue=np.array([0.1, 0.01]),
        per_spike_likelihood=per_spike_lik,
    )

    thresholds = DiagnosticThresholds(hpd_overlap=0.8, kl_divergence=2.0, predictive_pvalue=0.05)
    params = _params_for_short_run(n_time, n_cells)

    fig = compose_figure03(
        x_true,
        spikes,
        metrics,
        thresholds,
        params,
        place_field_centers=np.linspace(0, 1, n_cells),
    )
    try:
        # Diagnostic rows are ordered HPD (axis 3), -log(p) (axis 4),
        # KL (axis 5); axes 0-2 are the predictive/likelihood/raster stack.
        hpd_offsets = fig.axes[3].collections[0].get_offsets()
        np.testing.assert_array_equal(hpd_offsets[:, 0], [10, 10])
        np.testing.assert_allclose(hpd_offsets[:, 1], [0.25, 0.75])

        predictive_pvalue_offsets = fig.axes[4].collections[0].get_offsets()
        np.testing.assert_array_equal(predictive_pvalue_offsets[:, 0], [10, 10])
        # Plotted as -log(predictive_pvalue) (natural log); 0.1 -> -ln(0.1), 0.01 -> -ln(0.01).
        np.testing.assert_allclose(predictive_pvalue_offsets[:, 1], [-np.log(0.1), -np.log(0.01)])
    finally:
        plt.close(fig)
