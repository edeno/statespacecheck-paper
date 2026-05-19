"""Tests for plotting utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from statespacecheck_paper.analysis import DecodeParams, Thresholds, Transformed
from statespacecheck_paper.plotting import (
    compute_hpd_region,
    create_distribution_comparison_panel,
    extract_contiguous_regions,
    plot_combined_diagnostics,
    plot_misfit_examples,
    plot_original,
    plot_transformed,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_pdf() -> tuple[np.ndarray, np.ndarray]:
    """Standard normal PDF on a fine grid for HPD-region tests."""
    x = np.linspace(-5, 5, 1000)
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    return x, pdf


def _per_cell_metrics(rng: np.random.Generator, n_time: int, n_cells: int) -> dict[str, np.ndarray]:
    """Per-cell metric matrices ``(n_time, n_cells)`` with values in
    plausible ranges for the diagnostic plotting code."""
    return {
        "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
        "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
        "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
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
def thresholds_default() -> Thresholds:
    return Thresholds(hpd_overlap=0.8, kl_divergence=2.0, spike_prob=0.05)


def _combined_metrics(
    rng: np.random.Generator, n_time: int, n_bins: int, n_cells: int
) -> dict[str, Any]:
    """Build the full metrics dict accepted by ``plot_combined_diagnostics``."""
    spikes = rng.poisson(0.5, (n_time, n_cells))
    spike_lik = np.full((n_time, n_bins), np.nan)
    has_spk = spikes.sum(axis=1) > 0
    spike_lik[has_spk] = rng.dirichlet(np.ones(n_bins), size=int(has_spk.sum()))

    spike_time_ind, spike_cell_ind = np.nonzero(spikes[1:])
    spike_time_ind = spike_time_ind + 1
    n_spikes = max(len(spike_time_ind), 1)
    per_spike_lik = rng.dirichlet(np.ones(n_bins), size=n_spikes)[: len(spike_time_ind)]

    return {
        "spikes": spikes,
        "metrics": {
            "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
            "likelihood": rng.dirichlet(np.ones(n_bins), size=n_time),
            "spike_likelihood": spike_lik,
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            **_per_cell_metrics(rng, n_time, n_cells),
            "per_spike_likelihood": per_spike_lik,
            "spike_time_ind": spike_time_ind,
            "spike_cell_ind": spike_cell_ind,
            "event_time_ind": spike_time_ind,
            "event_cell_ind": spike_cell_ind,
            "event_hpd_overlap": rng.uniform(0, 1, len(spike_time_ind)),
            "event_kl_divergence": rng.uniform(0, 5, len(spike_time_ind)),
            "event_spike_prob": rng.uniform(0, 1, len(spike_time_ind)),
        },
    }


def _bidirectional_remap(n_cells: int) -> tuple[tuple[int, int], ...]:
    """Pairwise swaps across the cell index range."""
    half = n_cells // 2
    pairs: list[tuple[int, int]] = []
    for i in range(half):
        pairs.append((i, half + i))
        pairs.append((half + i, i))
    return tuple(pairs)


def _params_for_short_run(n_time: int, n_cells: int, sigx_pred: float = 0.5) -> DecodeParams:
    """DecodeParams with phase boundaries scaled to fit ``n_time``.

    Distributes the 10 phase boundaries (5 misfits with recovery between
    each) so every misfit window has at least a few timesteps. ``n_time``
    needs to be large enough that ``T_remap_start - 1000`` is positive
    (some downstream helpers index a 1000-timestep baseline preamble).
    """
    return DecodeParams(
        T_remap_start=int(n_time * 0.5),
        T_remap_end=int(n_time * 0.6),
        T_recovery1_end=int(n_time * 0.66),
        T_hist_dep_end=int(n_time * 0.74),
        T_recovery2_end=int(n_time * 0.8),
        T_drift_end=int(n_time * 0.85),
        T_recovery3_end=int(n_time * 0.9),
        T_wide_dynamics_end=int(n_time * 0.93),
        T_recovery4_end=int(n_time * 0.96),
        T_wiggly_end=int(n_time * 0.99),
        sigx_pred=sigx_pred,
        remap_from_to=_bidirectional_remap(n_cells),
    )


# ---------------------------------------------------------------------------
# compute_hpd_region
# ---------------------------------------------------------------------------


class TestComputeHpdRegion:
    def test_output_shape_and_dtype(self, gaussian_pdf: tuple) -> None:
        x, pdf = gaussian_pdf
        mask = compute_hpd_region(x, pdf, coverage=0.95)
        assert mask.shape == x.shape
        assert mask.dtype == bool

    def test_coverage_close_to_target(self, gaussian_pdf: tuple) -> None:
        x, pdf = gaussian_pdf
        mask = compute_hpd_region(x, pdf, coverage=0.95)
        dx = x[1] - x[0]
        pdf_normalized = pdf / (pdf.sum() * dx)
        actual_coverage = pdf_normalized[mask].sum() * dx
        # Discrete HPD slightly overshoots; tolerate up to 100%.
        assert 0.90 <= actual_coverage <= 1.0

    def test_hpd_is_contiguous_for_unimodal_distribution(self, gaussian_pdf: tuple) -> None:
        x, pdf = gaussian_pdf
        mask = compute_hpd_region(x, pdf, coverage=0.95)
        true_indices = np.where(mask)[0]
        expected_run = np.arange(true_indices[0], true_indices[-1] + 1)
        # Allow tiny gaps from discretization.
        assert len(true_indices) / len(expected_run) > 0.90

    @pytest.mark.parametrize(
        ("low_coverage", "high_coverage"),
        [(0.50, 0.95), (0.50, 0.99), (0.80, 0.95)],
    )
    def test_higher_coverage_includes_more_points(
        self, gaussian_pdf: tuple, low_coverage: float, high_coverage: float
    ) -> None:
        x, pdf = gaussian_pdf
        low_mask = compute_hpd_region(x, pdf, coverage=low_coverage)
        high_mask = compute_hpd_region(x, pdf, coverage=high_coverage)
        assert high_mask.sum() > low_mask.sum()

    def test_uniform_distribution_includes_almost_all_points(self) -> None:
        x = np.linspace(0, 10, 100)
        pdf = np.ones_like(x)
        mask = compute_hpd_region(x, pdf, coverage=0.95)
        assert mask.sum() / len(mask) > 0.90

    def test_coverage_near_one_does_not_overshoot_index(self) -> None:
        """Edge case: coverage very close to 1.0 must not index past the end."""
        x = np.linspace(-5, 5, 50)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        mask = compute_hpd_region(x, pdf, coverage=0.999)
        assert mask.shape == x.shape


# ---------------------------------------------------------------------------
# plot_original / plot_transformed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"phase_boundaries": (10, 20, 30, 40, 50, 60, 70, 80)},
        {"title": "Custom Test Title"},
    ],
    ids=["base", "phase_boundaries", "title"],
)
def test_plot_original_returns_figure(
    small_metrics: dict[str, Any],
    thresholds_default: Thresholds,
    extra_kwargs: dict[str, Any],
) -> None:
    fig = plot_original(
        small_metrics["xs"],
        small_metrics["x_true"],
        small_metrics["metrics"],
        thresholds_default,
        **extra_kwargs,
    )
    try:
        assert isinstance(fig, plt.Figure)
        # 4 panels: posterior + 3 metrics (HPDO, KL, spike_prob).
        assert len(fig.axes) >= 4
    finally:
        plt.close(fig)


def _make_transformed(rng: np.random.Generator, n_time: int, n_cells: int) -> Transformed:
    return Transformed(
        hpd_overlap=rng.uniform(0, 5, (n_time, n_cells)),
        kl_divergence=rng.uniform(0, 3, (n_time, n_cells)),
        spike_prob=rng.uniform(0, 10, (n_time, n_cells)),
        hpd_overlap_threshold=3.0,
        kl_divergence_threshold=2.0,
        spike_prob_threshold=5.0,
    )


@pytest.mark.parametrize(
    "extra_kwargs",
    [
        {},
        {"remap_window": (20, 40)},
        {"phase_boundaries": (30, 70)},
    ],
    ids=["base", "remap_window", "phase_boundaries"],
)
def test_plot_transformed_returns_figure(extra_kwargs: dict[str, Any]) -> None:
    rng = np.random.default_rng(42)
    n_time, n_bins, n_cells = 100, 50, 5
    xs = np.linspace(0, 1, n_bins)
    x_true = rng.uniform(0, n_bins - 1, n_time)
    posterior = rng.dirichlet(np.ones(n_bins), size=n_time)
    transformed = _make_transformed(rng, n_time, n_cells)

    fig = plot_transformed(xs, x_true, posterior, transformed, **extra_kwargs)
    try:
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_misfit_examples
# ---------------------------------------------------------------------------


def test_plot_misfit_examples_runs(rng: np.random.Generator) -> None:
    """Smoke test on a dataset large enough to satisfy the baseline window
    (the function indexes ``slice(1000, T_remap_start - 1000)``)."""
    n_time, n_bins, n_cells = 6000, 50, 10
    xs = np.linspace(0, 1, n_bins)
    x_true = rng.uniform(0, n_bins - 1, n_time)
    spikes = rng.poisson(0.5, (n_time, n_cells))
    metrics = {
        "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
        **_per_cell_metrics(rng, n_time, n_cells),
    }
    params = _params_for_short_run(n_time, n_cells)
    fig = plot_misfit_examples(
        xs,
        x_true,
        spikes,
        metrics,
        params,
        np.linspace(0, 1, n_cells),
        0.1,
        5.0,
    )
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_combined_diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_time", "n_bins", "n_cells"),
    [(6000, 50, 10), (3500, 30, 5)],
    ids=["large", "small"],
)
def test_plot_combined_diagnostics_runs(
    n_time: int, n_bins: int, n_cells: int, thresholds_default: Thresholds
) -> None:
    rng = np.random.default_rng(42)
    xs = np.linspace(0, 1, n_bins)
    x_true = rng.uniform(0, n_bins - 1, n_time)
    bundle = _combined_metrics(rng, n_time, n_bins, n_cells)
    params = _params_for_short_run(n_time, n_cells)

    fig = plot_combined_diagnostics(
        xs,
        x_true,
        bundle["spikes"],
        bundle["metrics"],
        thresholds_default,
        params,
        np.linspace(0, 1, n_cells),
    )
    try:
        assert isinstance(fig, plt.Figure)
    finally:
        plt.close(fig)


def test_plot_combined_diagnostics_uses_event_diagnostics_for_scatter() -> None:
    """When duplicate spike events fall in one bin, scatter plots must
    show each event independently — not collapse to the matrix value."""
    rng = np.random.default_rng(42)
    n_time, n_bins, n_cells = 3500, 30, 2
    xs = np.linspace(0, 1, n_bins)
    x_true = rng.uniform(0, n_bins - 1, n_time)
    spikes = np.zeros((n_time, n_cells), dtype=int)
    spikes[10, 0] = 2

    metrics = {
        "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
        "likelihood": rng.dirichlet(np.ones(n_bins), size=n_time),
        "spike_likelihood": np.full((n_time, n_bins), np.nan),
        "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
        "hpd_overlap": np.full((n_time, n_cells), np.nan),
        "kl_divergence": np.full((n_time, n_cells), np.nan),
        "spike_prob": np.full((n_time, n_cells), np.nan),
        "per_spike_likelihood": rng.dirichlet(np.ones(n_bins), size=2),
        "spike_time_ind": np.array([10, 10]),
        "spike_cell_ind": np.array([0, 0]),
        "event_time_ind": np.array([10, 10]),
        "event_cell_ind": np.array([0, 0]),
        "event_hpd_overlap": np.array([0.25, 0.75]),
        "event_kl_divergence": np.array([1.0, 3.0]),
        "event_spike_prob": np.array([0.1, 0.01]),
    }
    metrics["spike_likelihood"][10] = metrics["per_spike_likelihood"][0]
    metrics["hpd_overlap"][10, 0] = 0.5
    metrics["kl_divergence"][10, 0] = 2.0
    metrics["spike_prob"][10, 0] = 0.05

    thresholds = Thresholds(hpd_overlap=0.8, kl_divergence=2.0, spike_prob=0.05)
    params = _params_for_short_run(n_time, n_cells)

    fig = plot_combined_diagnostics(
        xs,
        x_true,
        spikes,
        metrics,
        thresholds,
        params,
        placefield_centers=np.linspace(0, 1, n_cells),
    )
    try:
        hpd_offsets = fig.axes[3].collections[0].get_offsets()
        np.testing.assert_array_equal(hpd_offsets[:, 0], [10, 10])
        np.testing.assert_allclose(hpd_offsets[:, 1], [0.25, 0.75])

        spike_prob_offsets = fig.axes[5].collections[0].get_offsets()
        np.testing.assert_array_equal(spike_prob_offsets[:, 0], [10, 10])
        # Plotted as -log10(spike_prob); 0.1 -> 1.0, 0.01 -> 2.0.
        np.testing.assert_allclose(spike_prob_offsets[:, 1], [1.0, 2.0])
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# extract_contiguous_regions
# ---------------------------------------------------------------------------


class TestExtractContiguousRegions:
    @pytest.fixture
    def x(self) -> np.ndarray:
        return np.linspace(0, 10, 100)

    def test_single_region(self, x: np.ndarray) -> None:
        regions = extract_contiguous_regions((x > 2) & (x < 8), x)
        assert len(regions) == 1
        start, end = regions[0]
        assert 2.0 < start < 2.5
        assert 7.5 < end < 8.0

    def test_multiple_regions(self, x: np.ndarray) -> None:
        mask = ((x > 1) & (x < 3)) | ((x > 6) & (x < 9))
        regions = extract_contiguous_regions(mask, x)
        assert len(regions) == 2

    def test_empty_mask_returns_empty_list(self, x: np.ndarray) -> None:
        assert extract_contiguous_regions(np.zeros_like(x, dtype=bool), x) == []

    def test_all_true_is_one_region_spanning_x(self, x: np.ndarray) -> None:
        regions = extract_contiguous_regions(np.ones_like(x, dtype=bool), x)
        assert len(regions) == 1
        assert regions[0] == (x[0], x[-1])

    def test_regions_at_both_edges(self, x: np.ndarray) -> None:
        regions = extract_contiguous_regions((x < 2) | (x > 8), x)
        assert len(regions) == 2
        assert regions[0][0] == x[0]
        assert regions[1][1] == x[-1]

    def test_single_point_region(self) -> None:
        """Edge case: a single True point still yields a (start, end) tuple."""
        x = np.linspace(0, 1, 5)
        mask = np.array([False, False, True, False, False])
        regions = extract_contiguous_regions(mask, x)
        assert len(regions) == 1
        assert regions[0] == (x[2], x[2])


# ---------------------------------------------------------------------------
# create_distribution_comparison_panel
# ---------------------------------------------------------------------------


class TestCreateDistributionComparisonPanel:
    """Visual contract: panel produces two distribution lines + HPD patches."""

    _X = np.linspace(-20, 20, 1000)
    _BASE_KWARGS: dict[str, Any] = {
        "predictive_params": (0, 1.5),
        "likelihood_params": (5, 1.5),
        "color_predictive": "blue",
        "color_likelihood": "orange",
    }

    def test_default_panel_has_lines_and_hpd_patches(
        self, fresh_axes: tuple[plt.Figure, plt.Axes]
    ) -> None:
        """With defaults, the panel produces both the two distribution
        lines AND the HPD bar patches — verify in one call to keep the
        test count proportional to the surface area."""
        _, ax = fresh_axes
        create_distribution_comparison_panel(ax, self._X, **self._BASE_KWARGS)
        # Predictive + likelihood lines (HPD bars are patches, not lines).
        assert len(ax.lines) >= 2
        assert len(ax.patches) >= 2

    def test_title_kwarg_sets_axis_title(self, fresh_axes: tuple[plt.Figure, plt.Axes]) -> None:
        _, ax = fresh_axes
        create_distribution_comparison_panel(ax, self._X, title="My Title", **self._BASE_KWARGS)
        assert ax.get_title() == "My Title"

    def test_show_labels_adds_text_annotations(
        self, fresh_axes: tuple[plt.Figure, plt.Axes]
    ) -> None:
        _, ax = fresh_axes
        create_distribution_comparison_panel(ax, self._X, show_labels=True, **self._BASE_KWARGS)
        # At minimum a "Predictive" and "Likelihood" annotation.
        assert len(ax.texts) >= 2

    @pytest.mark.parametrize(
        ("predictive_params", "likelihood_params"),
        [((0, 5.0), (2, 3.0)), ((-10, 1.0), (10, 1.0))],
        ids=["overlapping", "non_overlapping"],
    )
    def test_runs_for_overlap_and_non_overlap_scenarios(
        self,
        fresh_axes: tuple[plt.Figure, plt.Axes],
        predictive_params: tuple[float, float],
        likelihood_params: tuple[float, float],
    ) -> None:
        """Both configurations must complete without error and still
        produce HPD patches (regression: very-different-mean distributions
        used to crash HPD bar placement)."""
        _, ax = fresh_axes
        create_distribution_comparison_panel(
            ax,
            self._X,
            predictive_params=predictive_params,
            likelihood_params=likelihood_params,
            color_predictive="blue",
            color_likelihood="orange",
        )
        assert len(ax.patches) >= 2
