"""Tests for plotting utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from statespacecheck_paper.figure01_generation import create_distribution_comparison_panel
from statespacecheck_paper.plotting import (
    extract_contiguous_regions,
    negative_log_pvalue,
)


class TestNegativeLogPvalue:
    def test_matches_exact_transform_elementwise(self) -> None:
        x = np.array([1.0, 0.5, 0.1, 1e-10, 1e-12])
        expected = -np.log(x)
        np.testing.assert_array_equal(negative_log_pvalue(x), expected)

    @pytest.mark.parametrize("bad", [0.0, -0.1, np.inf, 1.1])
    def test_rejects_unrepresentable_or_invalid_values(self, bad: float) -> None:
        with pytest.raises(ValueError):
            negative_log_pvalue(bad)

    def test_scalar_threshold_transform(self) -> None:
        """The scalar overload used on flag thresholds is exact."""
        assert negative_log_pvalue(0.2) == -np.log(0.2)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


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
