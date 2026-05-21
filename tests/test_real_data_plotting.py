"""Tests for plotting helpers in ``real_data_plotting``.

Currently focused on ``plot_per_spike_metric_hexbin_row`` — the Figure 4(e)
panel introduced on this branch — which had no direct coverage before.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from statespacecheck_paper.real_data_plotting import (  # noqa: E402
    plot_per_spike_metric_hexbin_row,
)


@pytest.fixture
def paired_diagnostics() -> tuple[dict[str, Any], dict[str, Any]]:
    """Two per-spike diagnostic dicts with 50 matched spike events each.

    Synthesized in-test so the helper is exercised without requiring real-data
    fixtures. The two dicts share the same n_spikes so the same-length contract
    in ``plot_per_spike_metric_hexbin_row`` is satisfied; correlated noise
    gives a hexbin with mass on the identity line plus a spread.
    """
    rng = np.random.default_rng(0)
    n_spikes = 50

    hpd_a = rng.uniform(0.0, 1.0, n_spikes)
    hpd_b = np.clip(hpd_a + rng.normal(0.0, 0.05, n_spikes), 0.0, 1.0)

    kl_a = rng.gamma(2.0, 0.5, n_spikes)
    kl_b = kl_a + rng.normal(0.0, 0.1, n_spikes)

    sp_a = rng.uniform(0.01, 1.0, n_spikes)
    sp_b = np.clip(sp_a + rng.normal(0.0, 0.02, n_spikes), 1e-3, 1.0)

    diag_a = {
        "event_hpd_overlap": hpd_a,
        "event_kl_divergence": kl_a,
        "event_spike_prob": sp_a,
    }
    diag_b = {
        "event_hpd_overlap": hpd_b,
        "event_kl_divergence": kl_b,
        "event_spike_prob": sp_b,
    }
    return diag_a, diag_b


class TestPlotPerSpikeMetricHexbinRow:
    def test_renders_three_panels_with_hexbin_and_identity_line(
        self,
        paired_diagnostics: tuple[dict[str, Any], dict[str, Any]],
    ) -> None:
        """Every panel must carry a hexbin (PolyCollection) and an identity
        Line2D — the two load-bearing visual elements of the comparison.
        """
        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)

        for ax in axes:
            polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
            assert polys, f"axis {ax.get_title()!r} has no PolyCollection (hexbin)"
            lines = [child for child in ax.get_children() if isinstance(child, Line2D)]
            # At least one Line2D (the identity reference) should be present.
            assert lines, f"axis {ax.get_title()!r} has no Line2D (identity reference)"
            # Aspect ratio locked to equal so the identity line is at 45 degrees.
            # Matplotlib normalises ``set_aspect("equal")`` to the float 1.0.
            assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_drops_nans_from_count_annotation(
        self,
        paired_diagnostics: tuple[dict[str, Any], dict[str, Any]],
    ) -> None:
        """``n=...`` annotation reports the count of finite-in-both events,
        not the raw input length. Introduces 5 NaNs into each metric's
        arrays so the count should drop by exactly 5 in each panel.
        """
        diag_a, diag_b = paired_diagnostics
        n_total = diag_a["event_hpd_overlap"].size
        n_nans = 5

        # Same-position NaNs across the two arrays so the mask is unambiguous.
        nan_positions = [0, 7, 13, 24, 41]
        diag_a_nan = {k: v.copy() for k, v in diag_a.items()}
        diag_b_nan = {k: v.copy() for k, v in diag_b.items()}
        for key in ("event_hpd_overlap", "event_kl_divergence", "event_spike_prob"):
            diag_a_nan[key][nan_positions] = np.nan
            diag_b_nan[key][nan_positions] = np.nan

        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(diag_a_nan, diag_b_nan, axes)

        expected = f"n={n_total - n_nans:,}"
        for ax in axes:
            n_texts = [t.get_text() for t in ax.texts if t.get_text().startswith("n=")]
            assert n_texts, f"axis {ax.get_title()!r} has no n= annotation"
            assert any(t == expected for t in n_texts), (
                f"axis {ax.get_title()!r} reports {n_texts} but expected {expected}"
            )

        plt.close(fig)

    def test_validates_same_length(
        self,
        paired_diagnostics: tuple[dict[str, Any], dict[str, Any]],
    ) -> None:
        """Mismatched-shape inputs must raise — the helper would otherwise
        produce a plausible-looking hexbin on misaligned arrays.
        """
        diag_a, diag_b = paired_diagnostics
        diag_b_short = {k: v[:25] for k, v in diag_b.items()}
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(ValueError, match="same set of spike events"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b_short, axes)
        plt.close(fig)

    def test_rejects_wrong_axes_count(
        self,
        paired_diagnostics: tuple[dict[str, Any], dict[str, Any]],
    ) -> None:
        """The helper expects exactly three axes (one per metric)."""
        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="axes must have length 3"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)
        plt.close(fig)
