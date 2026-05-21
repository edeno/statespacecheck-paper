"""Tests for plotting helpers in ``real_data_plotting``.

Currently focused on ``plot_per_spike_metric_hexbin_row`` — the Figure 4(e)
panel introduced on this branch — which had no direct coverage before.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from statespacecheck_paper.analysis import PerCellDiagnostics  # noqa: E402
from statespacecheck_paper.real_data_plotting import (  # noqa: E402
    plot_per_spike_metric_hexbin_row,
)


def _per_spike_diagnostics(hpd: np.ndarray, kl: np.ndarray, sp: np.ndarray) -> PerCellDiagnostics:
    """Build a ``PerCellDiagnostics`` from per-spike metric arrays only.

    The hexbin helper consumes the three ``event_*`` arrays; the rest
    of the dataclass is required by the constructor but unused here.
    """
    n_spikes = hpd.shape[0]
    return PerCellDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=hpd,
        event_kl_divergence=kl,
        event_spike_prob=sp,
        hpd_overlap=None,
        kl_divergence=None,
        spike_prob=None,
        per_spike_likelihood=None,
    )


@pytest.fixture
def paired_diagnostics() -> tuple[PerCellDiagnostics, PerCellDiagnostics]:
    """Two ``PerCellDiagnostics`` with 50 matched spike events each.

    Synthesized in-test so the helper is exercised without requiring real-data
    fixtures. Same n_spikes so the same-length contract in
    ``plot_per_spike_metric_hexbin_row`` is satisfied; correlated noise
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

    return _per_spike_diagnostics(hpd_a, kl_a, sp_a), _per_spike_diagnostics(hpd_b, kl_b, sp_b)


class TestPlotPerSpikeMetricHexbinRow:
    def test_renders_three_panels_with_hexbin_and_identity_line(
        self,
        paired_diagnostics: tuple[PerCellDiagnostics, PerCellDiagnostics],
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
        paired_diagnostics: tuple[PerCellDiagnostics, PerCellDiagnostics],
    ) -> None:
        """``n=...`` annotation reports the count of finite-in-both events,
        not the raw input length. Introduces 5 NaNs into each metric's
        arrays so the count should drop by exactly 5 in each panel.
        """
        diag_a, diag_b = paired_diagnostics
        n_total = diag_a.event_hpd_overlap.size
        n_nans = 5

        # Same-position NaNs across the two arrays so the mask is unambiguous.
        nan_positions = [0, 7, 13, 24, 41]

        def with_nans(d: PerCellDiagnostics) -> PerCellDiagnostics:
            hpd = d.event_hpd_overlap.copy()
            kl = d.event_kl_divergence.copy()
            sp = d.event_spike_prob.copy()
            for arr in (hpd, kl, sp):
                arr[nan_positions] = np.nan
            return _per_spike_diagnostics(hpd, kl, sp)

        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(with_nans(diag_a), with_nans(diag_b), axes)

        import re

        expected_count = n_total - n_nans
        # Match any ``n=<digits>`` (with or without thousands separator
        # and surrounding whitespace) so a cosmetic format change
        # ("n = 45" or "n=45_000") doesn't break the test. The
        # behavioural contract is the integer in the annotation.
        pattern = re.compile(r"n\s*=\s*([\d,_]+)")
        for ax in axes:
            counts = [
                int(m.group(1).replace(",", "").replace("_", ""))
                for t in ax.texts
                for m in [pattern.match(t.get_text())]
                if m is not None
            ]
            assert counts, f"axis {ax.get_title()!r} has no n= annotation"
            assert expected_count in counts, (
                f"axis {ax.get_title()!r} reports {counts}; expected {expected_count}"
            )

        plt.close(fig)

    def test_validates_same_length(
        self,
        paired_diagnostics: tuple[PerCellDiagnostics, PerCellDiagnostics],
    ) -> None:
        """Mismatched-shape inputs must raise — the helper would otherwise
        produce a plausible-looking hexbin on misaligned arrays.
        """
        diag_a, diag_b = paired_diagnostics
        diag_b_short = _per_spike_diagnostics(
            diag_b.event_hpd_overlap[:25],
            diag_b.event_kl_divergence[:25],
            diag_b.event_spike_prob[:25],
        )
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(ValueError, match="same set of spike events"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b_short, axes)
        plt.close(fig)

    def test_rejects_wrong_axes_count(
        self,
        paired_diagnostics: tuple[PerCellDiagnostics, PerCellDiagnostics],
    ) -> None:
        """The helper expects exactly three axes (one per metric)."""
        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="axes must have length 3"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)
        plt.close(fig)
