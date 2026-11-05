"""Tests for the Figure-4 layout (event-time shift + composition contract)."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402


from statespacecheck_paper.figure04_layout import (  # noqa: E402
    Figure4Composition,
    _shift_diagnostic_event_times,
    compose_figure04,
)
from statespacecheck_paper.figure04_protocol import Figure4DetailWindow  # noqa: E402

from ._diagnostics import event_diagnostics  # noqa: E402
from ._figure04 import synthetic_render_data  # noqa: E402


class TestShiftDiagnosticEventTimes:
    def test_subtracts_offset(self) -> None:
        """Per-spike event times must be relative to the same time base as the
        figure axis — otherwise scatter points slide off the panels."""
        diagnostics = event_diagnostics(
            event_time=np.array([101.0, 101.5]), hpd=np.array([0.25, 0.75])
        )
        shifted = _shift_diagnostic_event_times(diagnostics, 100.0)
        np.testing.assert_allclose(shifted.event_time, [1.0, 1.5])
        # Original instance not mutated (frozen + write-protected).
        np.testing.assert_allclose(diagnostics.event_time, [101.0, 101.5])
        # Non-time arrays passed through by reference (zero-copy).
        assert shifted.event_hpd_overlap is diagnostics.event_hpd_overlap

    def test_passthrough_when_none(self) -> None:
        """Simulated-data path leaves ``event_time`` as ``None``; the shift is a
        no-op there, not a raise."""
        diagnostics = event_diagnostics(event_time=None, hpd=np.array([0.5]))
        assert _shift_diagnostic_event_times(diagnostics, 100.0) is diagnostics


# ---------------------------------------------------------------------------
# compose_figure04 end-to-end smoke test (synthetic Figure4RenderData)
# ---------------------------------------------------------------------------


def test_compose_figure04_produces_panels_and_finite_bbox() -> None:
    """End-to-end smoke test: with the detail window shrunk to the synthetic
    session, composition builds the a/b detail stacks, the c hexbin row, the
    track inset, and returns a finite tight bounding box."""
    render_data = synthetic_render_data()
    result = compose_figure04(
        render_data,
        diagnostic_thresholds={"hpd_overlap": 0.05, "predictive_pvalue": 0.05},
        detail_window=Figure4DetailWindow(center_index=20, half_width_samples=10),
    )

    assert isinstance(result, Figure4Composition)
    # Two 6-row detail stacks (a, b) + the 3-panel hexbin row (c) + the track
    # inset all live as axes on the returned figure.
    assert len(result.figure.axes) >= 12
    # The crop bbox must be finite (composition measured real artist extents).
    assert np.isfinite(np.asarray(result.bbox_inches.extents)).all()

    import matplotlib.pyplot as plt

    plt.close(result.figure)


def test_compose_figure04_aligns_the_two_detail_stacks_row_by_row() -> None:
    """Panel (b)'s rows sit at panel (a)'s heights although the two stacks lay
    out in separate subfigures with different decorations."""
    import matplotlib.pyplot as plt

    result = compose_figure04(
        synthetic_render_data(),
        diagnostic_thresholds={"hpd_overlap": 0.05, "predictive_pvalue": 0.05},
        detail_window=Figure4DetailWindow(center_index=20, half_width_samples=10),
    )
    result.figure.draw_without_rendering()
    axes_a, axes_b = result.figure.axes[:6], result.figure.axes[6:12]
    for ax_a, ax_b in zip(axes_a, axes_b, strict=True):
        pos_a, pos_b = ax_a.get_position(), ax_b.get_position()
        np.testing.assert_allclose((pos_b.y0, pos_b.height), (pos_a.y0, pos_a.height))
    plt.close(result.figure)


def test_compose_figure04_labels_rows_once_across_the_two_stacks() -> None:
    """Panel (a) carries the row labels, panel (b) the right-edge threshold and
    worse-fit labels (which the track inset aligns to); neither repeats them."""
    import matplotlib.pyplot as plt

    from statespacecheck_paper.figure04_layout import FIGURE04_DIAGNOSTIC_ANNOTATION_GIDS
    from statespacecheck_paper.plotting import THRESHOLD_LABEL_GID, WORSE_FIT_LABEL_GID

    result = compose_figure04(
        synthetic_render_data(),
        diagnostic_thresholds={"hpd_overlap": 0.05, "predictive_pvalue": 0.05},
        detail_window=Figure4DetailWindow(center_index=20, half_width_samples=10),
    )
    axes_a, axes_b = result.figure.axes[:6], result.figure.axes[6:12]

    assert axes_a[3].get_ylabel() == "HPD\noverlap"
    assert all(ax.get_ylabel() == "" for ax in axes_b)
    assert not any(
        text.get_gid() in FIGURE04_DIAGNOSTIC_ANNOTATION_GIDS for ax in axes_a for text in ax.texts
    )
    gids_b = [[text.get_gid() for text in ax.texts] for ax in axes_b[3:]]
    assert gids_b == [
        [THRESHOLD_LABEL_GID, WORSE_FIT_LABEL_GID],
        [THRESHOLD_LABEL_GID, WORSE_FIT_LABEL_GID],
        [WORSE_FIT_LABEL_GID],  # KL divergence has no threshold here
    ]
    plt.close(result.figure)
