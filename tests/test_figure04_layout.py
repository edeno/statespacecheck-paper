"""Tests for the Figure-4 layout (event-time shift + composition contract)."""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics
from statespacecheck_paper.figure04_layout import (
    Figure4Composition,
    _shift_diagnostic_event_times,
    compose_figure04,
)


def _make_per_cell_diagnostics(
    *, event_time: np.ndarray | None, event_hpd_overlap: np.ndarray
) -> SpikeEventDiagnostics:
    n_spikes = event_hpd_overlap.shape[0]
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=np.zeros(n_spikes),
        event_predictive_pvalue=np.zeros(n_spikes),
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
        event_time=event_time,
    )


class TestShiftDiagnosticEventTimes:
    def test_subtracts_offset(self) -> None:
        """Per-spike event times must be relative to the same time base as the
        figure axis — otherwise scatter points slide off the panels."""
        diagnostics = _make_per_cell_diagnostics(
            event_time=np.array([101.0, 101.5]),
            event_hpd_overlap=np.array([0.25, 0.75]),
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
        diagnostics = _make_per_cell_diagnostics(event_time=None, event_hpd_overlap=np.array([0.5]))
        assert _shift_diagnostic_event_times(diagnostics, 100.0) is diagnostics


def test_figure4_composition_is_frozen_with_figure_and_bbox() -> None:
    assert Figure4Composition.__dataclass_params__.frozen
    assert [f.name for f in dataclasses.fields(Figure4Composition)] == ["figure", "bbox_inches"]


def test_compose_figure04_signature() -> None:
    sig = inspect.signature(compose_figure04)
    assert list(sig.parameters) == ["render_data", "diagnostic_thresholds"]
    # ``diagnostic_thresholds`` is keyword-only.
    assert sig.parameters["diagnostic_thresholds"].kind is inspect.Parameter.KEYWORD_ONLY
