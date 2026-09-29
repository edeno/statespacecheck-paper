"""Shared builder for events-only ``SpikeEventDiagnostics`` test inputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics


def event_diagnostics(
    *,
    hpd: np.ndarray | None = None,
    kl: np.ndarray | None = None,
    pvalue: np.ndarray | None = None,
    time_ind: Sequence[int] | None = None,
    cell_ind: Sequence[int] | None = None,
    event_time: np.ndarray | None = None,
) -> SpikeEventDiagnostics:
    """Build ``SpikeEventDiagnostics`` carrying only per-spike event arrays.

    The dense ``(n_time, n_cells)`` matrices and per-spike likelihoods are
    omitted. Metrics not given are zero; event indices not given are zero.

    Parameters
    ----------
    hpd, kl, pvalue : np.ndarray, shape (n_spikes,), optional
        Per-event HPD overlap, KL divergence, and predictive p-value.
    time_ind, cell_ind : sequence of int, length n_spikes, optional
        Per-event time-bin and cell indices.
    event_time : np.ndarray, shape (n_spikes,), optional
        Per-event timestamps.
    """
    given = [a for a in (hpd, kl, pvalue, time_ind) if a is not None]
    n_spikes = len(given[0])
    zeros = np.zeros(n_spikes)
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp)
        if time_ind is None
        else np.asarray(time_ind, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp)
        if cell_ind is None
        else np.asarray(cell_ind, dtype=np.intp),
        event_hpd_overlap=zeros if hpd is None else hpd,
        event_kl_divergence=zeros if kl is None else kl,
        event_predictive_pvalue=zeros if pvalue is None else pvalue,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
        event_time=event_time,
    )
