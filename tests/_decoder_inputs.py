"""Shared decoder-test inputs used across the split analysis test modules.

``DecoderInputs`` bundles a small reproducible ``decode_and_diagnostics``
problem; ``_diag_dominant_transition`` builds a symmetric near-identity
transition matrix. Both are imported by the test modules that split out of the
old ``test_analysis.py`` (decoding, diagnostics, figure-3). The ``decoder_inputs``
fixture in ``conftest.py`` wraps ``DecoderInputs`` so it is available by name to
every test module without an import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from statespacecheck_paper.analysis import decode_and_diagnostics
from statespacecheck_paper.diagnostics import DecodingDiagnostics


@dataclass
class DecoderInputs:
    """Bundle of inputs for ``decode_and_diagnostics``."""

    spikes: np.ndarray
    xs: np.ndarray
    transition_matrix: np.ndarray
    pf_centers: np.ndarray
    pf_width: float
    rate_scale: float

    def call(self, **overrides: Any) -> DecodingDiagnostics:
        kwargs: dict[str, Any] = {
            "spikes": self.spikes,
            "xs": self.xs,
            "transition_matrix": self.transition_matrix,
            "pf_centers": self.pf_centers,
            "pf_width": self.pf_width,
            "rate_scale": self.rate_scale,
        }
        kwargs.update(overrides)
        return decode_and_diagnostics(**kwargs)


def _diag_dominant_transition(n_bins: int, peak: float = 0.9) -> np.ndarray:
    return np.eye(n_bins) * peak + (1.0 - peak) / n_bins


def make_decoder_inputs() -> DecoderInputs:
    """Small reproducible decoder problem with no misfit schedule."""
    rng = np.random.default_rng(42)
    n_time, n_cells, n_bins = 10, 3, 21
    return DecoderInputs(
        spikes=rng.poisson(1.0, size=(n_time, n_cells)),
        xs=np.linspace(0, 100, n_bins),
        transition_matrix=_diag_dominant_transition(n_bins),
        pf_centers=np.array([25.0, 50.0, 75.0]),
        pf_width=5.0,
        rate_scale=0.1,
    )
