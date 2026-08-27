"""Shared decoder-test inputs used across the split analysis test modules.

``DecoderInputs`` bundles a small reproducible ``decode_with_diagnostics``
problem; ``_diag_dominant_transition`` builds a symmetric near-identity
transition matrix. Both are imported by the test modules that split out of the
old ``test_analysis.py`` (decoding, figure-3). The ``decoder_inputs`` fixture in
``conftest.py`` wraps ``DecoderInputs`` so it is available by name to every test
module without an import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from statespacecheck_paper.decoding import decode_with_diagnostics
from statespacecheck_paper.diagnostics import DecodingDiagnostics


@dataclass
class DecoderInputs:
    """Bundle of inputs for ``decode_with_diagnostics``."""

    spike_counts: np.ndarray
    position_bins: np.ndarray
    transition_matrix: np.ndarray
    place_field_centers: np.ndarray
    place_field_std: float
    place_field_rate_scale: float

    def call(self, **overrides: Any) -> DecodingDiagnostics:
        kwargs: dict[str, Any] = {
            "spike_counts": self.spike_counts,
            "position_bins": self.position_bins,
            "transition_matrix": self.transition_matrix,
            "place_field_centers": self.place_field_centers,
            "place_field_std": self.place_field_std,
            "place_field_rate_scale": self.place_field_rate_scale,
        }
        kwargs.update(overrides)
        return decode_with_diagnostics(**kwargs)


def _diag_dominant_transition(n_bins: int, peak: float = 0.9) -> np.ndarray:
    return np.eye(n_bins) * peak + (1.0 - peak) / n_bins


def make_decoder_inputs() -> DecoderInputs:
    """Small reproducible decoder problem with no override schedule."""
    rng = np.random.default_rng(42)
    n_time, n_cells, n_bins = 10, 3, 21
    return DecoderInputs(
        spike_counts=rng.poisson(1.0, size=(n_time, n_cells)),
        position_bins=np.linspace(0, 100, n_bins),
        transition_matrix=_diag_dominant_transition(n_bins),
        place_field_centers=np.array([25.0, 50.0, 75.0]),
        place_field_std=5.0,
        place_field_rate_scale=0.1,
    )
