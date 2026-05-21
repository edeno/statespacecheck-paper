"""Tests for ``scripts/find_immobile_replay_windows.py``.

Currently focused on ``compute_multiunit_zscore`` — the zero-variance
guard introduced in Phase 1 had no direct coverage and the prior
``nan_to_num`` path silently turned degenerate input into "no
candidates" indistinguishably from a real no-replay outcome.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    find_immobile_replay = importlib.import_module("find_immobile_replay_windows")
finally:
    if str(SCRIPTS_DIR) in sys.path:
        sys.path.remove(str(SCRIPTS_DIR))

compute_multiunit_zscore = find_immobile_replay.compute_multiunit_zscore


class TestComputeMultiunitZscore:
    def test_raises_on_zero_variance(self) -> None:
        """Every cell silent → zero std → cannot z-score.

        Reverting the guard restores ``nan_to_num(z, nan=0.0)`` which
        produces an all-zero z-score and a "no candidate windows"
        outcome. The exception message must name the degeneracy so the
        operator can distinguish "preprocessing bug" from "real silence".
        """
        n_time, n_cells = 1000, 50
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        with pytest.raises(ValueError, match=r"degenerate.*std="):
            compute_multiunit_zscore(spike_counts)

    def test_normal_path_returns_finite_zscored_array(self) -> None:
        """Sanity check: a non-degenerate Poisson input z-scores cleanly.

        Guards against a future "loosen the guard" change accidentally
        re-introducing ``nan_to_num`` (which would silently pass this
        test only because it never raised on this input — the failure
        mode here is the zero-variance test above).
        """
        rng = np.random.default_rng(0)
        n_time, n_cells = 1000, 50
        spike_counts = rng.poisson(0.5, size=(n_time, n_cells)).astype(np.int64)
        z = compute_multiunit_zscore(spike_counts)
        assert z.shape == (n_time,)
        assert np.all(np.isfinite(z))
        assert abs(float(np.mean(z))) < 0.1
        assert abs(float(np.std(z)) - 1.0) < 0.1
