"""Tests for database-backed data loading helpers."""

from __future__ import annotations

import numpy as np

from statespacecheck_paper.load_data import _detect_coincident_spikes


class TestDetectCoincidentSpikes:
    """Tests for coincident-spike artifact filtering."""

    def test_removes_all_spikes_in_coincident_cluster(self) -> None:
        """A close multi-group cluster should not retain its first spike."""
        spike_times = [
            np.array([0.0, 1.0]),
            np.array([0.00001, 2.0]),
            np.array([0.00002, 3.0]),
        ]

        filtered_spike_times, filtered_indices = _detect_coincident_spikes(
            spike_times,
            spike_closeness_threshold=0.00004,
            max_coincident_fraction=0.33,
        )

        np.testing.assert_allclose(filtered_spike_times[0], [1.0])
        np.testing.assert_allclose(filtered_spike_times[1], [2.0])
        np.testing.assert_allclose(filtered_spike_times[2], [3.0])
        np.testing.assert_array_equal(filtered_indices[0], [1])
        np.testing.assert_array_equal(filtered_indices[1], [1])
        np.testing.assert_array_equal(filtered_indices[2], [1])

    def test_preserves_empty_group_slots_after_filtering(self) -> None:
        """Groups with all spikes removed should remain present as empty arrays."""
        spike_times = [
            np.array([0.0]),
            np.array([0.00001]),
            np.array([1.0]),
        ]

        filtered_spike_times, filtered_indices = _detect_coincident_spikes(
            spike_times,
            spike_closeness_threshold=0.00004,
            max_coincident_fraction=0.33,
        )

        assert len(filtered_spike_times) == 3
        assert len(filtered_indices) == 3
        np.testing.assert_array_equal(filtered_spike_times[0], [])
        np.testing.assert_array_equal(filtered_spike_times[1], [])
        np.testing.assert_array_equal(filtered_spike_times[2], [1.0])
        np.testing.assert_array_equal(filtered_indices[0], [])
        np.testing.assert_array_equal(filtered_indices[1], [])
        np.testing.assert_array_equal(filtered_indices[2], [0])
