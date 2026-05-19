"""Tests for database-backed data loading helpers."""

from __future__ import annotations

import numpy as np

from statespacecheck_paper.load_data import _detect_coincident_spikes


class TestDetectCoincidentSpikes:
    """Tests for coincident-spike artifact filtering.

    The filter removes spikes that fire across many electrode groups
    within ``spike_closeness_threshold`` seconds — these are typically
    electrical artifacts (e.g. stim, EMI), not single-unit activity.
    """

    def test_removes_all_spikes_in_coincident_cluster(self) -> None:
        """Three groups with synchronous spikes => those spikes drop from
        every group, but unique spikes survive."""
        spike_times = [
            np.array([0.0, 1.0]),
            np.array([0.00001, 2.0]),
            np.array([0.00002, 3.0]),
        ]
        filtered_times, filtered_indices = _detect_coincident_spikes(
            spike_times,
            spike_closeness_threshold=0.00004,
            max_coincident_fraction=0.33,
        )
        # The 0.0/0.00001/0.00002 cluster is dropped; the 1/2/3 spikes survive.
        np.testing.assert_allclose(filtered_times[0], [1.0])
        np.testing.assert_allclose(filtered_times[1], [2.0])
        np.testing.assert_allclose(filtered_times[2], [3.0])
        for idx_arr in filtered_indices:
            np.testing.assert_array_equal(idx_arr, [1])

    def test_preserves_empty_group_slots_after_filtering(self) -> None:
        """Groups whose only spikes are filtered out must still appear in
        the result as empty arrays (not be dropped from the list)."""
        spike_times = [
            np.array([0.0]),
            np.array([0.00001]),
            np.array([1.0]),
        ]
        filtered_times, filtered_indices = _detect_coincident_spikes(
            spike_times,
            spike_closeness_threshold=0.00004,
            max_coincident_fraction=0.33,
        )
        assert len(filtered_times) == 3
        assert len(filtered_indices) == 3
        np.testing.assert_array_equal(filtered_times[0], [])
        np.testing.assert_array_equal(filtered_times[1], [])
        np.testing.assert_array_equal(filtered_times[2], [1.0])
        np.testing.assert_array_equal(filtered_indices[0], [])
        np.testing.assert_array_equal(filtered_indices[1], [])
        np.testing.assert_array_equal(filtered_indices[2], [0])

    def test_no_coincidence_passes_everything_through(self) -> None:
        """Edge case: well-separated spikes survive intact."""
        spike_times = [
            np.array([0.0, 1.0, 2.0]),
            np.array([0.5, 1.5]),
        ]
        filtered_times, filtered_indices = _detect_coincident_spikes(
            spike_times,
            spike_closeness_threshold=0.001,  # Very tight: nothing coincident.
            max_coincident_fraction=0.5,
        )
        np.testing.assert_array_equal(filtered_times[0], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(filtered_times[1], [0.5, 1.5])
        np.testing.assert_array_equal(filtered_indices[0], [0, 1, 2])
        np.testing.assert_array_equal(filtered_indices[1], [0, 1])
