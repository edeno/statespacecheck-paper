"""Tests for the shared Figure-4 protocol choices."""

from __future__ import annotations

import pytest

from statespacecheck_paper.figure04_protocol import Figure4DetailWindow


class TestFigure4DetailWindow:
    def test_converts_center_and_half_width_to_slice(self) -> None:
        assert Figure4DetailWindow(center_index=20, half_width_samples=10).to_slice(40) == slice(
            10, 30
        )

    @pytest.mark.parametrize(
        ("center_index", "half_width_samples"),
        [(-1, 10), (20, 0), (20, -1), (20.0, 10)],
    )
    def test_rejects_invalid_values(self, center_index: int, half_width_samples: int) -> None:
        with pytest.raises(ValueError):
            Figure4DetailWindow(
                center_index=center_index,
                half_width_samples=half_width_samples,
            )

    def test_rejects_window_outside_recording(self) -> None:
        with pytest.raises(ValueError, match="outside the recording timeline"):
            Figure4DetailWindow(center_index=5, half_width_samples=10).to_slice(40)

    def test_rejects_invalid_recording_length(self) -> None:
        with pytest.raises(ValueError, match="n_time_samples"):
            Figure4DetailWindow(center_index=5, half_width_samples=2).to_slice(0)
