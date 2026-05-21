"""Property-based tests for simulation utilities using Hypothesis.

These tests focus on realistic scenarios within practical ranges.
For extreme edge cases, see the comprehensive unit tests.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from statespacecheck_paper.simulation import (
    normalize,
    reflect_into_interval,
    softmax_with_shift,
)


class TestNormalizeProperties:
    """Property-based tests for normalize function with realistic values."""

    @given(
        arr=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(
                min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_normalize_always_sums_to_one(self, arr: np.ndarray) -> None:
        """Property: normalized array always sums to approximately 1."""
        result = normalize(arr)

        # Check that sum is approximately 1
        np.testing.assert_allclose(np.sum(result), 1.0, rtol=1e-6, atol=1e-6)

    @given(
        arr=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(
                min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_normalize_produces_nonnegative_values(self, arr: np.ndarray) -> None:
        """Property: normalized array has all non-negative values."""
        result = normalize(arr)

        # All values should be >= 0
        assert np.all(result >= 0)


class TestReflectIntoIntervalProperties:
    """Property-based tests for reflect_into_interval function."""

    @given(
        x=st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
        xmin=st.floats(min_value=-100, max_value=-0.1, allow_nan=False, allow_infinity=False),
        xmax=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    def test_result_always_within_bounds(self, x: float, xmin: float, xmax: float) -> None:
        """Property: reflected value is always within [xmin, xmax]."""
        result = reflect_into_interval(x, xmin, xmax)

        # Result must be within bounds (with small tolerance for floating point)
        assert xmin <= result <= xmax + 1e-10, f"Result {result} not in [{xmin}, {xmax}]"

    @given(
        x=st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        xmin=st.floats(min_value=-50, max_value=0, allow_nan=False, allow_infinity=False),
        xmax=st.floats(min_value=1, max_value=50, allow_nan=False, allow_infinity=False),
    )
    def test_values_inside_bounds_unchanged(self, x: float, xmin: float, xmax: float) -> None:
        """Property: values already inside bounds are unchanged."""
        if xmin <= x <= xmax:
            result = reflect_into_interval(x, xmin, xmax)
            np.testing.assert_allclose(result, x, rtol=1e-10, atol=1e-10)

    @given(
        arr=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=100),
            elements=st.floats(
                min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False
            ),
        ),
        xmin=st.floats(min_value=-100, max_value=-0.1, allow_nan=False, allow_infinity=False),
        xmax=st.floats(min_value=0.1, max_value=100, allow_nan=False, allow_infinity=False),
    )
    def test_array_reflection_within_bounds(
        self, arr: np.ndarray, xmin: float, xmax: float
    ) -> None:
        """Property: all reflected array values are within bounds."""
        result = reflect_into_interval(arr, xmin, xmax)

        # All values must be within bounds (with small tolerance for floating point)
        assert np.all(result >= xmin - 1e-10)
        assert np.all(result <= xmax + 1e-10)


class TestSoftmaxWithShiftProperties:
    """``softmax_with_shift`` is the numerical primitive at the heart of
    the log-space decoder update. Three invariants define it
    operationally; any future "simplification" that breaks numerical
    stability or normalization fails one of them.
    """

    @given(
        ll=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=64),
            elements=st.floats(
                min_value=-500, max_value=500, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_sums_to_one(self, ll: np.ndarray) -> None:
        """The output is a probability distribution."""
        np.testing.assert_allclose(softmax_with_shift(ll).sum(), 1.0, rtol=1e-12)

    @given(
        ll=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=64),
            elements=st.floats(
                min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
            ),
        ),
        c=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_shift_invariant(self, ll: np.ndarray, c: float) -> None:
        """Adding a constant to every log-likelihood leaves the
        distribution unchanged — the load-bearing identity of the
        log-sum-exp trick. The bounded ``c`` range stays within float64
        headroom; the function's documented purpose is to handle
        unbounded ``ll`` via the internal shift, but the *property*
        only holds when adding ``c`` doesn't overflow."""
        np.testing.assert_allclose(
            softmax_with_shift(ll), softmax_with_shift(ll + c), rtol=1e-10, atol=1e-12
        )

    @given(
        ll=arrays(
            dtype=np.float64,
            shape=st.integers(min_value=1, max_value=64),
            elements=st.floats(
                min_value=-500, max_value=500, allow_nan=False, allow_infinity=False
            ),
        )
    )
    def test_outputs_are_nonneg(self, ll: np.ndarray) -> None:
        """The output is a probability distribution — every entry ≥ 0."""
        assert np.all(softmax_with_shift(ll) >= 0)
