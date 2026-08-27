"""Shared fixtures available to every test module."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ._decoder_inputs import DecoderInputs, make_decoder_inputs


@pytest.fixture
def rng() -> np.random.Generator:
    """Reproducible RNG seeded with 42 — the canonical seed for this suite."""
    return np.random.default_rng(42)


@pytest.fixture
def decoder_inputs() -> DecoderInputs:
    """Small reproducible decoder problem with no misfit schedule."""
    return make_decoder_inputs()


@pytest.fixture
def fresh_axes() -> Iterator[tuple[plt.Figure, plt.Axes]]:
    """A fresh ``(fig, ax)`` pair, automatically closed even if the test fails.

    Use this whenever a test inspects matplotlib state — manual close calls
    leak figures on assertion failure, which slows the suite and pollutes
    pyplot's global state across tests.
    """
    fig, ax = plt.subplots()
    try:
        yield fig, ax
    finally:
        plt.close(fig)
