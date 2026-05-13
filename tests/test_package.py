"""Tests for the top-level ``statespacecheck_paper`` package surface."""

from __future__ import annotations

import re

import statespacecheck_paper


def test_version_is_exposed_as_pep440_string() -> None:
    """``__version__`` is part of the public API: importable, non-empty,
    and a PEP 440-shaped string. Pinning the exact value here would just
    be a tautology with ``pyproject.toml``; pinning the *shape* catches
    regressions like dropping the attribute, returning ``None``, or
    accidentally setting it to a tuple."""
    version = statespacecheck_paper.__version__
    assert isinstance(version, str)
    assert version, "__version__ must not be empty"
    # PEP 440 release segment + optional pre/post/dev/local.
    assert re.match(r"^\d+(\.\d+)*([a-z]+\d*)?(\.dev\d+)?(\+[\w.]+)?$", version), (
        f"version {version!r} does not look like a PEP 440 release identifier"
    )
