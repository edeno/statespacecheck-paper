"""Tests for the top-level ``statespacecheck_paper`` package surface."""

from __future__ import annotations

import re

import statespacecheck_paper
from statespacecheck_paper.paths import REPO_ROOT


def test_version_is_exposed_as_pep440_string() -> None:
    version = statespacecheck_paper.__version__
    assert isinstance(version, str)
    assert version, "__version__ must not be empty"
    # PEP 440 release segment + optional pre/post/dev/local.
    assert re.match(r"^\d+(\.\d+)*([a-z]+\d*)?(\.dev\d+)?(\+[\w.]+)?$", version), (
        f"version {version!r} does not look like a PEP 440 release identifier"
    )


def test_version_is_the_one_pyproject_sets() -> None:
    """``__version__`` reports the checkout's version, not a stale install's."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^\[project\]$.*?^version = "([^"]*)"$', pyproject, re.MULTILINE | re.DOTALL)
    assert match is not None
    assert statespacecheck_paper.__version__ == match.group(1)
