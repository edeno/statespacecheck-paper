"""Tests for the top-level ``statespacecheck_paper`` package surface."""

from __future__ import annotations

import ast
import re
from pathlib import Path

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


def test_every_argsort_is_stable() -> None:
    """NumPy's default ``argsort`` orders equal values differently on CPUs with
    AVX2 than on others, which moved tied spike events and raster rows between
    machines. Every ``argsort`` in the package states a stable ``kind``."""
    package = Path(statespacecheck_paper.__file__).parent
    unstable = []
    for path in sorted(package.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"argsort", "sort_values"}
            ):
                continue
            kinds = [
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "kind" and isinstance(keyword.value, ast.Constant)
            ]
            if kinds not in (["stable"], ["mergesort"]):
                unstable.append(f"{path.relative_to(package)}:{node.lineno}")
    assert not unstable, f"argsort without a stable kind: {unstable}"
