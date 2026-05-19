"""Run module doctests under pytest so docstring examples can't rot.

A dedicated doctest runner is used here rather than ``--doctest-modules``
in ``[tool.pytest.ini_options] addopts``: that flag would import *every*
module under ``src/`` to collect doctests, including the ``interactive``
viewer modules whose PySide6 / pyqtgraph dependencies live in the
optional ``[interactive]`` extra and are absent from the default CI
environment — collection would then fail on ImportError.

This file instead checks the specific modules that (a) carry doctests
and (b) import cleanly without any optional extra. Add a module here
when it grows doctests worth enforcing.
"""

from __future__ import annotations

import doctest
import importlib

import pytest

# Modules whose docstring examples are executed and checked.
_DOCTEST_MODULES = [
    "statespacecheck_paper.analysis",
    "statespacecheck_paper.simulation",
    "statespacecheck_paper.style",
    "statespacecheck_paper.plotting",
    "statespacecheck_paper.figure03_demo",
]


@pytest.mark.parametrize("module_name", _DOCTEST_MODULES)
def test_module_doctests(module_name: str) -> None:
    """Every executable doctest in ``module_name`` passes.

    Examples with side effects (file writes) carry an inline
    ``# doctest: +SKIP`` and are not executed.
    """
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, (
        f"{module_name}: {results.failed} of {results.attempted} doctests failed "
        "(run `python -m doctest src/statespacecheck_paper/"
        f"{module_name.rsplit('.', 1)[-1]}.py` to see them)"
    )
