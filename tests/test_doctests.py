"""Run module doctests under pytest so docstring examples can't rot.

A dedicated doctest runner is used here rather than ``--doctest-modules``
in ``[tool.pytest.ini_options] addopts``: that flag imports *every* module
under ``src/`` to collect doctests. Importing ``spyglass_pipeline`` imports
Spyglass, which connects to the lab database, and importing
``interactive.__main__`` parses the command line and exits.

This file instead lists the modules whose docstrings contain executable
examples; ``test_every_module_with_doctests_is_listed`` fails when a module
gains one without being added here. Examples written entirely as comments
(``>>> # ...``) are illustrative and not listed.
"""

from __future__ import annotations

import doctest
import importlib
import re
from pathlib import Path

import pytest

import statespacecheck_paper

# Modules whose docstring examples are executed and checked.
_DOCTEST_MODULES = [
    "statespacecheck_paper.simulation",
    "statespacecheck_paper.diagnostics",
    "statespacecheck_paper.decoding",
    "statespacecheck_paper.style",
    "statespacecheck_paper.plotting",
    "statespacecheck_paper.figure01_generation",
    "statespacecheck_paper.schematic",
    "statespacecheck_paper.figure03_protocol",
    "statespacecheck_paper.figure03_simulation",
    "statespacecheck_paper.figure03_summary",
    "statespacecheck_paper.figure03_plotting",
    "statespacecheck_paper.figure04_diagnostics",
    "statespacecheck_paper.figure04_panels",
    "statespacecheck_paper.number_format",
    "statespacecheck_paper.reported_values",
    "statespacecheck_paper.spyglass_data",
]

_EXECUTABLE_EXAMPLE = re.compile(r"^\s*>>> (?!#)", re.MULTILINE)


def test_every_module_with_doctests_is_listed() -> None:
    """Each source module with an executable ``>>>`` example is in the list."""
    src = Path(statespacecheck_paper.__file__).resolve().parent
    with_examples = {
        "statespacecheck_paper." + ".".join(path.relative_to(src).with_suffix("").parts)
        for path in src.rglob("*.py")
        if _EXECUTABLE_EXAMPLE.search(path.read_text(encoding="utf-8"))
    }
    assert with_examples == set(_DOCTEST_MODULES)


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
