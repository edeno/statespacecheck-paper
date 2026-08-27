"""Enforce the intended module dependency direction.

``diagnostics.py`` is the leaf of the paper's dependency graph: it computes the
goodness-of-fit diagnostics from primitives plus the external ``statespacecheck``
package, so it must not import any sibling ``statespacecheck_paper`` module
(``analysis``, the forthcoming ``decoding`` / ``figure03_*`` layers, plotting,
etc.). Later phases extend this contract as modules move.
"""

from __future__ import annotations

import ast
from pathlib import Path

import statespacecheck_paper

_SRC = Path(statespacecheck_paper.__file__).resolve().parent


def _sibling_module_imports(module_filename: str) -> set[str]:
    """Return the set of sibling ``statespacecheck_paper`` modules imported."""
    tree = ast.parse((_SRC / module_filename).read_text())
    siblings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("statespacecheck_paper"):
                siblings.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("statespacecheck_paper"):
                    siblings.add(alias.name)
    return siblings


def test_diagnostics_imports_no_sibling_paper_module() -> None:
    """The diagnostics layer is a leaf: it imports numpy/scipy and the external
    ``statespacecheck`` package only, never a sibling paper module."""
    assert _sibling_module_imports("diagnostics.py") == set()
