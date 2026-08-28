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


def test_decoding_imports_only_diagnostics_and_simulation() -> None:
    """The general decoder depends only on the ``diagnostics`` and general
    ``simulation`` layers — never on ``analysis`` or a figure-specific module."""
    imported = _sibling_module_imports("decoding.py")
    assert imported <= {
        "statespacecheck_paper.diagnostics",
        "statespacecheck_paper.simulation",
    }


def test_figure03_protocol_is_a_leaf() -> None:
    """The Figure-3 protocol (config + phase ladder) imports no sibling module."""
    assert _sibling_module_imports("figure03_protocol.py") == set()


def test_figure03_family_dependency_edges_are_acyclic() -> None:
    """Each Figure-3 family module imports only from its allowed lower layers,
    keeping the dependency graph acyclic (protocol < simulation < summary <
    plotting, all above the general decoding/diagnostics/simulation layers)."""
    prefix = "statespacecheck_paper."
    allowed = {
        "figure03_protocol.py": set(),
        "figure03_simulation.py": {
            prefix + "figure03_protocol",
            prefix + "decoding",
            prefix + "diagnostics",
            prefix + "simulation",
        },
        "figure03_summary.py": {
            prefix + "figure03_protocol",
            prefix + "figure03_simulation",
            prefix + "diagnostics",
        },
        "figure03_plotting.py": {
            prefix + "figure03_protocol",
            prefix + "figure03_summary",
            prefix + "diagnostics",
            prefix + "plotting",
            prefix + "style",
        },
        "figure03_generation.py": {
            prefix + "figure03_plotting",
            prefix + "figure03_protocol",
            prefix + "figure03_simulation",
            prefix + "figure03_summary",
            prefix + "scientific_artifacts",
            prefix + "style",
        },
    }
    for module_file, permitted in allowed.items():
        assert _sibling_module_imports(module_file) <= permitted, module_file


def test_figure01_and_figure02_generation_dependencies_are_explicit() -> None:
    """The early figures keep composition in package recipes and leave their
    scripts as CLI adapters."""
    prefix = "statespacecheck_paper."
    allowed = {
        "figure01_generation.py": {
            prefix + "plotting",
            prefix + "schematic",
            prefix + "style",
        },
        "figure02_generation.py": {
            prefix + "figure02_panels",
            prefix + "style",
        },
    }
    for module_file, permitted in allowed.items():
        assert _sibling_module_imports(module_file) <= permitted, module_file


def test_figure04_family_dependency_edges_are_acyclic() -> None:
    """The Figure-4 family is layered cache < workflow < layout < generation:
    cache imports no other Figure-4 module; workflow imports cache; layout
    imports workflow (never cache/config/paths); generation ties them together.

    The analysis and plotting leaves (``figure04_decoder`` /
    ``figure04_place_fields`` < ``figure04_diagnostics`` and
    ``figure04_plot_primitives`` < ``figure04_track_plots`` < ``figure04_panels``)
    sit below this layering."""
    prefix = "statespacecheck_paper."
    allowed = {
        "figure04_decoder.py": set(),
        "figure04_place_fields.py": set(),
        "figure04_diagnostics.py": {
            prefix + "diagnostics",
            prefix + "figure04_place_fields",
        },
        "figure04_plot_primitives.py": {prefix + "style"},
        "figure04_track_plots.py": {prefix + "figure04_plot_primitives"},
        "figure04_panels.py": {
            prefix + "diagnostics",
            prefix + "figure04_diagnostics",
            prefix + "figure04_plot_primitives",
            prefix + "figure04_track_plots",
            prefix + "plotting",
            prefix + "style",
        },
        "figure04_cache.py": {
            prefix + "figure04_decoder",
            prefix + "load_local_data",
        },
        "figure04_workflow.py": {
            prefix + "figure04_cache",
            prefix + "figure04_decoder",
            prefix + "figure04_diagnostics",
            prefix + "figure04_place_fields",
            prefix + "diagnostics",
            prefix + "load_local_data",
        },
        "figure04_layout.py": {
            prefix + "figure04_workflow",
            prefix + "diagnostics",
            prefix + "figure04_panels",
            prefix + "figure04_plot_primitives",
            prefix + "figure04_track_plots",
        },
        "figure04_generation.py": {
            prefix + "figure04_cache",
            prefix + "figure04_workflow",
            prefix + "figure04_layout",
            prefix + "figure04_decoder",
            prefix + "paths",
            prefix + "scientific_artifacts",
            prefix + "style",
        },
    }
    for module_file, permitted in allowed.items():
        assert _sibling_module_imports(module_file) <= permitted, module_file
