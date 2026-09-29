"""Enforce the intended module dependency direction.

``diagnostics.py`` is a leaf of the paper's dependency graph: it computes the
goodness-of-fit diagnostics from primitives plus the external ``statespacecheck``
package, so it must not import any sibling ``statespacecheck_paper`` module
(``decoding``, the ``figure0*`` layers, plotting, etc.). The tests below pin the
allowed edges of the other layers, check that the whole module graph is
acyclic, and check that docs/figure-pipeline.md lists that graph exactly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import statespacecheck_paper
from statespacecheck_paper import figure04_cache

_SRC = Path(statespacecheck_paper.__file__).resolve().parent


def _sibling_module_imports(module_filename: str) -> set[str]:
    """Return the set of sibling ``statespacecheck_paper`` modules imported."""
    tree = ast.parse((_SRC / module_filename).read_text(encoding="utf-8"))
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
    """The diagnostics layer is a leaf: besides the standard library it imports
    numpy and the external ``statespacecheck`` package only, never a sibling
    paper module."""
    assert _sibling_module_imports("diagnostics.py") == set()


def test_decoding_imports_only_diagnostics_and_simulation() -> None:
    """The general decoder depends only on the ``diagnostics`` and general
    ``simulation`` layers — never on a figure-specific module."""
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
            prefix + "number_format",
            prefix + "plotting",
            prefix + "style",
        },
        "figure03_generation.py": {
            prefix + "diagnostics",
            prefix + "figure03_plotting",
            prefix + "figure03_protocol",
            prefix + "figure03_simulation",
            prefix + "figure03_summary",
            prefix + "paths",
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
            prefix + "diagnostics",
            prefix + "paths",
            prefix + "plotting",
            prefix + "schematic",
            prefix + "style",
        },
        "figure02_generation.py": {
            prefix + "figure02_panels",
            prefix + "paths",
            prefix + "style",
        },
    }
    for module_file, permitted in allowed.items():
        assert _sibling_module_imports(module_file) <= permitted, module_file


def test_figure04_family_dependency_edges_are_acyclic() -> None:
    """The Figure-4 family is layered fit/cache < workflow < summary/layout < generation:
    fit imports only the decoder construction, input, and place-field modules;
    cache imports only ``figure04_decoder`` (the configuration it hashes) and
    ``figure04_input`` (the input-file name and checksum); workflow imports
    fit and cache; summary and layout import workflow (never cache/config/paths);
    generation ties them together.

    The analysis and plotting modules (``figure04_decoder`` /
    ``figure04_place_fields`` < ``figure04_diagnostics`` and
    ``figure04_plot_primitives`` < ``figure04_track_plots`` < ``figure04_panels``)
    sit below this layering."""
    prefix = "statespacecheck_paper."
    allowed = {
        "figure04_models.py": set(),
        "figure04_decoder.py": {prefix + "diagnostics"},
        "figure04_place_fields.py": set(),
        "figure04_diagnostics.py": {
            prefix + "diagnostics",
            prefix + "figure04_place_fields",
        },
        "figure04_plot_primitives.py": {prefix + "figure04_place_fields", prefix + "style"},
        "figure04_track_plots.py": {prefix + "figure04_plot_primitives"},
        "figure04_panels.py": {
            prefix + "diagnostics",
            prefix + "figure04_diagnostics",
            prefix + "figure04_models",
            prefix + "figure04_place_fields",
            prefix + "figure04_plot_primitives",
            prefix + "figure04_track_plots",
            prefix + "plotting",
            prefix + "style",
        },
        "figure04_fit.py": {
            prefix + "figure04_decoder",
            prefix + "figure04_input",
            prefix + "figure04_place_fields",
        },
        "figure04_cache.py": {
            prefix + "figure04_decoder",
            prefix + "figure04_input",
        },
        "figure04_workflow.py": {
            prefix + "figure04_cache",
            prefix + "figure04_decoder",
            prefix + "figure04_diagnostics",
            prefix + "figure04_fit",
            prefix + "diagnostics",
            prefix + "figure04_input",
        },
        "figure04_summary.py": {
            prefix + "diagnostics",
            prefix + "figure04_diagnostics",
            prefix + "figure04_models",
            prefix + "figure04_workflow",
        },
        "figure04_layout.py": {
            prefix + "figure04_workflow",
            prefix + "diagnostics",
            prefix + "figure04_models",
            prefix + "figure04_panels",
            prefix + "figure04_track_plots",
            prefix + "plotting",
            prefix + "style",
        },
        "figure04_generation.py": {
            prefix + "diagnostics",
            prefix + "figure04_cache",
            prefix + "figure04_workflow",
            prefix + "figure04_summary",
            prefix + "figure04_layout",
            prefix + "figure04_decoder",
            prefix + "figure04_models",
            prefix + "paths",
            prefix + "scientific_artifacts",
            prefix + "style",
        },
    }
    for module_file, permitted in allowed.items():
        assert _sibling_module_imports(module_file) <= permitted, module_file


def test_reported_values_imports_no_analysis_module() -> None:
    """The macro emitter reads committed summaries; it needs only the shared
    rounding and the published-data identifiers, not the data stack."""
    prefix = "statespacecheck_paper."
    assert _sibling_module_imports("reported_values.py") <= {
        prefix + "number_format",
        prefix + "paths",
    }


def test_no_module_imports_the_download() -> None:
    """Editing the Zenodo download may be relabeled rather than regenerated
    (docs/development.md) only while no figure code imports it."""
    prefix = "statespacecheck_paper."
    modules = [path.relative_to(_SRC).as_posix() for path in sorted(_SRC.rglob("*.py"))]
    importers = [
        module
        for module in modules
        if prefix + "figure04_download" in _sibling_module_imports(module)
    ]
    assert importers == []


def test_no_figure_or_analysis_module_imports_the_spyglass_pipeline() -> None:
    """The Spyglass pipeline runs upstream of the figures, which read the archived
    input file instead; ``spyglass_pipeline.figure04_schema`` connects to the lab database
    when imported. Only the ``spyglass_pipeline`` package itself may import from it."""
    package = "statespacecheck_paper.spyglass_pipeline"
    short = "spyglass_pipeline"

    def imports_package(module_file: str) -> bool:
        tree = ast.parse((_SRC / module_file).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # ``from statespacecheck_paper import spyglass_pipeline`` and relative forms.
                if any(alias.name == short for alias in node.names) or module in {short, package}:
                    return True
                if module.startswith((package + ".", short + ".")):
                    return True
            elif isinstance(node, ast.Import) and any(
                alias.name == package or alias.name.startswith(package + ".")
                for alias in node.names
            ):
                return True
        return False

    modules = [path.relative_to(_SRC).as_posix() for path in sorted(_SRC.rglob("*.py"))]
    assert "spyglass_pipeline/figure04_input.py" in modules
    importers = [m for m in modules if not m.startswith(short + "/") and imports_package(m)]
    assert importers == []


def test_site_export_depends_only_on_analysis_layers() -> None:
    """The website export reads the figure pipelines' outputs and the reported
    values; it sits above both figure families and nothing imports it."""
    prefix = "statespacecheck_paper."
    assert _sibling_module_imports("site_export.py") <= {
        prefix + "decoding",
        prefix + "diagnostics",
        prefix + "figure03_generation",
        prefix + "figure03_protocol",
        prefix + "figure03_simulation",
        prefix + "figure03_summary",
        prefix + "figure04_cache",
        prefix + "figure04_decoder",
        prefix + "figure04_diagnostics",
        prefix + "figure04_generation",
        prefix + "figure04_layout",
        prefix + "figure04_models",
        prefix + "figure04_place_fields",
        prefix + "figure04_workflow",
        prefix + "number_format",
        prefix + "paths",
        prefix + "reported_values",
        prefix + "simulation",
        prefix + "style",
    }
    for path in sorted(_SRC.rglob("*.py")):
        if path.name == "site_export.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                # Covers ``from statespacecheck_paper(.site_export) import ...``
                # and the relative forms ``from . import site_export`` /
                # ``from .site_export import ...``.
                assert not (node.module or "").endswith("site_export"), path
                assert "site_export" not in names, path
            elif isinstance(node, ast.Import):
                assert all(not a.name.endswith("site_export") for a in node.names), path


# ---------------------------------------------------------------------------
# The complete module graph, and the copy of it in docs/figure-pipeline.md
# ---------------------------------------------------------------------------

_PACKAGE = "statespacecheck_paper"
_GRAPH_DOC = Path(__file__).resolve().parents[1] / "docs" / "figure-pipeline.md"
_MAIN_GUARDS = {"__name__ == '__main__'", '__name__ == "__main__"'}


def _module_name(path: Path) -> str:
    """Dotted name under the package, e.g. ``figure04_cache``, ``interactive.cache``.

    A subpackage's ``__init__`` is named after the subpackage.
    """
    parts = path.relative_to(_SRC).with_suffix("").parts
    if parts[-1] == "__init__" and len(parts) > 1:
        parts = parts[:-1]
    return ".".join(parts)


def _module_graph() -> dict[str, dict[str, set[str]]]:
    """Every source module's imports of sibling modules, by kind.

    ``top`` are module-level imports, ``lazy`` imports inside a function, and
    ``type_only`` imports under ``if TYPE_CHECKING``; each name appears only in
    the first of these that applies. Imports under ``if __name__ ==
    "__main__"`` run only when a module is executed as a script and are left
    out. Relative imports are resolved, and ``from package import module``
    counts as importing the module.
    """
    paths = sorted(_SRC.rglob("*.py"))
    names = {_module_name(path) for path in paths}

    def resolve(node: ast.Import | ast.ImportFrom, importer: str, is_package: bool) -> set[str]:
        if isinstance(node, ast.Import):
            return {
                alias.name.removeprefix(_PACKAGE + ".")
                for alias in node.names
                if alias.name.startswith(_PACKAGE + ".")
            }
        if node.level:
            package = importer if is_package else importer.rpartition(".")[0]
            base = ".".join(filter(None, [package, node.module]))
        elif node.module is not None and (
            node.module == _PACKAGE or node.module.startswith(_PACKAGE + ".")
        ):
            base = node.module.removeprefix(_PACKAGE).removeprefix(".")
        else:
            return set()
        targets = set()
        for alias in node.names:
            submodule = ".".join(filter(None, [base, alias.name]))
            if submodule in names:
                targets.add(submodule)
            elif base:
                targets.add(base)
        return targets

    def visit(
        node: ast.AST, kind: str, found: list[tuple[str, ast.Import | ast.ImportFrom]]
    ) -> None:
        for child in ast.iter_child_nodes(node):
            child_kind = kind
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                child_kind = "lazy" if kind == "top" else kind
            elif isinstance(child, ast.If):
                test = ast.unparse(child.test)
                if test == "TYPE_CHECKING":
                    child_kind = "type_only"
                elif test in _MAIN_GUARDS:
                    continue
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                found.append((child_kind, child))
            visit(child, child_kind, found)

    graph: dict[str, dict[str, set[str]]] = {}
    for path in paths:
        importer = _module_name(path)
        is_package = path.name == "__init__.py"
        found: list[tuple[str, ast.Import | ast.ImportFrom]] = []
        visit(ast.parse(path.read_text(encoding="utf-8")), "top", found)
        kinds: dict[str, set[str]] = {"top": set(), "lazy": set(), "type_only": set()}
        for kind, node in found:
            kinds[kind] |= resolve(node, importer, is_package)
        kinds["lazy"] -= kinds["top"]
        kinds["type_only"] -= kinds["top"] | kinds["lazy"]
        graph[importer] = {kind: targets - {importer} for kind, targets in kinds.items()}
    return graph


def test_module_graph_is_acyclic() -> None:
    """No chain of module-level or function-level imports returns to its start."""
    edges = {name: kinds["top"] | kinds["lazy"] for name, kinds in _module_graph().items()}
    visiting: set[str] = set()
    done: set[str] = set()

    def check(name: str, path: tuple[str, ...]) -> None:
        if name in done:
            return
        assert name not in visiting, f"import cycle: {' -> '.join((*path, name))}"
        visiting.add(name)
        for target in sorted(edges.get(name, set())):
            check(target, (*path, name))
        visiting.discard(name)
        done.add(name)

    for name in sorted(edges):
        check(name, ())


def _documented_graph() -> dict[str, dict[str, set[str]]]:
    """Parse the dependency-graph block of docs/figure-pipeline.md.

    Each line reads ``module → a, b; lazy: c; type-only: d``; ``(none)`` marks
    a module without sibling imports.
    """
    block = _GRAPH_DOC.read_text(encoding="utf-8").split("<!-- module-graph -->")[1]
    graph: dict[str, dict[str, set[str]]] = {}
    for line in block.splitlines():
        if "→" not in line:
            continue
        module, _, rest = (part.strip() for part in line.partition("→"))
        kinds: dict[str, set[str]] = {"top": set(), "lazy": set(), "type_only": set()}
        for field in rest.split(";"):
            label, _, listed = field.strip().rpartition(":")
            kind = {"": "top", "lazy": "lazy", "type-only": "type_only"}[label.strip()]
            kinds[kind] = {n.strip() for n in listed.split(",")} - {"", "(none)"}
        graph[module] = kinds
    return graph


def test_documented_module_graph_matches_the_source() -> None:
    """The graph in docs/figure-pipeline.md lists every module and exactly its imports."""
    assert _documented_graph() == _module_graph()


# ---------------------------------------------------------------------------
# The decode-cache fingerprint covers everything the fitted decode imports
# ---------------------------------------------------------------------------

# Paper modules that decode-hashed modules may import although their source is
# not in the decode fingerprint, each with the only names that may be imported
# from it. An entry must be unable to change the fitted decode; justify each.
_DECODE_NEUTRAL_IMPORTS: dict[str, frozenset[str]] = {
    # figure04_decoder: the default of Figure4DiagnosticsConfig.hpd_coverage.
    # The fit never reads it; the diagnostics fingerprint hashes the configured
    # coverage value.
    "diagnostics": frozenset({"HPD_COVERAGE"}),
    # figure04_input: the published epoch, which decides only whether a missing
    # input file's error message offers the download command. The decode
    # fingerprint hashes the requested epoch and the input file's content.
    "paths": frozenset({"FIGURE04_INPUTS_EPOCH"}),
}


def _names_imported_from(importer: str, target: str) -> set[str]:
    """Names the top-level module ``importer`` imports from sibling ``target``.

    ``import statespacecheck_paper.target`` and ``from statespacecheck_paper
    import target`` bring in the whole module, reported as ``"*"``.
    """
    tree = ast.parse((_SRC / f"{importer}.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == f"{_PACKAGE}.{target}" for alias in node.names):
                names.add("*")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if (node.level == 0 and module == f"{_PACKAGE}.{target}") or (
                node.level == 1 and module == target
            ):
                names |= {alias.name for alias in node.names}
            elif (node.level == 0 and module == _PACKAGE) or (node.level == 1 and not module):
                if any(alias.name == target for alias in node.names):
                    names.add("*")
    return names


def test_decode_hashed_modules_import_only_hashed_or_decode_neutral_modules() -> None:
    """Every paper module reachable from a decode-hashed module (module-level and
    function-level imports, transitively) is decode-hashed itself or named in
    ``_DECODE_NEUTRAL_IMPORTS``, and only the allowlisted names are imported
    from it. Otherwise an edit to an unhashed module could change the decode
    while a stale decode cache is still accepted."""
    graph = _module_graph()
    hashed = {name.removesuffix(".py") for name in figure04_cache._DECODE_SOURCE_FILES}
    assert hashed <= graph.keys()

    def imports(name: str) -> set[str]:
        return graph[name]["top"] | graph[name]["lazy"]

    reached: set[str] = set()
    pending = sorted(hashed)
    while pending:
        for target in sorted(imports(pending.pop()) - reached):
            reached.add(target)
            pending.append(target)
    unhashed = reached - hashed
    # Equality also catches an allowlist entry that is no longer needed.
    assert unhashed == _DECODE_NEUTRAL_IMPORTS.keys()
    for importer in sorted(hashed | unhashed):
        for target in sorted(imports(importer) & unhashed):
            imported = _names_imported_from(importer, target)
            assert imported <= _DECODE_NEUTRAL_IMPORTS[target], (importer, target, imported)
