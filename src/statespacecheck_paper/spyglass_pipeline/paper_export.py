"""Record the Figure-4 input fetch in a Spyglass paper export.

A Spyglass paper export logs exactly which database entries and files a paper
used and packages them for sharing. This module runs the Figure-4 input fetch
(:func:`~statespacecheck_paper.spyglass_pipeline.figure04_input.fetch_figure04_inputs`)
inside an export session, previews what would be logged without writing, lists
what was logged, and packages it; it refuses to run with a Spyglass whose export
tables do not match the database. ``scripts/spyglass_export_figure04.py`` is the
CLI. Every Spyglass import is inside a function.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from statespacecheck_paper.spyglass_pipeline.figure04_input import (
    FIGURE04_EPOCH_NAME,
    FIGURE04_NWB_FILE_NAME,
    Figure4Inputs,
    fetch_figure04_inputs,
)

# ``name [= default] : type`` lines of a DataJoint definition (not ``->`` or index lines).
_ATTRIBUTE_LINE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(?:=[^:#\n]*)?:", re.MULTILINE)


def declared_attribute_names(definition: str) -> set[str]:
    """Return the attribute names a DataJoint table definition declares directly.

    Attributes inherited through ``->`` references are not included.

    Parameters
    ----------
    definition : str
        DataJoint table definition.

    Returns
    -------
    set of str
        Declared attribute names.

    Examples
    --------
    >>> sorted(declared_attribute_names('''
    ...     -> master
    ...     table_id: int
    ...     ---
    ...     time=CURRENT_TIMESTAMP: timestamp  # when
    ...     unique index (export_id, table_id)
    ... '''))
    ['table_id', 'time']
    """
    return set(_ATTRIBUTE_LINE.findall(definition))


def check_export_tables_match_spyglass() -> None:
    """Refuse to export with a Spyglass that does not declare the database's export columns.

    The export tables of the lab database can be newer than the installed
    Spyglass (the version in ``uv.lock`` is), and an older Spyglass would write
    rows its successors do not expect. ``Export.make`` also requires packaging
    with the same Spyglass ``x.y.z`` that logged the selection, so a selection
    logged by a Spyglass that cannot package would be stranded. Run this before
    logging as well as before packaging. Read-only.

    Raises
    ------
    RuntimeError
        If a secondary column of the database's ``ExportSelection`` or ``Export``
        tables is not declared by the installed Spyglass.
    """
    from spyglass.common.common_usage import Export, ExportSelection

    tables = (
        ExportSelection,
        ExportSelection.Table,
        ExportSelection.File,
        Export,
        Export.Table,
        Export.File,
    )
    unknown = [
        f"{table.full_table_name}.{name}"
        for table in tables
        for name in table.heading.secondary_attributes
        if name not in declared_attribute_names(table.definition)
    ]
    if unknown:
        raise RuntimeError(
            f"Installed Spyglass predates the database's export tables (undeclared columns: "
            f"{unknown}); run the export with the lab's current Spyglass."
        )


def dry_run_figure04_export_log(
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> list[dict[str, str]]:
    """Return what an export session would log for the Figure-4 fetch, writing nothing.

    Runs :func:`fetch_figure04_inputs` with Spyglass export logging switched on
    (a placeholder export ID) and the ``ExportSelection.Table`` / ``.File`` inserts
    replaced by a recorder, then restores both. Reads the database and the analysis
    files like a real fetch. This relies on how Spyglass logs exports; if nothing is
    recorded, treat that as a changed mechanism, not as an empty export.

    Parameters
    ----------
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    list of dict of str to str
        One entry per row that would be inserted: ``part`` (``"Table"`` or
        ``"File"``) plus the row's fields (e.g. ``table_name`` and
        ``restriction``, or ``analysis_file_name``).
    """
    import os

    from spyglass.common.common_usage import ExportSelection
    from spyglass.utils.mixins.export import EXPORT_ENV_VAR

    recorded: list[dict[str, str]] = []

    def recorder(part: str) -> Any:
        def record(self: Any, rows: Any, *args: Any, **kwargs: Any) -> None:
            rows = [rows] if isinstance(rows, Mapping) else list(rows)
            recorded.extend({"part": part, **{k: str(v) for k, v in r.items()}} for r in rows)

        return record

    parts = {name: getattr(ExportSelection, name) for name in ("Table", "File")}
    originals = {
        (name, method): getattr(part, method)
        for name, part in parts.items()
        for method in ("insert", "insert1")
    }
    previous_export_id = os.environ.get(EXPORT_ENV_VAR)
    try:
        for (name, method), _ in originals.items():
            setattr(parts[name], method, recorder(name))
        os.environ[EXPORT_ENV_VAR] = "999999999"
        fetch_figure04_inputs(nwb_file_name, epoch_name)
    finally:
        if previous_export_id is None:
            os.environ.pop(EXPORT_ENV_VAR, None)
        else:
            os.environ[EXPORT_ENV_VAR] = previous_export_id
        for (name, method), original in originals.items():
            setattr(parts[name], method, original)
    return recorded


def unrestricted_log_entries(entries: Sequence[Mapping[str, str]]) -> list[str]:
    """Return the tables an export log would include whole.

    Spyglass's export logging records ``table & True`` (as in
    ``ensure_single_entry()``, used by ``fetch1_dataframe``) as an unrestricted
    fetch, and packaging then exports the entire table.

    Parameters
    ----------
    entries : sequence of Mapping of str to str
        Log rows, as returned by :func:`dry_run_figure04_export_log`.

    Returns
    -------
    list of str
        Names of tables with an unrestricted log entry.

    Examples
    --------
    >>> unrestricted_log_entries([
    ...     {"part": "Table", "table_name": "a", "restriction": "(True)"},
    ...     {"part": "Table", "table_name": "b", "restriction": "(x=1)"},
    ...     {"part": "File", "analysis_file_name": "f.nwb"},
    ... ])
    ['a']
    """
    return [
        entry["table_name"]
        for entry in entries
        if entry.get("part") == "Table"
        and entry.get("restriction", "").strip("() ") in ("", "True", "1")
    ]


def log_figure04_export(
    paper_id: str,
    analysis_id: str,
    nwb_file_name: str = FIGURE04_NWB_FILE_NAME,
    epoch_name: str = FIGURE04_EPOCH_NAME,
) -> Figure4Inputs:
    """Fetch the Figure-4 inputs inside a new Spyglass export session.

    **Writes to the lab database**: ``ExportSelection.start_export`` inserts a
    selection entry, and every Spyglass fetch until ``stop_export`` is logged
    against it. Before that, :func:`check_export_tables_match_spyglass` runs, the
    ``paper_id`` is checked to be new, and the fetch is rehearsed with
    :func:`dry_run_figure04_export_log` (refusing an export that would include a
    whole table). Packaging is :func:`package_figure04_export`, which must use the
    same Spyglass version.

    Parameters
    ----------
    paper_id : str
        New export paper ID (at most 32 characters).
    analysis_id : str
        Analysis label within the paper (at most 32 characters).
    nwb_file_name : str, optional
        Spyglass NWB file name. Default is the Figure-4 session.
    epoch_name : str, optional
        Epoch interval name. Default is the Figure-4 epoch.

    Returns
    -------
    Figure4Inputs
        The inputs fetched while logging.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`, or if the rehearsal
        records nothing or an unrestricted table.
    ValueError
        If ``paper_id`` already has export selections. Re-starting an existing
        ``(paper_id, analysis_id)`` deletes its packaged ``Export`` entry, and
        packaging rebuilds the paper's one package from all of its selections, so
        this refuses rather than change a previous export.
    """
    from spyglass.common.common_usage import ExportSelection

    check_export_tables_match_spyglass()
    selection = ExportSelection()
    if len(selection & {"paper_id": paper_id}) > 0:
        raise ValueError(f"paper_id {paper_id!r} already has export selections; choose a new one")
    # Rehearse the fetch with logging recorded, not written: this also imports every
    # module the fetch needs, so nothing can fail for that reason mid-export.
    rehearsal = dry_run_figure04_export_log(nwb_file_name, epoch_name)
    if not rehearsal:
        raise RuntimeError("The export rehearsal recorded nothing; Spyglass's logging changed")
    if unrestricted := unrestricted_log_entries(rehearsal):
        raise RuntimeError(f"The export would include whole tables: {sorted(set(unrestricted))}")

    selection.start_export(paper_id=paper_id, analysis_id=analysis_id)
    try:
        return fetch_figure04_inputs(nwb_file_name, epoch_name)
    finally:
        selection.stop_export()


def describe_figure04_export(paper_id: str) -> list[str]:
    """List the tables and files logged for an export (read-only).

    Parameters
    ----------
    paper_id : str
        Export paper ID.

    Returns
    -------
    list of str
        Printable lines: each logged table with its row count, then each file.
    """
    from spyglass.common.common_usage import ExportSelection

    selection = ExportSelection()
    tables = selection.preview_tables(paper_id=paper_id)
    files = selection.list_file_paths({"paper_id": paper_id}, as_dict=False)
    return [
        "Logged tables:",
        *(f"  {table.full_table_name}: {len(table)} rows" for table in tables),
        "Logged files:",
        *(f"  {path}" for path in sorted(files)),
    ]


def package_figure04_export(paper_id: str) -> None:
    """Package a logged export with ``Export().populate_paper``.

    **Writes to the lab database** (``Export`` entries listing the tables and
    files) and to the Spyglass export directory (a ``mysqldump`` script
    ``_ExportSQL_<paper_id>.sh``, the Spyglass version, and ``environment.yml``;
    it may also create ``~/.my.cnf``). Running that script produces the SQL dump.

    Parameters
    ----------
    paper_id : str
        Export paper ID, logged with the installed Spyglass version.

    Raises
    ------
    RuntimeError
        From :func:`check_export_tables_match_spyglass`, or from Spyglass when
        the selection was logged with a different Spyglass version.
    """
    from spyglass.common.common_usage import Export

    check_export_tables_match_spyglass()
    Export().populate_paper(paper_id=paper_id)
