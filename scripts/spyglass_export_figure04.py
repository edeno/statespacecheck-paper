r"""Capture the Figure-4 inputs in a Spyglass export (CLI; WRITES TO THE LAB DATABASE).

Runs the same fetches as ``scripts/fetch_figure04_inputs.py`` inside a Spyglass
export session, so the lab's export process records every table entry and
analysis NWB file the Figure-4 inputs came from:

1. ``ExportSelection.start_export(paper_id, analysis_id)`` inserts a selection
   entry; each Spyglass fetch is logged against it until ``stop_export``.
2. Optionally (``--output-dir``/``--compare-to``) the fetched inputs are written
   and compared with the exports the figure used, so the logged fetches are
   shown to reproduce them.
3. With ``--populate``, ``Export().populate_paper(paper_id)`` packages the export
   (``Export`` entries plus the SQL dump and file list in the Spyglass export
   directory). Without it, the logged selection can be previewed and packaged
   later.

Refuses a ``paper_id`` that already has export selections, since re-running an
existing selection or packaging a paper replaces its previous export.

Run on a lab server (the fetches read analysis NWB files) in an environment with
the lab's current Spyglass: ``--populate`` refuses a Spyglass older than the lab
database's export tables, such as the 0.5.5 in ``uv.lock``. See
``docs/data-lineage.md``.

Example (``REF`` holds the exports the figure used)::

    PYTHONPATH=src python scripts/spyglass_export_figure04.py \
        --paper-id <new-paper-id> --output-dir /tmp/figure04_export --compare-to REF
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from statespacecheck_paper.spyglass_data import (
    FIGURE04_EPOCH_NAME,
    FIGURE04_NWB_FILE_NAME,
    check_export_tables_match_spyglass,
    compare_figure04_exports,
    epoch_identifier,
    log_figure04_export,
    write_figure04_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Log (and optionally package) a Spyglass export of the Figure-4 inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper-id", required=True, help="New export paper ID (<= 32 chars).")
    parser.add_argument("--analysis-id", default="figure04_inputs", help="(<= 32 chars)")
    parser.add_argument("--output-dir", type=Path, help="Also write the fetched files here.")
    parser.add_argument("--compare-to", type=Path, help="Compare written files to these exports.")
    parser.add_argument(
        "--populate", action="store_true", help="Package the export with populate_paper."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)
    if args.compare_to is not None and args.output_dir is None:
        parser.error("--compare-to requires --output-dir")

    print(
        f"This writes to the lab database: ExportSelection entry paper_id={args.paper_id!r}, "
        f"analysis_id={args.analysis_id!r}"
        + (", then Export().populate_paper" if args.populate else "")
        + f". spyglass-neuro {version('spyglass-neuro')}."
    )
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted; nothing written.")
        return 1
    if args.populate:
        check_export_tables_match_spyglass()  # before any write

    inputs = log_figure04_export(args.paper_id, args.analysis_id)

    if args.output_dir is not None:
        animal_date_epoch = epoch_identifier(FIGURE04_NWB_FILE_NAME, FIGURE04_EPOCH_NAME)
        write_figure04_inputs(inputs, args.output_dir, animal_date_epoch)
        if args.compare_to is not None:
            matches = compare_figure04_exports(args.compare_to, args.output_dir, animal_date_epoch)
            for name, is_equal in matches.items():
                print(f"{'identical' if is_equal else 'DIFFERENT'}  {name}")
            if not all(matches.values()):
                print("Logged fetches do not reproduce the reference exports; not packaging.")
                return 1

    from spyglass.common.common_usage import Export, ExportSelection

    selection = ExportSelection()
    print("Logged tables:")
    for table in selection.preview_tables(paper_id=args.paper_id):
        print(f"  {table.full_table_name}: {len(table)} rows")
    print("Logged files:")
    for path in sorted(selection.list_file_paths({"paper_id": args.paper_id}, as_dict=False)):
        print(f"  {path}")

    if args.populate:
        Export().populate_paper(paper_id=args.paper_id)
        print(f"Packaged export for paper_id={args.paper_id!r}.")
    else:
        print("Not packaged. Rerun packaging with Export().populate_paper(paper_id=...).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
