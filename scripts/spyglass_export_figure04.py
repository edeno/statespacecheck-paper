r"""Capture the Figure-4 inputs in a Spyglass export (CLI; WRITES TO THE LAB DATABASE).

Runs the same fetches as ``scripts/fetch_figure04_inputs.py`` inside a Spyglass
export session, so the lab's export process records every table entry and
analysis NWB file the Figure-4 inputs came from:

1. Before anything is written: the output and reference paths are checked, you
   confirm, and the installed Spyglass is checked against the database's export
   tables.
2. ``ExportSelection.start_export(paper_id, analysis_id)`` inserts a selection
   entry; each Spyglass fetch is logged against it until ``stop_export``.
3. With ``--output-dir``/``--compare-to``, the fetched inputs are written and
   compared with the input file the figure used.
4. With ``--populate`` (which requires ``--compare-to``, and only if the
   comparison passed), ``Export().populate_paper`` packages the export: ``Export``
   entries in the database plus a ``mysqldump`` script in the Spyglass export
   directory. Packaging later needs the same Spyglass ``x.y.z``.

Refuses a ``paper_id`` that already has export selections. Run on a lab server
(the fetches read analysis NWB files) in an environment with the lab's current
Spyglass. See ``docs/data-lineage.md``.

Example (``REF`` holds the input file the figure used)::

    PYTHONPATH=src python scripts/spyglass_export_figure04.py \
        --paper-id <new-paper-id> --output-dir /tmp/figure04_export --compare-to REF --populate
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
    check_output_paths,
    describe_figure04_export,
    epoch_identifier,
    log_figure04_export,
    package_figure04_export,
    print_export_comparison,
    write_figure04_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Log (and optionally package) a Spyglass export of the Figure-4 inputs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paper-id", required=True, help="New export paper ID (<= 32 chars).")
    parser.add_argument("--analysis-id", default="figure04_inputs", help="(<= 32 chars)")
    parser.add_argument("--output-dir", type=Path, help="Write the fetched input file here.")
    parser.add_argument("--compare-to", type=Path, help="Compare the written file with this one.")
    parser.add_argument(
        "--populate", action="store_true", help="Package the export with populate_paper."
    )
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    args = parser.parse_args(argv)
    if args.compare_to is not None and args.output_dir is None:
        parser.error("--compare-to requires --output-dir")
    if args.populate and args.compare_to is None:
        parser.error("--populate requires --compare-to, so only verified fetches are packaged")
    animal_date_epoch = epoch_identifier(FIGURE04_NWB_FILE_NAME, FIGURE04_EPOCH_NAME)
    if args.output_dir is not None:
        try:
            check_output_paths(args.output_dir, animal_date_epoch, reference_dir=args.compare_to)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))

    populate_note = ", then Export().populate_paper" if args.populate else ""
    print(
        f"This writes to the lab database: ExportSelection entry paper_id={args.paper_id!r}, "
        f"analysis_id={args.analysis_id!r}{populate_note}. "
        f"spyglass-neuro {version('spyglass-neuro')}."
    )
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Aborted; nothing written.")
        return 1

    inputs = log_figure04_export(args.paper_id, args.analysis_id)
    leftover = f"The export selection for paper_id {args.paper_id!r} stays in the database"
    try:
        if args.output_dir is not None:
            write_figure04_inputs(inputs, args.output_dir, animal_date_epoch)
        if args.compare_to is not None and not print_export_comparison(
            args.compare_to, args.output_dir, animal_date_epoch
        ):
            print(
                f"Logged fetches do not reproduce the reference input file. {leftover}, unpackaged."
            )
            return 1
        print("\n".join(describe_figure04_export(args.paper_id)))
        if args.populate:
            package_figure04_export(args.paper_id)
            print(f"Packaged the export for paper_id={args.paper_id!r}.")
        else:
            print("Not packaged. Package it with the same Spyglass x.y.z via populate_paper.")
    except Exception:
        print(f"Failed after logging. {leftover}.", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
