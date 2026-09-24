r"""Rebuild the five Figure-4 input exports from Spyglass (CLI; read-only).

Fetches the Spyglass entries behind the Figure-4 recording and writes the five
files that :func:`statespacecheck_paper.load_local_data.load_neural_recording_from_files`
reads. With ``--compare-to``, checks the rebuilt files against a reference set
(e.g. the ``data/`` exports the figure used) by content. The recipe lives in
:mod:`statespacecheck_paper.spyglass_data`; this script is the thin CLI wrapper.

Requires Spyglass, lab database credentials, and the lab's analysis NWB store:
run on a lab server in an environment with the lab's Spyglass. Only reads from
the database. See ``docs/data-lineage.md``.

Example (``REF`` holds the exports the figure used)::

    PYTHONPATH=src python scripts/fetch_figure04_inputs.py \
        --output-dir /tmp/figure04_inputs --compare-to REF
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from statespacecheck_paper.spyglass_data import (
    FIGURE04_EPOCH_NAME,
    FIGURE04_NWB_FILE_NAME,
    check_output_paths,
    epoch_identifier,
    fetch_figure04_inputs,
    print_export_comparison,
    write_figure04_inputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Fetch, write, and optionally compare the Figure-4 input exports."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write the files.")
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Directory of reference exports to compare the rebuilt files against.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    parser.add_argument("--nwb-file-name", default=FIGURE04_NWB_FILE_NAME)
    parser.add_argument("--epoch-name", default=FIGURE04_EPOCH_NAME)
    args = parser.parse_args(argv)

    animal_date_epoch = epoch_identifier(args.nwb_file_name, args.epoch_name)
    try:
        check_output_paths(
            args.output_dir,
            animal_date_epoch,
            reference_dir=args.compare_to,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    inputs = fetch_figure04_inputs(args.nwb_file_name, args.epoch_name)
    for path in write_figure04_inputs(
        inputs, args.output_dir, animal_date_epoch, overwrite=args.overwrite
    ):
        print(f"wrote {path}")

    if args.compare_to is None:
        return 0
    return 0 if print_export_comparison(args.compare_to, args.output_dir, animal_date_epoch) else 1


if __name__ == "__main__":
    sys.exit(main())
