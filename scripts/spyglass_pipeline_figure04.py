r"""Set up and run the Figure-4 Spyglass pipeline one step at a time (CLI).

Steps, in order (see ``docs/spyglass-pipeline.md``)::

    position-group        PositionGroup of the epoch's v0 IntervalPositionInfo entry
    spike-sorting-output  register the v0 HPC sort in SpikeSortingOutput
    sorted-spikes-group   SortedSpikesGroup of the v0 HPC sort
    decoding-parameters   the two Figure-4 DecodingParameters
    decoding-selections   the two SortedSpikesDecodingSelection entries
    decode                populate SortedSpikesDecodingV1 for them
    diagnostics-schema    create the custom schema and its tables
    diagnostics-selection the Figure4DiagnosticsSelection entry
    diagnostics           populate Figure4Diagnostics
    check                 compare the stored summary with figure04_summary.json

Every step except ``check`` writes to the lab database, so each only prints what
it would write unless given ``--write`` (and then asks for confirmation unless
``--yes``). ``check`` only reads. The recipes live in
:mod:`statespacecheck_paper.spyglass_pipeline`; this script is the thin CLI wrapper.

Run on a lab server with the lab's current Spyglass and this repository's
``src/`` on the import path::

    PYTHONPATH=src python scripts/spyglass_pipeline_figure04.py --step position-group
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

STEPS = (
    "position-group",
    "spike-sorting-output",
    "sorted-spikes-group",
    "decoding-parameters",
    "decoding-selections",
    "decode",
    "diagnostics-schema",
    "diagnostics-selection",
    "diagnostics",
    "check",
)
SUMMARY_PATH = Path(__file__).resolve().parents[1] / "manuscript/figures/main/figure04_summary.json"


def _printable(entries: Any) -> Any:
    """Replace model objects with their class names for printing."""
    if isinstance(entries, list):
        return [_printable(entry) for entry in entries]
    if isinstance(entries, dict):
        return {
            key: type(value).__name__ if key == "decoding_params" else _printable(value)
            for key, value in entries.items()
        }
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    """Run one pipeline step (a dry run unless ``--write``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--write", action="store_true", help="Write to the lab database.")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH, help="For --step check.")
    args = parser.parse_args(argv)

    # Imports Spyglass, which connects to the lab database.
    from statespacecheck_paper import spyglass_pipeline as pipeline

    if args.step in ("diagnostics-selection", "diagnostics", "check"):
        pipeline.activate_schema()

    if args.step == "check":
        entry = pipeline.diagnostics_selection_entry()
        stored = (pipeline.Figure4Diagnostics() & entry).fetch_reported_statistics()
        committed = json.loads(args.summary.read_text())
        matches = {
            "n_units": stored["n_units"] == committed["dataset"]["n_units"],
            "diagnostic_means": stored["diagnostic_means"] == committed["diagnostic_means"],
            "flag_confusions": stored["flag_confusions"] == committed["flag_confusions"],
        }
        for name, is_equal in matches.items():
            print(f"{'identical' if is_equal else 'DIFFERENT'}  {name}")
        return 0 if all(matches.values()) else 1

    steps: dict[str, tuple[Callable[[], Any], Callable[[Any], None]]] = {
        "position-group": (pipeline.position_group_entry, pipeline.create_position_group),
        "spike-sorting-output": (
            pipeline.spike_sorting_output_entries,
            pipeline.register_spike_sorting_output,
        ),
        "sorted-spikes-group": (
            pipeline.sorted_spikes_group_entry,
            pipeline.create_sorted_spikes_group,
        ),
        "decoding-parameters": (
            pipeline.decoding_parameter_entries,
            pipeline.insert_decoding_parameters,
        ),
        "decoding-selections": (
            pipeline.decoding_selection_entries,
            pipeline.insert_decoding_selections,
        ),
        "decode": (pipeline.decoding_selection_entries, pipeline.populate_decoding),
        "diagnostics-selection": (
            pipeline.diagnostics_selection_entry,
            pipeline.insert_diagnostics_selection,
        ),
        "diagnostics": (pipeline.diagnostics_selection_entry, pipeline.populate_diagnostics),
    }
    if args.step == "diagnostics-schema":
        entries: Any = None
        print(f"Would create schema {pipeline.SCHEMA_NAME!r} and its tables.")
    else:
        describe, _ = steps[args.step]
        entries = describe()
        print(f"{args.step}:\n{json.dumps(_printable(entries), indent=1, default=str)}")
    if not args.write:
        print("Dry run: nothing written. Add --write to write this.")
        return 0
    if not args.yes and input("Write this to the lab database? [y/N] ").strip().lower() != "y":
        print("Aborted; nothing written.")
        return 1
    if args.step == "diagnostics-schema":
        pipeline.activate_schema(create=True)
    else:
        _, run = steps[args.step]
        run(entries)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
