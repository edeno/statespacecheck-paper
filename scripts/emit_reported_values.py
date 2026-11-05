"""Thin command-line entry point for emitting the manuscript's value macros.

Emitting runs offline: the cited ``statespacecheck`` DOI is read from the
committed ``manuscript/software_dois.json``. ``--refresh-dois`` first looks up
the recorded version's DOI on Zenodo (needs internet), adding it to that file if
new and failing if it disagrees with the recorded one.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from statespacecheck_paper.paths import (
    FIGURE03_SUMMARY_PATH,
    FIGURE04_SUMMARY_PATH,
    SOFTWARE_DOIS_PATH,
)
from statespacecheck_paper.reported_values import (
    load_summary,
    refresh_statespacecheck_doi,
    statespacecheck_version,
    write_macro_file,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Write ``manuscript/reported_values.tex`` from the figure summaries."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refresh-dois",
        action="store_true",
        help=(
            "Look up the cited statespacecheck version's DOI on Zenodo (needs internet) "
            f"and record or verify it in {SOFTWARE_DOIS_PATH.name} before emitting."
        ),
    )
    args = parser.parse_args(argv)
    if args.refresh_dois:
        version = statespacecheck_version(
            load_summary(FIGURE03_SUMMARY_PATH),
            load_summary(FIGURE04_SUMMARY_PATH),
        )
        doi = refresh_statespacecheck_doi(version)
        print(f"statespacecheck {version}: {doi} (recorded in {SOFTWARE_DOIS_PATH})")
    path = write_macro_file()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
