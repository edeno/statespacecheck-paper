"""Download the Figure-4 input file from Zenodo and verify its SHA-256 (CLI).

The file (75 MB) is version 1.0 of Zenodo record 10.5281/zenodo.23020757; the
recipe lives in :func:`statespacecheck_paper.load_local_data.download_figure04_inputs`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from statespacecheck_paper.load_local_data import download_figure04_inputs
from statespacecheck_paper.paths import DATA_PATH


def main(argv: Sequence[str] | None = None) -> None:
    """Download the file into the data directory (default ``DATA_PATH``)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-path", type=Path, default=DATA_PATH, help="Directory to save the file in."
    )
    args = parser.parse_args(argv)
    print(f"Verified {download_figure04_inputs(args.data_path)}")


if __name__ == "__main__":
    main()
