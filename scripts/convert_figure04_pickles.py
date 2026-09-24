"""Convert the legacy Figure-4 pickles to the ``.npz`` input file (CLI).

The recording was first exported as five pickles; Figure 4 now reads one
``{animal_date_epoch}_figure04_inputs.npz``. This writes that file next to (or
away from) the pickles and checks that it loads back identical to them. The
recipe is :func:`statespacecheck_paper.load_local_data.convert_legacy_pickle_exports`.

Example::

    uv run python scripts/convert_figure04_pickles.py --pickle-dir data --output-dir data
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from statespacecheck_paper.load_local_data import convert_legacy_pickle_exports
from statespacecheck_paper.paths import ANIMAL_DATE_EPOCH


def main(argv: Sequence[str] | None = None) -> None:
    """Convert and verify the legacy pickles for one epoch."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pickle-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--animal-date-epoch", default=ANIMAL_DATE_EPOCH)
    args = parser.parse_args(argv)
    path = convert_legacy_pickle_exports(args.pickle_dir, args.output_dir, args.animal_date_epoch)
    print(f"wrote {path} (verified identical to the pickles)")


if __name__ == "__main__":
    main()
