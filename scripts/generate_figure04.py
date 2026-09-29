"""Generate Figure 4: real hippocampal-data decoder diagnostics (CLI).

Figure 4 shows per-spike-event diagnostic metrics for the Continuous and
Continuous-Fragmented decoders on real neural recording data. The scientific
recipe lives in
:func:`statespacecheck_paper.figure04_generation.generate_figure04`; this script
is the thin CLI wrapper.

Requires:
- non_local_detector package for decoder models
- Pre-exported neural recording data under ``data/``
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from statespacecheck_paper.figure04_generation import generate_figure04


def main(argv: Sequence[str] | None = None) -> None:
    """Parse ``--force-recompute`` and run the Figure-4 generation recipe."""
    parser = argparse.ArgumentParser(description="Generate Figure 4.")
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help=(
            "Re-fit and re-decode both models and recompute their per-spike "
            "diagnostics instead of loading the decode and diagnostics caches "
            "under the data directory's intermediates/ (overwrites both caches)."
        ),
    )
    args = parser.parse_args(argv)
    generate_figure04(use_cache=not args.force_recompute)


if __name__ == "__main__":
    main()
