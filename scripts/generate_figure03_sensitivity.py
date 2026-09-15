"""Run the Figure-3 sensitivity settings (CLI).

Reruns the full calibration-then-evaluation pipeline at 50% and 80% HPD
coverage, with residual ordinary activity and a lower sparse-cell rate in the
sparse regime, and with the approximate reflected continuous trajectory, and
writes ``manuscript/figures/supplementary/figure03_sensitivity_summary.json``.
The recipe lives in
:func:`statespacecheck_paper.figure03_sensitivity.run_figure03_sensitivity`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from statespacecheck_paper.figure03_sensitivity import run_figure03_sensitivity


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the realization counts and run every sensitivity setting."""
    parser = argparse.ArgumentParser(description="Run the Figure 3 sensitivity settings.")
    parser.add_argument("--n-realizations", type=int, default=None)
    parser.add_argument("--n-calibration-realizations", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    kwargs: dict[str, int] = {"n_jobs": args.n_jobs}
    if args.n_realizations is not None:
        kwargs["n_realizations"] = args.n_realizations
    if args.n_calibration_realizations is not None:
        kwargs["n_calibration_realizations"] = args.n_calibration_realizations
    run_figure03_sensitivity(**kwargs)


if __name__ == "__main__":
    main()
