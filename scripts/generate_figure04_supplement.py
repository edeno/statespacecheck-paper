"""Generate the Figure-4 supplement: broadening, coverage, rate, and behavior (CLI).

The supplement characterizes the Continuous-model flag removal on the full
recording (predictive HPD-region sizes, uniform-mixture counterfactuals, HPD
coverage sensitivity) and the firing-rate / behavior associations of the flags.
The scientific recipe lives in
:func:`statespacecheck_paper.figure04_supplement_generation.generate_figure04_supplement`;
this script is the thin CLI wrapper. It reads the canonical Figure-4 caches
(run ``scripts/generate_figure04.py`` first) and refits nothing.
"""

from __future__ import annotations

from statespacecheck_paper.figure04_supplement_generation import generate_figure04_supplement


def main() -> None:
    """Run the Figure-4 supplement recipe."""
    generate_figure04_supplement()


if __name__ == "__main__":
    main()
