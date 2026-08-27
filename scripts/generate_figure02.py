"""Thin command-line entry point for generating Figure 2."""

from statespacecheck_paper.figure02_generation import generate_figure02


def main() -> None:
    """Generate the canonical Figure 2 artifacts."""
    generate_figure02()


if __name__ == "__main__":
    main()
