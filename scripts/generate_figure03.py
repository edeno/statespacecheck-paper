"""Thin command-line entry point for generating Figure 3."""

from statespacecheck_paper.figure03_generation import generate_figure03


def main() -> None:
    """Generate the canonical Figure 3 artifacts."""
    generate_figure03()


if __name__ == "__main__":
    main()
