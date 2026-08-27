"""Thin command-line entry point for generating Figure 1."""

from statespacecheck_paper.figure01_generation import generate_figure01


def main() -> None:
    """Generate the canonical Figure 1 artifacts."""
    generate_figure01()


if __name__ == "__main__":
    main()
