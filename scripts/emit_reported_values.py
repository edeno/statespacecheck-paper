"""Thin command-line entry point for emitting the manuscript's value macros."""

from statespacecheck_paper.reported_values import write_macro_file


def main() -> None:
    """Write ``manuscript/reported_values.tex`` from the figure summaries."""
    path = write_macro_file()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
