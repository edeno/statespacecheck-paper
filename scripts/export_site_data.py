"""Thin command-line entry point for regenerating the website's data files."""

import argparse

from statespacecheck_paper.site_export import export_site_data


def main() -> None:
    """Write ``site/data/*.json`` and the JavaScript parity fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-recording",
        action="store_true",
        help="Leave replay.json untouched (for machines without the Figure-4 data).",
    )
    args = parser.parse_args()
    for path in export_site_data(include_recording=not args.skip_recording):
        print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
