"""Application entry point and Qt setup for the interactive decoder viewer.

Owns the small amount of plumbing that runs once at process start:
choosing / configuring the ``QApplication``, instantiating
``DecoderDataSource`` and ``DecoderViewer``, and the
``argparse`` CLI for ``python -m statespacecheck_paper.interactive.viewer``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from .data_source import DecoderDataSource, ModelName


def configure_qt_application(app: QtWidgets.QApplication) -> None:
    """Set viewer-wide Qt defaults (font, pyqtgraph options).

    Setting an explicit application font sidesteps Qt's
    ``Populating font family aliases`` lookup, which can take ~250 ms
    at first show because it expands the missing ``"Sans Serif"``
    alias against the system font set.
    """
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    font = QtGui.QFont("Helvetica", 9)
    if not font.exactMatch():
        font = QtGui.QFont("Arial", 9)
    if not font.exactMatch():
        font = app.font()
    app.setFont(font)


def launch(cache_dir: Path | str, model: ModelName) -> int:
    """Open the viewer for ``model`` from ``cache_dir`` and run the event loop."""
    # Deferred to break the import cycle with ``viewer``: this module
    # is imported by ``viewer.py``'s re-export footer, so a top-level
    # ``from .viewer import DecoderViewer`` here would loop back.
    from .viewer import DecoderViewer  # noqa: PLC0415

    existing = QtWidgets.QApplication.instance()
    app: QtWidgets.QApplication = (
        existing
        if isinstance(existing, QtWidgets.QApplication)
        else QtWidgets.QApplication(sys.argv)
    )
    configure_qt_application(app)
    ds = DecoderDataSource(cache_dir, model)
    viewer = DecoderViewer(ds, cache_dir=cache_dir)
    viewer.show()
    try:
        return int(app.exec())
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="statespacecheck_paper.interactive.viewer")
    parser.add_argument("--cache-dir", required=True, help="Cache directory.")
    parser.add_argument(
        "--model",
        choices=("continuous", "contfrag"),
        default="continuous",
        help="Which model's cache to open.",
    )
    args = parser.parse_args(argv)
    return launch(args.cache_dir, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
