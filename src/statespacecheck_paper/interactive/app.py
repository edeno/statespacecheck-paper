"""Application entry point and Qt setup for the interactive decoder viewer.

Owns the small amount of plumbing that runs once at process start:
choosing / configuring the ``QApplication``, instantiating
``DecoderDataSource`` and ``DecoderViewer``, and the
``argparse`` CLI for ``python -m statespacecheck_paper.interactive``.
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


def launch(
    cache_dir: Path | str,
    model: ModelName | None = None,
    *,
    simulation: bool = False,
) -> int:
    """Open the viewer for the cache at ``cache_dir`` and run the event loop.

    Pass ``simulation=True`` to open the figure-3 simulation cache;
    otherwise pass ``model=`` to choose the real-data model. Exactly
    one of those must be specified.
    """
    # Deferred to break the import cycle with ``viewer``: this module
    # is imported by ``viewer.py``'s re-export footer, so a top-level
    # ``from .viewer import DecoderViewer`` here would loop back.
    from .viewer import DecoderViewer  # noqa: PLC0415

    if simulation and model is not None:
        raise ValueError("launch(): pass simulation=True OR model=, not both.")
    if not simulation and model is None:
        raise ValueError("launch(): pass model= for real-data or simulation=True.")

    existing = QtWidgets.QApplication.instance()
    app: QtWidgets.QApplication = (
        existing
        if isinstance(existing, QtWidgets.QApplication)
        else QtWidgets.QApplication(sys.argv)
    )
    configure_qt_application(app)
    if simulation:
        ds = DecoderDataSource.for_simulation(cache_dir)
    else:
        assert model is not None  # narrowed above
        ds = DecoderDataSource.for_model(cache_dir, model)
    viewer = DecoderViewer(ds, cache_dir=cache_dir)
    viewer.show()
    try:
        return int(app.exec())
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="statespacecheck_paper.interactive")
    parser.add_argument("--cache-dir", required=True, help="Cache directory.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--model",
        choices=("continuous", "contfrag"),
        default=None,
        help="Open the real-data cache for this model.",
    )
    target.add_argument(
        "--simulation",
        action="store_true",
        help="Open the figure-3 simulation cache (built via ``cache build-simulated``).",
    )
    args = parser.parse_args(argv)
    if args.simulation:
        return launch(args.cache_dir, simulation=True)
    # Default to ``continuous`` when the user hasn't specified either
    # flag (preserves the legacy CLI behaviour).
    model: ModelName = args.model if args.model is not None else "continuous"
    return launch(args.cache_dir, model=model)


if __name__ == "__main__":
    raise SystemExit(main())
