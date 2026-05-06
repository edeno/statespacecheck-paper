"""Interactive Figure 4 viewer (pyqtgraph desktop app).

See ``docs/figure04_interactive_viewer_plan.md`` for design.

The :func:`launch` helper opens the viewer for a given cache and runs
the Qt event loop; ``python -m statespacecheck_paper.interactive`` is
the equivalent CLI entry point (delegates to :mod:`.app`).
"""

from .app import configure_qt_application, launch, main

__all__ = ["configure_qt_application", "launch", "main"]
