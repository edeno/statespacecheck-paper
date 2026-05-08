"""Interactive decoder viewer (pyqtgraph desktop app).

The :func:`launch` helper opens the viewer for a given cache and runs
the Qt event loop; ``python -m statespacecheck_paper.interactive`` is
the equivalent CLI entry point (delegates to :mod:`.app`). See the
README for an end-to-end walkthrough.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time-only
    from .app import configure_qt_application, launch, main

__all__ = ["configure_qt_application", "launch", "main"]


def __getattr__(name: str) -> Any:
    """Deferred re-exports of ``app.{configure_qt_application, launch, main}``.

    Eagerly importing those at module-load time pulls the full Qt /
    pyqtgraph stack (via ``app`` → ``data_source`` → ``cache``), which
    causes a ``RuntimeWarning`` when ``runpy`` later tries to execute
    one of the submodules as ``__main__`` (e.g.
    ``python -m statespacecheck_paper.interactive.cache``) — the
    submodule is already in ``sys.modules`` from the transitive import
    chain. Defer via PEP 562 so plain submodule execution stays warning
    -free while ``from statespacecheck_paper.interactive import launch``
    keeps working.
    """
    if name in __all__:
        from . import app  # noqa: PLC0415

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
