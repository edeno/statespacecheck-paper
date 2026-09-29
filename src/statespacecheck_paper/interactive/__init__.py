"""Interactive decoder viewer (pyqtgraph desktop app).

``python -m statespacecheck_paper.interactive`` opens the viewer for a
given cache and runs the Qt event loop (see :func:`.app.main` and
:func:`.app.launch`). The package imports none of its submodules, so
running one as ``__main__`` (e.g.
``python -m statespacecheck_paper.interactive.cache``) does not pull in
the Qt stack. See docs/interactive.md for an end-to-end walkthrough.
"""
