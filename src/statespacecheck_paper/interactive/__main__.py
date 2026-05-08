"""``python -m statespacecheck_paper.interactive`` entry point.

Delegates to :func:`statespacecheck_paper.interactive.app.main`.
"""

from .app import main

raise SystemExit(main())
