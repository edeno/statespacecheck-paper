"""Run a Python script with DataJoint's database writes refused.

For checking Spyglass code against the shared lab database without changing it.
Every statement DataJoint sends passes through ``Connection.query``, which here
refuses anything but a read (``SELECT``, ``SHOW``, ``DESCRIBE``) or transaction
control before it reaches the server. That covers ``Schema.drop``, ``Table.alter``,
``Table.declare``, and SQL given to ``conn.query`` directly.

Schemas and tables are never created (``create_schema`` / ``create_tables`` are
forced off, so importing a module that defines tables the database lacks raises
instead of declaring them), and ``insert``, ``insert1``, ``delete``,
``delete_quick``, ``drop``, ``drop_quick``, and ``update1`` raise by name. That
includes the default rows DataJoint inserts into ``Lookup`` tables on import.
Reads work as usual. A cursor taken from the raw client connection, a stored
function called from a ``SELECT``, and files outside the database (e.g. analysis
NWB files) are not guarded.

Usage::

    PYTHONPATH=src python scripts/datajoint_read_only.py \
        scripts/spyglass_pipeline_figure04.py --step position-group
"""

from __future__ import annotations

import re
import runpy
import sys
from collections.abc import Callable
from typing import Any

# DataJoint's union queries open with a parenthesis. EXPLAIN is left out because
# EXPLAIN ANALYZE runs the statement it explains.
_READ_STATEMENT = re.compile(
    r"\s*\(*\s*(SELECT|SHOW|DESCRIBE|DESC|START\s+TRANSACTION|COMMIT|ROLLBACK)\b",
    re.IGNORECASE,
)

_WRITE_METHODS = (
    "insert",
    "insert1",
    "delete",
    "delete_quick",
    "drop",
    "drop_quick",
    "update1",
)


def _refuse(method: str) -> Callable[..., Any]:
    def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"read-only run: refused {type(self).__name__}.{method}")

    return refuse


def is_read_statement(statement: str) -> bool:
    """Return whether a SQL statement is a read or transaction control."""
    return _READ_STATEMENT.match(statement) is not None


def make_datajoint_read_only() -> None:
    """Patch DataJoint in this process so that database writes raise."""
    import datajoint as dj
    from datajoint.table import Table

    connection_query = dj.Connection.query
    schema_init, schema_activate = dj.Schema.__init__, dj.Schema.activate

    def query(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        if not is_read_statement(query):
            raise RuntimeError(f"read-only run: refused SQL {query[:80]!r}")
        return connection_query(self, query, *args, **kwargs)

    def init(self: Any, *args: Any, create_schema: bool = True, **kwargs: Any) -> None:
        kwargs.pop("create_tables", None)
        schema_init(self, *args, create_schema=False, create_tables=False, **kwargs)

    def activate(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("create_schema", None)
        kwargs.pop("create_tables", None)
        return schema_activate(self, *args, create_schema=False, create_tables=False, **kwargs)

    dj.Connection.query = query
    dj.Schema.__init__ = init
    dj.Schema.activate = activate
    for method in _WRITE_METHODS:
        setattr(Table, method, _refuse(method))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    make_datajoint_read_only()
    script, sys.argv = sys.argv[1], sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
