"""Tests for the read-only DataJoint guard.

The statement check runs everywhere. The guard's wiring needs DataJoint, which only
the lab's Spyglass environment installs, so CI runs that test in a separate step.
It replaces the send to the server and never connects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ._scripts import load_script

script = load_script("datajoint_read_only")


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT connection_id()",
        "  select `a` FROM `db`.`t`",
        "\nSHOW TABLES in `db`",
        "SHOW FULL COLUMNS FROM `t` IN `db`",
        "DESCRIBE `db`.`t`",
        "DESC `db`.`t`",
        "(SELECT `a` FROM `t`) UNION (SELECT `a` FROM `u`)",
        "START TRANSACTION WITH CONSISTENT SNAPSHOT",
        "COMMIT",
        "ROLLBACK",
    ],
)
def test_reads_are_allowed(statement: str) -> None:
    assert script.is_read_statement(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "DROP DATABASE `db`",
        "DROP TABLE `db`.`t`",
        "ALTER TABLE `db`.`t` ADD `a` int",
        "CREATE DATABASE `db`",
        "CREATE TABLE IF NOT EXISTS `db`.`t` (a int)",
        "INSERT INTO `db`.`t` VALUES (1)",
        "  insert into `db`.`t` VALUES (1)",
        "REPLACE INTO `db`.`t` VALUES (1)",
        "UPDATE `db`.`t` SET a=1",
        "DELETE FROM `db`.`t`",
        "TRUNCATE TABLE `db`.`t`",
        "RENAME TABLE `a` TO `b`",
        "GRANT ALL ON *.* TO user",
        "SET PASSWORD = PASSWORD('x')",
        "ALTER USER user() IDENTIFIED BY 'x'",
        "kill 12",
        "WITH d AS (SELECT 1) DELETE FROM `db`.`t`",
        "EXPLAIN ANALYZE DELETE FROM `db`.`a`, `db`.`b`",
        "/* note */ DROP DATABASE `db`",
        "SELECTED",
        "",
    ],
)
def test_everything_else_is_refused(statement: str) -> None:
    assert not script.is_read_statement(statement)


def test_patched_datajoint_refuses_a_write_before_sending_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dj = pytest.importorskip("datajoint")
    from datajoint.table import Table

    # The guard patches classes; these restore them after the test.
    monkeypatch.setattr(dj.Connection, "query", dj.Connection.query)
    monkeypatch.setattr(dj.Schema, "__init__", dj.Schema.__init__)
    monkeypatch.setattr(dj.Schema, "activate", dj.Schema.activate)
    for method in script._WRITE_METHODS:
        monkeypatch.setattr(Table, method, getattr(Table, method))

    sent: list[str] = []

    def send(cursor: Any, query: str, *args: Any) -> None:
        sent.append(query)

    monkeypatch.setattr(dj.Connection, "_execute_query", staticmethod(send))
    script.make_datajoint_read_only()

    connection = dj.Connection.__new__(dj.Connection)
    connection._query_cache = None
    connection._conn = SimpleNamespace(cursor=lambda cursor: None)

    connection.query("SELECT 1")
    with pytest.raises(RuntimeError, match="read-only run: refused SQL"):
        connection.query("DROP DATABASE `db`")
    with pytest.raises(RuntimeError, match="read-only run: refused Table.insert"):
        Table.insert(Table(), [])

    assert sent == ["SELECT 1"]
