# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the synthetic warehouse and executor."""

import pytest

from hugegraph_llm.semantic_layer.execution.executor import (
    ExecutionResult,
    SqliteExecutor,
)
from hugegraph_llm.semantic_layer.execution.synthetic import (
    build_sqlite_warehouse,
)
from hugegraph_llm.semantic_layer.readers import (
    ColumnRow,
    SemanticProjection,
    TableRow,
)


def _proj():
    """users 1--N orders 1--N order_items; plus a standalone table."""
    proj = SemanticProjection()
    for name, comment in (
        ("users", "用户表"),
        ("orders", "订单主表"),
        ("order_items", "订单明细表"),
        ("regions", "区域表"),
    ):
        proj.tables[name] = TableRow(name=name, comment=comment)
    for col in (
        ColumnRow(name="id", table="users", data_type="BIGINT", is_primary_key=True),
        ColumnRow(name="name", table="users", data_type="STRING"),
        ColumnRow(name="email", table="users", data_type="STRING"),
        ColumnRow(name="id", table="orders", data_type="BIGINT", is_primary_key=True),
        ColumnRow(name="user_id", table="orders", data_type="BIGINT"),
        ColumnRow(name="amount", table="orders", data_type="DECIMAL"),
        ColumnRow(name="created_at", table="orders", data_type="DATE"),
        ColumnRow(name="order_id", table="order_items", data_type="BIGINT"),
        ColumnRow(name="quantity", table="order_items", data_type="INT"),
        ColumnRow(name="code", table="regions", data_type="STRING", is_primary_key=True),
    ):
        proj.columns[col.qualified] = col
    proj.references["orders.user_id"] = ["users.id"]
    proj.reference_proven[("orders.user_id", "users.id")] = True
    proj.references["order_items.order_id"] = ["orders.id"]
    proj.reference_proven[("order_items.order_id", "orders.id")] = True
    return proj


@pytest.fixture
def executor():
    wh = build_sqlite_warehouse(_proj(), ":memory:")
    ex = SqliteExecutor(":memory:")
    # Swap the read-only connection for the in-memory writable one.
    ex._conn = wh.conn
    return ex, wh


def test_warehouse_contains_all_tables(executor):
    ex, wh = executor
    assert set(ex.table_names()) == {"users", "orders", "order_items", "regions"}
    assert wh.rows_total == sum(wh.tables.values())


def test_generated_rows_respect_types(executor):
    ex, _wh = executor
    res = ex.execute('SELECT id, email FROM "users" LIMIT 3')
    assert res.ok
    for row in res.rows:
        assert str(row[0]).startswith("USE")
        assert "@" in row[1]


def test_foreign_keys_point_at_real_parents(executor):
    """The core guarantee: every FK value exists in the parent table."""
    ex, _wh = executor
    orphans = ex.execute(
        'SELECT COUNT(*) FROM "orders" o '
        'LEFT JOIN "users" u ON o.user_id = u.id WHERE u.id IS NULL'
    )
    assert orphans.rows[0][0] == 0
    orphans2 = ex.execute(
        'SELECT COUNT(*) FROM "order_items" oi '
        'LEFT JOIN "orders" o ON oi.order_id = o.id WHERE o.id IS NULL'
    )
    assert orphans2.rows[0][0] == 0


def test_fact_tables_have_more_rows_than_dimensions(executor):
    ex, wh = executor
    assert wh.tables["orders"] > wh.tables["users"]


def test_gold_style_join_is_executable(executor):
    """A typical gold SQL against this model must just run."""
    ex, _wh = executor
    res = ex.execute(
        'SELECT u.name, SUM(o.amount) AS total FROM orders o '
        'JOIN users u ON o.user_id = u.id GROUP BY u.name'
    )
    assert res.ok, res.error
    assert res.columns == ["name", "total"]


def test_deterministic_generation():
    """Same seed -> same rows, so evaluation results are reproducible."""
    wh1 = build_sqlite_warehouse(_proj(), ":memory:", seed=7)
    wh2 = build_sqlite_warehouse(_proj(), ":memory:", seed=7)
    rows1 = wh1.conn.execute("SELECT * FROM orders ORDER BY id").fetchall()
    rows2 = wh2.conn.execute("SELECT * FROM orders ORDER BY id").fetchall()
    assert rows1 == rows2


def test_persisted_warehouse_reopens_readonly(tmp_path):
    path = str(tmp_path / "wh.db")
    build_sqlite_warehouse(_proj(), path)
    ex = SqliteExecutor(path)
    res = ex.execute("SELECT COUNT(*) FROM users")
    assert res.ok
    # Read-only: a write attempt must come back as an error, never succeed.
    write = ex.execute("INSERT INTO users VALUES (999, 'x', 'x@y.z')")
    assert not write.ok
    assert "readonly" in (write.error or "").lower()


# -- ExecutionResult --------------------------------------------------------


def test_error_result_is_not_ok():
    result = ExecutionResult(error="no such table")
    assert not result.ok
    assert result.to_dict()["error"] == "no such table"


def test_results_equal_ignores_row_order():
    a = ExecutionResult(columns=["t", "n"], rows=[["a", 1], ["b", 2]])
    b = ExecutionResult(columns=["t", "n"], rows=[["b", 2], ["a", 1]])
    assert a.results_equal(b)


def test_results_equal_checks_columns():
    a = ExecutionResult(columns=["t", "n"], rows=[["a", 1]])
    b = ExecutionResult(columns=["n", "t"], rows=[[1, "a"]])
    assert not a.results_equal(b)


def test_results_equal_rounds_floats():
    a = ExecutionResult(columns=["s"], rows=[[0.1 + 0.2]])
    b = ExecutionResult(columns=["s"], rows=[[0.3]])
    assert a.results_equal(b)


def test_error_results_never_equal():
    a = ExecutionResult(error="x")
    b = ExecutionResult(error="x")
    assert not a.results_equal(b)
