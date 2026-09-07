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

"""Generate a SQLite warehouse from a semantic projection.

Deterministic (fixed seed) so evaluation runs are reproducible: the same
projection always yields the same rows, so a gold SQL's expected result is
stable across runs and machines.

Value generation is column-name driven with domain hints from comments:

* ``id`` / ``*_id``  -> namespaced identifiers (``EMP001``, ``DEPT003``);
  parents generate first so foreign keys draw from real parent ids
* ``email``          -> fake but valid addresses
* ``amount/price/salary/revenue`` + NUMERIC/DECIMAL -> money amounts
* ``score/rate/count/quantity`` -> small numbers
* ``*_time/*_date/at`` -> dates in 2024
* ``status/type/level`` -> small integer enums (order status codes live in
  comments in real warehouses; here the enum is implicit)

Tables referenced by others (parents) get more rows than leaf tables --
``users`` must have at least as many distinct ids as ``orders`` needs.
"""

import random
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hugegraph_llm.semantic_layer.readers import SemanticProjection
from hugegraph_llm.utils.log import log

__all__ = ["SyntheticWarehouse", "build_sqlite_warehouse"]

_MONEY_RE = re.compile(r"amount|price|salary|revenue|budget|cost|total", re.I)
_SCORE_RE = re.compile(r"score|rate|health", re.I)
_COUNT_RE = re.compile(r"count|quantity|headcount|capacity", re.I)
_TIME_RE = re.compile(r"(_at|_time|_date)$|(created|updated|paid|due|resolved)", re.I)
_EMAIL_RE = re.compile(r"email", re.I)


@dataclass
class SyntheticWarehouse:
    """The generated warehouse plus where it lives."""

    path: str
    tables: Dict[str, int] = field(default_factory=dict)
    rows_total: int = 0

    def summary(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "tables": dict(sorted(self.tables.items())),
            "rows_total": self.rows_total,
        }


def build_sqlite_warehouse(
    projection: SemanticProjection,
    path: str = ":memory:",
    *,
    seed: int = 42,
    fact_rows: int = 120,
    dim_rows: int = 25,
) -> SyntheticWarehouse:
    """Create and populate a SQLite warehouse from ``projection``.

    :param path: ``:memory:`` for tests, a file path to persist.
    :param seed: fixed for reproducibility.
    :param fact_rows / dim_rows: row counts for tables that have outgoing
        foreign keys (facts) vs referenced-only tables (dimensions).
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(path if path != ":memory:" else ":memory:")

    parents, _referenced_by = _fk_topology(projection)
    result = SyntheticWarehouse(path=path)

    # Topological order over FK dependencies: a table is built only after
    # every table it references exists and has contributed its id pool.
    # (A simple "referenced first" flag is not enough -- `orders` and
    # `users` are BOTH referenced, and alphabetical order put `orders`
    # first, before its own parent's pool existed.)
    ordered = _topological_order(projection, parents)
    id_pools: Dict[str, List[object]] = {}

    for table in ordered:
        columns = projection.columns_of(table)
        if not columns:
            continue
        n = fact_rows if table in parents else dim_rows
        pk = _primary_key(projection, table)
        column_defs = ", ".join(
            f'"{c.name}" {_sqlite_type(c.data_type)}' for c in columns
        )
        conn.execute(f'CREATE TABLE "{table}" ({column_defs})')

        pool: List[object] = []
        for i in range(n):
            row = []
            for col in columns:
                value = _value_for(
                    col.name, col.data_type, table, i, pool, id_pools, rng
                )
                row.append(value)
                if col.name == pk:
                    pool.append(value)
            conn.execute(
                f'INSERT INTO "{table}" VALUES ({",".join("?" * len(row))})',
                row,
            )
        if pk:
            id_pools[table] = pool
        result.tables[table] = n
        result.rows_total += n

    conn.commit()
    if path != ":memory:":
        conn.close()
    log.info(
        "synthetic warehouse: %s tables / %s rows -> %s",
        len(result.tables), result.rows_total, path,
    )
    if path == ":memory:":
        result.conn = conn  # type: ignore[attr-defined]
    return result


def _fk_topology(
    projection: SemanticProjection,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Parents (referenced tables) and children (referencing tables)."""
    parents: Dict[str, List[str]] = {}
    children: Dict[str, List[str]] = {}
    for source, targets in projection.references.items():
        child = source.split(".", 1)[0]
        for target in targets:
            parent = target.split(".", 1)[0]
            parents.setdefault(child, []).append(parent)
            children.setdefault(parent, []).append(child)
    return parents, children


def _topological_order(
    projection: SemanticProjection, parents: Dict[str, List[str]]
) -> List[str]:
    """Kahn's algorithm over FK dependencies, alphabetical among ready ones.

    Cycles (self-references like ``employees.manager_id``) cannot be
    satisfied; those tables are appended at the end and their FK values
    fall back to generated ids.
    """
    import collections

    tables = set(projection.tables)
    indegree: Dict[str, int] = {t: 0 for t in tables}
    dependents: Dict[str, List[str]] = collections.defaultdict(list)
    for child, ps in parents.items():
        if child not in tables:
            continue
        for p in ps:
            if p in tables:
                indegree[child] += 1
                dependents[p].append(child)

    queue = sorted(t for t, d in indegree.items() if d == 0)
    order: List[str] = []
    while queue:
        table = queue.pop(0)
        order.append(table)
        for dep in sorted(dependents[table]):
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
    # Cycle members and tables missing endpoints go last, deterministically.
    order += sorted(tables - set(order))
    return order


def _find_parent_pool(
    column: str, id_pools: Dict[str, List[object]]
) -> Optional[List[object]]:
    """Match a FK column to its parent table's id pool.

    ``user_id`` rarely maps to a table literally called ``user`` -- the real
    name is usually plural (``users``), sometimes prefixed
    (``dim_user``). Exact match first, then stem-equality, then prefix/suffix
    containment; ambiguity falls back to the first stem match so generation
    stays deterministic.
    """
    base = column[: -len("_id")]
    if base in id_pools:
        return id_pools[base]
    stem = base.rstrip("s")
    for name in id_pools:
        if name.rstrip("s") == stem:
            return id_pools[name]
    for name in sorted(id_pools):
        if name.startswith(base) or base.startswith(name):
            return id_pools[name]
    return None


def _primary_key(projection: SemanticProjection, table: str) -> Optional[str]:
    """The column that identifies rows of ``table`` -- pool key only.

    The fallback is deliberately narrow: ``id`` or a table-specific
    ``{table}_id``. A generic ``*_id`` match would promote a foreign key
    (``order_items.order_id``) to "primary key" and pollute the id pool
    with values that exist only as references.
    """
    for col in projection.columns_of(table):
        if col.is_primary_key:
            return col.name
    for col in projection.columns_of(table):
        if col.name == "id" or col.name == f"{table}_id":
            return col.name
    return None


def _sqlite_type(data_type: str) -> str:
    """Map catalogue types onto SQLite storage classes.

    The projection carries warehouse types (``BIGINT``, ``DECIMAL``,
    ``STRING``...). SQLite only has INTEGER/REAL/TEXT/BLOB and applies type
    affinity, so a coarse mapping is correct -- TEXT for unknown types keeps
    any value storable.
    """
    t = (data_type or "").upper()
    if "INT" in t:
        return "INTEGER"
    if any(k in t for k in ("NUMERIC", "DECIMAL", "FLOAT", "DOUBLE", "REAL")):
        return "REAL"
    return "TEXT"


def _value_for(
    column: str,
    data_type: str,
    table: str,
    index: int,
    pool: List[object],
    id_pools: Dict[str, List[object]],
    rng: random.Random,
) -> object:
    t = (data_type or "").upper()

    # Foreign keys: draw from the referenced table's id pool.
    if column.endswith("_id") and column != "id":
        parent_pool = _find_parent_pool(column, id_pools)
        if parent_pool:
            return rng.choice(parent_pool)
        # No pool (parent not in projection): still an id-shaped value.
        base = column[: -len("_id")]
        return f"{base.upper()[:3]}{index + 1:04d}"

    if column == "id" or column.endswith("_id"):
        prefix = table.replace("_", "")[:3].upper()
        return f"{prefix}{index + 1:04d}"

    if _EMAIL_RE.search(column):
        return f"user{index + 1}@example.com"

    if _TIME_RE.search(column):
        day = rng.randint(1, 28)
        month = rng.randint(1, 12)
        date = f"2024-{month:02d}-{day:02d}"
        return f"{date} 10:{index % 60:02d}:00" if "TIMESTAMP" in t else date

    if "INT" in t:
        if _COUNT_RE.search(column) or "status" in column or "level" in column:
            return rng.randint(1, 5)
        return rng.randint(0, 999)

    if "REAL" in t or "DECIMAL" in t or "NUMERIC" in t:
        if _SCORE_RE.search(column):
            return round(rng.uniform(1.0, 5.0), 2)
        if _MONEY_RE.search(column):
            return round(rng.uniform(10.0, 5000.0), 2)
        return round(rng.uniform(0.0, 100.0), 2)

    if _SCORE_RE.search(column):
        return round(rng.uniform(1.0, 5.0), 2)

    # Plain string: enum-ish for status/type/level, else a labelled name.
    if re.search(r"status|type|level|region|category", column, re.I):
        return f"{column[:4].upper()}{rng.randint(1, 4)}"
    return f"{table}_{column}_{index + 1}"
