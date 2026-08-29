# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Property filter DSL (generalized from neo4j-graphrag-python filters.py).

Declarative ``PropertyFilter`` objects compile to one of two HugeGraph
execution targets:

  * ``compile_to_properties``  -> equality-only filters become the
    ``properties`` dict for ``getVertexByPage`` (HugeGraph REST does AND on
    the given key/value pairs; values travel as JSON, so they are naturally
    parameterized and injection-safe). Returns ``None`` when a filter is not
    equality (caller must use Gremlin).
  * ``compile_gremlin_has``    -> builds a ``.has(...)`` chain using standard
    Gremlin predicates (``P.neq/P.gt/...``) and HugeGraph's ``Text.regex``
    for LIKE. Values are serialized with :func:`escape_gremlin_literal` so
    user input is never able to break out of the query string (the
    pyhugegraph gremlin API does not support bindings).

Example::

    filters = [
        PropertyFilter("status", FilterOperator.EQ, "processed"),
        PropertyFilter("age", FilterOperator.GTE, 18),
    ]
    props = FilterCompiler.compile_to_properties(filters)      # None (GTE)
    fragment, _ = FilterCompiler.compile_gremlin_has(filters)  # .has('status','processed').has('age',P.gte(18))
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class FilterOperator(Enum):
    """Supported comparison operators (TinkerPop predicates + HugeGraph Text)."""

    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"
    LIKE = "like"


@dataclass
class PropertyFilter:
    """A single field/operator/value filter condition."""

    field: str
    operator: FilterOperator
    value: Any

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
        }


def escape_gremlin_literal(value: Any) -> str:
    """Serialize a value as a safe Gremlin literal.

    Strings are wrapped in single quotes with embedded quotes escaped;
    numbers/booleans are emitted as-is; ``None`` becomes ``null``. This is
    the injection barrier since the pyhugegraph gremlin API concatenates
    the query string (no bindings support).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    # string-ish: single-quoted with escaping
    text = str(value)
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


class FilterCompiler:
    """Compile :class:`PropertyFilter` lists to HugeGraph execution targets."""

    @staticmethod
    def compile_to_properties(
        filters: Sequence[PropertyFilter],
    ) -> Optional[Dict[str, Any]]:
        """Equality-only filters -> REST ``properties`` dict (AND semantics).

        Returns ``None`` if any filter is not ``EQ`` (the REST properties
        filter only supports equality); an empty dict for no filters.
        """
        props: Dict[str, Any] = {}
        for f in filters:
            if f.operator != FilterOperator.EQ:
                return None
            props[f.field] = f.value
        return props

    @staticmethod
    def compile_gremlin_has(
        filters: Sequence[PropertyFilter],
        label: Optional[str] = None,
    ) -> Tuple[str, List[str]]:
        """Build a ``.has(...)`` chain.

        Equality filters use the two-argument ``.has(key, value)`` form;
        other operators use Gremlin predicates ``.has(key, P.gt(...))`` and
        HugeGraph's ``Text.regex`` for LIKE. When ``label`` is given, the
        three-argument ``.has(label, key, ...)`` form is used (the first
        argument of a three-arg ``.has`` is a vertex/edge label, NOT a
        traverser alias).

        Returns ``(gremlin_fragment, operator_names)`` where the second
        element lists the operators used (for diagnostics).
        """
        prefix = f"'{label}', " if label else ""
        parts: List[str] = []
        operators: List[str] = []
        for f in filters:
            op = f.operator
            operators.append(op.value)
            field = escape_gremlin_literal(f.field)
            literal = escape_gremlin_literal(f.value)
            if op == FilterOperator.EQ:
                parts.append(f".has({prefix}{field}, {literal})")
            elif op == FilterOperator.NEQ:
                parts.append(f".has({prefix}{field}, P.neq({literal}))")
            elif op == FilterOperator.GT:
                parts.append(f".has({prefix}{field}, P.gt({literal}))")
            elif op == FilterOperator.GTE:
                parts.append(f".has({prefix}{field}, P.gte({literal}))")
            elif op == FilterOperator.LT:
                parts.append(f".has({prefix}{field}, P.lt({literal}))")
            elif op == FilterOperator.LTE:
                parts.append(f".has({prefix}{field}, P.lte({literal}))")
            elif op == FilterOperator.IN:
                parts.append(f".has({prefix}{field}, P.within({_csv(f.value)}))")
            elif op == FilterOperator.NIN:
                parts.append(f".has({prefix}{field}, P.without({_csv(f.value)}))")
            elif op == FilterOperator.LIKE:  # pragma: no branch - all operators are matched by earlier branches
                # HugeGraph Text.contains (single-arg; Text.regex does not exist)
                contains = escape_gremlin_literal(f.value)
                parts.append(f".has({prefix}{field}, Text.contains({contains}))")
        return "".join(parts), operators


def _csv(values: Any) -> str:
    """Serialize an IN/NIN value list as comma-separated Gremlin literals."""
    if not isinstance(values, (list, tuple)):
        values = [values]
    return ", ".join(escape_gremlin_literal(v) for v in values)
