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

import unittest

import pytest

from hugegraph_llm.operators.graph_op.kg_filters import (
    FilterCompiler,
    FilterOperator,
    PropertyFilter,
    escape_gremlin_literal,
)

pytestmark = [pytest.mark.unit]


class TestEscapeGremlinLiteral(unittest.TestCase):
    def test_none(self):
        self.assertEqual(escape_gremlin_literal(None), "null")

    def test_bool(self):
        self.assertEqual(escape_gremlin_literal(True), "true")
        self.assertEqual(escape_gremlin_literal(False), "false")

    def test_numbers(self):
        self.assertEqual(escape_gremlin_literal(10), "10")
        self.assertEqual(escape_gremlin_literal(1.5), "1.5")

    def test_plain_string(self):
        self.assertEqual(escape_gremlin_literal("abc"), "'abc'")

    def test_string_with_quote_escaped(self):
        # injection barrier: single quote cannot break out of the literal
        self.assertEqual(escape_gremlin_literal("a'b"), "'a\\'b'")

    def test_string_with_backslash_escaped(self):
        self.assertEqual(escape_gremlin_literal("a\\b"), "'a\\\\b'")


class TestPropertyFilter(unittest.TestCase):
    def test_to_dict(self):
        f = PropertyFilter("status", FilterOperator.EQ, "ok")
        self.assertEqual(
            f.to_dict(), {"field": "status", "operator": "eq", "value": "ok"}
        )


class TestFilterCompilerProperties(unittest.TestCase):
    def test_all_eq_compiles_to_dict(self):
        filters = [
            PropertyFilter("status", FilterOperator.EQ, "processed"),
            PropertyFilter("type", FilterOperator.EQ, "order"),
        ]
        self.assertEqual(
            FilterCompiler.compile_to_properties(filters),
            {"status": "processed", "type": "order"},
        )

    def test_empty_filters_returns_empty_dict(self):
        self.assertEqual(FilterCompiler.compile_to_properties([]), {})

    def test_non_eq_returns_none(self):
        filters = [PropertyFilter("age", FilterOperator.GTE, 18)]
        self.assertIsNone(FilterCompiler.compile_to_properties(filters))


class TestFilterCompilerGremlin(unittest.TestCase):
    def test_eq(self):
        fragment, ops = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("status", FilterOperator.EQ, "ok")]
        )
        self.assertEqual(fragment, ".has('status', 'ok')")
        self.assertEqual(ops, ["eq"])

    def test_all_operators(self):
        cases = [
            (FilterOperator.NEQ, "P.neq"),
            (FilterOperator.GT, "P.gt"),
            (FilterOperator.GTE, "P.gte"),
            (FilterOperator.LT, "P.lt"),
            (FilterOperator.LTE, "P.lte"),
        ]
        for op, pred in cases:
            with self.subTest(op=op):
                fragment, _ = FilterCompiler.compile_gremlin_has(
                    [PropertyFilter("age", op, 18)]
                )
                self.assertIn(f".has('age', {pred}(18))", fragment)

    def test_in_and_nin(self):
        fragment, _ = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("cat", FilterOperator.IN, ["a", "b"])]
        )
        self.assertEqual(fragment, ".has('cat', P.within('a', 'b'))")
        fragment, _ = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("cat", FilterOperator.NIN, ["a"])]
        )
        self.assertEqual(fragment, ".has('cat', P.without('a'))")

    def test_in_with_single_scalar(self):
        fragment, _ = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("cat", FilterOperator.IN, "only")]
        )
        self.assertEqual(fragment, ".has('cat', P.within('only'))")

    def test_like_uses_text_contains(self):
        fragment, ops = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("name", FilterOperator.LIKE, "tom")]
        )
        self.assertIn("Text.contains('tom')", fragment)
        self.assertEqual(ops, ["like"])

    def test_multiple_filters_chain(self):
        filters = [
            PropertyFilter("status", FilterOperator.EQ, "ok"),
            PropertyFilter("age", FilterOperator.GTE, 18),
        ]
        fragment, ops = FilterCompiler.compile_gremlin_has(filters)
        self.assertEqual(
            fragment,
            ".has('status', 'ok').has('age', P.gte(18))",
        )
        self.assertEqual(ops, ["eq", "gte"])

    def test_label_three_arg_form(self):
        fragment, _ = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("status", FilterOperator.EQ, "ok")], label="Table"
        )
        self.assertEqual(fragment, ".has('Table', 'status', 'ok')")

    def test_empty_filters(self):
        fragment, ops = FilterCompiler.compile_gremlin_has([])
        self.assertEqual(fragment, "")
        self.assertEqual(ops, [])

    def test_like_then_eq_chain(self):
        filters = [
            PropertyFilter("name", FilterOperator.LIKE, "tom"),
            PropertyFilter("status", FilterOperator.EQ, "ok"),
        ]
        fragment, ops = FilterCompiler.compile_gremlin_has(filters)
        self.assertIn("Text.contains('tom')", fragment)
        self.assertIn(".has('status', 'ok')", fragment)
        self.assertEqual(ops, ["like", "eq"])

    def test_escaping_injection_attempt(self):
        fragment, _ = FilterCompiler.compile_gremlin_has(
            [PropertyFilter("name", FilterOperator.EQ, "'; g.V().drop(); //")]
        )
        # the single quote inside the payload is escaped (\'), so it cannot
        # break out of the wrapping literal; the payload stays inert text
        self.assertIn("\\'", fragment)
        self.assertTrue(fragment.startswith(".has('name', '"))
        self.assertTrue(fragment.endswith("')"))


if __name__ == "__main__":
    unittest.main()
