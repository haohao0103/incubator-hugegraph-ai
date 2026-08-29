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

from hugegraph_llm.operators.hugegraph_op.kg_graph_schema import (
    ExtractionValidation,
    GraphSchema,
    NodeType,
    PropertyType,
    RelationshipType,
    _sanitize_value,
)

pytestmark = [pytest.mark.unit]


class TestSanitizeValue(unittest.TestCase):
    def test_text(self):
        self.assertEqual(_sanitize_value("TEXT", "abc"), "abc")
        self.assertEqual(_sanitize_value("TEXT", 42), "42")

    def test_int_and_long(self):
        self.assertEqual(_sanitize_value("INT", 30), 30)
        self.assertEqual(_sanitize_value("INT", "30"), 30)
        self.assertEqual(_sanitize_value("INT", 30.0), 30)
        self.assertEqual(_sanitize_value("LONG", "42"), 42)
        with self.assertRaises(ValueError):
            _sanitize_value("INT", "abc")
        with self.assertRaises(ValueError):
            _sanitize_value("INT", True)
        with self.assertRaises(ValueError):
            _sanitize_value("INT", 3.14)

    def test_double(self):
        self.assertEqual(_sanitize_value("DOUBLE", "0.8"), 0.8)
        self.assertEqual(_sanitize_value("DOUBLE", 1), 1.0)
        with self.assertRaises(ValueError):
            _sanitize_value("DOUBLE", "abc")
        with self.assertRaises(ValueError):
            _sanitize_value("DOUBLE", True)

    def test_boolean(self):
        self.assertIs(_sanitize_value("BOOLEAN", True), True)
        self.assertIs(_sanitize_value("BOOLEAN", "true"), True)
        self.assertIs(_sanitize_value("BOOLEAN", "FALSE"), False)
        with self.assertRaises(ValueError):
            _sanitize_value("BOOLEAN", "yes")

    def test_date(self):
        self.assertEqual(_sanitize_value("DATE", "2026-08-29"), "2026-08-29")
        with self.assertRaises(ValueError):
            _sanitize_value("DATE", "2026/08/29")

    def test_uuid_and_unsupported(self):
        self.assertEqual(_sanitize_value("UUID", "abc-123"), "abc-123")
        with self.assertRaises(ValueError):
            _sanitize_value("BLOB", "x")

    def test_double_rejects_uncoercible(self):
        with self.assertRaises(ValueError):
            _sanitize_value("DOUBLE", ["not", "a", "number"])


class TestPropertyType(unittest.TestCase):
    def test_valid(self):
        p = PropertyType("name", "TEXT", "SINGLE", "the name", True)
        self.assertEqual(p.to_dict()["data_type"], "TEXT")
        self.assertTrue(p.required)

    def test_invalid_data_type(self):
        with self.assertRaises(ValueError):
            PropertyType("x", "BLOB")

    def test_invalid_cardinality(self):
        with self.assertRaises(ValueError):
            PropertyType("x", "TEXT", "BOGUS")


class TestNodeType(unittest.TestCase):
    def test_valid_and_property_names(self):
        node = NodeType(
            "Person",
            properties=[PropertyType("name", required=True), PropertyType("age", "INT")],
        )
        self.assertEqual(node.property_names(), {"name", "age"})
        self.assertEqual(
            node.to_dict()["nullable_keys"], ["age"]  # only non-required
        )

    def test_invalid_label(self):
        with self.assertRaises(ValueError):
            NodeType("1person")
        with self.assertRaises(ValueError):
            NodeType("has-hyphen")

    def test_duplicate_properties(self):
        with self.assertRaises(ValueError):
            NodeType("Person", properties=[PropertyType("name"), PropertyType("name")])


class TestRelationshipType(unittest.TestCase):
    def test_valid(self):
        rel = RelationshipType("KNOWS", "Person", "Person", properties=[PropertyType("weight", "DOUBLE")])
        self.assertEqual(rel.property_names(), {"weight"})
        d = rel.to_dict()
        self.assertEqual(d["source_label"], "Person")
        self.assertEqual(d["target_label"], "Person")

    def test_invalid_labels(self):
        with self.assertRaises(ValueError):
            RelationshipType("KNOWS", "1person", "Person")
        with self.assertRaises(ValueError):
            RelationshipType("", "Person", "Person")


class TestGraphSchema(unittest.TestCase):
    def _schema(self, additional=True):
        return GraphSchema(
            nodes=[
                NodeType(
                    "Person",
                    properties=[PropertyType("name", required=True), PropertyType("age", "INT")],
                    additional_properties=additional,
                )
            ],
            relationships=[
                RelationshipType(
                    "KNOWS", "Person", "Person",
                    properties=[PropertyType("weight", "DOUBLE")],
                    additional_properties=additional,
                )
            ],
        )

    def test_duplicate_labels_raise(self):
        with self.assertRaises(ValueError):
            GraphSchema(nodes=[NodeType("Person"), NodeType("Person")])
        with self.assertRaises(ValueError):
            GraphSchema(relationships=[RelationshipType("KNOWS", "A", "B"), RelationshipType("KNOWS", "A", "B")])

    def test_lookup(self):
        s = self._schema()
        self.assertIsNotNone(s.node("Person"))
        self.assertIsNone(s.node("Ghost"))
        self.assertIsNotNone(s.relationship("KNOWS"))
        self.assertIsNone(s.relationship("NOPE"))

    def test_to_dict_dedup_propertykeys(self):
        s = GraphSchema(
            nodes=[
                NodeType("Person", properties=[PropertyType("name")]),
                NodeType("Company", properties=[PropertyType("name"), PropertyType("age", "INT")]),
            ],
            relationships=[RelationshipType("KNOWS", "Person", "Company",
                                              properties=[PropertyType("name"), PropertyType("weight", "DOUBLE")])],
        )
        d = s.to_dict()
        # name deduped across Person/Company/KNOWS; age + weight once each
        self.assertEqual(len(d["propertykeys"]), 3)
        self.assertEqual(len(d["vertexlabels"]), 2)
        self.assertEqual(len(d["edgelabels"]), 1)

    def test_validate_extraction_mixed(self):
        s = self._schema()
        v = s.validate_extraction(
            entities=[
                {"label": "Person", "properties": {"name": "Tom", "age": "30"}},
                {"label": "Ghost", "properties": {}},
                {"label": "Person", "properties": {"age": "30"}},  # missing required name
                {"label": "Person", "properties": {"name": "x", "age": "abc"}},  # bad int
            ],
            relationships=[
                {"label": "KNOWS", "source": "A", "target": "B", "properties": {"weight": "0.8"}},
                {"label": "KNOWS", "source": "A", "properties": {}},  # missing target
                {"label": "NOPE", "source": "A", "target": "B", "properties": {}},
            ],
        )
        self.assertEqual(len(v.valid_entities), 1)
        self.assertEqual(v.valid_entities[0]["properties"]["age"], 30)  # coerced
        reasons = [e["_reason"] for e in v.invalid_entities]
        self.assertEqual(len(reasons), 3)
        self.assertIn("unknown vertex label 'Ghost'", reasons)
        self.assertIn("missing required property 'name'", reasons)
        self.assertIn("expected int for INT", reasons[2])
        self.assertEqual(len(v.valid_relationships), 1)
        self.assertEqual(v.valid_relationships[0]["properties"]["weight"], 0.8)
        rel_reasons = [r["_reason"] for r in v.invalid_relationships]
        self.assertEqual(len(rel_reasons), 2)
        self.assertIn("missing source/target", rel_reasons)
        self.assertIn("unknown relationship label 'NOPE'", rel_reasons)

    def test_validate_strict_drops_unknown_properties(self):
        s = self._schema(additional=False)
        v = s.validate_extraction(
            entities=[{"label": "Person", "properties": {"name": "Tom", "extra": "x"}}],
            relationships=[{"label": "KNOWS", "source": "A", "target": "B",
                            "properties": {"weight": 1.0, "note": "y"}}],
        )
        self.assertEqual(v.valid_entities[0]["properties"], {"name": "Tom"})
        self.assertEqual(v.valid_relationships[0]["properties"], {"weight": 1.0})

    def test_validate_relationship_property_type_error(self):
        s = self._schema()
        v = s.validate_extraction(
            relationships=[{"label": "KNOWS", "source": "A", "target": "B",
                            "properties": {"weight": "not-a-number"}}],
        )
        self.assertEqual(len(v.valid_relationships), 0)
        self.assertIn("expected number", v.invalid_relationships[0]["_reason"])

    def test_validate_lenient_keeps_unknown_properties(self):
        s = self._schema(additional=True)
        v = s.validate_extraction(
            entities=[{"label": "Person", "properties": {"name": "Tom", "extra": "x"}}],
        )
        self.assertEqual(v.valid_entities[0]["properties"], {"name": "Tom", "extra": "x"})

    def test_validate_list_cardinality(self):
        s = GraphSchema(nodes=[NodeType("Doc", properties=[PropertyType("tags", "TEXT", "LIST")])])
        v = s.validate_extraction(
            entities=[{"label": "Doc", "properties": {"tags": ["a", "b"]}}],
        )
        self.assertEqual(v.valid_entities[0]["properties"]["tags"], ["a", "b"])
        # non-list value fails
        v2 = s.validate_extraction(
            entities=[{"label": "Doc", "properties": {"tags": "not-a-list"}}],
        )
        self.assertEqual(len(v2.valid_entities), 0)
        self.assertIn("expects a list", v2.invalid_entities[0]["_reason"])

    def test_empty_extraction(self):
        s = self._schema()
        v = s.validate_extraction()
        self.assertEqual(v.total, 0)
        self.assertEqual(v.to_dict()["valid_entities"], [])


class TestExtractionValidation(unittest.TestCase):
    def test_total_and_to_dict(self):
        v = ExtractionValidation(
            valid_entities=[{"label": "A"}],
            invalid_relationships=[{"label": "B"}],
        )
        self.assertEqual(v.total, 2)
        self.assertEqual(len(v.to_dict()["invalid_relationships"]), 1)


if __name__ == "__main__":
    unittest.main()
