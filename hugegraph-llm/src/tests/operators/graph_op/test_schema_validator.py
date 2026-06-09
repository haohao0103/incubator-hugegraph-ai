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

"""Tests for Sprint 8 Schema Constraint System."""

import pytest

from hugegraph_llm.operators.graph_op.schema_validator import (
    Cardinality,
    EntityLabelDef,
    PropertyDef,
    PropertyType,
    RelationDef,
    SchemaDefinition,
    SchemaValidator,
    Severity,
    ValidationResult,
    Violation,
)


# ============================================================
# Helper: build test schema
# ============================================================

def _build_test_schema() -> SchemaDefinition:
    schema = SchemaDefinition(version="1.0")
    schema.global_rules = {
        "max_properties_per_entity": 10,
        "allow_dynamic_labels": False,
    }
    schema.entity_labels["Person"] = EntityLabelDef(
        label="Person",
        primary_key="name",
        properties={
            "name": PropertyDef(
                name="name", required=True,
                cardinality=Cardinality.SINGLE,
                pattern=r"^.{1,100}$",
            ),
            "age": PropertyDef(
                name="age", prop_type=PropertyType.INTEGER,
                min_value=0, max_value=200,
            ),
            "email": PropertyDef(
                name="email", pattern=r"^[^@]+@[^@]+\.[^@]+$",
            ),
            "bio": PropertyDef(
                name="bio", prop_type=PropertyType.TEXT,
                max_length=500,
            ),
        }
    )
    schema.entity_labels["Company"] = EntityLabelDef(
        label="Company",
        primary_key="name",
        properties={
            "name": PropertyDef(
                name="name", required=True,
                cardinality=Cardinality.SINGLE,
            ),
            "revenue": PropertyDef(
                name="revenue", prop_type=PropertyType.FLOAT,
            ),
        }
    )
    schema.relations["works_at"] = RelationDef(
        label="works_at",
        source_labels=["Person"],
        target_labels=["Company"],
        description="Person works at Company",
    )
    schema.relations["knows"] = RelationDef(
        label="knows",
        source_labels=["Person"],
        target_labels=["Person"],
        description="Person knows Person",
        properties={
            "since": PropertyDef(
                name="since", prop_type=PropertyType.INTEGER,
            ),
        },
    )
    return schema


# ============================================================
# PropertyDef Tests
# ============================================================

class TestPropertyDef:
    def test_defaults(self):
        p = PropertyDef(name="test")
        assert p.prop_type == PropertyType.STRING
        assert p.cardinality == Cardinality.OPTIONAL
        assert p.required is False

    def test_custom(self):
        p = PropertyDef(
            name="age", prop_type=PropertyType.INTEGER,
            cardinality=Cardinality.SINGLE, required=True,
            min_value=0, max_value=200
        )
        assert p.prop_type == PropertyType.INTEGER
        assert p.required is True
        assert p.min_value == 0

    def test_all_types(self):
        for pt in PropertyType:
            p = PropertyDef(name="x", prop_type=pt)
            assert p.prop_type == pt


# ============================================================
# EntityLabelDef Tests
# ============================================================

class TestEntityLabelDef:
    def test_defaults(self):
        el = EntityLabelDef(label="Person")
        assert el.primary_key == "name"
        assert el.properties == {}

    def test_with_properties(self):
        el = EntityLabelDef(
            label="Person",
            properties={"name": PropertyDef(name="name", required=True)},
            primary_key="id",
        )
        assert el.primary_key == "id"
        assert "name" in el.properties


# ============================================================
# RelationDef Tests
# ============================================================

class TestRelationDef:
    def test_defaults(self):
        r = RelationDef(label="knows")
        assert r.source_labels == []
        assert r.target_labels == []

    def test_with_constraints(self):
        r = RelationDef(
            label="works_at",
            source_labels=["Person"],
            target_labels=["Company"],
        )
        assert "Person" in r.source_labels
        assert "Company" in r.target_labels


# ============================================================
# SchemaDefinition Tests
# ============================================================

class TestSchemaDefinition:
    def test_get_entity_label(self):
        schema = SchemaDefinition()
        schema.entity_labels["Person"] = EntityLabelDef(label="Person")
        assert schema.get_entity_label("Person") is not None
        assert schema.get_entity_label("Unknown") is None

    def test_is_entity_label(self):
        schema = SchemaDefinition()
        schema.entity_labels["X"] = EntityLabelDef(label="X")
        assert schema.is_entity_label("X") is True
        assert schema.is_entity_label("Y") is False

    def test_get_relation(self):
        schema = SchemaDefinition()
        schema.relations["knows"] = RelationDef(label="knows")
        assert schema.get_relation("knows") is not None

    def test_is_relation_label(self):
        schema = SchemaDefinition()
        assert schema.is_relation_label("x") is False


# ============================================================
# ValidationResult Tests
# ============================================================

class TestValidationResult:
    def test_default_valid(self):
        vr = ValidationResult()
        assert vr.is_valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_add_error(self):
        vr = ValidationResult()
        vr.add_violation("T1", Severity.ERROR, "fail")
        assert vr.is_valid is False
        assert len(vr.errors) == 1

    def test_add_warning_stays_valid(self):
        vr = ValidationResult()
        vr.add_violation("T1", Severity.WARNING, "warn")
        assert vr.is_valid is True
        assert len(vr.warnings) == 1

    def test_summary(self):
        vr = ValidationResult()
        vr.add_violation("E1", Severity.ERROR, "err")
        vr.add_violation("W1", Severity.WARNING, "warn")
        s = vr.summary()
        assert "Valid=False" in s
        assert "Errors=1" in s
        assert "Warnings=1" in s


# ============================================================
# SchemaValidator - Entity Validation
# ============================================================

class TestSchemaValidatorEntity:
    def setup_method(self):
        self.schema = _build_test_schema()
        self.validator = SchemaValidator(schema=self.schema)

    def test_valid_entity(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "age": 30, "email": "alice@example.com"
        })
        assert vr.is_valid is True

    def test_missing_required_property(self):
        vr = self.validator.validate_entity("Person", {"age": 30})
        assert vr.is_valid is False
        assert any("name" in v.message for v in vr.errors)

    def test_empty_label(self):
        vr = self.validator.validate_entity("", {"name": "X"})
        assert vr.is_valid is False

    def test_empty_properties(self):
        vr = self.validator.validate_entity("Person", {})
        assert len(vr.warnings) > 0  # no props warning + missing name

    def test_unknown_label_dynamic_allowed(self):
        val = SchemaValidator(strict_mode=False)  # default schema allows dynamic
        vr = val.validate_entity("NewLabel", {"name": "X"})
        assert vr.is_valid is True  # dynamic allowed

    def test_unknown_label_strict(self):
        vr = self.validator.validate_entity("Unknown", {"name": "X"})
        assert vr.is_valid is False
        assert any("Unknown" in v.message for v in vr.errors)

    def test_type_mismatch_age_string(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "age": "thirty"
        })
        assert vr.is_valid is True  # age not required, type mismatch is warning
        assert len(vr.warnings) > 0

    def test_age_out_of_range(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "age": 999
        })
        assert any("exceeds maximum" in v.message for v in vr.warnings)

    def test_age_negative(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "age": -5
        })
        assert any("below minimum" in v.message for v in vr.warnings)

    def test_email_pattern_mismatch(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "email": "not-an-email"
        })
        assert any("pattern" in v.message for v in vr.warnings)

    def test_email_pattern_valid(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "email": "alice@example.com"
        })
        # No pattern violation
        pattern_violations = [v for v in vr.violations if "pattern" in v.message]
        assert len(pattern_violations) == 0

    def test_bio_length_exceeded(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "bio": "x" * 600
        })
        assert any("exceeds max" in v.message for v in vr.warnings)

    def test_bio_length_ok(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "bio": "x" * 200
        })
        length_violations = [v for v in vr.violations if "length" in v.message]
        assert len(length_violations) == 0

    def test_unknown_property_warning(self):
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "unknown_prop": "value"
        })
        assert any("Unknown property" in v.message for v in vr.warnings)

    def test_unknown_property_strict_error(self):
        self.validator._strict_mode = True
        vr = self.validator.validate_entity("Person", {
            "name": "Alice", "unknown_prop": "value"
        })
        assert any("strict mode" in v.message for v in vr.errors)

    def test_null_required_property(self):
        vr = self.validator.validate_entity("Person", {
            "name": None
        })
        assert vr.is_valid is False

    def test_company_entity_valid(self):
        vr = self.validator.validate_entity("Company", {
            "name": "Acme", "revenue": 1000000.5
        })
        assert vr.is_valid is True

    def test_company_missing_name(self):
        vr = self.validator.validate_entity("Company", {"revenue": 100})
        assert vr.is_valid is False

    def test_violation_has_field(self):
        vr = self.validator.validate_entity("Person", {"age": 30})
        assert vr.errors[0].field == "name"


# ============================================================
# SchemaValidator - Relation Validation
# ============================================================

class TestSchemaValidatorRelation:
    def setup_method(self):
        self.schema = _build_test_schema()
        self.validator = SchemaValidator(schema=self.schema)

    def test_valid_relation(self):
        vr = self.validator.validate_relation(
            "works_at", "Person", "Company"
        )
        assert vr.is_valid is True

    def test_unknown_relation(self):
        vr = self.validator.validate_relation(
            "unknown_rel", "Person", "Company"
        )
        assert vr.is_valid is False

    def test_wrong_source_label(self):
        vr = self.validator.validate_relation(
            "works_at", "Company", "Company"
        )
        assert vr.is_valid is False
        assert any("Source label" in v.message for v in vr.errors)

    def test_wrong_target_label(self):
        vr = self.validator.validate_relation(
            "works_at", "Person", "Person"
        )
        assert vr.is_valid is False
        assert any("Target label" in v.message for v in vr.errors)

    def test_self_relation_valid(self):
        vr = self.validator.validate_relation(
            "knows", "Person", "Person"
        )
        assert vr.is_valid is True

    def test_relation_with_property(self):
        vr = self.validator.validate_relation(
            "knows", "Person", "Person",
            {"since": 2020}
        )
        assert vr.is_valid is True

    def test_relation_wrong_property_type(self):
        vr = self.validator.validate_relation(
            "knows", "Person", "Person",
            {"since": "not-a-year"}
        )
        assert len(vr.warnings) > 0

    def test_relation_unknown_property_strict(self):
        self.validator._strict_mode = True
        vr = self.validator.validate_relation(
            "knows", "Person", "Person",
            {"unknown": "value"}
        )
        assert any("strict mode" in v.message for v in vr.errors)

    def test_no_properties(self):
        vr = self.validator.validate_relation(
            "works_at", "Person", "Company", None
        )
        assert vr.is_valid is True


# ============================================================
# SchemaValidator - Batch Validation
# ============================================================

class TestSchemaValidatorBatch:
    def setup_method(self):
        self.schema = _build_test_schema()
        self.validator = SchemaValidator(schema=self.schema)

    def test_batch_mixed(self):
        items = [
            {"type": "entity", "label": "Person", "properties": {"name": "Alice"}},
            {"type": "entity", "label": "Person", "properties": {"age": 30}},
            {"type": "relation", "relation_label": "works_at",
             "source_label": "Person", "target_label": "Company"},
        ]
        results = self.validator.validate_batch(items)
        assert len(results) == 3
        assert results[0][1].is_valid is True
        assert results[1][1].is_valid is False
        assert results[2][1].is_valid is True

    def test_batch_empty(self):
        results = self.validator.validate_batch([])
        assert len(results) == 0

    def test_batch_unknown_type(self):
        items = [{"type": "unknown"}]
        results = self.validator.validate_batch(items)
        assert results[0][1].is_valid is False


# ============================================================
# SchemaValidator - Suggest Fix
# ============================================================

class TestSchemaValidatorSuggestFix:
    def setup_method(self):
        self.schema = _build_test_schema()
        self.validator = SchemaValidator(schema=self.schema)

    def test_suggest_fix_type_coercion(self):
        props = {"name": "Alice", "age": "30"}
        vr = self.validator.validate_entity("Person", props)
        fixed = self.validator.suggest_fix("Person", props, vr)
        assert isinstance(fixed["age"], int)
        assert fixed["age"] == 30

    def test_suggest_fix_boolean_coercion(self):
        props = {"name": "Alice", "age": 25}
        vr = self.validator.validate_entity("Person", props)
        # No coercion needed, should return same
        fixed = self.validator.suggest_fix("Person", props, vr)
        assert fixed["age"] == 25

    def test_suggest_fix_unknown_label(self):
        props = {"name": "Alice"}
        vr = self.validator.validate_entity("Unknown", props)
        fixed = self.validator.suggest_fix("Unknown", props, vr)
        assert fixed == props  # No changes possible


# ============================================================
# SchemaValidator - Schema Evolution
# ============================================================

class TestSchemaEvolution:
    def test_evolve_add_entity(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        new_schema = val.evolve_schema({
            "add_entity_labels": [
                EntityLabelDef(label="City", properties={
                    "name": PropertyDef(name="name", required=True)
                })
            ]
        })
        assert "City" in new_schema.entity_labels
        assert new_schema.version == "1.1"

    def test_evolve_add_relation(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        new_schema = val.evolve_schema({
            "add_relations": [
                RelationDef(
                    label="located_in",
                    source_labels=["Person"],
                    target_labels=["City"],
                )
            ]
        })
        assert "located_in" in new_schema.relations

    def test_evolve_remove_entity(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        new_schema = val.evolve_schema({
            "remove_entity_labels": ["Company"]
        })
        assert "Company" not in new_schema.entity_labels

    def test_evolve_add_property(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        new_schema = val.evolve_schema({
            "add_properties": {
                "Person": {
                    "phone": PropertyDef(name="phone", pattern=r"^\\d+$")
                }
            }
        })
        assert "phone" in new_schema.entity_labels["Person"].properties

    def test_evolve_update_global_rules(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        val.evolve_schema({
            "update_global_rules": {"allow_dynamic_labels": True}
        })
        assert val._schema.global_rules["allow_dynamic_labels"] is True

    def test_version_increments(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        assert val._schema.version == "1.0"
        val.evolve_schema({})
        assert val._schema.version == "1.1"
        val.evolve_schema({})
        assert val._schema.version == "1.2"


# ============================================================
# SchemaValidator - Serialization
# ============================================================

class TestSchemaSerialization:
    def test_to_dict(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        d = val.to_dict()
        assert d["version"] == "1.0"
        assert "Person" in d["entity_labels"]
        assert "works_at" in d["relations"]

    def test_from_dict_round_trip(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        d = val.to_dict()
        val2 = SchemaValidator.from_dict(d)
        assert val2._schema.version == schema.version
        assert "Person" in val2._schema.entity_labels
        assert "works_at" in val2._schema.relations

    def test_from_dict_empty(self):
        val = SchemaValidator.from_dict({})
        assert val._schema.version == "1.0"


# ============================================================
# SchemaValidator - Schema Summary
# ============================================================

class TestSchemaSummary:
    def test_summary_output(self):
        val = SchemaValidator(schema=_build_test_schema())
        s = val.schema_summary()
        assert "Schema v1.0" in s
        assert "Person" in s
        assert "works_at" in s

    def test_default_schema_summary(self):
        val = SchemaValidator()
        s = val.schema_summary()
        assert "Entity" in s
        assert "Community" in s


# ============================================================
# SchemaValidator - Operator Protocol
# ============================================================

class TestSchemaValidatorOperator:
    def test_run_with_entities(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        ctx = {
            "extracted_entities": [
                {"label": "Person", "properties": {"name": "Alice", "age": 30}},
                {"label": "Person", "properties": {"age": 25}},  # missing name
            ],
            "extracted_relations": [
                {
                    "relation_label": "works_at",
                    "source_label": "Person",
                    "target_label": "Company",
                    "properties": {}
                }
            ]
        }
        result = val.run(ctx)
        assert "validation_result" in result
        assert "validated_entities" in result
        assert "validation_errors" in result
        assert len(result["validated_entities"]) == 1
        assert len(result["validation_errors"]) == 1

    def test_run_empty(self):
        val = SchemaValidator()
        ctx = {"extracted_entities": [], "extracted_relations": []}
        result = val.run(ctx)
        assert result["validated_entities"] == []
        assert result["validated_relations"] == []

    def test_run_with_custom_schema(self):
        custom = SchemaDefinition(version="2.0")
        custom.global_rules = {"allow_dynamic_labels": True}
        val = SchemaValidator()
        ctx = {
            "schema": custom,
            "extracted_entities": [
                {"label": "NewType", "properties": {"name": "X"}}
            ]
        }
        result = val.run(ctx)
        assert len(result["validated_entities"]) == 1

    def test_run_validates_relations(self):
        schema = _build_test_schema()
        val = SchemaValidator(schema=schema)
        ctx = {
            "extracted_entities": [],
            "extracted_relations": [
                {
                    "relation_label": "works_at",
                    "source_label": "Company",  # wrong source
                    "target_label": "Company",
                }
            ]
        }
        result = val.run(ctx)
        assert len(result["validated_relations"]) == 0
        assert len(result["validation_errors"]) == 1


# ============================================================
# SchemaValidator - Default Schema
# ============================================================

class TestDefaultSchema:
    def test_default_has_entity(self):
        val = SchemaValidator()
        assert val._schema.is_entity_label("Entity")

    def test_default_has_community(self):
        val = SchemaValidator()
        assert val._schema.is_entity_label("Community")

    def test_default_has_chunk(self):
        val = SchemaValidator()
        assert val._schema.is_entity_label("Chunk")

    def test_default_has_relations(self):
        val = SchemaValidator()
        assert val._schema.is_relation_label("relates_to")
        assert val._schema.is_relation_label("belongs_to")
        assert val._schema.is_relation_label("has_chunk")

    def test_default_allows_dynamic_labels(self):
        val = SchemaValidator()
        assert val._schema.global_rules.get("allow_dynamic_labels") is True

    def test_default_version(self):
        val = SchemaValidator()
        assert val._schema.version == "1.0"

    def test_entity_required_name(self):
        val = SchemaValidator()
        vr = val.validate_entity("Entity", {})
        assert vr.is_valid is False


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_boolean_true(self):
        val = SchemaValidator()
        # Default schema doesn't have boolean props,
        # but the type checker should work
        assert SchemaValidator._check_type(True, PropertyType.BOOLEAN) is True
        assert SchemaValidator._check_type(False, PropertyType.BOOLEAN) is True

    def test_integer_not_bool(self):
        # bool is subclass of int in Python
        assert SchemaValidator._check_type(True, PropertyType.INTEGER) is False
        assert SchemaValidator._check_type(1, PropertyType.INTEGER) is True

    def test_float_accepts_int(self):
        assert SchemaValidator._check_type(5, PropertyType.FLOAT) is True
        assert SchemaValidator._check_type(5.5, PropertyType.FLOAT) is True

    def test_list_types(self):
        assert SchemaValidator._check_type([], PropertyType.LIST_STRING) is True
        assert SchemaValidator._check_type([1], PropertyType.LIST_INTEGER) is True
        assert SchemaValidator._check_type("x", PropertyType.LIST_STRING) is False

    def test_date_string(self):
        assert SchemaValidator._check_type("2024-01-01", PropertyType.DATE) is True

    def test_try_coerce_int_from_string(self):
        result = SchemaValidator._try_coerce(
            "42", PropertyDef(name="x", prop_type=PropertyType.INTEGER)
        )
        assert result == 42

    def test_try_coerce_float_from_string(self):
        result = SchemaValidator._try_coerce(
            "3.14", PropertyDef(name="x", prop_type=PropertyType.FLOAT)
        )
        assert abs(result - 3.14) < 0.001

    def test_try_coerce_boolean_from_string(self):
        result = SchemaValidator._try_coerce(
            "true", PropertyDef(name="x", prop_type=PropertyType.BOOLEAN)
        )
        assert result is True

    def test_try_coerce_fail(self):
        result = SchemaValidator._try_coerce(
            "not_a_number", PropertyDef(name="x", prop_type=PropertyType.INTEGER)
        )
        assert result is None

    def test_violation_dataclass(self):
        v = Violation(
            rule_id="T1", severity=Severity.ERROR,
            message="test error", field="name",
            suggested_fix="add name"
        )
        assert v.rule_id == "T1"
        assert v.severity == Severity.ERROR

    def test_multiple_violations_ordered(self):
        vr = ValidationResult()
        vr.add_violation("E1", Severity.ERROR, "err1")
        vr.add_violation("W1", Severity.WARNING, "warn1")
        vr.add_violation("E2", Severity.ERROR, "err2")
        assert len(vr.violations) == 3
        assert vr.violations[0].rule_id == "E1"
        assert vr.violations[2].rule_id == "E2"
