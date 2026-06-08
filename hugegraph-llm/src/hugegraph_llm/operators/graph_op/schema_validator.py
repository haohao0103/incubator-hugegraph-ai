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

"""
Schema Constraint System for GraphRAG entity extraction.

Enforces ontology/structure rules on extracted entities and relations:
- Validates entity labels against a defined schema
- Checks property types and cardinality constraints
- Enforces allowed relationship types between entity types
- Supports schema evolution (add/remove labels/properties)
- Provides validation reports for data quality monitoring

Reference: Neo4j GraphRAG schema constraints, MS GraphRAG entity types.
"""

import copy
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from hugegraph_llm.utils.log import log


# ============================================================
# Data Models
# ============================================================

class PropertyType(Enum):
    """Supported property data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    LIST_STRING = "list<string>"
    LIST_INTEGER = "list<integer>"
    TEXT = "text"  # Long text field


class Cardinality(Enum):
    """Property cardinality constraints."""
    SINGLE = "single"      # Exactly one value (required)
    OPTIONAL = "optional"  # Zero or one value
    MULTI = "multi"        # Zero or more values (list)


@dataclass
class PropertyDef:
    """Definition of a property in the schema."""
    name: str
    prop_type: PropertyType = PropertyType.STRING
    cardinality: Cardinality = Cardinality.OPTIONAL
    required: bool = False
    pattern: Optional[str] = None  # Regex pattern for string properties
    min_value: Optional[float] = None  # For numeric properties
    max_value: Optional[float] = None  # For numeric properties
    max_length: Optional[int] = None   # For string/text properties
    description: str = ""


@dataclass
class EntityLabelDef:
    """Definition of an entity label in the schema."""
    label: str
    properties: Dict[str, PropertyDef] = field(default_factory=dict)
    primary_key: str = "name"  # Default primary key property
    description: str = ""
    parent_labels: List[str] = field(default_factory=list)  # Inheritance


@dataclass
class RelationDef:
    """Definition of a relation/edge type in the schema."""
    label: str
    source_labels: List[str] = field(default_factory=list)  # Allowed source labels
    target_labels: List[str] = field(default_factory=list)  # Allowed target labels
    properties: Dict[str, PropertyDef] = field(default_factory=dict)
    source_cardinality: str = "single"  # single | multi
    target_cardinality: str = "single"
    description: str = ""


@dataclass
class SchemaDefinition:
    """Complete schema definition for the graph."""
    entity_labels: Dict[str, EntityLabelDef] = field(default_factory=dict)
    relations: Dict[str, RelationDef] = field(default_factory=dict)
    global_rules: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"

    def get_entity_label(self, label: str) -> Optional[EntityLabelDef]:
        return self.entity_labels.get(label)

    def get_relation(self, label: str) -> Optional[RelationDef]:
        return self.relations.get(label)

    def is_entity_label(self, label: str) -> bool:
        return label in self.entity_labels

    def is_relation_label(self, label: str) -> bool:
        return label in self.relations


# ============================================================
# Validation Result
# ============================================================

class Severity(Enum):
    ERROR = "error"    # Must fix (blocks insertion)
    WARNING = "warning"  # Should fix (advisory)
    INFO = "info"      # Informational


@dataclass
class Violation:
    """A single schema violation."""
    rule_id: str
    severity: Severity
    message: str
    field: str = ""
    suggested_fix: str = ""


@dataclass
class ValidationResult:
    """Result of schema validation."""
    is_valid: bool = True
    violations: List[Violation] = field(default_factory=list)

    def add_violation(self, rule_id: str, severity: Severity,
                      message: str, field: str = "",
                      suggested_fix: str = ""):
        v = Violation(rule_id=rule_id, severity=severity,
                      message=message, field=field,
                      suggested_fix=suggested_fix)
        self.violations.append(v)
        if severity == Severity.ERROR:
            self.is_valid = False

    @property
    def errors(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[Violation]:
        return [v for v in self.violations if v.severity == Severity.WARNING]

    def summary(self) -> str:
        return (
            f"Valid={self.is_valid}, "
            f"Errors={len(self.errors)}, "
            f"Warnings={len(self.warnings)}, "
            f"Total={len(self.violations)}"
        )


# ============================================================
# Schema Validator
# ============================================================

class SchemaValidator:
    """Validates entities and relations against a SchemaDefinition.

    Usage:
        schema = SchemaDefinition()
        schema.entity_labels["Person"] = EntityLabelDef(
            label="Person",
            properties={
                "name": PropertyDef(name="name", required=True,
                                     cardinality=Cardinality.SINGLE),
                "age": PropertyDef(name="age", prop_type=PropertyType.INTEGER),
            }
        )
        validator = SchemaValidator(schema)
        result = validator.validate_entity("Person", {"name": "Alice", "age": 30})
    """

    def __init__(self, schema: Optional[SchemaDefinition] = None,
                 strict_mode: bool = False):
        """
        :param schema: Schema definition to validate against.
                       If None, uses a built-in default schema.
        :param strict_mode: If True, unknown properties cause errors.
                             If False, they cause warnings.
        """
        self._schema = schema or self._default_schema()
        self._strict_mode = strict_mode

    @staticmethod
    def _default_schema() -> SchemaDefinition:
        """Built-in default schema for GraphRAG."""
        schema = SchemaDefinition(version="1.0")
        schema.global_rules = {
            "max_properties_per_entity": 50,
            "max_relation_types": 200,
            "allow_dynamic_labels": True,
        }

        # Common entity labels
        schema.entity_labels["Entity"] = EntityLabelDef(
            label="Entity",
            properties={
                "name": PropertyDef(
                    name="name", required=True,
                    cardinality=Cardinality.SINGLE,
                    description="Entity display name"
                ),
                "description": PropertyDef(
                    name="description",
                    prop_type=PropertyType.TEXT,
                    description="Entity description"
                ),
                "entity_type": PropertyDef(
                    name="entity_type",
                    description="Semantic type classification"
                ),
            }
        )

        schema.entity_labels["Community"] = EntityLabelDef(
            label="Community",
            properties={
                "community_id": PropertyDef(
                    name="community_id", required=True,
                    cardinality=Cardinality.SINGLE,
                    description="Community identifier"
                ),
                "level": PropertyDef(
                    name="level", prop_type=PropertyType.INTEGER,
                    description="Hierarchy level"
                ),
                "summary": PropertyDef(
                    name="summary", prop_type=PropertyType.TEXT,
                    description="Community summary report"
                ),
            }
        )

        schema.entity_labels["Chunk"] = EntityLabelDef(
            label="Chunk",
            properties={
                "content": PropertyDef(
                    name="content", prop_type=PropertyType.TEXT,
                    required=True, description="Chunk text content"
                ),
                "chunk_id": PropertyDef(
                    name="chunk_id", required=True,
                    cardinality=Cardinality.SINGLE,
                    description="Unique chunk identifier"
                ),
                "source_doc": PropertyDef(
                    name="source_doc",
                    description="Source document name"
                ),
            }
        )

        # Common relation types
        schema.relations["relates_to"] = RelationDef(
            label="relates_to",
            description="Generic relation between entities"
        )
        schema.relations["belongs_to"] = RelationDef(
            label="belongs_to",
            description="Entity belongs to a community"
        )
        schema.relations["has_chunk"] = RelationDef(
            label="has_chunk",
            description="Entity has associated text chunk"
        )

        return schema

    # ---- Public API ----

    def validate_entity(self, label: str, properties: Dict[str, Any],
                        context: Optional[Dict] = None) -> ValidationResult:
        """Validate an entity against the schema.

        :param label: Entity label (vertex label).
        :param properties: Entity properties dict.
        :param context: Optional context for cross-validation.
        :return: ValidationResult.
        """
        result = ValidationResult()
        if not label:
            result.add_violation(
                "ENTITY_001", Severity.ERROR,
                "Entity label is empty", field="label"
            )
            return result

        if not properties:
            result.add_violation(
                "ENTITY_002", Severity.WARNING,
                "Entity has no properties", field=str(properties)
            )
            # Do NOT return early — still check required fields below

        label_def = self._schema.get_entity_label(label)

        # Unknown label check
        if label_def is None:
            if self._schema.global_rules.get("allow_dynamic_labels", True):
                result.add_violation(
                    "ENTITY_003", Severity.INFO,
                    f"Entity label '{label}' not in schema (dynamic label allowed)",
                    field="label"
                )
            else:
                result.add_violation(
                    "ENTITY_004", Severity.ERROR,
                    f"Entity label '{label}' not defined in schema",
                    field="label",
                    suggested_fix=f"Add '{label}' to schema or use a known label"
                )
            return result

        # Check required properties
        for prop_name, prop_def in label_def.properties.items():
            if prop_def.required and prop_name not in properties:
                result.add_violation(
                    "ENTITY_010", Severity.ERROR,
                    f"Required property '{prop_name}' missing",
                    field=prop_name,
                    suggested_fix=f"Provide a value for '{prop_name}'"
                )

        # Validate each property
        for prop_name, value in properties.items():
            prop_def = label_def.properties.get(prop_name)
            if prop_def is None:
                if self._strict_mode:
                    result.add_violation(
                        "ENTITY_020", Severity.ERROR,
                        f"Unknown property '{prop_name}' on label '{label}' "
                        f"(strict mode)",
                        field=prop_name,
                        suggested_fix="Remove property or add to schema"
                    )
                else:
                    result.add_violation(
                        "ENTITY_021", Severity.WARNING,
                        f"Unknown property '{prop_name}' on label '{label}'",
                        field=prop_name
                    )
            else:
                self._validate_property(result, prop_def, value, label)

        return result

    def validate_relation(self, relation_label: str,
                         source_label: str, target_label: str,
                         properties: Optional[Dict[str, Any]] = None
                         ) -> ValidationResult:
        """Validate a relation/edge against the schema.

        :param relation_label: Edge label.
        :param source_label: Source vertex label.
        :param target_label: Target vertex label.
        :param properties: Edge properties dict.
        :return: ValidationResult.
        """
        result = ValidationResult()
        properties = properties or {}

        rel_def = self._schema.get_relation(relation_label)

        # Unknown relation
        if rel_def is None:
            if self._schema.global_rules.get("allow_dynamic_labels", True):
                result.add_violation(
                    "REL_001", Severity.INFO,
                    f"Relation '{relation_label}' not in schema (dynamic allowed)",
                    field=relation_label
                )
            else:
                result.add_violation(
                    "REL_002", Severity.ERROR,
                    f"Relation '{relation_label}' not defined in schema",
                    field=relation_label
                )
            return result

        # Check source label constraint
        if rel_def.source_labels:
            if source_label not in rel_def.source_labels:
                result.add_violation(
                    "REL_010", Severity.ERROR,
                    f"Source label '{source_label}' not allowed for relation "
                    f"'{relation_label}'. Allowed: {rel_def.source_labels}",
                    field="source",
                    suggested_fix=f"Use one of: {rel_def.source_labels}"
                )

        # Check target label constraint
        if rel_def.target_labels:
            if target_label not in rel_def.target_labels:
                result.add_violation(
                    "REL_011", Severity.ERROR,
                    f"Target label '{target_label}' not allowed for relation "
                    f"'{relation_label}'. Allowed: {rel_def.target_labels}",
                    field="target",
                    suggested_fix=f"Use one of: {rel_def.target_labels}"
                )

        # Validate relation properties
        for prop_name, value in properties.items():
            prop_def = rel_def.properties.get(prop_name)
            if prop_def is None:
                if self._strict_mode:
                    result.add_violation(
                        "REL_020", Severity.ERROR,
                        f"Unknown property '{prop_name}' on relation "
                        f"'{relation_label}' (strict mode)",
                        field=prop_name
                    )
            else:
                self._validate_property(result, prop_def, value, relation_label)

        return result

    def validate_batch(self, items: List[Dict[str, Any]]
                        ) -> List[Tuple[Dict[str, Any], ValidationResult]]:
        """Validate a batch of entities/relations.

        Each item should have:
          - type: "entity" or "relation"
          - For entity: label, properties
          - For relation: relation_label, source_label, target_label, properties

        :param items: List of item dicts.
        :return: List of (item, ValidationResult) tuples.
        """
        results = []
        for item in items:
            item_type = item.get("type", "entity")
            if item_type == "entity":
                vr = self.validate_entity(
                    item["label"],
                    item.get("properties", {})
                )
            elif item_type == "relation":
                vr = self.validate_relation(
                    item.get("relation_label", ""),
                    item.get("source_label", ""),
                    item.get("target_label", ""),
                    item.get("properties", {})
                )
            else:
                vr = ValidationResult()
                vr.add_violation(
                    "BATCH_001", Severity.ERROR,
                    f"Unknown item type: {item_type}"
                )
            results.append((item, vr))
        return results

    def suggest_fix(self, label: str, properties: Dict[str, Any],
                     result: ValidationResult) -> Dict[str, Any]:
        """Generate suggested fixes for validation violations.

        :param label: Entity label.
        :param properties: Original properties.
        :param result: Validation result with violations.
        :return: Fixed properties dict.
        """
        fixed = copy.deepcopy(properties)
        label_def = self._schema.get_entity_label(label)
        if not label_def:
            return fixed

        for v in result.violations:
            if v.field and v.severity in (Severity.ERROR, Severity.WARNING):
                if v.field in fixed:
                    # Try type coercion for simple type mismatches
                    prop_def = label_def.properties.get(v.field)
                    if prop_def:
                        coerced = self._try_coerce(fixed[v.field], prop_def)
                        if coerced is not None:
                            fixed[v.field] = coerced
        return fixed

    def evolve_schema(self, changes: Dict[str, Any]) -> SchemaDefinition:
        """Apply schema evolution changes.

        :param changes: Dict with optional keys:
            - add_entity_labels: List[EntityLabelDef]
            - add_relations: List[RelationDef]
            - remove_entity_labels: List[str]
            - remove_relations: List[str]
            - add_properties: Dict[label, Dict[prop_name, PropertyDef]]
            - update_global_rules: Dict[str, Any]
        :return: Updated SchemaDefinition (new instance).
        """
        new_schema = copy.deepcopy(self._schema)
        new_schema.version = str(float(new_schema.version) + 0.1)

        # Add entity labels
        for el_def in changes.get("add_entity_labels", []):
            if isinstance(el_def, EntityLabelDef):
                new_schema.entity_labels[el_def.label] = el_def
            elif isinstance(el_def, dict):
                el = EntityLabelDef(**el_def)
                new_schema.entity_labels[el.label] = el

        # Add relations
        for rel_def in changes.get("add_relations", []):
            if isinstance(rel_def, RelationDef):
                new_schema.relations[rel_def.label] = rel_def

        # Remove entity labels
        for label in changes.get("remove_entity_labels", []):
            new_schema.entity_labels.pop(label, None)

        # Remove relations
        for label in changes.get("remove_relations", []):
            new_schema.relations.pop(label, None)

        # Add properties to existing labels
        for label, props in changes.get("add_properties", {}).items():
            el = new_schema.entity_labels.get(label)
            if el:
                for pname, pdef in props.items():
                    el.properties[pname] = pdef

        # Update global rules
        for k, v in changes.get("update_global_rules", {}).items():
            new_schema.global_rules[k] = v

        self._schema = new_schema
        log.info("Schema evolved to version %s", new_schema.version)
        return new_schema

    def schema_summary(self) -> str:
        """Human-readable summary of the current schema."""
        lines = [f"Schema v{self._schema.version}"]
        lines.append(f"  Entity labels: {len(self._schema.entity_labels)}")
        for name, el in self._schema.entity_labels.items():
            req = sum(1 for p in el.properties.values() if p.required)
            lines.append(
                f"    {name}: {len(el.properties)} props ({req} required), "
                f"PK={el.primary_key}"
            )
        lines.append(f"  Relations: {len(self._schema.relations)}")
        for name, rel in self._schema.relations.items():
            lines.append(f"    {name}: {rel.source_labels} -> {rel.target_labels}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize schema to dict."""
        def _serialize_pd(pd_obj):
            if isinstance(pd_obj, PropertyDef):
                return {
                    "name": pd_obj.name,
                    "type": pd_obj.prop_type.value,
                    "cardinality": pd_obj.cardinality.value,
                    "required": pd_obj.required,
                    "pattern": pd_obj.pattern,
                }
            return pd_obj

        d = {
            "version": self._schema.version,
            "entity_labels": {},
            "relations": {},
            "global_rules": self._schema.global_rules,
        }
        for name, el in self._schema.entity_labels.items():
            d["entity_labels"][name] = {
                "label": el.label,
                "primary_key": el.primary_key,
                "description": el.description,
                "properties": {
                    pn: _serialize_pd(pv)
                    for pn, pv in el.properties.items()
                },
            }
        for name, rel in self._schema.relations.items():
            d["relations"][name] = {
                "label": rel.label,
                "source_labels": rel.source_labels,
                "target_labels": rel.target_labels,
                "description": rel.description,
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaValidator":
        """Deserialize schema from dict."""
        schema = SchemaDefinition(
            version=data.get("version", "1.0"),
            global_rules=data.get("global_rules", {}),
        )
        for name, el_data in data.get("entity_labels", {}).items():
            props = {}
            for pn, pv in el_data.get("properties", {}).items():
                props[pn] = PropertyDef(
                    name=pn,
                    prop_type=PropertyType(pv.get("type", "string")),
                    cardinality=Cardinality(
                        pv.get("cardinality", "optional")
                    ),
                    required=pv.get("required", False),
                    pattern=pv.get("pattern"),
                )
            schema.entity_labels[name] = EntityLabelDef(
                label=name,
                properties=props,
                primary_key=el_data.get("primary_key", "name"),
                description=el_data.get("description", ""),
            )
        for name, rel_data in data.get("relations", {}).items():
            schema.relations[name] = RelationDef(
                label=name,
                source_labels=rel_data.get("source_labels", []),
                target_labels=rel_data.get("target_labels", []),
                description=rel_data.get("description", ""),
            )
        return cls(schema=schema)

    # ---- Operator Protocol ----

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute schema validation via operator protocol.

        Reads from context:
            extracted_entities:  List of entity dicts with label + properties
            extracted_relations: List of relation dicts
            schema:              Optional SchemaDefinition to use
        Writes to context:
            validation_result:   ValidationResult summary
            validated_entities:  Entities that passed validation
            validation_errors:   List of violations for failed entities
        """
        entities = context.get("extracted_entities", [])
        relations = context.get("extracted_relations", [])
        custom_schema = context.get("schema")
        if custom_schema:
            self._schema = custom_schema

        all_violations = []
        valid_entities = []
        error_entities = []

        for i, ent in enumerate(entities):
            label = ent.get("label", "Entity")
            props = ent.get("properties", ent)
            vr = self.validate_entity(label, props)
            if vr.is_valid:
                valid_entities.append(ent)
            else:
                error_entities.append({
                    "index": i,
                    "entity": ent,
                    "violations": [v.message for v in vr.violations],
                })
            all_violations.extend(vr.violations)

        valid_relations = []
        for i, rel in enumerate(relations):
            rel_label = rel.get("relation_label", rel.get("label", ""))
            src = rel.get("source_label", rel.get("source", ""))
            tgt = rel.get("target_label", rel.get("target", ""))
            props = rel.get("properties", {})
            vr = self.validate_relation(rel_label, src, tgt, props)
            if vr.is_valid:
                valid_relations.append(rel)
            else:
                error_entities.append({
                    "index": i,
                    "entity": rel,
                    "violations": [v.message for v in vr.violations],
                })
            all_violations.extend(vr.violations)

        overall = ValidationResult()
        overall.violations = all_violations
        overall.is_valid = len(overall.errors) == 0

        context["validation_result"] = overall.summary()
        context["validated_entities"] = valid_entities
        context["validated_relations"] = valid_relations
        context["validation_errors"] = error_entities

        return context

    # ---- Internal Helpers ----

    def _validate_property(self, result: ValidationResult,
                           prop_def: PropertyDef,
                           value: Any,
                           context_label: str):
        """Validate a single property value against its definition."""
        # Null/empty check
        if value is None or value == "":
            if prop_def.required:
                result.add_violation(
                    "PROP_010", Severity.ERROR,
                    f"Required property '{prop_def.name}' is null/empty "
                    f"on '{context_label}'",
                    field=prop_def.name
                )
            return

        # Type check
        if not self._check_type(value, prop_def.prop_type):
            result.add_violation(
                "PROP_020", Severity.WARNING,
                f"Property '{prop_def.name}' value type mismatch: "
                f"expected {prop_def.prop_type.value}, "
                f"got {type(value).__name__} on '{context_label}'",
                field=prop_def.name,
                suggested_fix=f"Convert value to {prop_def.prop_type.value}"
            )

        # Pattern check (for strings)
        if prop_def.pattern and isinstance(value, str):
            if not re.match(prop_def.pattern, value):
                result.add_violation(
                    "PROP_030", Severity.WARNING,
                    f"Property '{prop_def.name}' value '{value[:50]}' "
                    f"does not match pattern '{prop_def.pattern}' "
                    f"on '{context_label}'",
                    field=prop_def.name
                )

        # Range check (for numeric)
        if isinstance(value, (int, float)):
            if prop_def.min_value is not None and value < prop_def.min_value:
                result.add_violation(
                    "PROP_040", Severity.WARNING,
                    f"Property '{prop_def.name}' value {value} "
                    f"below minimum {prop_def.min_value} "
                    f"on '{context_label}'",
                    field=prop_def.name
                )
            if prop_def.max_value is not None and value > prop_def.max_value:
                result.add_violation(
                    "PROP_041", Severity.WARNING,
                    f"Property '{prop_def.name}' value {value} "
                    f"exceeds maximum {prop_def.max_value} "
                    f"on '{context_label}'",
                    field=prop_def.name
                )

        # Length check (for strings)
        if isinstance(value, str) and prop_def.max_length:
            if len(value) > prop_def.max_length:
                result.add_violation(
                    "PROP_050", Severity.WARNING,
                    f"Property '{prop_def.name}' length {len(value)} "
                    f"exceeds max {prop_def.max_length} on '{context_label}'",
                    field=prop_def.name,
                    suggested_fix=f"Truncate to {prop_def.max_length} chars"
                )

    @staticmethod
    def _check_type(value: Any, expected: PropertyType) -> bool:
        """Check if value matches expected type."""
        if expected == PropertyType.STRING:
            return isinstance(value, str)
        if expected == PropertyType.TEXT:
            return isinstance(value, str)
        if expected == PropertyType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == PropertyType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == PropertyType.BOOLEAN:
            return isinstance(value, bool)
        if expected == PropertyType.DATE:
            return isinstance(value, str)  # Simplified: accept ISO string
        if expected == PropertyType.DATETIME:
            return isinstance(value, str)
        if expected in (PropertyType.LIST_STRING, PropertyType.LIST_INTEGER):
            return isinstance(value, list)
        return True  # Unknown type, pass

    @staticmethod
    def _try_coerce(value: Any, prop_def: PropertyDef) -> Optional[Any]:
        """Try to coerce a value to the expected type."""
        try:
            if prop_def.prop_type == PropertyType.INTEGER and \
                    not isinstance(value, int):
                return int(value)
            if prop_def.prop_type == PropertyType.FLOAT and \
                    not isinstance(value, (int, float)):
                return float(value)
            if prop_def.prop_type == PropertyType.BOOLEAN and \
                    not isinstance(value, bool):
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
        except (ValueError, TypeError):
            pass
        return None
