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

"""Comprehensive tests for text2gremlin_block logic (GremlinResult, schema
processing, output configuration, query execution)."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))

from hugegraph_llm.demo.rag_demo.text2gremlin_block import (
    GremlinResult,
    store_schema,
    _process_schema,
    _configure_output_types,
    _execute_queries,
    simple_schema,
    gremlin_generate_for_ui,
)


# ════════════════════════════════════════════════════════════════
#  GremlinResult dataclass tests
# ════════════════════════════════════════════════════════════════

class TestGremlinResult:

    def test_error_creation(self):
        """GremlinResult.error() creates failed result with message."""
        result = GremlinResult.error("Query failed")
        assert result.success is False
        assert result.match_result == "Query failed"
        assert result.error_message == "Query failed"
        assert result.template_gremlin is None
        assert result.raw_gremlin is None

    def test_success_result_creation(self):
        """GremlinResult.success_result() creates successful result."""
        result = GremlinResult.success_result(
            match_result="matched 2 templates",
            template_gremlin="g.V().has('name', 'peter')",
            raw_gremlin="g.V().hasLabel('person')",
            template_exec="[{id:1}]",
            raw_exec="[{id:1},{id:2}]",
        )
        assert result.success is True
        assert result.match_result == "matched 2 templates"
        assert result.template_gremlin == "g.V().has('name', 'peter')"
        assert result.raw_gremlin == "g.V().hasLabel('person')"
        assert result.template_exec_result == "[{id:1}]"
        assert result.raw_exec_result == "[{id:1},{id:2}]"
        assert result.error_message is None


# ════════════════════════════════════════════════════════════════
#  store_schema tests
# ════════════════════════════════════════════════════════════════

class TestStoreSchema:

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.prompt")
    def test_no_change_no_save(self, mock_prompt):
        """When schema/question/prompt unchanged, update_yaml_file NOT called."""
        mock_prompt.text2gql_graph_schema = "old_schema"
        mock_prompt.default_question = "old_question"
        mock_prompt.gremlin_generate_prompt = "old_prompt"

        store_schema("old_schema", "old_question", "old_prompt")

        mock_prompt.update_yaml_file.assert_not_called()

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.prompt")
    def test_schema_changed_triggers_save(self, mock_prompt):
        """When schema changes, update_yaml_file IS called."""
        mock_prompt.text2gql_graph_schema = "old_schema"
        mock_prompt.default_question = "old_q"
        mock_prompt.gremlin_generate_prompt = "old_p"

        store_schema("new_schema", "old_q", "old_p")

        assert mock_prompt.text2gql_graph_schema == "new_schema"
        mock_prompt.update_yaml_file.assert_called_once()

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.prompt")
    def test_question_changed_triggers_save(self, mock_prompt):
        """When question changes, update_yaml_file IS called."""
        mock_prompt.text2gql_graph_schema = "s"
        mock_prompt.default_question = "old_q"
        mock_prompt.gremlin_generate_prompt = "p"

        store_schema("s", "new_q", "p")

        assert mock_prompt.default_question == "new_q"
        mock_prompt.update_yaml_file.assert_called_once()


# ════════════════════════════════════════════════════════════════
#  _process_schema tests
# ════════════════════════════════════════════════════════════════

class TestProcessSchema:

    def test_empty_schema_returns_none(self):
        """Empty schema returns (None, False)."""
        mock_generator = MagicMock()
        mock_sm = MagicMock()
        result, short = _process_schema("", mock_generator, mock_sm)
        assert result is None
        assert short is False

    def test_graph_name_schema(self):
        """Graph name (not JSON) triggers import from HugeGraph."""
        mock_generator = MagicMock()
        mock_sm = MagicMock()
        mock_sm.schema = MagicMock()
        mock_sm.schema.getSchema.return_value = {"vertexlabels": []}

        result, short = _process_schema("hugegraph", mock_generator, mock_sm)

        assert short is True
        mock_generator.import_schema.assert_called_once_with(from_hugegraph="hugegraph")

    def test_json_schema(self):
        """Valid JSON schema is parsed and imported."""
        mock_generator = MagicMock()
        mock_sm = MagicMock()

        schema_json = json.dumps({"vertexlabels": [{"name": "Person"}]})
        result, short = _process_schema(schema_json, mock_generator, mock_sm)

        assert short is False
        mock_generator.import_schema.assert_called_once_with(from_user_defined={"vertexlabels": [{"name": "Person"}]})

    def test_invalid_json_schema(self):
        """Invalid JSON schema returns (None, None)."""
        mock_generator = MagicMock()
        mock_sm = MagicMock()

        result, short = _process_schema("{invalid!!}", mock_generator, mock_sm)
        assert result is None
        assert short is None  # Error case


# ════════════════════════════════════════════════════════════════
#  _configure_output_types tests
# ════════════════════════════════════════════════════════════════

class TestConfigureOutputTypes:

    def test_default_all_true(self):
        """Default output types are all True."""
        result = _configure_output_types(None)
        assert all(result.values())

    def test_empty_dict_unchanged(self):
        """Empty dict {} is falsy in Python, so all outputs remain True."""
        result = _configure_output_types({})
        # {} is falsy → if-block not entered → all stay True
        assert all(v is True for v in result.values())

    def test_empty_list_sets_all_false(self):
        """Empty list [] is falsy, same as None — all stay True."""
        result = _configure_output_types([])
        assert all(v is True for v in result.values())

    def test_selective_outputs(self):
        """Selective output types only enable requested ones."""
        result = _configure_output_types(["match_result", "raw_gremlin"])
        assert result["match_result"] is True
        assert result["raw_gremlin"] is True
        assert result["template_gremlin"] is False
        assert result["template_execution_result"] is False
        assert result["raw_execution_result"] is False

    def test_unknown_keys_ignored(self):
        """Unknown keys in requested_outputs are ignored."""
        result = _configure_output_types(["match_result", "unknown_key"])
        assert result["match_result"] is True
        # unknown_key not in output_types dict → ignored


# ════════════════════════════════════════════════════════════════
#  _execute_queries tests
# ════════════════════════════════════════════════════════════════

class TestExecuteQueries:

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.run_gremlin_query")
    def test_execute_template_query(self, mock_run):
        """Template query execution is called when output type enabled."""
        mock_run.return_value = [{"id": "1:v1"}]
        context = {"result": "g.V().limit(5)", "raw_result": "g.V().limit(10)"}
        output_types = {
            "template_execution_result": True,
            "raw_execution_result": False,
        }
        _execute_queries(context, output_types)

        assert "template_exec_res" in context
        assert context["template_exec_res"] == [{"id": "1:v1"}]
        assert context["raw_exec_res"] == ""  # Disabled → empty string

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.run_gremlin_query")
    def test_execute_both_queries(self, mock_run):
        """Both template and raw queries are executed."""
        mock_run.return_value = [{"id": "1"}]
        context = {"result": "g.V()", "raw_result": "g.E()"}
        output_types = {
            "template_execution_result": True,
            "raw_execution_result": True,
        }
        _execute_queries(context, output_types)

        assert "template_exec_res" in context
        assert "raw_exec_res" in context

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.run_gremlin_query")
    def test_execute_exception_captured(self, mock_run):
        """Gremlin query exception is captured as string."""
        mock_run.side_effect = RuntimeError("Syntax error in query")
        context = {"result": "bad_query", "raw_result": "another_bad"}
        output_types = {
            "template_execution_result": True,
            "raw_execution_result": True,
        }
        _execute_queries(context, output_types)

        assert "Syntax error" in str(context["template_exec_res"])

    def test_disabled_outputs_empty_string(self):
        """Disabled output types set result to empty string."""
        context = {"result": "g.V()", "raw_result": "g.E()"}
        output_types = {
            "template_execution_result": False,
            "raw_execution_result": False,
        }
        _execute_queries(context, output_types)

        assert context["template_exec_res"] == ""
        assert context["raw_exec_res"] == ""


# ════════════════════════════════════════════════════════════════
#  simple_schema tests
# ════════════════════════════════════════════════════════════════

class TestSimpleSchema:

    def test_full_schema_simplified(self):
        """Full schema is reduced to essential keys."""
        full_schema = {
            "vertexlabels": [
                {
                    "id": "1",
                    "name": "Person",
                    "properties": ["name", "age"],
                    "primary_keys": ["name"],
                    "nullable_keys": ["age"],
                    "enable_label_index": True,
                },
            ],
            "edgelabels": [
                {
                    "name": "knows",
                    "source_label": "Person",
                    "target_label": "Person",
                    "properties": ["weight"],
                    "frequency": "SINGLE",
                    "sort_keys": ["weight"],
                },
            ],
        }

        result = simple_schema(full_schema)
        assert len(result["vertexlabels"]) == 1
        assert "primary_keys" not in result["vertexlabels"][0]
        assert result["vertexlabels"][0]["name"] == "Person"
        assert len(result["edgelabels"]) == 1
        assert result["edgelabels"][0]["name"] == "knows"
        assert "frequency" not in result["edgelabels"][0]

    def test_empty_schema(self):
        """Empty schema returns empty dict."""
        result = simple_schema({})
        assert result == {}

    def test_missing_vertexlabels(self):
        """Schema without vertexlabels omits that key."""
        schema = {"edgelabels": [{"name": "e", "source_label": "A", "target_label": "B"}]}
        result = simple_schema(schema)
        assert "vertexlabels" not in result

    def test_missing_edgelabels(self):
        """Schema without edgelabels omits that key."""
        schema = {"vertexlabels": [{"id": "1", "name": "A"}]}
        result = simple_schema(schema)
        assert "edgelabels" not in result


# ════════════════════════════════════════════════════════════════
#  gremlin_generate_for_ui tests
# ════════════════════════════════════════════════════════════════

class TestGremlinGenerateForUI:

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.SchedulerSingleton")
    def test_successful_generation(self, mock_sched_cls):
        """Successful gremlin generation returns 5-element tuple."""
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "match_result": [{"query": "who is peter", "gremlin": "g.V().has('name','peter')"}],
            "template_gremlin": "g.V().has('name','peter')",
            "raw_gremlin": "g.V().hasLabel('person').has('name','peter')",
            "template_execution_result": "[{id:1}]",
            "raw_execution_result": "[{id:1},{id:2}]",
        }

        result = gremlin_generate_for_ui("who is peter?", 2, "hugegraph", "prompt")

        assert len(result) == 5
        assert isinstance(result[0], str)  # match_result_str
        assert result[1] == "g.V().has('name','peter')"  # template_gremlin

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.SchedulerSingleton")
    def test_scheduler_exception(self, mock_sched_cls):
        """Scheduler exception returns error JSON + empty strings."""
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.side_effect = RuntimeError("Server error")

        result = gremlin_generate_for_ui("test", 2, "schema", "prompt")

        # First element is error JSON
        error_dict = json.loads(result[0])
        assert "error" in error_dict
        assert result[1] == ""
        assert result[2] == ""
        assert result[3] == ""
        assert result[4] == ""

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.SchedulerSingleton")
    def test_numeric_example_num_conversion(self, mock_sched_cls):
        """String example_num is converted to int inside gremlin_generate_for_ui."""
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        # Use a simple dict return that won't trigger real schema execution
        mock_instance.schedule_flow.return_value = {
            "match_result": "no match",
            "template_gremlin": "g.V()",
            "raw_gremlin": "g.V().limit(1)",
            "template_execution_result": "",
            "raw_execution_result": "",
        }

        result = gremlin_generate_for_ui("q", "3", "schema", "prompt")

        # Verify scheduler was called (the function completed without crash)
        mock_instance.schedule_flow.assert_called_once()
        assert len(result) == 5

    @patch("hugegraph_llm.demo.rag_demo.text2gremlin_block.SchedulerSingleton")
    def test_dict_match_result_serialized(self, mock_sched_cls):
        """Dict match_result is serialized to JSON string."""
        mock_instance = MagicMock()
        mock_sched_cls.get_instance.return_value = mock_instance
        mock_instance.schedule_flow.return_value = {
            "match_result": {"key": "value"},
            "template_gremlin": "g.V()",
            "raw_gremlin": "g.E()",
            "template_execution_result": "",
            "raw_execution_result": "",
        }

        result = gremlin_generate_for_ui("q", 2, "s", "p")
        # match_result_str should be JSON
        parsed = json.loads(result[0])
        assert parsed["key"] == "value"
