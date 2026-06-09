# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# under the Apache License, Version 2.0 (the
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

"""Unit tests for GremlinValidator and GremlinRetryLoop."""

import json
import unittest
from unittest.mock import MagicMock

from hugegraph_llm.operators.llm_op.gremlin_validator import (
    GremlinRetryLoop,
    GremlinValidator,
)

SAMPLE_SCHEMA = """
Vertex labels: Person, Company, Product
Edge labels: works_at, supplies, produces
Properties: name, age, revenue
"""


class TestGremlinValidatorInit(unittest.TestCase):
    def test_default_language(self):
        v = GremlinValidator()
        self.assertEqual(v._language, "en")

    def test_custom_language(self):
        v = GremlinValidator(language="cn")
        self.assertEqual(v._language, "cn")

    def test_llm_can_be_none(self):
        v = GremlinValidator(llm=None)
        self.assertIsNone(v._llm)


class TestGremlinValidatorValidate(unittest.TestCase):
    def test_empty_gremlin_invalid(self):
        v = GremlinValidator(llm=None)
        result = v.validate("", SAMPLE_SCHEMA)
        self.assertFalse(result["valid"])
        self.assertTrue(len(result["issues"]) > 0)

    def test_none_gremlin_invalid(self):
        v = GremlinValidator(llm=None)
        result = v.validate(None, SAMPLE_SCHEMA)
        self.assertFalse(result["valid"])

    def test_whitespace_gremlin_invalid(self):
        v = GremlinValidator(llm=None)
        result = v.validate("   ", SAMPLE_SCHEMA)
        self.assertFalse(result["valid"])

    def test_no_llm_optimistic_pass(self):
        v = GremlinValidator(llm=None)
        result = v.validate("g.V().has('name','test')", SAMPLE_SCHEMA)
        self.assertTrue(result["valid"])

    def test_valid_gremlin(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "valid": True, "issues": [], "fixed_query": ""
        })
        v = GremlinValidator(llm=mock_llm)
        result = v.validate("g.V().has('name','test')", SAMPLE_SCHEMA)
        self.assertTrue(result["valid"])

    def test_invalid_gremlin_with_issues(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "valid": False,
            "issues": ["Unknown vertex label 'Foo'"],
            "fixed_query": "g.V().has('name','test')"
        })
        v = GremlinValidator(llm=mock_llm)
        result = v.validate("g.V('Foo')", SAMPLE_SCHEMA)
        self.assertFalse(result["valid"])
        self.assertEqual(len(result["issues"]), 1)

    def test_json_in_code_block(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = (
            '```json\n{"valid": false, "issues": ["syntax error"], "fixed_query": ""}\n```'
        )
        v = GremlinValidator(llm=mock_llm)
        result = v.validate("bad query", SAMPLE_SCHEMA)
        self.assertFalse(result["valid"])

    def test_malformed_json_fallback(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not json at all"
        v = GremlinValidator(llm=mock_llm)
        result = v.validate("g.V()", SAMPLE_SCHEMA)
        self.assertTrue(result["valid"])  # optimistic fallback

    def test_llm_error_fallback(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API error")
        v = GremlinValidator(llm=mock_llm)
        result = v.validate("g.V()", SAMPLE_SCHEMA)
        self.assertTrue(result["valid"])  # optimistic fallback

    def test_cn_language_uses_cn_prompt(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "valid": True, "issues": [], "fixed_query": ""
        })
        v = GremlinValidator(llm=mock_llm, language="cn")
        v.validate("g.V()", SAMPLE_SCHEMA)
        call_args = mock_llm.generate.call_args
        prompt = call_args.kwargs.get("prompt", "")
        self.assertIn("验证", prompt)


class TestGremlinRetryLoopInit(unittest.TestCase):
    def test_default_max_retries(self):
        loop = GremlinRetryLoop()
        self.assertEqual(loop._max_retries, 3)

    def test_custom_max_retries(self):
        loop = GremlinRetryLoop(max_retries=5)
        self.assertEqual(loop._max_retries, 5)

    def test_max_retries_clamped_upper(self):
        loop = GremlinRetryLoop(max_retries=10)
        self.assertEqual(loop._max_retries, 5)

    def test_max_retries_clamped_lower(self):
        loop = GremlinRetryLoop(max_retries=0)
        self.assertEqual(loop._max_retries, 1)


class TestExtractGremlin(unittest.TestCase):
    def test_extract_from_code_block(self):
        loop = GremlinRetryLoop()
        result = loop._extract_gremlin(
            'Here is the query:\n```gremlin\ng.V().has("name", "test")\n```\nDone.'
        )
        self.assertEqual(result, 'g.V().has("name", "test")')

    def test_extract_no_code_block(self):
        loop = GremlinRetryLoop()
        result = loop._extract_gremlin("g.V().has('name','test')")
        self.assertEqual(result, "g.V().has('name','test')")

    def test_extract_strips_whitespace(self):
        loop = GremlinRetryLoop()
        result = loop._extract_gremlin("  g.V()  ")
        self.assertEqual(result, "g.V()")


class TestIsEmptyResult(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertTrue(GremlinRetryLoop._is_empty_result(None))

    def test_empty_list_is_empty(self):
        self.assertTrue(GremlinRetryLoop._is_empty_result([]))

    def test_nonempty_list_not_empty(self):
        self.assertFalse(GremlinRetryLoop._is_empty_result([1, 2]))

    def test_empty_dict_data_is_empty(self):
        self.assertTrue(GremlinRetryLoop._is_empty_result({"data": []}))

    def test_nonempty_dict_data_not_empty(self):
        self.assertFalse(GremlinRetryLoop._is_empty_result({"data": [{"name": "test"}]}))

    def test_dict_without_data_key_not_empty(self):
        self.assertFalse(GremlinRetryLoop._is_empty_result({"error": "test"}))


class TestGenerateAndExecute(unittest.TestCase):
    def test_success_on_first_attempt(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '```gremlin\ng.V().has("name", "test")\n```',
        ]
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": [{"name": "test"}]}

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person named test")

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(len(result["history"]), 0)

    def test_retry_on_validation_failure_then_success(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '```gremlin\ng.V("Foo")\n```',          # attempt 1: bad label
            '```gremlin\ng.V().has("name","test")\n```',  # attempt 2: fixed
        ]
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            {"valid": False, "issues": ["Unknown label Foo"]},  # validation fail
            {"valid": True, "issues": []},                        # validation pass
        ]
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": [{"name": "test"}]}

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person named test")

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["history"][0]["status"], "validation_failed")

    def test_retry_on_execution_failure_then_success(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '```gremlin\ng.V().has("name","test")\n```',
            '```gremlin\ng.V().has("name","test2")\n```',
        ]
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.side_effect = [
            Exception("Syntax error"),
            {"data": [{"name": "test2"}]},
        ]

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person")

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["history"][0]["status"], "execution_failed")

    def test_all_retries_exhausted_fallback(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            'bad gremlin 1',
            'bad gremlin 2',
            'bad gremlin 3',
        ]
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {
            "valid": False, "issues": ["error"]
        }

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=None,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person")

        self.assertFalse(result["success"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["fallback"], "bfs")
        self.assertEqual(len(result["history"]), 3)

    def test_empty_result_triggers_retry(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            '```gremlin\ng.V().has("age", 999)\n```',
            '```gremlin\ng.V().has("name","test")\n```',
        ]
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.side_effect = [
            {"data": []},  # empty result → retry
            {"data": [{"name": "test"}]},
        ]

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person")

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)

    def test_generation_failure_continues(self):
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = [
            Exception("LLM error"),
            '```gremlin\ng.V().has("name","test")\n```',
        ]
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": [{"name": "test"}]}

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=3,
        )
        result = loop.generate_and_execute("Find person")

        self.assertTrue(result["success"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["history"][0]["status"], "generation_failed")

    def test_no_client_returns_none_result(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '```gremlin\ng.V()\n```'
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=None,
            schema=SAMPLE_SCHEMA, max_retries=1,
        )
        result = loop.generate_and_execute("Find person")

        # No client → execute returns None → treated as empty → fallback
        self.assertFalse(result["success"])
        self.assertEqual(result["fallback"], "bfs")


class TestRunOperator(unittest.TestCase):
    def test_run_writes_to_context(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '```gremlin\ng.V()\n```'
        mock_validator = MagicMock()
        mock_validator.validate.return_value = {"valid": True, "issues": []}
        mock_client = MagicMock()
        mock_client.gremlin.return_value.exec.return_value = {"data": [{"name": "test"}]}

        loop = GremlinRetryLoop(
            llm=mock_llm, validator=mock_validator, graph_client=mock_client,
            schema=SAMPLE_SCHEMA, max_retries=1,
        )
        context = {"query": "Find test", "call_count": 0}
        result = loop.run(context)

        self.assertTrue(result["gremlin_retry_result"]["success"])
        self.assertEqual(result["call_count"], 1)

    def test_run_empty_query(self):
        loop = GremlinRetryLoop()
        context = {"query": ""}
        result = loop.run(context)

        self.assertFalse(result["gremlin_retry_result"]["success"])
        self.assertEqual(result["gremlin_retry_result"]["fallback"], "no_query")


if __name__ == "__main__":
    unittest.main()
