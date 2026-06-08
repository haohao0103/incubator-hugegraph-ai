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
Gremlin query validator with retry loop for self-correction.

Implements Sprint 5: Text2Gremlin self-correction.
When an LLM-generated Gremlin query fails, this module:
1. Validates the Gremlin query (syntax + schema alignment)
2. If validation fails, feeds the error back to the LLM
3. Retries generation (up to MAX_RETRIES)
4. Falls back to BFS subgraph traversal if all retries fail

This significantly reduces the BFS fallback rate for Text2Gremlin.
"""

import json
import re
from typing import Any, Dict, List, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log

VALIDATE_PROMPT = (
    "Validate the following Gremlin query for correctness.\n\n"
    "## Graph Schema\n{schema}\n\n"
    "## Gremlin Query\n{gremlin}\n\n"
    "## Checks\n"
    "1. Is the Gremlin syntax correct?\n"
    "2. Do the vertex labels, edge labels, and properties exist in the schema?\n"
    "3. Is the query logically sound (correct traversal direction)?\n\n"
    "Output ONLY valid JSON:\n"
    '{{"valid": true/false, "issues": ["issue1", "issue2"], '
    '"fixed_query": "optional corrected gremlin"}}'
)

VALIDATE_PROMPT_CN = (
    "验证以下 Gremlin 查询的正确性。\n\n"
    "## 图 Schema\n{schema}\n\n"
    "## Gremlin 查询\n{gremlin}\n\n"
    "## 检查项\n"
    "1. Gremlin 语法是否正确？\n"
    "2. 引用的顶点标签、边标签和属性是否存在于 Schema 中？\n"
    "3. 查询逻辑是否正确（遍历方向等）？\n\n"
    "仅输出 JSON：\n"
    '{{"valid": true/false, "issues": ["问题1", "问题2"], '
    '"fixed_query": "可选的修正后 gremlin"}}'
)


class GremlinValidator:
    """LLM-driven Gremlin query validator.

    Checks:
    1. Syntax correctness
    2. Schema alignment (labels and properties exist)
    3. Logical soundness (traversal direction, etc.)
    """

    def __init__(self, llm: Optional[BaseLLM] = None, language: str = "en"):
        self._llm = llm
        self._language = language

    def validate(self, gremlin: str, schema: str) -> Dict[str, Any]:
        """Validate a Gremlin query against the schema.

        :param gremlin: The Gremlin query to validate.
        :param schema: The graph schema string.
        :return: Dict with "valid" (bool), "issues" (list), "fixed_query" (str).
        """
        if not gremlin or not gremlin.strip():
            return {"valid": False, "issues": ["Empty Gremlin query"], "fixed_query": ""}

        if not self._llm:
            # No LLM available — optimistic pass-through
            return {"valid": True, "issues": [], "fixed_query": gremlin}

        prompt = VALIDATE_PROMPT_CN if self._language == "cn" else VALIDATE_PROMPT
        prompt_text = prompt.format(schema=schema, gremlin=gremlin)

        try:
            response = self._llm.generate(prompt=prompt_text)
            return self._parse_validation_response(response)
        except Exception as e:
            log.warning("Gremlin validation failed: %s", e)
            return {"valid": True, "issues": [], "fixed_query": gremlin}

    @staticmethod
    def _parse_validation_response(response: str) -> Dict[str, Any]:
        """Parse the LLM validation response as JSON."""
        text = response.strip()

        # Direct JSON
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "valid" in result:
                return {
                    "valid": bool(result.get("valid", True)),
                    "issues": result.get("issues", []),
                    "fixed_query": result.get("fixed_query", ""),
                }
        except (json.JSONDecodeError, TypeError):
            pass

        # JSON in code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, dict) and "valid" in result:
                    return {
                        "valid": bool(result.get("valid", True)),
                        "issues": result.get("issues", []),
                        "fixed_query": result.get("fixed_query", ""),
                    }
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: optimistic
        return {"valid": True, "issues": [], "fixed_query": ""}


class GremlinRetryLoop:
    """Self-correcting Text2Gremlin with retry loop.

    Flow: Generate → Validate → Execute → Fail? → Feedback → Retry

    Usage::

        retry_loop = GremlinRetryLoop(
            llm=my_llm,
            validator=GremlinValidator(llm=my_llm),
            graph_client=my_client,
            schema=my_schema,
        )
        result = retry_loop.generate_and_execute("Find all suppliers of part A")
        if result["success"]:
            print(result["result"])
        else:
            print("Fallback to BFS")
    """

    MAX_RETRIES = 3

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        validator: Optional[GremlinValidator] = None,
        graph_client: Optional[Any] = None,
        schema: Optional[str] = None,
        gremlin_prompt: Optional[str] = None,
        max_retries: int = 3,
        language: str = "en",
    ):
        self._llm = llm
        self._validator = validator
        self._graph_client = graph_client
        self._schema = schema or ""
        self._gremlin_prompt = gremlin_prompt
        self._max_retries = max(1, min(max_retries, 5))
        self._language = language

    def _get_llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = LLMs().get_text2gql_llm()
        return self._llm

    def _get_validator(self) -> GremlinValidator:
        if self._validator is None:
            self._validator = GremlinValidator(llm=self._llm, language=self._language)
        return self._validator

    def generate_and_execute(self, query: str) -> Dict[str, Any]:
        """Generate a Gremlin query with self-correction retry loop.

        :param query: Natural language query.
        :return: Dict with success, gremlin, result, attempts, history, fallback.
        """
        llm = self._get_llm()
        validator = self._get_validator()

        # Build initial prompt
        prompt = self._build_prompt(query)
        history: List[Dict[str, Any]] = []

        for attempt in range(1, self._max_retries + 1):
            # Step 1: Generate Gremlin
            try:
                gremlin = llm.generate(prompt=prompt)
                gremlin = self._extract_gremlin(gremlin)
            except Exception as e:
                log.error("Gremlin generation failed (attempt %d): %s", attempt, e)
                history.append({
                    "attempt": attempt,
                    "gremlin": None,
                    "status": "generation_failed",
                    "error": str(e),
                })
                continue

            # Step 2: Validate
            validation = validator.validate(gremlin, self._schema)
            if not validation["valid"]:
                issues_str = "; ".join(validation.get("issues", []))
                log.warning("Gremlin validation failed (attempt %d): %s", attempt, issues_str)
                prompt += (
                    f"\n\nAttempt {attempt}: Generated Gremlin failed validation.\n"
                    f"Gremlin: {gremlin}\n"
                    f"Issues: {issues_str}\n"
                    f"Please fix the Gremlin query and regenerate."
                )
                history.append({
                    "attempt": attempt,
                    "gremlin": gremlin,
                    "status": "validation_failed",
                    "error": issues_str,
                })
                continue

            # Step 3: Execute
            try:
                result = self._execute_gremlin(gremlin)
                if result is not None and not self._is_empty_result(result):
                    return {
                        "success": True,
                        "gremlin": gremlin,
                        "result": result,
                        "attempts": attempt,
                        "history": history,
                    }
                else:
                    error_msg = "Query returned empty result"
            except Exception as e:
                error_msg = str(e)

            # Step 4: Feedback and retry
            log.warning("Gremlin execution failed (attempt %d): %s", attempt, error_msg)
            prompt += (
                f"\n\nAttempt {attempt}: Generated Gremlin failed execution.\n"
                f"Gremlin: {gremlin}\n"
                f"Error: {error_msg}\n"
                f"Please analyze the error and generate a corrected Gremlin query."
            )
            history.append({
                "attempt": attempt,
                "gremlin": gremlin,
                "status": "execution_failed",
                "error": error_msg,
            })

        # All retries exhausted → fallback
        return {
            "success": False,
            "gremlin": None,
            "result": None,
            "attempts": self._max_retries,
            "history": history,
            "fallback": "bfs",
        }

    def _build_prompt(self, query: str) -> str:
        """Build the initial Text2Gremlin prompt."""
        if self._gremlin_prompt:
            return self._gremlin_prompt.format(
                query=query,
                schema=self._schema,
            )
        # Default prompt
        return (
            f"Convert the following natural language question to a Gremlin query.\n\n"
            f"## Graph Schema\n{self._schema}\n\n"
            f"## Question\n{query}\n\n"
            f"## Instructions\n"
            f"Generate ONLY the Gremlin query. Use HugeGraph Gremlin dialect.\n"
            f"Output the query in a ```gremlin``` code block.\n\n"
            f"Gremlin:"
        )

    def _extract_gremlin(self, response: str) -> str:
        """Extract Gremlin query from LLM response."""
        # Try code block first
        match = re.search(r"```gremlin\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return full response stripped
        return response.strip()

    def _execute_gremlin(self, gremlin: str) -> Any:
        """Execute a Gremlin query against the graph."""
        if self._graph_client is None:
            return None
        result = self._graph_client.gremlin(gremlin).exec()
        return result

    @staticmethod
    def _is_empty_result(result: Any) -> bool:
        """Check if a query result is empty."""
        if result is None:
            return True
        if isinstance(result, dict):
            if "data" in result:
                data = result["data"]
                if isinstance(data, list) and len(data) == 0:
                    return True
            return False
        if isinstance(result, list):
            return len(result) == 0
        return False

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run as an operator (context in → context out)."""
        query = context.get("query", "")
        if not query:
            context["gremlin_retry_result"] = {
                "success": False,
                "gremlin": None,
                "result": None,
                "attempts": 0,
                "history": [],
                "fallback": "no_query",
            }
            return context

        result = self.generate_and_execute(query)
        context["gremlin_retry_result"] = result
        context["call_count"] = context.get("call_count", 0) + result["attempts"]
        return context
