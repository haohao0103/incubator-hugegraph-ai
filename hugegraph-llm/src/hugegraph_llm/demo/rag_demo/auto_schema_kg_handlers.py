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

"""Handlers for the AutoSchemaKG Gradio tab.

Generate a HugeGraph schema draft from a document and commit it after review.
"""

import json
from typing import Any, Dict, Tuple

from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.operators.hugegraph_op.commit_to_hugegraph import Commit2Graph
from hugegraph_llm.operators.llm_op.auto_schema_kg import (
    AutoSchemaKGOperator,
    SchemaReviewResult,
)
from hugegraph_llm.utils.log import log


# Cache the latest generated draft so the approve step can use it without
# round-tripping through the JSON editor.  This is a module-level cache keyed
# by a session token (empty in the simple single-user demo).
_LATEST_DRAFT: Dict[str, Any] = {}


def generate_schema_draft(
    document: str,
    instructions: str = "",
    auto_commit: bool = False,
) -> Tuple[str, str, str]:
    """Generate a schema draft from ``document``.

    Returns:
        A tuple of (markdown_preview, schema_json, status_message).
    """
    if not document or not document.strip():
        return (
            "",
            "",
            "Error: please provide a non-empty document.",
        )
    try:
        llm = LLMs().get_extract_llm()
        operator = AutoSchemaKGOperator(
            llm=llm,
            schema_commit_client=None,
            review_callback=None,
            allow_commit=False,
            instructions=instructions or "",
        )
        result = operator.run(document)
        _LATEST_DRAFT["draft"] = result.draft
        schema_json = json.dumps(result.draft.to_schema_dict(), ensure_ascii=False, indent=2)
        return (
            result.draft.to_human_readable(),
            schema_json,
            "Schema draft generated. Review the Markdown and JSON, then click approve to commit.",
        )
    except Exception as e:  # pylint: disable=broad-except
        log.error("AutoSchemaKG generate failed: %s", e)
        return (
            "",
            "",
            f"Error: {e}",
        )


def approve_and_commit(schema_json: str) -> Tuple[str, str]:
    """Approve the schema and commit it to HugeGraph.

    ``schema_json`` is the current content of the JSON editor. If it has been
    edited by the user, the edited version is committed. If it is empty, the
    cached draft from the last generate call is used.
    """
    draft = None
    if schema_json and schema_json.strip():
        try:
            schema_dict = json.loads(schema_json)
        except json.JSONDecodeError as e:
            return ("", f"Error: invalid JSON schema - {e}")
        from hugegraph_llm.operators.llm_op.auto_schema_kg import (
            EdgeLabelDef,
            PropertyKeyDef,
            SchemaDraft,
            VertexLabelDef,
        )

        try:
            draft = SchemaDraft(
                property_keys=[PropertyKeyDef(**p) for p in schema_dict.get("propertykeys", [])],
                vertex_labels=[VertexLabelDef(**v) for v in schema_dict.get("vertexlabels", [])],
                edge_labels=[EdgeLabelDef(**e) for e in schema_dict.get("edgelabels", [])],
            )
        except Exception as e:  # pylint: disable=broad-except
            return ("", f"Error: cannot parse schema dict - {e}")
    elif _LATEST_DRAFT.get("draft") is not None:
        draft = _LATEST_DRAFT["draft"]
    else:
        return ("", "Error: no schema draft available. Please generate a draft first.")

    try:
        commit_client = Commit2Graph()
        operator = AutoSchemaKGOperator(
            llm=LLMs().get_extract_llm(),
            schema_commit_client=commit_client,
            review_callback=lambda _d: SchemaReviewResult(approved=True),
            allow_commit=True,
        )
        # operator.run() would regenerate the draft from a document. We already
        # have the draft, so we reuse the review/commit path directly.
        review = SchemaReviewResult(approved=True)
        committed, commit_error = operator._commit_if_allowed(review.effective_schema(draft))  # pylint: disable=protected-access
        if committed:
            return (
                draft.to_human_readable(),
                f"Committed successfully to HugeGraph. {len(draft.vertex_labels)} vertex labels, "
                f"{len(draft.edge_labels)} edge labels.",
            )
        return (draft.to_human_readable(), f"Commit failed: {commit_error}")
    except Exception as e:  # pylint: disable=broad-except
        log.error("AutoSchemaKG commit failed: %s", e)
        return ("", f"Error: {e}")


def reset_schema_draft() -> Tuple[str, str, str]:
    """Clear the cached draft and editor content."""
    _LATEST_DRAFT.clear()
    return ("", "", "Draft cleared.")
