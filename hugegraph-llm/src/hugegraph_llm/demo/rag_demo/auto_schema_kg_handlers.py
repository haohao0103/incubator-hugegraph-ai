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
    BatchAutoSchemaKGOperator,
    SchemaDraft,
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


def generate_batch_schema_draft(
    documents_text: str,
    instructions: str = "",
    document_separator: str = "\n\n",
) -> Tuple[str, str, str]:
    """Generate a merged schema draft from multiple documents.

    Args:
        documents_text: Text containing one or more documents separated by ``document_separator``.
        instructions: Optional domain guidance.
        document_separator: String used to split ``documents_text`` into separate documents.

    Returns:
        A tuple of (markdown_preview, schema_json, status_message).
    """
    if not documents_text or not documents_text.strip():
        return (
            "",
            "",
            "Error: please provide one or more non-empty documents.",
        )

    documents = [doc.strip() for doc in documents_text.split(document_separator) if doc.strip()]
    if not documents:
        documents = [documents_text.strip()]

    try:
        llm = LLMs().get_extract_llm()
        batch = BatchAutoSchemaKGOperator(
            llm=llm,
            schema_commit_client=None,
            review_callback=None,
            allow_commit=False,
            instructions=instructions or "",
        )
        result = batch.run(documents)
        _LATEST_DRAFT["draft"] = result.merged_draft

        schema_json = json.dumps(result.merged_draft.to_schema_dict(), ensure_ascii=False, indent=2)
        status_lines = [
            f"Merged schema from {len(result.per_document_results)} document(s).",
        ]
        if result.conflicts:
            status_lines.append(f"Detected {len(result.conflicts)} conflict(s); review before commit.")
        else:
            status_lines.append("No conflicts detected.")

        return (
            result.merged_draft.to_human_readable(),
            schema_json,
            "\n".join(status_lines),
        )
    except Exception as e:  # pylint: disable=broad-except
        log.error("Batch AutoSchemaKG generate failed: %s", e)
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
