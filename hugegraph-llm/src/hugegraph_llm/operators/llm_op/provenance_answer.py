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

"""Provenance-aware answer synthesis with source citations.

Extends the existing AnswerSynthesize to append document citations
at the bottom of answers, tracing entities back to their source chunks.
"""

from typing import Any, Dict, List, Optional

from hugegraph_llm.config import prompt
from hugegraph_llm.operators.hugegraph_op.provenance_manager import (
    ProvenanceManager,
    ProvenanceRecord,
)
from hugegraph_llm.utils.log import log


class ProvenanceAnswerSynthesize:
    """Generate answers with source citations from the provenance chain.

    This extends the standard AnswerSynthesize by:
    1. Calling the LLM for answer generation (with provenance-aware prompt)
    2. Querying the provenance chain for entities mentioned in the answer
    3. Appending a "## Sources" section with citations

    Usage:
        synth = ProvenanceAnswerSynthesize(llm=chat_llm, provenance_manager=pm)
        context = synth.run(context)
    """

    CITATION_PROMPT_TEMPLATE = """You are an expert in knowledge graphs and natural language processing.

Answer the following query based on the provided context. After your answer,
include a brief "## Sources" section listing the documents and text passages
that support your answer.

Context information:
---------------------
{context_str}
---------------------
Query: {query_str}

Answer with citations to the source documents.
"""

    def __init__(
        self,
        llm: Any = None,
        provenance_manager: ProvenanceManager = None,
        max_citations: int = 5,
        max_citation_text_len: int = 300,
    ):
        """Initialize the provenance-aware answer synthesizer.

        Args:
            llm: LLM instance for answer generation.
            provenance_manager: ProvenanceManager for querying source chains.
            max_citations: Maximum number of citations to include.
            max_citation_text_len: Max length of each citation text.
        """
        self._llm = llm
        self._pm = provenance_manager
        self._max_citations = max_citations
        self._max_citation_text_len = max_citation_text_len

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a provenance-aware answer.

        Reads from context:
            query: The user's question.
            graph_result: Graph context (may contain entity references).
            vector_result: Vector search context.
            match_vids: Matched vertex IDs.
            answer_prompt: Optional custom prompt template.

        Writes to context:
            answer: The answer with citations appended.
            citations: List of citation strings.
            provenance_records: Raw provenance records used.
        """
        query = context.get("query", "")
        graph_result = context.get("graph_result", "")
        vector_result = context.get("vector_result", "")
        answer_prompt_tpl = context.get("answer_prompt") or prompt.answer_prompt

        # Combine context
        context_str = ""
        if graph_result:
            context_str += f"Graph Knowledge:\n{graph_result}\n\n"
        if vector_result:
            context_str += f"Document Passages:\n{vector_result}\n\n"

        # Generate answer
        if self._llm and query:
            prompt_text = answer_prompt_tpl.format(
                context_str=context_str, query_str=query
            )
            try:
                answer = self._llm.generate(prompt=prompt_text)
            except Exception as e:
                log.error("Answer generation failed: %s", e)
                answer = ""
        else:
            answer = ""

        # Get entity IDs mentioned in the answer
        match_vids = context.get("match_vids", [])
        if isinstance(match_vids, list) and match_vids:
            entity_ids = [v for v in match_vids if isinstance(v, str)]
        else:
            entity_ids = []

        # Query provenance
        citations = []
        records = []
        if self._pm and entity_ids:
            provenance_map = self._pm.get_provenance_for_answer(
                entity_ids, max_per_entity=1
            )
            seen_texts = set()
            for eid, recs in provenance_map.items():
                for rec in recs:
                    if len(records) >= self._max_citations:
                        break
                    # Deduplicate by chunk text
                    if rec.chunk_text[:100] not in seen_texts:
                        seen_texts.add(rec.chunk_text[:100])
                        records.append(rec)
                        citations.append(
                            rec.to_citation(max_text_len=self._max_citation_text_len)
                        )

        # Append citations to answer
        if citations and answer:
            numbered = "\n".join(
                f"{i}. {c}" for i, c in enumerate(citations, 1)
            )
            answer = f"{answer}\n\n## 来源 / Sources\n{numbered}"

        context["answer"] = answer
        context["citations"] = citations
        context["provenance_records"] = [r.__dict__ for r in records]
        return context
