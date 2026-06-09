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

"""GraphRAG Question-Answering Chain for LangChain integration."""

from typing import Any, Dict, List, Optional

from hugegraph_llm.utils.log import log

_QA_PROMPT = """You are a helpful knowledge assistant. Use the following
retrieved context to answer the user's question. If the context does not
contain enough information, say so honestly.

## Context
{context}

## Question
{question}

## Answer
"""

_SUMMARY_PROMPT = """Summarize the following context into a concise answer
for the question. Focus on factual accuracy.

## Context
{context}

## Question
{question}

## Summary
"""


class HugeGraphQAChain:
    """GraphRAG QA chain combining retrieval with LLM synthesis.

    Wraps a HugeGraphRetriever with an LLM to produce answers from
    graph + vector retrieval results.

    Usage::

        chain = HugeGraphQAChain(
            retriever=my_retriever,
            llm=my_llm,
        )
        result = chain.run("What is the relationship between X and Y?")
        # result = {"answer": "...", "sources": [...], "context": "..."}
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        llm: Optional[Any] = None,
        prompt_template: Optional[str] = None,
        max_context_length: int = 4000,
        include_sources: bool = True,
    ):
        self._retriever = retriever
        self._llm = llm
        self._prompt_template = prompt_template or _QA_PROMPT
        self._max_context_length = max_context_length
        self._include_sources = include_sources

    def _build_context(self, documents: List[Dict]) -> str:
        """Build a context string from retrieved documents.

        Truncates total length to max_context_length.
        """
        parts = []
        total_len = 0
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            chunk = f"[{i+1}] (source: {source}) {content}"
            if total_len + len(chunk) > self._max_context_length:
                remaining = self._max_context_length - total_len
                if remaining > 50:
                    parts.append(chunk[:remaining] + "...")
                break
            parts.append(chunk)
            total_len += len(chunk)
        return "\n\n".join(parts)

    def run(self, question: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Run the QA chain: retrieve then synthesize.

        :param question: User question.
        :param k: Number of documents to retrieve.
        :return: Dict with "answer", "sources", "context".
        """
        # Step 1: Retrieve
        documents = []
        if self._retriever:
            try:
                documents = self._retriever.get_relevant_documents(question, k=k)
            except Exception as e:
                log.error("QAChain retrieval failed: %s", e)

        context = self._build_context(documents)

        # Step 2: Generate answer
        answer = ""
        if self._llm and context:
            try:
                prompt = self._prompt_template.format(
                    context=context, question=question
                )
                answer = self._llm.generate(prompt=prompt)
            except Exception as e:
                log.error("QAChain LLM generation failed: %s", e)
                answer = "[Error generating answer]"
        elif not context:
            answer = "No relevant context found for this question."

        # Build result
        result = {
            "answer": answer,
            "context": context,
        }
        if self._include_sources:
            result["sources"] = [
                doc.get("metadata", {}).get("source", "unknown")
                for doc in documents
            ]
        return result

    def summarize(self, question: str, context_text: str) -> str:
        """Summarize a given context for a question (skip retrieval).

        :param question: User question.
        :param context_text: Raw context text.
        :return: Summary string.
        """
        if not self._llm:
            return context_text[:500]
        try:
            prompt = _SUMMARY_PROMPT.format(context=context_text, question=question)
            return self._llm.generate(prompt=prompt)
        except Exception as e:
            log.error("QAChain summarize failed: %s", e)
            return context_text[:500]
