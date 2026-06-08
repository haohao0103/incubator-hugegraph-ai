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
HyDE (Hypothetical Document Embeddings) query enhancement operator.

Generates a hypothetical answer passage for a given query using LLM,
then uses the passage embedding for vector retrieval instead of the
raw query. This significantly improves recall for short/ambiguous queries.

Three modes:
- off: No HyDE enhancement (backward compatible default)
- prefix: Original query + HyDE passage concatenated for retrieval
- full: Pure HyDE passage used for retrieval
"""

from typing import Any, Dict, Optional

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.models.llms.init_llm import LLMs
from hugegraph_llm.utils.log import log

DEFAULT_HYDE_PROMPT = (
    "Please write a short passage that answers the following question. "
    "The passage should be factual and contain key terms that would appear "
    "in relevant documents. Do not worry if the information is not perfectly "
    "accurate — the goal is to generate a plausible passage that captures "
    "the semantics of the answer.\n\n"
    "Question: {query}\n\n"
    "Passage:"
)

DEFAULT_HYDE_PROMPT_CN = (
    "请写一段简短的文字来回答以下问题。"
    "这段文字应该是事实性的，并包含相关文档中会出现的核心关键词。"
    "不必担心信息不完全准确——目标是生成一段合理的、能体现答案语义的文字。\n\n"
    "问题：{query}\n\n"
    "段落："
)


class HyDEGenerate:
    """HyDE (Hypothetical Document Embeddings) query enhancement operator.

    Solves the sparse short-query problem by generating a hypothetical
    answer passage via LLM, then using the passage for vector retrieval.

    Usage::

        enhancer = HyDEGenerate(llm=my_llm, mode="prefix")
        context = enhancer.run({"query": "What is HugeGraph?"})
        # context["hyde_query"] contains the enhanced query text
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        mode: str = "prefix",
        max_query_length: int = 100,
        prompt_template: Optional[str] = None,
        prompt_template_cn: Optional[str] = None,
    ):
        """Initialize HyDE enhancer.

        :param llm: LLM instance for generating hypothetical passages.
            If None, uses the default LLM from config.
        :param mode: Enhancement mode:
            - "off": no enhancement (passthrough)
            - "prefix": original query + HyDE passage
            - "full": pure HyDE passage
        :param max_query_length: Queries longer than this (chars) skip
            HyDE, since they are already information-rich.
        :param prompt_template: Custom HyDE prompt (English). Uses default
            if not provided.
        :param prompt_template_cn: Custom HyDE prompt (Chinese). Uses
            default if not provided.
        """
        self._llm = llm
        self._mode = mode
        self._max_query_length = max_query_length
        self._prompt_template = prompt_template or DEFAULT_HYDE_PROMPT
        self._prompt_template_cn = prompt_template_cn or DEFAULT_HYDE_PROMPT_CN

    @property
    def mode(self) -> str:
        """Return the current HyDE mode."""
        return self._mode

    def _get_llm(self) -> BaseLLM:
        """Lazy-initialize LLM if not provided."""
        if self._llm is None:
            self._llm = LLMs().get_general_llm()
        return self._llm

    def _should_enhance(self, query: str) -> bool:
        """Determine if HyDE should be applied.

        Skip when:
        - mode is "off"
        - query is empty or None
        - query is already long enough (information-rich)
        """
        if self._mode == "off":
            return False
        if not query or not query.strip():
            return False
        if len(query) > self._max_query_length:
            log.debug("HyDE skipped: query length %d > max %d", len(query), self._max_query_length)
            return False
        return True

    def _generate_hypothetical(self, query: str, language: str = "en") -> str:
        """Generate a hypothetical answer passage for the query.

        :param query: The original user query.
        :param language: "en" or "cn" for prompt selection.
        :return: Generated hypothetical passage.
        """
        template = self._prompt_template_cn if language == "cn" else self._prompt_template
        prompt_text = template.format(query=query)

        try:
            llm = self._get_llm()
            passage = llm.generate(prompt=prompt_text)
            if passage:
                passage = passage.strip()
            return passage or ""
        except Exception as e:
            log.warning("HyDE generation failed: %s, falling back to original query", e)
            return ""

    def enhance(self, query: str, language: str = "en") -> str:
        """Generate the enhanced query text.

        :param query: Original user query.
        :param language: "en" or "cn" for prompt selection.
        :return: Enhanced query string (or original if HyDE not applicable).
        """
        if not self._should_enhance(query):
            return query

        hypothetical = self._generate_hypothetical(query, language)
        if not hypothetical:
            log.warning("HyDE generated empty passage, using original query")
            return query

        if self._mode == "full":
            return hypothetical
        # prefix mode: original query + hypothetical passage
        return f"{query}\n\n{hypothetical}"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run HyDE enhancement as an operator.

        Reads ``query`` from context and writes ``hyde_query`` and
        ``hyde_applied`` back.

        :param context: Pipeline context dict with "query" key.
        :return: Updated context with HyDE enhancement.
        """
        query = context.get("query", "")
        language = context.get("language", "en")
        enhanced = self.enhance(query, language)

        applied = enhanced != query
        if applied:
            log.info(
                "HyDE applied (mode=%s): query length %d -> %d",
                self._mode,
                len(query),
                len(enhanced),
            )

        context["hyde_query"] = enhanced
        context["hyde_applied"] = applied
        context["original_query"] = query
        context["call_count"] = context.get("call_count", 0) + (1 if applied else 0)
        return context
