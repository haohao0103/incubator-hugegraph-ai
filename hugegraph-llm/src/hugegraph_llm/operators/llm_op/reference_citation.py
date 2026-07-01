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

"""Reference ID citation system for answer provenance.

Inspired by LightRAG's ``generate_reference_list_from_chunks()`` (utils.py),
adapted for HugeGraph-AI's ProvenanceAnswerSynthesize.

Key differences from LightRAG:
- Works with our ProvenanceRecord/ProvenanceManager instead of raw chunks
- Generates [1], [2] style inline citations with file_path mapping
- Integrates with ProvenanceAnswerSynthesize's citation prompt

Usage:
    from hugegraph_llm.operators.llm_op.reference_citation import (
        ReferenceIdGenerator, ReferenceCitationBuilder,
    )

    generator = ReferenceIdGenerator()
    ref_list = generator.generate_from_chunks(chunks)

    builder = ReferenceCitationBuilder()
    answer_with_refs = builder.build(answer, ref_list)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from hugegraph_llm.utils.log import log


@dataclass
class ReferenceEntry:
    """A single reference entry mapping reference_id to source document."""
    reference_id: str
    file_path: str
    chunk_text: str = ""
    frequency: int = 1  # How many chunks cite this source

    def to_dict(self) -> Dict[str, str]:
        return {
            "reference_id": self.reference_id,
            "file_path": self.file_path,
        }

    def to_citation_str(self, max_text_len: int = 200) -> str:
        """Format as a citation string: [1] file_path — snippet."""
        snippet = self.chunk_text[:max_text_len] if self.chunk_text else ""
        if snippet:
            return f"[{self.reference_id}] {self.file_path} — {snippet}…"
        return f"[{self.reference_id}] {self.file_path}"


@dataclass
class ReferenceList:
    """Complete reference list from a query answer."""
    entries: List[ReferenceEntry] = field(default_factory=list)
    id_to_file_path: Dict[str, str] = field(default_factory=dict)

    def format_reference_section(self, max_text_len: int = 200) -> str:
        """Format as a markdown reference section."""
        if not self.entries:
            return ""
        lines = ["## References"]
        for entry in self.entries:
            lines.append(entry.to_citation_str(max_text_len=max_text_len))
        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict[str, str]]:
        """Export as a list of dicts (compatible with LightRAG format)."""
        return [e.to_dict() for e in self.entries]


class ReferenceIdGenerator:
    """Generate sequential reference IDs from chunks, prioritized by frequency.

    Directly inspired by LightRAG's ``generate_reference_list_from_chunks()``.

    Algorithm:
    1. Extract file_paths from chunks
    2. Count occurrences per file_path
    3. Sort by frequency (descending) + first appearance order
    4. Assign sequential IDs (1, 2, 3...) to file_paths
    5. Inject reference_id back into each chunk

    This ensures the most frequently cited source gets reference_id=1.
    """

    def __init__(
        self,
        unknown_source_marker: str = "unknown_source",
    ):
        self._unknown_marker = unknown_source_marker

    def generate_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Tuple[ReferenceList, List[Dict[str, Any]]]:
        """Generate reference list from chunks.

        Args:
            chunks: List of chunk dicts, each with "file_path" key.

        Returns:
            Tuple of (ReferenceList, updated_chunks_with_reference_ids).
        """
        if not chunks:
            return ReferenceList(), []

        # Step 1: Count file_path occurrences
        file_path_counts: Dict[str, int] = {}
        for chunk in chunks:
            fp = chunk.get("file_path", "")
            if fp and fp != self._unknown_marker:
                file_path_counts[fp] = file_path_counts.get(fp, 0) + 1

        # Step 2: Sort by frequency (descending) + first appearance
        file_path_with_indices: List[Tuple[str, int, int]] = []
        seen_paths: set = set()
        for i, chunk in enumerate(chunks):
            fp = chunk.get("file_path", "")
            if fp and fp != self._unknown_marker and fp not in seen_paths:
                file_path_with_indices.append((fp, file_path_counts[fp], i))
                seen_paths.add(fp)

        sorted_paths = sorted(file_path_with_indices, key=lambda x: (-x[1], x[2]))
        unique_paths = [item[0] for item in sorted_paths]

        # Step 3: Create file_path → reference_id mapping
        file_path_to_ref_id: Dict[str, str] = {}
        for i, fp in enumerate(unique_paths):
            file_path_to_ref_id[fp] = str(i + 1)

        # Step 4: Build reference entries
        entries: List[ReferenceEntry] = []
        id_to_file_path: Dict[str, str] = {}
        for i, fp in enumerate(unique_paths):
            ref_id = str(i + 1)
            # Find representative chunk text
            rep_text = ""
            for chunk in chunks:
                if chunk.get("file_path") == fp:
                    rep_text = chunk.get("content", chunk.get("text", ""))
                    break

            entries.append(ReferenceEntry(
                reference_id=ref_id,
                file_path=fp,
                chunk_text=rep_text[:500] if rep_text else "",
                frequency=file_path_counts[fp],
            ))
            id_to_file_path[ref_id] = fp

        # Step 5: Inject reference_id into chunks
        updated_chunks: List[Dict[str, Any]] = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            fp = chunk_copy.get("file_path", "")
            if fp and fp != self._unknown_marker:
                chunk_copy["reference_id"] = file_path_to_ref_id[fp]
            else:
                chunk_copy["reference_id"] = ""
            updated_chunks.append(chunk_copy)

        return ReferenceList(entries=entries, id_to_file_path=id_to_file_path), updated_chunks

    def generate_from_records(
        self,
        records: List[Any],
        max_text_len: int = 300,
    ) -> ReferenceList:
        """Generate reference list from ProvenanceRecords.

        Adaptation for HugeGraph-AI's ProvenanceManager output.

        Args:
            records: List of ProvenanceRecord objects.
            max_text_len: Max length of chunk text in reference entries.

        Returns:
            ReferenceList with entries.
        """
        if not records:
            return ReferenceList()

        # Convert ProvenanceRecords to chunk-like dicts
        chunks = []
        for rec in records:
            fp = getattr(rec, "document_path", getattr(rec, "file_path", ""))
            text = getattr(rec, "chunk_text", getattr(rec, "text", ""))
            chunks.append({"file_path": fp, "text": text})

        ref_list, _ = self.generate_from_chunks(chunks)

        # Trim chunk_text to max_text_len
        for entry in ref_list.entries:
            if len(entry.chunk_text) > max_text_len:
                entry.chunk_text = entry.chunk_text[:max_text_len]

        return ref_list


class ReferenceCitationBuilder:
    """Build answers with inline [reference_id] citations.

    Inspired by LightRAG's prompt pattern where the LLM is instructed
    to use [reference_id] format in responses. This builder:
    1. Injects reference instructions into the answer prompt
    2. Post-processes the LLM response to append reference section
    """

    CITATION_INSTRUCTION = (
        "\n\nIMPORTANT: When citing facts in your response, use the format "
        "[reference_id] to refer to source documents. "
        "For example: 'According to [1], the capital of France is Paris.' "
        "The reference_id numbers correspond to the Reference Document List below.\n\n"
        "Reference Document List:\n{reference_list_str}\n"
    )

    def build_prompt_with_references(
        self,
        base_prompt: str,
        ref_list: ReferenceList,
    ) -> str:
        """Inject reference list instructions into an answer prompt.

        Args:
            base_prompt: Original prompt template with {context_str} and {query_str}.
            ref_list: Reference list to inject.

        Returns:
            Enhanced prompt with citation instructions.
        """
        if not ref_list.entries:
            return base_prompt

        ref_lines = []
        for entry in ref_list.entries:
            ref_lines.append(f"[{entry.reference_id}] {entry.file_path}")

        reference_list_str = "\n".join(ref_lines)
        citation_suffix = self.CITATION_INSTRUCTION.format(
            reference_list_str=reference_list_str,
        )

        return base_prompt + citation_suffix

    def build_answer_with_references(
        self,
        answer: str,
        ref_list: ReferenceList,
        max_citation_text_len: int = 200,
    ) -> str:
        """Post-process answer by appending reference section.

        Also validates that [reference_id] citations in the answer
        are consistent with the reference list.

        Args:
            answer: LLM-generated answer text.
            ref_list: Reference list for citation validation.
            max_citation_text_len: Max text length per citation.

        Returns:
            Answer with appended reference section.
        """
        if not answer:
            return answer

        # Validate inline citations
        cited_ids = set(re.findall(r'\[(\d+)\]', answer))
        valid_ids = set(ref_list.id_to_file_path.keys())

        invalid_ids = cited_ids - valid_ids
        if invalid_ids:
            log.warning(
                "ReferenceCitationBuilder: invalid citations %s in answer",
                invalid_ids,
            )

        # Append reference section
        ref_section = ref_list.format_reference_section(
            max_text_len=max_citation_text_len,
        )

        if ref_section:
            # Avoid double-append if answer already has ## References
            if "## References" not in answer and "## 来源" not in answer:
                return f"{answer}\n\n{ref_section}"

        return answer

    def extract_cited_reference_ids(self, answer: str) -> List[str]:
        """Extract all [reference_id] citations from an answer."""
        return re.findall(r'\[(\d+)\]', answer)
