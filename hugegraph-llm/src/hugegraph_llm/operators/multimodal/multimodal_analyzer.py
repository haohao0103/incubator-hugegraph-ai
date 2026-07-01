# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal analysis operator with modality-specific VLM prompts.

Adapted from LightRAG prompt_multimodal.py. Three specialized prompts
for image, table, and equation analysis, each producing structured JSON
output that feeds into KG entity injection.

Operator protocol: run(context) -> context
  context["multimodal_sidecars"] = {
    "drawings": {item_id: {llm_analyze_result: {...}}},
    "tables": {item_id: {llm_analyze_result: {...}}},
    "equations": {item_id: {llm_analyze_result: {...}}},
  }
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

log = logging.getLogger(__name__)

# ============================================================================
# Image type enumeration (LightRAG prompt_multimodal.py IMAGE_TYPE_ENUM)
# ============================================================================

IMAGE_TYPE_ENUM: tuple[str, ...] = (
    "Photo", "Illustration", "Screenshot", "Icon", "Chart",
    "Table", "Infographic", "Flowchart", "Chat Log", "Wireframe",
    "Texture", "Other",
)
IMAGE_TYPE_FALLBACK = "Other"

# ============================================================================
# VLM Prompt Templates (adapted from LightRAG prompt_multimodal.py)
# ============================================================================

MULTIMODAL_PROMPTS: Dict[str, str] = {}

MULTIMODAL_PROMPTS["image_analysis"] = """You are an expert image analyzer. Analyze the provided image and return a single JSON object describing its content.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Examine the image carefully and identify:
   - The primary subject(s), scene, or composition.
   - Salient visual elements (objects, people, text overlays, diagrams, charts, screenshots, etc.).
   - Spatial layout when meaningful (e.g. left/right, foreground/background, panels of a figure).
   - Any visible text — quote it verbatim when short; summarize when long.

2. USE OF ADDITIONAL CONTEXT
   - Captions      : caption attached to the image ("n/a" = none)
   - Footnotes     : footnote attached to the image ("n/a" = none)
   - Leading Text  : text appearing immediately BEFORE the image ("n/a" = none)
   - Trailing Text : text appearing immediately AFTER the image ("n/a" = none)

   Rules:
   - Use context to disambiguate abbreviations, units, named entities, and the image's purpose.
   - The IMAGE ITSELF takes priority when it conflicts with context.
   - Only mention a relationship between the image and context if clearly supported.

3. NAMING (`name`) — concise, distinctive (3-8 words, snake_case preferred).
   Good: `crispr_cas9_workflow_diagram`, `q4_revenue_bar_chart`
   Bad: `image`, `figure`, `picture_1`

4. TYPE (`type`) — pick exactly one from:
   Photo, Illustration, Screenshot, Icon, Chart, Table, Infographic, Flowchart, Chat Log, Wireframe, Texture, Other

5. DESCRIPTION (`description`, ≤500 words, natural prose)
   Cover: what the image depicts, primary subjects, quantitative findings if chart/diagram,
   visible text content that carries meaning, relationships with surrounding context if clearly supported.

6. OUTPUT RULES
   - ONE valid JSON object only. No markdown, no code fences, no preamble.
   - All string values properly escaped JSON strings.
   - Output values for `name` and `description` must be in {language}.

================ ADDITIONAL CONTEXT ================
- Captions: {captions}
- Footnotes: {footnotes}
- Leading Text:
```
{leading}
```
- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{{{"name": "<name>", "type": "<type>", "description": "<description>"}}}}

Output:
"""

MULTIMODAL_PROMPTS["table_analysis"] = """You are an expert table analyzer. The exact format (HTML or JSON 2-D array) is declared in the TABLE CONTENT section. Analyze and return a single JSON object.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Identify: overall structure, column headers, units, key data points,
   patterns and trends, empty/null cells, footnote markers.

2. USE OF ADDITIONAL CONTEXT
   - Captions: table caption ("n/a" = none)
   - Footnotes: table footnote ("n/a" = none)
   - Leading Text: text before the table ("n/a" = none)
   - Trailing Text: text after the table ("n/a" = none)

   Rules:
   - TABLE CONTENT TAKES PRIORITY over context when they conflict.
   - NEVER invent rows, columns, values, units, or entities not visible.

3. NAMING (`name`) — concise, distinctive (3-8 words, snake_case).
   Good: `q4_2024_revenue_by_region`, `model_benchmark_accuracy_latency`
   Bad: `table`, `data_table`, `results`

4. DESCRIPTION (`description`, ≤500 words, natural prose)
   Cover: what the table is about, row/column meaning, units, time range,
   most important patterns/trends/outliers with specific values cited.

5. OUTPUT RULES
   - ONE valid JSON object only. No markdown, no preamble.
   - Output values for `name` and `description` in {language}.

================ TABLE CONTENT ================
The table below is in {content_format}.
```
{content}
```

================ ADDITIONAL CONTEXT ================
- Captions: {captions}
- Footnotes: {footnotes}
- Leading Text:
```
{leading}
```
- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{{{"name": "<name>", "description": "<description>"}}}}

Output:
"""

MULTIMODAL_PROMPTS["equation_analysis"] = """You are an expert analyzer of mathematical and chemical equations. The input is LaTeX or Markdown. Analyze and return a single JSON object.

================ INSTRUCTIONS ================

1. CONTENT RECOGNITION
   Identify: expression type, mathematical/chemical meaning, variables/constants/operators,
   application domain, physical/statistical significance, whether it matches a known named formula.

2. USE OF ADDITIONAL CONTEXT
   - Captions: equation caption/label ("n/a" = none)
   - Footnotes: equation footnote ("n/a" = none)
   - Leading Text: text before the equation ("n/a" = none)
   - Trailing Text: text after the equation ("n/a" = none)

   Rules:
   - THE EQUATION ITSELF TAKES PRIORITY over context.
   - NEVER invent variables or interpretations not justified.

3. NAMING (`name`) — what the equation IS/DOES, not just "equation".
   Good: `bayes_theorem_posterior`, `softmax_cross_entropy_loss`, `ideal_gas_law`
   Bad: `equation`, `formula`, `math`, `eq_1`

4. NORMALIZED EQUATION (`equation`) — math-mode BODY ONLY. No $ delimiters.
   Strip outer wrappers. Keep semantic inner environments (aligned, cases, pmatrix).
   Preserve all symbols, subscripts, superscripts faithfully.

5. DESCRIPTION (`description`, ≤300 words, natural prose)
   Cover: what the equation expresses, its role, named formula if any,
   brief clarification of non-obvious symbols, relationship with context if clearly supported.

6. OUTPUT RULES
   - ONE valid JSON object only. No markdown, no preamble.
   - LaTeX backslashes in `equation` string must be double-escaped.
   - Output values for `name` and `description` in {language}.

================ EQUATION BODY ================
```
{content}
```

================ ADDITIONAL CONTEXT ================
- Captions: {captions}
- Footnotes: {footnotes}
- Leading Text:
```
{leading}
```
- Trailing Text:
```
{trailing}
```

================ OUTPUT FORMAT ================
{{{{"name": "<name>", "equation": "<normalized LaTeX>", "description": "<description>"}}}}

Output:
"""

# ============================================================================
# Table format labels
# ============================================================================

_TABLE_FORMAT_LABELS: Dict[str, str] = {
    "html": "HTML format — a <table> fragment where merged cells use rowspan/colspan",
    "json": "JSON format — a 2-D array where rows[i][j] is the cell at row i, column j",
}


def table_content_format_label(fmt: str) -> str:
    """Human-readable format declaration for table_analysis prompt."""
    key = (fmt or "").strip().lower()
    try:
        return _TABLE_FORMAT_LABELS[key]
    except KeyError:
        raise ValueError(f"unknown table format {fmt!r}; expected 'html' or 'json'") from None


# ============================================================================
# MultimodalAnalyzer Operator
# ============================================================================


@dataclass
class MultimodalAnalyzerConfig:
    """Configuration for MultimodalAnalyzer operator."""
    language: str = "English"
    enabled_modalities: Set[str] = field(default_factory=lambda: {"drawings", "tables", "equations"})
    max_content_tokens: int = 8000
    surrounding_max_tokens: int = 2000


class MultimodalAnalyzer:
    """HG-AI operator for multimodal content analysis using VLM prompts.

    Adapts LightRAG's analyze_multimodal flow into the HG-AI operator
    protocol (run(context) -> context).

    Usage:
        analyzer = MultimodalAnalyzer(llm_func=my_llm_call)
        context = analyzer.run(context)

    The context dict should contain:
      - "multimodal_sidecars": {
          "drawings": {item_id: {...}},
          "tables": {item_id: {...}},
          "equations": {item_id: {...}},
        }
      - "blocks_content_by_id": {blockid: content_str}
    """

    def __init__(
        self,
        config: Optional[MultimodalAnalyzerConfig] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        llm_func_async: Optional[Callable] = None,
    ):
        self.config = config or MultimodalAnalyzerConfig()
        self.llm_func = llm_func
        self.llm_func_async = llm_func_async

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """HG-AI operator protocol: analyze all multimodal sidecar items.

        Args:
            context: Must contain "multimodal_sidecars" dict with
                     drawings/tables/equations sub-dicts, and optionally
                     "blocks_content_by_id" for surrounding context.

        Returns:
            context with "multimodal_analysis_results" added.
        """
        sidecars = context.get("multimodal_sidecars", {})
        blocks_content = context.get("blocks_content_by_id", {})
        results: Dict[str, Dict[str, Any]] = {
            "drawings": {},
            "tables": {},
            "equations": {},
        }

        # Process each modality
        for modality in ("drawings", "tables", "equations"):
            if modality not in self.config.enabled_modalities:
                continue
            items = sidecars.get(modality, {})
            if not items:
                continue

            prompt_key = _modality_to_prompt_key(modality)

            for item_id, item in items.items():
                if not isinstance(item, dict):
                    continue
                # Check if already analyzed
                if "llm_analyze_result" in item:
                    results[modality][item_id] = item
                    continue

                # Build prompt
                prompt = self._build_prompt(modality, item, blocks_content)
                if not prompt:
                    continue

                # Call LLM
                try:
                    if self.llm_func:
                        response = self.llm_func(prompt)
                    else:
                        log.warning("No LLM function provided; skipping analysis for %s/%s",
                                    modality, item_id)
                        continue

                    parsed = _parse_json_response(response)
                    if parsed:
                        item["llm_analyze_result"] = parsed
                        # Validate image type enum
                        if modality == "drawings" and "type" in parsed:
                            if parsed["type"] not in IMAGE_TYPE_ENUM:
                                parsed["type"] = IMAGE_TYPE_FALLBACK
                        results[modality][item_id] = item
                    else:
                        log.warning("Failed to parse LLM response for %s/%s",
                                    modality, item_id)

                except Exception as e:
                    log.error("Error analyzing %s/%s: %s", modality, item_id, e)

        context["multimodal_analysis_results"] = results
        return context

    def _build_prompt(
        self,
        modality: str,
        item: Dict[str, Any],
        blocks_content: Dict[str, str],
    ) -> Optional[str]:
        """Build modality-specific VLM analysis prompt."""
        prompt_key = _modality_to_prompt_key(modality)
        template = MULTIMODAL_PROMPTS.get(prompt_key)
        if not template:
            return None

        content = item.get("content", "")
        captions = item.get("caption", "") or "n/a"
        footnotes_list = item.get("footnotes", [])
        footnotes = " | ".join(footnotes_list) if footnotes_list else "n/a"
        language = self.config.language

        # Build surrounding context
        blockid = item.get("blockid", "")
        block_content = blocks_content.get(blockid, "")
        surrounding = item.get("surrounding", {})
        leading = surrounding.get("leading", "n/a") if surrounding else "n/a"
        trailing = surrounding.get("trailing", "n/a") if surrounding else "n/a"

        # If no surrounding yet, try to build basic context from block content
        if leading == "n/a" and trailing == "n/a" and block_content:
            # Simple context: first/last 500 chars of block content
            leading = block_content[:500] if len(block_content) > 500 else block_content
            trailing = block_content[-500:] if len(block_content) > 500 else ""

        format_vars = {
            "content": content,
            "captions": captions,
            "footnotes": footnotes,
            "language": language,
            "leading": leading or "n/a",
            "trailing": trailing or "n/a",
        }

        # Add content_format for tables
        if modality == "tables":
            fmt = item.get("format", "json")
            format_vars["content_format"] = table_content_format_label(fmt)

        try:
            return template.format(**format_vars)
        except KeyError as e:
            log.warning("Missing template variable for %s prompt: %s", modality, e)
            return None


# ============================================================================
# Helpers
# ============================================================================

def _modality_to_prompt_key(modality: str) -> str:
    """Map sidecar root key to prompt key."""
    mapping = {
        "drawings": "image_analysis",
        "tables": "table_analysis",
        "equations": "equation_analysis",
    }
    return mapping.get(modality, "")


def _parse_json_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response as JSON, tolerating markdown wrappers."""
    text = response.strip()

    # Strategy 1: direct JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract ```json ... ``` code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: extract first {...} pair
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None
