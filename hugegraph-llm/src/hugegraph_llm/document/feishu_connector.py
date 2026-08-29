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

"""Feishu (Lark) Wiki + Docx connector for knowledge-base ingestion.

PoC scope (as agreed):
- Content: Wiki knowledge base + cloud Docx text only (no sheets/bitable/chat).
- Update model: manual / scheduled full + incremental import. No event
  subscription (WebSocket/webhook) is wired in this PoC.

The connector walks a Wiki space node tree (``wiki.v2``), resolves each Docx
node (``docx.v1``), and converts its block structure (headings, paragraphs,
lists, quotes, code, tables) into markdown-ish plain text. Each document is
returned as a :class:`~hugegraph_llm.document.document_loader.Document` with
a stable ``doc_id`` (``feishu:{obj_token}``) and metadata (title, token,
updated_time, url, permission scope) so the caller can drive incremental
re-indexing and later ACL filtering through the company permission system.

``lark-oapi`` is imported lazily so this module can be imported (and its pure
parsing helpers unit-tested) without the SDK installed.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from hugegraph_llm.document.document_loader import BaseLoader, Document

log = logging.getLogger(__name__)

# Feishu Docx block types (subset used by the PoC).
BLOCK_PAGE = 1
BLOCK_TEXT = 2
BLOCK_HEADING1 = 3
BLOCK_HEADING2 = 4
BLOCK_HEADING3 = 5
BLOCK_HEADING4 = 6
BLOCK_HEADING5 = 7
BLOCK_HEADING6 = 8
BLOCK_HEADING7 = 9
BLOCK_HEADING8 = 10
BLOCK_HEADING9 = 11
BLOCK_BULLET = 12
BLOCK_ORDERED = 13
BLOCK_CODE = 14
BLOCK_QUOTE = 15
BLOCK_TABLE = 22
BLOCK_TABLE_CELL = 23

_HEADING_TYPES = {
    BLOCK_HEADING1: 1,
    BLOCK_HEADING2: 2,
    BLOCK_HEADING3: 3,
    BLOCK_HEADING4: 4,
    BLOCK_HEADING5: 5,
    BLOCK_HEADING6: 6,
    BLOCK_HEADING7: 7,
    BLOCK_HEADING8: 8,
    BLOCK_HEADING9: 9,
}


def _extract_text(text_obj: Any) -> str:
    """Extract plain text from a Feishu ``Text``-like object.

    Handles ``TextRun`` (content) and ``MentionDoc`` (@-mentions of docs)
    elements defensively so the parser works even if the SDK model shape
    differs slightly across versions.
    """
    if text_obj is None:
        return ""
    elements = getattr(text_obj, "elements", None) or []
    parts = []
    for element in elements:
        text_run = getattr(element, "text_run", None)
        if text_run is not None:
            content = getattr(text_run, "content", None)
            if content:
                parts.append(content)
            continue
        mention = getattr(element, "mention_doc", None)
        if mention is not None:
            title = getattr(mention, "title", None)
            if title:
                parts.append(f"《{title}》")
    return "".join(parts)


def _block_text(block: Any) -> str:
    """Extract text for the block-type-specific content field."""
    for attr in (
        "heading1",
        "heading2",
        "heading3",
        "heading4",
        "heading5",
        "heading6",
        "heading7",
        "heading8",
        "heading9",
        "text",
        "bullet",
        "ordered",
        "quote",
        "code",
        "table_cell",
    ):
        value = getattr(block, attr, None)
        if value is not None:
            return _extract_text(value)
    return ""


def _block_to_dict(block: Any) -> Dict[str, Any]:
    """Convert a lark ``Block`` object to a plain dict for markdown rendering."""
    block_type = int(getattr(block, "block_type", 0))
    children = list(getattr(block, "children", None) or [])
    info: Dict[str, Any] = {
        "block_id": getattr(block, "block_id", ""),
        "block_type": block_type,
        "children": children,
        "text": _block_text(block),
    }
    if block_type == BLOCK_TABLE:
        table = getattr(block, "table", None)
        prop = getattr(table, "property", None) if table is not None else None
        info["row_size"] = int(getattr(prop, "row_size", 0) or 0)
        info["column_size"] = int(getattr(prop, "column_size", 0) or 0)
    return info


def blocks_to_markdown(blocks: List[Dict[str, Any]]) -> str:
    """Render a list of simplified block dicts into markdown-ish text.

    This is a pure function (no lark SDK) so it can be unit-tested directly.
    Expected dict keys: ``block_id``, ``block_type``, ``children`` (list of
    block ids), ``text``; tables also carry ``row_size``/``column_size`` and
    their cells are separate ``BLOCK_TABLE_CELL`` blocks referenced via
    ``children`` in row-major order.
    """
    by_id = {b["block_id"]: b for b in blocks if b.get("block_id")}

    lines: List[str] = []
    ordered_counter = 0

    def render(block_id: str, depth: int) -> None:
        nonlocal ordered_counter
        block = by_id.get(block_id)
        if block is None:
            return
        block_type = block.get("block_type", 0)
        text = block.get("text", "") or ""

        if block_type in _HEADING_TYPES:
            level = _HEADING_TYPES[block_type]
            lines.append(f"{'#' * level} {text}".rstrip())
            ordered_counter = 0
        elif block_type == BLOCK_TEXT:
            lines.append(text)
            ordered_counter = 0
        elif block_type == BLOCK_BULLET:
            lines.append(f"{'  ' * depth}- {text}")
        elif block_type == BLOCK_ORDERED:
            ordered_counter += 1
            lines.append(f"{'  ' * depth}{ordered_counter}. {text}")
        elif block_type == BLOCK_QUOTE:
            for line in text.splitlines() or [""]:
                lines.append(f"> {line}")
            ordered_counter = 0
        elif block_type == BLOCK_CODE:
            lines.append("```")
            lines.append(text)
            lines.append("```")
            ordered_counter = 0
        elif block_type == BLOCK_TABLE:
            render_table(block)
            ordered_counter = 0
        # BLOCK_TABLE_CELL / BLOCK_PAGE / image / file are handled elsewhere.

        for child_id in block.get("children", []) or []:
            render(child_id, depth + 1)

    def render_table(table_block: Dict[str, Any]) -> None:
        row_size = int(table_block.get("row_size", 0))
        column_size = int(table_block.get("column_size", 0))
        cell_ids = list(table_block.get("children", []) or [])
        if row_size <= 0 or column_size <= 0:
            return
        rows: List[List[str]] = []
        for r in range(row_size):
            row: List[str] = []
            for c in range(column_size):
                idx = r * column_size + c
                cell = by_id.get(cell_ids[idx]) if idx < len(cell_ids) else None
                row.append((cell or {}).get("text", "") or "")
            rows.append(row)
        if not rows:
            return
        header = rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            padded = (row + [""] * len(header))[: len(header)]
            lines.append("| " + " | ".join(padded) + " |")

    # Render from page root(s): blocks whose type is page, or all top-level
    # blocks when no page block is present.
    page_blocks = [b for b in blocks if b.get("block_type") == BLOCK_PAGE]
    if page_blocks:
        for page in page_blocks:
            for child_id in page.get("children", []) or []:
                render(child_id, 0)
    else:
        for block in blocks:
            render(block.get("block_id", ""), 0)

    return "\n\n".join(line for line in lines if line.strip()).strip()


class FeishuConnector(BaseLoader):
    """Load Feishu Wiki knowledge-base documents as :class:`Document` objects.

    Args:
        app_id: Feishu self-built app id.
        app_secret: Feishu self-built app secret.
        base_url: Open platform domain (``https://open.feishu.cn`` for Feishu,
            ``https://open.larksuite.com`` for Lark international).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        app_id: str = "",
        app_secret: str = "",
        base_url: str = "https://open.feishu.cn",
        timeout: int = 30,
    ):
        self.app_id = app_id
        self.app_secret = app_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Any = None

    # -- client / auth -------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import lark_oapi as lark
            except ImportError as e:  # pragma: no cover - depends on env
                raise RuntimeError(
                    "FeishuConnector requires the 'lark-oapi' package. Install it with `uv sync --extra llm`."
                ) from e
            self._client = (
                lark.Client.builder()
                .app_id(self.app_id)
                .app_secret(self.app_secret)
                .domain(lark.FEISHU_DOMAIN if "feishu" in self.base_url else lark.LARK_SUITE_DOMAIN)
                .build()
            )
        return self._client

    # -- wiki traversal ------------------------------------------------------

    def list_wiki_docs(
        self,
        space_id: str,
        parent_node_token: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return all Docx nodes under a Wiki space (recursively).

        Each entry: ``{node_token, obj_token, obj_type, title, updated_time}``.
        ``updated_time`` is a Unix epoch seconds integer, used for incremental
        import.
        """
        client = self._get_client()
        import lark_oapi.api.wiki.v2 as wiki_v2

        result: List[Dict[str, Any]] = []

        def walk(node_token: Optional[str]) -> None:
            page_token = ""
            while True:
                req = (
                    wiki_v2.ListSpaceNodeRequest.builder()
                    .space_id(space_id)
                    .page_token(page_token)
                    .page_size(50)
                    .build()
                )
                if node_token:
                    req.parent_node_token = node_token
                resp = client.wiki.v2.space.list_node(req)
                if not resp.success():
                    log.warning("list wiki node failed: %s", resp.msg)
                    return
                data = resp.data
                for node in data.items or []:
                    obj_type = getattr(node, "obj_type", "")
                    if obj_type == "docx":
                        result.append(
                            {
                                "node_token": node.node_token,
                                "obj_token": node.obj_token,
                                "obj_type": obj_type,
                                "title": getattr(node, "title", ""),
                                "updated_time": _to_epoch_seconds(getattr(node, "updated_time", None)),
                            }
                        )
                    if getattr(node, "has_child", False):
                        walk(node.node_token)
                if not data.has_more or not data.page_token:
                    break
                page_token = data.page_token

        walk(parent_node_token)
        return result

    # -- public load API -----------------------------------------------------

    def load(self, sources: List[str]) -> List[Document]:
        """Load the given Docx object tokens (targeted re-index)."""
        return self.load_docs(sources)

    def load_wiki_space(self, space_id: str) -> List[Document]:
        """Full import of every Docx under a Wiki space."""
        return self._load_nodes(self.list_wiki_docs(space_id))

    def load_wiki_space_incremental(self, space_id: str, since_timestamp: int) -> List[Document]:
        """Import only Docx nodes updated after ``since_timestamp`` (epoch sec)."""
        nodes = [n for n in self.list_wiki_docs(space_id) if int(n.get("updated_time") or 0) > since_timestamp]
        return self._load_nodes(nodes)

    def load_docs(self, doc_tokens: List[str]) -> List[Document]:
        """Load specific Docx object tokens (used for targeted re-index)."""
        nodes = [{"obj_token": t, "title": "", "updated_time": 0, "node_token": t} for t in doc_tokens]
        return self._load_nodes(nodes)

    # -- docx fetch ----------------------------------------------------------

    def _load_nodes(self, nodes: List[Dict[str, Any]]) -> List[Document]:
        docs: List[Document] = []
        for node in nodes:
            try:
                doc = self._load_docx(node)
                if doc is not None:
                    docs.append(doc)
            except Exception as e:
                log.error("failed to load feishu docx %s: %s", node.get("obj_token"), e)
        return docs

    def _load_docx(self, node: Dict[str, Any]) -> Optional[Document]:
        obj_token = node.get("obj_token", "")
        if not obj_token:
            return None
        client = self._get_client()
        import lark_oapi.api.docx.v1 as docx_v1

        # Document title + revision.
        title = node.get("title", "")
        get_req = docx_v1.GetDocumentRequest.builder().document_id(obj_token).build()
        get_resp = client.docx.v1.document.get(get_req)
        if get_resp.success() and get_resp.data and get_resp.data.document:
            title = getattr(get_resp.data.document, "title", "") or title

        # Blocks -> markdown.
        blocks = self._fetch_blocks(obj_token)
        markdown = blocks_to_markdown([_block_to_dict(b) for b in blocks])
        if not markdown.strip():
            log.warning("feishu docx %s has empty content, skipped", obj_token)
            return None

        node_token = node.get("node_token", "") or obj_token
        metadata = {
            "source_type": "feishu_wiki",
            "source": "feishu",
            "title": title,
            "doc_token": obj_token,
            "node_token": node_token,
            "updated_time": node.get("updated_time", 0),
            "url": f"{self.base_url}/wiki/{node_token}",
        }
        return Document(
            content=markdown,
            metadata=metadata,
            doc_id=f"feishu:{obj_token}",
        )

    def _fetch_blocks(self, doc_token: str) -> List[Any]:
        client = self._get_client()
        import lark_oapi.api.docx.v1 as docx_v1

        blocks: List[Any] = []
        page_token = ""
        while True:
            req = (
                docx_v1.ListDocumentBlockRequest.builder()
                .document_id(doc_token)
                .page_token(page_token)
                .page_size(500)
                .document_revision_id(-1)
                .build()
            )
            resp = client.docx.v1.document_block.list(req)
            if not resp.success():
                log.warning("list docx block failed: %s", resp.msg)
                break
            data = resp.data
            if data.items:
                blocks.extend(data.items)
            if not data.has_more or not data.page_token:
                break
            page_token = data.page_token
        return blocks


def _to_epoch_seconds(value: Any) -> int:
    """Normalize a Feishu timestamp to Unix epoch seconds."""
    if value is None:
        return 0
    try:
        if isinstance(value, (int, float)):
            return int(value)
        return int(datetime.fromisoformat(str(value)).timestamp())
    except (TypeError, ValueError):
        return 0
