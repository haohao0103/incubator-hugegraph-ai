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

"""Sidecar writer — emits spec-compliant *.parsed/ directory from IR.

Adapted from LightRAG sidecar/writer.py. This module owns the single
executable specification of the sidecar format. Parser adapters hand it
an IRDoc; it emits the parsed/ directory with blocks.jsonl, tables.json,
drawings.json, equations.json, and assets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hugegraph_llm.operators.multimodal.sidecar_ir import (
    AssetSpec,
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRTable,
)
from hugegraph_llm.operators.multimodal.sidecar_placeholder import (
    render_drawing_tag,
    render_equation_tag,
    render_table_tag,
    render_template,
    table_body_for_rows,
)

log = logging.getLogger(__name__)


def write_sidecar(
    ir: IRDoc,
    *,
    parsed_dir: Path,
    doc_id: str,
    engine: str = "native",
    clean_parsed_dir: bool = True,
    block_drawing_path_style: str = "with_prefix",
) -> Dict[str, Any]:
    """Emit a spec-compliant *.parsed/ directory from an IR.

    Args:
        ir: Document IR produced by a parser adapter.
        parsed_dir: Output directory.
        doc_id: doc-<md5> identifier; doc_hash for sidecar ids is
                 the 32-char tail after stripping 'doc-' prefix.
        engine: Parser engine name (native/mineru/docling/legacy).
        clean_parsed_dir: When True (default), rmtree parsed_dir first.
        block_drawing_path_style: "with_prefix" or "basename_only".

    Returns:
        Dict with doc_id, file_path, content, blocks_path.
    """
    valid_styles = {"with_prefix", "basename_only"}
    if block_drawing_path_style not in valid_styles:
        raise ValueError(
            f"block_drawing_path_style must be one of {valid_styles}, "
            f"got {block_drawing_path_style!r}"
        )

    if clean_parsed_dir and parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(ir.document_name).stem or ir.document_name
    blocks_path = parsed_dir / f"{base_name}.blocks.jsonl"
    tables_path = parsed_dir / f"{base_name}.tables.json"
    drawings_path = parsed_dir / f"{base_name}.drawings.json"
    equations_path = parsed_dir / f"{base_name}.equations.json"
    assets_dir = parsed_dir / f"{base_name}.blocks.assets"

    # Stage 1: realize assets
    asset_paths = _materialize_assets(ir.assets, assets_dir)

    # Stage 2: walk blocks, allocate ids, render templates
    doc_hash = doc_id.removeprefix("doc-")
    tables: Dict[str, Dict[str, Any]] = {}
    drawings: Dict[str, Dict[str, Any]] = {}
    equations: Dict[str, Dict[str, Any]] = {}
    blocks_lines: List[str] = []
    merged_parts: List[str] = []

    table_seq = 0
    drawing_seq = 0
    equation_seq = 0
    asset_prefix = f"{assets_dir.name}/"

    for block_index, block in enumerate(ir.blocks):
        # Allocate ids
        table_id_by_key: Dict[str, str] = {}
        for table in block.tables:
            table_seq += 1
            tb_id = f"tb-{doc_hash}-{table_seq:04d}"
            table_id_by_key[table.placeholder_key] = tb_id

        drawing_id_by_key: Dict[str, str] = {}
        for drawing in block.drawings:
            drawing_seq += 1
            im_id = f"im-{doc_hash}-{drawing_seq:04d}"
            drawing_id_by_key[drawing.placeholder_key] = im_id

        equation_id_by_key: Dict[str, str] = {}
        for equation in block.equations:
            if not equation.is_block:
                continue
            equation_seq += 1
            eq_id = f"eq-{doc_hash}-{equation_seq:04d}"
            equation_id_by_key[equation.placeholder_key] = eq_id

        # Render placeholder template
        rendered = _render_block_content(
            block,
            table_id_by_key=table_id_by_key,
            drawing_id_by_key=drawing_id_by_key,
            equation_id_by_key=equation_id_by_key,
            asset_paths=asset_paths,
            asset_prefix=asset_prefix,
            block_drawing_path_style=block_drawing_path_style,
        )

        rendered = rendered.strip()
        if not rendered:
            continue

        blockid = hashlib.md5(
            f"{doc_id}:{block_index}:{block.heading}:{rendered}".encode("utf-8")
        ).hexdigest()

        # Realize sidecar items
        for table in block.tables:
            tb_id = table_id_by_key[table.placeholder_key]
            if tb_id not in rendered:
                log.warning(
                    "Orphan table id=%s on block %d; skipping sidecar entry",
                    tb_id, block_index,
                )
                continue
            tables[tb_id] = _table_item_dict(
                tb_id, blockid, block.heading, block.parent_headings, table
            )

        for drawing in block.drawings:
            im_id = drawing_id_by_key[drawing.placeholder_key]
            if im_id not in rendered:
                log.warning(
                    "Orphan drawing id=%s on block %d; skipping sidecar entry",
                    im_id, block_index,
                )
                continue
            drawings[im_id] = _drawing_item_dict(
                im_id, blockid, block.heading, block.parent_headings,
                drawing, asset_paths, asset_prefix,
            )

        for equation in block.equations:
            if not equation.is_block:
                continue
            eq_id = equation_id_by_key[equation.placeholder_key]
            if eq_id not in rendered:
                log.warning(
                    "Orphan equation id=%s on block %d; skipping sidecar entry",
                    eq_id, block_index,
                )
                continue
            equations[eq_id] = _equation_item_dict(
                eq_id, blockid, block.heading, block.parent_headings, equation
            )

        row: Dict[str, Any] = {
            "type": "content",
            "blockid": blockid,
            "format": "plain_text",
            "content": rendered,
            "heading": block.heading,
            "parent_headings": list(block.parent_headings),
            "level": int(block.level),
            "session_type": block.session_type or "body",
            "table_slice": block.table_slice or "none",
            "positions": [p.to_jsonable() for p in block.positions],
        }
        if block.table_header:
            row["table_header"] = block.table_header
        blocks_lines.append(json.dumps(row, ensure_ascii=False))
        merged_parts.append(rendered)

    # Stage 3: doc-level metadata
    merged_text = "\n\n".join(p for p in merged_parts if p.strip())
    document_hash = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()
    parse_time = datetime.now(timezone.utc).isoformat()

    asset_dir_present = assets_dir.exists() and any(assets_dir.iterdir())
    if not asset_dir_present and assets_dir.exists():
        try:
            assets_dir.rmdir()
        except OSError:
            pass

    meta: Dict[str, Any] = {
        "type": "meta",
        "format": "lightrag",
        "version": "1.0",
        "document_name": ir.document_name,
        "document_format": ir.document_format,
        "document_hash": f"sha256:{document_hash}",
        "table_file": bool(tables),
        "equation_file": bool(equations),
        "drawing_file": bool(drawings),
        "asset_dir": asset_dir_present,
    }
    split_option = dict(ir.split_option or {})
    if split_option:
        meta["split_option"] = split_option
    meta.update({
        "blocks": len(blocks_lines),
        "doc_id": doc_id,
        "parse_engine": engine,
        "parse_time": parse_time,
        "doc_title": ir.doc_title,
    })
    if ir.bbox_attributes is not None:
        meta["bbox_attributes"] = dict(ir.bbox_attributes)

    blocks_path.write_text(
        "\n".join([json.dumps(meta, ensure_ascii=False)] + blocks_lines) + "\n",
        encoding="utf-8",
    )

    # Sidecar JSONs
    if tables:
        tables_path.write_text(
            json.dumps({"version": "1.0", "tables": tables},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if drawings:
        drawings_path.write_text(
            json.dumps({"version": "1.0", "drawings": drawings},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if equations:
        equations_path.write_text(
            json.dumps({"version": "1.0", "equations": equations},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    log.info(
        "Wrote %d blocks for doc_id=%s (%d tables, %d drawings, %d equations, "
        "assets=%s, engine=%s)",
        len(blocks_lines), doc_id, len(tables), len(drawings),
        len(equations), asset_dir_present, engine,
    )

    return {
        "doc_id": doc_id,
        "file_path": ir.document_name,
        "parse_format": "lightrag",
        "content": merged_text,
        "blocks_path": str(blocks_path),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _materialize_assets(
    assets: List[AssetSpec],
    assets_dir: Path,
) -> Dict[str, str]:
    """Materialize AssetSpec objects into assets_dir.

    Returns: {ref: filename_inside_assets_dir}.
    """
    if not assets:
        return {}

    assets_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, str] = {}
    used_names: set[str] = set()

    for spec in assets:
        target_name = _allocate_unique_name(spec.suggested_name, used_names)
        target_path = assets_dir / target_name

        try:
            target_path.resolve().relative_to(assets_dir.resolve())
        except ValueError:
            log.warning("Unsafe asset target for ref=%s (%s); skipping",
                        spec.ref, spec.suggested_name)
            continue

        if isinstance(spec.source, Path):
            src_path = Path(str(spec.source))
            if not src_path.exists():
                log.warning("Asset source missing for ref=%s; skipping copy", spec.ref)
                continue
            if src_path.resolve() != target_path.resolve():
                shutil.copyfile(src_path, target_path)
        elif isinstance(spec.source, bytes):
            target_path.write_bytes(spec.source)
        elif spec.source is None:
            if not target_path.exists():
                log.warning("Asset ref=%s declared in place but %s is absent",
                            spec.ref, target_path)
                continue
        else:
            log.warning("Unsupported AssetSpec.source type for ref=%s: %s",
                        spec.ref, type(spec.source).__name__)
            continue

        used_names.add(target_name)
        out[spec.ref] = target_name

    return out


def _allocate_unique_name(suggested: str, used: set[str]) -> str:
    """Make suggested unique within used: foo.png -> foo-2.png."""
    suggested = _safe_asset_filename(suggested)
    if suggested not in used:
        return suggested
    stem = Path(suggested).stem
    suffix = Path(suggested).suffix
    n = 2
    while True:
        cand = f"{stem}-{n}{suffix}"
        if cand not in used:
            return cand
        n += 1


def _safe_asset_filename(suggested: str) -> str:
    """Collapse parser-suggested asset names to a safe filename."""
    name = Path(str(suggested).replace("\\", "/")).name
    name = "".join(c for c in name.strip() if ord(c) >= 32 and c != "\x7f").strip(".")
    return name or "asset"


def _render_block_content(
    block: IRBlock,
    *,
    table_id_by_key: Dict[str, str],
    drawing_id_by_key: Dict[str, str],
    equation_id_by_key: Dict[str, str],
    asset_paths: Dict[str, str],
    asset_prefix: str,
    block_drawing_path_style: str = "with_prefix",
) -> str:
    """Expand placeholder tokens in block.content_template."""
    tables_by_key = {t.placeholder_key: t for t in block.tables}
    drawings_by_key = {d.placeholder_key: d for d in block.drawings}
    equations_by_key = {e.placeholder_key: e for e in block.equations}

    def _table(key: str) -> str:
        table = tables_by_key.get(key)
        if table is None:
            return ""
        tb_id = table_id_by_key.get(key, "")
        if table.body_override is not None:
            fmt = "json" if table.rows is not None else "html"
            return render_table_tag(tb_id, fmt, table.body_override)
        if table.rows is not None:
            return render_table_tag(tb_id, "json", table_body_for_rows(table.rows))
        return render_table_tag(tb_id, "html", table.html or "")

    def _drawing(key: str) -> str:
        drawing = drawings_by_key.get(key)
        if drawing is None:
            return ""
        im_id = drawing_id_by_key.get(key, "")
        if drawing.path_override is not None:
            path = drawing.path_override
        else:
            filename = asset_paths.get(drawing.asset_ref, "")
            if not filename:
                path = ""
            elif block_drawing_path_style == "basename_only":
                path = filename
            else:
                path = f"{asset_prefix}{filename}"
        return render_drawing_tag(im_id, drawing.fmt, drawing.caption, path, drawing.src)

    def _equation(key: str) -> str:
        eq = equations_by_key.get(key)
        if eq is None:
            return ""
        if not eq.is_block:
            return render_equation_tag(None, eq.latex, eq.caption)
        eq_id = equation_id_by_key.get(key, "")
        return render_equation_tag(eq_id, eq.latex, eq.caption)

    def _inline_equation(key: str) -> str:
        eq = equations_by_key.get(key)
        if eq is None:
            return ""
        return render_equation_tag(None, eq.latex, eq.caption)

    return render_template(
        block.content_template,
        table_renderer=_table,
        drawing_renderer=_drawing,
        equation_renderer=_equation,
        inline_equation_renderer=_inline_equation,
    )


def _table_item_dict(
    table_id: str, blockid: str, heading: str,
    parent_headings: List[str], table: IRTable,
) -> Dict[str, Any]:
    if table.rows is not None:
        fmt = "json"
        content = table_body_for_rows(table.rows)
    else:
        fmt = "html"
        content = table.html or ""

    item: Dict[str, Any] = {
        "id": table_id,
        "blockid": blockid,
        "heading": heading,
        "parent_headings": list(parent_headings),
        "dimension": [int(table.num_rows), int(table.num_cols)],
        "format": fmt,
        "content": content,
        "caption": table.caption,
        "footnotes": list(table.footnotes),
    }
    if table.table_header is not None:
        if fmt == "html" and isinstance(table.table_header, str):
            item["table_header"] = table.table_header
        elif fmt == "json" and isinstance(table.table_header, list):
            item["table_header"] = json.dumps(table.table_header, ensure_ascii=False)
    if table.self_ref:
        item["self_ref"] = table.self_ref
    if table.extras:
        item["extras"] = dict(table.extras)
    return item


def _drawing_item_dict(
    drawing_id: str, blockid: str, heading: str,
    parent_headings: List[str], drawing: IRDrawing,
    asset_paths: Dict[str, str], asset_prefix: str,
) -> Dict[str, Any]:
    if drawing.path_override is not None:
        path = drawing.path_override
    else:
        filename = asset_paths.get(drawing.asset_ref, "")
        path = f"{asset_prefix}{filename}" if filename else ""
    item: Dict[str, Any] = {
        "id": drawing_id,
        "blockid": blockid,
        "heading": heading,
        "parent_headings": list(parent_headings),
        "format": drawing.fmt,
        "path": path,
        "src": drawing.src,
        "caption": drawing.caption,
        "footnotes": list(drawing.footnotes),
    }
    if drawing.self_ref:
        item["self_ref"] = drawing.self_ref
    if drawing.extras:
        item["extras"] = dict(drawing.extras)
    return item


def _equation_item_dict(
    eq_id: str, blockid: str, heading: str,
    parent_headings: List[str], equation: IREquation,
) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": eq_id,
        "blockid": blockid,
        "heading": heading,
        "parent_headings": list(parent_headings),
        "format": "latex",
        "content": equation.latex.strip(),
        "caption": equation.caption,
        "footnotes": list(equation.footnotes),
    }
    if equation.self_ref:
        item["self_ref"] = equation.self_ref
    if equation.extras:
        item["extras"] = dict(equation.extras)
    return item
