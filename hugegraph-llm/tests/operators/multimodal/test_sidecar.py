# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.

"""Tests for multimodal sidecar IR, placeholder, writer modules."""

import json
import tempfile
from pathlib import Path

import pytest

from hugegraph_llm.operators.multimodal.sidecar_ir import (
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRPosition,
    IRTable,
    AssetSpec,
)
from hugegraph_llm.operators.multimodal.sidecar_placeholder import (
    render_table_tag,
    render_drawing_tag,
    render_equation_tag,
    render_template,
    table_body_for_rows,
    xml_attr_escape,
    caption_attr,
    _TOKEN_RE,
)
from hugegraph_llm.operators.multimodal.sidecar_writer import (
    write_sidecar,
    _materialize_assets,
    _allocate_unique_name,
    _safe_asset_filename,
)


# ============================================================================
# IR Dataclass Tests
# ============================================================================

class TestIRDataclasses:
    def test_ir_position_to_jsonable(self):
        pos = IRPosition(type="bbox", anchor=[10, 20, 30, 40], origin="LEFTTOP")
        result = pos.to_jsonable()
        assert result["type"] == "bbox"
        assert result["anchor"] == [10, 20, 30, 40]
        assert result["origin"] == "LEFTTOP"

    def test_ir_position_skips_none_fields(self):
        pos = IRPosition(type="heading")
        result = pos.to_jsonable()
        assert "anchor" not in result
        assert "range" not in result
        assert "charspan" not in result
        assert "origin" not in result

    def test_ir_table_defaults(self):
        tbl = IRTable(placeholder_key="t1")
        assert tbl.rows is None
        assert tbl.html is None
        assert tbl.caption == ""
        assert tbl.footnotes == []
        assert tbl.extras == {}

    def test_ir_drawing_defaults(self):
        drw = IRDrawing(placeholder_key="d1", asset_ref="a1")
        assert drw.fmt == ""
        assert drw.caption == ""
        assert drw.path_override is None

    def test_ir_equation_inline(self):
        eq = IREquation(placeholder_key="eq1", latex="x^2", is_block=False)
        assert eq.is_block is False
        assert eq.caption == ""

    def test_ir_block_with_items(self):
        tbl = IRTable(placeholder_key="t1", rows=[["a", "b"]], num_rows=1, num_cols=2)
        drw = IRDrawing(placeholder_key="d1", asset_ref="img1", fmt="png")
        eq = IREquation(placeholder_key="eq1", latex="E=mc^2")
        block = IRBlock(
            content_template="See {{TBL:t1}} and {{IMG:d1}}. Formula: {{EQ:eq1}}",
            heading="Section 1",
            level=1,
            tables=[tbl],
            drawings=[drw],
            equations=[eq],
        )
        assert len(block.tables) == 1
        assert len(block.drawings) == 1
        assert len(block.equations) == 1

    def test_ir_doc(self):
        doc = IRDoc(
            document_name="test.pdf",
            document_format="pdf",
            doc_title="Test Document",
            blocks=[IRBlock(content_template="Hello world")],
        )
        assert doc.document_name == "test.pdf"
        assert len(doc.blocks) == 1

    def test_asset_spec(self):
        spec = AssetSpec(ref="img1", suggested_name="fig1.png", source=None)
        assert spec.ref == "img1"
        assert spec.source is None


# ============================================================================
# Placeholder Tests
# ============================================================================

class TestPlaceholders:
    def test_xml_attr_escape(self):
        assert xml_attr_escape("a<b>c&d\"e") == "a&lt;b&gt;c&amp;d&quot;e"

    def test_caption_attr_present(self):
        result = caption_attr("Table 1")
        assert result == ' caption="Table 1"'

    def test_caption_attr_empty(self):
        assert caption_attr("") == ""

    def test_render_table_tag(self):
        result = render_table_tag("tb-abc-0001", "json", '[["a","b"]]')
        assert 'id="tb-abc-0001"' in result
        assert 'format="json"' in result
        assert '[["a","b"]]' in result

    def test_render_drawing_tag(self):
        result = render_drawing_tag("im-abc-0001", "png", "Fig 1", "assets/fig.png", "")
        assert 'id="im-abc-0001"' in result
        assert 'format="png"' in result
        assert 'caption="Fig 1"' in result
        assert 'path="assets/fig.png"' in result

    def test_render_equation_tag_block(self):
        result = render_equation_tag("eq-abc-0001", "E=mc^2", "Energy formula")
        assert 'id="eq-abc-0001"' in result
        assert 'format="latex"' in result
        assert "E=mc^2" in result
        assert 'caption="Energy formula"' in result

    def test_render_equation_tag_inline(self):
        result = render_equation_tag(None, "x^2")
        assert "id=" not in result
        assert 'format="latex"' in result
        assert "x^2" in result

    def test_table_body_for_rows(self):
        rows = [["Name", "Age"], ["Alice", "30"]]
        result = table_body_for_rows(rows)
        parsed = json.loads(result)
        assert parsed == rows

    def test_render_template_all_tokens(self):
        template = "Before {{TBL:t1}} mid {{IMG:d1}} after {{EQ:eq1}} inline {{EQI:ie1}} end"
        result = render_template(
            template,
            table_renderer=lambda k: f"[TABLE:{k}]",
            drawing_renderer=lambda k: f"[DRAW:{k}]",
            equation_renderer=lambda k: f"[EQ:{k}]",
            inline_equation_renderer=lambda k: f"[IEQ:{k}]",
        )
        assert "[TABLE:t1]" in result
        assert "[DRAW:d1]" in result
        assert "[EQ:eq1]" in result
        assert "[IEQ:ie1]" in result
        assert "Before" in result
        assert "end" in result

    def test_token_regex_matches(self):
        assert _TOKEN_RE.match("{{TBL:t1}}")
        assert _TOKEN_RE.match("{{IMG:d1}}")
        assert _TOKEN_RE.match("{{EQ:eq1}}")
        assert _TOKEN_RE.match("{{EQI:ie1}}")

    def test_token_regex_no_match(self):
        assert _TOKEN_RE.match("{{OTHER:x}}") is None


# ============================================================================
# Writer Tests
# ============================================================================

class TestWriter:
    def test_write_sidecar_basic(self):
        """Test writing a minimal IRDoc to parsed directory."""
        doc = IRDoc(
            document_name="test.txt",
            document_format="text",
            doc_title="Test",
            blocks=[
                IRBlock(content_template="Hello world", heading="Intro", level=1),
                IRBlock(content_template="Second block", heading="Body", level=2),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir) / "test.parsed"
            result = write_sidecar(
                doc,
                parsed_dir=parsed_dir,
                doc_id="doc-abcdef1234567890",
                engine="native",
            )

            assert result["doc_id"] == "doc-abcdef1234567890"
            assert result["content"] == "Hello world\n\nSecond block"
            assert parsed_dir.exists()

            # Check blocks.jsonl exists and has content
            blocks_path = parsed_dir / "test.blocks.jsonl"
            assert blocks_path.exists()
            lines = blocks_path.read_text().strip().split("\n")
            assert len(lines) == 3  # meta + 2 content blocks

            # Check first line is meta
            meta = json.loads(lines[0])
            assert meta["type"] == "meta"
            assert meta["format"] == "lightrag"
            assert meta["blocks"] == 2

    def test_write_sidecar_with_table(self):
        """Test writing an IRDoc with a table."""
        tbl = IRTable(
            placeholder_key="t1",
            rows=[["Name", "Age"], ["Alice", "30"]],
            num_rows=2,
            num_cols=2,
            caption="People table",
        )
        block = IRBlock(
            content_template="Data: {{TBL:t1}}",
            heading="Section 1",
            tables=[tbl],
        )
        doc = IRDoc(
            document_name="data.txt",
            document_format="text",
            doc_title="Data Doc",
            blocks=[block],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir) / "data.parsed"
            result = write_sidecar(doc, parsed_dir=parsed_dir, doc_id="doc-hash123")

            # Check tables.json exists
            tables_path = parsed_dir / "data.tables.json"
            assert tables_path.exists()
            tables_data = json.loads(tables_path.read_text())
            assert "tables" in tables_data
            assert len(tables_data["tables"]) == 1

            # Check table item has correct fields
            tbl_item = list(tables_data["tables"].values())[0]
            assert tbl_item["format"] == "json"
            assert tbl_item["caption"] == "People table"
            assert tbl_item["dimension"] == [2, 2]

    def test_write_sidecar_with_drawing(self):
        """Test writing an IRDoc with a drawing."""
        asset = AssetSpec(ref="img1", suggested_name="fig.png", source=b"PNG_DATA")
        drw = IRDrawing(
            placeholder_key="d1",
            asset_ref="img1",
            fmt="png",
            caption="Figure 1",
        )
        block = IRBlock(
            content_template="See {{IMG:d1}}",
            drawings=[drw],
        )
        doc = IRDoc(
            document_name="fig.txt",
            document_format="text",
            doc_title="Figure Doc",
            blocks=[block],
            assets=[asset],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir) / "fig.parsed"
            result = write_sidecar(doc, parsed_dir=parsed_dir, doc_id="doc-fig123")

            # Check drawings.json exists
            drawings_path = parsed_dir / "fig.drawings.json"
            assert drawings_path.exists()
            drawings_data = json.loads(drawings_path.read_text())
            assert len(drawings_data["drawings"]) == 1

            # Check asset was materialized
            assets_dir = parsed_dir / "fig.blocks.assets"
            assert assets_dir.exists()

    def test_write_sidecar_with_equation(self):
        """Test writing an IRDoc with a block equation."""
        eq = IREquation(
            placeholder_key="eq1",
            latex="E=mc^2",
            caption="Energy-mass equivalence",
        )
        block = IRBlock(
            content_template="Formula: {{EQ:eq1}}",
            equations=[eq],
        )
        doc = IRDoc(
            document_name="math.txt",
            document_format="text",
            doc_title="Math Doc",
            blocks=[block],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir) / "math.parsed"
            result = write_sidecar(doc, parsed_dir=parsed_dir, doc_id="doc-math123")

            equations_path = parsed_dir / "math.equations.json"
            assert equations_path.exists()
            eq_data = json.loads(equations_path.read_text())
            assert len(eq_data["equations"]) == 1
            eq_item = list(eq_data["equations"].values())[0]
            assert eq_item["format"] == "latex"
            assert eq_item["content"] == "E=mc^2"

    def test_write_sidecar_empty_blocks_skipped(self):
        """Empty blocks should be dropped."""
        doc = IRDoc(
            document_name="empty.txt",
            document_format="text",
            doc_title="Empty Doc",
            blocks=[
                IRBlock(content_template="  "),  # whitespace-only
                IRBlock(content_template="Content"),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed_dir = Path(tmpdir) / "empty.parsed"
            result = write_sidecar(doc, parsed_dir=parsed_dir, doc_id="doc-empty123")
            assert result["content"] == "Content"

    def test_write_sidecar_invalid_path_style_raises(self):
        doc = IRDoc(document_name="x.txt", document_format="text", doc_title="X", blocks=[])
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="block_drawing_path_style"):
                write_sidecar(doc, parsed_dir=Path(tmpdir)/"x.parsed",
                             doc_id="doc-x", block_drawing_path_style="invalid")


# ============================================================================
# Asset Materialization Tests
# ============================================================================

class TestAssetMaterialization:
    def test_materialize_bytes_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            result = _materialize_assets(
                [AssetSpec(ref="img1", suggested_name="fig.png", source=b"PNG_BYTES")],
                assets_dir,
            )
            assert "img1" in result
            assert result["img1"] == "fig.png"
            assert (assets_dir / "fig.png").exists()
            assert (assets_dir / "fig.png").read_bytes() == b"PNG_BYTES"

    def test_materialize_path_asset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            src = Path(tmpdir) / "source.png"
            src.write_bytes(b"SRC_DATA")
            assets_dir = Path(tmpdir) / "assets"
            result = _materialize_assets(
                [AssetSpec(ref="img1", suggested_name="fig.png", source=src)],
                assets_dir,
            )
            assert "img1" in result
            assert (assets_dir / "fig.png").read_bytes() == b"SRC_DATA"

    def test_materialize_none_asset_missing_warns(self):
        """AssetSpec with source=None expects file already in place."""
        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            result = _materialize_assets(
                [AssetSpec(ref="img1", suggested_name="fig.png", source=None)],
                assets_dir,
            )
            # File not in place, so ref is skipped
            assert "img1" not in result

    def test_allocate_unique_name(self):
        used = {"fig.png"}
        result = _allocate_unique_name("fig.png", used)
        assert result == "fig-2.png"

    def test_allocate_unique_name_first_use(self):
        used = set()
        result = _allocate_unique_name("fig.png", used)
        assert result == "fig.png"

    def test_safe_asset_filename(self):
        assert _safe_asset_filename("normal.png") == "normal.png"
        assert _safe_asset_filename("path/to/file.png") == "file.png"
        assert _safe_asset_filename("") == "asset"
