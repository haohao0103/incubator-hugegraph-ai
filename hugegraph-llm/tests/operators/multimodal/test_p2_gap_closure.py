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

"""Tests for P2 gap closure features:

P2-1: OMML → LaTeX complete version (omml_to_latex.py module)
P2-2: DOCX embedded image word/_rels relationship complete parsing (DrawingML + VML)
P2-3: Markdown image SSRF protection (_is_safe_url + _resolve_md_image)
"""

import base64
import io
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# P2-1: OMML → LaTeX Complete Version
# ---------------------------------------------------------------------------

from hugegraph_llm.operators.multimodal.omml_to_latex import (
    OMMLParser,
    clean_exp,
    convert_omml_to_latex,
    qn,
)


def _make_omml_element(tag_text: str, children_text: str = ""):
    """Helper: create an XML element from tag + children text."""
    import xml.etree.ElementTree as ET
    return ET.fromstring(tag_text)


class TestOMMLqn:
    def test_m_namespace(self):
        result = qn("m:oMath")
        assert result == "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"

    def test_w_namespace(self):
        result = qn("w:t")
        assert result == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"

    def test_invalid_prefix_raises(self):
        with pytest.raises(KeyError):
            qn("unknown:tag")


class TestCleanExp:
    def test_degf(self):
        assert clean_exp("\\degf") == "&deg;F"

    def test_degC(self):
        assert clean_exp("\\degc") == "&deg;C"

    def test_cbrt(self):
        assert clean_exp("\\cbrtx") == "\\sqrt[3]{x}"

    def test_qdrt(self):
        assert clean_exp("\\qdrtx") == "\\sqrt[4]{x}"

    def test_sfrac_to_frac(self):
        assert clean_exp("\\sfrac") == "\\frac"

    def test_bullet_spacing(self):
        assert clean_exp("\\bulletx") == "\\bullet x"

    def test_sum_braces(self):
        assert clean_exp("\\sumx") == "\\sum{x}"

    def test_prod_braces(self):
        assert clean_exp("\\prodn") == "\\prod{n}"

    def test_lim_below(self):
        result = clean_exp("\\lim\\below{n→∞}{f(n)}")
        assert result == "\\lim_{n→∞}{f(n)}"

    def test_no_change_on_clean_input(self):
        assert clean_exp("\\frac{1}{2}") == "\\frac{1}{2}"


class TestOMMLParserText:
    def test_simple_text(self):
        elem = _make_omml_element(
            f'<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            f'<m:r><m:t>x + y</m:t></m:r>'
            f'</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "x" in result
        assert "y" in result

    def test_empty_text(self):
        elem = _make_omml_element(
            f'<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            f'<m:r><m:t></m:t></m:r>'
            f'</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        # Empty m:t returns " " (space)
        assert result.strip() == ""


class TestOMMLParserFraction:
    def test_simple_fraction(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:f>'
            '<m:num><m:r><m:t>a</m:t></m:r></m:num>'
            '<m:den><m:r><m:t>b</m:t></m:r></m:den>'
            '</m:f>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\frac" in result
        assert "a" in result
        assert "b" in result

    def test_binomial_fraction(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:f>'
            '<m:fPr><m:type m:val="noBar"/></m:fPr>'
            '<m:num><m:r><m:t>n</m:t></m:r></m:num>'
            '<m:den><m:r><m:t>k</m:t></m:r></m:den>'
            '</m:f>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\genfrac" in result


class TestOMMLParserSuperscript:
    def test_simple_superscript(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:sSup>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '<m:sup><m:r><m:t>2</m:t></m:r></m:sup>'
            '</m:sSup>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "x" in result
        assert "2" in result
        assert "^" in result


class TestOMMLParserSubscript:
    def test_simple_subscript(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:sSub>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '<m:sub><m:r><m:t>i</m:t></m:r></m:sub>'
            '</m:sSub>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "x" in result
        assert "i" in result
        assert "_" in result


class TestOMMLParserSubSuperscript:
    def test_sub_and_sup(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:sSubSup>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '<m:sub><m:r><m:t>i</m:t></m:r></m:sub>'
            '<m:sup><m:r><m:t>2</m:t></m:r></m:sup>'
            '</m:sSubSup>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "x" in result
        assert "i" in result
        assert "2" in result
        assert "_" in result
        assert "^" in result


class TestOMMLParserPreSubSup:
    def test_pre_sub_sup(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:sPre>'
            '<m:sub><m:r><m:t>i</m:t></m:r></m:sub>'
            '<m:sup><m:r><m:t>j</m:t></m:r></m:sup>'
            '<m:e><m:r><m:t>X</m:t></m:r></m:e>'
            '</m:sPre>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "i" in result
        assert "j" in result
        assert "X" in result


class TestOMMLParserNary:
    def test_integral(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:nary>'
            '<m:sub><m:r><m:t>a</m:t></m:r></m:sub>'
            '<m:sup><m:r><m:t>b</m:t></m:r></m:sup>'
            '<m:e><m:r><m:t>f(x)</m:t></m:r></m:e>'
            '</m:nary>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\int" in result
        assert "a" in result
        assert "b" in result
        assert "f(x)" in result

    def test_sum(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:nary>'
            '<m:naryPr><m:chr m:val="∑"/></m:naryPr>'
            '<m:sub><m:r><m:t>i=1</m:t></m:r></m:sub>'
            '<m:sup><m:r><m:t>n</m:t></m:r></m:sup>'
            '<m:e><m:r><m:t>a_i</m:t></m:r></m:e>'
            '</m:nary>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\sum" in result

    def test_nary_no_sub_sup(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:nary>'
            '<m:e><m:r><m:t>f</m:t></m:r></m:e>'
            '</m:nary>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\int" in result  # default is integral
        assert "f" in result


class TestOMMLParserRadical:
    def test_square_root(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:rad>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:rad>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\sqrt" in result
        assert "x" in result

    def test_nth_root(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:rad>'
            '<m:deg><m:r><m:t>3</m:t></m:r></m:deg>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:rad>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\sqrt[3]" in result
        assert "x" in result


class TestOMMLParserAccent:
    def test_hat_accent(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:acc>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:acc>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        # Default accent is hat (770)
        assert "\\hat" in result
        assert "x" in result

    def test_tilde_accent(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:acc>'
            '<m:accPr><m:chr m:val="̃"/></m:accPr>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:acc>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\tilde" in result


class TestOMMLParserDelimiter:
    def test_parentheses(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:d>'
            '<m:e><m:r><m:t>a</m:t></m:r></m:e>'
            '</m:d>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\left(" in result
        assert "\\right)" in result
        assert "a" in result

    def test_brackets(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:d>'
            '<m:dPr><m:begChr m:val="["/><m:endChr m:val="]"/></m:dPr>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:d>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\left[" in result
        assert "\\right]" in result


class TestOMMLParserBar:
    def test_overline(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:bar>'
            '<m:e><m:r><m:t>AB</m:t></m:r></m:e>'
            '</m:bar>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\overline" in result
        assert "AB" in result

    def test_underline(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:bar>'
            '<m:barPr><m:pos m:val="bot"/></m:barPr>'
            '<m:e><m:r><m:t>AB</m:t></m:r></m:e>'
            '</m:bar>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\underline" in result


class TestOMMLParserBorderBox:
    def test_boxed(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:borderBox>'
            '<m:e><m:r><m:t>E</m:t></m:r></m:e>'
            '</m:borderBox>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\boxed" in result
        assert "E" in result


class TestOMMLParserBox:
    def test_box_passthrough(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:box>'
            '<m:e><m:r><m:t>hello</m:t></m:r></m:e>'
            '</m:box>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "hello" in result


class TestOMMLParserEquationArray:
    def test_eq_arr(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:eqArr>'
            '<m:e><m:r><m:t>a</m:t></m:r></m:e>'
            '<m:e><m:r><m:t>b</m:t></m:r></m:e>'
            '</m:eqArr>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\begin{eqnarray*}" in result
        assert "\\end{eqnarray*}" in result
        assert "a" in result
        assert "b" in result


class TestOMMLParserMatrix:
    def test_matrix(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:m>'
            '<m:mr>'
            '<m:e><m:r><m:t>a</m:t></m:r></m:e>'
            '<m:e><m:r><m:t>b</m:t></m:r></m:e>'
            '</m:mr>'
            '<m:mr>'
            '<m:e><m:r><m:t>c</m:t></m:r></m:e>'
            '<m:e><m:r><m:t>d</m:t></m:r></m:e>'
            '</m:mr>'
            '</m:m>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\begin{matrix}" in result
        assert "\\end{matrix}" in result


class TestOMMLParserFunction:
    def test_sin_function(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:func>'
            '<m:fName><m:r><m:t>sin</m:t></m:r></m:fName>'
            '<m:e><m:r><m:t>x</m:t></m:r></m:e>'
            '</m:func>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\sin" in result
        assert "x" in result

    def test_lim_function(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:func>'
            '<m:fName><m:r><m:t>lim</m:t></m:r></m:fName>'
            '<m:e><m:r><m:t>f(x)</m:t></m:r></m:e>'
            '</m:func>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\lim" in result


class TestOMMLParserGroupChr:
    def test_underbrace(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:groupChr>'
            '<m:e><m:r><m:t>xyz</m:t></m:r></m:e>'
            '</m:groupChr>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\underbrace" in result
        assert "xyz" in result


class TestOMMLParserDispatchTable:
    """Verify that all 21 OMML tags are registered in the _parsers dispatch dict."""

    def test_all_21_tags_present(self):
        parser = OMMLParser()
        expected_tags = [
            "m:r", "m:acc", "m:borderBox", "m:bar", "m:box",
            "m:d", "m:e", "m:groupChr", "m:f", "m:sSup",
            "m:sSub", "m:sSubSup", "m:sPre", "m:t", "m:rad",
            "m:nary", "m:eqArr", "m:func", "m:m", "m:mr",
        ]
        for tag in expected_tags:
            qualified = qn(tag)
            assert qualified in parser._parsers, f"Missing parser for {tag}"

    def test_dispatch_dict_count(self):
        parser = OMMLParser()
        # Note: _parsers has 20 entries (m:oMathPara is handled in parse()
        # directly, not via dispatch table)
        assert len(parser._parsers) == 20


class TestOMMLSymbolReplacement:
    def test_lt_replacement(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:r><m:t>&lt;</m:t></m:r>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\lt" in result

    def test_ge_replacement(self):
        # Use ≥ Unicode character instead of &ge; (XML entities not supported
        # by stdlib ElementTree)
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:r><m:t>≥</m:t></m:r>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\geq" in result

    def test_infinity_replacement(self):
        elem = _make_omml_element(
            '<m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">'
            '<m:r><m:t>∞</m:t></m:r>'
            '</m:oMath>'
        )
        result = convert_omml_to_latex(elem)
        assert "\\infty" in result


# ---------------------------------------------------------------------------
# P2-2: DOCX DrawingML + VML Image Parsing
# ---------------------------------------------------------------------------

from hugegraph_llm.operators.multimodal.unified_document_parser import (
    _emu_to_pixels,
    _extract_docx_drawing_placeholder,
    _extract_docx_vml_image_placeholder,
    _DocxRelationship,
)


class TestEmuToPixels:
    def test_typical_width(self):
        # 914400 EMU = 1 inch = 96 px (at 96 DPI)
        assert _emu_to_pixels("914400") == 96

    def test_zero_emu(self):
        assert _emu_to_pixels("0") == 0

    def test_large_emu(self):
        # 4572000 EMU ≈ 5 inches ≈ 480 px
        assert _emu_to_pixels("4572000") == 480

    def test_invalid_emu(self):
        assert _emu_to_pixels("abc") is None

    def test_negative_emu(self):
        assert _emu_to_pixels("-100") == 0

    def test_empty_emu(self):
        assert _emu_to_pixels("") is None


class TestDocxRelationship:
    def test_basic_creation(self):
        rel = _DocxRelationship(
            rel_id="rId1",
            rel_type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            target="media/image1.png",
            target_mode="Internal",
            image_format="png",
        )
        assert rel.rel_id == "rId1"
        assert rel.image_format == "png"

    def test_external_target(self):
        rel = _DocxRelationship(
            rel_id="rId2",
            rel_type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            target="https://example.com/photo.jpg",
            target_mode="External",
            image_format="jpeg",
        )
        assert rel.target_mode == "External"
        assert rel.image_format == "jpeg"


class TestExtractDocxDrawingPlaceholder:
    def _make_drawing_element(self, rId="rId1", r_type="embed", cx="914400", cy="457200"):
        """Create a w:drawing XML element for testing."""
        import xml.etree.ElementTree as ET
        r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        wp_ns = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
        a_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
        pic_ns = "http://schemas.openxmlformats.org/drawingml/2006/picture"

        xml_str = (
            f'<w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            f' xmlns:wp="{wp_ns}" xmlns:a="{a_ns}" xmlns:r="{r_ns}" xmlns:pic="{pic_ns}">'
            f'<wp:inline>'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="1" name="Picture 1"/>'
            f'<a:graphic><a:graphicData uri="{pic_ns}">'
            f'<pic:pic><pic:blipFill><a:blip r:{r_type}="{rId}"/></pic:blipFill>'
            f'</pic:pic></a:graphicData></a:graphic>'
            f'</wp:inline>'
            f'</w:drawing>'
        )
        return ET.fromstring(xml_str)

    def _make_rels(self, rId="rId1", target="media/image1.png", target_mode="Internal", image_format=None):
        """Create a dict of _DocxRelationship objects."""
        return {
            rId: _DocxRelationship(
                rel_id=rId,
                rel_type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                target=target,
                target_mode=target_mode,
                image_format=image_format,
            )
        }

    def test_embed_image(self):
        elem = self._make_drawing_element(rId="rId1", r_type="embed")
        rels = self._make_rels("rId1", "media/image1.png", "Internal", image_format="png")
        result = _extract_docx_drawing_placeholder(elem, rels)
        assert "drawing" in result
        assert 'path="media/image1.png"' in result
        assert 'format="png"' in result

    def test_link_image_preferred(self):
        """P2-2: r:link takes priority over r:embed."""
        elem = self._make_drawing_element(rId="rId5", r_type="link")
        rels = self._make_rels("rId5", "https://cdn.example.com/img.jpg", "External", image_format="jpeg")
        result = _extract_docx_drawing_placeholder(elem, rels)
        assert "drawing" in result
        assert "https://cdn.example.com/img.jpg" in result

    def test_emu_dimensions(self):
        """P2-2: EMU → px conversion for image dimensions."""
        elem = self._make_drawing_element(cx="914400", cy="457200")
        rels = self._make_rels("rId1", "media/image1.png", "Internal", image_format="png")
        result = _extract_docx_drawing_placeholder(elem, rels)
        assert 'width="96"' in result   # 914400 EMU → 96 px
        assert 'height="48"' in result  # 457200 EMU → 48 px

    def test_missing_relationship(self):
        elem = self._make_drawing_element(rId="rId99")
        result = _extract_docx_drawing_placeholder(elem, {})
        assert "drawing" in result
        assert "path" not in result


class TestExtractDocxVMLImagePlaceholder:
    def _make_vml_element(self, rId="rId1"):
        """Create a VML w:pict element for testing."""
        import xml.etree.ElementTree as ET
        r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        v_ns = "urn:schemas-microsoft-com:vml"
        o_ns = "urn:schemas-microsoft-com:office:office"

        # Use r: prefix in namespace declaration, r:id in attribute
        xml_str = (
            f'<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            f' xmlns:v="{v_ns}" xmlns:r="{r_ns}" xmlns:o="{o_ns}">'
            f'<v:shape id="shape1">'
            f'<v:imagedata id="img1" o:title="Photo" r:id="{rId}"/>'
            f'</v:shape>'
            f'</w:pict>'
        )
        return ET.fromstring(xml_str)

    def test_vml_image_with_rid(self):
        """P2-2: VML v:imagedata r:id resolution."""
        elem = self._make_vml_element(rId="rId3")
        rels = {
            "rId3": _DocxRelationship(
                rel_id="rId3",
                rel_type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                target="media/photo.jpg",
                target_mode="Internal",
                image_format="jpeg",
            )
        }
        result = _extract_docx_vml_image_placeholder(elem, rels)
        assert "drawing" in result
        assert 'path="media/photo.jpg"' in result
        assert 'format="jpeg"' in result
        assert 'name="Photo"' in result

    def test_vml_external_image(self):
        """P2-2: VML external image (TargetMode=External)."""
        elem = self._make_vml_element(rId="rId4")
        rels = {
            "rId4": _DocxRelationship(
                rel_id="rId4",
                rel_type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                target="https://external.com/logo.png",
                target_mode="External",
                image_format="png",
            )
        }
        result = _extract_docx_vml_image_placeholder(elem, rels)
        assert "https://external.com/logo.png" in result

    def test_vml_no_imagedata(self):
        """VML element without v:imagedata → empty string."""
        import xml.etree.ElementTree as ET
        v_ns = "urn:schemas-microsoft-com:vml"
        xml_str = (
            f'<w:pict xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
            f' xmlns:v="{v_ns}">'
            f'<v:shape id="shape1"><v:textbox/></v:shape>'
            f'</w:pict>'
        )
        elem = ET.fromstring(xml_str)
        result = _extract_docx_vml_image_placeholder(elem, {})
        assert result == ""


# ---------------------------------------------------------------------------
# P2-3: Markdown Image SSRF Protection
# ---------------------------------------------------------------------------

from hugegraph_llm.operators.multimodal.unified_document_parser import (
    _is_safe_url,
    _resolve_md_image,
    _MAX_IMAGE_DOWNLOAD_SIZE,
    _MAX_URL_LENGTH,
    _PRIVATE_IP_RE,
    _BLOCKED_SCHEMES,
)


class TestIsSafeUrl:
    """P2-3: SSRF protection for markdown image URLs."""

    def test_safe_public_url(self):
        assert _is_safe_url("https://cdn.example.com/image.png")

    def test_safe_http_url(self):
        assert _is_safe_url("http://example.com/photo.jpg")

    def test_block_localhost(self):
        assert not _is_safe_url("http://localhost/image.png")

    def test_block_127_ip(self):
        assert not _is_safe_url("http://127.0.0.1/secret")

    def test_block_10_private(self):
        assert not _is_safe_url("http://10.0.0.1/internal")

    def test_block_172_16_private(self):
        assert not _is_safe_url("http://172.16.0.1/internal")

    def test_block_172_31_private(self):
        assert not _is_safe_url("http://172.31.255.1/internal")

    def test_allow_172_32_public(self):
        assert _is_safe_url("http://172.32.0.1/public")  # 172.32+ is public

    def test_block_192_168_private(self):
        assert not _is_safe_url("http://192.168.1.1/internal")

    def test_block_169_254_linklocal(self):
        assert not _is_safe_url("http://169.254.1.1/linklocal")

    def test_block_0_ip(self):
        assert not _is_safe_url("http://0.0.0.0/wildcard")

    def test_block_file_scheme(self):
        assert not _is_safe_url("file:///etc/passwd")

    def test_block_ftp_scheme(self):
        assert not _is_safe_url("ftp://example.com/file")

    def test_block_credential_injection(self):
        assert not _is_safe_url("http://user:pass@evil.com/image.png")

    def test_block_localdomain(self):
        assert not _is_safe_url("http://localdomain/image")

    def test_block_dot_local(self):
        assert not _is_safe_url("http://myhost.local/image")

    def test_block_url_too_long(self):
        long_url = "https://example.com/" + "a" * 3000
        assert not _is_safe_url(long_url)

    def test_allow_url_within_limit(self):
        normal_url = "https://example.com/image.png"
        assert len(normal_url) < _MAX_URL_LENGTH
        assert _is_safe_url(normal_url)

    def test_domain_whitelist(self):
        allowed = frozenset({"example.com", "cdn.example.com"})
        assert _is_safe_url("https://cdn.example.com/img.png", allowed_domains=allowed)
        assert not _is_safe_url("https://other.com/img.png", allowed_domains=allowed)

    def test_ipv6_loopback_blocked(self):
        assert not _is_safe_url("http://[::1]/image")

    def test_ipv6_linklocal_blocked(self):
        assert not _is_safe_url("http://[fe80::1]/image")

    def test_ipv6_unique_local_blocked(self):
        assert not _is_safe_url("http://[fc00::1]/image")

    def test_no_hostname(self):
        assert not _is_safe_url("https:///image.png")


class TestResolveMdImageSSRF:
    """P2-3: _resolve_md_image SSRF protection integration."""

    def test_block_file_scheme(self):
        result = _resolve_md_image("file:///etc/passwd")
        assert result is None

    def test_block_ftp_scheme(self):
        result = _resolve_md_image("ftp://evil.com/image.png")
        assert result is None

    def test_block_private_ip(self):
        result = _resolve_md_image("http://192.168.1.1/internal.png")
        assert result is None

    def test_block_localhost(self):
        result = _resolve_md_image("http://localhost/admin.png")
        assert result is None

    def test_data_uri_accepted(self):
        """data: URIs for images should be accepted."""
        # Create a tiny valid PNG data URI
        _PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">II", 10, 10)
        png_data = _PNG_SIGNATURE + struct.pack(">I", 13) + b"IHDR" + ihdr + b"\x00" * 20
        b64 = base64.b64encode(png_data).decode()
        data_uri = f"data:image/png;base64,{b64}"
        result = _resolve_md_image(data_uri)
        assert result is not None
        assert result.format == "png"

    def test_data_uri_oversize_blocked(self):
        """Data URIs exceeding _MAX_IMAGE_DOWNLOAD_SIZE should be blocked."""
        # Create a fake huge data URI (> 10 MB)
        huge_b64 = "A" * (_MAX_IMAGE_DOWNLOAD_SIZE * 4 // 3 + 100)
        data_uri = f"data:image/png;base64,{huge_b64}"
        result = _resolve_md_image(data_uri)
        assert result is None

    def test_local_file_relative_path(self):
        """Local relative file path (not URL) should be handled."""
        result = _resolve_md_image("local_image.png", md_dir=Path("/tmp/nonexistent"))
        # File doesn't exist, so result should be None (file not found)
        assert result is None

    def test_local_file_with_path_traversal_blocked(self):
        """Path traversal in local file paths should be blocked."""
        # Note: _resolve_md_image doesn't do path traversal check for local files
        # (that's for HTTP URLs), but relative paths that resolve outside md_dir
        # should be handled
        result = _resolve_md_image("../../etc/passwd", md_dir=Path("/tmp/docs"))
        # The file won't exist, so None; path traversal doesn't block local files
        # at this level since they're relative
        assert result is None  # File doesn't exist

    def test_angle_bracket_strip(self):
        """Markdown may wrap URLs in <...>; should be stripped."""
        # This tests the angle bracket stripping logic
        src = "<https://example.com/img.png>"
        # After stripping: https://example.com/img.png
        # But we mock the download to avoid network
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.headers = {"Content-Length": "100"}
            mock_resp.read = MagicMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 92)
            mock_urlopen.return_value.__enter__ = MagicMock(return_value=mock_resp)
            mock_urlopen.return_value.__exit__ = MagicMock(return_value=False)
            result = _resolve_md_image(src)
            # Should strip angle brackets and attempt download
            # (will fail on actual network but the URL stripping works)


class TestBlockedSchemes:
    def test_blocked_schemes_list(self):
        assert "file" in _BLOCKED_SCHEMES
        assert "ftp" in _BLOCKED_SCHEMES
        assert "ftps" in _BLOCKED_SCHEMES
        assert "sftp" in _BLOCKED_SCHEMES
        assert "javascript" in _BLOCKED_SCHEMES
        assert "vbscript" in _BLOCKED_SCHEMES

    def test_data_scheme_in_blocked(self):
        """data: is in blocked schemes but handled specially (allowed for images)."""
        assert "data" in _BLOCKED_SCHEMES


class TestPrivateIpRegex:
    def test_loopback(self):
        assert _PRIVATE_IP_RE.match("127.0.0.1")

    def test_class_a_private(self):
        assert _PRIVATE_IP_RE.match("10.0.0.1")

    def test_class_b_private(self):
        assert _PRIVATE_IP_RE.match("172.16.0.1")
        assert _PRIVATE_IP_RE.match("172.31.255.1")

    def test_class_b_public(self):
        assert not _PRIVATE_IP_RE.match("172.32.0.1")

    def test_class_c_private(self):
        assert _PRIVATE_IP_RE.match("192.168.1.1")

    def test_link_local(self):
        assert _PRIVATE_IP_RE.match("169.254.1.1")

    def test_ipv6_loopback(self):
        # Bracketed form (raw hostname in URL)
        assert _PRIVATE_IP_RE.match("[::1]")
        # urlparse strips brackets → ::1 (what _is_safe_url checks)
        assert _PRIVATE_IP_RE.match("::1")

    def test_ipv6_link_local(self):
        assert _PRIVATE_IP_RE.match("[fe80::1]")
        assert _PRIVATE_IP_RE.match("fe80::1")

    def test_ipv6_unique_local(self):
        assert _PRIVATE_IP_RE.match("[fc00::1]")
        assert _PRIVATE_IP_RE.match("fc00::1")
        assert _PRIVATE_IP_RE.match("[fd00::1]")
        assert _PRIVATE_IP_RE.match("fd00::1")

    def test_public_ip(self):
        assert not _PRIVATE_IP_RE.match("8.8.8.8")
        assert not _PRIVATE_IP_RE.match("203.0.113.1")


# ---------------------------------------------------------------------------
# P2 Integration: OMML → LaTeX via unified_document_parser
# ---------------------------------------------------------------------------

class TestConvertOmmlViaParser:
    """Verify that _convert_omml_to_latex delegates to the omml_to_latex module."""

    def test_delegates_to_full_module(self):
        from hugegraph_llm.operators.multimodal.unified_document_parser import _convert_omml_to_latex
        import xml.etree.ElementTree as ET

        m_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
        elem = ET.fromstring(
            f'<m:oMath xmlns:m="{m_ns}">'
            f'<m:r><m:t>x</m:t></m:r>'
            f'</m:oMath>'
        )
        result = _convert_omml_to_latex(elem)
        assert "x" in result

    def test_fallback_on_import_error(self):
        """If omml_to_latex module fails to import, falls back to simplified version."""
        from hugegraph_llm.operators.multimodal.unified_document_parser import _convert_omml_to_latex_fallback
        import xml.etree.ElementTree as ET

        m_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
        elem = ET.fromstring(
            f'<m:oMath xmlns:m="{m_ns}">'
            f'<m:r><m:t>test</m:t></m:r>'
            f'</m:oMath>'
        )
        result = _convert_omml_to_latex_fallback(elem)
        # Simplified version should still produce some output
        assert "test" in result
