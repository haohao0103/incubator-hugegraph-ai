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

"""Unit tests for the Feishu connector block->markdown parsing.

These tests exercise the pure ``blocks_to_markdown`` helper (no lark SDK / no
network), which is the core text-fidelity logic of the connector.
"""

import pytest

from hugegraph_llm.document.feishu_connector import (
    BLOCK_BULLET,
    BLOCK_CODE,
    BLOCK_HEADING1,
    BLOCK_HEADING2,
    BLOCK_ORDERED,
    BLOCK_PAGE,
    BLOCK_QUOTE,
    BLOCK_TABLE,
    BLOCK_TABLE_CELL,
    BLOCK_TEXT,
    blocks_to_markdown,
)

pytestmark = pytest.mark.unit


def _block(block_id, block_type, text="", children=None, **extra):
    block = {
        "block_id": block_id,
        "block_type": block_type,
        "children": children or [],
        "text": text,
    }
    block.update(extra)
    return block


def test_headings_and_paragraphs():
    blocks = [
        _block("p1", BLOCK_PAGE, children=["h1", "t1"]),
        _block("h1", BLOCK_HEADING1, "第一章 概述"),
        _block("t1", BLOCK_TEXT, "这是正文段落。"),
    ]
    md = blocks_to_markdown(blocks)
    assert "# 第一章 概述" in md
    assert "这是正文段落。" in md


def test_heading_levels():
    blocks = [
        _block("h1", BLOCK_HEADING1, "一级"),
        _block("h2", BLOCK_HEADING2, "二级"),
    ]
    md = blocks_to_markdown(blocks)
    assert "# 一级" in md
    assert "## 二级" in md


def test_bullet_and_ordered_lists():
    blocks = [
        _block("b1", BLOCK_BULLET, "条目A"),
        _block("b2", BLOCK_BULLET, "条目B"),
        _block("o1", BLOCK_ORDERED, "第一步"),
        _block("o2", BLOCK_ORDERED, "第二步"),
    ]
    md = blocks_to_markdown(blocks)
    assert "- 条目A" in md
    assert "- 条目B" in md
    assert "1. 第一步" in md
    assert "2. 第二步" in md


def test_quote_and_code():
    blocks = [
        _block("q1", BLOCK_QUOTE, "引用的内容"),
        _block("c1", BLOCK_CODE, "print('hello')"),
    ]
    md = blocks_to_markdown(blocks)
    assert "> 引用的内容" in md
    assert "```" in md
    assert "print('hello')" in md


def test_table_rendering():
    blocks = [
        _block("tbl", BLOCK_TABLE, "", children=["c00", "c01", "c10", "c11"], row_size=2, column_size=2),
        _block("c00", BLOCK_TABLE_CELL, "表头1"),
        _block("c01", BLOCK_TABLE_CELL, "表头2"),
        _block("c10", BLOCK_TABLE_CELL, "值1"),
        _block("c11", BLOCK_TABLE_CELL, "值2"),
    ]
    md = blocks_to_markdown(blocks)
    assert "| 表头1 | 表头2 |" in md
    assert "| --- | --- |" in md
    assert "| 值1 | 值2 |" in md


def test_nested_bullet_children():
    blocks = [
        _block("parent", BLOCK_BULLET, "父条目", children=["child"]),
        _block("child", BLOCK_BULLET, "子条目"),
    ]
    md = blocks_to_markdown(blocks)
    assert "- 父条目" in md
    assert "- 子条目" in md
