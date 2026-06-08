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

"""BM25 keyword index for full-text retrieval.

.. note::
    This module re-exports BM25FullTextBackend from ``indices.fulltext``
    for backward compatibility. For new code, use the ``fulltext`` package
    directly with configurable backends:

    ::

        from hugegraph_llm.indices.fulltext import BM25FullTextBackend
        from hugegraph_llm.indices.fulltext import OceanBaseFTSBackend

The ``BM25Index`` class is an alias for ``BM25FullTextBackend``.
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Union

import jieba

from hugegraph_llm.config import resource_path
from hugegraph_llm.indices.fulltext.base import FullTextBase
from hugegraph_llm.indices.fulltext.bm25_fulltext import (
    BM25FullTextBackend,
)

log = logging.getLogger(__name__)

BM25_INDEX_FILE = "bm25_index.json"
BM25_DOCS_FILE = "bm25_docs.json"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> List[str]:
    """Tokenize text using jieba for Chinese and whitespace for others.

    Strips punctuation and normalizes whitespace. Handles mixed
    Chinese/English text by applying jieba segmentation first,
    then splitting English tokens on whitespace/punctuation.

    Args:
        text: Input text to tokenize.

    Returns:
        List of lowercase tokens.
    """
    if not text or not text.strip():
        return []
    raw_tokens = jieba.lcut(text)
    cleaned = []
    for tok in raw_tokens:
        tok = tok.strip().lower()
        if tok and re.match(r"^[\w\u4e00-\u9fff]+$", tok):
            cleaned.append(tok)
    return cleaned


# BM25Index = BM25FullTextBackend (backward-compatible alias)
BM25Index = BM25FullTextBackend
