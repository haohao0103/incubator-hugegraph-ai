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

"""Full-text search index backends.

Provides pluggable implementations:
- BM25FullTextBackend: local file-based BM25 (dev/testing)
- OceanBaseFTSBackend: OceanBase FULLTEXT INDEX (production)
"""

from hugegraph_llm.indices.fulltext.base import FullTextBase
from hugegraph_llm.indices.fulltext.bm25_fulltext import BM25FullTextBackend
from hugegraph_llm.indices.fulltext.oceanbase_fulltext import OceanBaseFTSBackend

__all__ = [
    "FullTextBase",
    "BM25FullTextBackend",
    "OceanBaseFTSBackend",
]
