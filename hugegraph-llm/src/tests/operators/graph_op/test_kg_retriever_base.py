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

import unittest
from typing import Any

import pytest

from hugegraph_llm.operators.graph_op.kg_retriever_base import (
    KGRetriever,
    RetrieverResult,
    RetrieverResultItem,
)

pytestmark = [pytest.mark.unit]


class _ListRetriever(KGRetriever):
    """Minimal subclass returning a plain list."""

    def __init__(self, items):
        self._items = items

    def get_search_results(self, query, **kwargs):
        return self._items


class _ObjRetriever(KGRetriever):
    """Subclass returning an object exposing provenance + chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def get_search_results(self, query, **kwargs):
        return _FakeRawResult(self._chunks)


class _FakeRawResult:
    def __init__(self, chunks):
        self.chunks = chunks
        self.provenance = {"original_query": "q", "chunk_count": len(chunks)}


class _TextItem:
    """Chunk-like item with text/score/chunk_id attributes."""

    def __init__(self, text, score=1.0, chunk_id="c1"):
        self.text = text
        self.score = score
        self.chunk_id = chunk_id


class _PassthroughRetriever(KGRetriever):
    """Subclass returning a RetrieverResult directly."""

    def get_search_results(self, query, **kwargs):
        return RetrieverResult(
            items=[RetrieverResultItem(content="direct", metadata={"a": 1})],
            metadata={"origin": "passthrough"},
        )


class TestRetrieverResultItem(unittest.TestCase):
    def test_defaults(self):
        item = RetrieverResultItem(content="x")
        self.assertEqual(item.metadata, {})
        self.assertIsNone(item.score)

    def test_to_dict(self):
        item = RetrieverResultItem(content="x", metadata={"k": "v"}, score=0.9)
        self.assertEqual(
            item.to_dict(), {"content": "x", "metadata": {"k": "v"}, "score": 0.9}
        )


class TestRetrieverResult(unittest.TestCase):
    def test_to_dict(self):
        result = RetrieverResult(
            items=[RetrieverResultItem(content="a")], metadata={"m": 1}
        )
        self.assertEqual(
            result.to_dict(),
            {"items": [{"content": "a", "metadata": {}, "score": None}], "metadata": {"m": 1}},
        )

    def test_is_empty(self):
        self.assertTrue(RetrieverResult().is_empty)
        self.assertFalse(
            RetrieverResult(items=[RetrieverResultItem(content="a")]).is_empty
        )


class TestGetParameters(unittest.TestCase):
    def test_infers_required_and_optional(self):
        class _P(KGRetriever):
            def get_search_results(self, query: str, rewrite=None, **kwargs):
                return []

        params = _P().get_parameters(parameter_descriptions={"query": "the question"})
        self.assertEqual(params["required"], ["query"])
        self.assertEqual(params["properties"]["query"]["type"], "str")
        self.assertEqual(params["properties"]["query"]["description"], "the question")
        self.assertTrue(params["properties"]["query"]["required"])
        self.assertFalse(params["properties"]["rewrite"]["required"])
        self.assertNotIn("kwargs", params["properties"])

    def test_skips_args_and_normalizes_types(self):
        class _P(KGRetriever):
            def get_search_results(self, *args, query: Any = "x"):
                return []

        params = _P().get_parameters()
        self.assertNotIn("args", params["properties"])
        self.assertEqual(params["properties"]["query"]["type"], "string")  # Any -> string

    def test_returns_empty_for_no_params(self):
        class _P(KGRetriever):
            def get_search_results(self):
                return []

        params = _P().get_parameters()
        self.assertEqual(params, {"properties": {}, "required": []})


class TestKGRetriever(unittest.TestCase):
    def test_abstract_cannot_instantiate(self):
        with self.assertRaises(TypeError):
            KGRetriever()  # type: ignore[abstract]

    def test_search_over_list(self):
        r = _ListRetriever([RetrieverResultItem(content="a", score=0.5)])
        result = r.search("q")
        self.assertIsInstance(result, RetrieverResult)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].content, "a")
        self.assertEqual(result.metadata["__retriever"], "_ListRetriever")

    def test_search_empty_list(self):
        result = _ListRetriever([]).search("q")
        self.assertTrue(result.is_empty)

    def test_search_none_result(self):
        class _NoneRetriever(KGRetriever):
            def get_search_results(self, query, **kwargs):
                return None

        self.assertTrue(_NoneRetriever().search("q").is_empty)

    def test_search_object_with_chunks_and_provenance(self):
        r = _ObjRetriever([_TextItem(text="hello", score=0.8)])
        result = r.search("q")
        self.assertEqual(result.items[0].content, "hello")
        self.assertEqual(result.items[0].score, 0.8)
        self.assertEqual(result.items[0].metadata, {"id": "c1"})
        self.assertEqual(result.metadata["original_query"], "q")
        self.assertEqual(result.metadata["chunk_count"], 1)
        self.assertEqual(result.metadata["__retriever"], "_ObjRetriever")

    def test_search_retriever_result_passthrough(self):
        result = _PassthroughRetriever().search("q")
        self.assertEqual(result.items[0].content, "direct")
        self.assertEqual(result.metadata["origin"], "passthrough")
        self.assertEqual(result.metadata["__retriever"], "_PassthroughRetriever")

    def test_search_custom_formatter(self):
        class _FmtRetriever(KGRetriever):
            def get_search_results(self, query, **kwargs):
                return ["raw1", "raw2"]

            def get_result_formatter(self):
                return lambda item: RetrieverResultItem(content=f"formatted-{item}")

        result = _FmtRetriever().search("q")
        self.assertEqual([i.content for i in result.items], ["formatted-raw1", "formatted-raw2"])

    def test_default_formatter_variants(self):
        r = _ListRetriever([])
        # RetrieverResultItem passthrough
        item = RetrieverResultItem(content="x", metadata={"m": 1}, score=1.0)
        converted = r._default_formatter(item)
        self.assertIs(converted, item)
        # object with content attr
        class _WithContent:
            content = "c"
            metadata = {"k": 2}
            score = 0.3

        converted = r._default_formatter(_WithContent())
        self.assertEqual(converted.content, "c")
        self.assertEqual(converted.metadata, {"k": 2})
        self.assertEqual(converted.score, 0.3)
        # text-like object
        converted = r._default_formatter(_TextItem("t", score=0.7, chunk_id="c9"))
        self.assertEqual(converted.content, "t")
        self.assertEqual(converted.metadata, {"id": "c9"})
        self.assertEqual(converted.score, 0.7)
        # plain string
        converted = r._default_formatter("plain")
        self.assertEqual(converted.content, "plain")

    def test_result_items_variants(self):
        r = _ListRetriever([])
        self.assertEqual(r._result_items(None), [])
        self.assertEqual(r._result_items(("a", "b")), ["a", "b"])
        self.assertEqual(r._result_items(["x"]), ["x"])
        obj = _FakeRawResult(["c1"])
        self.assertEqual(r._result_items(obj), ["c1"])
        result = RetrieverResult(items=[RetrieverResultItem(content="z")])
        self.assertEqual(r._result_items(result), result.items)
        # generic iterable fallback
        self.assertEqual(r._result_items("abc"), ["a", "b", "c"])

    def test_result_metadata_variants(self):
        r = _ListRetriever([])
        obj = _FakeRawResult([])
        self.assertEqual(r._result_metadata(obj), {"original_query": "q", "chunk_count": 0})
        result = RetrieverResult(items=[], metadata={"m": 1})
        self.assertEqual(r._result_metadata(result), {"m": 1})
        self.assertEqual(r._result_metadata(["x"]), {})


if __name__ == "__main__":
    unittest.main()
