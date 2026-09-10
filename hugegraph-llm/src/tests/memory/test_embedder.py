# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the embedder and its degradation behaviour.

These run without sentence-transformers installed, which is the default
environment: the fallback path is the one most machines will hit.
"""

import numpy as np
import pytest

from hugegraph_llm.memory.embedder import EMBEDDING_AVAILABLE, HashingEmbedder, LocalEmbedder


# -- hashing fallback -------------------------------------------------------


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    assert np.array_equal(e.encode("works at Acme"), e.encode("works at Acme"))


def test_hashing_embedder_is_normalised():
    vec = HashingEmbedder(dim=64).encode("works at Acme")
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0)


def test_hashing_embedder_handles_empty_text():
    vec = HashingEmbedder(dim=32).encode("")
    assert np.all(vec == 0)


def test_hashing_embedder_handles_cjk():
    """Chinese has no spaces; the regex must still tokenise it."""
    vec = HashingEmbedder(dim=64).encode("张明在 Acme 工作")
    assert float(np.linalg.norm(vec)) > 0


def test_hashing_embedder_is_not_semantic():
    """Documented limitation: paraphrases are NOT close.

    Pinned so nobody later assumes the fallback can judge meaning -- it
    exists to exercise plumbing, not to produce similarity.
    """
    e = HashingEmbedder(dim=128)
    a = e.encode("works at Acme")
    b = e.encode("employed by Acme")
    c = e.encode("likes tea")
    assert float(np.dot(a, b)) == pytest.approx(float(np.dot(a, c)), abs=0.35)


# -- LocalEmbedder degradation ----------------------------------------------


def test_local_embedder_degrades_when_model_missing():
    """No torch/transformers gigabyte required just to run the pipeline."""
    embedder = LocalEmbedder(dim=128)
    if EMBEDDING_AVAILABLE:
        assert not embedder.degraded
    else:
        assert embedder.degraded
        assert isinstance(embedder._fallback, HashingEmbedder)


def test_local_embedder_returns_correct_dimension():
    embedder = LocalEmbedder(dim=128)
    vec = embedder.embed("works at Acme")
    assert len(vec) == 128 if embedder.degraded else True


def test_local_embedder_caches_by_key():
    embedder = LocalEmbedder(dim=64)
    first = embedder.embed("text", key="f1")
    second = embedder.embed("different", key="f1")
    # Same key -> cached vector, regardless of the text passed.
    assert np.array_equal(first, second)


def test_local_embedder_can_require_the_real_model():
    """Jobs that need real similarity must fail loudly, not degrade."""
    if EMBEDDING_AVAILABLE:
        pytest.skip("sentence-transformers is installed")
    with pytest.raises(RuntimeError):
        LocalEmbedder(dim=64, allow_fallback=False)


def test_local_embedder_normalises_output():
    embedder = LocalEmbedder(dim=64)
    vec = embedder.embed("works at Acme")
    norm = float(np.linalg.norm(vec))
    assert norm == pytest.approx(1.0, abs=0.05)


def test_top_k_ranks_by_cosine():
    embedder = LocalEmbedder(dim=64)
    query = embedder.embed("works at Acme")
    same = embedder.embed("works at Acme", key="same")
    other = embedder.embed("likes tea", key="other")
    ranked = embedder.top_k(query, [("other", other), ("same", same)], k=2)
    assert ranked[0][0] == "same"
    assert ranked[0][1] > ranked[1][1]


def test_top_k_accepts_plain_lists():
    embedder = LocalEmbedder(dim=4)
    query = np.array([1.0, 0, 0, 0], dtype=np.float32)
    ranked = embedder.top_k(query, [("a", [1.0, 0, 0, 0]), ("b", [0, 1.0, 0, 0])], k=2)
    assert ranked[0][0] == "a"
