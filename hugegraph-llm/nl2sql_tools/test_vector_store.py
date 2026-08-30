"""Smoke tests for the nl2sql SchemaVectorStore abstraction.

Covers:
  1. NumpySchemaVectorStore basic semantics (top-k, cosine ordering, upsert).
  2. Parity: default linker vs linker with explicitly injected Numpy store.
  3. MilvusSchemaVectorStore degrade: dead server -> P2 disabled, lexical works.
  4. LegacyVectorStoreAdapter: wraps a VectorStoreBase-like store end-to-end.

Run:  PYTHONPATH=incubator-hugegraph-ai/hugegraph-llm/src python scripts/test_vector_store.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import build_warehouse_schema  # noqa: E402
from p2_embedder import make_embedder  # noqa: E402

from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from hugegraph_llm.nl2sql.vector_store import (  # noqa: E402
    LegacyVectorStoreAdapter,
    MilvusSchemaVectorStore,
    NumpySchemaVectorStore,
    as_schema_store,
)


def check(name, cond, extra=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {extra}")
    return cond


def main():
    ok = True
    schema = build_warehouse_schema()
    embedder = make_embedder()

    # ---- 1. numpy store basics ----
    s = NumpySchemaVectorStore()
    s.upsert(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    hits = s.search([1, 0, 0], top_k=3)
    ok &= check("numpy top1", hits[0] == ("a", 1.0), str(hits))
    ok &= check("numpy top2 is c", hits[1][0] == "c", str(hits))
    ok &= check("numpy dim", s.dimension == 3, f"dim={s.dimension}")
    ok &= check("numpy zero query -> []", s.search([0, 0, 0], 3) == [])
    s2 = NumpySchemaVectorStore()
    ok &= check("numpy empty search", s2.search([1, 0, 0], 3) == [])

    # ---- 2. parity: default vs explicit numpy store ----
    qs = ["毛利", "客户数", "支付金额是多少", "门店销售额和退款的差额"]
    lk_default = SchemaLinker(schema, embedder=embedder, top_k_vector=3)
    lk_numpy = SchemaLinker(
        schema, embedder=embedder, top_k_vector=3,
        vector_store=NumpySchemaVectorStore(),
    )
    for q in qs:
        r1 = [i.node_id for i in lk_default.link(q, top_k=10)]
        r2 = [i.node_id for i in lk_numpy.link(q, top_k=10)]
        ok &= check(f"parity '{q}'", r1 == r2, "" if r1 == r2 else f"{r1} vs {r2}")

    # ---- 3. milvus dead server -> clean degrade to lexical ----
    lk_milvus = SchemaLinker(
        schema, embedder=embedder, top_k_vector=3,
        vector_store=MilvusSchemaVectorStore(
            uri="http://127.0.0.1:1", collection_name="smoke_dead", dim=512,
        ),
    )
    try:
        items = lk_milvus.link("支付金额是多少", top_k=5)
        ok &= check("milvus-dead: P2 disabled", lk_milvus._vector_disabled is True)
        ok &= check("milvus-dead: lexical still works", len(items) > 0,
                    str([i.name for i in items][:3]))
    except Exception as exc:  # pragma: no cover
        ok &= check("milvus-dead: no crash (got exception)", False, repr(exc))

    # ---- 4. legacy adapter over a VectorStoreBase-like store ----
    class FakeStore:
        def __init__(self):
            self._rows = []

        def add(self, vectors, props):
            self._rows = list(zip(props, vectors))

        def search(self, q, top_k, dis_threshold=0.9):
            import numpy as np
            scored = []
            for pid, vec in self._rows:
                a, b = np.asarray(q), np.asarray(vec)
                sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                scored.append((sim, pid))
            scored.sort(reverse=True)
            return [pid for _, pid in scored[:top_k]]

    store = as_schema_store(FakeStore())
    ok &= check("as_schema_store wraps legacy", isinstance(store, LegacyVectorStoreAdapter))
    lk_legacy = SchemaLinker(
        schema, embedder=embedder, top_k_vector=3, vector_store=store,
    )
    items = lk_legacy.link("毛利", top_k=10)
    ok &= check(
        "legacy adapter surfaces gross_profit for 毛利",
        "column:dw.ads_daily_sales.gross_profit"
        in [i.node_id for i in items],
        str([i.node_id for i in items][:5]),
    )

    print("\n" + ("ALL PASS" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
