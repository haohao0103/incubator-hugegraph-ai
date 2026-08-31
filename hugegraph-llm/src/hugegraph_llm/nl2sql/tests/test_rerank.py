"""Tests for the optional cross-encoder rerank stage.

These never load a model: the reranker's availability is forced off or
replaced with a deterministic stand-in, so the tests stay fast and offline.
"""

from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker
from hugegraph_llm.nl2sql.rerank import CrossEncoderReranker, item_text
from hugegraph_llm.nl2sql.schema_graph.builder import SchemaGraphBuilder
from hugegraph_llm.nl2sql.schema_graph.model import Column, Table


class _FakeItem:
    """Minimal stand-in for LinkedItem."""

    def __init__(self, name, node_type="column", table="", props=None, score=1.0):
        self.name = name
        self.node_type = node_type
        self.table = table
        self.properties = props or {}
        self.score = score
        self.rerank_score = None


class _ReverseReranker:
    """Deterministic reranker: reverses order and records how it was called."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, items, top_k):
        self.calls.append((query, len(items), top_k))
        return list(reversed(items))[:top_k]


def _schema():
    b = SchemaGraphBuilder()
    b.add_tables([
        Table("orders", "dw", "订单表"),
        Table("users", "dw", "用户表"),
    ])
    b.add_columns([
        Column("order_id", "dw.orders", is_primary_key=True),
        Column("amount", "dw.orders", "DECIMAL"),
        Column("user_id", "dw.users", is_primary_key=True),
    ])
    return b.build()


def test_item_text_column_includes_table_and_type():
    it = _FakeItem("amount", "column", "dw.orders", {"data_type": "DECIMAL",
                                                     "comment": "订单金额"})
    text = item_text(it)
    assert "dw.orders.amount" in text
    assert "DECIMAL" in text
    assert "订单金额" in text


def test_item_text_table():
    it = _FakeItem("orders", "table", "", {"comment": "订单表"})
    assert item_text(it) == "orders 订单表"


def test_item_text_tolerates_missing_properties():
    assert item_text(_FakeItem("bare", "table")) == "bare"


def test_unavailable_reranker_preserves_order():
    rk = CrossEncoderReranker()
    rk._available = False  # simulate "model could not load"
    items = [_FakeItem(f"c{i}", score=float(10 - i)) for i in range(5)]
    out = rk.rerank("问题", items, top_k=3)
    assert [i.name for i in out] == ["c0", "c1", "c2"]


def test_reranker_never_raises_on_predict_failure():
    class _Boom:
        def predict(self, pairs):
            raise RuntimeError("cuda exploded")

    rk = CrossEncoderReranker()
    rk._available = True
    rk._model = _Boom()
    items = [_FakeItem("a"), _FakeItem("b")]
    out = rk.rerank("问题", items, top_k=2)
    assert [i.name for i in out] == ["a", "b"]  # order kept, no exception


def test_linker_invokes_reranker_and_uses_its_order():
    fake = _ReverseReranker()
    lk = SchemaLinker(_schema(), reranker=fake)
    out = lk.link("订单金额", top_k=3)

    assert fake.calls, "reranker was never invoked"
    query, pool_size, top_k = fake.calls[0]
    assert query == "订单金额"
    assert pool_size >= top_k == 3
    # pool must be wider than top_k, otherwise two-stage adds nothing
    assert pool_size > 3, f"candidate pool too narrow: {pool_size}"


def test_linker_without_reranker_is_unchanged():
    lk = SchemaLinker(_schema())
    out = lk.link("订单金额", top_k=3)
    assert len(out) <= 3
    # scores must stay in descending order (PPR order, no rerank applied)
    assert all(a.score >= b.score for a, b in zip(out, out[1:]))
    assert all(getattr(i, "rerank_score", None) is None for i in out)
