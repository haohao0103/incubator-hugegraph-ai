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

"""Tests for context rendering and token budgeting."""

from hugegraph_llm.semantic_layer.budget import fit_context
from hugegraph_llm.semantic_layer.context import (
    FULL,
    KEYS_ONLY,
    NAME_ONLY,
    ColumnContext,
    TableContext,
    estimate_tokens,
)


def _table(name="orders", columns=3, **kwargs):
    cols = [
        ColumnContext(
            name=f"col{i}",
            data_type="int",
            comment=f"column number {i}",
            key_type="primary" if i == 0 else "",
            references=["other.id"] if i == 1 else [],
        )
        for i in range(columns)
    ]
    return TableContext(
        table_name=name,
        table_description="All orders",
        columns=cols,
        primary_key=["col0"],
        **kwargs,
    )


# -- token estimation ------------------------------------------------------


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_counts_cjk_per_char():
    """len/4 would badly undercount Chinese metadata."""
    chinese = "订单表" * 10  # 30 chars
    assert estimate_tokens(chinese) == 30


def test_estimate_tokens_latin_is_quarter():
    assert estimate_tokens("a" * 100) == 25


def test_estimate_tokens_mixed():
    mixed = "订单" + "a" * 40
    assert estimate_tokens(mixed) == 2 + 10


# -- render levels ---------------------------------------------------------


def test_full_render_includes_comments_and_samples():
    ctx = _table()
    data = ctx.to_dict(FULL)
    assert data["num_columns"] == 3
    assert data["columns"][0]["comment"] == "column number 0"
    assert data["columns"][1]["references"] == ["other.id"]


def test_keys_only_keeps_keys_and_references_only():
    """Enough to write an ON clause, nothing more."""
    data = _table().to_dict(KEYS_ONLY)
    assert data["columns"][0] == {"name": "col0", "key_type": "primary"}
    assert data["columns"][1] == {"name": "col1", "references": ["other.id"]}
    assert "comment" not in data["columns"][1]


def test_keys_only_drops_plain_columns():
    data = _table().to_dict(KEYS_ONLY)
    # col2 has neither key nor reference, so nothing survives to render.
    assert len(data["columns"]) == 2


def test_name_only_has_no_columns():
    data = _table().to_dict(NAME_ONLY)
    assert data["columns"] == []
    assert data["num_columns"] == 0
    assert data["table_name"] == "orders"


def test_num_columns_reports_rendered_count():
    """A model must be able to tell the view is truncated."""
    data = _table(columns=5).to_dict(KEYS_ONLY)
    assert data["num_columns"] == len(data["columns"]) == 2


def test_stale_table_is_flagged():
    ctx = _table(confidence=0.3)
    assert ctx.is_stale
    assert "low confidence" in ctx.to_dict(FULL)["warning"]


def test_fresh_table_has_no_warning():
    assert "warning" not in _table(confidence=0.9).to_dict(FULL)


# -- budgeting -------------------------------------------------------------


def test_fit_respects_budget():
    result = fit_context([_table(f"t{i}") for i in range(5)], max_tokens=100_000)
    assert result.ok
    assert result.used_tokens <= result.budget
    assert len(result.contexts) == 5


def test_fit_degrades_rather_than_exceeding():
    """With room for one full table, the second must degrade, not vanish."""
    contexts = [_table(f"table_number_{i}", columns=8) for i in range(4)]
    one_full = estimate_tokens(contexts[0].render(FULL))
    result = fit_context(contexts, max_tokens=one_full + 20)
    assert result.ok
    assert result.used_tokens <= result.budget
    assert len(result.contexts) >= 2
    assert any(lvl != FULL for lvl in result.levels.values())


def test_fit_drops_when_even_name_only_does_not_fit():
    result = fit_context([_table(f"t{i}") for i in range(50)], max_tokens=60)
    assert result.dropped
    assert result.used_tokens <= result.budget


def test_fit_reserve_tokens_is_subtracted():
    """Reserve shrinks the usable budget before anything is placed."""
    contexts = [_table("t")]
    generous = fit_context(contexts, max_tokens=1000)
    stingy = fit_context(contexts, max_tokens=1000, reserve_tokens=999)
    assert generous.contexts and not stingy.contexts
    assert stingy.dropped == ["t"]


def test_zero_budget_drops_everything():
    result = fit_context([_table("a"), _table("b")], max_tokens=0)
    assert result.dropped == ["a", "b"]
    assert result.contexts == []


def test_seeds_rank_before_expanded():
    seed = _table("seed", is_seed=True, score=0.1)
    expanded = _table("expanded", is_seed=False, score=0.9)
    result = fit_context([expanded, seed], max_tokens=100_000)
    assert [c.table_name for c in result.contexts] == ["seed", "expanded"]


def test_higher_score_wins_among_seeds():
    strong = _table("strong", is_seed=True, score=0.9)
    weak = _table("weak", is_seed=True, score=0.1)
    result = fit_context([weak, strong], max_tokens=100_000)
    assert [c.table_name for c in result.contexts] == ["strong", "weak"]


def test_stale_tables_are_pushed_last_but_kept():
    """Stale must never crowd out trusted context, and must never vanish."""
    stale = _table("stale", is_seed=True, score=1.0, confidence=0.1)
    fresh = _table("fresh", is_seed=True, score=0.1, confidence=1.0)
    result = fit_context([stale, fresh], max_tokens=100_000)
    assert [c.table_name for c in result.contexts] == ["fresh", "stale"]


def test_render_uses_chosen_level():
    contexts = [_table("t", columns=6)]
    result = fit_context(contexts, max_tokens=1_000_000)
    assert result.render() == contexts[0].render(FULL)


def test_to_dicts_uses_chosen_level():
    contexts = [_table("t", columns=6)]
    result = fit_context(contexts, max_tokens=1_000_000)
    assert result.to_dicts()[0]["num_columns"] == 6
