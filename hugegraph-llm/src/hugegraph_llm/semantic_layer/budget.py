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

"""Token budgeting for retrieved context.

neocarta has no explicit budgeter: it returns whatever the search produced
and relies on the caller to cope. That is fine for a 30-table demo and
breaks on a real warehouse, where "retrieve the schema" can mean 300k tokens.

The contract here is: **never exceed the budget, degrade instead**. A table
that does not fit at full detail is retried at ``keys_only`` (enough to write
a JOIN) and then ``name_only`` (enough to know the table exists). Dropping a
table entirely is the last resort, and the caller is told what was dropped.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from hugegraph_llm.semantic_layer.context import (
    FULL,
    KEYS_ONLY,
    NAME_ONLY,
    TableContext,
    estimate_tokens,
)

__all__ = ["BudgetResult", "fit_context", "DEGRADATION_LEVELS"]

#: Order in which detail is given up.
DEGRADATION_LEVELS: Tuple[str, ...] = (FULL, KEYS_ONLY, NAME_ONLY)


@dataclass
class BudgetResult:
    """Outcome of fitting contexts into a token budget."""

    #: Rendered contexts, in the order they should appear in the prompt.
    contexts: List[TableContext] = field(default_factory=list)
    #: ``table_name -> level`` actually used.
    levels: Dict[str, str] = field(default_factory=dict)
    #: Tables that could not fit even at ``name_only``.
    dropped: List[str] = field(default_factory=list)
    used_tokens: int = 0
    budget: int = 0

    @property
    def ok(self) -> bool:
        return self.used_tokens <= self.budget

    def render(self) -> str:
        """Render every context at the level chosen for it."""
        return "\n\n".join(
            ctx.render(self.levels.get(ctx.table_name, FULL))
            for ctx in self.contexts
        )

    def to_dicts(self) -> List[dict]:
        return [
            ctx.to_dict(self.levels.get(ctx.table_name, FULL))
            for ctx in self.contexts
        ]


def fit_context(
    contexts: List[TableContext],
    max_tokens: int,
    *,
    reserve_tokens: int = 0,
) -> BudgetResult:
    """Fit ``contexts`` into ``max_tokens``, degrading detail to make room.

    Ordering is deliberate: a directly-recalled table is more likely to be
    the answer than one reached by graph expansion, and among equals the
    larger table is the more likely fact source. Stale (low-confidence)
    tables are pushed to the end -- they must never crowd out a trusted
    definition, but they are still shown.

    :param reserve_tokens: tokens held back for the rest of the prompt
        (instructions, the question itself). Subtracted before fitting.
    """
    budget = max(0, max_tokens - reserve_tokens)
    result = BudgetResult(budget=max_tokens)
    if budget == 0:
        result.dropped = [c.table_name for c in contexts]
        return result

    ordered = sorted(
        contexts,
        key=lambda c: (c.is_stale, not c.is_seed, -c.score, -c.row_count),
    )

    used = 0
    for ctx in ordered:
        chosen: Optional[str] = None
        cost = 0
        for level in DEGRADATION_LEVELS:
            candidate = estimate_tokens(ctx.render(level))
            if used + candidate <= budget:
                chosen, cost = level, candidate
                break
        if chosen is None:
            result.dropped.append(ctx.table_name)
            continue
        result.contexts.append(ctx)
        result.levels[ctx.table_name] = chosen
        used += cost

    result.used_tokens = used
    return result
