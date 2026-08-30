"""End-to-end NL2SQL orchestration over the metadata KG (P0 + P1 wiring).

This is the glue that turns the standalone NL2SQL building blocks into one
deterministic, auditable pipeline and emits the *same* ``QueryStage`` /
``UnifiedQueryResponse`` contract the unified query API uses (so it drops into
``unified_query`` as ``mode="nl2sql"``):

  问题 --(P1-2 黑话)--> schema linking (P0-1/2)
       --(LLM 生成候选, 可注入)--> 候选 SQL 列表
       --(P0-3 校验)--> 逐个验证
       --(P1-4 投票)--> 选出最优 SQL
       --(P1-3 权威 / P1-1 血缘 / P0-4 golden)--> 审计与回灌

Every step except the LLM generation is **deterministic and LLM-free**, so the
pipeline is cheap, testable and safe to run on every request. The LLM step is an
injectable ``generate_fn`` -- unit tests pass candidates directly (no network),
while the live path (``unified_query`` mode ``nl2sql``) uses glm-5.3 via the
project's ``LLMs().get_text2gql_llm()``.

Typical use::

    pipe = KgNL2SQLPipeline(question="各城市订单总额", client=client)
    resp = pipe.run()                       # real LLM generation
    # or fully deterministic:
    resp = pipe.run(candidates=["SELECT SUM(order.amount) FROM order GROUP BY city"])
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from hugegraph_llm.api.models.unified_requests import (
    QueryStageBuilder,
    UnifiedQueryResponse,
)
from hugegraph_llm.operators.graph_op.kg_rule_engine import (
    KgRuleEngine,
    GraphData,
)
from hugegraph_llm.operators.graph_op.kg_schema_linker import KgSchemaLinker
from hugegraph_llm.operators.graph_op.kg_sql_validator import KgSqlValidator
from hugegraph_llm.operators.graph_op.kg_sql_voter import KgSqlVoter
from hugegraph_llm.operators.graph_op.kg_jargon_map import KgJargonMap
from hugegraph_llm.operators.graph_op.kg_metric_authority import KgMetricAuthority
from hugegraph_llm.operators.graph_op.kg_lineage_api import KgLineageApi
from hugegraph_llm.operators.graph_op.kg_golden_sql import KgGoldenSqlStore
from hugegraph_llm.utils.log import log

# (question, prompt_context) -> List[str] of candidate SQLs
GenerateFn = Callable[[str, str], List[str]]


def _extract_sql_candidates(text: str) -> List[str]:
    """Pull ```sql fenced blocks; fall back to the whole response as one SQL."""
    blocks = re.findall(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    cleaned = text.strip()
    return [cleaned] if cleaned else []


def _truncate(s: str, n: int = 800) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "...(truncated)"


class KgNL2SQLPipeline:
    """Compose the NL2SQL building blocks into one staged, auditable pipeline."""

    def __init__(
        self,
        question: str,
        graph_data: Optional[GraphData] = None,
        client: Optional[Any] = None,
        graph_name: Optional[str] = None,
        generate_fn: Optional[GenerateFn] = None,
        golden_store: Optional[KgGoldenSqlStore] = None,
        jargon: Optional[KgJargonMap] = None,
        store_best: bool = False,
        domain: Optional[str] = None,
    ) -> None:
        if not question or not str(question).strip():
            raise ValueError("question must not be empty")
        if graph_data is None and client is None:
            raise ValueError("KgNL2SQLPipeline requires graph_data or client")
        self._question = str(question)
        self._graph_name = graph_name
        self._client = client
        self._data = graph_data
        self._generate_fn = generate_fn
        self._use_default_llm = generate_fn is None
        self._golden_store = golden_store
        self._store_best = store_best
        self._domain = domain

        self._jargon = jargon or KgJargonMap()
        self._linker = KgSchemaLinker(
            client=client, synonyms=self._jargon.to_synonym_map()
        )

        golden_records: List[Any] = []
        if golden_store is not None:
            try:
                golden_records = golden_store.get_similar(self._question, top_k=3)
            except Exception:  # pragma: no cover - store IO guard
                golden_records = []

        # one graph snapshot + one validator per request, shared with the
        # voter (index build is the dominant cost on large metadata graphs)
        self._graph_snapshot = self._graph_data()
        self._validator = KgSqlValidator(self._graph_snapshot, graph_name)
        self._voter = KgSqlVoter(
            question=self._question,
            graph_data=self._graph_snapshot,
            validator=self._validator,
            golden_records=golden_records,
        )
        self._authority = KgMetricAuthority(
            graph_data=graph_data, client=client, graph_name=graph_name
        )
        self._lineage = KgLineageApi(
            data=graph_data, client=client, graph_name=graph_name
        )

    # -- graph access --------------------------------------------------------

    def _graph_data(self) -> GraphData:
        if self._data is not None:
            return self._data
        if self._client is not None:
            self._data = KgRuleEngine(self._client, self._graph_name).load_graph()
            return self._data
        return {"vertices": {}, "edges": {}}  # pragma: no cover - unreachable

    # -- LLM generation (default, lazy) -------------------------------------

    def _default_generate(self, prompt_context: str) -> List[str]:
        """Real glm-5.3 generation via the project's text2gql LLM role."""
        from hugegraph_llm.models.llms.init_llm import LLMs

        llm = LLMs().get_text2gql_llm()
        prompt = (
            "You are a data-warehouse NL2SQL engine. Using ONLY the tables, "
            "fields and metric formulas in the SCHEMA below, write SQL to answer "
            "the QUESTION. Return up to 3 candidate SQL statements, each wrapped "
            "in a ```sql fenced block. Respect each metric's canonical formula "
            "(e.g. SUM vs COUNT) and only reference schema entities that exist.\n\n"
            f"QUESTION: {self._question}\n\nSCHEMA:\n{prompt_context}\n"
        )
        try:
            resp = llm.generate(prompt=prompt)
        except Exception as exc:  # pragma: no cover - network/LLM guard
            log.warning("nl2sql LLM generation failed: %s", exc)
            return []
        return _extract_sql_candidates(resp)

    # -- run ----------------------------------------------------------------

    def run(self, candidates: Optional[List[str]] = None) -> UnifiedQueryResponse:
        question = self._question
        stages: List[Any] = []

        # 1) schema linking (P0-1/2) + jargon (P1-2 already in linker synonyms)
        ctx = self._linker.link(question, data=self._graph_data())
        prompt_context = ctx.to_prompt_context()
        stages.append(
            QueryStageBuilder.make(
                "linking",
                input={"question": question},
                output={
                    "tables": [t.get("name") for t in ctx.tables],
                    "fields": [f.get("name") for f in ctx.fields],
                    "metrics": [m.get("name") for m in ctx.metrics],
                    "evidence": ctx.evidence[:3],
                },
            )
        )

        # 2) candidate generation
        if candidates is not None:
            source = "provided"
        elif self._generate_fn is not None:
            try:
                candidates = self._generate_fn(question, prompt_context) or []
            except Exception as exc:
                log.warning("generate_fn failed: %s", exc)
                candidates = []
            source = "custom"
        elif self._use_default_llm:
            candidates = self._default_generate(prompt_context)
            source = "llm"
        else:  # pragma: no cover - unreachable (use_default_llm True when no fn)
            candidates = []
            source = "none"
        candidates = [str(c).strip() for c in (candidates or []) if str(c).strip()]
        stages.append(
            QueryStageBuilder.make(
                "sql_generation",
                input={"prompt_context": _truncate(prompt_context)},
                output={"candidates": candidates, "source": source},
            )
        )

        # 3) validation (P0-3) + voting (P1-4)
        votes = self._voter.vote(candidates) if candidates else []
        stages.append(
            QueryStageBuilder.make(
                "sql_validation",
                output={
                    "validated": [
                        {
                            "sql": v.sql,
                            "valid": v.valid,
                            "issue_count": v.issue_count,
                            "caliber": v.breakdown.get("caliber", 0),
                        }
                        for v in votes
                    ]
                },
            )
        )
        best_sql = votes[0].sql if votes else ""
        stages.append(
            QueryStageBuilder.make(
                "sql_voting",
                output={
                    "ranked": [
                        {"sql": v.sql, "score": v.score, "valid": v.valid}
                        for v in votes
                    ],
                    "chosen": best_sql,
                },
            )
        )

        # 4) metric authority (P1-3) -- only surface when a name conflicts
        authority_notes = self._authority_notes(ctx)
        if authority_notes:
            stages.append(
                QueryStageBuilder.make("authority", output={"resolved": authority_notes})
            )

        # 5) lineage audit (P1-1)
        stages.append(
            QueryStageBuilder.make(
                "lineage", output={"explain": self._lineage_for(best_sql, ctx, votes)}
            )
        )

        # 6) golden feedback (P0-4) -- optionally store the winning SQL
        stored = self._maybe_store(best_sql, votes)
        if stored is not None:
            stages.append(QueryStageBuilder.make("golden_feedback", output=stored))

        return UnifiedQueryResponse(
            answer=best_sql,
            route="nl2sql",
            citations=[],
            subgraph={
                "tables": [t.get("name") for t in ctx.tables],
                "metrics": [m.get("name") for m in ctx.metrics],
            },
            raw={"votes": [v.to_dict() for v in votes], "prompt_context": prompt_context},
            stages=stages,
        )

    # -- helpers -------------------------------------------------------------

    def _authority_notes(self, ctx: Any) -> Dict[str, Any]:
        notes: Dict[str, Any] = {}
        seen: set = set()
        for m in ctx.metrics:
            name = m.get("name")
            if not name or name in seen:
                continue
            seen.add(name)
            ranked = self._authority.ranked(name)
            if len(ranked) > 1:
                notes[name] = {
                    "authoritative": ranked[0].get("name"),
                    "alternatives": [r.get("name") for r in ranked[1:]],
                }
        return notes

    def _lineage_for(self, best_sql: str, ctx: Any, votes: List[Any]) -> str:
        target: Optional[str] = None
        if votes:
            refs = getattr(votes[0].report, "tables_referenced", None) or []
            if refs:
                target = refs[0]
        if not target and ctx.tables:
            target = ctx.tables[0].get("name")
        if not target and ctx.metrics:
            target = ctx.metrics[0].get("name")
        return self._lineage.explain(target or "")

    def _maybe_store(self, best_sql: str, votes: List[Any]) -> Optional[Dict[str, Any]]:
        if not (self._store_best and self._golden_store and best_sql):
            return None
        valid = bool(votes) and votes[0].valid
        if not valid:
            return None
        try:
            vid = self._golden_store.add(self._question, best_sql, domain=self._domain)
            return {"stored": vid is not None, "vertex_id": vid}
        except Exception as exc:  # pragma: no cover - store IO guard
            log.warning("golden store failed: %s", exc)
            return {"stored": False, "error": str(exc)}
