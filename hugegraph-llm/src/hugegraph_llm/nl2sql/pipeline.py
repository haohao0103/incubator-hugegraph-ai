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

"""
End-to-end orchestration, with every stage independently callable.

Design goal: an upstream platform may consume the whole pipeline, or pick
only the piece it needs. Each of these is a standalone entry point:

- ``link(question)``            -> top-k relevant tables/columns (L1 only)
- ``join_path(...)``            -> join path between tables (L2 only)
- ``communities()``             -> subject-domain partitioning (L3 only)
- ``schema_context(question)``  -> narrowed, flat schema string for a prompt
- ``run(question)``             -> full pipeline (requires an injected LLM)

``schema_context`` is the piece most worth understanding. Borrowing from
SuperSonic: what gets handed to the model is a **flat, narrowed view** —
business names, types and comments, but no overwhelming catalog dump. Unlike
SuperSonic, proven joins can optionally be included, because not every
upstream platform has its own semantic layer to generate them.

A single ``engine`` is shared by every layer. It defaults to in-process
networkx, so nothing external is required; injecting a ``VermeerEngine`` moves
the PPR, shortest-path, Steiner and community work onto a Vermeer cluster
while the API above stays byte-identical.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hugegraph_llm.utils.log import log

from .engine.base import EngineCapabilities, GraphEngine
from .engine.local import LocalEngine
from .evaluation.evaluator import SQLScore, SQLEvaluator
from .join_path.path_finder import JoinPath, JoinPathFinder
from .linking.schema_linker import LinkedItem, SchemaLinker
from .rerank import CrossEncoderReranker
from .schema_graph.model import SchemaGraph


@dataclass
class PipelineResult:
    """Outcome of a full pipeline run."""

    question: str
    linked_items: List[LinkedItem] = field(default_factory=list)
    schema_context: str = ""
    sql: str = ""
    join_path: Optional[JoinPath] = None

    @property
    def tables(self) -> List[str]:
        return sorted({i.name for i in self.linked_items
                       if i.node_type == "table"})


class NL2SQLPipeline:
    """Graph-enhanced Text2SQL, usable whole or in parts."""

    def __init__(
        self,
        schema: SchemaGraph,
        llm: Optional[Callable[[str], str]] = None,
        top_k: int = 10,
        engine: Optional[GraphEngine] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        top_k_vector: int = 5,
        vector_weight: float = 0.9,
        keyword_extractor: Optional[Callable[[str], List[str]]] = None,
        reranker: Optional["CrossEncoderReranker"] = None,
    ):
        """
        :param schema: Schema Graph from :class:`SchemaGraphBuilder`.
        :param llm: Optional callable taking a prompt and returning SQL.
                    Required only for :meth:`run` (full pipeline).
        :param top_k: Default number of schema elements to retrieve.
        :param engine: Graph compute engine shared by all layers. Defaults to
                       an in-process :class:`LocalEngine`.
        :param embedder: optional ``Callable[[str], list[float]]`` enabling
                         semantic (P2) schema linking. Forwarded to
                         :class:`SchemaLinker`; ``None`` = lexical only.
        :param top_k_vector: semantic neighbours contributed per question.
        :param vector_weight: weight ceiling for the top semantic seed.
        :param keyword_extractor: optional ``Callable[[str], list[str]]`` that
                                  pulls retrieval keywords out of a question
                                  before linking (LLM-gated); seeds merge by
                                  max weight with the raw question.
        """
        self._schema = schema
        self._llm = llm
        self._top_k = top_k
        self._engine = engine if engine is not None else LocalEngine(schema)
        self._keyword_extractor = keyword_extractor
        self._understanding = None  # lazy QueryUnderstanding
        self._linker = SchemaLinker(
            schema,
            engine=self._engine,
            embedder=embedder,
            top_k_vector=top_k_vector,
            vector_weight=vector_weight,
            reranker=reranker,
        )
        self._join_finder = JoinPathFinder(schema, engine=self._engine)
        self._permission_rules = None  # optional tenant column allow-lists

    def set_permission_rules(self, rules) -> None:
        """Enable tenant-level column permissions (see nl2sql.permissions)."""
        self._permission_rules = rules

    # ---- engine introspection ----

    @property
    def engine(self) -> GraphEngine:
        """The engine backing every layer of this pipeline."""
        return self._engine

    def prebuild(self) -> None:
        """Pre-build linker indexes (BM25 + vector) at load time, so the first
        user question does not pay the cold-start cost."""
        self._linker.prebuild()

    @property
    def capabilities(self) -> EngineCapabilities:
        """What the current engine does and does not guarantee."""
        return self._engine.capabilities

    # ---- L1: schema linking (standalone) ----

    def _link_texts(self, question: str) -> List[str]:
        """Question plus optional LLM-extracted keywords (fail-open)."""
        texts = [question]
        if self._keyword_extractor is not None:
            try:
                kws = self._keyword_extractor(question)
                if kws:
                    texts.extend(str(k) for k in kws if str(k).strip())
            except Exception as exc:  # noqa: BLE001 -- LLM down? keep raw
                log.warning("keyword extraction failed; raw question only: %s", exc)
        return texts

    def link(self, question: str, top_k: Optional[int] = None) -> List[LinkedItem]:
        intent = self.understand(question).intent_type
        return self._linker.link_multi(
            self._link_texts(question), top_k or self._top_k, intent=intent
        )

    def understand(self, question: str):
        """Rule-based query understanding (intent + jargon expansion), offline."""
        from .query_understanding import QueryUnderstanding

        if self._understanding is None:
            self._understanding = QueryUnderstanding()
        return self._understanding.understand(question)

    # ---- L2: join path (standalone) ----

    def join_path(self, source: str, target: str) -> Optional[JoinPath]:
        return self._join_finder.shortest_path(source, target)

    def connect_tables(self, tables: List[str]) -> Optional[JoinPath]:
        return self._join_finder.connect(tables)

    # ---- L3: subject domains (standalone) ----

    def communities(
        self, resolution: float = 1.0, algorithm: str = "louvain"
    ) -> Dict[str, List[str]]:
        """Partition the warehouse into subject domains.

        Community detection over the join graph answers a question the linker
        cannot: *which tables belong together at all*. A candidate set that
        straddles two domains is usually a linking mistake, not a legitimate
        cross-domain query.

        :returns: domain id -> sorted table names.
        """
        return self._join_finder.domains(
            resolution=resolution, algorithm=algorithm
        )

    def domain_of(self, table: str) -> Optional[str]:
        """Subject domain containing ``table`` (``db.name``), if any."""
        for cid, names in self.communities().items():
            if table in names:
                return cid
        return None

    # ---- metric authority + SQL quality gate (optional capabilities) ----

    def authoritative_metric(self, term_names: List[str]) -> Optional[str]:
        """Pick the authoritative metric among candidate term names, using the
        per-Term ``authoritative / priority / source`` properties (governance
        stamp dominates). ``None`` when no candidate term exists."""
        from .metric_authority import resolve_metric

        cands = []
        for n in term_names:
            node = self._schema.nodes.get(f"term:{n}")
            if node is not None:
                cands.append((n, node.properties))
        if not cands:
            return None
        winner = resolve_metric(cands)
        return winner[0] if winner else None

    def vote_sql(
        self,
        candidates: List[str],
        linked_ids: Optional[List[str]] = None,
        metric_agg: Optional[str] = None,
    ) -> List[tuple]:
        """Deterministically rank candidate SQLs (validity / 口径 / overlap).
        No LLM call. Returns ``[(sql, score, report), ...]`` best first."""
        from .sql_ops import SqlVoter

        return SqlVoter(self._schema).vote(candidates, linked_ids, metric_agg)

    # ---- narrowed prompt context (standalone) ----

    def schema_context(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_joins: bool = False,
        include_global: bool = False,
        tenant: Optional[str] = None,
    ) -> str:
        """Build a flat, narrowed schema description for a prompt.

        Only linked elements are described, so the model is not asked to
        choose among hundreds of columns — which is exactly where real-world
        Text2SQL accuracy collapses. ``include_global`` appends same-subject-
        domain sibling tables (Global context, mirroring production
        metadata-GraphRAG systems) so the model can discover related tables
        without an explicit join path.

        ``tenant`` is accepted for forward compatibility but **does not filter**
        yet — current stage only *flags* sensitive columns (``[SENSITIVE]``);
        permission enforcement waits for the governance allow-list.
        """
        items = self.link(question, top_k)
        if not items:
            return ""

        lines: List[str] = []
        tables = [i for i in items if i.node_type == "table"]
        columns = [i for i in items if i.node_type == "column"]

        if tables:
            lines.append("Tables:")
            for t in tables:
                comment = t.properties.get("comment", "")
                suffix = f" -- {comment}" if comment else ""
                lines.append(f"  {t.name}{suffix}")

        if columns:
            lines.append("Columns:")
            for c in columns:
                col_type = c.properties.get("data_type", "")
                comment = c.properties.get("comment", "")
                parts = [p for p in (col_type, comment) if p]
                suffix = f" -- {' '.join(str(p) for p in parts)}" if parts else ""
                owner = f"{c.table}." if c.table else ""
                from .permissions import is_sensitive

                tag = " [SENSITIVE]" if is_sensitive(c.name, comment) else ""
                lines.append(f"  {owner}{c.name}{suffix}{tag}")

        if include_joins and len(tables) >= 2:
            # Pass full table names (db.table) so they match the join-graph
            # node ids ("table:dw.orders"), not the bare short names.
            table_full = [t.node_id.split(":", 1)[1] for t in tables]
            path = self.connect_tables(table_full)
            if path and path.steps:
                lines.append("Joins:")
                for step in path.steps:
                    if step.proven:
                        lines.append(f"  {step.to_sql()}")

        if include_global and tables:
            # Same-subject-domain siblings of the linked tables (Global view).
            domains = self.communities()
            linked_full = {t.node_id.split(":", 1)[1] for t in tables}
            sibs: List[str] = []
            for t in tables:
                full = t.node_id.split(":", 1)[1]
                for dom_id, names in domains.items():
                    if full in names:
                        for sib in names:
                            if sib not in linked_full and sib not in sibs:
                                sibs.append(sib)
            if sibs:
                lines.append("Related tables (same subject domain):")
                lines.append("  " + ", ".join(sibs))

        return "\n".join(lines)

    # ---- full pipeline ----

    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_joins: bool = True,
    ) -> PipelineResult:
        """Run the full pipeline; requires an LLM to have been injected."""
        if self._llm is None:
            raise ValueError("full pipeline requires an injected llm callable")
        items = self.link(question, top_k)
        context = self.schema_context(question, top_k, include_joins)
        prompt = _build_prompt(question, context)
        sql = self._llm(prompt)
        log.info("pipeline produced SQL for question: %s", question)
        return PipelineResult(
            question=question,
            linked_items=items,
            schema_context=context,
            sql=sql,
        )

    # ---- evaluation (standalone) ----

    def evaluate(
        self,
        pairs: List[tuple],
        executor: Optional[Callable[[str], List]] = None,
    ) -> Dict[str, object]:
        """Score predicted/gold SQL pairs at clause level."""
        ev = SQLEvaluator(executor=executor)
        agg = ev.evaluate_batch(pairs)
        return {
            "count": agg.count,
            "exact_match_rate": agg.exact_match_rate,
            "execution_accuracy": agg.execution_accuracy,
            "mean_f1": agg.mean_f1,
            "per_clause": agg.per_clause,
        }

    # ---- lifecycle ----

    def close(self) -> None:
        """Release engine resources (a no-op for the local engine)."""
        self._engine.close()

    def __enter__(self) -> "NL2SQLPipeline":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _build_prompt(question: str, schema_context: str) -> str:
    """Compose the SQL-generation prompt.

    Mirrors the constraint style that makes semantic-layer Text2SQL work:
    the model may only reference what appears in the provided schema.
    """
    return (
        "You are a data analyst writing SQL for a data warehouse.\n"
        "Only use tables and columns listed in the Schema below; "
        "do not invent names.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )
