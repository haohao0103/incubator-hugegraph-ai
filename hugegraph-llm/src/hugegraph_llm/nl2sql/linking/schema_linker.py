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
Schema Linking: pick the right tables and columns for a question.

This is the single highest-leverage step in warehouse Text2SQL. Benchmarks
make the point bluntly: models score ~91% on the clean academic Spider 1.0
but only ~21% on Spider 2.0, which uses real enterprise schemas averaging
hundreds of columns. The model rarely fails at *writing* SQL -- it fails at
*choosing* among an overwhelming number of plausible tables.

Approach: Personalized PageRank (PPR) over the Schema Graph.
- Seed nodes come from business-term bindings, literal name/comment matches,
  and (optionally) semantic vector recall. PPR then propagates relevance along
  lineage, foreign keys, and co-occurrence, so a column that is strongly
  connected to the seeds rises to the top even though its own name never
  appears in the question.

Two seed sources are layered, both feeding the *same* PPR:
- **Lexical (P0, always on)**: term names/aliases/definitions + table/column
  names/comments, with Chinese-aware tokenisation so a question like
  "支付总额" can reach the metric whose Chinese definition is "支付总额".
- **Semantic (P2, optional)**: if an ``embedder`` is supplied, every schema
  node is embedded once and the question's nearest neighbours (cosine top-k)
  are added as seeds. This is what connects "总额" to "金额" when they share
  no surface string. The embedder is fully pluggable -- wire in an OpenAI /
  Ollama / LiteLLM embedding via ``hugegraph_llm.models.embeddings`` or any
  ``Callable[[str], list[float]]``.

The PPR itself is delegated to a :class:`~hugegraph_llm.nl2sql.engine.base.
GraphEngine`. In-process networkx is the default; a Vermeer cluster can be
injected instead when the catalog outgrows Python, without any change to the
seeding logic or to the caller's code.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from hugegraph_llm.utils.log import log

from ..engine.base import GraphEngine
from ..engine.local import LocalEngine
from ..schema_graph.model import NodeType, SchemaGraph
from ..synonym_dict import JargonMap
from ..vector_store import NumpySchemaVectorStore, SchemaVectorStore

# Latin / digit runs and CJK codepoints -- the two token classes we index.
_LATIN_RE = re.compile(r"[a-z0-9_]+")
_CJK_RE = re.compile(r"[一-鿿]")


def _cjk_segment(text: str) -> List[str]:
    """Chinese word segmentation, with a char-level fallback.

    ``jieba`` gives proper multi-character words ("支付金额" -> 支付/金额) which
    makes fuzzy token overlap far more precise than naive character splits. If
    jieba is unavailable we degrade to single-character unigrams.
    """
    try:
        import jieba

        return [w for w in jieba.cut(text) if _CJK_RE.search(w)]
    except Exception:  # pragma: no cover - jieba present in our runtime
        return list(text)


def _tokens(text: str) -> set:
    """Mixed Latin + CJK token set (snake_case / camelCase aware, lowercased).

    ``orderAmount x_y`` -> ``{order, amount, x, y}``; ``order_id`` -> ``{order,
    id}``. Splitting identifiers into their constituent words matters for fuzzy
    matching: a question token ``order`` must meet the column ``order_id``.
    """
    text = str(text)
    # camelCase boundary first (before lowercasing destroys it): lower->Upper.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.lower()
    toks = set()
    for run in re.findall(r"[a-z0-9_]+", text):
        toks.update(part for part in run.split("_") if part)
    cjk = "".join(_CJK_RE.findall(text))
    if cjk:
        toks.update(_cjk_segment(cjk))
    return toks


class _BM25Index:
    """Minimal BM25 (k1=1.5, b=0.75) over tokenised node surfaces.

    Built once per linker from every table/column surface, so the fuzzy
    fallback is a proper lexical scorer (IDF-aware) instead of a raw token
    overlap count.
    """

    def __init__(self, docs: List[tuple]):
        # docs: [(node_id, List[token])]
        self._doc_ids = [d[0] for d in docs]
        self._doc_lens = [len(d[1]) for d in docs]
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)) if docs else 0.0
        df: Dict[str, int] = {}
        for _, toks in docs:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5))
                     for t, f in df.items()}
        self._postings: Dict[str, List[tuple]] = defaultdict(list)
        for i, (_, toks) in enumerate(docs):
            for t, tf in Counter(toks).items():
                self._postings[t].append((i, tf))

    def score(self, q_tokens, k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for t in q_tokens:
            if t not in self._postings:
                continue
            idf = self._idf.get(t, 0.0)
            for i, tf in self._postings[t]:
                dl = self._doc_lens[i]
                if self._avgdl:
                    denom = tf + k1 * (1 - b + b * dl / self._avgdl)
                else:
                    denom = tf + k1
                out[self._doc_ids[i]] = out.get(self._doc_ids[i], 0.0) + \
                    idf * tf * (k1 + 1) / denom
        return out


@dataclass
class LinkedItem:
    """A schema element retrieved for a question."""

    node_id: str
    name: str
    node_type: str
    score: float
    table: str = ""
    properties: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.node_type}:{self.name}({self.score:.4f})"


class SchemaLinker:
    """Retrieves the top-k most relevant schema elements for a question."""

    def __init__(
        self,
        schema: SchemaGraph,
        alpha: float = 0.85,
        engine: Optional[GraphEngine] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
        top_k_vector: int = 5,
        vector_weight: float = 0.9,
        vector_store: Optional["SchemaVectorStore"] = None,
        use_bm25: bool = True,
        use_importance: bool = True,
        importance_weight: float = 0.15,
        use_jargon: bool = True,
        fusion: Optional[str] = None,
    ):
        """
        :param schema: Schema Graph built by :class:`SchemaGraphBuilder`.
        :param alpha: PPR damping factor. Higher means relevance stays closer
                      to the seeds; lower spreads it further through the graph.
        :param engine: Graph compute engine. Defaults to an in-process
                       :class:`LocalEngine`; pass a ``VermeerEngine`` to push
                       the PPR onto a cluster.
        :param embedder: optional ``Callable[[str], list[float]]`` enabling
                         semantic (P2) seed recall. When set, each schema node
                         is embedded lazily on first use and the question's
                         cosine top-k neighbours become seeds. Leave ``None`` for
                         lexical-only linking.
        :param top_k_vector: how many nearest nodes the embedder contributes.
        :param vector_weight: weight assigned to the top semantic seed (scaled
                             by cosine similarity, then capped at this).
        :param vector_store: where node embeddings live. Defaults to an
                             in-process :class:`NumpySchemaVectorStore`; pass a
                             ``MilvusSchemaVectorStore`` / ``OceanBaseSchema
                             VectorStore`` / any RAG ``VectorStoreBase`` (auto-
                             wrapped) to serve P2 seeds from an external vector
                             DB. Failures degrade to lexical linking.
        :param use_bm25: replace the token-overlap fuzzy fallback with a real
                         BM25 scorer (jieba tokens + IDF). Default on.
        :param use_importance: rerank table/column results by entity importance
                               (lineage downstream count, row count, fact flag),
                               scaled by ``importance_weight``. Mirrors the
                               manual-boost entity weighting of production
                               metadata-GraphRAG systems.
        :param importance_weight: how much the normalised importance boosts a
                                  result's PPR score (score *= 1 + w*imp).
        """
        self._schema = schema
        self._alpha = alpha
        self._engine = engine if engine is not None else LocalEngine(schema)
        self._embedder = embedder
        self._top_k_vector = top_k_vector
        self._vector_weight = vector_weight
        self._vector_store: Optional[SchemaVectorStore] = vector_store
        self._vector_index: Optional[bool] = None  # None=unbuilt, True=built
        self._vector_disabled = False
        self._use_bm25 = use_bm25
        self._use_importance = use_importance
        self._importance_weight = importance_weight
        self._use_jargon = use_jargon
        self._fusion = fusion  # None | "rrf" | "score"
        self._jargon = JargonMap() if use_jargon else None
        self._bm25 = None  # lazy BM25 index over schema node surfaces
        self._importance: Optional[Dict[str, float]] = None  # node_id -> [0,1]

    @property
    def engine(self) -> GraphEngine:
        """The engine actually running the PPR."""
        return self._engine

    def prebuild(self) -> None:
        """Eagerly build BM25 + vector indexes to avoid a first-query stall.

        Both indexes are lazily built on first ``link`` otherwise; for a
        production service the pipeline calls this right after construction
        (load time cost, controlled) instead of paying it on the first user
        question. Failures degrade internally (vector build disables P2).
        """
        if self._use_bm25:
            self._ensure_bm25()
        if self._embedder is not None:
            self._ensure_vector_index()

    # ---- public API (individually callable) ----

    def link(
        self,
        question: str,
        top_k: int = 10,
        include_tables: bool = True,
        include_columns: bool = True,
        intent: Optional[str] = None,
    ) -> List[LinkedItem]:
        """Return the top-k schema elements relevant to ``question``."""
        return self.link_multi([question], top_k, include_tables,
                               include_columns, intent=intent)

    def link_multi(
        self,
        texts: List[str],
        top_k: int = 10,
        include_tables: bool = True,
        include_columns: bool = True,
        intent: Optional[str] = None,
    ) -> List[LinkedItem]:
        """Seed from several texts (question + LLM keywords + jargon canonicals)
        and run one PPR. Seeds merge by max weight.

        With ``fusion`` set, each recall path (lexical / vector / BM25) is
        ranked independently and the rankings are fused at the result level
        (RRF or weighted score-sum) instead of the seed level.

        ``intent`` ("table"/"field"/"metric") applies type-weighted re-ranking
        — "在哪个表" surfaces tables, "口径是多少" surfaces the metric column.
        """
        if self._use_jargon:
            extra: List[str] = []
            for t in texts:
                extra.extend(self._jargon.expand(str(t)))
            texts = list(texts) + [e for e in extra if e not in texts]

        if self._fusion in ("rrf", "score"):
            return self._fused_link(texts, top_k, include_tables,
                                    include_columns, intent=intent)

        seeds: Dict[str, float] = {}
        for t in texts:
            if not t or not str(t).strip():
                continue
            for nid, w in self._seed_nodes(str(t)).items():
                seeds[nid] = max(seeds.get(nid, 0.0), w)
        if self._embedder is not None:
            for t in texts:
                if not t or not str(t).strip():
                    continue
                for nid, w in self._vector_seeds(str(t)).items():
                    seeds[nid] = max(seeds.get(nid, 0.0), w)
        if not seeds:
            log.warning("no schema seed matched question: %s", texts[0] if texts else "")
            return []
        scores = self._ppr(seeds)
        items = self._to_items(scores, include_tables, include_columns)
        items = self._rerank(items, intent=intent)
        return items[:top_k]

    def _fused_link(
        self,
        texts: List[str],
        top_k: int,
        include_tables: bool,
        include_columns: bool,
        intent: Optional[str] = None,
    ) -> List[LinkedItem]:
        """Result-level fusion: rank each recall path, fuse the rankings."""
        from ..fusion import rrf_fuse, score_fuse

        def _path(seeds: Dict[str, float]) -> List[LinkedItem]:
            if not seeds:
                return []
            scores = self._ppr(seeds)
            items = self._to_items(scores, include_tables, include_columns)
            return self._rerank(items, intent=intent)

        def _lexical() -> Dict[str, float]:
            seeds: Dict[str, float] = {}
            for t in texts:
                if not t or not str(t).strip():
                    continue
                for nid, w in self._seed_nodes(str(t)).items():
                    seeds[nid] = max(seeds.get(nid, 0.0), w)
            return seeds

        def _vector() -> Dict[str, float]:
            if self._embedder is None:
                return {}
            seeds: Dict[str, float] = {}
            for t in texts:
                if not t or not str(t).strip():
                    continue
                for nid, w in self._vector_seeds(str(t)).items():
                    seeds[nid] = max(seeds.get(nid, 0.0), w)
            return seeds

        def _bm25() -> Dict[str, float]:
            if not self._use_bm25:
                return {}
            seeds: Dict[str, float] = {}
            for t in texts:
                q = _tokens(str(t))
                if q:
                    for nid, w in self._bm25_seeds(q).items():
                        seeds[nid] = max(seeds.get(nid, 0.0), w)
            return seeds

        path_items: List[List[LinkedItem]] = []
        sources: List[str] = []
        for name, seeds in (("lexical", _lexical()), ("vector", _vector()),
                            ("bm25", _bm25())):
            items = _path(seeds)
            if items:
                path_items.append(items[: max(20, top_k * 3)])
                sources.append(name)
        if not path_items:
            return []
        weights = {s: 1.0 for s in sources}
        if self._fusion == "rrf":
            fused = rrf_fuse(path_items, key_fn=lambda it: it.node_id,
                             weights=weights)
        else:
            fused = score_fuse(path_items, key_fn=lambda it: it.node_id,
                               score_fn=lambda it: it.score, weights=weights)
        out: List[LinkedItem] = []
        for it, fscore in fused:
            it.score = fscore
            out.append(it)
        return out[:top_k]

    def link_columns(self, question: str, top_k: int = 10) -> List[LinkedItem]:
        return self.link(question, top_k, include_tables=False,
                         include_columns=True)

    def link_tables(self, question: str, top_k: int = 5) -> List[LinkedItem]:
        return self.link(question, top_k, include_tables=True,
                         include_columns=False)

    # ---- internals: lexical seeds (P0) ----

    def _seed_nodes(self, question: str) -> Dict[str, float]:
        """Find seed nodes from term bindings and literal name matches."""
        seeds: Dict[str, float] = {}
        text = question.lower()

        # Business terms carry the most weight: curated vocabulary, not incidental
        # string overlap. Match the English identifier AND the Chinese
        # definition/comment so "支付总额" reaches the metric whose definition is
        # exactly that. Synonym expansion: a hit also seeds same-meaning terms
        # ("实际车型" reaches the metric stored as "物理车型").
        for term in self._schema.terms():
            props = term.properties
            names = [term.name] + list(props.get("aliases", []))
            definition = str(props.get("definition", "") or props.get("comment", ""))
            syns = props.get("synonyms", [])
            if any(n and str(n).lower() in text for n in names):
                seeds[f"term:{term.name}"] = 1.0
                for syn in syns:
                    seeds.setdefault(f"term:{syn}", 0.7)
            elif definition and definition.lower() in text:
                seeds[f"term:{term.name}"] = 0.7
                for syn in syns:
                    seeds.setdefault(f"term:{syn}", 0.5)

        # Literal column / table name + comment matches (substring, CJK-aware).
        for node_id, node in self._schema.nodes.items():
            if node.node_type == NodeType.TERM:
                continue
            if node.name and node.name.lower() in text:
                seeds[node_id] = 0.8
            elif node.properties.get("comment"):
                comment = str(node.properties["comment"]).lower()
                if comment and comment in text:
                    seeds[node_id] = 0.5

        if not seeds:
            # Fall back to token overlap so that a question with no exact
            # match still gets some entry point into the graph.
            seeds = self._fuzzy_seeds(text)
        return seeds

    def _fuzzy_seeds(self, text: str) -> Dict[str, float]:
        """BM25 lexical recall as the last-resort seed source (CJK-aware)."""
        q_tokens = _tokens(text)
        if not q_tokens:
            return {}
        if self._use_bm25:
            return self._bm25_seeds(q_tokens)
        # legacy token-overlap containment
        seeds: Dict[str, float] = {}
        for node_id, node in self._schema.nodes.items():
            if node.node_type == NodeType.TERM:
                continue
            surface = " ".join([
                node.name or "",
                str(node.properties.get("comment", "")),
                str(node.properties.get("table", "")),
            ])
            n_tokens = _tokens(surface)
            hit = len(q_tokens & n_tokens)
            if hit:
                seeds[node_id] = 0.3 * hit / len(q_tokens)
        return seeds

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        docs: List[tuple] = []
        for node_id, node in self._schema.nodes.items():
            if node.node_type == NodeType.TERM:
                continue
            surface = " ".join([
                node.name or "",
                str(node.properties.get("comment", "")),
                str(node.properties.get("table", "")),
            ])
            docs.append((node_id, list(_tokens(surface))))
        self._bm25 = _BM25Index(docs)

    def _bm25_seeds(self, q_tokens) -> Dict[str, float]:
        self._ensure_bm25()
        scores = self._bm25.score(q_tokens)
        mx = max(scores.values()) if scores else 0.0
        if not mx:
            return {}
        # Fuzzy seeds are a last resort: keep them weak (0.1) so they never
        # out-compete precise lexical or semantic seeds in the PPR mix.
        return {nid: 0.1 * s / mx for nid, s in scores.items() if s > 0}

    # ---- internals: entity importance + rerank (P1) ----

    def _ensure_importance(self) -> None:
        """Table importance in [0,1]: lineage downstream, row count, fact flag."""
        if self._importance is not None:
            return
        downstream: Dict[str, int] = defaultdict(int)
        for e in self._schema.edges:
            if e.edge_type.value == "lineage":
                downstream[e.source] += 1
        imp: Dict[str, float] = {}
        for node in self._schema.tables():
            props = node.properties
            try:
                row = int(props.get("row_count", 0) or 0)
            except (TypeError, ValueError):
                row = 0
            fact = 1.0 if props.get("is_fact") else 0.0
            down = downstream.get(node.node_id, 0)
            imp[node.node_id] = (math.log1p(down) * 0.6
                                 + math.log1p(row) / 15.0 + fact * 0.5)
        mx = max(imp.values()) if imp else 0.0
        self._importance = (
            {k: (v / mx if mx else 0.0) for k, v in imp.items()} if imp else {}
        )

    def _rerank(self, items: List[LinkedItem],
                intent: Optional[str] = None) -> List[LinkedItem]:
        """Boost results by entity importance: score *= (1 + w * imp), then by
        question intent (table/field/metric) via type-weighted multipliers."""
        from ..query_understanding import intent_boost

        if not self._use_importance and not intent:
            return items

        w = self._importance_weight
        imp = {}
        if self._use_importance:
            self._ensure_importance()
            imp = self._importance or {}

        tboost = intent_boost(intent) if intent else {}
        for it in items:
            boost = 0.0
            if it.node_type == NodeType.TABLE.value:
                boost = imp.get(it.node_id, 0.0)
            elif it.node_type == NodeType.COLUMN.value and it.table:
                boost = imp.get(f"table:{it.table}", 0.0)
            factor = 1.0
            if tboost:
                factor = tboost.get(it.node_type, 1.0)
            if boost or factor != 1.0:
                it.score = it.score * (1.0 + w * boost) * factor
        items.sort(key=lambda x: x.score, reverse=True)
        return items

    # ---- internals: semantic seeds (P2) ----

    def _ensure_vector_index(self) -> None:
        """Embed every schema node once (lazy) into the vector store.

        Table / column nodes embed their ``name + comment + table`` surface.
        Term nodes — which carry the curated business vocabulary — are embedded
        too (``name + aliases + comment/definition``) so a question's semantic
        neighbours can include the term that binds to the right column. Terms
        never appear in link() output; they only seed the PPR.

        The store defaults to an in-process numpy index; an injected Milvus /
        OceanBase / RAG ``VectorStoreBase`` store is used when provided. Any
        build failure disables P2 (lexical-only) rather than crashing.
        """
        if self._vector_index is not None or self._vector_disabled:
            return

        ids: List[str] = []
        texts: List[str] = []
        for node_id, node in self._schema.nodes.items():
            if node.node_type == NodeType.TERM:
                aliases = " ".join(
                    str(a) for a in node.properties.get("aliases", []) if a
                )
                surface = " ".join([
                    node.name or "",
                    aliases,
                    str(node.properties.get("comment", "")),
                    str(node.properties.get("definition", "")),
                ])
            else:
                surface = " ".join([
                    node.name or "",
                    str(node.properties.get("comment", "")),
                    str(node.properties.get("table", "")),
                ])
            ids.append(node_id)
            texts.append(surface)

        try:
            vecs = [self._embedder(t) for t in texts]
            store = self._vector_store
            if store is None:
                store = NumpySchemaVectorStore()
                self._vector_store = store
            store.upsert(ids, [list(v) for v in vecs])
            self._vector_index = True
        except Exception as exc:  # model/network/store failure -> lexical only
            log.warning("vector index build failed; P2 disabled: %s", exc)
            self._vector_disabled = True
            self._vector_index = True

    def _vector_seeds(self, question: str) -> Dict[str, float]:
        """Top-k nearest nodes to the question, as cosine-weighted seeds."""
        if self._vector_disabled:
            return {}
        self._ensure_vector_index()
        if not self._vector_index or self._vector_store is None:
            return {}

        try:
            q = list(self._embedder(question))
        except Exception as exc:
            log.warning("question embedding failed; P2 skipped: %s", exc)
            return {}
        try:
            hits = self._vector_store.search(q, self._top_k_vector)
        except Exception as exc:
            log.warning("vector search failed; P2 skipped: %s", exc)
            return {}
        return {
            nid: float(self._vector_weight * max(0.0, sim))
            for nid, sim in hits
            if sim > 0
        }

    # ---- internals: ppr + materialisation ----

    def _ppr(self, seeds: Dict[str, float]) -> Dict[str, float]:
        """Propagate seed relevance through the schema graph."""
        return self._engine.personalized_pagerank(seeds, alpha=self._alpha)

    def _to_items(
        self,
        scores: Dict[str, float],
        include_tables: bool,
        include_columns: bool,
    ) -> List[LinkedItem]:
        items: List[LinkedItem] = []
        for node_id, score in scores.items():
            node = self._schema.nodes.get(node_id)
            if node is None:
                continue
            if node.node_type == NodeType.TABLE and not include_tables:
                continue
            if node.node_type == NodeType.COLUMN and not include_columns:
                continue
            if node.node_type == NodeType.TERM:
                continue
            table = ""
            if node.node_type == NodeType.COLUMN:
                table = str(node.properties.get("table", ""))
            items.append(
                LinkedItem(
                    node_id=node_id,
                    name=node.name,
                    node_type=node.node_type.value,
                    score=score,
                    table=table,
                    properties=dict(node.properties),
                )
            )
        items.sort(key=lambda x: x.score, reverse=True)
        return items
