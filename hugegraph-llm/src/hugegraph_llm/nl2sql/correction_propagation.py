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
L3 correction recall with semantic-edge graph propagation (NL2SQL).

Why this exists
---------------
The NL2SQL chain records analyst corrections as ``CorrectionDecision``
vertices with provenance edges (``correctionAppliesToTerm/Field/Caliber``).
A naive recall only surfaces corrections attached to the *exact* seed nodes a
question hits lexically. But analysts attach corrections to whatever node was
at fault — often a caliber (``GMV 只统计 paid``), a synonym, or a node one
hop up/down a metric chain. Rephrasing the question must still recall them.

This module propagates the seed set along the *semantic* edges of the
in-memory :class:`~hugegraph_llm.nl2sql.schema_graph.model.SchemaGraph`:

    term <-> column   (TERM_MAPS edge)
    column -> table   (BELONGS_TO edge)
    term  <-> term    (synonym, via term.properties["synonyms"])
    term  <-> caliber (caliber, via term.properties["calibers"])

and collects every ``node.properties["corrections"]`` reached, deduped by
correction id (one correction may attach to several endpoints, and a
propagated path may hit several endpoints of the same correction).

Deliberately **not** propagated: lineage / foreign-key / co-occurrence edges
— join structure is irrelevant to correction semantics and would scatter
corrections onto unrelated tables (same decision as the PoC ``SEMANTIC_EDGES``).

The propagation runs on the in-memory graph the loader already built, so no
per-node Gremlin round-trips are needed (unlike the PoC, which ran per-node
``bothE(...).bothV()`` against HugeGraph).
"""

from typing import Dict, List, Set

from hugegraph_llm.utils.log import log

from .schema_graph.model import EdgeType, SchemaGraph


def semantic_neighbors(schema: SchemaGraph, node_id: str) -> Set[str]:
    """Return the semantic neighbours of ``node_id`` in the SchemaGraph.

    Semantic edges only: term<->column bindings, column->table ownership,
    term synonyms and term calibers (the latter two live in node properties
    rather than as graph edges).
    """
    out: Set[str] = set()
    for e in schema.edges:
        if e.edge_type not in (EdgeType.TERM_MAPS, EdgeType.BELONGS_TO):
            continue
        if e.source == node_id:
            out.add(e.target)
        elif e.target == node_id:
            out.add(e.source)
    node = schema.nodes.get(node_id)
    if node is None:
        return out
    props = node.properties or {}
    for syn in props.get("synonyms", []) or []:
        out.add(f"term:{syn}")
    return out


def propagate_seeds(
    schema: SchemaGraph,
    seed_ids: Set[str],
    hops: int = 2,
) -> Set[str]:
    """BFS from ``seed_ids`` along semantic edges for up to ``hops`` hops.

    Returns the full reachable set (including the seeds themselves).
    """
    reached: Set[str] = set(seed_ids)
    frontier: Set[str] = set(seed_ids)
    for _ in range(hops):
        nxt: Set[str] = set()
        for nid in frontier:
            for nb in semantic_neighbors(schema, nid):
                if nb not in reached:
                    reached.add(nb)
                    nxt.add(nb)
        frontier = nxt
        if not frontier:
            break
    return reached


def fetch_corrections(
    schema: SchemaGraph,
    seed_ids: Set[str],
    hops: int = 2,
) -> tuple:
    """Recall corrections from all nodes reachable via semantic propagation.

    :param schema: in-memory SchemaGraph (corrections folded into
        ``node.properties["corrections"]`` by the HG loader).
    :param seed_ids: seed node ids (``term:...`` / ``column:...`` / ``table:...``).
    :param hops: propagation depth along semantic edges.
    :return: ``(corrections, stats)`` — corrections is a list of
        ``{id, question, wrong_sql, correct_sql, correction_reason}`` deduped
        by id; stats is ``{"seed", "propagated", "reached"}`` for logging the
        difference graph propagation made vs seed-only recall.
    """
    reached = propagate_seeds(schema, seed_ids, hops=hops)
    out: List[dict] = []
    seen: Set[str] = set()
    for nid in reached:
        node = schema.nodes.get(nid)
        if node is None:
            continue
        for corr in node.properties.get("corrections", []) or []:
            cid = corr.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                out.append(corr)
    stats = {
        "seed": sorted(seed_ids),
        "propagated": sorted(reached - set(seed_ids)),
        "reached": sorted(reached),
    }
    if out:
        log.info(
            "correction recall: seed=%d propagated=%d reached=%d corrections=%d",
            len(seed_ids), len(stats["propagated"]), len(stats["reached"]),
            len(out),
        )
    return out, stats
