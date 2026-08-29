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

"""Real-HugeGraph validation of EntityResolution.merge_mode="synonym_edge".

Uses the EXISTING schema on the live `hugegraph` graph (zero schema changes):
  - vertex label: Person   (CUSTOMIZE_STRING id; props: name, type, confidence)
  - edge label:   KNOWS    (source=Person, target=Person, NO required properties)

To avoid touching production data, resolution is scoped to exactly two freshly
inserted duplicate Person vertices (passed in the run context), so only those two
get a KNOWS synonym edge. Both vertices must survive, exactly one KNOWS edge must
be written, and it must be readable back. Test vertices are removed afterwards.
"""

import json

from pyhugegraph.client import PyHugeClient

from hugegraph_llm.operators.graph_op.entity_resolution import EntityResolution

HOST, PORT, GRAPH = "localhost", 8080, "hugegraph"
USER, PWD = "admin", "admin"
VLABEL = "Person"
ELABEL = "KNOWS"
UNIQUE_NAME = "ERSynonymValidationApple_2026"
ID1, ID2 = "ersyn_v1", "ersyn_v2"

# reuse the operator's Gremlin-safe id quoting
g_id = EntityResolution._g_id


def log(msg):
    print(msg, flush=True)


def _count(resp):
    if isinstance(resp, dict) and "data" in resp:
        data = resp["data"]
        return data[0] if isinstance(data, list) and data else 0
    return resp


def _edges(resp):
    if isinstance(resp, dict) and "data" in resp:
        return resp["data"]
    return resp if isinstance(resp, list) else []


def _drop_named_vertices(client, label, name):
    resp = client.gremlin().exec(
        "g.V().hasLabel('%s').project('id','name').by(id()).by('name').toList()" % label
    )
    rows = resp.get("data", []) if isinstance(resp, dict) else resp
    dropped = 0
    for r in rows or []:
        if isinstance(r, dict) and r.get("name") == name:
            vid = r.get("id")
            client.gremlin().exec("g.V(%s).bothE().drop().iterate()" % g_id(vid))
            client.gremlin().exec("g.V(%s).drop().iterate()" % g_id(vid))
            dropped += 1
    return dropped


def main():
    client = PyHugeClient("http://%s:%s" % (HOST, PORT), GRAPH, USER, PWD)
    log("connected to %s:%s graph=%s" % (HOST, PORT, GRAPH))

    # idempotent pre-cleanup of any orphaned vertices from a prior run
    for label in (VLABEL, "TestLabel"):
        n = _drop_named_vertices(client, label, UNIQUE_NAME)
        if n:
            log("pre-cleanup: removed %d orphan vertex(es) on '%s'" % (n, label))

    # 1) insert two duplicate Person vertices (same name, distinct custom ids)
    client.graph().addVertex(VLABEL, {"name": UNIQUE_NAME, "type": "person"}, id=ID1)
    client.graph().addVertex(VLABEL, {"name": UNIQUE_NAME, "type": "person"}, id=ID2)
    log("created duplicate vertices: %s, %s" % (ID1, ID2))

    try:
        # 2) resolve ONLY these two vertices (passed in context -> graph store
        #    is untouched beyond the synonym-edge write for this pair).
        resolver = EntityResolution(
            client=client,
            strategy="exact_match",
            merge_mode="synonym_edge",
            synonym_edge_label=ELABEL,
            resolve_properties=["name"],
        )
        context = resolver.run(
            {
                "vertices": [
                    {"id": ID1, "label": VLABEL, "properties": {"name": UNIQUE_NAME}, "degree": 0},
                    {"id": ID2, "label": VLABEL, "properties": {"name": UNIQUE_NAME}, "degree": 0},
                ]
            }
        )
        rr = context["resolution_result"]
        log("resolution_result=%s" % json.dumps(rr, ensure_ascii=False))

        # 3) verify both vertices still exist
        both_exist = client.gremlin().exec("g.V(%s,%s).count().next()" % (g_id(ID1), g_id(ID2)))
        log("vertex count after resolution (expect 2): %s" % both_exist)

        # 4) verify the synonym edge exists and is readable
        edge_check = client.gremlin().exec(
            "g.V(%s).bothE().where(label().is('%s')).dedup().toList()" % (g_id(ID1), ELABEL)
        )
        edges = _edges(edge_check)
        log("synonym edges touching %s (expect 1): %s" % (ID1, json.dumps(edges, ensure_ascii=False)))

        # 5) assertions
        assert rr["merged_count"] == 1, "merged_count should be 1, got %s" % rr["merged_count"]
        assert rr["synonym_edges"] == 1, "synonym_edges should be 1, got %s" % rr["synonym_edges"]
        assert rr["deprecated_vids"] == [], \
            "synonym mode must not deprecate, got %s" % rr["deprecated_vids"]
        assert _count(both_exist) == 2, "both vertices must survive, got %s" % both_exist
        assert len(edges) == 1, "expected exactly 1 %s edge, got %s" % (ELABEL, edges)
        log("RESULT: PASS - synonym_edge mode verified on real HugeGraph "
            "(both vertices preserved, 1 %s edge written & read back)" % ELABEL)
    finally:
        # 6) cleanup: drop the two test vertices and their edges
        try:
            for vid in (ID1, ID2):
                client.gremlin().exec("g.V(%s).bothE().drop().iterate()" % g_id(vid))
                client.gremlin().exec("g.V(%s).drop().iterate()" % g_id(vid))
            log("cleanup: test vertices %s,%s and their edges removed" % (ID1, ID2))
        except Exception as e:  # noqa: BLE001
            log("cleanup warning: %s" % e)


if __name__ == "__main__":
    main()
