"""P0-only smoke test: build corpus, run lexical linker, verify gold surfaced.
No embedder / network needed. Validates corpus + linker wiring."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import (build_warehouse_schema, WAREHOUSE_QUESTIONS,
                       gold_node_ids, gold_table_ids)
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker

g = build_warehouse_schema()
print(f"tables={len(g.tables())} cols={len(g.columns())} terms={len(g.terms())} edges={len(g.edges)}")
lk = SchemaLinker(g)
TOP = 10
ok = 0
for q in WAREHOUSE_QUESTIONS:
    items = lk.link(q["q"], top_k=TOP)
    ids = [i.node_id for i in items]
    gids = set(gold_node_ids(q))
    hit = bool(gids & set(ids))
    if hit:
        ok += 1
    print(f"[{q['category']:8}] {'OK ' if hit else 'MISS'} {q['q']:18} gold={q['gold']} top3={[i.node_id.replace('column:dw.','') for i in items[:3]]}")
print(f"\nP0 lexical recall@10 = {ok}/{len(WAREHOUSE_QUESTIONS)} = {ok/len(WAREHOUSE_QUESTIONS):.2f}")
