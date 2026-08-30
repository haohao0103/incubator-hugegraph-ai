"""Ablation: BM25 / importance rerank effect on the best combined config.

Compares w=0.5 k=3 across: default(BM25+imp) / no-importance / no-bm25 /
neither, to attribute the R@5/MRR delta after the P0/P1 upgrades.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from p2_corpus import build_warehouse_schema  # noqa: E402
from p2_embedder import make_embedder  # noqa: E402
from hugegraph_llm.nl2sql.linking.schema_linker import SchemaLinker  # noqa: E402
from p2_stress import evaluate, aggregate  # noqa: E402

schema = build_warehouse_schema()
embedder = make_embedder()

configs = [
    ("default(bm25+imp)", {}),
    ("no-importance", {"use_importance": False}),
    ("no-bm25", {"use_bm25": False}),
    ("neither", {"use_bm25": False, "use_importance": False}),
    ("imp_w=0.1", {"importance_weight": 0.1}),
]
print(f"{'config':20} | R@1  R@3  R@5   MRR  cov | semR@5 p0missR@5")
for name, kw in configs:
    lk = SchemaLinker(schema, embedder=embedder, top_k_vector=3,
                      vector_weight=0.5, **kw)
    rows = evaluate(lk, "P0+P2")
    a = aggregate(rows)
    sem = aggregate(rows, "semantic")
    miss = [r for r in rows if not r["any_hit"]]
    print(f"{name:20} | {a['recall@1']:4.2f} {a['recall@3']:4.2f} "
          f"{a['recall@5']:4.2f} {a['mrr']:5.3f} {a['coverage']:4.2f} "
          f"| {sem['recall@5']:5.2f} {aggregate(miss)['recall@5']:7.2f}")
