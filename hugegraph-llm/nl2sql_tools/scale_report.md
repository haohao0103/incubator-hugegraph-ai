# NL2SQL 规模基准（合成 schema）

- vermeer: http://127.0.0.1:6688 | queries: 15

| tables | nodes | BM25 build(s) | local p50(ms) | local p99(ms) | vm load(s) | vm p50(ms) | vm p99(ms) | top5 agree |
|---|---|---|---|---|---|---|---|---|
| 100 | 1105 | 0.7 | 5.57 | 8.19 | 0.02 | 0.5 | 0.54 | 1.0 |
| 300 | 3315 | 0.37 | 9.92 | 11.45 | 0.15 | 1.56 | 2.23 | 1.0 |
