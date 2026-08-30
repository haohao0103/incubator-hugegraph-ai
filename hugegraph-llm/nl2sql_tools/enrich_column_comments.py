"""Enrich SchemaMetadata with LLM-generated Chinese comments for columns
that lack one.

Short physical names ("pay_amt") embed poorly; a generated business comment
("支付金额，订单支付流水金额") gives both the lexical matcher (P0) and the
semantic embedder (P2) real signal. Batch mode asks the LLM for several
columns at once to keep call count low; per-column fallback on parse failure.

Usage:
  python scripts/enrich_column_comments.py --meta <SchemaMetadata.json> \
      --out <enriched.json> [--batch 20] [--dry-run]
"""
import argparse
import json
import os
import sys
import time

LOG_PATH = "_out/enrich/logs/enrich.log"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _llm(prompt: str) -> str:
    from hugegraph_llm.models.llms.init_llm import LLMs
    return LLMs().get_chat_llm().generate(prompt=prompt)


_BATCH_PROMPT = (
    "你是数仓元数据专家。为下列每个字段生成一句简洁的中文业务注释"
    "（15字以内，说明业务含义，不要写表名本身）。严格按这个格式输出，每行一条：\n"
    "字段名 -> 注释\n"
    "{cols}\n"
    "如果某字段无法推断含义，输出：字段名 -> NULL"
)


def _parse_batch(raw: str) -> dict:
    out = {}
    for line in raw.splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        k, _, v = line.partition("->")
        k = k.strip()
        v = v.strip()
        if k and v and v.upper() != "NULL":
            out[k] = v
    return out


def enrich(meta: dict, batch: int = 20, dry_run: bool = False) -> dict:
    cols = meta.get("columns", [])
    todo = [c for c in cols if not (c.get("comment") or "").strip()]
    log(f"columns={len(cols)} missing_comment={len(todo)} batch={batch} dry={dry_run}")
    if dry_run:
        return meta
    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        lines = []
        for c in chunk:
            tbl = c.get("table", "")
            lines.append(f"{tbl}.{c['name']}")
        prompt = _BATCH_PROMPT.format(cols="\n".join(lines))
        try:
            raw = _llm(prompt)
            parsed = _parse_batch(raw)
        except Exception as exc:  # noqa: BLE001
            log(f"batch {i // batch} failed: {exc}; fallback per-column")
            parsed = {}
            for c in chunk:
                try:
                    r2 = _llm(_BATCH_PROMPT.format(cols=f"{c.get('table','')}.{c['name']}"))
                    parsed.update(_parse_batch(r2))
                except Exception as exc2:  # noqa: BLE001
                    log(f"  {c['name']} failed: {exc2}")
        hit = 0
        for c in chunk:
            key = f"{c.get('table', '')}.{c['name']}"
            if key in parsed:
                c["comment"] = parsed[key]
                hit += 1
        done += hit
        log(f"batch {i // batch + 1}: enriched {hit}/{len(chunk)} "
            f"(running {done}/{len(todo)})")
    log(f"done: enriched {done} columns")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log("=== enrich column comments ===")
    with open(args.meta, encoding="utf-8") as f:
        meta = json.load(f)
    meta = enrich(meta, batch=args.batch, dry_run=args.dry_run)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"wrote {args.out}")
    log("DONE")


if __name__ == "__main__":
    main()
