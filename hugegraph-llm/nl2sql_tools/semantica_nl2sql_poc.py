# -*- coding: utf-8 -*-
"""场景一 PoC 骨架：semantica 作为 NL2SQL 的「语义中间层」。

三层架构（本文件演示全链路，可本机直接跑通）：
  L1 混合检索：在 HugeGraph「业务术语本体」上做 (a) 术语/字段/表的词法召回
              + (b) 图遍历（term→field 映射、field→table、table join path、
                metric→caliber 口径）。
  L2 提示组装：把召回的 schema 上下文 + 历史纠错拼成 prompt，交给 MiMo 生成 SQL。
  L3 纠错决策记录：用户修正 SQL 时，落一条带 provenance 的 CorrectionDecision 节点，
              通过 appliesTo 边挂回相关术语/口径 —— 这是相对 Vanna.ai 的核心差异化
              （Vanna 只存 (question, sql) 对，不存「为什么错、挂到哪个口径」）。
              召回时**沿语义边做图传播**（pocSynonym / pocComputedFrom* /
              pocTermField / pocMetricField / pocHasColumn / pocHasCaliber），
              挂在同义词、指标链上游/下游节点的纠错，换种说法提问也能被召回。

关键约束（已踩坑，务必遵守）：
  * HugeGraph 1.7 的 Gremlin 通道只跑 Gremlin，semantica 高层 DecisionRecorder/
    DecisionQuery/ContextRetriever 会吐 Cypher -> 在本后端直接调用必崩。
    因此本 PoC 只用通用 GraphStore API：create_node / create_relationship /
    execute_query(gremlin)，不走高层 Cypher API。
  * 图创建接口本实例被禁（405），直接复用已存在的 kg_rag。
  * 检索的图扩展一律用 g.V('<逻辑id>') 按主键 + 边遍历，避免依赖 HG 的搜索/二级索引
    （textContains / has('prop',val) 在无索引时会报 No index）。
  * 标签命名冲突（本次踩坑）：kg_rag 里已有 e2e 灌库创建的标签（顶点 Metric 为
    PRIMARY_KEY id 策略、边 hasColumn/synonym/computedFromField/lineage 为固定
    端点对）。同名 CUSTOMIZE_STRING 顶点写入会报 "Can't customize vertex id when
    id strategy is 'PRIMARY_KEY'"；复用已有边标签且端点标签对不一致时，store 会走
    「删旧标签再重建」路径（异步删除有竞态，且会连带删掉该标签下其它边——本次已
    误删 e2e 的 hasColumn 标签）。因此本 PoC 的标签一律加 PoC 前缀
    （PoCTable/PoCField/PoCMetric），且**每条边标签严格对应唯一 (源标签,目标标签)
    对**（pocHasColumn / pocTermField / pocMetricField / pocComputedFromMetric /
    pocComputedFromTerm / pocJoinPath / pocHasCaliber / pocSynonym /
    pocAppliesToTerm|Caliber|Field），与仓库 schema 完全隔离。
  * HG 的 Gremlin 对「未定义的 label」做 hasLabel 会抛 Undefined vertex label，
    整条查询作废（reset 必须先查 schema 再 drop，见 reset_ontology）。
  * 逻辑 id 禁止含冒号（本次踩坑）：semantica hg-backend 的 create_relationship
    会把 ``<id>:<suffix>`` 误当作 HG 复合 id（``<numeric>:<label>``），把 source/
    target 标签解析成错误值（如 "orders"），边标签创建直接 400。本 PoC 全部 id
    用下划线命名（tbl_orders / fld_orders_amount / term_gmv / metric_gmv /
    cal_gmv_paid / corr_xxxx）。

运行方式（本机）：
  cd <repo_root>
  PYTHONPATH=/Users/mac/Desktop/apache-code/hugegraph-dev/semantica-hg-backend \\
  OPENAI_CHAT_API_BASE=https://api.xiaomimimo.com/v1 \\
  OPENAI_CHAT_API_KEY=<你的 MiMo key> \\
  OPENAI_CHAT_LANGUAGE_MODEL=mimo-v2.5-pro \\
  /Users/mac/.venvs/semantica/bin/python3.12 \\
      hugegraph-llm/nl2sql_tools/semantica_nl2sql_poc.py

（若未设置 OPENAI_CHAT_API_KEY，L2 会跳过真实调用、用占位 SQL 完成 L1/L3 演示，
 不会报错退出。）
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

# ---- 路径（绝对路径，符合项目规范）------------------------------------------
REPO = Path(__file__).resolve().parents[2]                      # .../incubator-hugegraph-ai
LOG_PATH = REPO / "_out" / "semantica_poc" / "logs" / "semantica_poc.log"
OUT_JSON = REPO / "_out" / "semantica_poc" / "semantica_poc.json"

# 复用已存在的图（创建被 405 禁）
HG_URL = "http://127.0.0.1:8081"
HG_GRAPH = "kg_rag"

# MiMo（OpenAI 兼容）配置，优先读环境变量，缺省给小米默认
MIMO_BASE = os.environ.get("OPENAI_CHAT_API_BASE", "https://api.xiaomimimo.com/v1")
MIMO_KEY = os.environ.get("OPENAI_CHAT_API_KEY", "")
MIMO_MODEL = os.environ.get("OPENAI_CHAT_LANGUAGE_MODEL", "mimo-v2.5-pro")


# --------------------------------------------------------------------------- #
# 日志（落盘 + 打印，符合「所有跑数脚本必须留日志」规范）
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def unwrap(props: dict) -> dict:
    """TinkerPop valueMap 把单值包成 list，这里拆回标量。"""
    out = {}
    for k, v in (props or {}).items():
        if isinstance(v, list) and len(v) == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


def gq(store, script: str) -> list:
    """执行 Gremlin，返回 records（归一化后的 dict 列表）。"""
    return store.execute_query(script).get("records", []) or []


def keywordize(text: str) -> list:
    """极简中文/英文切词：去标点、按空格/常见分隔拆，过滤单字噪声。"""
    import re
    toks = re.split(r"[\s,，。、？?；;：:（）()\[\]]+", text.lower())
    return [t for t in toks if len(t) >= 1]


# --------------------------------------------------------------------------- #
# 本体写入（语义层 = semantica 的通用 GraphStore，不用 Cypher 高层 API）
# --------------------------------------------------------------------------- #
def reset_ontology(store) -> None:
    """清空本 PoC 写入的节点（按 label 批量 drop，detach 级联删边）。

    坑：HG 的 Gremlin 对「未定义的 label」做 hasLabel 会直接抛
    "Undefined vertex label"，整个查询作废。因此先查现有 vertex labels，
    只 drop 确实存在的 PoC 标签。
    """
    import requests as _rq
    poc_labels = ("BusinessTerm", "PoCMetric", "Caliber", "PoCField",
                  "PoCTable", "CorrectionDecision")
    try:
        r = _rq.get(f"{HG_URL}/graphs/{HG_GRAPH}/schema/vertexlabels", timeout=15)
        existing = {v["name"] for v in r.json().get("vertexlabels", [])}
        to_drop = [l for l in poc_labels if l in existing]
        if not to_drop:
            log("reset: 无遗留 PoC 标签")
            return
        ls = ",".join(f"'{l}'" for l in to_drop)
        gq(store, f"g.V().hasLabel({ls}).drop()")
        log(f"reset: dropped {to_drop}")
    except Exception as e:  # noqa: BLE001
        log(f"reset: 跳过（可能本就为空） {type(e).__name__}: {str(e)[:100]}")


def add_node(store, label: str, nid: str, props: dict) -> None:
    store.create_node([label], {**props, "id": nid})


def add_edge(store, src: str, dst: str, rel: str, props: dict | None = None) -> None:
    store.create_relationship(src, dst, rel, props or {})


def build_demo_ontology(store) -> None:
    """搭一个小型「销售数仓」业务术语本体，作为语义中间层示范。

    重要：逻辑 id 一律**不含冒号**。semantica hg-backend 的 create_relationship
    会把 ``<id>:<suffix>`` 误当作 HG 复合 id（``<numeric>:<label>``），导致边标签
    source/target 被解析成错误值（如 "orders"），HG 直接 400。见模块 docstring。
    """
    # ---- 表 ----
    tables = {
        "tbl_orders":      {"name": "orders", "comment": "订单主表", "row_count": 1_200_000},
        "tbl_users":       {"name": "users", "comment": "用户表", "row_count": 800_000},
        "tbl_products":    {"name": "products", "comment": "商品表", "row_count": 50_000},
        "tbl_order_items": {"name": "order_items", "comment": "订单明细表", "row_count": 5_000_000},
    }
    for nid, p in tables.items():
        add_node(store, "PoCTable", nid, p)

    # ---- 字段 ----
    fields = {
        "fld_orders_id":         {"name": "id", "table": "orders", "data_type": "bigint", "comment": "订单号"},
        "fld_orders_user_id":    {"name": "user_id", "table": "orders", "data_type": "bigint", "comment": "下单用户"},
        "fld_orders_amount":     {"name": "amount", "table": "orders", "data_type": "decimal", "comment": "订单金额(元)"},
        "fld_orders_status":     {"name": "status", "table": "orders", "data_type": "string", "comment": "订单状态 paid/refund"},
        "fld_orders_created_at": {"name": "created_at", "table": "orders", "data_type": "timestamp", "comment": "下单时间"},
        "fld_users_id":          {"name": "id", "table": "users", "data_type": "bigint", "comment": "用户号"},
        "fld_users_city":        {"name": "city", "table": "users", "data_type": "string", "comment": "注册城市"},
        "fld_users_reg_channel": {"name": "reg_channel", "table": "users", "data_type": "string", "comment": "注册渠道"},
        "fld_products_id":       {"name": "id", "table": "products", "data_type": "bigint", "comment": "商品号"},
        "fld_products_category": {"name": "category", "table": "products", "data_type": "string", "comment": "商品品类"},
        "fld_order_items_order_id":   {"name": "order_id", "table": "order_items", "data_type": "bigint", "comment": "所属订单"},
        "fld_order_items_product_id": {"name": "product_id", "table": "order_items", "data_type": "bigint", "comment": "商品"},
        "fld_order_items_qty":        {"name": "qty", "table": "order_items", "data_type": "int", "comment": "购买数量"},
    }
    for nid, p in fields.items():
        add_node(store, "PoCField", nid, p)

    # ---- 业务术语（BusinessTerm）----
    terms = {
        "term_gmv":      {"name": "GMV", "aliases": "成交总额,毛交易额,gross merchandise value",
                          "definition": "一段时间内所有已支付订单的金额之和", "domain": "交易"},
        "term_paid_orders": {"name": "支付订单数", "aliases": "支付单量,paid orders",
                            "definition": "status='paid' 的订单条数", "domain": "交易"},
        "term_aov":      {"name": "客单价", "aliases": "平均客单,avg order value",
                          "definition": "GMV 除以支付订单数", "domain": "交易"},
        "term_dau":      {"name": "DAU", "aliases": "日活,日活跃用户",
                          "definition": "当日活跃用户数", "domain": "用户"},
        "term_new_users": {"name": "新增用户", "aliases": "新客,new users",
                          "definition": "首次注册的用户数", "domain": "用户"},
    }
    for nid, p in terms.items():
        add_node(store, "BusinessTerm", nid, p)

    # ---- 指标（PoCMetric，带公式/口径引用）----
    metrics = {
        "metric_gmv": {"name": "gmv", "formula": "sum(orders.amount) where status='paid'",
                       "grain": "order", "unit": "yuan"},
        "metric_aov": {"name": "aov", "formula": "gmv / paid_orders",
                       "grain": "order", "unit": "yuan"},
    }
    for nid, p in metrics.items():
        add_node(store, "PoCMetric", nid, p)

    # ---- 口径（Caliber，口径统一是语义层核心价值）----
    calibers = {
        "cal_gmv_paid": {"name": "GMV口径", "dimension": "status",
                         "description": "GMV 仅统计 status='paid' 的订单 amount 之和；退款订单不计，不含未支付"},
        "cal_aov_def":  {"name": "客单价口径", "dimension": "grain",
                         "description": "客单价=GMV/支付订单数，分母为支付订单数而非下单订单数"},
    }
    for nid, p in calibers.items():
        add_node(store, "Caliber", nid, p)

    # ---- 边（HG 边标签的 source/target 是固定标签对；不同端点对必须用不同
    #      边标签，否则会触发 store 的「删旧建新」路径——删除是异步的、有竞态，
    #      且会连带删掉同标签下的其他边。因此这里按 (源标签,目标标签) 一一命名）----
    # pocHasColumn: PoCTable -> PoCField
    col_map = {
        "tbl_orders": ["fld_orders_id", "fld_orders_user_id", "fld_orders_amount",
                       "fld_orders_status", "fld_orders_created_at"],
        "tbl_users": ["fld_users_id", "fld_users_city", "fld_users_reg_channel"],
        "tbl_products": ["fld_products_id", "fld_products_category"],
        "tbl_order_items": ["fld_order_items_order_id", "fld_order_items_product_id", "fld_order_items_qty"],
    }
    for t, fs in col_map.items():
        for f in fs:
            add_edge(store, t, f, "pocHasColumn", {})

    # pocTermField: BusinessTerm -> PoCField / pocMetricField: PoCMetric -> PoCField
    add_edge(store, "term_gmv", "fld_orders_amount", "pocTermField", {"role": "measure"})
    add_edge(store, "term_paid_orders", "fld_orders_status", "pocTermField", {"role": "filter"})
    add_edge(store, "term_aov", "metric_gmv", "pocComputedFromMetric", {"expr": "gmv"})
    add_edge(store, "term_aov", "term_paid_orders", "pocComputedFromTerm", {"expr": "paid_orders"})
    add_edge(store, "metric_gmv", "fld_orders_amount", "pocMetricField", {"role": "measure"})
    add_edge(store, "metric_gmv", "fld_orders_status", "pocMetricField", {"role": "filter"})

    # pocJoinPath: PoCTable -> PoCTable（带上 join 条件）
    add_edge(store, "tbl_orders", "tbl_users", "pocJoinPath",
             {"on": "orders.user_id = users.id", "type": "many_to_one"})
    add_edge(store, "tbl_order_items", "tbl_orders", "pocJoinPath",
             {"on": "order_items.order_id = orders.id", "type": "many_to_one"})
    add_edge(store, "tbl_order_items", "tbl_products", "pocJoinPath",
             {"on": "order_items.product_id = products.id", "type": "many_to_one"})

    # pocHasCaliber: PoCMetric -> Caliber
    add_edge(store, "metric_gmv", "cal_gmv_paid", "pocHasCaliber", {})
    add_edge(store, "metric_aov", "cal_aov_def", "pocHasCaliber", {})

    # pocSynonym: BusinessTerm -> BusinessTerm
    add_node(store, "BusinessTerm", "term_gmv_en",
             {"name": "Gross Merchandise Value", "aliases": "gmv", "definition": "GMV 的英文全称", "domain": "交易"})
    add_edge(store, "term_gmv", "term_gmv_en", "pocSynonym", {})

    log("ontology: 已写入 4 表 / 13 字段 / 6 术语 / 2 指标 / 2 口径 + 关联边")


# --------------------------------------------------------------------------- #
# L1 混合检索
# --------------------------------------------------------------------------- #
def fetch_all_terms(store) -> dict:
    """一次性拉全 PoC 术语节点做词法召回缓存（按 label 过滤无需索引）。"""
    recs = gq(store, "g.V().hasLabel('BusinessTerm','PoCMetric','Caliber','PoCField',"
                  "'PoCTable').project('id','label','props')"
                  ".by(id).by(label).by(valueMap())")
    cache = {}
    for r in recs:
        cache[r["id"]] = {"label": r["label"], "props": unwrap(r.get("props", {}))}
    return cache


def lexical_match(cache: dict, question: str) -> list:
    """双向词法召回（中文友好）：

    ① 节点 name/aliases 的 token 出现在问题里（强信号，如问题含 "GMV"、
       "客单价"）；② 问题中的词出现在节点文本里（弱信号，如 "城市" 命中
       city 字段注释）。
    返回 [(id, score)]，降序。中文没有天然分隔符，必须按 token-in-question
    方向匹配，而不是把整个中文问题当一个 token 去子串匹配。
    """
    q = question.lower()
    hits = []
    for nid, info in cache.items():
        props = info["props"]
        score = 0
        # ① 节点自身 token（名称/别名/定义）出现在问题里
        for key in ("name", "aliases", "definition"):
            weight = 3 if key in ("name", "aliases") else 1
            for tok in keywordize(str(props.get(key, ""))):
                if len(tok) >= 2 and tok in q:
                    score += weight
        # ② 问题 token 出现在节点文本里（英文/数字词为主，如 "city"、"2024"）
        blob = " ".join(str(v) for v in props.values()).lower()
        for kw in keywordize(q):
            if len(kw) >= 2 and kw in blob:
                score += 2
        if score:
            hits.append((nid, score))
    hits.sort(key=lambda x: -x[1])
    return hits


def expand_context(store, seed_ids: set) -> dict:
    """从命中的种子节点做图遍历，收集字段/表/join路径/口径。

    要点：
      * join 做两跳闭包（orders -> order_items -> products），把间接相关的
        表也带进上下文（如 Q2 的客单价需要 products.category）；
      * 上下文内所有表的字段全量列出，否则 LLM 不知道 users.city、
        products.category 这类列存在，只能瞎编；
      * computedFrom 链上的 BusinessTerm 也回灌 seed，继续展开。
    """
    fields, tables, calibers = set(), set(), set()
    joins = {}  # tbl -> [(other_tbl, on_cond)]（已去重）

    def tq(script):
        return gq(store, script)

    # 1) 种子 -> 字段 / 口径 / 指标链 / 关联术语
    for mid in list(seed_ids):
        for r in tq(f"g.V('{mid}').outE('pocTermField','pocMetricField',"
                    f"'pocComputedFromMetric','pocComputedFromTerm','pocHasCaliber')"
                    f".inV().project('id','label','props').by(id).by(label).by(valueMap())"):
            lid, lbl = r["id"], r["label"]
            if lbl == "PoCField":
                fields.add(lid)
            elif lbl == "Caliber":
                calibers.add(lid)
            elif lbl == "PoCMetric":  # computedFrom 链：再拉它的字段和口径
                for r2 in tq(f"g.V('{lid}').outE('pocMetricField').inV()"
                             f".project('id','label','props').by(id).by(label).by(valueMap())"):
                    if r2["label"] == "PoCField":
                        fields.add(r2["id"])
                for r2 in tq(f"g.V('{lid}').outE('pocHasCaliber').inV()"
                             f".project('id','label','props').by(id).by(label).by(valueMap())"):
                    if r2["label"] == "Caliber":
                        calibers.add(r2["id"])
            elif lbl == "BusinessTerm":
                seed_ids.add(lid)  # 继续展开该术语的 termField

    # 2) 字段 -> 所属表 与 关联术语（反向）
    for fid in list(fields):
        for r in tq(f"g.V('{fid}').inE('pocHasColumn').outV()"
                    f".project('id','label','props').by(id).by(label).by(valueMap())"):
            if r["label"] == "PoCTable":
                tables.add(r["id"])
        for r in tq(f"g.V('{fid}').inE('pocTermField','pocMetricField').outV()"
                    f".project('id','label','props').by(id).by(label).by(valueMap())"):
            if r["label"] in ("BusinessTerm", "PoCMetric"):
                seed_ids.add(r["id"])

    # 3) 表 -> join 两跳闭包（去重）
    seen_joins = set()
    frontier = set(tables)
    for _ in range(2):
        nxt = set()
        for tid in frontier:
            for r in tq(f"g.V('{tid}').bothE('pocJoinPath')"
                        f".project('on','other').by('on').by(otherV().id())"):
                other, on = r.get("other"), r.get("on")
                key = (tid, other, on)
                if key not in seen_joins:
                    seen_joins.add(key)
                    joins.setdefault(tid, []).append((other, on))
                if other and other not in tables:
                    tables.add(other)
                    nxt.add(other)
        frontier = nxt

    # 4) 上下文内所有表的字段全量列出
    for tid in list(tables):
        for r in tq(f"g.V('{tid}').outE('pocHasColumn').inV()"
                    f".project('id','label','props').by(id).by(label).by(valueMap())"):
            if r["label"] == "PoCField":
                fields.add(r["id"])

    # 5) 口径详情
    caliber_info = {}
    for cid in calibers:
        recs = tq(f"g.V('{cid}').project('id','label','props')"
                  f".by(id).by(label).by(valueMap())")
        if recs:
            caliber_info[cid] = unwrap(recs[0].get("props", {}))

    # 命中术语详情（直接由 matched_terms 带回，无需单独 term_info）

    return {
        "matched_terms": {k: cache_global[k]["props"] for k in seed_ids if k in cache_global},
        "fields": sorted(fields),
        "tables": sorted(tables),
        "joins": {k: v for k, v in joins.items()},
        "calibers": caliber_info,
    }


# 沿图传播只走「语义边」：术语↔字段/指标/口径/同义词、表↔字段。
# 不走 pocJoinPath（表间 join 关系与纠错语义无关，避免把纠错扩散到无关表）。
SEMANTIC_EDGES = ("pocTermField", "pocMetricField", "pocComputedFromMetric",
                  "pocComputedFromTerm", "pocHasCaliber", "pocSynonym", "pocHasColumn")


def propagate_seeds(store, seed_ids: set, hops: int = 2) -> set:
    """从命中种子出发，沿语义边做多跳传播，返回可达节点集合。

    目的：纠错挂在 term_gmv 上，用户换种说法问「GMV 英文全称」/「成交总额」，
    或经指标链（term_aov --pocComputedFromMetric--> metric_gmv）的提问，也能
    通过同义词/指标链邻居召回挂在 term_gmv_en / metric_aov 等节点上的纠错。

    坑：对「不存在的边标签」做 bothE 会抛 Undefined edge label（换图/空本体时
    SEMANTIC_EDGES 里的边未必建过）。先查 schema 现有边标签，只对存在的传播。
    """
    import requests as _rq
    r = _rq.get(f"{HG_URL}/graphs/{HG_GRAPH}/schema/edgelabels", timeout=15)
    existing = {e["name"] for e in r.json().get("edgelabels", [])}
    labels = [e for e in SEMANTIC_EDGES if e in existing]
    if not labels:
        return set(seed_ids)
    labels_arg = ",".join(f"'{e}'" for e in labels)
    reached = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(hops):
        nxt = set()
        for nid in frontier:
            # bothE(...).bothV() 会把起点也带回，reached 去重即可
            for rec in gq(store, f"g.V('{nid}').bothE({labels_arg}).bothV()"
                                 f".project('id').by(id)"):
                oid = rec.get("id")
                if oid and oid not in reached:
                    reached.add(oid)
                    nxt.add(oid)
        frontier = nxt
        if not frontier:
            break
    return reached


def fetch_corrections(store, seed_ids: set) -> tuple:
    """L3 前置：沿语义边传播后，对所有可达节点召回历史纠错（provenance 复用）。

    返回 (纠错列表, 传播统计 dict)；传播统计含 seed / propagated / reached，
    用于演示「沿图传播」相对「仅种子召回」多召回了哪些节点的纠错。

    坑：Gremlin 对不存在的**边标签**做 inE 同样会抛 Undefined edge label
    （首轮运行尚未记录任何纠错时，pocAppliesTo* 标签还不存在）。因此先查
    现有边标签，只对存在的标签做 inE。
    """
    import requests as _rq
    r = _rq.get(f"{HG_URL}/graphs/{HG_GRAPH}/schema/edgelabels", timeout=15)
    existing_edges = {e["name"] for e in r.json().get("edgelabels", [])}
    apply_labels = [l for l in ("pocAppliesToTerm", "pocAppliesToCaliber",
                                "pocAppliesToField") if l in existing_edges]
    if not apply_labels:
        return [], {"seed": sorted(seed_ids), "propagated": [], "reached": sorted(seed_ids)}
    labels_arg = ",".join(f"'{l}'" for l in apply_labels)

    reached = propagate_seeds(store, seed_ids)
    out, seen = [], set()
    for nid in reached:
        recs = gq(store, f"g.V('{nid}').inE({labels_arg}).outV()"
                        f".hasLabel('CorrectionDecision')"
                        f".project('id','label','props').by(id).by(label).by(valueMap())")
        for r_ in recs:
            p = unwrap(r_.get("props", {}))
            # 一条纠错可能挂多个端点（term/caliber/field），传播后多端点都会
            # inE 到同一条 —— 按纠错 id 去重，避免 prompt 重复注入。
            if p.get("id") and p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
    stats = {
        "seed": sorted(seed_ids),
        "propagated": sorted(reached - seed_ids),
        "reached": sorted(reached),
    }
    return out, stats


def retrieve(store, question: str, cache: dict) -> dict:
    hits = lexical_match(cache, question)
    seed = set(h for h, _ in hits[:6])
    ctx = expand_context(store, seed)
    ctx["lexical_hits"] = hits[:6]
    ctx["corrections"], ctx["correction_stats"] = fetch_corrections(store, seed)
    return ctx


# --------------------------------------------------------------------------- #
# L2 提示组装 + MiMo 生成
# --------------------------------------------------------------------------- #
def assemble_prompt(question: str, ctx: dict) -> str:
    L = []
    L.append("你是一个数仓 Text2SQL 助手。只依据下面「语义层召回」的表/字段/口径生成 SQL。")
    L.append(f"问题：{question}")
    L.append("")
    L.append("【召回的业务术语/指标】")
    for tid, p in ctx.get("matched_terms", {}).items():
        L.append(f"  - {p.get('name','')}（{tid}）：{p.get('definition', p.get('formula',''))}")
    L.append("")
    L.append("【涉及表】")
    for t in ctx.get("tables", []):
        name = cache_global.get(t, {}).get("props", {}).get("name", t)
        L.append(f"  - {name}（{t}）")
    L.append("")
    L.append("【涉及字段】")
    for f in ctx.get("fields", []):
        p = cache_global.get(f, {}).get("props", {})
        L.append(f"  - {p.get('table','')}.{p.get('name','')}（{p.get('data_type','')}）：{p.get('comment','')}")
    L.append("")
    L.append("【JOIN 路径】")
    for t, edges in ctx.get("joins", {}).items():
        tname = cache_global.get(t, {}).get("props", {}).get("name", t)
        for other, on in edges:
            oname = cache_global.get(other, {}).get("props", {}).get("name", other)
            L.append(f"  - {tname} JOIN {oname} ON {on}")
    L.append("")
    if ctx.get("calibers"):
        L.append("【口径约束（必须严格遵守）】")
        for cid, p in ctx["calibers"].items():
            L.append(f"  - {p.get('name','')}：{p.get('description','')}")
    if ctx.get("corrections"):
        L.append("")
        L.append("【历史纠错（重要，避免重犯）】")
        for c in ctx["corrections"]:
            L.append(f"  - 曾被纠正：错误SQL={c.get('wrong_sql','')}")
            L.append(f"    正确SQL={c.get('correct_sql','')}")
            L.append(f"    原因={c.get('correction_reason','')}")
    L.append("")
    L.append("要求：仅使用上述字段与 JOIN；输出 SQL 本身，不要解释、不要 markdown 代码块。")
    return "\n".join(L)


def call_mimo(prompt: str) -> str:
    if not MIMO_KEY:
        log("L2: 未设置 OPENAI_CHAT_API_KEY，跳过真实 MiMo 调用（返回占位 SQL）")
        return "-- [占位] 未配置 MiMo key；请注入 OPENAI_CHAT_API_KEY 后重跑以生成真实 SQL"
    import requests
    try:
        r = requests.post(
            f"{MIMO_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {MIMO_KEY}", "Content-Type": "application/json"},
            json={"model": MIMO_MODEL, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        log(f"L2: MiMo 调用失败 {type(e).__name__} {str(e)[:160]}")
        return f"-- [MiMo error] {type(e).__name__}"


# --------------------------------------------------------------------------- #
# L3 纠错决策记录（带 provenance）
# --------------------------------------------------------------------------- #
def _applies_edge_label(nid: str) -> str:
    """按被挂端点标签选择 appliesTo 边标签（HG 边标签端点对固定，不可混用）。"""
    info = cache_global.get(nid)
    lbl = info["label"] if info else (
        "BusinessTerm" if nid.startswith("term_") else
        "Caliber" if nid.startswith("cal_") else "PoCField")
    return {"BusinessTerm": "pocAppliesToTerm",
            "Caliber": "pocAppliesToCaliber",
            "PoCField": "pocAppliesToField"}[lbl]


def record_correction(store, question: str, wrong_sql: str, correct_sql: str,
                      reason: str, applies_to: list) -> str:
    cid = f"corr_{uuid.uuid4().hex[:8]}"
    props = {
        "id": cid,
        "question": question,
        "wrong_sql": wrong_sql,
        "correct_sql": correct_sql,
        "correction_reason": reason,
        "decision_maker": "analyst",
        "confidence": 0.95,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    add_node(store, "CorrectionDecision", cid, props)
    for t in applies_to:
        try:
            add_edge(store, cid, t, _applies_edge_label(t), {})
        except Exception as e:  # noqa: BLE001
            log(f"L3: appliesTo 边写入跳过（端点可能不存在）{t}: {type(e).__name__}")
    log(f"L3: 已记录纠错决策 {cid}，provenance 挂到 {applies_to}")
    return cid


# --------------------------------------------------------------------------- #
# 演示编排
# --------------------------------------------------------------------------- #
cache_global: dict = {}  # 模块级，供 expand_context/assemble_prompt 使用


def run_demo(store) -> dict:
    reset_ontology(store)
    build_demo_ontology(store)
    global cache_global
    cache_global = fetch_all_terms(store)
    log(f"L1: 术语缓存 {len(cache_global)} 个节点")

    questions = {
        "Q1": "上个月各城市的GMV是多少？",
        "Q2": "客单价最高的品类有哪些？",
    }
    result = {"questions": {}, "correction": None}

    # ---- Phase A：Q1 首轮（未纠错）----
    q = questions["Q1"]
    ctx = retrieve(store, q, cache_global)
    prompt = assemble_prompt(q, ctx)
    sql = call_mimo(prompt)
    log(f"PhaseA [{q}] SQL={sql[:80]}")
    result["questions"]["Q1_first"] = {
        "question": q, "sql": sql,
        "matched_terms": list(ctx["matched_terms"].keys()),
        "tables": ctx["tables"], "fields": ctx["fields"],
        "joins": {k: [o for o, _ in v] for k, v in ctx["joins"].items()},
        "calibers": list(ctx["calibers"].keys()),
    }

    # ---- Phase B：用户纠错（模拟分析师修正）----
    wrong = sql
    correct = ("SELECT u.city, SUM(o.amount) AS gmv "
               "FROM orders o JOIN users u ON o.user_id = u.id "
               "WHERE o.status = 'paid' AND o.created_at >= date_trunc('month', now()) - interval '1 month' "
               "GROUP BY u.city ORDER BY gmv DESC")
    reason = ("GMV 仅统计 status='paid' 的订单（口径 cal:gmv.paid）；"
              "city 在 users 表，必须 JOIN users；时间用 created_at 过滤上月")
    cid = record_correction(store, q, wrong, correct, reason,
                            applies_to=["term_gmv", "cal_gmv_paid", "fld_users_city"])
    result["correction"] = {"id": cid, "reason": reason, "correct_sql": correct}

    # ---- Phase B2：纠错挂在「非种子节点」上（口径 cal_gmv_paid）。Q1 提问命中
    #       term_gmv/metric_gmv，metric_gmv --pocHasCaliber--> cal_gmv_paid，
    #       cal_gmv_paid 不在词法种子内，仅靠「沿图传播」才能召回这条纠错 ——
    #       验证 L3 升级（同义词/指标链/口径可达也召回）。----
    cid2 = record_correction(
        store, "GMV 口径的英文表述问题",
        "SELECT SUM(orders.amount) FROM orders",
        "SELECT SUM(orders.amount) FROM orders WHERE orders.status = 'paid'",
        "GMV 口径（cal:gmv.paid）仅统计 status='paid' 的订单金额之和；"
        "该口径约束已由分析师确认过，换英文/别名表述提问同样适用",
        applies_to=["cal_gmv_paid"])
    result["correction_propagation"] = {"id": cid2, "applies_to": "cal_gmv_paid"}

    # ---- Phase C：同问题再问（应召回纠错 -> 生成正确 SQL）----
    ctx2 = retrieve(store, q, cache_global)
    st = ctx2["correction_stats"]
    log(f"PhaseC: 召回历史纠错 {len(ctx2['corrections'])} 条 "
        f"(seed={len(st['seed'])} propagated={len(st['propagated'])} reached={len(st['reached'])})")
    prop_ids = [c.get("id") for c in ctx2["corrections"]]
    log(f"PhaseC: 传播额外召回 {cid2}: {cid2 in prop_ids}  "
        f"(挂在非种子节点 cal_gmv_paid, 不在 seed={st['seed']})")
    prompt2 = assemble_prompt(q, ctx2)
    sql2 = call_mimo(prompt2)
    log(f"PhaseC [{q}] SQL={sql2[:80]}")
    result["questions"]["Q1_after_correction"] = {
        "question": q, "sql": sql2,
        "retrieved_corrections": len(ctx2["corrections"]),
        "correction_stats": st,
    }

    # ---- Q2 顺带跑一遍（验证另一路径的检索）----
    ctx3 = retrieve(store, questions["Q2"], cache_global)
    sql3 = call_mimo(assemble_prompt(questions["Q2"], ctx3))
    log(f"Q2 [{questions['Q2']}] SQL={sql3[:80]}")
    result["questions"]["Q2"] = {
        "question": questions["Q2"], "sql": sql3,
        "matched_terms": list(ctx3["matched_terms"].keys()),
        "tables": ctx3["tables"], "fields": ctx3["fields"],
        "joins": {k: [o for o, _ in v] for k, v in ctx3["joins"].items()},
        "calibers": list(ctx3["calibers"].keys()),
    }

    return result


def main() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log("=== 场景一 PoC：semantica 语义中间层 (L1检索 / L2生成 / L3纠错) ===")
    log(f"target: {HG_URL} graph={HG_GRAPH}  mimo_model={MIMO_MODEL} "
        f"key_set={bool(MIMO_KEY)}")

    from semantica.graph_store import GraphStore
    store = GraphStore(backend="hugegraph", host="127.0.0.1", port=8081, graph=HG_GRAPH)
    store.connect()
    log("connected to HugeGraph OK")

    result = run_demo(store)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"wrote {OUT_JSON}")
    log("DONE")


if __name__ == "__main__":
    main()
