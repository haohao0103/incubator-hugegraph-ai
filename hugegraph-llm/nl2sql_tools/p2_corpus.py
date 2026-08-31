"""Larger-scale synthetic warehouse schema + labeled question set.

Goal: stress-test P2 (semantic vector recall) vs P0 (lexical) schema linking on
a realistic multi-table catalog where many questions require *semantic* (not
surface) matching.

Everything is deterministic and offline. Build with ``build_warehouse_schema()``;
the labeled questions live in ``WAREHOUSE_QUESTIONS``.

Gold answers are column full-names (``table.column``). A question's gold is what
a correct schema linker must surface. Lexical controls share characters with the
gold comment/name (P0 should win); semantic-only questions share *no* character
with the gold text so P0 produces no seed and P2 must carry the recall.
"""
from hugegraph_llm.nl2sql.schema_graph.model import (
    Column, Edge, EdgeType, NodeType, SchemaGraph, Table, Term,
)

DB = "dw"


def _col_id(table, col):
    return f"column:{DB}.{table}.{col}"


def _table_id(table):
    return f"table:{DB}.{table}"


def _term_id(name):
    return f"term:{name}"


# ---------------------------------------------------------------------------
# Raw metadata
# ---------------------------------------------------------------------------
TABLES = [
    ("orders", "订单表", True),
    ("payments", "支付表", True),
    ("users", "用户表", False),
    ("products", "商品表", False),
    ("order_items", "订单明细表", False),
    ("refunds", "退款表", True),
    ("coupons", "优惠券表", False),
    ("stores", "门店表", False),
    ("regions", "地区表", False),
    ("logistics", "物流表", True),
    ("ads_daily_sales", "每日销售汇总", False),
    ("ads_user_profile", "用户画像汇总", False),
]

# (table, column, data_type, comment, is_pk, is_fk_ref_table_or_None)
COLUMNS = [
    ("orders", "order_id", "bigint", "订单编号", True, None),
    ("orders", "user_id", "bigint", "用户编号", False, "users"),
    ("orders", "store_id", "bigint", "门店编号", False, "stores"),
    ("orders", "gmv", "decimal", "成交总额", False, None),
    ("orders", "order_status", "string", "订单状态", False, None),
    ("orders", "create_time", "timestamp", "下单时间", False, None),
    ("orders", "pay_time", "timestamp", "支付时间", False, None),
    ("orders", "channel", "string", "下单渠道", False, None),
    ("orders", "item_count", "int", "商品件数", False, None),
    ("orders", "discount_amount", "decimal", "优惠金额", False, None),

    ("payments", "payment_id", "bigint", "支付流水号", True, None),
    ("payments", "order_id", "bigint", "订单编号", False, "orders"),
    ("payments", "pay_amount", "decimal", "支付金额", False, None),
    ("payments", "settlement_amount", "decimal", "店铺结算流水金额", False, None),
    ("payments", "pay_method", "string", "支付方式", False, None),
    ("payments", "refund_amount", "decimal", "退款金额", False, None),
    ("payments", "currency", "string", "币种", False, None),

    ("users", "user_id", "bigint", "用户编号", True, None),
    ("users", "city", "string", "所在城市", False, None),
    ("users", "reg_time", "timestamp", "注册时间", False, None),
    ("users", "gender", "string", "性别", False, None),
    ("users", "age", "int", "年龄", False, None),
    ("users", "vip_level", "int", "会员等级", False, None),

    ("products", "product_id", "bigint", "商品编号", True, None),
    ("products", "product_name", "string", "商品名称", False, None),
    ("products", "category", "string", "商品类目", False, None),
    ("products", "price", "decimal", "售价", False, None),
    ("products", "cost", "decimal", "成本价", False, None),
    ("products", "brand", "string", "品牌", False, None),

    ("order_items", "item_id", "bigint", "明细编号", True, None),
    ("order_items", "order_id", "bigint", "订单编号", False, "orders"),
    ("order_items", "product_id", "bigint", "商品编号", False, "products"),
    ("order_items", "quantity", "int", "购买数量", False, None),
    ("order_items", "subtotal", "decimal", "小计金额", False, None),
    ("order_items", "unit_price", "decimal", "单价", False, None),

    ("refunds", "refund_id", "bigint", "退款编号", True, None),
    ("refunds", "order_id", "bigint", "订单编号", False, "orders"),
    ("refunds", "refund_amount", "decimal", "退款金额", False, None),
    ("refunds", "refund_reason", "string", "退款原因", False, None),

    ("coupons", "coupon_id", "bigint", "券编号", True, None),
    ("coupons", "coupon_name", "string", "券名称", False, None),
    ("coupons", "discount_value", "decimal", "抵扣面额", False, None),
    ("coupons", "threshold", "decimal", "使用门槛", False, None),
    ("coupons", "expire_time", "timestamp", "过期时间", False, None),

    ("stores", "store_id", "bigint", "门店编号", True, None),
    ("stores", "store_name", "string", "门店名称", False, None),
    ("stores", "city", "string", "所在城市", False, None),
    ("stores", "region", "string", "所属大区", False, None),
    ("stores", "open_date", "timestamp", "开业日期", False, None),

    ("regions", "region_id", "bigint", "大区编号", True, None),
    ("regions", "region_name", "string", "大区名称", False, None),
    ("regions", "province", "string", "省份", False, None),

    ("logistics", "logistics_id", "bigint", "物流编号", True, None),
    ("logistics", "order_id", "bigint", "订单编号", False, "orders"),
    ("logistics", "fulfillment_hours", "decimal", "订单从支付到签收时长", False, None),
    ("logistics", "ship_time", "timestamp", "发货时间", False, None),
    ("logistics", "receive_time", "timestamp", "签收时间", False, None),
    ("logistics", "carrier", "string", "承运商", False, None),

    ("ads_daily_sales", "stat_date", "date", "统计日期", True, None),
    ("ads_daily_sales", "store_id", "bigint", "门店编号", False, "stores"),
    ("ads_daily_sales", "gmv", "decimal", "成交总额", False, None),
    ("ads_daily_sales", "avg_order_value", "decimal", "平均每单成交金额", False, None),
    ("ads_daily_sales", "gross_profit", "decimal", "商品销售收入减成本", False, None),
    ("ads_daily_sales", "sell_through_rate", "decimal", "有销售的商品占比", False, None),
    ("ads_daily_sales", "return_rate", "decimal", "退货订单占比", False, None),
    ("ads_daily_sales", "order_cnt", "int", "订单数", False, None),
    ("ads_daily_sales", "user_cnt", "int", "下单用户数", False, None),

    ("ads_user_profile", "stat_date", "date", "统计日期", True, None),
    ("ads_user_profile", "user_id", "bigint", "用户编号", False, "users"),
    ("ads_user_profile", "repurchase_rate", "decimal", "重复购买用户占比", False, None),
    ("ads_user_profile", "acquisition_channel", "string", "用户首次来源渠道", False, None),
    ("ads_user_profile", "clv", "decimal", "用户生命周期价值", False, None),
    ("ads_user_profile", "city_tier", "string", "城市等级", False, None),
]

# (term_name, [aliases], definition, binds_to_column(table, col))
TERMS = [
    ("支付总额", [], "支付总额", ("payments", "pay_amount")),
    ("成交额", [], "成交额", ("orders", "gmv")),
    ("营收", [], "营收", ("ads_daily_sales", "gmv")),
    ("客单价", [], "客单价", ("ads_daily_sales", "avg_order_value")),
    ("毛利额", [], "毛利额", ("ads_daily_sales", "gross_profit")),
    ("复购", [], "复购", ("ads_user_profile", "repurchase_rate")),
    ("退货", [], "退货", ("refunds", "refund_amount")),
    ("履约", [], "履约", ("logistics", "fulfillment_hours")),
    ("动销", [], "动销", ("ads_daily_sales", "sell_through_rate")),
    ("获客", [], "获客", ("ads_user_profile", "acquisition_channel")),
    ("结算", [], "结算", ("payments", "settlement_amount")),
]

# (term_name, caliber_name, dimension, description)
# 口径 = 业务指标的统一计算约束（语义层核心价值）。每个 term 可挂 0..n 个口径；
# 口径会随术语进 ingest（Caliber 顶点 + hasCaliber 边）并在 prompt 组装时注入。
CALIBERS = [
    ("成交额", "GMV口径", "status",
     "成交额仅统计订单状态为 paid（已支付）的订单金额之和；退款/取消订单不计入"),
    ("支付总额", "实付口径", "status",
     "支付总额 = 用户实际支付成功的金额合计（pay_amount），不含退款金额"),
    ("客单价", "客单价口径", "grain",
     "客单价 = 成交总额 / 订单数，分母为订单数而非用户数"),
    ("营收", "营收口径", "status",
     "营收 = 已支付订单的成交额合计，与 GMV 口径一致（status='paid'）"),
]


# L3 纠错历史（CorrectionDecision）。applies_to 形如 "term:X"/"field:Y"/"caliber:Z"，
# 挂到语义边上；召回时沿 TERM_MAPS/BELONGS_TO/synonym/caliber 传播，非种子节点上的
# 纠错也能被命中（见 correction_propagation.py）。三条分别覆盖：
#   corr_gmv_status    挂 term 成交额（词法种子直接命中）
#   corr_gmv_caliber   挂 caliber GMV口径（非种子，需沿图传播召回）
#   corr_pay_amount    挂 field payments.pay_amount（column 种子，TERM_MAPS 反查）
CORRECTIONS = [
    {
        "id": "corr_gmv_status",
        "question": "订单的成交总额",
        "wrong_sql": "SELECT SUM(gmv) AS total_gmv FROM orders;",
        "correct_sql": "SELECT SUM(gmv) AS total_gmv FROM orders WHERE order_status = 'paid';",
        "correction_reason": "成交额仅统计 status='paid' 的订单（GMV口径）；未加过滤会把未支付/退款订单计入。",
        "applies_to": ["term:成交额"],
    },
    {
        "id": "corr_gmv_caliber",
        "question": "买卖盘子有多大",
        "wrong_sql": "SELECT SUM(gmv) AS total FROM orders;",
        "correct_sql": "SELECT SUM(gmv) AS total FROM orders WHERE order_status = 'paid';",
        "correction_reason": "GMV口径同样约束挂该口径的汇总指标（ads_daily_sales.gmv），聚合时需按口径过滤。",
        "applies_to": ["caliber:GMV口径"],
    },
    {
        "id": "corr_pay_amount",
        "question": "支付金额是多少",
        "wrong_sql": "SELECT pay_amount FROM payments;",
        "correct_sql": "SELECT SUM(pay_amount) AS total_payment FROM payments;",
        "correction_reason": "问总额时 pay_amount 需 SUM 聚合；实付口径只统计支付成功的金额。",
        "applies_to": ["field:payments.pay_amount"],
    },
]


# (child_col(table,col), parent_col(table,col))
FOREIGN_KEYS = [
    (("orders", "user_id"), ("users", "user_id")),
    (("orders", "store_id"), ("stores", "store_id")),
    (("payments", "order_id"), ("orders", "order_id")),
    (("order_items", "order_id"), ("orders", "order_id")),
    (("order_items", "product_id"), ("products", "product_id")),
    (("refunds", "order_id"), ("orders", "order_id")),
    (("logistics", "order_id"), ("orders", "order_id")),
    (("ads_daily_sales", "store_id"), ("stores", "store_id")),
    (("ads_user_profile", "user_id"), ("users", "user_id")),
]

CO_OCCUR = [
    ("orders", "payments"), ("orders", "order_items"), ("orders", "users"),
    ("orders", "stores"), ("payments", "stores"), ("ads_daily_sales", "stores"),
    ("ads_user_profile", "users"), ("products", "order_items"),
    ("logistics", "orders"), ("refunds", "orders"), ("orders", "coupons"),
]

LINEAGE = [
    ("orders", "ads_daily_sales"), ("payments", "ads_daily_sales"),
    ("order_items", "ads_daily_sales"), ("orders", "ads_user_profile"),
    ("users", "ads_user_profile"),
]


def build_warehouse_schema() -> SchemaGraph:
    g = SchemaGraph()
    for name, comment, is_fact in TABLES:
        g.add_node(Table(name=name, database=DB, comment=comment,
                         is_fact=is_fact).to_node())
    for table, col, dtype, comment, is_pk, fk in COLUMNS:
        g.add_node(Column(name=col, table=f"{DB}.{table}", data_type=dtype,
                          comment=comment, is_primary_key=bool(is_pk),
                          is_foreign_key=fk is not None).to_node())
    for tname, aliases, definition, bind in TERMS:
        # Term has no `definition` attr; the linker reads `definition or comment`,
        # so store the business definition in `comment`.
        cals = [
            {"name": cname, "dimension": dim, "description": desc}
            for t_, cname, dim, desc in CALIBERS
            if t_ == tname
        ]
        g.add_node(Term(name=tname, aliases=list(aliases),
                        comment=definition,
                        properties={"calibers": cals} if cals else {}).to_node())

    def add(edge):
        g.add_edge(edge)

    for table, col, *_ in COLUMNS:
        add(Edge(source=_col_id(table, col), target=_table_id(table),
                 edge_type=EdgeType.BELONGS_TO, weight=1.0))
    for (ct, cc), (pt, pc) in FOREIGN_KEYS:
        add(Edge(source=_col_id(ct, cc), target=_col_id(pt, pc),
                 edge_type=EdgeType.FOREIGN_KEY, weight=1.0))
    for tname, _, _, (bt, bc) in TERMS:
        add(Edge(source=_term_id(tname), target=_col_id(bt, bc),
                 edge_type=EdgeType.TERM_MAPS, weight=1.0))
    for a, b in CO_OCCUR:
        add(Edge(source=_table_id(a), target=_table_id(b),
                 edge_type=EdgeType.CO_OCCUR, weight=1.0))
    for a, b in LINEAGE:
        add(Edge(source=_table_id(a), target=_table_id(b),
                 edge_type=EdgeType.LINEAGE, weight=1.0))
    return g


# ---------------------------------------------------------------------------
# Labeled questions. category: "lexical" (P0 should win) or "semantic"
# (P0 should miss; P2 carries recall). gold = list of column full-names.
# ---------------------------------------------------------------------------
WAREHOUSE_QUESTIONS = [
    # ---- lexical controls (shared characters with gold text) ----
    {"q": "支付金额是多少", "gold": ["payments.pay_amount"], "category": "lexical"},
    {"q": "订单的成交总额", "gold": ["orders.gmv"], "category": "lexical"},
    {"q": "商品成本价", "gold": ["products.cost"], "category": "lexical"},
    {"q": "每个门店的结算金额", "gold": ["payments.settlement_amount"], "category": "lexical"},
    {"q": "退货订单占比", "gold": ["ads_daily_sales.return_rate"], "category": "lexical"},
    {"q": "复购率怎么算", "gold": ["ads_user_profile.repurchase_rate"], "category": "lexical"},
    {"q": "支付方式有哪些", "gold": ["payments.pay_method"], "category": "lexical"},
    {"q": "商品类目分布", "gold": ["products.category"], "category": "lexical"},

    # ---- semantic-only (NO shared character with gold text; P0 empty) ----
    {"q": "毛利", "gold": ["ads_daily_sales.gross_profit"], "category": "semantic"},
    {"q": "买一份花多少", "gold": ["ads_daily_sales.avg_order_value"], "category": "semantic"},
    {"q": "引流途径", "gold": ["ads_user_profile.acquisition_channel"], "category": "semantic"},
    {"q": "送货要几天", "gold": ["logistics.fulfillment_hours"], "category": "semantic"},
    {"q": "买卖盘子有多大", "gold": ["orders.gmv"], "category": "semantic"},
    {"q": "一个人从头到尾贡献", "gold": ["ads_user_profile.clv"], "category": "semantic"},
    {"q": "客户数", "gold": ["users.user_id"], "category": "semantic"},
    {"q": "哪些货卖得动", "gold": ["ads_daily_sales.sell_through_rate"], "category": "semantic"},
    {"q": "到账的钱", "gold": ["payments.settlement_amount"], "category": "semantic"},
    {"q": "用户首次从哪来", "gold": ["ads_user_profile.acquisition_channel"], "category": "semantic"},
    {"q": "从付款到收货多久", "gold": ["logistics.fulfillment_hours"], "category": "semantic"},
    {"q": "老客重复买的比例", "gold": ["ads_user_profile.repurchase_rate"], "category": "semantic"},

    # ---- join-intent (multi-element gold) ----
    {"q": "用户下的订单支付了多少",
     "gold": ["payments.pay_amount", "orders.order_id", "users.user_id"],
     "category": "join"},
    {"q": "每个用户的复购频次",
     "gold": ["ads_user_profile.repurchase_rate", "users.user_id"],
     "category": "join"},
    {"q": "门店销售额和退款的差额",
     "gold": ["ads_daily_sales.gmv", "refunds.refund_amount", "stores.store_id"],
     "category": "join"},
]


def gold_node_ids(question):
    ids = []
    for col in question["gold"]:
        table, _, c = col.rpartition(".")
        ids.append(f"column:{DB}.{table}.{c}")
    return ids


def gold_table_ids(question):
    tables = set()
    for col in question["gold"]:
        table, _, _ = col.rpartition(".")
        tables.add(f"table:{DB}.{table}")
    return list(tables)
