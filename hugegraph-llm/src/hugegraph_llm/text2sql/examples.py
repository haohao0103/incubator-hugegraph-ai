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

"""A minimal, self-contained order-domain semantic model for PoC + tests."""

from hugegraph_llm.text2sql.model import (
    Column,
    Filter,
    Join,
    Metric,
    QueryPattern,
    SemanticModel,
    Table,
    Term,
    Value,
)


def build_order_domain_model() -> SemanticModel:
    """Return a small order-domain semantic layer.

    The metric 口径 here is illustrative — in production it must come from a
    metric platform / data dictionary and pass human review before entering the
    graph.
    """
    order = Table(
        name="order",
        description="订单主表",
        aliases=["订单", "order"],
        domain="order",
        time_column="pay_time",
        partition_column="dt",
        columns=[
            Column("id", "BIGINT", "订单ID", role="pk"),
            Column("user_id", "BIGINT", "下单用户ID", role="fk"),
            Column("driver_id", "BIGINT", "接单司机ID", role="fk"),
            Column("status", "TINYINT", "订单状态", role="value"),
            Column("amount", "DECIMAL", "订单金额", role="measure"),
            Column("pay_time", "DATETIME", "支付时间", role="time"),
            Column("created_at", "DATETIME", "下单时间", role="time"),
            Column("dt", "STRING", "分区日期", role="partition"),
        ],
    )

    order_detail = Table(
        name="order_detail",
        description="订单明细表",
        aliases=["订单明细", "明细"],
        domain="order",
        columns=[
            Column("order_id", "BIGINT", "订单ID", role="fk"),
            Column("product_id", "BIGINT", "商品ID", role="fk"),
            Column("amount", "DECIMAL", "明细金额", role="measure"),
            Column("quantity", "INT", "数量", role="measure"),
        ],
    )

    user = Table(
        name="user",
        description="用户表",
        aliases=["用户", "顾客"],
        domain="order",
        columns=[
            Column("id", "BIGINT", "用户ID", role="pk"),
            Column("name", "STRING", "用户姓名", role="dimension"),
            Column("level", "STRING", "用户等级", role="dimension"),
        ],
    )

    driver = Table(
        name="driver",
        description="司机表",
        aliases=["司机", "骑手"],
        domain="order",
        columns=[
            Column("id", "BIGINT", "司机ID", role="pk"),
            Column("name", "STRING", "司机姓名", role="dimension"),
            Column("region", "STRING", "所属区域", role="dimension"),
        ],
    )

    model = SemanticModel(
        name="order_domain",
        domain="order",
        tables=[order, order_detail, user, driver],
        joins=[
            Join("order", "user", "order.user_id = user.id", "LEFT JOIN", "none"),
            Join("order_detail", "order", "order_detail.order_id = order.id", "LEFT JOIN", "one_to_many"),
            Join("order", "driver", "order.driver_id = driver.id", "LEFT JOIN", "none"),
        ],
        terms=[
            Term(
                "GMV", aliases=["成交额", "支付金额", "gmv"], column_refs=["order_detail.amount"], metric_refs=["gmv"]
            ),
            # Ambiguous term: one word maps to multiple columns -> needs disambiguation.
            Term("金额", aliases=[], column_refs=["order.amount", "order_detail.amount"]),
            Term("订单量", aliases=["订单数", "单量"], metric_refs=["order_count"]),
            Term("司机", aliases=["骑手", "driver"], column_refs=["driver.name"]),
            Term("用户", aliases=["顾客", "客户"], column_refs=["user.name"]),
        ],
        metrics=[
            Metric(
                name="gmv",
                description="成交总额（已支付+已完成订单的明细金额求和）",
                measure="order_detail.amount",
                agg_func="SUM",
                time_column="order.pay_time",
                time_granularity="month",
                filters=[Filter(column="order.status", operator="IN", values=["2", "3"])],
                dedup=True,
            ),
            Metric(
                name="order_count",
                description="订单量",
                measure="order.id",
                agg_func="COUNT",
                time_column="order.created_at",
                time_granularity="day",
                filters=[],
                dedup=False,
            ),
        ],
        values=[
            Value("order.status", "1", "待支付"),
            Value("order.status", "2", "已支付"),
            Value("order.status", "3", "已完成"),
            Value("order.status", "4", "已取消"),
        ],
        query_patterns=[
            QueryPattern(
                question="上个月 GMV 是多少",
                sql="SELECT SUM(od.amount) AS gmv FROM order_detail od "
                "JOIN `order` o ON od.order_id = o.id "
                "WHERE o.status IN (2, 3) AND o.pay_time >= '2024-01-01' AND o.pay_time < '2024-02-01'",
                tables=["order", "order_detail"],
                metrics=["gmv"],
            ),
            QueryPattern(
                question="每个司机的接单量",
                sql="SELECT d.name AS driver, COUNT(o.id) AS order_count FROM driver d "
                "LEFT JOIN `order` o ON d.id = o.driver_id GROUP BY d.name",
                tables=["driver", "order"],
                metrics=["order_count"],
            ),
        ],
    )
    return model
