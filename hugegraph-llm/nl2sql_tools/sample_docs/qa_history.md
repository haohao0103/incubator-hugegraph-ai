# 历史问答记录（示例，含错误 SQL 纠错）

本文档收录线上 BI 问答中产生过错误 SQL 的历史记录，供语义层抽取纠错样本。
每条记录包含用户问题、曾经给出的错误 SQL、修正后的正确 SQL 与纠错原因。

## 1. 销售额查询用了不存在的列

- 用户问题：这个月销售额是多少？
- 错误 SQL：
  SELECT SUM(amount) FROM orders WHERE month = '2026-08'
- 正确 SQL：
  SELECT SUM(pay_amount) AS sales_amount
  FROM payments
  WHERE pay_status = 'paid'
    AND pay_time >= '2026-08-01'
- 纠错原因：orders 表没有 amount 列（幻觉列名）；销售额应取 payments.pay_amount，且需过滤支付成功状态。
- 纠错对象：field:payments.pay_amount

## 2. 成交额漏加已支付过滤

- 用户问题：上周各渠道成交额分别是多少？
- 错误 SQL：
  SELECT channel, SUM(gmv) FROM orders GROUP BY channel
- 正确 SQL：
  SELECT o.channel, SUM(o.gmv) AS gmv
  FROM orders o
  WHERE o.status = 'paid'
    AND o.pay_time >= '2026-08-24'
  GROUP BY o.channel
- 纠错原因：GMV 口径要求仅统计 paid 状态订单，漏过滤会把取消/退款订单计入，金额虚高。
- 纠错对象：term:成交额

## 3. 客单价分子分母口径错误

- 用户问题：北京门店的平均客单价是多少？
- 错误 SQL：
  SELECT AVG(avg_order_value) FROM ads_daily_sales
- 正确 SQL：
  SELECT SUM(gmv) / SUM(order_cnt) AS avg_order_value
  FROM ads_daily_sales
  WHERE store_id IN (SELECT store_id FROM stores WHERE city = '北京')
- 纠错原因：客单价 = 成交总额 / 订单数（聚合粒度上的除法），直接 AVG 汇总表字段会得到错误的"日均客单价平均"。
- 纠错对象：term:客单价

## 4. 履约时长混入未签收订单

- 用户问题：平均履约时长是多少小时？
- 错误 SQL：
  SELECT AVG(fulfillment_hours) FROM logistics
- 正确 SQL：
  SELECT AVG(fulfillment_hours)
  FROM logistics
  WHERE status = 'received'
- 纠错原因：履约时效口径只统计已签收订单；未签收订单的 fulfillment_hours 为空或无效值。
- 纠错对象：term:履约时长

## 5. 用户复购率 join 少了用户维度

- 用户问题：各城市用户的复购率是多少？
- 错误 SQL：
  SELECT city, repurchase_rate FROM ads_user_profile
- 正确 SQL：
  SELECT u.city, p.repurchase_rate
  FROM ads_user_profile p
  JOIN users u ON p.user_id = u.user_id
- 纠错原因：复购率按用户维度统计，需关联 users 表取城市；缺 join 会把用户画像表当成用户维度表。
- 纠错对象：field:users.city
