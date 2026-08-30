INSERT OVERWRITE TABLE dw.ads_daily_sales
SELECT
  o.store_id,
  SUM(o.gmv) AS gmv,
  AVG(o.gmv) AS avg_order_value,
  COUNT(DISTINCT o.order_id) AS order_cnt
FROM dw.orders o
LEFT JOIN dw.stores s ON o.store_id = s.store_id
WHERE o.pay_time >= '2026-01-01'
GROUP BY o.store_id;

INSERT OVERWRITE TABLE dw.ads_user_profile
SELECT u.user_id, COUNT(DISTINCT o.order_id) AS order_cnt
FROM dw.users u
JOIN dw.orders o ON u.user_id = o.user_id
GROUP BY u.user_id;

CREATE TABLE IF NOT EXISTS dw.dim_payment_summary AS
SELECT p.payment_id, p.order_id, p.pay_amount
FROM dw.payments p
JOIN dw.orders o ON p.order_id = o.order_id;

-- pure analytical query: no write target, still yields join key + co-occurrence
SELECT o.order_id, p.pay_amount
FROM dw.orders o
LEFT JOIN dw.payments p ON o.order_id = p.order_id
WHERE p.pay_amount > 100;

-- with a CTE; the CTE name must NOT be treated as a physical table
WITH pay_agg AS (
  SELECT order_id, SUM(pay_amount) AS total
  FROM dw.payments
  GROUP BY order_id
)
INSERT OVERWRITE TABLE dw.ads_order_payment
SELECT o.order_id, a.total
FROM dw.orders o
LEFT JOIN pay_agg a ON o.order_id = a.order_id;
