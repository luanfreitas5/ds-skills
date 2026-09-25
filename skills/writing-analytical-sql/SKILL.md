---
name: writing-analytical-sql
description: Use when writing SQL for analysis, reporting, or feature extraction (Postgres, DuckDB, BigQuery, Snowflake, Spark SQL), when joining one-to-many tables before aggregating, when sums or counts look inflated, when using NOT IN, window functions, or "latest row per group".
---

# Writing Analytical SQL

## Overview

Analytical SQL fails silently: a join that multiplies rows still returns a plausible number. **Core principle:** know the grain (what one row means) of every CTE, aggregate each child table to the target grain **before** joining, and assert the grain after every join.

## Rules

1. **One CTE, one responsibility**, named after its grain: `orders_per_customer`, `payments_per_order`.
2. **Aggregate children before joining.** `orders ⋈ order_items ⋈ payments` then `SUM` counts each payment once per item (fan-out). Collapse `order_items` and `payments` to one row per `order_id` first.
3. **Assert grain** before and after each join (query below). A join to a dimension must not change the row count.
4. **Count the entity, not the rows:** `COUNT(DISTINCT order_id)` when rows can repeat; `COUNT(col)` skips NULLs, `COUNT(*)` does not.
5. **Anti-join with `NOT EXISTS`**, never `NOT IN (subquery)`: one NULL in the subquery makes `x NOT IN (...)` return NULL for every row → zero rows.
6. **Windows:** `PARTITION BY` = the entity; `ORDER BY` needs a deterministic tiebreaker (`order_date DESC, order_id DESC`). Rolling frames use `ROWS BETWEEN n PRECEDING AND CURRENT ROW`. Latest row per group: `QUALIFY ROW_NUMBER() OVER (...) = 1` (DuckDB, BigQuery, Snowflake); Postgres has no `QUALIFY` — filter `rn = 1` in an outer query.
7. **NULL semantics explicit:** `COALESCE` only where zero is the true value (no orders → 0 revenue); NULL categories handled on purpose, not dropped by an inner join.
8. **Features for a model:** every event filtered to `< cutoff_date` of the label row (REQUIRED SUB-SKILL: building-point-in-time-features). "All history until today" leaks the future.

## Implementation

```sql
WITH items_per_order AS (          -- grain: 1 linha por order_id
    SELECT order_id, SUM(quantity * unit_price) AS revenue
    FROM order_items GROUP BY order_id
),
payments_per_order AS (            -- grain: 1 linha por order_id
    SELECT order_id, SUM(amount) AS paid
    FROM payments GROUP BY order_id
),
orders_enriched AS (               -- grain: 1 linha por order_id
    SELECT o.order_id, o.customer_id, o.order_date,
           COALESCE(i.revenue, 0) AS revenue, COALESCE(p.paid, 0) AS paid
    FROM orders o
    LEFT JOIN items_per_order i USING (order_id)
    LEFT JOIN payments_per_order p USING (order_id)
),
last_3_orders AS (                 -- grain: 1 linha por customer_id
    SELECT customer_id, AVG(revenue) AS avg_revenue_last_3
    FROM (
        SELECT customer_id, revenue,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC, order_id DESC) AS rn
        FROM orders_enriched
    ) t
    WHERE rn <= 3
    GROUP BY customer_id
)
SELECT c.customer_id,
       COALESCE(SUM(e.revenue), 0)     AS revenue_total,
       COALESCE(SUM(e.paid), 0)        AS paid_total,
       COUNT(DISTINCT e.order_id)      AS n_orders,
       MAX(e.order_date)               AS last_order_date,
       l.avg_revenue_last_3,
       NOT EXISTS (                    -- anti-join seguro com NULL
           SELECT 1 FROM orders o2
           JOIN order_items oi USING (order_id)
           JOIN products pr USING (product_id)
           WHERE o2.customer_id = c.customer_id AND pr.category = 'eletronicos'
       ) AS never_bought_electronics
FROM customers c
LEFT JOIN orders_enriched e USING (customer_id)
LEFT JOIN last_3_orders l USING (customer_id)
GROUP BY c.customer_id, l.avg_revenue_last_3;
```

Grain assertion — must return zero rows (run after each CTE in development, and as a test in CI):

```sql
SELECT customer_id, COUNT(*) FROM features GROUP BY customer_id HAVING COUNT(*) > 1;
-- e: (SELECT COUNT(*) FROM features) = (SELECT COUNT(*) FROM customers)
```

Automate these checks on the result table with a data contract (REQUIRED SUB-SKILL: validating-data-contracts): `customer_id` unique, `revenue_total >= 0`, `paid_total` within tolerance of `revenue_total`.

## Common Mistakes

| Mistake | Symptom | Fix |
|---|---|---|
| Join items and payments, then `SUM` | Revenue × number of payments | Aggregate each to `order_id` first |
| `COUNT(*)` after a 1:N join | Order count = item count | `COUNT(DISTINCT order_id)` |
| `WHERE x NOT IN (SELECT y ...)` with NULL `y` | Zero rows returned | `NOT EXISTS` |
| `ROW_NUMBER() OVER (ORDER BY date)` without `PARTITION BY` | "Last 3" across all customers | `PARTITION BY customer_id` |
| Ties in window `ORDER BY` | Different result per run | Add unique tiebreaker column |
| `INNER JOIN` to dimension with NULL keys | Rows vanish silently | `LEFT JOIN` + count check |
| `QUALIFY` in Postgres | Syntax error | Subquery with `rn = 1` |
