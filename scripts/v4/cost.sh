#!/usr/bin/env bash
# 実課金の確認（BigQuery の課金エクスポート）。クレジット差し引き前の総額＝無料クレジットの消費量。
# 予算: 無料クレジット ¥47,813（$300）。中止規則: ¥36,000 で新規系統停止、¥42,000 で全停止（docs/research_design_v4.md §7）
set -euo pipefail
P=pro-plasma-510112-m7
T="${P}.billing_export.gcp_billing_export_v1_01FBCA_9D91E5_175028"
bq --project_id=$P query --use_legacy_sql=false --format=pretty "
SELECT service.description AS service,
       ROUND(SUM(cost), 0) AS gross_jpy,
       ROUND(SUM(IFNULL((SELECT SUM(c.amount) FROM UNNEST(credits) c), 0)), 0) AS credits_jpy,
       MIN(usage_start_time) AS first_usage, MAX(usage_end_time) AS last_usage
FROM \`${T}\` WHERE project.id = '${P}'
GROUP BY service ORDER BY gross_jpy DESC"
bq --project_id=$P query --use_legacy_sql=false --format=pretty "
SELECT ROUND(SUM(cost), 0) AS total_gross_jpy,
       ROUND(47813 - SUM(cost), 0) AS remaining_credit_jpy_est,
       ROUND(100 * SUM(cost) / 47813, 1) AS used_pct
FROM \`${T}\` WHERE project.id = '${P}'"
