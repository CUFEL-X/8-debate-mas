# 数据契约 (Data Contract)

本技能依赖表名：`etf_daily`

| 列名 (Standard) | 可接受原始列名 | 类型 | 说明 |
| :--- | :--- | :--- | :--- |
| code | code | str | ETF 代码（如 510300） |
| date | date / data | date/datetime | 交易日期（用于 `apply_date_filter`） |
| close | close | float | 收盘价（核心计算字段） |
| amount | amount | float | 成交额（用于流动性过滤；缺失时自动跳过过滤） |

## 说明
- `amount_latest`：只使用最新交易日的 `amount` 与 `min_amount` 对比。
- `amihud`：
  - 使用 `amount_yuan = amount * amount_scale`（amount_scale 默认 1000）
  - 对每只 ETF 取最近 `window` 天，计算 illiq，保留 illiq 较小的 `illiq_quantile` 分位以内标的。