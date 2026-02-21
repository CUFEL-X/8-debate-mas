# 数据契约（theme_miner）

## 依赖表

| 表名 | 必需列 | 可选列 | 说明 |
| :--- | :--- | :--- | :--- |
| govcn | title, content | date/pub_date/time, industry_name, context | 政策文本与行业字段（industry_frequency 需要 industry_name） |
| etf_basic | code, cname | symbol/ts_code/masterfundcode, csname/name/extname | ETF 基础信息，用于名称模糊匹配召回 |

## 依赖配置（references）
- `references/ontology.yaml`：供 `ontology_mapping` 使用（aliases/expands_to/weight）
- `references/mappings.yaml`：
  - `INDUSTRY_FUZZY_MAP`：供 `industry_frequency` 使用（行业 -> ETF 名称检索词）
  - `THEME_KEYWORDS_MAP`：供 `guardrail_pool` 使用（Bucket -> ETF 名称检索词）
  - `GUARDRAIL_BUCKETS`: 默认的兜底资产桶列表