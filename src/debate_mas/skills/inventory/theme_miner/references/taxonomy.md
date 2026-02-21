# 配置指南（Operator Manual）

本技能遵循“配置与代码分离”：业务人员改 YAML，即可调整召回覆盖面，无需改代码。

---

## 1） ontology.yaml（主观先验：主题词典）
供 `ontology_mapping` 使用。

你主要改三块：

### (1) aliases：解决“用户搜不到”
用户输入这些词，都能命中同一个主题。

### (2) expands_to：解决“召回不全”
主题会扩展成这些检索词，用于在 ETF 名称里模糊匹配召回。

### (3) weight：静态重要性（解释用）
当前 handler 会把 weight 写入 reason/extra 用于解释；它不是收益预测模型。

---

## 2） mappings.yaml（行业/主题 -> 名称检索词）
### (1) INDUSTRY_FUZZY_MAP（industry_frequency 主用）
用于把 `govcn.industry_name` 翻译成 ETF 名称里更容易命中的 token。

注意：
- key 不需要 100% 精确匹配：handler 会先做清洗，再做包含式兜底
- value 是“可能出现在 ETF 名称里的词”，越贴近真实基金命名越好

### (2) THEME_KEYWORDS_MAP & GUARDRAIL_BUCKETS（guardrail_pool 主用）
用于 Hunter 的结构兜底召回。
- **THEME_KEYWORDS_MAP**: 定义每个桶（如 Bond）包含哪些关键词（如 国债, 信用债）。
- **GUARDRAIL_BUCKETS**: 定义默认要检查哪些桶。

---

## 3） 一个推荐的工作流
1) 发现新叙事：先在 ontology.yaml 加一个 concepts 条目（aliases + expands_to）
2) 发现 industry_frequency 命中行业但召回少：在 mappings.yaml 的 INDUSTRY_FUZZY_MAP 补充 token
3) 最终排序：把 theme_miner 输出候选作为 universe 交给 quantitative_sniper