### 📰 主题召回结果 (Mode: {{ mode }})

> **参数**: Keyword={{ keyword }}, LookbackDays={{ days }}, TopK={{ top_k }}

#### 1） 证据摘要（解释用）
- **政策命中**: {{ policy_docs }} 条
- **证据摘要**: {{ evidence_str }}
- **政策强度**: {{ policy_strength }}

#### 2） 召回候选（Recall，不是排序结果）
| 排名 | 代码 | 召回证据分 | 推荐理由 |
| :--- | :--- | :---: | :--- |
| 1 | {{ symbol }} | {{ score }} | {{ reason }} |
| 2 | ... | ... | ... |

*注：score 为“召回证据强弱”的展示分（上限 60）；若要量化排序，请将候选作为 universe 交给 quantitative_sniper。*