### 🎯 量化选股结果 (策略: {{ strategy }})

> **计算参数**
> Window={{ window }},
> TopK={{ top_k }},
> LiquidityFilter={{ liquidity_filter }},
> Mode={{ threshold_mode }},
> Universe={{ universe_size if universe_size else 'ALL' }},
> ScoreScale=percentile_0_100

| 排名 | 代码 | 得分(0~100) | 推荐理由 |
| :--- | :--- | :---: | :--- |
| 1 | {{ symbol }} | {{ score }} | {{ reason }} |
| 2 | ... | ... | ... |

*注：得分越高代表该策略信号越强；raw 指标与更多细节见 extra 字段。*