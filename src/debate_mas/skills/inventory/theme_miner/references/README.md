# 主题挖掘机 (Theme Miner)

本技能包是 Hunter 的“雷达”，用于从政策文本和行业数据中挖掘投资线索，并将其映射到具体的 ETF 标的。

## 🧠 核心逻辑 (Architecture)

本代码采用 **策略分发 (Strategy Dispatch)** 模式，支持三种召回路径：

1.  **Ontology Mapping (知识驱动)**:
    - 逻辑: 关键词 -> 知识图谱扩展 -> 政策验证 -> ETF 映射
    - 适用: 捕捉“新质生产力”、“低空经济”等宏观概念。

2.  **Industry Frequency (数据驱动)**:
    - 逻辑: 统计高频出现的行业词 -> 映射到 ETF
    - 适用: 发现新闻里正在热议的板块 (Hot Topics)。

3.  **Guardrail Pool (规则驱动)**:
    - 逻辑: 强制召回 Bond/Gold/Cash 等防守资产
    - 适用: 保证组合的安全性，防止“All-in”高风险资产。

## 👨‍💻 开发者指南 (Student Lab)

我们预留了 `_user_custom_logic` 接口，供练习者实现自定义的召回策略。

**练习目标**: 实现一个“混合加权检索”策略。
- **Step 1**: 分别对 Title 和 Content 列进行搜索。
- **Step 2**: 对 Title 命中的结果给予更高权重 (Weight=2.0)。
- **Step 3**: 合并结果并按总分排序。

请打开 `scripts/handler.py`，找到 `_user_custom_logic` 函数开始你的代码填空！