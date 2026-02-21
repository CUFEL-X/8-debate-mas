# 决策打分规则

## 1. 线性融合 (Current)
目前采用加权减法模型：
$$Score_{final} = Score_{Hunter} - (Score_{Risk} \times k)$$

- **Score_Hunter**: 来源于 `quantitative_sniper` 或 `theme_miner`，范围 [0, 100]。
- **Score_Risk**: 来源于 `market_sentry` 或 `forensic_detective`，范围 [0, 100]。
- **k (Risk Penalty)**: 风险厌恶系数，默认 1.0。
  - 牛市可设为 0.5 (激进)。
  - 熊市可设为 2.0 (保守)。

## 2. 深度学习融合 (Future)
*预留接口 `method="dl_model"`*
计划训练一个 MLP 或 XGBoost 模型，输入多空双方的原始特征，直接预测胜率。