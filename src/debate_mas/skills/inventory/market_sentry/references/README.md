# 盘面哨兵 (Market Sentry)

本技能负责对 ETF 进行量化风控审计（基于日线行情）。

## 审计逻辑

### 1. 流动性 (Liquidity)
- **目标**: 必须保证“买得进、卖得出”，防止陷入流动性陷阱。
- **核心指标**: `Amount_MA20` (最近20个交易日成交额均值)。
- **默认阈值**: `min_amount=2000` (即 200万元，若 CSV amount 单位为千元)。
- **判定**: 若 `MA20 < min_amount` -> **REJECT** (Risk Score +60)。

### 2. 波动率 (Volatility)
采用 **“非对称风控”** 逻辑，避免误杀良性上涨的高波动标的。

- **指标定义**:
  1.  **Total Volatility (总体波动)**: $\sigma = Std(R)$
  2.  **Downside Volatility (下行波动)**: $\sigma_{down} = Std(R \mid R < 0)$

- **判定规则**:
  - **Case A (低波)**: 若 $\sigma \le Threshold$ -> **PASS**。
  - **Case B (良性高波)**: 若 $\sigma > Threshold$ 但 **下跌天数占比 < 15%** -> **PASS** (视为单边上涨)。
  - **Case C (恶性高波)**: 若 $\sigma > Threshold$ 且 $\sigma_{down} > Threshold$ -> **REJECT** (Risk Score +40)。
  - **Case D (暴跌)**: 若单日跌幅超过 $2 \times Threshold$ -> **REJECT**。