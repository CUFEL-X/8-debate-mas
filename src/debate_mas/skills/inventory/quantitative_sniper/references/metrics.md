# 指标公式 (Formulas)

本文档定义 algo.py 中使用的关键指标与阈值逻辑。

---

## 0. 百分位得分（统一输出）
对任一横截面数值序列 $x_i$，定义：
- 百分位分数：$pct_i \in [0,100]$
- 缺失值：用 50（中性）填充

对应代码：`_pct_rank_0_100`

---

## 1. 动量 (Momentum)
近 $n$ 日涨幅：
$$mom\_raw = \frac{P_t - P_{t-n}}{P_{t-n}}$$
最终得分：
$$score = pct\_rank(mom\_raw)$$

代码位置：`_scan_momentum`

---

## 2. 反转 (Reversal / Bias)
均线：
$$MA_{n,t} = \frac{1}{n}\sum_{k=0}^{n-1} P_{t-k}$$
乖离率：
$$bias = \frac{P_t - MA_{n,t}}{MA_{n,t}}$$
反转信号（超跌越多越大）：
$$rev\_raw = -bias$$
策略仅保留 $rev\_raw>0$（即 $bias<0$）

最终得分：
$$score = pct\_rank(rev\_raw)$$

代码位置：`_scan_reversal`

---

## 3. 夏普比率 (Sharpe Ratio)
日收益率：
$$r_t = \frac{P_t}{P_{t-1}} - 1$$
年化夏普（实现中默认 $R_f=0$）：
$$SR = \frac{\mathbb{E}[r]}{\sigma(r)}\sqrt{252}$$

代码位置：`_select_by_sharpe`

---

## 4. 概率夏普比率 (Probabilistic Sharpe Ratio, PSR)
$$PSR(SR^*) = \Phi \left( \frac{(SR^* - SR_{ref}) \sqrt{n-1}}{\sqrt{1 - \gamma_3 SR^* + \frac{\gamma_4 + 1}{4} SR^{*2}}} \right)$$

其中：
- $\Phi$：标准正态 CDF
- $\gamma_3$：偏度（skew）
- $\gamma_4$：超额峰度（excess kurtosis）
- $n$：样本数（收益率序列长度）

实现中的“去噪强度”（用于排序）：
$$sharpe\_adj = SR \times PSR$$
最终得分：
$$score = pct\_rank(sharpe\_adj)$$

代码位置：`_probabilistic_sharpe_ratio`, `_select_by_sharpe`

---

## 5. Composite（三因子融合）
先分别得到三因子的百分位：
- $mom\_pct = pct\_rank(mom\_raw)$
- $sharpe\_pct = pct\_rank(sharpe\_adj)$
- $rev\_pct = pct\_rank(rev\_raw)$（缺失用 50）

权重归一化后：
$$score = w_{mom} \cdot mom\_pct + w_{sharpe} \cdot sharpe\_pct + w_{rev} \cdot rev\_pct$$

代码位置：`_scan_composite`, `_normalize_weights`

---

## 6. Amihud 非流动性过滤（可选）
对每只 ETF 取最近 window 天：
$$illiq = \frac{1}{n}\sum_{t}\frac{|r_t|}{amount\_yuan_t}$$
其中：
$$amount\_yuan_t = amount_t \times amount\_scale$$

保留 $illiq$ 较小（流动性更好）的 `illiq_quantile` 分位以内标的。

代码位置：`_filter_liquidity`