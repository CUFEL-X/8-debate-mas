# 量化狙击手 (Quantitative Sniper)

本技能包是多头 (Hunter) 的核心量化工具，用于在给定候选池或全市场范围内做量化排序与筛选。

## 输出特点
- **统一得分**：所有策略 score ∈ [0,100]（横截面百分位）
- **强可解释**：raw 指标与分位都保留在 extra；reason 也会包含关键 raw
- **可复用**：EtfCandidate.extra 合并“全局元信息 + 单标的指标”，PM/后续 rerank 可直接读 extra

## 策略说明 (Strategies)
- **Momentum**：追涨逻辑（近 N 日涨幅排序）
- **Sharpe + PSR**：稳健逻辑（用 PSR 去噪后排序，可选 PSR 阈值过滤）
- **Reversal**：抄底逻辑（均线下方超跌，Bias<0）
- **Composite**：三因子融合（mom/sharpe/rev 百分位加权）

## 👨‍💻 开发者指南 (Developer Guide)

handler.py 遵循“数据准备 -> 过滤 -> 计算 -> 阈值 -> 封装”的范式，适合做填空式教学扩展。

| 策略名称 | 对应函数 | 教学/修改位置 |
| :--- | :--- | :--- |
| Momentum | `_scan_momentum` | Phase 2: 计算 `mom_raw` |
| Sharpe | `_select_by_sharpe` | Phase 2: 计算 `sharpe` / `psr` / `sharpe_adj` |
| Reversal | `_scan_reversal` | Phase 2: 计算 `bias` / `rev_raw` |
| Composite | `_scan_composite` | Phase 2: 一次遍历算三因子 + 权重融合 |
| User Defined | `_user_defined_strategy` | [练习点] 自定义 RSI/MACD |

### 扩展步骤
1. 打开 `scripts/handler.py`
2. 实现 `_user_defined_strategy`
3. 模仿其他策略输出：`score(0~100 百分位) + reason + extra(raw/pct)`
4. 调用时传入 `strategy='user_defined'`
