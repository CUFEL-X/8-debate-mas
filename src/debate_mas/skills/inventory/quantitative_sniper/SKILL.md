---
name: quantitative_sniper
chinese_name: 量化狙击手
version: 1.2.0
role: hunter
group: investment
tags: [quant, momentum, reversal, sharpe, psr, composite, liquidity, percentile]
description: >
  标准化的量化筛选/排序工具。
  采用“数据准备 -> Universe过滤 -> 指标计算 -> 动态阈值 -> 结果封装”的结构化流程。
  支持动量(Momentum)、夏普(Sharpe+PSR)、反转(Reversal)、三因子融合(Composite)策略，并预留自定义策略接口。
  所有策略统一输出 score ∈ [0,100] 的横截面百分位分数；raw 指标保留在 extra 中便于追溯解释。
data_dependencies:
  - table: etf_daily
    source_file: etf_2025_data.csv
    required_columns: [code, date, close]
    optional_columns: [amount, vol, open, high, low, pre_close, change, pct_chg, adj_factor]
outputs:
  type: EtfCandidateList
schema_notes: >
    返回标准化的候选对象。
    extra 字段包含完整的策略元数据 (strategy, window, threshold_meta 等)，确保决策可追溯。
---

# Action Guide
## 你是谁
你是 Alpha Hunter 手中的精密武器。你不依赖直觉，而是执行严格的代码逻辑。你的任务是将模糊的交易直觉（涨得好/稳得住/超跌反弹）转化为精确的数学排序。

## 什么时候调用我
- 触发条件：
- 用户要“强势/趋势/动量” → `strategy="momentum"`
- 用户要“稳健/高夏普/低回撤倾向” → `strategy="sharpe"`（可选 `threshold_mode="psr"`）
- 用户要“超跌反弹/抄底/均线下方” → `strategy="reversal"`
- 用户要“综合三因子排序” → `strategy="composite"`
- 用户要自定义指标（练习/扩展） → `strategy="user_defined"`
- 禁止条件：
  - 案卷中缺失 `etf_daily` 表。
  - `ref_date` 为空（必须指定回测日期）。

## 怎么调用（必须遵守）
1. 读取数据：`ctx.dossier.get_table("etf_daily")`
2. 时间过滤：`apply_date_filter(df, ctx.ref_date)`（防未来函数）
3. 字段标准化：列名 lower，兼容 `data -> date`
4. Universe 过滤：支持 list / dict / EtfCandidate-like / str(json 或逗号分隔)
5. 策略路由：momentum / sharpe / reversal / composite / user_defined
6. 阈值过滤：
   - `none`：直接排序取 TopK
   - `quantile`：动态分位阈值（含 fallback meta）
   - `psr`：仅对 sharpe 策略生效（不足 TopK 会自动放宽或回退）
7. 输出：统一封装为 EtfCandidateList，score=0~100 百分位，raw/pct 写入 extra

## Inputs

### 核心参数
- `strategy` (str): `momentum`(默认) / `sharpe` / `reversal` / `composite` / `user_defined`
- `window` (int): 计算窗口，默认 20
- `top_k` (int): 返回数量，默认 5
- `universe` (list | str | None):
  - None：全市场
  - list：可为 `["510300","159934"]` 或 `[{symbol/code:..}, ...]` 或 EtfCandidate-like
  - str：支持 `'["159934","511360"]'` 或 `'159934,511360'`

### Phase 1: 流动性过滤 (Liquidity)
- `min_amount` (float): 成交额阈值（单位与数据源一致）
- `liquidity_filter` (str):
  - `amount_latest`（默认）：只看最新交易日成交额
  - `amihud`：Amihud 非流动性（价格冲击）过滤
- `amount_scale` (float): 将 amount 换算为元的系数（`amount_yuan = amount * amount_scale`）
- `illiq_quantile` (float): Amihud 过滤保留分位（默认 0.8：保留 illiq 较小的前 80%）

### Phase 3: 动态阈值 (Threshold)
- `threshold_mode` (str):
  - `none`
  - `quantile`
  - `psr`（主要用于 `sharpe`）
- `quantile_q` (float | None): 分位阈值 0~1（None 时按 top_k 自动推导）
- `psr_confidence` (float): PSR 置信度阈值（默认 0.95）
- `psr_ref_sharpe` (float): PSR 参考夏普（默认 0.0）

### Composite 组合权重
- `composite_weights` (dict | None): `{"mom":..., "sharpe":..., "rev":...}`，会自动归一化；缺省等权

## Outputs
- 成功：`SkillResult.ok`
  - `data.type="EtfCandidateList"`
  - `data.items=[EtfCandidate,...]`
  - `data.meta`：策略参数与阈值 meta
- 失败：`SkillResult.fail("原因")`