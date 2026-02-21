from __future__ import annotations

import json
from math import erf, sqrt
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd

# =========================
# 基础工具
# =========================
def pct_rank_0_100(values: pd.Series, *, neutral: float = 50.0) -> pd.Series:
    """横截面百分位 rank 映射到 [0, 100]"""
    v = pd.to_numeric(values, errors="coerce")
    if v.empty:
        return v
    pct = v.rank(pct=True, method="average") * 100.0
    pct = pct.where(~pct.isna(), other=float(neutral))
    if pct.isna().all():
        pct[:] = float(neutral)
    return pct

def fmt(x: Any, *, nd: int = 2, na: str = "NA") -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return na
        return f"{float(x):.{nd}f}"
    except Exception:
        return na
    
def normalize_weights(w: Optional[Dict[str, float]]) -> Dict[str, float]:
    """接受 keys: mom, sharpe, rev，并归一化到 sum=1"""
    base = {"mom": 1.0 / 3.0, "sharpe": 1.0 / 3.0, "rev": 1.0 / 3.0}
    if not w:
        return base
    tmp = dict(base)
    for k in ("mom", "sharpe", "rev"):
        if k in w and isinstance(w[k], (int, float)) and np.isfinite(w[k]):
            tmp[k] = float(w[k])
    s = tmp["mom"] + tmp["sharpe"] + tmp["rev"]
    if s <= 1e-12:
        return base
    return {k: tmp[k] / s for k in tmp}

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def probabilistic_sharpe_ratio(sr_hat: float, sr_ref: float, n: int, skew: float, ex_kurt: float) -> float:
    """PSR: Probabilistic Sharpe Ratio (简化实现，与你原版一致的计算结构)"""
    if n <= 2:
        return 0.0
    denom = 1.0 - (skew * sr_hat) + ((ex_kurt + 1.0) / 4.0) * (sr_hat**2)
    if np.isnan(denom) or denom <= 0:
        denom = 1e-12
    z = (sr_hat - sr_ref) * sqrt(max(n - 1, 1)) / sqrt(denom)
    return float(min(max(norm_cdf(float(z)), 0.0), 1.0))

# =========================
# 流动性过滤
# =========================
def filter_liquidity(
    df: pd.DataFrame,
    *,
    min_amount: float,
    liquidity_filter: str,
    amount_scale: float,
    window: int,
    illiq_quantile: float,
) -> np.ndarray | List[str]:
    """流动性过滤器"""
    if "amount" not in df.columns: 
        return df["code"].unique()
        
    # 模式 1: 最新日成交额
    if liquidity_filter == "amount_latest":
        latest_date = df["date"].max()
        latest_df = df[df["date"] == latest_date]
        valid = latest_df[latest_df["amount"] > min_amount]["code"].unique()
        return valid
    
    # 模式 2: Amihud 非流动性因子
    if liquidity_filter == "amihud":
        d = df.sort_values("date").copy()
        d["ret"] = d.groupby("code")["close"].pct_change()
        d["amount_yuan"] = d["amount"] * float(amount_scale) # 换算成元

        d = d.dropna(subset=["ret", "amount_yuan"])
        d = d[d["amount_yuan"] > 0]
        if d.empty: 
            return df["code"].unique()
 
        d = d.groupby("code").tail(window)
        illiq = d.groupby("code").apply(lambda g: float(np.mean(np.abs(g["ret"]) / g["amount_yuan"])))
        if illiq.empty: 
            return df["code"].unique()

        cutoff = illiq.quantile(float(illiq_quantile))
        valid = illiq[illiq <= cutoff].index.astype(str).tolist()
        return valid

    return df["code"].unique()


# =========================
# 阈值过滤
# =========================
def apply_threshold_quantile(
    df_score: pd.DataFrame,
    *,
    top_k: int,
    quantile_q: Optional[float],
    enabled: bool,
) -> pd.DataFrame:
    """[Threshold] 动态分位阈值"""
    if not enabled or df_score.empty:
        return df_score

    n = len(df_score)
    if quantile_q is None:
        q = 1.0 - (float(top_k) / float(max(n, 1)))
        q = float(min(max(q, 0.0), 1.0))
    else:
        q = float(min(max(float(quantile_q), 0.0), 1.0))

    cutoff = float(df_score["score"].quantile(q))
    before_cnt = len(df_score)
    filtered = df_score[df_score["score"] >= cutoff].copy()

    if len(filtered) < int(top_k):
        df_score.attrs["threshold_meta"] = {
            "mode": "quantile",
            "q": q,
            "fallback": "none (passed < top_k)",
            "passed": len(filtered),
            "before": before_cnt,
        }
        return df_score

    filtered.attrs["threshold_meta"] = {
        "mode": "quantile",
        "q": q,
        "cutoff": cutoff,
        "passed": len(filtered),
        "before": before_cnt,
    }
    return filtered

def apply_threshold_psr(df_score: pd.DataFrame, *, top_k: int, psr_confidence: float) -> pd.DataFrame:
    """PSR 概率阈值（不足 top_k 自动放宽或回退）"""
    before_cnt = len(df_score)
    conf1 = float(psr_confidence)

    filtered = df_score[df_score["psr"] >= conf1].copy()
    effective_conf = conf1
    fallback_used: Optional[str] = None

    if len(filtered) < int(top_k):
        conf2 = min(conf1, 0.90)
        filtered2 = df_score[df_score["psr"] >= conf2].copy()
        if len(filtered2) > 0:
            filtered = filtered2
            effective_conf = conf2
            fallback_used = f"relax_to_{conf2}"
        else:
            filtered = df_score.copy()
            effective_conf = 0.0
            fallback_used = "fallback_to_none"

    filtered.attrs["threshold_meta"] = {
        "mode": "psr",
        "confidence": psr_confidence,
        "effective": effective_conf,
        "fallback": fallback_used,
        "passed": len(filtered),
        "before": before_cnt,
    }
    return filtered

# =========================
# 核心策略实现：返回 df_score
# df_score 必须至少包含：symbol, score, reason, extra
# =========================
def scan_momentum(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """[策略] 动量：计算近 N 日涨幅"""
    window = int(params["window"])
    top_k = int(params["top_k"])
    min_amount = float(params["min_amount"])
    liquidity_filter = str(params["liquidity_filter"])
    amount_scale = float(params["amount_scale"])
    illiq_quantile = float(params["illiq_quantile"])
    threshold_mode = str(params["threshold_mode"])
    quantile_q = params.get("quantile_q")

    # --- Phase 1: 流动性过滤 (Liquidity Filter) ---
    valid_codes = filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
    target_df = df[df["code"].isin(valid_codes)].copy()

    # --- Phase 2: 核心指标计算 (Core Calculation) ---
    rows: List[Dict[str, Any]] = []
    for code, g in target_df.sort_values("date").groupby("code"):
        if len(g) < window + 1:
            continue
        curr = float(g.iloc[-1]["close"])
        prev = float(g.iloc[-(window + 1)]["close"])
        if prev <= 0:
            continue
        mom_raw = (curr - prev) / prev
        rows.append({"symbol": str(code), "mom_raw": float(mom_raw)})

    # --- Phase 3: 结果处理 (Result Processing) ---
    if not rows:
        return pd.DataFrame()

    df_score = pd.DataFrame(rows)
    df_score["mom_pct"] = pct_rank_0_100(df_score["mom_raw"], neutral=50.0)
    df_score["score"] = df_score["mom_pct"]
    df_score["reason"] = df_score.apply(
        lambda r: f"近{window}日涨幅 {r['mom_raw']*100:.2f}% | pct {r['mom_pct']:.1f}",
        axis=1,
    )
    df_score["extra"] = df_score.apply(
        lambda r: {"mom_raw": float(r["mom_raw"]), "mom_pct": float(r["mom_pct"])},
        axis=1,
    )

    df_score = apply_threshold_quantile(df_score, top_k, quantile_q, enabled=(threshold_mode == "quantile"))
    return df_score

def select_by_sharpe(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """[策略] 夏普比率：稳健优选 (支持 PSR 概率调整)"""
    window = int(params["window"])
    top_k = int(params["top_k"])
    min_amount = float(params["min_amount"])
    liquidity_filter = str(params["liquidity_filter"])
    amount_scale = float(params["amount_scale"])
    illiq_quantile = float(params["illiq_quantile"])
    threshold_mode = str(params["threshold_mode"])
    psr_confidence = float(params["psr_confidence"])
    psr_ref_sharpe = float(params["psr_ref_sharpe"])

    # --- Phase 1: 流动性过滤 ---
    valid_codes = filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
    target_df = df[df["code"].isin(valid_codes)].copy()

    # --- Phase 2: 核心指标计算 ---
    rows: List[Dict[str, Any]] = []
    for code, g in target_df.sort_values("date").groupby("code"):
        if len(g) < window + 1:
            continue

        recent_data = g.iloc[-(window + 1) :].copy()
        recent_data["ret"] = recent_data["close"].pct_change()
        rets = recent_data["ret"].dropna()

        if len(rets) < max(5, window // 3):
            continue

        mu = float(rets.mean())
        sig = float(rets.std())
        if sig <= 1e-6 or np.isnan(sig):
            continue

        sharpe = (mu / sig) * np.sqrt(252)

        skew = float(rets.skew()) if len(rets) > 2 else 0.0
        ex_kurt = float(rets.kurt()) if len(rets) > 2 else 0.0
        n = int(len(rets))

        psr = probabilistic_sharpe_ratio(sharpe, float(psr_ref_sharpe), n, skew, ex_kurt)
        sharpe_adj = float(sharpe) * float(psr)

        rows.append(
            {
                "symbol": str(code),
                "sharpe": float(sharpe),
                "psr": float(psr),
                "n": n,
                "sharpe_adj": float(sharpe_adj),
            }
        )

    # --- Phase 3: 结果处理 (包含 PSR 专用逻辑) ---
    if not rows:
        return pd.DataFrame()

    df_score = pd.DataFrame(rows)
    df_score["sharpe_pct"] = pct_rank_0_100(df_score["sharpe_adj"], neutral=50.0)
    df_score["score"] = df_score["sharpe_pct"]

    if threshold_mode == "psr":
        df_score = apply_threshold_psr(df_score, top_k, psr_confidence)

    df_score["reason"] = df_score.apply(
        lambda r: f"夏普 {r['sharpe']:.2f} | PSR {r['psr']:.2f} | pct {r['sharpe_pct']:.1f} | win={window}",
        axis=1,
    )
    # 对齐你原版：extra 不塞 n（原版没塞）
    df_score["extra"] = df_score.apply(
        lambda r: {
            "sharpe_raw": float(r["sharpe"]),
            "psr": float(r["psr"]),
            "sharpe_adj": float(r["sharpe_adj"]),
            "sharpe_pct": float(r["sharpe_pct"]),
        },
        axis=1,
    )
    return df_score

def scan_reversal(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """[策略] 超跌反弹：负乖离率 (Bias < 0)"""
    window = int(params["window"])
    top_k = int(params["top_k"])
    min_amount = float(params["min_amount"])
    liquidity_filter = str(params["liquidity_filter"])
    amount_scale = float(params["amount_scale"])
    illiq_quantile = float(params["illiq_quantile"])
    threshold_mode = str(params["threshold_mode"])
    quantile_q = params.get("quantile_q")

    adj_min_amount = (min_amount / 2.0) if liquidity_filter == "amount_latest" else min_amount
    
    # --- Phase 1: 流动性过滤 ---
    valid_codes = filter_liquidity(df, adj_min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
    target_df = df[df["code"].isin(valid_codes)].copy()

    # --- Phase 2: 核心指标计算 ---
    rows: List[Dict[str, Any]] = []
    for code, g in target_df.sort_values("date").groupby("code"):
        if len(g) < window:
            continue
        curr = float(g.iloc[-1]["close"])
        ma = float(g["close"].tail(window).mean())
        if not np.isfinite(ma) or ma == 0:
            continue
        bias = (curr - ma) / ma
        rev_raw = -bias
        rows.append({"symbol": str(code), "bias": float(bias), "rev_raw": float(rev_raw)})

    # --- Phase 3: 结果处理 ---    
    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)
    df_all["rev_pct"] = pct_rank_0_100(df_all["rev_raw"], neutral=50.0)

    df_score = df_all[df_all["rev_raw"] > 0].copy()
    if df_score.empty:
        return pd.DataFrame()

    df_score["score"] = df_score["rev_pct"]
    df_score["reason"] = df_score.apply(
        lambda r: f"乖离率 {r['bias']*100:.2f}% | oversold {r['rev_raw']*100:.2f}% | pct {r['rev_pct']:.1f}",
        axis=1,
    )
    df_score["extra"] = df_score.apply(
        lambda r: {"rev_raw": float(r["rev_raw"]), "rev_pct": float(r["rev_pct"]), "bias": float(r["bias"])},
        axis=1,
    )

    df_score = apply_threshold_quantile(df_score, top_k, quantile_q, enabled=(threshold_mode == "quantile"))
    return df_score

def scan_composite(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Composite：一次遍历计算 mom/sharpe/rev 三因子，并统一转成 0~100 百分位后融合（默认等权）。"""
    window = int(params["window"])
    top_k = int(params["top_k"])
    min_amount = float(params["min_amount"])
    liquidity_filter = str(params["liquidity_filter"])
    amount_scale = float(params["amount_scale"])
    illiq_quantile = float(params["illiq_quantile"])
    threshold_mode = str(params["threshold_mode"])
    quantile_q = params.get("quantile_q")
    psr_ref_sharpe = float(params["psr_ref_sharpe"])

    composite_weights = params.get("composite_weights", None)
    if isinstance(composite_weights, str):
        try:
            composite_weights = json.loads(composite_weights)
        except Exception:
            composite_weights = None
    if composite_weights is not None and not isinstance(composite_weights, dict):
        composite_weights = None

    valid_codes = filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
    target_df = df[df["code"].isin(valid_codes)].copy()

    w = normalize_weights(composite_weights)

    rows: List[Dict[str, Any]] = []
    for code, g in target_df.sort_values("date").groupby("code"):
        mom_raw = np.nan
        if len(g) >= window + 1:
            curr = float(g.iloc[-1]["close"])
            prev = float(g.iloc[-(window + 1)]["close"])
            if prev > 0:
                mom_raw = (curr - prev) / prev

        rev_raw = np.nan
        bias = np.nan
        if len(g) >= window:
            curr = float(g.iloc[-1]["close"])
            ma = float(g["close"].tail(window).mean())
            if np.isfinite(ma) and ma != 0:
                bias = (curr - ma) / ma
                rev_raw = -bias

        sharpe = np.nan
        psr = np.nan
        sharpe_adj = np.nan
        if len(g) >= window + 1:
            recent = g.iloc[-(window + 1) :].copy()
            recent["ret"] = recent["close"].pct_change()
            rets = recent["ret"].dropna()
            if len(rets) >= max(5, window // 3):
                mu = float(rets.mean())
                sig = float(rets.std())
                if sig > 1e-6 and not np.isnan(sig):
                    sharpe = (mu / sig) * np.sqrt(252)
                    skew = float(rets.skew()) if len(rets) > 2 else 0.0
                    ex_kurt = float(rets.kurt()) if len(rets) > 2 else 0.0
                    n = int(len(rets))
                    psr = probabilistic_sharpe_ratio(sharpe, float(psr_ref_sharpe), n, skew, ex_kurt)
                    sharpe_adj = float(sharpe) * float(psr)

        rows.append(
            {
                "symbol": str(code),
                "mom_raw": mom_raw,
                "rev_raw": rev_raw,
                "bias": bias,
                "sharpe_raw": sharpe,
                "psr": psr,
                "sharpe_adj": sharpe_adj,
            }
        )

    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)

    d["mom_pct"] = pct_rank_0_100(d["mom_raw"], neutral=50.0)
    d["rev_pct"] = pct_rank_0_100(d["rev_raw"], neutral=50.0)
    d["sharpe_pct"] = pct_rank_0_100(d["sharpe_adj"], neutral=50.0)

    d["score"] = w["mom"] * d["mom_pct"] + w["sharpe"] * d["sharpe_pct"] + w["rev"] * d["rev_pct"]
    d["composite_score"] = d["score"]

    d["reason"] = d.apply(
        lambda r: (
            f"Comp {r['score']:.1f} | mom {r['mom_pct']:.0f}, sharpe {r['sharpe_pct']:.0f}, rev {r['rev_pct']:.0f} "
            f"(raw: mom {fmt(r['mom_raw']*100 if np.isfinite(r['mom_raw']) else np.nan, nd=2)}%, "
            f"sr {fmt(r['sharpe_raw'], nd=2)}, psr {fmt(r['psr'], nd=2)}, "
            f"rev {fmt(r['rev_raw']*100 if np.isfinite(r['rev_raw']) else np.nan, nd=2)}%)"
        ),
        axis=1,
    )

    d["extra"] = d.apply(
        lambda r: {
            "mom_raw": float(r["mom_raw"]) if np.isfinite(r["mom_raw"]) else None,
            "mom_pct": float(r["mom_pct"]),
            "sharpe_raw": float(r["sharpe_raw"]) if np.isfinite(r["sharpe_raw"]) else None,
            "psr": float(r["psr"]) if np.isfinite(r["psr"]) else None,
            "sharpe_adj": float(r["sharpe_adj"]) if np.isfinite(r["sharpe_adj"]) else None,
            "sharpe_pct": float(r["sharpe_pct"]),
            "rev_raw": float(r["rev_raw"]) if np.isfinite(r["rev_raw"]) else None,
            "rev_pct": float(r["rev_pct"]),
            "bias": float(r["bias"]) if np.isfinite(r["bias"]) else None,
            "composite_weights": dict(w),
            "composite_score": float(r["composite_score"]),
        },
        axis=1,
    )

    if threshold_mode == "quantile":
        d = apply_threshold_quantile(d, top_k, quantile_q, enabled=True)

    return d

def user_defined_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    [TODO] 自定义策略（练习入口）

    输入:
      - df: 已完成防未来(ref_date)与字段标准化后的 etf_daily，至少含 code/date/close（amount 可能存在）
      - params: handler 传入的参数字典（至少包含 window/top_k/min_amount/liquidity_filter 等）

    输出（必须）:
      - 返回 df_score: DataFrame
        - 必须至少包含列：symbol, score, reason, extra
        - score 建议映射到 0~100（百分位）以与其他策略一致
        - extra 必须包含你关键中间变量（便于解释/评分）

    验收:
      - 运行 strategy="user_defined"
      - 检查 SkillResult.data.type == "EtfCandidateList"
      - 检查 items[*].extra 是否包含你的关键中间变量
    """
    # 学生只需要在这里写策略逻辑，并返回 df_score
    # 例如：
    # window = int(params["window"])
    # ... 计算 raw 指标 -> pct_rank_0_100 -> score -> reason -> extra
    # return df_score

    raise NotImplementedError("自定义策略尚未实现：请在 algo.py 的 user_defined_strategy 中实现并返回 df_score。")

def run_strategy(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    s = str(params["strategy"])
    if s == "momentum":
        return scan_momentum(df, params)
    if s == "sharpe":
        return select_by_sharpe(df, params)
    if s == "reversal":
        return scan_reversal(df, params)
    if s == "composite":
        return scan_composite(df, params)
    if s == "user_defined":
        return user_defined_strategy(df, params)
    raise ValueError(f"unknown strategy: {s}")