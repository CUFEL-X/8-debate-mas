# skills/inventory/quantitative_sniper/scripts/handler.py
from __future__ import annotations

import json
from math import erf, sqrt
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from debate_mas.protocol import EtfCandidate, SkillResult
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

from .dataloader import load_etf_daily
from .algo import run_strategy

Strategy = Literal["momentum", "sharpe", "reversal", "composite", "user_defined"]


class SkillHandler(BaseFinanceSkill):
    """[Hunter] 量化狙击手：取数/路由/包装"""

    # =========================
    # 基础工具：输入归一化
    # =========================
    @staticmethod
    def _normalize_universe(universe: Optional[Union[List[str], str]]) -> Optional[List[str]]:
        """
        universe 允许三种输入：
        - None
        - List[str] / List[dict] / List[EtfCandidate-like]
        - str: '["159934","511360"]' 或 '159934,511360'
        """
        if universe is None:
            return None

        def _coerce_one(x: Any) -> Optional[str]:
            if x is None:
                return None
            # dict: 优先取 symbol/code
            if isinstance(x, dict):
                v = x.get("symbol") or x.get("code")
                return str(v).strip() if v else None
            # EtfCandidate-like:取 symbol 属性
            if hasattr(x, "symbol"):
                v = getattr(x, "symbol")
                return str(v).strip() if v else None
            return str(x).strip()

        # list -> list[str]
        if isinstance(universe, list):
            out: List[str] = []
            seen = set()
            for x in universe:
                s = _coerce_one(x)
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
            return out or None

        # str -> 优先尝试 json -> 失败则按逗号拆分
        if isinstance(universe, str):
            s = universe.strip()
            if not s:
                return None
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return SkillHandler._normalize_universe(obj)
                if isinstance(obj, (str, int, float)):
                    t = str(obj).strip()
                    return [t] if t else None
            except Exception:
                pass

            parts = [p.strip() for p in s.replace("\n", ",").split(",")]
            return SkillHandler._normalize_universe(parts)

        t = _coerce_one(universe)
        return [t] if t else None

    # 百分位得分工具（统一映射到 0~100）
    @staticmethod
    def _pct_rank_0_100(values: pd.Series, *, neutral: float = 50.0) -> pd.Series:
        """横截面百分位 rank 映射到 [0, 100]"""
        v = pd.to_numeric(values, errors="coerce")
        if v.empty:
            return v
        pct = v.rank(pct=True, method="average") * 100.0
        pct = pct.where(~pct.isna(), other=float(neutral))
        # 全 NaN 情况：直接全填 neutral
        if pct.isna().all():
            pct[:] = float(neutral)
        return pct

    @staticmethod
    def _fmt(x: Any, *, nd: int = 2, na: str = "NA") -> str:
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return na
            return f"{float(x):.{nd}f}"
        except Exception:
            return na

    @staticmethod
    def _normalize_weights(w: Optional[Dict[str, float]]) -> Dict[str, float]:
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

    # =========================
    # 主入口：execute
    # - 输入表: etf_daily 必须至少包含: code/date/close
    # - 输出: SkillResult.data.type == "EtfCandidateList"
    # =========================
    def execute(
        self,
        ctx: SkillContext,
        strategy: Strategy = "momentum",
        window: int = 20,
        top_k: int = 5,
        min_amount: float = 1000,
        universe: Optional[Union[List[str], str]] = None,
        liquidity_filter: Literal["amount_latest", "amihud"] = "amount_latest",
        amount_scale: float = 1000.0,
        illiq_quantile: float = 0.8,
        threshold_mode: Literal["none", "quantile", "psr"] = "none",
        quantile_q: Optional[float] = None,         
        psr_confidence: float = 0.95,            
        psr_ref_sharpe: float = 0.0,
        composite_weights: Optional[Union[Dict[str, float], str]] = None,
    ) -> SkillResult:
        
        # 1. 数据准备 (Data Preparation)
        df = ctx.dossier.get_table("etf_daily")
        if df is None or df.empty:
            return SkillResult.fail("案卷中找不到 'etf_daily' 数据。")

        # 时间切片（防未来函数）
        df = self.apply_date_filter(df, ctx.ref_date)
        if df is None or df.empty:
            return SkillResult.fail(f"截止 {ctx.ref_date} 无可用行情数据。")

        # 字段标准化
        df = df.rename(columns=lambda x: str(x).strip().lower())
        if "data" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"data": "date"})
        
        # 强制类型转换 (Robustness)
        try:
            if "code" in df.columns:
                df["code"] = df["code"].astype(str)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            if "amount" in df.columns:
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        except Exception as e:
            return SkillResult.fail(f"数据清洗失败: {e}")

        df = df.dropna(subset=["code", "date", "close"])
        if df.empty:
            return SkillResult.fail("清洗后数据为空（code/date/close缺失）。")
        
        # 2. Universe 过滤 (Universe Filtering)
        universe_list = self._normalize_universe(universe)
        universe_set = set(universe_list) if universe_list else None
        if universe_set:
            df = df[df["code"].isin(universe_set)].copy()
            if df.empty:
                return SkillResult.fail(
                    f"universe 过滤后为空：传入 {len(universe_set)} 个代码，但行情表无匹配。"
                )
            
        # 3. 策略分发 (Strategy Dispatch)
        if quantile_q is not None and not isinstance(quantile_q, (int, float)):
            try:
                quantile_q = float(quantile_q)
            except Exception:
                quantile_q = None

        params: Dict[str, Any] = {
            "strategy": strategy,
            "window": int(window),
            "top_k": int(top_k),
            "min_amount": float(min_amount),
            "liquidity_filter": liquidity_filter,
            "amount_scale": float(amount_scale),
            "illiq_quantile": float(illiq_quantile),
            "threshold_mode": threshold_mode,
            "quantile_q": quantile_q,
            "psr_confidence": float(psr_confidence),
            "psr_ref_sharpe": float(psr_ref_sharpe),
            "universe_size": len(universe_set) if universe_set else None,
            "ref_date": ctx.ref_date,
            "agent_role": ctx.agent_role,
            "composite_weights": composite_weights,
        }

        if strategy == "momentum":
            return self._scan_momentum(df, **params)
        if strategy == "sharpe":
            return self._select_by_sharpe(df, **params)
        if strategy == "reversal":
            return self._scan_reversal(df, **params)
        if strategy == "composite":
            return self._scan_composite(df, **params)
        if strategy == "user_defined":
            return self._user_defined_strategy(df, window=int(window), top_k=int(top_k), min_amount=float(min_amount))

        return SkillResult.fail(f"不支持的策略类型: {strategy}")
    
    # =========================================================================
    # 核心策略实现 (Core Strategies)
    # =========================================================================
    def _scan_momentum(
        self,
        df: pd.DataFrame,
        window: int,
        top_k: int,
        min_amount: float,
        liquidity_filter: str,
        amount_scale: float,
        illiq_quantile: float,
        threshold_mode: str,
        quantile_q: Optional[float],
        universe_size: Optional[int],
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        """[策略] 动量：计算近 N 日涨幅"""
        # --- Phase 1: 流动性过滤 (Liquidity Filter) ---
        valid_codes = self._filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
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
            rows.append({"symbol": code, "mom_raw": mom_raw})

        # --- Phase 3: 结果处理 (Result Processing) ---
        if not rows:
            return self._wrap_empty_result("momentum", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

        df_score = pd.DataFrame(rows)
        df_score["mom_pct"] = self._pct_rank_0_100(df_score["mom_raw"], neutral=50.0)
        df_score["score"] = df_score["mom_pct"]

        df_score["reason"] = df_score.apply(
            lambda r: f"近{window}日涨幅 {r['mom_raw']*100:.2f}% | pct {r['mom_pct']:.1f}",
            axis=1,
        )
        df_score["extra"] = df_score.apply(
            lambda r: {"mom_raw": float(r["mom_raw"]), "mom_pct": float(r["mom_pct"])},
            axis=1,
        )

        df_score = self._apply_threshold_quantile(df_score, top_k, quantile_q, enabled=(threshold_mode == "quantile"))
        return self._finalize_result(df_score, top_k, "momentum", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

    def _select_by_sharpe(
        self,
        df: pd.DataFrame,
        window: int,
        top_k: int,
        min_amount: float,
        liquidity_filter: str,
        amount_scale: float,
        illiq_quantile: float,
        threshold_mode: str,
        psr_confidence: float,
        psr_ref_sharpe: float,
        universe_size: Optional[int],
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        """[策略] 夏普比率：稳健优选 (支持 PSR 概率调整)"""
        # --- Phase 1: 流动性过滤 ---
        valid_codes = self._filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
        target_df = df[df["code"].isin(valid_codes)].copy()

        # --- Phase 2: 核心指标计算 ---
        rows: List[Dict[str, Any]] = []
        for code, g in target_df.sort_values("date").groupby("code"):
            if len(g) < window + 1:
                continue

            recent_data = g.iloc[-(window + 1):].copy()
            recent_data["ret"] = recent_data["close"].pct_change()
            rets = recent_data["ret"].dropna()
            
            if len(rets) < max(5, window // 3): continue

            mu = float(rets.mean())
            sig = float(rets.std())
            if sig <= 1e-6 or np.isnan(sig): continue # 防除零

            sharpe = (mu / sig) * np.sqrt(252)

            skew = float(rets.skew()) if len(rets) > 2 else 0.0
            ex_kurt = float(rets.kurt()) if len(rets) > 2 else 0.0
            n = int(len(rets))
            psr = self._probabilistic_sharpe_ratio(sharpe, float(psr_ref_sharpe), n, skew, ex_kurt)
            sharpe_adj = float(sharpe) * float(psr) 
            rows.append({"symbol": code, "sharpe": float(sharpe), "psr": float(psr), "n": n, "sharpe_adj": float(sharpe_adj)})

        # --- Phase 3: 结果处理 (包含 PSR 专用逻辑) ---
        if not rows:
            return self._wrap_empty_result("sharpe", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

        df_score = pd.DataFrame(rows)
        df_score["sharpe_pct"] = self._pct_rank_0_100(df_score["sharpe_adj"], neutral=50.0)
        df_score["score"] = df_score["sharpe_pct"]

        if threshold_mode == "psr":
            df_score = self._apply_threshold_psr(df_score, top_k, psr_confidence)

        df_score["reason"] = df_score.apply(
            lambda r: f"夏普 {r['sharpe']:.2f} | PSR {r['psr']:.2f} | pct {r['sharpe_pct']:.1f} | win={window}",
            axis=1,
        )
        df_score["extra"] = df_score.apply(
            lambda r: {
                "sharpe_raw": float(r["sharpe"]),
                "psr": float(r["psr"]),
                "sharpe_adj": float(r["sharpe_adj"]),
                "sharpe_pct": float(r["sharpe_pct"]),
            },
            axis=1,
        )
        return self._finalize_result(df_score, top_k, "sharpe", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

    def _scan_reversal(
        self,
        df: pd.DataFrame,
        window: int,
        top_k: int,
        min_amount: float,
        liquidity_filter: str,
        amount_scale: float,
        illiq_quantile: float,
        threshold_mode: str,
        quantile_q: Optional[float],
        universe_size: Optional[int],
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        """[策略] 超跌反弹：负乖离率 (Bias < 0)"""
        adj_min_amount = (min_amount / 2) if liquidity_filter == "amount_latest" else min_amount
        
        # --- Phase 1: 流动性过滤 ---
        valid_codes = self._filter_liquidity(df, adj_min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
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
            rows.append({"symbol": code, "bias": float(bias), "rev_raw": float(rev_raw)})

        # --- Phase 3: 结果处理 ---
        if not rows:
            return self._wrap_empty_result("reversal", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

        df_all = pd.DataFrame(rows)
        df_all["rev_pct"] = self._pct_rank_0_100(df_all["rev_raw"], neutral=50.0)

        df_score = df_all[df_all["rev_raw"] > 0].copy()
        if df_score.empty:
            return self._wrap_empty_result("reversal", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

        df_score["score"] = df_score["rev_pct"]
        df_score["reason"] = df_score.apply(
            lambda r: f"乖离率 {r['bias']*100:.2f}% | oversold {r['rev_raw']*100:.2f}% | pct {r['rev_pct']:.1f}",
            axis=1,
        )
        df_score["extra"] = df_score.apply(
            lambda r: {"rev_raw": float(r["rev_raw"]), "rev_pct": float(r["rev_pct"]), "bias": float(r["bias"])},
            axis=1,
        )
        df_score = self._apply_threshold_quantile(df_score, top_k, quantile_q, enabled=(threshold_mode == "quantile"))
        return self._finalize_result(df_score, top_k, "reversal", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

    def _scan_composite(
        self,
        df: pd.DataFrame,
        window: int,
        top_k: int,
        min_amount: float,
        liquidity_filter: str,
        amount_scale: float,
        illiq_quantile: float,
        threshold_mode: str,
        quantile_q: Optional[float],
        psr_confidence: float,
        psr_ref_sharpe: float,
        universe_size: Optional[int],
        composite_weights: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SkillResult:
        """Composite：一次遍历计算 mom/sharpe/rev 三因子，并统一转成 0~100 百分位后融合（默认等权）。"""
        if isinstance(composite_weights, str):
            try:
                composite_weights = json.loads(composite_weights)
            except Exception:
                composite_weights = None

        valid_codes = self._filter_liquidity(df, min_amount, liquidity_filter, amount_scale, window, illiq_quantile)
        target_df = df[df["code"].isin(valid_codes)].copy()

        w = self._normalize_weights(composite_weights)

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
                recent = g.iloc[-(window + 1):].copy()
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
                        psr = self._probabilistic_sharpe_ratio(sharpe, float(psr_ref_sharpe), n, skew, ex_kurt)
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
            return self._wrap_empty_result("composite", window, universe_size, liquidity_filter, threshold_mode, meta=meta)

        d = pd.DataFrame(rows)

        d["mom_pct"] = self._pct_rank_0_100(d["mom_raw"], neutral=50.0)
        d["rev_pct"] = self._pct_rank_0_100(d["rev_raw"], neutral=50.0)
        d["sharpe_pct"] = self._pct_rank_0_100(d["sharpe_adj"], neutral=50.0)

        d["score"] = w["mom"] * d["mom_pct"] + w["sharpe"] * d["sharpe_pct"] + w["rev"] * d["rev_pct"]
        d["composite_score"] = d["score"]

        d["reason"] = d.apply(
            lambda r: (
                f"Comp {r['score']:.1f} | mom {r['mom_pct']:.0f}, sharpe {r['sharpe_pct']:.0f}, rev {r['rev_pct']:.0f} "
                f"(raw: mom {self._fmt(r['mom_raw']*100 if np.isfinite(r['mom_raw']) else np.nan, nd=2)}%, "
                f"sr {self._fmt(r['sharpe_raw'], nd=2)}, psr {self._fmt(r['psr'], nd=2)}, "
                f"rev {self._fmt(r['rev_raw']*100 if np.isfinite(r['rev_raw']) else np.nan, nd=2)}%)"
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
            d = self._apply_threshold_quantile(d, top_k, quantile_q, enabled=True)

        return self._finalize_result(d, top_k, "composite", window, universe_size, liquidity_filter, threshold_mode, meta=meta)
    
    # =========================================================================
    # 辅助逻辑 (Helper Methods - Engineering Clean)
    # =========================================================================
    def _finalize_result(
        self,
        df_score: pd.DataFrame,
        top_k: int,
        strategy: str,
        window: int,
        universe_size: Optional[int],
        liquidity_filter: str,
        threshold_mode: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        """[Helper] 统一的排序、截断与结果包装"""
        final_df = df_score.sort_values("score", ascending=False).head(top_k)
        cols = ["symbol", "score", "reason"]
        if "extra" in final_df.columns:
            cols.append("extra")
        candidates = final_df[cols].to_dict("records")

        threshold_meta = df_score.attrs.get("threshold_meta")
        return self._wrap_result(
            candidates, strategy=strategy, window=window, 
            universe_size=universe_size,
            liquidity_filter=liquidity_filter, 
            threshold_mode=threshold_mode,
            threshold_meta=threshold_meta,
            meta=meta
        )

    def _wrap_empty_result(
        self,
        strategy: str,
        window: int,
        universe_size: Optional[int],
        liquidity_filter: str,
        threshold_mode: str,
        meta: Optional[Dict[str, Any]] = None, 
    ) -> SkillResult:
        """[Helper] 快速返回空结果"""
        return self._wrap_result([], strategy, window, universe_size, liquidity_filter, threshold_mode, threshold_meta=None, meta=meta)

    def _apply_threshold_quantile(
        self,
        df_score: pd.DataFrame,
        top_k: int,
        quantile_q: Optional[float],
        enabled: bool,
    ) -> pd.DataFrame:
        """[Threshold] 动态分位阈值"""
        if not enabled or df_score.empty: return df_score

        N = len(df_score)
        if quantile_q is None:
            q = 1.0 - (float(top_k) / float(max(N, 1)))
            q = float(min(max(q, 0.0), 1.0))
        else:
            q = float(min(max(quantile_q, 0.0), 1.0))

        cutoff = float(df_score["score"].quantile(q))
        before_cnt = len(df_score)
        filtered = df_score[df_score["score"] >= cutoff].copy()

        if len(filtered) < int(top_k):
            df_score.attrs["threshold_meta"] = {"mode": "quantile", "q": q, "fallback": "none (passed < top_k)", "passed": len(filtered)}
            return df_score

        filtered.attrs["threshold_meta"] = {"mode": "quantile", "q": q, "cutoff": cutoff, "passed": len(filtered), "before": before_cnt}
        return filtered

    def _apply_threshold_psr(self, df_score: pd.DataFrame, top_k: int, psr_confidence: float):
        """[Threshold] PSR 概率夏普阈值 (特化逻辑)"""
        before_cnt = len(df_score)
        conf1 = float(psr_confidence)

        filtered = df_score[df_score["psr"] >= conf1].copy()
        effective_conf = conf1
        fallback_used = None

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
            "mode": "psr", "confidence": psr_confidence, "effective": effective_conf, 
            "fallback": fallback_used, "passed": len(filtered), "before": before_cnt
        }
        return filtered

    # ================= 基础工具 (Infrastructure) =================
    def _norm_cdf(self, x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _probabilistic_sharpe_ratio(self, sr_hat: float, sr_ref: float, n: int, skew: float, ex_kurt: float) -> float:
        if n <= 2:
            return 0.0
        denom = 1.0 - (skew * sr_hat) + ((ex_kurt + 1.0) / 4.0) * (sr_hat**2)
        if np.isnan(denom) or denom <= 0:
            denom = 1e-12
        z = (sr_hat - sr_ref) * sqrt(max(n - 1, 1)) / sqrt(denom)
        return float(min(max(self._norm_cdf(float(z)), 0.0), 1.0))

    def _filter_liquidity(
        self,
        df: pd.DataFrame,
        min_amount: float,
        liquidity_filter: str,
        amount_scale: float,
        window: int,
        illiq_quantile: float,
    ):
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

    def _user_defined_strategy(self, df, window, top_k, min_amount):
        """
        [TODO]
        输入: df 为已完成防未来(ref_date)与字段标准化后的 etf_daily，至少含 code/date/close（amount 可能存在）
        输出: 必须返回 SkillResult，且 data.type == "EtfCandidateList"，items 内每项包含 symbol/score/reason/extra
        验收: 运行 strategy="user_defined"；检查 items[*].extra 是否包含你的关键中间变量（便于解释/评分）
        """
        return SkillResult.fail("自定义策略尚未实现，请在 handler.py 中完善 _user_defined_strategy 函数。")
    
    def _wrap_result(
        self,
        candidates_list,
        strategy,
        window,
        universe_size,
        liquidity_filter,
        threshold_mode,
        threshold_meta=None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> SkillResult:
        base_extra = {
            "strategy": strategy,
            "window": int(window),
            "liquidity_filter": liquidity_filter,
            "threshold_mode": threshold_mode,
            "threshold_meta": threshold_meta,
            "universe_size": universe_size,
            "score_scale": "percentile_0_100",
        }

        objs: List[EtfCandidate] = []
        for item in candidates_list:
            item_extra = item.get("extra") if isinstance(item, dict) else None
            merged_extra = dict(base_extra)
            if isinstance(item_extra, dict):
                merged_extra.update(item_extra)

            objs.append(
                EtfCandidate(
                    symbol=str(item["symbol"]),
                    score=float(item["score"]),
                    reason=str(item["reason"]),
                    source_skill="quantitative_sniper",
                    extra=merged_extra,
                )
            )

        scope = "全市场" if universe_size is None else f"Pool({universe_size})"
        if not objs:
            insight = f"[{scope}] {strategy} 未筛选出符合条件的标的。"
        else:
            insight = f"[{scope}] {strategy} | {threshold_mode} 产出 {len(objs)} 只，首选 {objs[0].symbol} ({objs[0].score:.1f})."
        if threshold_meta:
            insight += f" | Meta: {threshold_meta}"

        data = {
            "type": "EtfCandidateList",
            "items": [o.model_dump() for o in objs],
            "meta": {
                "strategy": strategy,
                "window": int(window),
                "top_k": int(len(objs)),
                "universe_size": universe_size,
                "liquidity_filter": liquidity_filter,
                "threshold_mode": threshold_mode,
                "threshold_meta": threshold_meta,
                "score_scale": "percentile_0_100",
                **(meta or {}),
            },
        }
        return SkillResult.ok(data=data, insight=insight)