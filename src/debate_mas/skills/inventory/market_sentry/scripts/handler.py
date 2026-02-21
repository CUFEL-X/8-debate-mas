from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from debate_mas.protocol import EtfRiskReport, SkillResult
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

class SkillHandler(BaseFinanceSkill):
    """
    [Auditor] 盘面哨兵 - 核心执行脚本
    职责：基于日线行情，检查流动性(Liquidity)和波动率(Volatility)
    """
    # 常量集中
    SKILL_NAME = "market_sentry"  
    TABLE_DAILY = "etf_daily"

    def execute(self, 
                ctx: SkillContext, 
                symbols: List[str] = None, 
                min_amount: float = 2000,   
                vol_threshold: float = 0.02, 
                window: int = 20) -> SkillResult:
        # --- 1. 数据加载 ---
        if not symbols: 
            return SkillResult.fail("必须提供待审计的 symbols 列表")
        
        str_symbols = [str(s) for s in symbols]

        df = ctx.dossier.get_table(self.TABLE_DAILY)
        if df is None or df.empty:
            return SkillResult.fail("案卷中缺失 'etf_daily' 行情表，无法审计。")

        df = self.apply_date_filter(df, ctx.ref_date)
        if df.empty:
            return SkillResult.fail(f"截止 {ctx.ref_date} 无可用行情数据。")

        df = df.copy()
        if "code" in df.columns:
            df["code"] = df["code"].astype(str)

        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        if "close" in df.columns:
            df["close"] = pd.to_numeric(df["close"], errors="coerce")

        df_subset = df[df["code"].isin(str_symbols)].copy()

        date_col = self._infer_date_col(df_subset)
        if not date_col:
            return SkillResult.fail("行情表缺失 date/data/tradingdate 列，无法进行时间序审计。")

        df_subset[date_col] = pd.to_datetime(df_subset[date_col], errors="coerce")
        df_subset = df_subset.dropna(subset=[date_col]).sort_values(date_col)

        # --- 2. 逐个审计 ---
        reports: List[EtfRiskReport] = []
        window = int(window)
        
        for sym in str_symbols:
            single_df = df_subset[df_subset['code'] == sym]
            
            # 如果数据太少 
            if len(single_df) < max(5, window // 2):
                reports.append(self._create_report(sym, risk_score=100.0, reason=f"数据严重不足 (仅{len(single_df)}天)"))
                continue
                
            # A. 流动性检查
            liq_pass, liq_val, liq_msg = self._check_liquidity(single_df, window, min_amount)
            # B. 波动率检查
            vol_pass, vol_val, vol_msg = self._check_volatility(single_df, window, vol_threshold)
            # C. 用户自定义检查 (扩展接口)
            custom_pass, custom_msg = self._user_defined_check(single_df)
            
            # --- 3. 综合判定 --- 
            risk_msgs = []
            if not liq_pass: risk_msgs.append(liq_msg)
            if not vol_pass: risk_msgs.append(vol_msg)
            if not custom_pass: risk_msgs.append(custom_msg)
            
            # 评分逻辑 
            risk_score = 0
            if not liq_pass: risk_score += 60  # 流动性枯竭：直接死刑
            if not vol_pass: risk_score += 40  # 波动率过大：严重警告 (接近死刑)
            
            status_label = self._status_label(risk_score)

            final_notes = [status_label] + risk_msgs
            
            report = EtfRiskReport(
                symbol=sym,
                risk_score=float(risk_score),
                liquidity_flag="ok" if liq_pass else "illiquid",
                sentiment_flag="normal", 
                notes=final_notes
            )
            reports.append(report)

        # --- 4. 汇总结果 ---
        pass_count = sum(1 for r in reports if r.risk_score < 50)
        insight = f"审计完成: {len(symbols)} 只标的，{pass_count} 只通过风控，{len(reports)-pass_count} 只被标记为高风险。"
        
        data = {
            "type": "EtfRiskReportList",
            "items": [r.model_dump() for r in reports],
            "meta": {
                "ref_date": ctx.ref_date,
                "agent_role": ctx.agent_role,
                "symbols_n": len(str_symbols),
                "window": int(window),
                "min_amount": float(min_amount),
                "vol_threshold": float(vol_threshold),
            },
        }
        return SkillResult.ok(data=data, insight=insight)

    # =========================
    # 审计逻辑细节
    # =========================
    def _check_liquidity(self, df, window, threshold):
        """检查 N 日均成交额"""
        if "amount" not in df.columns:
            return False, 0.0, "缺失 amount 列"

        recent = df["amount"].tail(int(window))
        avg_amount = float(recent.mean())
            
        if pd.isna(avg_amount):
            return False, 0.0, "流动性数据缺失 (NaN)"
        if avg_amount < float(threshold):
            return False, avg_amount, f"流动性枯竭 (均额 {int(avg_amount)} < 阈值 {threshold})"

        return True, avg_amount, ""

    def _check_volatility(self, df, window, threshold):
        """
        检查日收益率波动（非对称风控）
        - 总体波动率达标：PASS
        - 总体波动率超标但下跌日比例低：可豁免
        - 关注下行波动率：防“恶性下跌”
        """
        if "close" not in df.columns:
            return False, float("nan"), "缺失 close 列"

        pct_change = df["close"].pct_change().tail(int(window))
        std_dev = float(pct_change.std())

        if pd.isna(std_dev):
            return False, float("nan"), "波动率无法计算 (数据不足)"

        if std_dev <= float(threshold):
            return True, std_dev, ""

        negative_rets = pct_change[pct_change < 0]
        down_ratio = len(negative_rets) / max(1, len(pct_change))
        if down_ratio < 0.15:
            return True, std_dev, f"波动虽高({std_dev:.1%})但下跌日少({down_ratio:.0%}<15%)"

        downside_vol = float(negative_rets.std())
        if pd.isna(downside_vol) or downside_vol > float(threshold):
            if pd.isna(downside_vol):
                max_drop = float(negative_rets.min()) if not negative_rets.empty else 0.0
                if max_drop < -float(threshold) * 2:
                    return False, std_dev, f"单日暴跌 ({max_drop:.1%})"
                return True, std_dev, "波动主要来自上涨"

            return False, downside_vol, f"下行波动剧烈 ({downside_vol:.1%} > {threshold:.0%})"

        return True, std_dev, f"波动较高({std_dev:.1%})但下行风险可控"

    def _user_defined_check(self, df):
        """
        [Student TODO]
        输入: df 为单个 symbol 的时间序列（已按 date 排序），至少含 close/amount（若存在）
        输出: (pass_flag: bool, msg: str)；若 pass_flag=False，msg 会写入 EtfRiskReport.notes
        验收: notes 出现你的自定义告警；risk_score 仍由流动性/波动率主规则决定（本钩子仅补充文本提示）
        例如：检查是否连续跌停、检查换手率是否过高
        """
        return True, ""

    # =========================
    # 小工具
    # =========================
    def _infer_date_col(self, df: pd.DataFrame) -> Optional[str]:
        return next((c for c in df.columns if c.lower() in ["date", "data", "tradingdate"]), None)

    def _create_report(self, sym: str, risk_score: float, reason: str = "") -> EtfRiskReport:
        return EtfRiskReport(
            symbol=str(sym),
            risk_score=float(risk_score),
            notes=["[REJECT]", str(reason) if reason else "数据不足/审计失败，默认否决"],
        )
    
    def _status_label(self, risk_score: float) -> str:
        if float(risk_score) >= 50:
            return "[REJECT]"
        if float(risk_score) > 0:
            return "[WARNING]"
        return "[PASS]"