from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

import pandas as pd

from debate_mas.protocol import EtfRiskReport, SkillResult
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

class SkillHandler(BaseFinanceSkill):
    """
    [Auditor] 取证侦探 - 核心执行脚本
    职责：结构性审计 (Profile) + 舆情取证 (News)
    """
    # 常量集中
    SKILL_NAME = "forensic_detective"
    TABLE_BASIC = "etf_basic"
    TABLE_NEWS = "csrc"
    # 负面语义词典 
    NEGATIVE_TERMS = [
        "处罚", "违规", "警示", "立案", "调查", 
        "罚款", "整改", "责令", "警告", "通报", 
        "暂停", "退市", "造假", "虚假"
    ]
    def execute(
        self,
        ctx: SkillContext,
        symbols: Optional[List[str]] = None,
        max_fee: float = 0.5,   # 管理费 > 0.5% 报警
        min_days: int = 60,     # 成立/上市 < 60 天 报警
        lookback: int = 90,     # 舆情回溯天数
    ) -> SkillResult:
        
        # --- 1. 数据加载 ---
        if not symbols: 
            return SkillResult.fail("必须提供 symbols")
        
        str_symbols = [str(s) for s in symbols]
        
        df_basic = ctx.dossier.get_table(self.TABLE_BASIC)
        df_news = ctx.dossier.get_table(self.TABLE_NEWS)
        
        if df_basic is None:
            return SkillResult.fail(f"案卷中缺失 '{self.TABLE_BASIC}' 表，无法查证档案。")
            
        # 预处理 Basic 表
        df_basic = df_basic.copy()
        if "code" in df_basic.columns:
            df_basic["code"] = df_basic["code"].astype(str)

        target_basic = df_basic[df_basic["code"].isin(str_symbols)].copy()

        if df_news is not None and not df_news.empty:
            df_news = self.apply_date_filter(df_news, ctx.ref_date)

        ref_date = pd.to_datetime(ctx.ref_date) if ctx.ref_date else pd.Timestamp.now()
        
        # --- 2. 逐个侦查 ---
        reports: List[EtfRiskReport] = []

        for sym in str_symbols:
            row_df = target_basic[target_basic["code"] == sym]
            if row_df.empty:
                reports.append(self._create_report(sym, 100, ["[REJECT]", "档案缺失 (Basic表中未找到)"]))
                continue

            row = row_df.iloc[0]
            risk_score = 0
            risk_msgs: List[str] = []
            
            # === A. 结构性审计 (Check Profile) ===
            profile_score, profile_msgs = self._check_profile_risk(row, ref_date, max_fee, min_days)
            risk_score += profile_score
            risk_msgs.extend(profile_msgs)
            
            # === B. 舆情取证 (Audit News) ===
            news_score, news_msgs = 0, []
            if df_news is not None and not df_news.empty:
                news_score, news_msgs = self._audit_news(sym, row, df_news, ref_date, lookback)
                risk_score += news_score
                risk_msgs.extend(news_msgs)
            
            # === C. [扩展] 成分穿透 (Constituents) ===
            const_score, const_msgs = self._scan_constituents(sym)
            risk_score += const_score
            risk_msgs.extend(const_msgs)
            
            # === D. [扩展] 用户自定义逻辑 (User Defined) ===
            # 这里是完全开放的 Hook
            custom_score, custom_msgs = self._user_defined_check(sym, row)
            risk_score += custom_score
            risk_msgs.extend(custom_msgs)
            
            # === E. 结案报告 ===
            status = self._status_label(risk_score)
            final_notes = [status] + risk_msgs

            report = EtfRiskReport(
                symbol=sym,
                risk_score=float(min(100, risk_score)),
                liquidity_flag="ok",
                sentiment_flag="negative" if news_score > 0 else "normal",
                notes=final_notes,
            )
            reports.append(report)
            
        # 统计
        pass_count = sum(1 for r in reports if r.risk_score < 50)
        insight = f"取证完成: {len(symbols)} 只标的，{pass_count} 只结构合规，{len(reports)-pass_count} 只存在结构性隐患或舆情风险。"

        data = {
            "type": "EtfRiskReportList",
            "items": [r.model_dump() for r in reports],
            "meta": {
                "ref_date": ctx.ref_date,
                "agent_role": ctx.agent_role,
                "symbols_n": len(str_symbols),
                "max_fee": float(max_fee),
                "min_days": int(min_days),
                "lookback": int(lookback),
                "news_table": "csrc",
            },
        }
        return SkillResult.ok(data=data, insight=insight)

    # =========================
    # 核心审计逻辑
    # =========================
    def _check_profile_risk(self, row: pd.Series, ref_date: pd.Timestamp, max_fee: float, min_days: int):
        """
        结构性陷阱排查：费率刺客 + 次新不稳定
        Return: (score, [msgs])
        """
        score = 0
        msgs: List[str] = []
        
        # 1. 费率刺客 (Fee Assassin)
        raw_fee = str(row.get("mgt_fee", "0"))
        try:
            fee_val = float(raw_fee.replace("%", ""))
            # 兼容：有些数据用小数（0.005=0.5%）
            if 0 < fee_val <= 0.05:
                fee_val = fee_val * 100
        except Exception:
            fee_val = 0.0

        if fee_val > float(max_fee):
            score += 20
            msgs.append(f"费率刺客 (管理费 {fee_val}% > {max_fee}%)")
            
        # 2. 次新风险 (Greenhorn Risk)
        check_date_str = str(row.get("list_date", "") or "")
        date_type = "上市"
        if check_date_str in ("nan", "", "None"):
            check_date_str = str(row.get("setup_date", "") or "")
            date_type = "成立"

        try:
            check_dt = pd.to_datetime(check_date_str, errors="raise")
            age_days = (ref_date - check_dt).days
            if age_days < int(min_days):
                score += 10
                msgs.append(f"次新风险 ({date_type}仅 {age_days} 天 < {min_days} 天)")
        except Exception:
            pass

        return score, msgs

    def _audit_news(
        self,
        sym: str,
        row: pd.Series,
        df_news: pd.DataFrame,
        ref_date: pd.Timestamp,
        lookback: int,
    ) -> tuple[int, List[str]]:
        """
        舆情取证：在监管数据中搜索 ETF 名称/代码，并命中负面词
        Return: (score, msgs)
        """
        score = 0
        msgs = []
        
        # 1. 确定时间窗口
        start_date = ref_date - timedelta(days=int(lookback))

        date_col = next((c for c in df_news.columns if c.lower() in ["date", "time", "pub_date"]), None)
        if not date_col:
            return 0, []

        try:
            news_subset = df_news.copy()
            news_subset[date_col] = pd.to_datetime(news_subset[date_col], errors="coerce")
            news_subset = news_subset.dropna(subset=[date_col])
            news_subset = news_subset[(news_subset[date_col] >= start_date) & (news_subset[date_col] <= ref_date)]
        except Exception:
            return 0, []
            
        if news_subset.empty: 
            return 0, []
        
        # 2. 构造搜索关键词
        # 搜索：ETF简称 (cname)
        # 进阶(TODO): 搜索基金公司 (management)、基金经理
        keywords: List[str] = []
        if sym:
            keywords.append(sym)

        cname = str(row.get("cname", "") or "")
        if cname and cname not in ("nan", "None") and len(cname) > 2:
            keywords.append(cname)

        if not keywords:
            return 0, []

        text_cols = [c for c in news_subset.columns if c.lower() in ["title", "content", "summary"]]
        if not text_cols:
            return 0, []
        
        hit_record = ""
        for _, news_row in news_subset.iterrows():
            combined_text = " ".join([str(news_row.get(c, "")) for c in text_cols])

            hit_kw = next((kw for kw in keywords if kw in combined_text), "")
            if not hit_kw:
                continue

            if any(neg in combined_text for neg in self.NEGATIVE_TERMS):
                title = str(news_row.get("title", "监管公告"))
                hit_record = f"{hit_kw}涉及{title}"
                break

        if hit_record:
            return 50, [f"监管舆情命中 ({hit_record})"]

        return 0, []

    # =========================
    # 模板/扩展接口
    # =========================
    def _scan_constituents(self, sym: str):
        """
        [Template] 成分股穿透
        TODO: 需要 index_weight.csv 数据支持
        """
        return 0, []


    def _user_defined_check(self, sym: str, row: pd.Series):
        """
        [TODO]
        输入: sym 为 ETF 代码字符串；row 为 df_basic 中该 ETF 的一行（含 cname/mgt_fee/list_date/setup_date 等字段）
        输出: (score_delta: int, msgs: List[str])；score_delta 会累加到 risk_score，msgs 会进入 notes
        验收: 在 EtfRiskReport.notes 中看到你的自定义提示；risk_score 变化能触发 [PASS]/[WARNING]/[REJECT]
        """
        return 0, []

    # =========================
    # 小工具
    # =========================
    def _status_label(self, risk_score: float) -> str:
        if float(risk_score) >= 50:
            return "[REJECT]"
        if float(risk_score) > 0:
            return "[WARNING]"
        return "[PASS]"

    def _create_report(self, sym: str, risk_score: float, notes: List[str]) -> EtfRiskReport:
        return EtfRiskReport(symbol=str(sym), risk_score=float(risk_score), notes=notes)