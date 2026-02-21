# skills/inventory/portfolio_allocator/scripts/handler.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from debate_mas.protocol import (
    DecisionAction,
    EtfCandidate,
    EtfDecision,
    EtfRiskReport,
    SkillResult,
)
from debate_mas.skills.base import BaseFinanceSkill, SkillContext

# ============================
# 显式 args_schema：宽进（dict），内部再强转模型
# ============================
class PortfolioAllocatorArgs(BaseModel):
    """
    输入就按“字典列表”写即可（可填空式复现/拓展）。
    内部仍会把 dict 转成 EtfCandidate / EtfRiskReport 做强校验。
    """
    model_config = ConfigDict(extra="allow")  

    candidates: List[Dict[str, Any]] | None = Field(default=None, description="Hunter 输出候选列表（dict）")
    risk_reports: List[Dict[str, Any]] | None = Field(default=None, description="Auditor 输出风险报告列表（dict）")

    method: str = "linear_voting"
    sizing_method: str = "kelly"
    risk_penalty: float = 1.0,      # 风险厌恶系数
    max_position: float = 0.2,      # 单标的最大仓位
    buy_threshold: float = 50.0,    # BUY 硬门槛
    target_exposure: float = 1.0,   # 总仓位目标
    max_buys: int = 10

    @model_validator(mode="before")
    @classmethod
    def _coerce_inputs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        def _normalize(v: Any, *, field_name: str) -> Any:
            if v is None:
                return None

            # 1) JSON string -> python object
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    return None
                try:
                    v = json.loads(s)
                except Exception as e:
                    raise ValueError(f"{field_name} 解析失败：传入的是字符串但不是合法 JSON：{e}")

            # 2) 包了一层 {"type": "...", "items": [...]}
            if isinstance(v, dict) and "items" in v and isinstance(v["items"], list):
                v = v["items"]

            # 3) 单个 dict -> 包成 list[dict]
            if isinstance(v, dict):
                return [v]

            # 4) list -> 保留
            if isinstance(v, list):
                return v

            raise ValueError(f"{field_name} 输入类型不支持：{type(v)}（期望 list/dict/JSON-string）")

        if "candidates" in data:
            data["candidates"] = _normalize(data.get("candidates"), field_name="candidates")
        if "risk_reports" in data:
            data["risk_reports"] = _normalize(data.get("risk_reports"), field_name="risk_reports")

        return data

class SkillHandler(BaseFinanceSkill):
    """
    [PM] 资产配置官 - 核心决策脚本
    职责：多空信号融合 + 仓位管理
    """
    SKILL_NAME = "portfolio_allocator"
    OUTPUT_TYPE = "EtfDecisionList"

    args_schema = PortfolioAllocatorArgs

    # ------------------------
    # 输入强转小工具（保持跨链路稳健）
    # ------------------------
    def _coerce_candidates(self, candidates: Optional[List[Union[EtfCandidate, Dict[str, Any], str]]]) -> List[EtfCandidate]:
        out: List[EtfCandidate] = []
        for x in candidates or []:
            if isinstance(x, EtfCandidate):
                out.append(x)
            elif isinstance(x, str):
                try:
                    obj = json.loads(x)
                    if isinstance(obj, dict):
                        out.append(EtfCandidate(**obj))
                    elif isinstance(obj, list):
                        out.extend(EtfCandidate(**d) for d in obj if isinstance(d, dict))
                except Exception:
                    pass
            elif isinstance(x, dict):
                out.append(EtfCandidate(**x))
        return out

    def _coerce_risk_reports(self, risk_reports: Optional[List[Union[EtfRiskReport, Dict[str, Any], str]]]) -> List[EtfRiskReport]:
        out: List[EtfRiskReport] = []
        for x in risk_reports or []:
            if isinstance(x, EtfRiskReport):
                out.append(x)
            elif isinstance(x, str):
                try:
                    obj = json.loads(x)
                    if isinstance(obj, dict):
                        out.append(EtfRiskReport(**obj))
                    elif isinstance(obj, list):
                        out.extend(EtfRiskReport(**d) for d in obj if isinstance(d, dict))
                except Exception:
                    pass
            elif isinstance(x, dict):
                out.append(EtfRiskReport(**x))
        return out
    
    def execute(
        self,
        ctx: SkillContext,
        candidates: Optional[list[Union[EtfCandidate, Dict[str, Any]]]] = None,  
        risk_reports: Optional[list[Union[EtfRiskReport, Dict[str, Any]]]] = None, 
        method: str = "linear_voting",
        sizing_method: str = "kelly",
        risk_penalty: float = 1.0,   
        max_position: float = 0.2,   
        buy_threshold: float = 50.0,
        target_exposure: float = 1.0, 
        max_buys: int = 10,
    ) -> SkillResult:
        try:
            candidates_m = self._coerce_candidates(candidates) 
            risk_reports_m = self._coerce_risk_reports(risk_reports)  
        except Exception as e:
            return SkillResult.fail(f"输入解析失败（dict->模型强转失败）: {e}")  
        
        # --- 1) 输入检查 ---
        if not candidates_m:
            return SkillResult.fail("没有 Hunter 的候选标的，无法决策。")

        # --- 2) 候选去重：同一 symbol 取最高分 ---
        best: Dict[str, EtfCandidate] = {}
        for c in candidates_m:
            sym = str(c.symbol)
            if sym not in best or float(c.score) > float(best[sym].score):
                best[sym] = c
        unique_candidates = list(best.values())

        # --- 3) 风险映射（Auditor -> symbol -> report） ---
        risk_map: Dict[str, EtfRiskReport] = {str(r.symbol): r for r in (risk_reports_m or [])}

        decisions: List[EtfDecision] = []

        for cand in unique_candidates:
            sym = str(cand.symbol)
            rpt = risk_map.get(sym)

            risk_score = float(rpt.risk_score) if rpt else 0.0
            risk_notes = list(rpt.notes) if (rpt and rpt.notes) else []

            # --- 4) Hard Veto：系统级规则 ---
            if risk_score >= float(buy_threshold):
                decisions.append(
                    EtfDecision(
                        symbol=sym,
                        action=DecisionAction.REJECT,         
                        weight=0.0,
                        final_score=0.0,
                        key_reasons=[f"Auditor 否决 (风险分 {risk_score:.1f} >= 50)"],
                        risk_warnings=risk_notes,
                    )
                )
                continue

            # --- 5) 多空融合评分 ---
            hunter_score = float(cand.score)
            final_score = float(self._calculate_final_score(method, hunter_score, risk_score, risk_penalty))

            # --- 6) 动作与仓位 ---
            action = DecisionAction.WATCH
            weight = 0.0

            if final_score >= float(buy_threshold):
                weight = float(
                    self._calculate_sizing(
                        method=sizing_method,
                        score=final_score,
                        max_pos=float(max_position),
                    )
                )
                action = DecisionAction.BUY

            reason_str = f"H:{hunter_score:.1f} - A:{risk_score:.1f}*k({risk_penalty:.2f}) -> Final:{final_score:.1f}"

            decisions.append(
                EtfDecision(
                    symbol=sym,
                    action=action,                         
                    weight=round(weight, 6),
                    final_score=round(final_score, 2),
                    key_reasons=[reason_str, str(cand.reason)],
                    risk_warnings=risk_notes,
                )
            )

        # --- 7) 排序（高分在前） ---
        decisions.sort(key=lambda x: float(x.final_score), reverse=True)
        
        # --- 8) TopK BUY 限制 ---
        try:
            k = int(max_buys)
        except Exception:
            k = 0
        if k > 0:
            buy_list = [d for d in decisions if d.action == DecisionAction.BUY and float(d.weight) > 0]
            if len(buy_list) > k:
                for d in buy_list[k:]:
                    d.action = DecisionAction.WATCH
                    d.weight = 0.0
                    d.key_reasons = [f"超出 Top{k} BUY 上限，改为 WATCH"] + (d.key_reasons or [])

        # --- 9) BUY 权重归一化到 target_exposure ---
        buys = [d for d in decisions if d.action == DecisionAction.BUY and float(d.weight) > 0]
        total_w = float(sum(float(d.weight) for d in buys))
        if total_w > 0 and total_w > float(target_exposure):
            scale = float(target_exposure) / total_w
            for d in buys:
                d.weight = round(float(d.weight) * scale, 6)

        # --- 10) 统计 ---
        n_buy = sum(1 for d in decisions if d.action == DecisionAction.BUY)
        n_watch = sum(1 for d in decisions if d.action == DecisionAction.WATCH)
        n_reject = sum(1 for d in decisions if d.action == DecisionAction.REJECT)

        insight = f"决策完成: 买入 {n_buy} 只, 观望 {n_watch} 只, 否决 {n_reject} 只。"

        data: Dict[str, Any] = {
            "type": self.OUTPUT_TYPE,
            "items": [d.model_dump() for d in decisions],
            "meta": {
                "ref_date": ctx.ref_date,
                "agent_role": ctx.agent_role,
                "buy_threshold": float(buy_threshold),
                "target_exposure": float(target_exposure),
                "max_position": float(max_position),
                "risk_penalty": float(risk_penalty),
                "method": method,
                "sizing_method": sizing_method,
                "max_buys": int(max_buys),
            },
            "summary": {
                "n_buy": int(n_buy),
                "n_watch": int(n_watch),
                "n_reject": int(n_reject),
                "total_buy_weight": round(sum(float(d.weight) for d in buys), 6),
            },
        }

        return SkillResult.ok(data=data, insight=insight)

    # ================= 🧠 决策引擎 =================
    def _calculate_final_score(self, method: str, h_score: float, a_risk: float, penalty: float) -> float:
        """
        最简单可解释融合：Final = max(0, HunterScore - RiskScore * penalty)
        这是“能讲清楚”的 baseline，学生可以替换为更高级模型
        """
        h_score = float(h_score)
        a_risk = float(a_risk)
        penalty = float(penalty)
        return max(0.0, h_score - a_risk * penalty)

    def _calculate_sizing(self, method: str, score: float, max_pos: float) -> float:
        """Kelly 思想 + 边界控制（不产生负权重、不爆仓）"""
        score = float(score)
        max_pos = float(max_pos)

        if method == "kelly":
            # 60->0.51, 100->0.65（更稳，避免过度激进）
            win_rate = 0.51 + (score - 60.0) * (0.14 / 40.0)
            win_rate = float(np.clip(win_rate, 0.51, 0.65))

            odds = 2.0
            kelly = (win_rate * odds - (1.0 - win_rate)) / odds
            raw = max(0.0, float(kelly))
        else:
            raw = 0.1

        return min(raw, max_pos)
