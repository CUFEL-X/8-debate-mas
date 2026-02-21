# protocol/schema.py
"""
协议定义层 (Protocol Schema)
定义全系统通用的数据交互标准：
- Layer 3 (Skills) 生产这些数据
- Layer 2 (Core)   消费/合并这些数据
- Layer 4 (Output) 展示/落盘这些数据
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# 0) 统一枚举（跨层对齐：Persona / Core / Skills / Renderer）
# =============================================================================
class DecisionAction(str, Enum):
    BUY = "BUY"
    WATCH = "WATCH"
    REJECT = "REJECT"


# =============================================================================
# 1) 通用返回：SkillResult
# =============================================================================
class SkillResult(BaseModel):
    """
    [通用协议] 技能执行结果标准

    Teaching Notes:
    - success/insight: 给人看
    - data: 给机器消费（可为 List[BusinessObject] 或 Dict 或 None）
    - visuals: 图表路径留痕（可为空）
    - error_msg: 失败原因（只在 success=False 时应出现）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="执行是否成功")
    data: Any = Field(default=None, description="结构化返回（对象/列表/字典均可）")
    insight: str = Field(..., description="自然语言结论")
    visuals: List[str] = Field(default_factory=list, description="生成的图表路径列表")
    error_msg: Optional[str] = Field(default=None, description="错误信息（仅失败时）")

    @classmethod
    def ok(cls, data: Any = None, insight: str = "", visuals: Optional[List[str]] = None) -> "SkillResult":
        """快捷构造成功结果（data 可以是 List / Dict / None）"""
        return cls(
            success=True,
            data=data,
            insight=insight or "",
            visuals=visuals or [],
            error_msg=None
        )

    @classmethod
    def fail(cls, error_msg: str, data: Any = None) -> "SkillResult":
        """快捷构造失败结果（允许把 debug data 带回）"""
        return cls(
            success=False,
            data=data,
            insight=f"执行失败: {error_msg}",
            visuals=[],
            error_msg=error_msg
        )


# =============================================================================
# 2) 业务中间产物（Skills 产出 / Core 消费 / Output 展示）
# =============================================================================
class EtfCandidate(BaseModel):
    """Hunter 的产物：候选标的（可追溯）"""
    symbol: str
    score: float
    reason: str
    source_skill: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class EtfRiskReport(BaseModel):
    """Auditor 的产物：风险报告（可解释）"""
    symbol: str
    liquidity_flag: Optional[str] = None
    premium_flag: Optional[str] = None
    sentiment_flag: Optional[str] = None
    risk_score: float = 0.0
    notes: List[str] = Field(default_factory=list)


class EtfDecision(BaseModel):
    """
    PM 的产物：最终决策（可执行）
    Action 统一为 BUY/WATCH/REJECT，与 Persona 和 Renderer 对齐
    """
    symbol: str
    action: DecisionAction
    weight: float = Field(0.0, ge=0.0, le=1.0)
    final_score: float = 0.0
    key_reasons: List[str] = Field(default_factory=list)
    risk_warnings: List[str] = Field(default_factory=list)


# =============================================================================
# 3) Layer 4 交付协议（“决策备忘录”结构）——可选但强烈建议
# =============================================================================
class ToolTraceEntry(BaseModel):
    """工具调用留痕（Core 可选写入，Renderer 只负责落盘）"""
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    insight: str = ""
    error_msg: Optional[str] = None
    visuals: List[str] = Field(default_factory=list)


class DebateMeta(BaseModel):
    """
    交付物元信息（强烈建议 Core 填）
    Teaching Tip: 可以先只填 mission/ref_date/rounds，其他留空
    """
    mission: str = ""
    ref_date: Optional[str] = None
    rounds: int = 0
    stop_reason: Optional[str] = None  
    tool_trace: List[ToolTraceEntry] = Field(default_factory=list)
    dossier_meta: Dict[str, Any] = Field(default_factory=dict)  
    extras: Dict[str, Any] = Field(default_factory=dict)        # 预留扩展


class DebateLog(BaseModel):
    """最终交付 JSON 的结构化协议（机器可读、可回放）"""
    timestamp: str
    meta: DebateMeta = Field(default_factory=DebateMeta)
    decisions: List[EtfDecision] = Field(default_factory=list)
    visuals: List[str] = Field(default_factory=list)