"""
Protocol Package Initialization
将 schema.py 中的核心对象暴露到 protocol 包的顶层，
方便外部使用 'from debate_mas.protocol import EtfCandidate' 这种写法。
"""
from .schema import (
    SkillResult,
    EtfCandidate,
    EtfRiskReport,
    EtfDecision,
    DecisionAction,  
)

from .renderer import DebateRenderer

__all__ = [
    "SkillResult",
    "EtfCandidate",
    "EtfRiskReport",
    "EtfDecision",
    "DecisionAction", 
    "DebateRenderer",
]