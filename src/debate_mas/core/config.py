# core/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

# ======================= 简单工具 =======================
def _default_base_dir() -> str:
    """把 BASE_DIR 默认值计算抽成函数"""
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )


@dataclass(frozen=True)
class SystemConfig:
    """
    【核心配置中心】

    结构分层：
    A) 通用框架参数（DebateMas 通用：无论你辩论的是 ETF / 方案 / 文档审阅都适用）
    B) ETF 业务参数（与“ETF候选/风控/组合/ETF技能”强绑定）
    """

    # ============================================================
    # A) 通用框架参数（Framework-Level）
    # ============================================================
    # --- 模型与温度（LLM backend：通用） ---
    HUNTER_MODEL: str = "qwen3-max"
    PM_MODEL: str = "qwen3-max"
    AUDITOR_MODEL: str = "qwen3-max"

    ROLE_TEMPERATURE: Dict[str, float] = field(default_factory=lambda: {
        "hunter": 0.9,   # 发散找机会（通用：提案/探索角色）
        "auditor": 0.3,  # 更稳（通用：审计/质疑角色）
        "pm": 0.1,       # 严谨决策（通用：裁决/定稿角色）
    })

    # --- Token 预算（通用） ---
    MAX_TOKENS_DEFAULT: int = 3000  # 全局默认 max_tokens
    ROLE_MAX_TOKENS: Dict[str, int] = field(default_factory=lambda: {
        "hunter": 3000,
        "auditor": 3000,
        "pm": 3000,
    })

    # --- 运行与证据策略（通用） ---
    VERBOSE: bool = True  # 是否打印“增量摘要”
    ENFORCE_TOOL_ON_NEED_EVIDENCE: bool = True  # 若出现 NEED_EVIDENCE，下一轮强制补证据（通用机制）

    # --- 辩论流程控制（通用：收敛与终止） ---
    MAX_ROUNDS: int = 3            # 最多辩论几轮（硬停）
    EXIT_ON_CONSENSUS: bool = True # 双方都 STOP 则提前结束

    # --- Tool Calling 框架硬约束（通用：工具治理） ---
    ROLE_TOOL_MAX_CALLS: Dict[str, int] = field(default_factory=lambda: {
        "hunter": 4,
        "auditor": 2,
        "pm": 1,
    })
    FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND: bool = True  # 同轮同参去重/防刷

    # --- 候选数量硬约束（机制通用，但在本项目用于“候选池”） ---
    ENFORCE_MIN_CANDIDATES: bool = True
    HUNTER_MIN_CANDIDATES: int = 10

    # --- 路径配置（通用：项目路径/默认数据目录） ---
    BASE_DIR: str = field(default_factory=_default_base_dir)
    DATA_DIR: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "DATA_DIR", os.path.join(self.BASE_DIR, "data_test"))

    # ============================================================
    # B) ETF 业务参数（ETF Domain-Level）
    # ============================================================
    # --- 角色工具白名单（ETF技能名本身是业务绑定的；机制在 A4） ---
    ROLE_TOOL_ALLOWLIST: Dict[str, List[str]] = field(default_factory=lambda: {
        "hunter": [
            "theme_miner",
            "quantitative_sniper",
        ],
        "auditor": [
            "market_sentry",
            "forensic_detective",
        ],
        "pm": [
            "portfolio_allocator",
        ],
    })

    # --- Hunter 两阶段 pipeline（ETF候选生成逻辑：recall → rerank） ---
    HUNTER_DETERMINISTIC_PIPELINE: bool = True
    HUNTER_PIPELINE_MODE: str = "two_stage"  # 可扩展：未来加 "free" / "ablation"

    # Round0: 多策略召回
    HUNTER_RECALL_STRATEGIES: List[str] = field(default_factory=lambda: ["momentum", "sharpe", "reversal"])
    HUNTER_RECALL_MIN_STRATEGIES: int = 2
    HUNTER_RECALL_TOPK_PER_STRATEGY: int = 10

    # Round1+: 统一标尺 rerank
    HUNTER_RERANK_STRATEGY: str = "composite"
    HUNTER_RERANK_OUTPUT_TOPN: int = 20

    # pipeline 中“默认主策略”（用于教学/日志固定锚点）
    HUNTER_PIPELINE_SNIPER_STRATEGY: str = "momentum"

    # --- 候选融合（ETF：多来源候选合并权重） ---
    HUNTER_BLEND: Dict[str, float] = field(default_factory=lambda: {
        "theme_miner": 0.3,
        "quantitative_sniper": 0.7,
    })

    # --- 业务阈值（ETF：风控/剔除阈值） ---
    RISK_SCORE_THRESHOLD: float = 50.0

    # --- sniper / theme_miner 默认参数（ETF：因子与主题召回配置） ---
    SNIPER_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "window": 20,
        "top_k": 10,
        "min_amount": 1000.0,
        "threshold_mode": "none",
        "psr_confidence": 0.95,
    })

    SNIPER_PROFILES: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "momentum": {
            "window": 20,
            "threshold_mode": "none",
            "psr_confidence": 0.95,
        },
        "sharpe": {
            "window": 60,
            "threshold_mode": "psr",
            "psr_confidence": 0.95,
        },
        "reversal": {
            "window": 10,
            "threshold_mode": "none",
        },
        "composite": {},
    })

    SNIPER_ENFORCE: Dict[str, Any] = field(default_factory=lambda: {
        "min_amount": 1000.0,
    })

    SNIPER_LIMITS: Dict[str, Any] = field(default_factory=lambda: {
        "max_top_k": 200,
    })

    THEME_MINER_DEFAULTS: Dict[str, Any] = field(default_factory=lambda: {
        "top_k": 10,
        "days": 30,
        "top_industries": 3,
    })

    # --- Auditor 参数（ETF：风险审计工具约束） ---
    AUDITOR_MARKET_SENTRY_ENFORCE: Dict[str, Any] = field(default_factory=lambda: {
        "window": 20,
        "min_amount": 1000.0,
        "vol_threshold": 0.03,
    })

    AUDITOR_FORENSIC_DETECTIVE_ENFORCE: Dict[str, Any] = field(default_factory=lambda: {
        "lookback": 60,
        "max_fee": 0.5,
        "min_days": 60,
    })

    # --- PM 参数（ETF：组合构建与买入规则） ---
    PM_PORTFOLIO_ALLOCATOR_ENFORCE: Dict[str, Any] = field(default_factory=lambda: {
        "method": "linear_voting",
        "sizing_method": "kelly",
        "risk_penalty": 1.0,
        "max_position": 0.4,
        "buy_threshold": 60.0,
        "target_exposure": 0.95,
        "max_buys": 10,
    })

    # ============================================================
    # 调试快照
    # ============================================================
    def get_model_config(self) -> Dict[str, Any]:
        return {
            "Hunter": self.HUNTER_MODEL,
            "Auditor": self.AUDITOR_MODEL,
            "PM": self.PM_MODEL,
            "Data_Dir": self.DATA_DIR,
            "Blend": self.HUNTER_BLEND,
            "Max_Tokens": {
                "default": self.MAX_TOKENS_DEFAULT,
                "by_role": self.ROLE_MAX_TOKENS,
            },
            "Min_Candidates": {
                "enabled": self.ENFORCE_MIN_CANDIDATES,
                "min": self.HUNTER_MIN_CANDIDATES,
            },
            "Sniper": {
                "defaults": self.SNIPER_DEFAULTS,
                "profiles": self.SNIPER_PROFILES,
                "enforce": self.SNIPER_ENFORCE,
                "limits": self.SNIPER_LIMITS,
                "pipeline": {
                    "mode": self.HUNTER_PIPELINE_MODE,
                    "recall_strategies": self.HUNTER_RECALL_STRATEGIES,
                    "recall_min_strategies": self.HUNTER_RECALL_MIN_STRATEGIES,
                    "recall_topk_per_strategy": self.HUNTER_RECALL_TOPK_PER_STRATEGY,
                    "rerank_strategy": self.HUNTER_RERANK_STRATEGY,
                    "rerank_output_topn": self.HUNTER_RERANK_OUTPUT_TOPN,
                },
            },
            "Auditor_Enforce": {
                "market_sentry": self.AUDITOR_MARKET_SENTRY_ENFORCE,
                "forensic_detective": self.AUDITOR_FORENSIC_DETECTIVE_ENFORCE,
            },
            "PM_Enforce": {
                "portfolio_allocator": self.PM_PORTFOLIO_ALLOCATOR_ENFORCE,
            },
            "Max_Rounds": self.MAX_ROUNDS,
            "Tool_Allowlist": self.ROLE_TOOL_ALLOWLIST,
            "Tool_MaxCalls": self.ROLE_TOOL_MAX_CALLS,
            "Dedup_SameToolSameArgs": self.FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND,
            "Risk_Threshold": self.RISK_SCORE_THRESHOLD,
            "Hunter_Pipeline": {
                "enabled": self.HUNTER_DETERMINISTIC_PIPELINE,
                "sniper_strategy": self.HUNTER_PIPELINE_SNIPER_STRATEGY,
            },
        }


CONFIG = SystemConfig()