from __future__ import annotations

import json
from typing import Any, List, Optional, Union, Callable, Tuple

import pandas as pd
import numpy as np

from debate_mas.protocol import SkillResult
from debate_mas.skills.base import SkillContext

REQ_COLS = ("code", "date", "close")


def normalize_universe(universe: Optional[Union[List[str], str, list[Any]]]) -> Optional[List[str]]:
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
        # EtfCandidate-like: 取 symbol 属性
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
                return normalize_universe(obj)
            if isinstance(obj, (str, int, float)):
                t = str(obj).strip()
                return [t] if t else None
        except Exception:
            pass

        parts = [p.strip() for p in s.replace("\n", ",").split(",")]
        return normalize_universe(parts)

    t = _coerce_one(universe)
    return [t] if t else None

def load_etf_daily(
    ctx: SkillContext,
    universe: Optional[Union[List[str], str, list[Any]]],
    *,
    apply_date_filter: Callable[[pd.DataFrame, str], pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[int]] | SkillResult:
    """
    负责：
    1) 从 dossier 取 etf_daily
    2) 防未来(ref_date)
    3) 列名标准化/类型清洗/必需字段检查
    4) universe 过滤
    返回：(df, universe_size) 或 SkillResult.fail(...)
    """
    df = ctx.dossier.get_table("etf_daily")
    if df is None or df.empty:
        return SkillResult.fail("案卷中找不到 'etf_daily' 数据。")

    # 时间切片（防未来函数）
    df = ctx.skill.apply_date_filter(df, ctx.ref_date) if hasattr(ctx, "skill") else df  # 你也可以把 apply_date_filter 传进来
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

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        return SkillResult.fail(f"行情表缺失必需字段: {missing}（要求至少包含 code/date/close）。")

    df = df.dropna(subset=list(REQ_COLS))
    if df.empty:
        return SkillResult.fail("清洗后数据为空（code/date/close缺失）。")
        
    #Universe 过滤 (Universe Filtering)
    universe_list = normalize_universe(universe)
    universe_set = set(universe_list) if universe_list else None
    if universe_set:
        df = df[df["code"].isin(universe_set)].copy()
        if df.empty:
            return SkillResult.fail(
                f"universe 过滤后为空：传入 {len(universe_set)} 个代码，但行情表无匹配。"
            )
    universe_size = len(universe_set) if universe_set else None
    return df, universe_size