# core/state.py
from __future__ import annotations

import json
import hashlib
from typing import Any, Dict, List, Optional, Set, TypedDict
from langchain_core.messages import BaseMessage

from debate_mas.loader.dossier import Dossier

_HISTORY_DEFAULT: Dict[str, List[Any]] = {"candidates": [], "objections": [], "diffs": [], "decisions": []}
def _ensure_history(st: "DebateState") -> None:
    """统一 history 初始化，避免每个 push_* 都写一遍 setdefault。"""
    st.setdefault("history", {"candidates": [], "objections": [], "diffs": [], "decisions": []})

def _stable_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)

def _fp(obj: Any) -> str:
    return hashlib.sha1(_stable_dumps(obj).encode("utf-8")).hexdigest()


class DebateState(TypedDict, total=False):
    # --- 输入 ---
    mission: str
    ref_date: Optional[str]
    dossier: Dossier
    dossier_view: Dict[str, Any]

    # --- 对话与轮次 ---
    messages: List[BaseMessage]
    round_idx: int

    # --- 本轮/角色控制（graph 用）---
    hunter_stop_suggest: str
    auditor_stop_suggest: str
    pm_stop_suggest: str
    _last_speaker_role: str

    # --- “最新版”结构化产物（用于 graph/judge） ---
    candidates_cur: List[Dict[str, Any]]
    objections_cur: List[Dict[str, Any]]
    diff_cur: Dict[str, Any]             
    decisions_cur: List[Dict[str, Any]]

    # --- 历史（教学/审计/渲染都很有用） ---
    history: Dict[str, List[Any]]        

    # --- 收敛状态 ---
    stable_rounds: int
    phase: str                          

    # --- 兼容旧字段（尽量不断你现有模块） ---
    candidates: List[Dict[str, Any]]
    risk_reports: List[Dict[str, Any]]
    decisions: List[Dict[str, Any]]

    # --- 可审计留痕 ---
    tool_trace: List[Dict[str, Any]]
    stop_reason: Optional[str]
    artifacts: Optional[Dict[str, str]]
    tool_cache: Dict[str, Any]

    # --- Tool Guard（每轮重置）---
    _round_tool_calls: Dict[str, int]
    _round_tool_calls_ok: Dict[str, int]
    _round_fingerprints: Set[str]
    _round_guard_denied: bool

    # --- NEED_EVIDENCE 协议驱动（跨轮控制）---
    _need_evidence: bool                       
    _need_evidence_symbols: List[str]         
    _need_evidence_actions: List[str]       
    _round_missing_evidence: bool           

    # --- MIN_CANDIDATES 跨轮控制 ---
    _need_more_candidates: bool
    _need_more_candidates_min: int
    _need_more_candidates_have: int
    _need_more_candidates_missing: int
    _need_more_candidates_reason: str

    # --- 稳定性指纹 ---
    _last_stable_fp: str
    _force_hunter_tool: bool

    # --- Two-stage pipeline ---
    _hunter_pipeline_stage: str  # "recall" | "rerank"
    survivor_universe: List[str]

    _need_recall_diversity: bool
    _need_recall_diversity_reason: str
    _need_rerank_composite: bool
    _need_rerank_composite_reason: str

    # --- “策略使用记录”硬状态 ---
    _hunter_round_sniper_strategies: List[str]

def push_candidates_merge(st: DebateState, incoming: List[Dict[str, Any]]) -> None:
    """Hunter 只能“补充/修订”，不能“偷偷删池子”。"""
    prev_items = st.get("candidates_cur", []) or []
    by_sym: Dict[str, Dict[str, Any]] = {}

    for it in prev_items:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol", "") or "").strip()
        if sym:
            by_sym[sym] = dict(it)

    for it in incoming or []:
        if not isinstance(it, dict):
            continue
        sym = str(it.get("symbol", "") or "").strip()
        if not sym:
            continue
        by_sym[sym] = dict(it)

    merged = list(by_sym.values())
    merged.sort(key=lambda x: (float(x.get("score", 0.0) or 0.0), str(x.get("symbol", ""))), reverse=True)

    st["candidates_cur"] = merged
    st["candidates"] = merged 
    _ensure_history(st)
    st["history"]["candidates"].append({"round": int(st.get("round_idx", 0) or 0), "items": st["candidates_cur"]})

# init/reset
def init_state(
    mission: str,
    dossier: Dossier,
    ref_date: Optional[str] = None,
    messages: Optional[List[BaseMessage]] = None,
) -> DebateState:
    st: DebateState = {
        "mission": mission,
        "ref_date": ref_date,
        "dossier": dossier,
        "dossier_view": dossier.frozen_view() if hasattr(dossier, "frozen_view") else {},
        "messages": messages or [],
        "round_idx": 0,
        "candidates_cur": [],
        "objections_cur": [],
        "diff_cur": {},
        "decisions_cur": [],
        "history": {"candidates": [], "objections": [], "diffs": [], "decisions": []},
        "stable_rounds": 0,
        "phase": "init",
        "candidates": [],
        "risk_reports": [],
        "decisions": [],
        "tool_trace": [],
        "stop_reason": None,
        "artifacts": None,
        "tool_cache": {},
        "_need_evidence": False,
        "_need_evidence_symbols": [],
        "_need_evidence_actions": [],
        "_round_missing_evidence": False,
        "_need_more_candidates": False,
        "_need_more_candidates_min": 0,
        "_need_more_candidates_have": 0,
        "_need_more_candidates_missing": 0,
        "_need_more_candidates_reason": "",
        "_last_stable_fp": "",
        "_force_hunter_tool": False,
        "_hunter_pipeline_stage": "recall",
        "survivor_universe": [],
        "_need_recall_diversity": False,
        "_need_recall_diversity_reason": "",
        "_need_rerank_composite": False,
        "_need_rerank_composite_reason": "",
        "_hunter_round_sniper_strategies": [],
    }
    reset_round_runtime(st)
    return st


def reset_round_runtime(st: DebateState) -> None:
    st["_round_tool_calls"] = {"hunter": 0, "auditor": 0, "pm": 0}
    st["_round_tool_calls_ok"] = {"hunter": 0, "auditor": 0, "pm": 0}
    st["_round_fingerprints"] = set()
    st["_round_guard_denied"] = False
    st["_round_missing_evidence"] = False


def mark_guard_denied(st: DebateState) -> None:
    st["_round_guard_denied"] = True


def bump_round(st: DebateState) -> None:
    st["round_idx"] = int(st.get("round_idx", 0) or 0) + 1
    reset_round_runtime(st)

# MIN_CANDIDATES helpers
def set_need_more_candidates(
    st: DebateState,
    *,
    min_required: int,
    have: int,
    missing: int,
    reason: str,
) -> None:
    st["_need_more_candidates"] = True
    st["_need_more_candidates_min"] = int(min_required or 0)
    st["_need_more_candidates_have"] = int(have or 0)
    st["_need_more_candidates_missing"] = int(missing or 0)
    st["_need_more_candidates_reason"] = str(reason or "").strip()


def clear_need_more_candidates(st: DebateState) -> None:
    st["_need_more_candidates"] = False
    st["_need_more_candidates_min"] = 0
    st["_need_more_candidates_have"] = 0
    st["_need_more_candidates_missing"] = 0
    st["_need_more_candidates_reason"] = ""

# push helpers
def push_candidates(st: DebateState, items: List[Dict[str, Any]]) -> None:
    st["candidates_cur"] = list(items or [])
    st["candidates"] = st["candidates_cur"] 
    st.setdefault("history", {"candidates": [], "objections": [], "diffs": [], "decisions": []})
    st["history"]["candidates"].append({"round": int(st.get("round_idx", 0) or 0), "items": st["candidates_cur"]})


def push_objections(st: DebateState, items: List[Dict[str, Any]]) -> None:
    st["objections_cur"] = list(items or [])
    _ensure_history(st)
    st["history"]["objections"].append({"round": int(st.get("round_idx", 0) or 0), "items": st["objections_cur"]})


def push_diff(st: DebateState, diff_obj: Dict[str, Any]) -> None:
    st["diff_cur"] = dict(diff_obj or {})
    st.setdefault("history", {"candidates": [], "objections": [], "diffs": [], "decisions": []})
    st["history"]["diffs"].append({"round": int(st.get("round_idx", 0) or 0), "diff": st["diff_cur"]})


def push_decisions(st: DebateState, items: List[Dict[str, Any]]) -> None:
    st["decisions_cur"] = list(items or [])
    st["decisions"] = st["decisions_cur"] 
    st.setdefault("history", {"candidates": [], "objections": [], "diffs": [], "decisions": []})
    st["history"]["decisions"].append({"round": int(st.get("round_idx", 0) or 0), "items": st["decisions_cur"]})

# 收敛：稳定性指纹
def bump_stable_rounds(st: DebateState, *, reset_if_changed: bool = True) -> int:
    """用“候选 + objections + diff”生成指纹；若不变则 stable_rounds += 1。"""
    cur = {
        "candidates": st.get("candidates_cur", []) or [],
        "objections": st.get("objections_cur", []) or [],
        "diff": st.get("diff_cur", {}) or {},
    }
    cur_fp = _fp(cur)
    prev_fp = str(st.get("_last_stable_fp", "") or "")

    if prev_fp and cur_fp == prev_fp:
        st["stable_rounds"] = int(st.get("stable_rounds", 0) or 0) + 1
    else:
        if reset_if_changed:
            st["stable_rounds"] = 0
        st["_last_stable_fp"] = cur_fp

    return int(st.get("stable_rounds", 0) or 0)
