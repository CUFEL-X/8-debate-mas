# core/tools.py
from __future__ import annotations

import ast
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable, Set

from contextvars import ContextVar

from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from debate_mas.skills.registry import SkillRegistry
from debate_mas.skills.base import SkillContext
from debate_mas.protocol import SkillResult

from .config import CONFIG
from .state import DebateState, mark_guard_denied

# ============================================================
# SECTION 0) 类型与运行时 state 注入
# ============================================================
_CURRENT_STATE: ContextVar[Optional[DebateState]] = ContextVar("_CURRENT_STATE", default=None)
ToolRunner = Callable[[DebateState], DebateState]

def _get_runtime_state(fallback: DebateState) -> DebateState:
    """优先取 ToolNode 注入的“运行时 state”，避免闭包捕获旧 state。"""
    st = _CURRENT_STATE.get()
    return st if isinstance(st, dict) else fallback

# ============================================================
# SECTION 1) 指纹 / 稳定序列化
# ============================================================
def _json_dumps_stable(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(obj), ensure_ascii=False)

def fingerprint(tool_name: str, tool_args: Dict[str, Any]) -> str:
    raw = tool_name + "|" + _json_dumps_stable(tool_args or {})
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

# ============================================================
# SECTION 2) ctx 构建（注入 role/ref_date/dossier）
# ============================================================
def build_ctx(dossier, role: str, ref_date: Optional[str]) -> SkillContext:
    try:
        return SkillContext(dossier=dossier, agent_role=role, ref_date=ref_date)
    except Exception:
        return SkillContext.model_construct(dossier=dossier, agent_role=role, ref_date=ref_date)
    
# ============================================================
# SECTION 3) schema 过滤工具
# ============================================================
def _is_empty_value(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")

def _schema_keys_from_tool(base_tool: StructuredTool) -> Optional[Set[str]]:
    schema = getattr(base_tool, "args_schema", None)
    if schema is None:
        return None
    if hasattr(schema, "model_fields"):
        try:
            return set(schema.model_fields.keys())
        except Exception:
            return None
    if hasattr(schema, "__fields__"):
        try:
            return set(schema.__fields__.keys())  
        except Exception:
            return None
    return None

def _filter_to_schema(args: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    if schema_keys is None:
        return args
    return {k: v for k, v in (args or {}).items() if k in schema_keys}

def _fill_missing(args: Dict[str, Any], defaults: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    out = dict(args or {})
    for k, v in (defaults or {}).items():
        if schema_keys is not None and k not in schema_keys:
            continue
        if k not in out or _is_empty_value(out.get(k)):
            out[k] = v
    return out

def _force_override(args: Dict[str, Any], overrides: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    out = dict(args or {})
    for k, v in (overrides or {}).items():
        if schema_keys is not None and k not in schema_keys:
            continue
        out[k] = v
    return out

# ============================================================
# SECTION 4) policy 应用
# ============================================================
def _apply_tool_policy(tool_name: str, tool_args: Dict[str, Any], schema_keys: Optional[Set[str]]) -> Dict[str, Any]:
    args = dict(tool_args or {})

    # --- Hunter: quantitative_sniper ---
    if tool_name == "quantitative_sniper":
        # composite_weights：允许 list/str 兜底纠错 -> dict[str,float]
        if "composite_weights" in args:
            cw = args.get("composite_weights")

            if isinstance(cw, str):
                s = cw.strip()
                if s:
                    try:
                        cw = json.loads(s)
                    except Exception: pass

            if isinstance(cw, list):
                try: cw = {str(k): v for (k, v) in cw}
                except Exception: pass

            if isinstance(cw, dict):
                cw2 = {}
                for k, v in cw.items():
                    try: cw2[str(k)] = float(v)
                    except Exception: continue
                cw = cw2

            # 只有在 schema 支持时才写回，避免“未知字段”导致校验失败
            if schema_keys is None or "composite_weights" in schema_keys:
                args["composite_weights"] = cw
        # 1) strategy：优先 LLM 传；没传则用 pipeline strategy
        strategy = args.get("strategy")
        if _is_empty_value(strategy):
            strategy = getattr(CONFIG, "HUNTER_PIPELINE_SNIPER_STRATEGY", "momentum")
            if schema_keys is None or "strategy" in schema_keys:
                args["strategy"] = strategy

        # 2) defaults：只补齐缺失
        args = _fill_missing(args, getattr(CONFIG, "SNIPER_DEFAULTS", {}) or {}, schema_keys)

        # 3) profile：强控覆盖（user_defined 不覆盖）
        profiles = getattr(CONFIG, "SNIPER_PROFILES", {}) or {}
        if isinstance(strategy, str) and strategy != "user_defined":
            args = _force_override(args, profiles.get(strategy, {}) or {}, schema_keys)
        
        # 如果是 composite 策略，强制 top_k = 配置值
        strategy = args.get("strategy")
        if strategy == "composite":
            rerank_n = int(getattr(CONFIG, "HUNTER_RERANK_OUTPUT_TOPN", 30))
            args["top_k"] = rerank_n

        # 4) enforce：少量硬参数强控（top_k/min_amount）
        args = _force_override(args, getattr(CONFIG, "SNIPER_ENFORCE", {}) or {}, schema_keys)

        # 5) top_k 上限保护（但允许 rerank 用 top_k=len(universe)）
        limits = getattr(CONFIG, "SNIPER_LIMITS", {}) or {}
        max_top_k = int(limits.get("max_top_k", 200) or 200)
        if "top_k" in args:
            try:
                k = int(args.get("top_k") or 0)
                if k > max_top_k:
                    args["top_k"] = max_top_k
            except Exception: pass

        return _filter_to_schema(args, schema_keys)
    
    # --- Auditor: market_sentry ---
    if tool_name == "market_sentry":
        args = _force_override(args, getattr(CONFIG, "AUDITOR_MARKET_SENTRY_ENFORCE", {}) or {}, schema_keys)
        return _filter_to_schema(args, schema_keys)
    # --- Auditor: forensic_detective ---
    if tool_name == "forensic_detective":
        args = _force_override(args, getattr(CONFIG, "AUDITOR_FORENSIC_DETECTIVE_ENFORCE", {}) or {}, schema_keys)
        return _filter_to_schema(args, schema_keys)
    # --- PM: portfolio_allocator ---
    if tool_name == "portfolio_allocator":
        args = _force_override(args, getattr(CONFIG, "PM_PORTFOLIO_ALLOCATOR_ENFORCE", {}) or {}, schema_keys)
        return _filter_to_schema(args, schema_keys)

    return _filter_to_schema(args, schema_keys)

# ============================================================
# SECTION 5) Guard：白名单/上限/去重
# ============================================================
def tool_guard_check(
    role: str,
    tool_name: str,
    tool_args: Dict[str, Any],
    state: DebateState,
) -> Tuple[bool, str]:
    allowlist = CONFIG.ROLE_TOOL_ALLOWLIST.get(role, [])
    if tool_name not in allowlist:
        return False, f"tool '{tool_name}' 不在角色 '{role}' 白名单内"

    # max calls
    max_calls = int(CONFIG.ROLE_TOOL_MAX_CALLS.get(role, 0))
    used_calls = int(state.get("_round_tool_calls", {}).get(role, 0))
    if used_calls >= max_calls:
        return False, f"角色 '{role}' 本轮 tool 调用次数已达上限 {max_calls}"

    # dedup same tool+args within same round
    if CONFIG.FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND:
        fp = fingerprint(tool_name, tool_args or {})
        if fp in state.get("_round_fingerprints", set()):
            return False, "同回合同 tool 同参数重复调用已被禁止（dedup hit）"

    return True, "ok"

def _guard_deny_payload(reason: str) -> str:
    res = SkillResult.fail(error_msg=f"[GUARD_DENY] {reason}")
    return json.dumps(res.model_dump(), ensure_ascii=False)

# ============================================================
# SECTION 6) tool 输出解析：用于 trace 统计 produced_n
# ============================================================
def _try_parse_tool_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = str(text).strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _count_produced(obj: Optional[Dict[str, Any]]) -> int:
    """从 SkillResult 结构里统计产出条数（items/candidates/results）。"""
    if not isinstance(obj, dict):
        return 0
    
    data = obj.get("data")
    if isinstance(data, dict):
        for k in ("items", "candidates", "results"):
            v = data.get(k)
            if isinstance(v, list):
                return len(v)
            
    # 兜底：有些 skill 直接在顶层放 items
    if isinstance(obj.get("items"), list):
        return len(obj.get("items"))
    return 0

def _append_tool_trace(
    state: DebateState,
    *,
    role: str,
    tool: str,
    args: Dict[str, Any],
    ok: bool,
    insight: str = "",
    error_msg: Optional[str] = None,
    visuals: Optional[List] = None,
    elapsed_ms: Optional[int] = None,
    denied: bool = False,
    produced_n: Optional[int] = None,
) -> None:
    """trace 增强：写 produced_n，summary 才能“真实统计”而不是猜。"""
    state.setdefault("tool_trace", [])
    state["tool_trace"].append({
        "kind": "tool",
        "role": role,
        "tool": tool,
        "args": args or {},
        "ok": bool(ok),
        "denied": bool(denied),
        "insight": insight or "",
        "error_msg": error_msg,
        "visuals": visuals or [],
        "elapsed_ms": int(elapsed_ms or 0),
        "produced_n": int(produced_n or 0),
        "round_idx": int(state.get("round_idx", 0) or 0),
        "ts": time.time(),
    })

# ============================================================
# SECTION 7) Tool 构建：给 LangChain 的 StructuredTool
# ============================================================
def build_tools_for_role(
    role: str,
    ctx: SkillContext,
    state: DebateState,
) -> List[StructuredTool]:
    """只返回该 role 白名单内的 tools。"""
    SkillRegistry.load_all_skills()
    tools: List[StructuredTool] = []
    allowlist = CONFIG.ROLE_TOOL_ALLOWLIST.get(role, [])

    for tool_name in allowlist:
        skill = SkillRegistry.get_skill(tool_name)
        base_tool = skill.to_langchain_tool(ctx)
        tools.append(_wrap_tool_with_guard(role=role, tool_name=tool_name, base_tool=base_tool, state=state))

    return tools

def _wrap_tool_with_guard(
    *,
    role: str,
    tool_name: str,
    base_tool: StructuredTool,
    state: DebateState,
) -> StructuredTool:
    def _func(**kwargs):
        st = _get_runtime_state(state)
         # 1) 先按 args_schema 过滤 + 应用 policy（少而硬强控）
        schema_keys = _schema_keys_from_tool(base_tool)
        tool_args = _apply_tool_policy(tool_name, dict(kwargs or {}), schema_keys)

        # 2) PM allocator 需要吃“硬状态”字段（仍然先判断 schema 是否支持）
        if tool_name == "portfolio_allocator":
            if schema_keys is None or "candidates" in schema_keys:
                tool_args["candidates"] = st.get("candidates_cur") or st.get("candidates") or []
            if schema_keys is None or "risk_reports" in schema_keys:
                tool_args["risk_reports"] = st.get("risk_reports") or []

        # 3) Guard & Dedup
        allowed, reason = tool_guard_check(role, tool_name, tool_args, st)
        if not allowed:
            mark_guard_denied(st)
            payload_json = _guard_deny_payload(reason)
            payload_obj = _try_parse_tool_json(payload_json) or {}
            _append_tool_trace(
                st,
                role=role,
                tool=tool_name,
                args=tool_args,
                ok=False,
                insight=str(payload_obj.get("insight", "")),
                error_msg=str(payload_obj.get("error_msg", reason)),
                elapsed_ms=0,
                denied=True,
                produced_n=0,
            )
            return payload_json
        
        # 4) call count
        st.setdefault("_round_tool_calls", {"hunter": 0, "auditor": 0, "pm": 0})
        st["_round_tool_calls"][role] = int(st["_round_tool_calls"].get(role, 0)) + 1

        # 5) dedup fingerprints
        if CONFIG.FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND:
            fp = fingerprint(tool_name, tool_args)
            st.setdefault("_round_fingerprints", set())
            st["_round_fingerprints"].add(fp)


        # 6) invoke & trace
        t0 = time.time()
        try:
            out = base_tool.invoke(tool_args)
            if isinstance(out, str):
                out_json = out
            elif isinstance(out, dict):
                out_json = json.dumps(out, ensure_ascii=False)
            else:
                out_json = str(out)

            out_obj = _try_parse_tool_json(out_json) or {
                "success": True,
                "data": out_json,
                "insight": "",
                "error_msg": None,
            }
            st.setdefault("tool_cache", {})
            st["tool_cache"][tool_name] = out_obj
            ok = bool(out_obj.get("success", True))
            produced_n = _count_produced(out_obj)

            _append_tool_trace(
                st,
                role=role,
                tool=tool_name,
                args=tool_args,
                ok=ok,
                insight=str(out_obj.get("insight", "")),
                error_msg=out_obj.get("error_msg"),
                elapsed_ms=int((time.time() - t0) * 1000),
                denied=False,
                produced_n=produced_n,
            )

            st.setdefault("_round_tool_calls_ok", {"hunter": 0, "auditor": 0, "pm": 0})
            if ok:
                st["_round_tool_calls_ok"][role] = int(st["_round_tool_calls_ok"].get(role, 0)) + 1

            # 7) “策略使用记录”写入硬状态
            if ok and role == "hunter" and tool_name == "quantitative_sniper":
                strat = str((tool_args or {}).get("strategy", "") or "").strip()
                if strat:
                    st.setdefault("_hunter_round_sniper_strategies", [])
                    used = st["_hunter_round_sniper_strategies"]
                    if strat not in used:
                        used.append(strat)

            return out_json

        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            fail_obj = SkillResult.fail(error_msg=f"tool '{tool_name}' 执行异常: {e}").model_dump()
            _append_tool_trace(
                st,
                role=role,
                tool=tool_name,
                args=tool_args,
                ok=False,
                insight=str(fail_obj.get("insight", "")),
                error_msg=str(fail_obj.get("error_msg", str(e))),
                elapsed_ms=elapsed,
                denied=False,
                produced_n=0,
            )
            return json.dumps(fail_obj, ensure_ascii=False)

    return StructuredTool(
        name=base_tool.name,
        description=base_tool.description,
        args_schema=getattr(base_tool, "args_schema", None),
        func=_func,
    )

# ============================================================
# SECTION 8) ToolNode 构建（给 graph.py 用）
# ============================================================
def build_tool_node_for_role(
    role: str,
    tools: List[StructuredTool],
    state: DebateState,
) -> Optional[ToolRunner]:
    """返回 ToolRunner（callable），graph 里不再区分 ToolNode 类型。"""
    if not tools:
        return None

    raw_node = ToolNode(tools=tools)

    def _node(state_in: DebateState) -> DebateState:
        ctx_token = _CURRENT_STATE.set(state_in)
        try:
            msgs = state_in.get("messages", [])
            if msgs and isinstance(msgs[-1], AIMessage):
                last_msg = msgs[-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    fixed_calls = []
                    for tc in last_msg.tool_calls:
                        new_args = dict(tc.get("args", {}))
                        
                        # 全局清洗 universe / symbols 字段
                        for k in ["universe", "symbols"]:
                            if k in new_args:
                                val = new_args[k]
                                if isinstance(val, str):
                                    s = val.strip()
                                    # 尝试解开 "['a','b']" 或 "['a', 'b']"
                                    if s.startswith("[") and s.endswith("]"):
                                        try:
                                            # 处理潜在的单引号问题
                                            parsed = json.loads(s.replace("'", '"'))
                                            if isinstance(parsed, list):
                                                new_args[k] = parsed
                                        except Exception:
                                            pass
                        
                        tc["args"] = new_args
                        fixed_calls.append(tc)
                    
                    last_msg.tool_calls = fixed_calls
            out = raw_node.invoke({"messages": msgs})
            tool_msgs: List[Any] = out.get("messages", []) if isinstance(out, dict) else []
            state_in["messages"] = msgs + tool_msgs
            return state_in
        finally:
            _CURRENT_STATE.reset(ctx_token)

    return _node



# ============================================================
# SECTION 9) 便利函数：按 role 直接装配
# ============================================================
def build_role_tools_and_node(
    *,
    role: str,
    dossier,
    ref_date: Optional[str],
    state: DebateState,
) -> Tuple[List[StructuredTool], ToolRunner, SkillContext]:
    ctx = build_ctx(dossier, role=role, ref_date=ref_date)
    tools = build_tools_for_role(role, ctx, state)
    node = build_tool_node_for_role(role, tools, state)
    if node is None:
        raise ValueError(f"role={role} 没有可用 tools，无法构建 tool node")
    return tools, node, ctx