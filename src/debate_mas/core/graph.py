# core/graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from langchain_core.messages import SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .config import CONFIG
from .state import (
    DebateState,
    bump_round,
    push_candidates_merge,
    push_objections,
    push_diff,
    push_decisions,
    bump_stable_rounds,
    set_need_more_candidates,
    clear_need_more_candidates,
)

from debate_mas.protocol.etf_debate import try_parse_payload_with_span, validate_payload

# ============================================================
# 1) 通用：system prompt 置顶
# ============================================================
def _append_system_prompt(messages: List[BaseMessage], system_prompt: str) -> List[BaseMessage]:
    return [SystemMessage(content=system_prompt)] + (messages or [])

def _last_ai_has_tool_calls(state: DebateState) -> bool:
    msgs = state.get("messages", []) or []
    for m in reversed(msgs):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None)
            if tc:
                return True
            ak = getattr(m, "additional_kwargs", {}) or {}
            return bool(ak.get("tool_calls"))
    return False

# ============================================================
# 2) payload 抽取：支持“辩论文字 + 末尾 JSON”
# ============================================================
def _extract_last_payload(state: DebateState, *, expected_type: str) -> Optional[Dict[str, Any]]:
    msgs = state.get("messages", []) or []
    for m in reversed(msgs):
        if not isinstance(m, AIMessage):
            continue
        obj, _start = try_parse_payload_with_span(m.content or "")
        if not obj:
            continue
        try:
            validate_payload(obj)
        except Exception:
            continue
        t = str(obj.get("type", "") or "").strip().upper()
        if t == expected_type:
            return obj
    return None

def _get_stop_suggest(obj: Optional[Dict[str, Any]]) -> str:
    return str((obj or {}).get("stop_suggest", "") or "").strip().upper()

def _extract_need_evidence(objections: List[Dict[str, Any]]) -> Tuple[bool, List[str], List[str]]:
    need = False
    syms: List[str] = []
    acts: List[str] = []
    for it in objections or []:
        verdict = str(it.get("verdict", "") or "").strip().upper()
        if verdict == "NEED_EVIDENCE":
            need = True
            sym = str(it.get("symbol", "") or "").strip()
            if sym:
                syms.append(sym)
            ra = it.get("required_actions", None)
            if isinstance(ra, list):
                for x in ra:
                    x = str(x or "").strip()
                    if x:
                        acts.append(x)

    def _uniq(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out
    return need, _uniq(syms), _uniq(acts)

# ============================================================
# 3) MIN_CANDIDATES：unique 计数 + 判定
# ============================================================
def _unique_candidate_count(items: List[Dict[str, Any]]) -> int:
    seen = set()
    for it in items or []:
        sym = str(it.get("symbol", "") or "").strip()
        if sym:
            seen.add(sym)
    return len(seen)

def _min_candidates_required() -> int:
    if not bool(getattr(CONFIG, "ENFORCE_MIN_CANDIDATES", True)):
        return 0
    try:
        return int(getattr(CONFIG, "HUNTER_MIN_CANDIDATES", 0) or 0)
    except Exception:
        return 0

def _min_candidates_status(state: DebateState) -> Tuple[int, int, int]:
    mn = _min_candidates_required()
    have = _unique_candidate_count(state.get("candidates_cur", []) or [])
    missing = max(0, mn - have) if mn > 0 else 0
    return mn, have, missing

def _hunter_used_sniper_strategies_this_round(state: DebateState) -> List[str]:
    """直接读硬状态，不再从 tool_trace 推断。"""
    xs = state.get("_hunter_round_sniper_strategies", []) or []
    out: List[str] = []
    seen = set()
    for x in xs:
        x = str(x or "").strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ============================================================
# 4) survivor_universe：系统自动算（存活池）
# ============================================================
def _compute_survivor_universe(state: DebateState) -> List[str]:
    """
    计算存活池 U1：
    - 基于 candidates_cur 的 symbol
    - 剔除 auditor verdict=REJECT
    - 剔除 risk_reports 里 liquidity_flag=illiquid（你也可以扩展更多硬剔除规则）
    - 剔除 risk_score >= CONFIG.RISK_SCORE_THRESHOLD
    """
    cand_syms = []
    for it in (state.get("candidates_cur", []) or []):
        sym = str((it or {}).get("symbol", "") or "").strip()
        if sym: cand_syms.append(sym)

    # 1) objections: REJECT 剔除
    reject = set()
    for ob in (state.get("objections_cur", []) or []):
        if not isinstance(ob, dict): continue
        sym = str(ob.get("symbol", "") or "").strip()
        verdict = str(ob.get("verdict", "") or "").strip().upper()
        if sym and verdict == "REJECT": reject.add(sym)

    # 2) risk_reports: illiquid / high_risk 剔除
    illiq = set()
    high_risk = set()
    thr = float(getattr(CONFIG, "RISK_SCORE_THRESHOLD", 50.0) or 50.0)
    for rr in (state.get("risk_reports", []) or []):
        if not isinstance(rr, dict): continue
        sym = str(rr.get("symbol", "") or "").strip()
        liq = str(rr.get("liquidity_flag", "ok") or "ok").strip().lower()
        if sym and liq == "illiquid": illiq.add(sym)
        try:
            rs = float(rr.get("risk_score", 0.0) or 0.0)
            if rs >= thr:
                high_risk.add(sym)
        except Exception:
            pass

    out: List[str] = []
    seen = set()
    for s in cand_syms:
        if s in seen or s in reject or s in illiq or s in high_risk:
            continue
        seen.add(s)
        out.append(s)
    return out

# ============================================================
# 5) DIFF：系统自动算
# ============================================================
def _index_by_symbol(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items or []:
        sym = str(it.get("symbol", "")).strip()
        if sym:
            out[sym] = it
    return out

def _compute_candidates_diff(
    prev_items: List[Dict[str, Any]],
    cur_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prev = _index_by_symbol(prev_items or [])
    cur = _index_by_symbol(cur_items or [])

    prev_syms = set(prev.keys())
    cur_syms = set(cur.keys())

    patches: List[Dict[str, Any]] = []
    for sym in sorted(cur_syms - prev_syms):
        patches.append({"op": "ADD", "symbol": sym, "note": "新增候选"})

    # 同标的：看 score/reason 是否变化（轻量即可）
    for sym in sorted(prev_syms & cur_syms):
        p = prev.get(sym, {})
        c = cur.get(sym, {})
        if p.get("score", None) != c.get("score", None) and (p.get("score") is not None) and (c.get("score") is not None):
            patches.append({"op": "SCORE_UPDATE", "symbol": sym, "note": f"score: {p.get('score')} -> {c.get('score')}"})
        p_rs = str(p.get("reason", "") or "").strip()
        c_rs = str(c.get("reason", "") or "").strip()
        if p_rs and c_rs and p_rs != c_rs:
            patches.append({"op": "REASON_UPDATE", "symbol": sym, "note": "reason 已更新"})

    return {"type": "DIFF", "items": patches}


# ============================================================
# 6) 软 trace：解释“为什么继续/为什么强制/为什么 diff”
# ============================================================
def _append_soft_trace(
    state: DebateState,
    *,
    role: str,
    tool: str,
    insight: str,
    args: Optional[Dict[str, Any]] = None,
    ok: bool = True,
) -> None:
    """kind="trace"：软 trace（解释“为什么继续/为什么强制/为什么 diff”）"""
    state.setdefault("tool_trace", [])
    state["tool_trace"].append(
        {
            "kind": "trace", 
            "role": role,
            "tool": tool,
            "args": args or {},
            "ok": bool(ok),
            "denied": False,
            "produced_n": 0,
            "elapsed_ms": 0,
            "round_idx": int(state.get("round_idx", 0) or 0),
            "insight": insight,
            "error_msg": None,
            "visuals": [],
        }
    )

# ============================================================
# 7) RoleBlock：注入 llm + tools + postprocess
# ============================================================
ToolRunner = Callable[[DebateState], DebateState]
@dataclass(frozen=True)
class RoleBlock:
    role: str
    system_prompt: str
    llm_invoke: Callable[[List[BaseMessage]], AIMessage]
    tool_node: Optional[ToolRunner]
    postprocess: Callable[[DebateState], None]


def _make_tool_wrapper(tool_node: ToolRunner) -> ToolRunner:
    def _tools(state: DebateState) -> DebateState:
        return tool_node(state)
    return _tools

# ============================================================
# 8) Two-stage pipeline sys prompt helper
# ============================================================
def _build_hunter_pipeline_sys_prompt(state: DebateState) -> Optional[str]:
    """把 Two-stage pipeline 的 sys prompt 拼接抽成 helper："""
    if not bool(getattr(CONFIG, "HUNTER_DETERMINISTIC_PIPELINE", True)):
        return None
    if str(getattr(CONFIG, "HUNTER_PIPELINE_MODE", "two_stage")) != "two_stage":
        return None

    stage = str(state.get("_hunter_pipeline_stage", "recall") or "recall").strip().lower()
    recall_strats = getattr(CONFIG, "HUNTER_RECALL_STRATEGIES", []) or []
    min_strats = int(getattr(CONFIG, "HUNTER_RECALL_MIN_STRATEGIES", 2) or 2)
    topk_each = int(getattr(CONFIG, "HUNTER_RECALL_TOPK_PER_STRATEGY", 10) or 10)

    if stage == "recall":
        sys_lines = [
            "【Two-Stage Pipeline | STAGE=RECALL】本轮先多策略召回，再输出候选池（别漏机会）。",
            f"- 至少调用 quantitative_sniper {min_strats} 次（不同 strategy），strategy ∈ {recall_strats}。",
            f"- 每次召回建议 top_k≈{topk_each}。",
            "- 将各次结果 union 去重得到候选池 U。",
            "- 输出 CANDIDATES：items 覆盖 MIN_CANDIDATES；extra.sources 记录来自哪个 strategy。",
            "- 注意：composite 不是可选策略；它只用于下一阶段 rerank。",
        ]
    else:
        u1 = state.get("survivor_universe", []) or []
        sys_lines = [
            "【Two-Stage Pipeline | STAGE=RERANK】对存活池做 composite 统一再排序（同一把尺子）。",
            f"- 存活池 survivor_universe size={len(u1)}。",
            "- 必须调用 quantitative_sniper(strategy='composite', universe=survivor_universe, top_k=len(universe))。",
            "- 输出 CANDIDATES：score 使用 composite_score(0~100) 并排序；extra 保留 raw 指标。",
        ]

    return "\n".join(sys_lines)

# ============================================================
# 9) postprocess：把“本轮产物”写入 state
# ============================================================
def _normalize_candidate_items(items: List[Dict[str, Any]], prev_by_sym: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """不做复杂校验，只做“缺字段补齐”，避免 PM 强转模型时崩溃"""
    out: List[Dict[str, Any]] = []
    autofill = 0
    prev_by_sym = prev_by_sym or {}

    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        it = dict(raw)

        sym = str(it.get("symbol", "") or "").strip()
        if not sym:
            continue
        it["symbol"] = sym
        prev = prev_by_sym.get(sym, {})
        if "score" not in it:
            it["score"] = prev.get("score", 0.0)
            autofill += 1
        if "reason" not in it:
            it["reason"] = prev.get("reason", "")
            autofill += 1
        if "source_skill" not in it:
            it["source_skill"] = prev.get("source_skill", "unknown")
            autofill += 1

        if "extra" not in it or it["extra"] is None:
            old_extra = prev.get("extra", {})
            it["extra"] = old_extra if isinstance(old_extra, dict) else {}
            autofill += 1
        elif not isinstance(it["extra"], dict):
            it["extra"] = {"raw_extra": it["extra"]}
            autofill += 1

        out.append(it)

    return out, autofill

def postprocess_hunter(state: DebateState) -> None:
    obj = _extract_last_payload(state, expected_type="CANDIDATES")
    if not obj: return

    cur_items = obj.get("items") or []
    if not isinstance(cur_items, list): return

    prev_items = state.get("candidates_cur", []) or []
    prev_map = _index_by_symbol(prev_items)
    cur_items_norm, _autofill = _normalize_candidate_items(cur_items, prev_map)

    push_candidates_merge(state, cur_items_norm)

    # Rerank 阶段强制 TopN 截断，防止 Token 爆炸
    is_pipeline = bool(getattr(CONFIG, "HUNTER_DETERMINISTIC_PIPELINE", True))
    stage = str(state.get("_hunter_pipeline_stage", "recall") or "recall").strip().lower()
    
    if is_pipeline and stage == "rerank":
        top_n = int(getattr(CONFIG, "HUNTER_RERANK_OUTPUT_TOPN", 20)) # 默认为 20
        current_list = state.get("candidates_cur", [])
        if len(current_list) > top_n:
            current_list.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
            
            # 裁剪->写回
            trimmed_list = current_list[:top_n]
            state["candidates_cur"] = trimmed_list
            state["candidates"] = trimmed_list
            
            _append_soft_trace(
                state, 
                role="system", 
                tool="__rerank_cutoff__", 
                insight=f"Rerank 阶段强制截断：从 {len(current_list)} 缩减至 Top {top_n}"
            )
    merged_items = state.get("candidates_cur", []) or []

    diff_obj = _compute_candidates_diff(prev_items, merged_items)
    push_diff(state, diff_obj)

    state["hunter_stop_suggest"] = _get_stop_suggest(obj)

    mn, have, _missing = _min_candidates_status(state)
    if mn > 0 and have >= mn and bool(state.get("_need_more_candidates", False)):
        clear_need_more_candidates(state)
        _append_soft_trace(
            state,
            role="system",
            tool="__min_candidates_satisfied__",
            insight=f"候选池已达标：unique={have} >= MIN={mn}，清理 need_more_candidates 标志。",
        )

    if bool(state.get("_force_hunter_tool", False)):
        ok_calls = int((state.get("_round_tool_calls_ok", {}) or {}).get("hunter", 0) or 0)
        if ok_calls <= 0:
            state["_round_missing_evidence"] = True
            state["hunter_stop_suggest"] = "CONTINUE"
            _append_soft_trace(
                state,
                role="system",
                tool="__hunter_missing_tool__",
                insight="本轮要求 hunter 至少调用 1 次工具，但未发生有效工具调用 -> 强制继续下一轮。",
            )
        state["_force_hunter_tool"] = False

    _append_soft_trace(state, role="hunter", tool="__hunter_output__", insight="hunter 输出 CANDIDATES（末尾 JSON）")
    _append_soft_trace(state, role="system", tool="__diff__", insight=f"系统计算 DIFF patches={len(diff_obj.get('items', []))}")

    if bool(getattr(CONFIG, "HUNTER_DETERMINISTIC_PIPELINE", True)) and str(getattr(CONFIG, "HUNTER_PIPELINE_MODE", "two_stage")) == "two_stage":
        stage = str(state.get("_hunter_pipeline_stage", "recall") or "recall").strip().lower()
        used = _hunter_used_sniper_strategies_this_round(state)

        if stage == "recall":
            allow = set(getattr(CONFIG, "HUNTER_RECALL_STRATEGIES", []) or [])
            used_in_allow = [x for x in used if x in allow]
            min_n = int(getattr(CONFIG, "HUNTER_RECALL_MIN_STRATEGIES", 2) or 2)

            if len(used_in_allow) < min_n:
                state["_need_recall_diversity"] = True
                state["_need_recall_diversity_reason"] = f"recall 阶段需>={min_n}个策略，但本轮仅记录 {used_in_allow or used}"
                state["hunter_stop_suggest"] = "CONTINUE"
                _append_soft_trace(state, role="system", tool="__pipeline_recall_not_met__", insight=state["_need_recall_diversity_reason"])
            else:
                state["_need_recall_diversity"] = False
                state["_need_recall_diversity_reason"] = ""

        if stage == "rerank":
            if "composite" not in used:
                state["_need_rerank_composite"] = True
                state["_need_rerank_composite_reason"] = f"rerank 阶段必须调用 composite，但本轮仅记录 {used}"
                state["hunter_stop_suggest"] = "CONTINUE"
                _append_soft_trace(state, role="system", tool="__pipeline_rerank_not_met__", insight=state["_need_rerank_composite_reason"])
            else:
                state["_need_rerank_composite"] = False
                state["_need_rerank_composite_reason"] = ""

def _extract_risk_items_from_cache(state: DebateState, tool_name: str) -> List[Dict[str, Any]]:
    cache = (state.get("tool_cache", {}) or {}).get(tool_name)
    if not isinstance(cache, dict):
        return []
    data = cache.get("data")
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    return items if isinstance(items, list) else []


def _merge_risk_reports(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """合并多个 EtfRiskReportList"""
    by_sym: Dict[str, Dict[str, Any]] = {}

    def _flag_max(a: str, b: str, *, order: List[str]) -> str:
        ia = order.index(a) if a in order else 0
        ib = order.index(b) if b in order else 0
        return b if ib > ia else a

    for items in lists:
        for it in items or []:
            if not isinstance(it, dict):
                continue
            sym = str(it.get("symbol", "") or "").strip()
            if not sym:
                continue

            risk = float(it.get("risk_score", 0.0) or 0.0)
            notes = it.get("notes") or []
            notes = [str(x) for x in notes if str(x).strip()]

            liq = str(it.get("liquidity_flag", "ok") or "ok")
            sent = str(it.get("sentiment_flag", "normal") or "normal")

            if sym not in by_sym:
                by_sym[sym] = {"symbol": sym, "risk_score": 0.0, "liquidity_flag": liq, "sentiment_flag": sent, "notes": []}

            cur = by_sym[sym]
            cur["risk_score"] = float(min(100.0, float(cur["risk_score"]) + risk))
            cur["liquidity_flag"] = _flag_max(str(cur["liquidity_flag"]), liq, order=["ok", "illiquid"])
            cur["sentiment_flag"] = _flag_max(str(cur["sentiment_flag"]), sent, order=["normal", "negative"])

            # notes 去重保序
            seen = set(cur["notes"])
            for n in notes:
                if n not in seen:
                    cur["notes"].append(n)
                    seen.add(n)

    out = list(by_sym.values())
    out.sort(key=lambda x: float(x.get("risk_score", 0.0)), reverse=True)
    return out

def postprocess_auditor(state: DebateState) -> None:
    obj = _extract_last_payload(state, expected_type="OBJECTIONS")
    if not obj:
        return

    items = obj.get("items") or []
    prev_u1 = list(state.get("survivor_universe", []) or [])
    if not isinstance(items, list):
        return

    push_objections(state, items)
    ms = _extract_risk_items_from_cache(state, "market_sentry")
    fd = _extract_risk_items_from_cache(state, "forensic_detective")
    state["risk_reports"] = _merge_risk_reports(ms, fd)
    state["survivor_universe"] = _compute_survivor_universe(state)

    new_u1 = state["survivor_universe"]
    removed_syms = set(prev_u1) - set(new_u1)

    if removed_syms:
        hard_patches = []
        for sym in sorted(removed_syms):
            hard_patches.append({"op": "REMOVE", "symbol": sym, "note": "系统硬剔除"})
        
        diff = state.get("diff_cur", {}) or {"type": "DIFF", "items": []}
        if not isinstance(diff.get("items"), list):
            diff["items"] = []
        diff["items"].extend(hard_patches)
        push_diff(state, diff)
        
    bump_stable_rounds(state, reset_if_changed=True)

    state["auditor_stop_suggest"] = _get_stop_suggest(obj)
    need, syms, acts = _extract_need_evidence(items)
    state["_need_evidence"] = bool(need)
    state["_need_evidence_symbols"] = syms
    state["_need_evidence_actions"] = acts

    _append_soft_trace(state, role="auditor", tool="__auditor_output__", insight="auditor 输出 OBJECTIONS")


def postprocess_pm(state: DebateState) -> None:
    obj = _extract_last_payload(state, expected_type="DECISIONS")
    if not obj: return

    items = obj.get("items") or []
    if not isinstance(items, list): return

    push_decisions(state, items)
    state["pm_stop_suggest"] = _get_stop_suggest(obj)
    _append_soft_trace(state, role="pm", tool="__pm_output__", insight="pm 输出 DECISIONS（JSON-only）")

# ============================================================
# 10) judge：决定下一步走向（attack/patch 的收敛规则）
# ============================================================
def _should_end_debate(state: DebateState) -> str:
    """
    【教学抽题点（第二段练习）】
    1) MAX_ROUNDS / stop 条件：让学生改 N 轮、或增加 early_stop 规则
    2) MIN_CANDIDATES / pipeline gate：让学生改“何时强制继续”
    注意：这里是“收敛裁决器”，改动最直观、最可测试。
    """
    ridx = int(state.get("round_idx", 0) or 0)
    max_rounds = int(getattr(CONFIG, "MAX_ROUNDS", 3) or 3)

    if ridx >= max_rounds - 1:
        state["stop_reason"] = "MAX_ROUNDS_DEBATE"
        clear_need_more_candidates(state)
        return "pm"

    if bool(state.get("_round_guard_denied", False)):
        state["stop_reason"] = "GUARD_DENIED"
        return "next_round"

    hs = str(state.get("hunter_stop_suggest", "") or "").strip().upper()
    ads = str(state.get("auditor_stop_suggest", "") or "").strip().upper()

    mn, have, missing = _min_candidates_status(state)
    if mn > 0 and missing > 0:
        state["stop_reason"] = "MIN_CANDIDATES_NOT_MET"
        _append_soft_trace(state, role="system", tool="__min_candidates__", insight=f"候选池不足：unique={have} < MIN={mn}（missing={missing}）-> 强制继续下一轮补齐。")
        return "next_round"

    if bool(state.get("_need_recall_diversity", False)):
        state["stop_reason"] = "PIPELINE_RECALL_DIVERSITY_NOT_MET"
        return "next_round"
    if bool(state.get("_need_rerank_composite", False)):
        state["stop_reason"] = "PIPELINE_RERANK_NOT_MET"
        return "next_round"

    # 双方都认为可停：直接进 PM
    allow_early_exit = bool(getattr(CONFIG, "EXIT_ON_CONSENSUS", True))
    if allow_early_exit and hs == "STOP" and ads == "STOP":
        state["stop_reason"] = "CONSENSUS_STOP"
        return "pm"

    # 稳定轮数达到阈值：认为已收敛（默认 1 就够，教学更直观）
    stable = int(state.get("stable_rounds", 0) or 0)
    if stable >= 1 and ads == "STOP":
        state["stop_reason"] = "STABLE_AND_AUDITOR_STOP"
        return "pm"

    state["stop_reason"] = "CONTINUE_DEBATE"
    return "next_round"


# ============================================================
# 11) 构建图：Hunter ↔ Auditor（attack/patch）→ PM
# ============================================================

def build_etf_attack_patch_graph(
    *,
    hunter: RoleBlock,
    auditor: RoleBlock,
    pm: RoleBlock,
) -> StateGraph:
    g = StateGraph(DebateState)

    def add_role(rb: RoleBlock) -> Tuple[str, str]:
        role = rb.role
        agent_n = f"{role}_agent"
        tools_n = f"{role}_tools"
        post_n = f"{role}_postprocess"

        def _agent(state: DebateState) -> DebateState:
            msgs = state.get("messages", []) or []
            prompt_msgs = _append_system_prompt(msgs, rb.system_prompt)

            # auditor：强制工具调用提示
            if role == "auditor":
                prompt_msgs = [SystemMessage(content="【强制】本轮至少调用 1 次工具（优先 market_sentry），并在 Final JSON 的 evidence 中引用【本轮】ToolMessage 输出。")] + prompt_msgs

            # hunter：Two-stage pipeline 提示
            if role == "hunter":
                sys_prompt = _build_hunter_pipeline_sys_prompt(state)  # 【MOD】
                if sys_prompt:
                    prompt_msgs = [SystemMessage(content=sys_prompt)] + prompt_msgs  # 【MOD】

            state["_last_speaker_role"] = role
            state["phase"] = role

            ai = rb.llm_invoke(prompt_msgs)
            state["messages"] = (msgs or []) + [ai]
            return state

        def _post(state: DebateState) -> DebateState:
            rb.postprocess(state)
            return state

        g.add_node(agent_n, _agent)
        g.add_node(post_n, _post)

        if rb.tool_node is not None:
            g.add_node(tools_n, _make_tool_wrapper(rb.tool_node)) 

            def _route(_state: DebateState) -> str:
                return "tools" if _last_ai_has_tool_calls(_state) else "post"

            g.add_conditional_edges(agent_n, _route, {"tools": tools_n, "post": post_n})
            g.add_edge(tools_n, agent_n)
        else:
            g.add_edge(agent_n, post_n)

        return agent_n, post_n

    hunter_agent, hunter_post = add_role(hunter)
    auditor_agent, auditor_post = add_role(auditor)
    pm_agent, pm_post = add_role(pm)

    # hunter -> auditor
    g.add_edge(hunter_post, auditor_agent)

    # next_round：推进轮次并重置 guard 计数
    def _next_round(state: DebateState) -> DebateState:
        # 记录“为何继续”，用于决定下一轮是否强制 hunter 用工具
        stop_reason = str(state.get("stop_reason", "") or "").strip().upper()
        mn, have, missing = _min_candidates_status(state)

        bump_round(state)
        state["phase"] = "next_round"

        need_more = (mn > 0 and missing > 0)
        if need_more:
            set_need_more_candidates(
                state,
                min_required=mn,
                have=have,
                missing=missing,
                reason="候选池不足，需补齐给 PM 足够选择空间",
            )
            _append_soft_trace(
                state,
                role="system",
                tool="__min_candidates_flag__",
                insight=f"进入下一轮前设置补齐标志：unique={have} < MIN={mn}（missing={missing}）-> 强制 hunter 调工具补齐。",
            )
        else:
            clear_need_more_candidates(state)

        need_evidence = bool(state.get("_need_evidence", False)) and bool(getattr(CONFIG, "ENFORCE_TOOL_ON_NEED_EVIDENCE", True))
        guard_denied = (stop_reason == "GUARD_DENIED")

        # two-stage：下一轮阶段推进
        if bool(getattr(CONFIG, "HUNTER_DETERMINISTIC_PIPELINE", True)) and str(getattr(CONFIG, "HUNTER_PIPELINE_MODE", "two_stage")) == "two_stage":
            if need_more or bool(state.get("_need_recall_diversity", False)):
                state["_hunter_pipeline_stage"] = "recall"
            else:
                state["_hunter_pipeline_stage"] = "rerank"

        pipeline_fix = bool(state.get("_need_recall_diversity", False)) or bool(state.get("_need_rerank_composite", False))
        state["_force_hunter_tool"] = bool(need_more or need_evidence or guard_denied or pipeline_fix)

        if state["_force_hunter_tool"]:
            why: List[str] = []
            if need_more:
                why.append("MIN_CANDIDATES")
            if need_evidence:
                why.append("NEED_EVIDENCE")
            if guard_denied:
                why.append("GUARD_DENIED")
            if bool(state.get("_need_recall_diversity", False)):
                why.append("PIPELINE_RECALL")
            if bool(state.get("_need_rerank_composite", False)):
                why.append("PIPELINE_RERANK")

            _append_soft_trace(
                state,
                role="system",
                tool="__force_hunter_tool__",
                insight=f"下一轮强制 hunter 调工具：reasons={why}, stage={state.get('_hunter_pipeline_stage')}",
            )

        return state

    g.add_node("next_round", _next_round)
    g.add_conditional_edges(auditor_post, _should_end_debate, {"next_round": "next_round", "pm": pm_agent})
    g.add_edge("next_round", hunter_agent)

    g.add_edge(pm_post, END)
    g.set_entry_point(hunter_agent)
    return g.compile()
