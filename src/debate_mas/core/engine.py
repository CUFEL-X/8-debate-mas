# core/engine.py
from __future__ import annotations

import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from debate_mas.loader.dual_mode_loader import DualModeLoader
from debate_mas.skills.registry import SkillRegistry
from debate_mas.protocol.renderer import DebateRenderer
from debate_mas.protocol.schema import EtfDecision

from debate_mas.protocol.etf_debate import try_parse_payload_with_span, validate_payload

from .config import CONFIG
from .state import init_state, DebateState
from .personas import build_role_prompts_etf

from .graph import (
    RoleBlock,
    build_etf_attack_patch_graph,
    postprocess_hunter,
    postprocess_auditor,
    postprocess_pm,
)

from .blend_rank import merge_candidates, explain_merge
from .tools import build_role_tools_and_node

load_dotenv()

# ============================================================
# 0) LLM / coercion helpers
# ============================================================
def _build_llm(
    model_name: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 4000,
) -> ChatOpenAI:
    """统一 LLM 构造器：ChatOpenAI 连接 LLM。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    api_base = os.getenv("DASHSCOPE_BASE_URL", "")

    if not api_key or not api_base:
        raise RuntimeError(
            "缺少环境变量：DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL。"
            "请在项目根目录创建 .env 并写入这两个字段。"
        )
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _coerce_decisions(decisions: List[Dict[str, Any]]) -> List[EtfDecision]:
    """把 PM 输出的 dict 统一转成 EtfDecision"""
    out: List[EtfDecision] = []
    for d in decisions or []:
        if isinstance(d, EtfDecision):
            out.append(d)
        elif isinstance(d, dict):
            out.append(EtfDecision(**d))
    return out


def _coerce_tool_trace(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """ToolTraceEntry 需要字段：tool/args/ok/insight/error_msg/visuals"""
    out: List[Dict[str, Any]] = []
    for t in trace or []:
        if not isinstance(t, dict):
            continue
        x = dict(t)
        x.setdefault("kind", "tool")  
        x.setdefault("args", {})
        x.setdefault("ok", True)
        x.setdefault("denied", False)
        x.setdefault("insight", "")
        x.setdefault("error_msg", None)
        x.setdefault("visuals", [])
        x.setdefault("produced_n", 0)
        x.setdefault("elapsed_ms", 0)
        x.setdefault("round_idx", 0)
        x.setdefault("role", "unknown")
        out.append(x)
    return out


# ============================================================
# 1) Transcript 序列化工具
# ============================================================
def _infer_role(m: BaseMessage) -> str:
    if isinstance(m, HumanMessage):
        return "user"
    if isinstance(m, AIMessage):
        return "assistant"
    if isinstance(m, ToolMessage):
        return "tool"
    t = getattr(m, "type", None)
    return str(t) if t else "unknown"


def _serialize_messages(msgs: List[BaseMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in msgs or []:
        d: Dict[str, Any] = {"role": _infer_role(m), "content": (getattr(m, "content", "") or "")}

        tc = getattr(m, "tool_calls", None)
        if tc:
            d["tool_calls"] = tc

        ak = getattr(m, "additional_kwargs", None) or {}
        if isinstance(ak, dict) and ak.get("tool_calls"):
            d["tool_calls"] = ak.get("tool_calls")

        name = getattr(m, "name", None)
        if name:
            d["name"] = name

        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            d["tool_call_id"] = tool_call_id

        out.append(d)
    return out

# ============================================================
# 2) run() 的 3 个“纯组装函数”
# ============================================================
def _setup_dossier_and_state(
    *,
    mission: str,
    ref_date: Optional[str],
    folder_path: Optional[str],
    seed_user_message: Optional[str],
) -> Tuple[Any, DebateState]:
    # 1) 加载案卷（dossier）
    loader = DualModeLoader()
    dossier = loader.load_from_folder(mission=mission, folder_path=folder_path or CONFIG.DATA_DIR)

    # 2) 初始化 state
    msgs: List[BaseMessage] = []
    if seed_user_message:
        msgs.append(HumanMessage(content=seed_user_message))
    st: DebateState = init_state(mission=mission, dossier=dossier, ref_date=ref_date, messages=msgs)

    return dossier, st


def _setup_prompts_tools_llms(
    *,
    mission: str,
    dossier: Any,
    ref_date: Optional[str],
    st: DebateState,
) -> Tuple[Dict[str, str], RoleBlock, RoleBlock, RoleBlock]:
    # 1) 构建 personas
    prompts = build_role_prompts_etf(
        mission=mission,
        dossier_view=st.get("dossier_view", {}) or {},
        allowlist_by_role=CONFIG.ROLE_TOOL_ALLOWLIST,
    )

    # 2) 构建 tools + tool nodes
    hunter_tools, hunter_tool_node, _ = build_role_tools_and_node(role="hunter", dossier=dossier, ref_date=ref_date, state=st)
    auditor_tools, auditor_tool_node, _ = build_role_tools_and_node(role="auditor", dossier=dossier, ref_date=ref_date, state=st)
    pm_tools, pm_tool_node, _ = build_role_tools_and_node(role="pm", dossier=dossier, ref_date=ref_date, state=st)

    temps = getattr(CONFIG, "ROLE_TEMPERATURE", {}) or {}

    role_max_tokens = getattr(CONFIG, "ROLE_MAX_TOKENS", {}) or {}
    max_tokens_default = int(getattr(CONFIG, "MAX_TOKENS_DEFAULT", 3000) or 3000)

    # 3) 构建 LLM
    hunter_llm = _build_llm(CONFIG.HUNTER_MODEL, temperature=float(temps.get("hunter", 0.9)), max_tokens=role_max_tokens.get("hunter", max_tokens_default)).bind_tools(hunter_tools)
    auditor_llm = _build_llm(CONFIG.AUDITOR_MODEL, temperature=float(temps.get("auditor", 0.3)), max_tokens=role_max_tokens.get("auditor", max_tokens_default)).bind_tools(auditor_tools)
    pm_llm = _build_llm(CONFIG.PM_MODEL, temperature=float(temps.get("pm", 0.1)), max_tokens=role_max_tokens.get("pm", max_tokens_default)).bind_tools(pm_tools)

    # 4) 组装 RoleBlock
    hunter_block = RoleBlock(
        role="hunter",
        system_prompt=prompts["hunter"],
        llm_invoke=lambda ms: hunter_llm.invoke(ms),
        tool_node=hunter_tool_node,
        postprocess=postprocess_hunter,
    )
    auditor_block = RoleBlock(
        role="auditor",
        system_prompt=prompts["auditor"],
        llm_invoke=lambda ms: auditor_llm.invoke(ms),
        tool_node=auditor_tool_node,
        postprocess=postprocess_auditor,
    )
    pm_block = RoleBlock(
        role="pm",
        system_prompt=prompts["pm"],
        llm_invoke=lambda ms: pm_llm.invoke(ms),
        tool_node=pm_tool_node,
        postprocess=postprocess_pm,
    )

    return prompts, hunter_block, auditor_block, pm_block


def _run_graph_and_render(
    *,
    mission: str,
    ref_date: Optional[str],
    output_dir: str,
    st: DebateState,
    hunter_block: RoleBlock,
    auditor_block: RoleBlock,
    pm_block: RoleBlock,
    verbose_summary: bool,
) -> Dict[str, str]:
    # 1) 构建并运行图
    app = build_etf_attack_patch_graph(hunter=hunter_block, auditor=auditor_block, pm=pm_block)

    final_state: DebateState = st
    print("\n🟦 VERBOSE_MODE=summary：辩论级摘要（按轮/角色工具摘要 + 自然语言）\n")
    
    if verbose_summary:
        last_tool_trace_len = len(st.get("tool_trace", []) or [])
        last_msg_len = len(st.get("messages", []) or [])

        for step_state in app.stream(st, stream_mode="values"):
            final_state = step_state

            # 1) tool_trace 增量摘要
            tool_trace_now = final_state.get("tool_trace", []) or []
            last_tool_trace_len = _print_tool_trace_increment(tool_trace_now, last_tool_trace_len)

            # 2) messages 增量：打印 Debate + payload 一行摘要
            msgs_now = final_state.get("messages", []) or []
            last_msg_len = _print_assistant_messages_increment(msgs_now, last_msg_len, max_chars=900, state=final_state)
    else:
        final_state = app.invoke(st)

    print("\n🟦 VERBOSE END\n")

    # 2) 候选融合留痕
    cand = final_state.get("candidates_cur", None)
    if cand is None:
        cand = final_state.get("candidates", []) or []

    merged = merge_candidates([cand], source_weights=CONFIG.HUNTER_BLEND)
    final_state["candidates"] = merged
    final_state["candidates_cur"] = merged

    merge_notes = explain_merge(merged, top_n=5)

    transcript_msgs: List[BaseMessage] = final_state.get("messages", []) or []
    transcript = _serialize_messages(transcript_msgs)

    # 3) 渲染输出
    round_idx = int(final_state.get("round_idx", 0) or 0)
    rounds_done = round_idx + 1

    renderer = DebateRenderer(output_dir=output_dir)
    tool_trace = _coerce_tool_trace(final_state.get("tool_trace", []) or [])

    extra_meta = {
        "ref_date": ref_date,
        "rounds": rounds_done,
        "stop_reason": final_state.get("stop_reason"),
        "tool_trace": tool_trace,
        "dossier_meta": final_state.get("dossier_view", {}) or {},
        "extras": {
            "merge_notes": merge_notes,
            "config_snapshot": CONFIG.get_model_config(),
            "transcript": transcript,
            "candidates_cur": final_state.get("candidates_cur", []),
            "objections_cur": final_state.get("objections_cur", []),
            "diff_cur": final_state.get("diff_cur", {}),
        },
    }

    decisions_raw = final_state.get("decisions", []) or []
    decisions = _coerce_decisions(decisions_raw)

    artifacts = renderer.render(mission=mission, decisions=decisions, extra_meta=extra_meta)
    final_state["artifacts"] = artifacts

    # 4) transcript 落盘
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = os.path.join(output_dir, f"{ts}_transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "mission": mission,
                    "ref_date": ref_date,
                    "rounds": rounds_done,
                    "stop_reason": final_state.get("stop_reason"),
                    "transcript": transcript,
                    "candidates_cur": final_state.get("candidates_cur", []),
                    "objections_cur": final_state.get("objections_cur", []),
                    "diff_cur": final_state.get("diff_cur", {}),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if isinstance(artifacts, dict):
            artifacts["transcript"] = transcript_path

        if verbose_summary:
            print(f"📝 transcript 已落盘: {transcript_path}")

    except Exception as e:
        if verbose_summary:
            print(f"⚠️ transcript 落盘失败: {e}")

    return artifacts


# ============================================================
# 3) 一键运行入口
# ============================================================
def run(
    mission: str,
    *,
    ref_date: Optional[str] = None,
    folder_path: Optional[str] = None,
    output_dir: str = "./output_reports",
    seed_user_message: Optional[str] = None,
) -> Dict[str, str]:
    """
    一键运行入口：
    - folder_path：本地文件夹模式（最适合教学与业务人员）
    - 输出：log.json + memo.md + rebalance.csv（由 renderer 负责）
    """
    verbose_summary = CONFIG.VERBOSE

    # 1) 加载 skills
    SkillRegistry.load_all_skills(force_reload=False)

    # 2) 准备 dossier + state
    dossier, st = _setup_dossier_and_state(
        mission=mission,
        ref_date=ref_date,
        folder_path=folder_path,
        seed_user_message=seed_user_message,
    )

    # 3) 准备 prompts/tools/llms
    _prompts, hunter_block, auditor_block, pm_block = _setup_prompts_tools_llms(
        mission=mission,
        dossier=dossier,
        ref_date=ref_date,
        st=st,
    )

    # 4) 运行图 + 渲染输出
    return _run_graph_and_render(
        mission=mission,
        ref_date=ref_date,
        output_dir=output_dir,
        st=st,
        hunter_block=hunter_block,
        auditor_block=auditor_block,
        pm_block=pm_block,
        verbose_summary=verbose_summary,
    )

# ============================================================
# VERBOSE SECTION
# ============================================================
def _summarize_structured_payload(obj: Dict[str, Any]) -> Optional[str]:
    """结构化 JSON（CANDIDATES/OBJECTIONS/DIFF/DECISIONS）只打印一行摘要。"""
    t = str(obj.get("type", "")).strip().upper()
    if t not in {"CANDIDATES", "OBJECTIONS", "DIFF", "DECISIONS"}:
        return None
    items = obj.get("items")
    n = len(items) if isinstance(items, list) else 0
    stop = str(obj.get("stop_suggest", "") or "").strip().upper()
    extra = f" stop={stop}" if stop else ""
    return f"[{t}] items={n}{extra}"


def _infer_assistant_role_hint(ai: AIMessage, state: DebateState) -> str:
    """用 _last_speaker_role 获取当前 AIMessage 属于哪个 role。"""
    # 1) 最优：ai.name（我们在 graph._agent 里写入）
    name = getattr(ai, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    # 2) 其次：additional_kwargs["_speaker_role"]
    ak = getattr(ai, "additional_kwargs", {}) or {}
    if isinstance(ak, dict):
        sr = ak.get("_speaker_role")
        if isinstance(sr, str) and sr.strip():
            return sr.strip()

    # 3) 兜底：state
    return str(state.get("_last_speaker_role", "assistant"))


def _tool_trace_entry_digest(t: Dict[str, Any]) -> Dict[str, Any]:
    """produced_n 只信 trace 里的 produced_n。"""
    kind = str(t.get("kind", "tool") or "tool").strip().lower()
    tool = str(t.get("tool", "unknown"))
    role = str(t.get("role", "unknown"))
    ok = bool(t.get("ok", True))
    denied = bool(t.get("denied", False))
    round_idx = int(t.get("round_idx", 0) or 0)
    insight = (t.get("insight") or "").strip()
    err = (t.get("error_msg") or "").strip()
    elapsed_ms = int(t.get("elapsed_ms", 0) or 0)

    produced_n: Optional[int] = None
    if "produced_n" in t:
        try:
            produced_n = int(t.get("produced_n") or 0)
        except Exception:
            produced_n = None

    return {
        "kind": kind,
        "round_idx": round_idx,
        "role": role,
        "tool": tool,
        "args": t.get("args", {}),
        "ok": ok,
        "denied": denied,
        "elapsed_ms": elapsed_ms,
        "produced_n": produced_n,
        "insight": insight,
        "error": err,
    }

def _print_tool_trace_increment(tool_trace: List[Dict[str, Any]], start_idx: int) -> int:
    """
    工具调用 vs 软trace 分开打印：
    - 🛠️ Tool Calls：kind="tool"
    - 📝 Trace：kind="trace"
    """
    new_entries = (tool_trace or [])[start_idx:]
    if not new_entries:
        return start_idx

    digests = [_tool_trace_entry_digest(t) for t in new_entries]

    buckets: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = {}
    for d in digests:
        key = (d["round_idx"], d["role"])
        buckets.setdefault(key, {"tool": [], "trace": []})
        buckets[key]["tool" if d["kind"] == "tool" else "trace"].append(d)

    for (ridx, role), grp in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        ridx_disp = ridx + 1
        tool_items = grp.get("tool", [])
        trace_items = grp.get("trace", [])

        if tool_items:
            print(f"\n🛠️ Round {ridx_disp} | Role={role} | tool_calls+={len(tool_items)}")
            for it in tool_items:
                status = "OK" if it["ok"] else "FAIL"
                if it["denied"]:
                    status += "/DENIED"

                produced = f" | produced={it['produced_n']}" if it["produced_n"] is not None else ""
                cost = f" | {it['elapsed_ms']}ms" if it["elapsed_ms"] else ""
                args_str = str(it['args'])
                if len(args_str) > 100: args_str = args_str[:100] + "..."
                msg = f"  - {it['tool']} -> {status}{produced}{cost}"

                insight = it["insight"] or ""
                if insight:
                    if len(insight) > 180:
                        insight = insight[:180] + "...[truncated]"
                    msg += f"\n    insight: {insight}"

                if (not it["ok"]) and it["error"]:
                    err = it["error"]
                    if len(err) > 300: err = err[:300] + "..."
                    msg += f"\n    error: {err}"

                print(msg)

        if trace_items:
            print(f"\n📝 Round {ridx_disp} | Role={role} | trace+={len(trace_items)}")
            for it in trace_items:
                insight = it["insight"] or ""
                if len(insight) > 220:
                    insight = insight[:220] + "...[truncated]"
                print(f"  · {it['tool']}: {insight}")

    return len(tool_trace or [])


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        s2 = s[first_nl + 1 :] if first_nl != -1 else ""
        if s2.rstrip().endswith("```"):
            s2 = s2.rstrip()[:-3]
        return s2.strip()
    return s

def _split_debate_and_payload(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """支持“辩论文字 + 末尾 JSON”。"""
    raw = (text or "").strip()
    if not raw:
        return "", None

    s = _strip_code_fences(raw)
    obj, start_idx = try_parse_payload_with_span(s)
    if not obj:
        return s, None

    try:
        validate_payload(obj)
    except Exception:
        return s, None

    if start_idx is None or start_idx <= 0:
        return "", obj

    return s[:start_idx].rstrip(), obj


def _print_assistant_messages_increment(
    msgs: List[BaseMessage],
    start_idx: int,
    *,
    max_chars: int = 900,
    state: DebateState
) -> int:
    """messages 增量：打印 Debate + payload 一行摘要（PM 通常 JSON-only）。"""
    new_msgs = (msgs or [])[start_idx:]
    if not new_msgs:
        return start_idx

    for m in new_msgs:
        if not isinstance(m, AIMessage):
            continue

        content_raw = (m.content or "").strip()
        if not content_raw:
            continue

        role_hint = _infer_assistant_role_hint(m, state)
        debate_text, payload = _split_debate_and_payload(content_raw)

        if debate_text:
            text_to_print = debate_text[:max_chars] + (" ...[truncated]" if len(debate_text) > max_chars else "")
            print(f"\n[{role_hint}]")
            print(text_to_print)

        if payload:
            one_liner = _summarize_structured_payload(payload)
            if one_liner:
                print(f"  ↳ {one_liner}")

        if (not debate_text) and (not payload):
            text_to_print = content_raw[:max_chars] + (" ...[truncated]" if len(content_raw) > max_chars else "")
            print(f"\n[{role_hint}]")
            print(text_to_print)

    return len(msgs or [])
