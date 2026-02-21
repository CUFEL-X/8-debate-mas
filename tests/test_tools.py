import json
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from debate_mas.protocol import SkillResult
from debate_mas.skills.base import SkillContext
from debate_mas.core import tools as t


class _FakeConfig:
    ROLE_TOOL_ALLOWLIST = {
        "hunter": ["quantitative_sniper"],
        "pm": ["portfolio_allocator"],
        "auditor": [],
    }
    ROLE_TOOL_MAX_CALLS = {"hunter": 1, "pm": 2, "auditor": 0}
    FORBID_SAME_TOOL_SAME_ARGS_IN_SAME_ROUND = True

    HUNTER_PIPELINE_SNIPER_STRATEGY = "composite"
    HUNTER_RERANK_OUTPUT_TOPN = 3

    SNIPER_DEFAULTS = {"top_k": 5, "min_amount": 0}
    SNIPER_PROFILES = {
        "composite": {"top_k": 9},  
        "momentum": {"top_k": 7},
    }
    SNIPER_ENFORCE = {}  
    SNIPER_LIMITS = {"max_top_k": 4}  
    PM_PORTFOLIO_ALLOCATOR_ENFORCE = {}


class _DummyDossier:
    """最小 dossier 占位。测试不依赖其内容。"""


def _ctx_stub(role: str) -> SkillContext:
    return SkillContext.model_construct(dossier=_DummyDossier(), agent_role=role, ref_date="2025-01-01")


class _SniperArgs(BaseModel):
    strategy: Optional[str] = None
    top_k: int = 10
    composite_weights: Optional[dict] = None


class _AllocatorArgs(BaseModel):
    candidates: List[dict] = Field(default_factory=list)
    risk_reports: List[dict] = Field(default_factory=list)


class _FakeStructuredTool:
    """
    轻量替身：只提供 invoke() / args_schema / name / description
    （避免 LangChain 版本差异导致测试脆弱）
    """
    def __init__(self, *, name: str, args_schema, handler):
        self.name = name
        self.description = ""
        self.args_schema = args_schema
        self._handler = handler

    def invoke(self, tool_args: Dict[str, Any]):
        return self._handler(tool_args)


class _FakeSkill:
    def __init__(self, name: str, base_tool: _FakeStructuredTool):
        self.name = name
        self._tool = base_tool

    def to_langchain_tool(self, ctx: SkillContext):
        return self._tool

def test_build_ctx_injects_fields():
    ctx = t.build_ctx(_DummyDossier(), role="hunter", ref_date="2025-01-01")
    assert ctx.agent_role == "hunter"
    assert ctx.ref_date == "2025-01-01"
    assert ctx.dossier is not None


def test_build_tools_for_role_uses_allowlist_only(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

    called = {"get": [], "load": 0}

    def fake_load_all_skills():
        called["load"] += 1

    def fake_get_skill(name: str):
        called["get"].append(name)
        tool = _FakeStructuredTool(
            name=name,
            args_schema=_SniperArgs,
            handler=lambda _: json.dumps(SkillResult.ok(data={"ping": 1}).model_dump(), ensure_ascii=False),
        )
        return _FakeSkill(name, tool)

    monkeypatch.setattr(t.SkillRegistry, "load_all_skills", staticmethod(fake_load_all_skills), raising=True)
    monkeypatch.setattr(t.SkillRegistry, "get_skill", staticmethod(fake_get_skill), raising=True)

    st = {}
    ctx = _ctx_stub("hunter")
    tools = t.build_tools_for_role("hunter", ctx, st)

    assert called["load"] == 1
    assert called["get"] == ["quantitative_sniper"]
    assert len(tools) == 1
    assert tools[0].name == "quantitative_sniper"


def test_guard_denies_not_in_allowlist(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
    st = {"_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
    ok, reason = t.tool_guard_check("hunter", "not_allowed", {}, st)
    assert ok is False
    assert "白名单" in reason


def test_guard_denies_over_max_calls(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
    st = {"_round_tool_calls": {"hunter": 1}, "_round_fingerprints": set()}
    ok, reason = t.tool_guard_check("hunter", "quantitative_sniper", {}, st)
    assert ok is False
    assert "上限" in reason


def test_guard_denies_dedup_same_tool_same_args(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)
    args = {"x": 1}
    fp = t.fingerprint("quantitative_sniper", args)

    st = {"_round_tool_calls": {"hunter": 0}, "_round_fingerprints": {fp}}
    ok, reason = t.tool_guard_check("hunter", "quantitative_sniper", args, st)
    assert ok is False
    assert "dedup" in reason.lower() or "重复" in reason


def test_wrap_tool_with_guard_denied_returns_json_and_writes_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

    base_tool = _FakeStructuredTool(
        name="quantitative_sniper",
        args_schema=_SniperArgs,
        handler=lambda _: (_ for _ in ()).throw(RuntimeError("should not call")),
    )

    st = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
    wrapped = t._wrap_tool_with_guard(role="hunter", tool_name="not_allowed", base_tool=base_tool, state=st)

    out = wrapped.invoke({"top_k": 10})
    obj = json.loads(out)
    assert obj["success"] is False
    assert "[GUARD_DENY]" in (obj.get("error_msg") or "")
    assert st.get("tool_trace") and st["tool_trace"][-1]["denied"] is True


def test_wrap_tool_with_guard_ok_returns_json_and_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

    def handler(args: Dict[str, Any]) -> str:
        payload = SkillResult.ok(data={"items": [1, 2]}, insight="ok").model_dump()
        payload["data"]["seen_top_k"] = args.get("top_k")
        payload["data"]["seen_strategy"] = args.get("strategy")
        return json.dumps(payload, ensure_ascii=False)

    base_tool = _FakeStructuredTool(
        name="quantitative_sniper",
        args_schema=_SniperArgs,
        handler=handler,
    )

    st = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
    wrapped = t._wrap_tool_with_guard(role="hunter", tool_name="quantitative_sniper", base_tool=base_tool, state=st)

    out = wrapped.invoke({})
    obj = json.loads(out)

    assert obj["success"] is True
    assert st["tool_trace"][-1]["produced_n"] == 2
    assert st["tool_trace"][-1]["ok"] is True

    assert obj["data"]["seen_strategy"] == "composite"
    assert int(obj["data"]["seen_top_k"]) <= 4


def test_portfolio_allocator_injects_state_fields_when_schema_supports(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

    def handler(args: Dict[str, Any]) -> str:
        payload = SkillResult.ok(data={"candidates": args.get("candidates"), "risk_reports": args.get("risk_reports")}).model_dump()
        return json.dumps(payload, ensure_ascii=False)

    base_tool = _FakeStructuredTool(
        name="portfolio_allocator",
        args_schema=_AllocatorArgs,
        handler=handler,
    )

    st = {
        "round_idx": 0,
        "_round_tool_calls": {"pm": 0},
        "_round_fingerprints": set(),
        "candidates_cur": [{"code": "510300"}],
        "risk_reports": [{"code": "510300", "risk": "low"}],
    }

    wrapped = t._wrap_tool_with_guard(role="pm", tool_name="portfolio_allocator", base_tool=base_tool, state=st)
    out = wrapped.invoke({}) 
    obj = json.loads(out)

    assert obj["success"] is True
    assert obj["data"]["candidates"] == [{"code": "510300"}]
    assert obj["data"]["risk_reports"] == [{"code": "510300", "risk": "low"}]


def test_runtime_state_contextvar_affects_wrapper_trace(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(t, "CONFIG", _FakeConfig, raising=True)

    def handler(_args: Dict[str, Any]) -> str:
        payload = SkillResult.ok(data={"items": [1]}).model_dump()
        return json.dumps(payload, ensure_ascii=False)

    base_tool = _FakeStructuredTool(
        name="quantitative_sniper",
        args_schema=_SniperArgs,
        handler=handler,
    )

    fallback_state = {"round_idx": 0, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
    wrapped = t._wrap_tool_with_guard(
        role="hunter",
        tool_name="quantitative_sniper",
        base_tool=base_tool,
        state=fallback_state,
    )

    state_in = {"round_idx": 7, "_round_tool_calls": {"hunter": 0}, "_round_fingerprints": set()}
    token = t._CURRENT_STATE.set(state_in)
    try:
        out = wrapped.invoke({})
        assert json.loads(out)["success"] is True
    finally:
        t._CURRENT_STATE.reset(token)

    assert state_in.get("tool_trace")
    assert state_in["tool_trace"][-1]["round_idx"] == 7

