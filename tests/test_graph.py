import json
import re
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from debate_mas.core import graph as g


def _patch_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """直接 monkeypatch 模块变量 g.CONFIG 为一个可变对象（stub）。"""
    cfg = SimpleNamespace(
        MAX_ROUNDS=3,
        EXIT_ON_CONSENSUS=True,

        ENFORCE_MIN_CANDIDATES=False,
        HUNTER_MIN_CANDIDATES=0,

        HUNTER_DETERMINISTIC_PIPELINE=True,
        HUNTER_PIPELINE_MODE="two_stage",
        HUNTER_RECALL_STRATEGIES=["momentum", "liquidity", "composite"],
        HUNTER_RECALL_MIN_STRATEGIES=2,
        HUNTER_RECALL_TOPK_PER_STRATEGY=10,
        HUNTER_RERANK_OUTPUT_TOPN=20,

        RISK_SCORE_THRESHOLD=50.0,

        ENFORCE_TOOL_ON_NEED_EVIDENCE=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)

    monkeypatch.setattr(g, "CONFIG", cfg, raising=True)

def _patch_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    把协议层解析/校验 monkeypatch 成“可控且稳定”的版本，
    让本关测试聚焦在 graph 编排与状态机逻辑，而不是 JSON schema 细节。
    """
    def fake_parse(text: str):
        m = re.search(r"(\{.*\})\s*$", text, re.S)
        if not m:
            return None, -1
        try:
            return json.loads(m.group(1)), m.start(1)
        except Exception:
            return None, -1

    def fake_validate(_obj):
        return None

    monkeypatch.setattr(g, "try_parse_payload_with_span", fake_parse, raising=True)
    monkeypatch.setattr(g, "validate_payload", fake_validate, raising=True)


def _mk_ai(payload: dict) -> AIMessage:
    return AIMessage(content=json.dumps(payload, ensure_ascii=False))

def test_append_system_prompt_prepends_and_keeps_history_order():
    msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
    out = g._append_system_prompt(msgs, system_prompt="SYS")
    assert out[0].content == "SYS"
    assert [m.content for m in out[1:]] == ["hi", "hello"]


def test_last_ai_has_tool_calls_detects_both_storage_styles():
    state = {"messages": [AIMessage(content="x")]}
    assert g._last_ai_has_tool_calls(state) is False

    state = {
        "messages": [
            AIMessage(
                content="x",
                additional_kwargs={
                    "tool_calls": [
                        {"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}
                    ]
                },
            )
        ]
    }
    assert g._last_ai_has_tool_calls(state) is True

    m = AIMessage(content="x")
    setattr(m, "tool_calls", [{"name": "bar"}])
    state = {"messages": [m]}
    assert g._last_ai_has_tool_calls(state) is True


def test_get_stop_suggest_uppercase_and_strip():
    assert g._get_stop_suggest({"stop_suggest": " stop "}) == "STOP"
    assert g._get_stop_suggest({"stop_suggest": "continue"}) == "CONTINUE"
    assert g._get_stop_suggest(None) == ""


def test_extract_last_payload_supports_debate_plus_tail_json(monkeypatch: pytest.MonkeyPatch):
    _patch_protocol(monkeypatch)

    payload = {"type": "CANDIDATES", "stop_suggest": "STOP", "items": []}
    msg = AIMessage(content="some debate...\n" + json.dumps(payload, ensure_ascii=False))
    state = {"messages": [msg]}

    out = g._extract_last_payload(state, expected_type="CANDIDATES")
    assert out is not None
    assert out["type"] == "CANDIDATES"


def test_should_end_debate_max_rounds_go_pm(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=2, ENFORCE_MIN_CANDIDATES=False)

    state = {"round_idx": 1, "stable_rounds": 0, "messages": []}
    nxt = g._should_end_debate(state)
    assert nxt == "pm"
    assert state.get("stop_reason") == "MAX_ROUNDS_DEBATE"


def test_should_end_debate_guard_denied_forces_next_round(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=False)

    state = {"round_idx": 0, "_round_guard_denied": True, "messages": []}
    nxt = g._should_end_debate(state)
    assert nxt == "next_round"
    assert state.get("stop_reason") == "GUARD_DENIED"


def test_should_end_debate_min_candidates_gate(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=True, HUNTER_MIN_CANDIDATES=3)

    state = {
        "round_idx": 0,
        "messages": [],
        "candidates_cur": [{"symbol": "510300"}, {"symbol": "510500"}],  # unique=2 < 3
        "hunter_stop_suggest": "STOP",
        "auditor_stop_suggest": "STOP",
    }
    nxt = g._should_end_debate(state)
    assert nxt == "next_round"
    assert state.get("stop_reason") == "MIN_CANDIDATES_NOT_MET"


def test_should_end_debate_pipeline_gates(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, ENFORCE_MIN_CANDIDATES=False)

    s1 = {"round_idx": 0, "_need_recall_diversity": True, "messages": []}
    assert g._should_end_debate(s1) == "next_round"
    assert s1["stop_reason"] == "PIPELINE_RECALL_DIVERSITY_NOT_MET"

    s2 = {"round_idx": 0, "_need_rerank_composite": True, "messages": []}
    assert g._should_end_debate(s2) == "next_round"
    assert s2["stop_reason"] == "PIPELINE_RERANK_NOT_MET"


def test_should_end_debate_consensus_stop(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=True, ENFORCE_MIN_CANDIDATES=False)

    state = {
        "round_idx": 0,
        "messages": [],
        "hunter_stop_suggest": "STOP",
        "auditor_stop_suggest": "STOP",
        "stable_rounds": 0,
    }
    nxt = g._should_end_debate(state)
    assert nxt == "pm"
    assert state.get("stop_reason") == "CONSENSUS_STOP"


def test_should_end_debate_stable_and_auditor_stop(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=False, ENFORCE_MIN_CANDIDATES=False)

    state = {
        "round_idx": 0,
        "messages": [],
        "hunter_stop_suggest": "CONTINUE",
        "auditor_stop_suggest": "STOP",
        "stable_rounds": 1,
    }
    nxt = g._should_end_debate(state)
    assert nxt == "pm"
    assert state.get("stop_reason") == "STABLE_AND_AUDITOR_STOP"


def test_should_end_debate_default_continue(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, MAX_ROUNDS=99, EXIT_ON_CONSENSUS=True, ENFORCE_MIN_CANDIDATES=False)

    state = {
        "round_idx": 0,
        "messages": [],
        "hunter_stop_suggest": "CONTINUE",
        "auditor_stop_suggest": "CONTINUE",
        "stable_rounds": 0,
    }
    nxt = g._should_end_debate(state)
    assert nxt == "next_round"
    assert state.get("stop_reason") == "CONTINUE_DEBATE"


def test_graph_compiles_and_runs_one_cycle(monkeypatch: pytest.MonkeyPatch):
    """
    最小端到端：能 compile + invoke，且走完 hunter->auditor->pm。
    不校验业务字段，只校验“主循环能跑通 + stop_reason 写入”。
    """
    _patch_protocol(monkeypatch)
    _patch_config(monkeypatch, MAX_ROUNDS=1, ENFORCE_MIN_CANDIDATES=False)

    hunter_rb = g.RoleBlock(
        role="hunter",
        system_prompt="HUNTER_SYS",
        llm_invoke=lambda _msgs: _mk_ai(
            {
                "type": "CANDIDATES",
                "stop_suggest": "STOP",
                "items": [
                    {"symbol": "510300", "score": 80.0, "reason": "x", "source_skill": "demo", "extra": {}}
                ],
            }
        ),
        tool_node=None,
        postprocess=g.postprocess_hunter,
    )

    auditor_rb = g.RoleBlock(
        role="auditor",
        system_prompt="AUDITOR_SYS",
        llm_invoke=lambda _msgs: _mk_ai({"type": "OBJECTIONS", "stop_suggest": "STOP", "items": []}),
        tool_node=None,
        postprocess=g.postprocess_auditor,
    )

    pm_rb = g.RoleBlock(
        role="pm",
        system_prompt="PM_SYS",
        llm_invoke=lambda _msgs: _mk_ai({"type": "DECISIONS", "stop_suggest": "STOP", "items": []}),
        tool_node=None,
        postprocess=g.postprocess_pm,
    )

    graph = g.build_etf_attack_patch_graph(hunter=hunter_rb, auditor=auditor_rb, pm=pm_rb)

    init_state = {
        "messages": [],
        "round_idx": 0,
        "stable_rounds": 0,
        "tool_trace": [],
        "tool_cache": {},
        "candidates_cur": [],
        "objections_cur": [],
        "diff_cur": {"type": "DIFF", "items": []},
        "risk_reports": [],
        "survivor_universe": [],
        "_round_tool_calls_ok": {},
        "_hunter_round_sniper_strategies": [],
    }

    out = graph.invoke(init_state)

    assert out.get("_last_speaker_role") == "pm"

    assert out.get("round_idx") == 0

    assert any(it.get("symbol") == "510300" for it in (out.get("candidates_cur") or []))

    ai_n = len([m for m in (out.get("messages") or []) if isinstance(m, AIMessage)])
    assert ai_n >= 3

    assert out.get("stop_reason") in (None, "MAX_ROUNDS_DEBATE")