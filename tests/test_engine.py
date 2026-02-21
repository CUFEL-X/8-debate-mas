import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from debate_mas.core import engine as e

def _patch_config(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    """
    直接 monkeypatch 模块变量 e.CONFIG 为一个可变对象（stub），避免 FrozenInstanceError。
    """
    cfg = SimpleNamespace(
        DATA_DIR="__DATA_DIR__",
        VERBOSE=False,

        HUNTER_MODEL="stub-hunter",
        AUDITOR_MODEL="stub-auditor",
        PM_MODEL="stub-pm",

        ROLE_TEMPERATURE={"hunter": 0.9, "auditor": 0.3, "pm": 0.1},
        ROLE_MAX_TOKENS={"hunter": 1200, "auditor": 800, "pm": 800},
        MAX_TOKENS_DEFAULT=1000,

        ROLE_TOOL_ALLOWLIST={"hunter": [], "auditor": [], "pm": []},

        HUNTER_BLEND={"demo": 1.0},
    )

    def _get_model_config():
        return {"stub": True}

    cfg.get_model_config = _get_model_config

    for k, v in overrides.items():
        setattr(cfg, k, v)

    monkeypatch.setattr(e, "CONFIG", cfg, raising=True)


class DummyLLM:
    """
    最小 LLM stub：
    - bind_tools(tools) -> self
    - invoke(messages) -> AIMessage
    """
    def __init__(self, role: str):
        self.role = role
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, _messages):
        return AIMessage(content=f"[{self.role}] ok")


def _patch_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_llm(model_name: str, *, temperature: float = 0.2, max_tokens: int = 4000):
        role = "unknown"
        if "hunter" in model_name: role = "hunter"
        if "auditor" in model_name: role = "auditor"
        if "pm" in model_name: role = "pm"
        return DummyLLM(role=role)

    monkeypatch.setattr(e, "_build_llm", fake_build_llm, raising=True)


def _patch_loader_and_state(monkeypatch: pytest.MonkeyPatch):
    calls = {"load": [], "init_state": []}

    class DummyLoader:
        def load_from_folder(self, *, mission: str, folder_path: str):
            calls["load"].append({"mission": mission, "folder_path": folder_path})
            return {"dossier": True, "folder": folder_path}

    def fake_init_state(*, mission, dossier, ref_date, messages):
        calls["init_state"].append(
            {"mission": mission, "dossier": dossier, "ref_date": ref_date, "messages": messages}
        )
        return {
            "mission": mission,
            "ref_date": ref_date,
            "dossier": dossier,
            "dossier_view": {"meta": "stub"},
            "messages": list(messages or []),

            "round_idx": 0,
            "stable_rounds": 0,
            "tool_trace": [],
            "tool_cache": {},

            "candidates_cur": [],
            "objections_cur": [],
            "diff_cur": {"type": "DIFF", "items": []},
            "risk_reports": [],
            "survivor_universe": [],

            "decisions": [],
        }

    monkeypatch.setattr(e, "DualModeLoader", DummyLoader, raising=True)
    monkeypatch.setattr(e, "init_state", fake_init_state, raising=True)
    return calls


def _patch_prompts_tools(monkeypatch: pytest.MonkeyPatch):
    calls = {"prompts": [], "tools": []}

    def fake_build_prompts_etf(*, mission, dossier_view, allowlist_by_role):
        calls["prompts"].append(
            {"mission": mission, "dossier_view": dossier_view, "allowlist_by_role": allowlist_by_role}
        )
        return {"hunter": "HUNTER_SYS", "auditor": "AUDITOR_SYS", "pm": "PM_SYS"}

    def fake_build_role_tools_and_node(*, role, dossier, ref_date, state):
        calls["tools"].append({"role": role, "ref_date": ref_date})
        return ([], None, None)

    monkeypatch.setattr(e, "build_role_prompts_etf", fake_build_prompts_etf, raising=True)
    monkeypatch.setattr(e, "build_role_tools_and_node", fake_build_role_tools_and_node, raising=True)
    return calls


def _patch_graph_and_renderer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    让 _run_graph_and_render 真正跑到：
    - build graph -> invoke -> final_state
    - transcript 序列化 + 落盘
    - renderer.render 返回 artifacts
    """
    class DummyGraph:
        def __init__(self, final_state):
            self._final_state = final_state

        def invoke(self, _st):
            return self._final_state

        def stream(self, _st, stream_mode="values"):
            yield self._final_state

    def fake_build_graph(*, hunter, auditor, pm):
        final_state = {
            "round_idx": 0,
            "stable_rounds": 0,
            "stop_reason": "MAX_ROUNDS_DEBATE",

            "_last_speaker_role": "pm",

            "messages": [
                AIMessage(content='{"type":"CANDIDATES","stop_suggest":"STOP","items":[{"symbol":"510300","score":80.0,"reason":"x","source_skill":"demo","extra":{}}]}'),
                AIMessage(content='{"type":"OBJECTIONS","stop_suggest":"STOP","items":[]}'),
                AIMessage(content='{"type":"DECISIONS","stop_suggest":"STOP","items":[]}'),
            ],

            "tool_trace": [{"kind": "trace", "role": "system", "tool": "__test__", "ok": True, "insight": "ok"}],
            "tool_cache": {},

            "candidates_cur": [{"symbol": "510300", "score": 80.0, "reason": "x", "source_skill": "demo", "extra": {}}],
            "objections_cur": [],
            "diff_cur": {"type": "DIFF", "items": []},

            "decisions": [], 
            "dossier_view": {"meta": "stub"},
        }
        return DummyGraph(final_state)

    class FakeRenderer:
        def __init__(self, output_dir: str):
            self.output_dir = output_dir

        def render(self, *, mission: str, decisions, extra_meta):
            return {"memo": str(Path(self.output_dir) / "memo.md")}

    def fake_merge_candidates(cands_list, source_weights=None):
        if not cands_list:
            return []
        x = cands_list[0] or []
        return list(x)

    def fake_explain_merge(_merged, top_n=5):
        return "merge_notes: stub"

    monkeypatch.setattr(e, "build_etf_attack_patch_graph", fake_build_graph, raising=True)
    monkeypatch.setattr(e, "DebateRenderer", FakeRenderer, raising=True)
    monkeypatch.setattr(e, "merge_candidates", fake_merge_candidates, raising=True)
    monkeypatch.setattr(e, "explain_merge", fake_explain_merge, raising=True)


def _patch_skill_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(e.SkillRegistry, "load_all_skills", lambda force_reload=False: None, raising=True)

def test_infer_role_and_serialize_messages():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
        ToolMessage(content="tool ok", tool_call_id="t1"),
    ]

    out = e._serialize_messages(msgs)

    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"
    assert out[2]["role"] == "tool"
    assert out[2]["tool_call_id"] == "t1"


def test_serialize_messages_captures_tool_calls_in_both_styles():
    m1 = AIMessage(
        content="x",
        additional_kwargs={
            "tool_calls": [{"id": "1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]
        },
    )
    m2 = AIMessage(content="y")
    setattr(m2, "tool_calls", [{"name": "bar"}])

    out = e._serialize_messages([m1, m2])
    assert "tool_calls" in out[0]
    assert "tool_calls" in out[1]



def test_setup_dossier_and_state_injects_seed_message_and_defaults(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch, DATA_DIR="DATA_DEFAULT")
    calls = _patch_loader_and_state(monkeypatch)

    dossier, st = e._setup_dossier_and_state(
        mission="m",
        ref_date="2025-10-26",
        folder_path=None, 
        seed_user_message="seed",
    )

    assert dossier["folder"] == "DATA_DEFAULT"
    assert isinstance(st["messages"][0], HumanMessage)
    assert st["messages"][0].content == "seed"

    assert calls["load"][0]["folder_path"] == "DATA_DEFAULT"
    assert calls["init_state"][0]["mission"] == "m"
    assert calls["init_state"][0]["ref_date"] == "2025-10-26"


def test_setup_prompts_tools_llms_builds_roleblocks(monkeypatch: pytest.MonkeyPatch):
    _patch_config(monkeypatch)
    _patch_llm(monkeypatch)
    calls = _patch_prompts_tools(monkeypatch)

    st = {"dossier_view": {"meta": "stub"}}
    prompts, hunter_rb, auditor_rb, pm_rb = e._setup_prompts_tools_llms(
        mission="m",
        dossier={"dossier": True},
        ref_date="2025-10-26",
        st=st,
    )

    assert prompts["hunter"] == "HUNTER_SYS"
    assert hunter_rb.role == "hunter"
    assert auditor_rb.role == "auditor"
    assert pm_rb.role == "pm"

    assert len(calls["prompts"]) == 1
    assert {c["role"] for c in calls["tools"]} == {"hunter", "auditor", "pm"}

    assert isinstance(hunter_rb.llm_invoke([]), AIMessage)


def test_run_graph_and_render_creates_transcript_and_returns_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, VERBOSE=False)
    _patch_graph_and_renderer(monkeypatch, tmp_path)

    st = {
        "messages": [],
        "tool_trace": [],
        "tool_cache": {},
        "candidates_cur": [],
        "objections_cur": [],
        "diff_cur": {"type": "DIFF", "items": []},
        "dossier_view": {"meta": "stub"},
        "round_idx": 0,
        "stable_rounds": 0,
    }

    rb = e.RoleBlock(
        role="hunter",
        system_prompt="SYS",
        llm_invoke=lambda _ms: AIMessage(content="x"),
        tool_node=None,
        postprocess=lambda _st: None,
    )

    artifacts = e._run_graph_and_render(
        mission="m",
        ref_date="2025-10-26",
        output_dir=str(tmp_path),
        st=st,
        hunter_block=rb,
        auditor_block=rb,
        pm_block=rb,
        verbose_summary=False,
    )

    assert isinstance(artifacts, dict)
    assert "memo" in artifacts


    transcript_path = artifacts.get("transcript")
    if transcript_path:
        p = Path(transcript_path)
        assert p.exists()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["mission"] == "m"
        assert data["ref_date"] == "2025-10-26"
        assert isinstance(data["transcript"], list)
    else:
        files = list(Path(tmp_path).glob("*_transcript.json"))
        assert len(files) >= 1


def test_run_entrypoint_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _patch_config(monkeypatch, VERBOSE=False, DATA_DIR="DATA_DEFAULT")
    _patch_skill_registry(monkeypatch)
    _patch_llm(monkeypatch)
    _patch_loader_and_state(monkeypatch)
    _patch_prompts_tools(monkeypatch)
    _patch_graph_and_renderer(monkeypatch, tmp_path)

    artifacts = e.run(
        "m",
        ref_date="2025-10-26",
        folder_path=None,
        output_dir=str(tmp_path),
        seed_user_message="seed",
    )

    assert isinstance(artifacts, dict)
    assert "memo" in artifacts