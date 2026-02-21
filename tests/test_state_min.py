from __future__ import annotations

from typing import Any

from debate_mas.core.state import (
    init_state,
    push_candidates,
    push_objections,
    push_diff,
    push_decisions,
    bump_round,
)

class FakeDossier:
    """只提供 frozen_view()，避免依赖真实 Dossier 初始化参数。"""
    def frozen_view(self) -> dict[str, Any]:
        return {"ok": True}


def _assert_has_keys(st: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in st]
    assert not missing, f"missing keys: {missing}"


def test_init_state_has_core_fields() -> None:
    st = init_state("x", FakeDossier(), ref_date="2025-10-26")

    core_keys = [
        "mission",
        "ref_date",
        "dossier",
        "dossier_view",
        "messages",
        "round_idx",
        "candidates_cur",
        "objections_cur",
        "diff_cur",
        "decisions_cur",
        "history",
        "stop_reason",
        "tool_trace",
    ]
    _assert_has_keys(st, core_keys)

    assert st["round_idx"] == 0
    assert isinstance(st["messages"], list)
    assert isinstance(st["dossier_view"], dict)

    for k in ["candidates", "objections", "diffs", "decisions"]:
        assert k in st["history"], f"history missing {k}"
        assert isinstance(st["history"][k], list), f"history[{k}] must be list"


def test_push_writes_cur_and_history() -> None:
    st = init_state("x", FakeDossier())

    push_candidates(st, [{"id": "A", "score": 1}])
    push_objections(st, [{"id": "A", "risk": "x"}])
    push_diff(st, {"changed": True})
    push_decisions(st, [{"action": "WATCH"}])

    assert len(st["candidates_cur"]) == 1
    assert len(st["objections_cur"]) == 1
    assert isinstance(st["diff_cur"], dict)
    assert len(st["decisions_cur"]) == 1

    assert len(st["history"]["candidates"]) == 1
    assert len(st["history"]["objections"]) == 1
    assert len(st["history"]["diffs"]) == 1
    assert len(st["history"]["decisions"]) == 1

    assert st["history"]["candidates"][0]["round"] == 0
    assert st["history"]["objections"][0]["round"] == 0
    assert st["history"]["diffs"][0]["round"] == 0
    assert st["history"]["decisions"][0]["round"] == 0


def test_bump_round_resets_runtime() -> None:
    st = init_state("x", FakeDossier())

    st["_round_tool_calls"]["hunter"] = 7
    bump_round(st)

    assert st["round_idx"] == 1
    assert st["_round_tool_calls"]["hunter"] == 0
    assert st["_round_guard_denied"] is False
    assert st["_round_missing_evidence"] is False